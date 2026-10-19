"""L0 — op-level numerics, judged against the error the ORIGINAL implementation already carries.

THE TOLERANCE PROBLEM, AND WHY THIS AVOIDS IT. "Is rtol=1e-2 ok?" has no defensible answer: too
tight and every honest bf16 rewrite fails, too loose and a real bug passes. So nothing here is
compared against a hand-picked constant. The frozen capture gives:

    captured    what the ORIGINAL fused op produced, in its own precision (out, dgrad, wgrad)
    ref_fp32    an INDEPENDENT pure-PyTorch reference, in fp32, on the same frozen inputs
    err_base    |captured - ref_fp32|      <- the error the baseline already carries

and the rule is: **a candidate is acceptable iff it is no further from ref_fp32 than the original
is** (times a small margin). That is not arbitrary, and it never penalises a candidate for rounding
the baseline also does.

The reference must be an independent implementation, not a replay of the fused op: the captured call
carries runtime handles (ProcessGroup, WeightGradStore, the owning module) that cannot be rebuilt
offline. See `meta.json:opaque_args`.

Second, non-negotiable check: the error must not be ONE-SIDED. A small systematic bias passes every
magnitude check and still destroys convergence over thousands of steps. That failure is invisible to
a tolerance and obvious to a sign test.
"""
from __future__ import annotations

import json
import sys

import torch


def load_record(path):
    return torch.load(path, weights_only=False)


def _t(node, dtype=None, dev="cpu"):
    if isinstance(node, dict) and node.get("__t__"):
        x = node["data"].to(dev)
        if dtype is not None and x.is_floating_point():
            x = x.to(dtype)
        return x
    return None


def error_stats(actual, reference):
    """Magnitude AND shape of the error distribution. Both are required to judge."""
    a = actual.detach().float().flatten()
    r = reference.detach().float().flatten()
    d = a - r
    denom = r.abs().mean().clamp_min(1e-12)
    pos = (d > 0).sum().item()
    nz = (d != 0).sum().item()
    rr = (r * r).sum()
    return {
        "max_abs": d.abs().max().item(),
        "mean_abs": d.abs().mean().item(),
        "rel_mean_abs": (d.abs().mean() / denom).item(),
        # 0.5 for unbiased rounding noise; near 0 or 1 means an ADDITIVE systematic bias.
        "sign_balance": (pos / nz) if nz else 0.5,
        # The sign test alone is NOT enough, and a negative test proved it: a candidate scaled by
        # 1.005 produces d = 0.005*r, whose sign simply follows r -- so sign_balance stays at ~0.5
        # and the magnitude check passes too (the original's own bf16 error is LARGER than the bias).
        # A MULTIPLICATIVE bias only shows up in the correlation between the error and the value:
        # <d,r>/<r,r> is exactly the implied scale offset (0.005 for the case above, ~0 for rounding).
        "scale_bias": ((d * r).sum() / rr).item() if float(rr) > 0 else 0.0,
        "offset_bias": (d.mean() / denom).item(),
        "nonzero_frac": nz / d.numel(),
        "max_ulp_bf16": max_ulp_bf16(d, r),
    }


def max_ulp_bf16(d, r):
    """Largest |error| in units of ONE bf16 ulp of the reference value at that element.

    WHY THIS EXISTS. When the reference is itself rounded to bf16 (every gradient the harness takes
    onto a bf16 leaf is), a correctly-rounded candidate differs from it by 0 or exactly 1 ulp, and
    WHERE the 1-ulp disagreements land is luck. Measured on the swiglu_bwd_fwd_bw oracle (2026-10-10):
    the original's ~860 disagreements all fell on |dgrad| ~1e-4, so its max_abs was 4.8e-7, and an
    fp64-exact-then-rounded candidate -- more accurate than the original -- scored 1.9e-6 and FAILED
    the x1.25 max_abs rule. Measured in ulps both are 0.99. An element whose reference is 0 gets the
    smallest bf16 subnormal as its ulp, so any non-zero error there is enormous, as it should be.
    """
    a = r.abs()
    tiny = torch.tensor(2.0 ** -133, device=r.device)
    ulp = torch.where(a > 0, torch.exp2(torch.floor(torch.log2(a.clamp_min(2.0 ** -126))) - 7), tiny)
    return (d.abs() / ulp).max().item()


def verdict(base_err, cand_err, margin=1.25, sign_tol=0.15):
    """Acceptable iff no further from the reference than the original, and not one-sided."""
    fails = []
    for key in sorted(set(base_err) & set(cand_err)):
        b, c = base_err[key], cand_err[key]
        if c["max_abs"] > b["max_abs"] * margin:
            fails.append(f"{key}: max_abs {c['max_abs']:.3e} exceeds the original's "
                         f"{b['max_abs']:.3e} x{margin}")
        if abs(c["sign_balance"] - 0.5) > sign_tol and abs(b["sign_balance"] - 0.5) <= sign_tol:
            fails.append(f"{key}: error is one-sided (sign_balance {c['sign_balance']:.3f}) while "
                         f"the original's is not ({b['sign_balance']:.3f}) — a systematic bias, "
                         f"which no magnitude check would catch")
    return {"pass": not fails, "reason": "; ".join(fails) or
            "no further from the fp32 reference than the original, and not one-sided",
            "base": base_err, "candidate": cand_err}


def rmsnorm_linear_reference(x, ln_w, w, b, eps=1e-6):
    """Independent reference for TE's _LayerNormLinear under RMSNorm (what this model uses)."""
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rms * ln_w.float()
    out = torch.nn.functional.linear(y, w.float(), b.float() if b is not None else None)
    return out


def rmsnorm_linear_swiglu_reference(x, ln_weight, weight, bias=None, eps=1e-6):
    """Independent fp32 reference for the MLP fc1 + SwiGLU COMPOSITE (the `mlp_fc1_swiglu` head).

    RMSNorm -> fc1 (gate|up, 2 x ffn) -> SiLU(gate) * up, i.e. exactly
    `_LayerNormLinear` followed by `megatron.core.fusions.fused_bias_swiglu.swiglu`.

    Three things here are semantics, not style:

    * THE SPLIT IS `chunk(2, -1)`, NOT interleaved. Megatron's `swiglu()` is
      `y_1, y_2 = torch.chunk(y, 2, -1); silu(y_1) * y_2`, so the FIRST half of the 37888 columns is
      the gate and the second is the up-projection. (The weight is stored with stride=2 for TP
      resharding, which is a checkpoint-layout concern and does NOT interleave the runtime columns
      at TP1.) Swapping the halves produces a same-shaped, wrong, self-consistent answer.
    * THE ACTIVATION IS PART OF THE OP. The candidate's premise is that the pointwise pass
      disappears into the fc1 epilogue, so the frozen op's output must be POST-activation
      [s, b, ffn]; an oracle cut before the activation cannot score the win it exists to measure.
    * fp32 THROUGHOUT, including the SiLU. The original computes the activation in bf16 from a bf16
      GEMM output; this is the truth that error is measured against, not a re-run of it.
    """
    y = rmsnorm_linear_reference(x, ln_weight, weight, bias, eps)
    gate, up = torch.chunk(y, 2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def linear_reference(w, x, b=None):
    """Independent reference for TE's _Linear. NOTE the argument order: `_Linear.apply` takes the
    WEIGHT first, unlike `_LayerNormLinear`. Reading the order off the capture instead of assuming it
    is the difference between an oracle and a plausible-looking pile of wrong numbers."""
    return torch.nn.functional.linear(x.float(), w.float(), b.float() if b is not None else None)


def linear_pair_reference(weight_fc2, x_fc2, weight_oproj, x_oproj):
    """Independent fp32 reference for the TWO TE `_Linear` call sites of one layer, as ONE op:
    the MLP fc2 (18944 -> 3584) and the attention o_proj (3584 -> 3584). Head `mlp_fc2_oproj_gemm`.

    WHY A PAIR. Both run through the same autograd.Function (`_Linear.apply`) and the same hipBLASLt
    solution families, and the head's lever (exact-shape solution re-selection) is a property of the
    Function, not of one shape. Freezing only fc2 would let a lane "win" on fc2 while silently
    changing o_proj's solution outside the oracle -- the lmhead anti-pattern. Both run 56x per rank
    per step, so the unweighted per-call sum this op times is the per-step ratio exactly.

    No bias on either (disable-bias-linear). fp32 throughout; the two products are independent.
    """
    return (linear_reference(weight_fc2, x_fc2), linear_reference(weight_oproj, x_oproj))


def attention_varlen_reference(cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded,
                               cu_seqlens_kv_padded, q, k, v,
                               attn_scale=None, causal=True, qkv_format="thd"):
    """Independent fp32 reference for TE's `FusedAttnFunc` in thd (varlen / group) mode with GQA.

    Three things here are semantics, not style, and getting any of them wrong produces an oracle
    that looks plausible and is not:

    * ISOLATION. `thd` packs several sequences into one buffer, and the whole point of the
      cu_seqlens plumbing is that a token must NOT attend across a segment boundary. A reference
      that runs one dense SDPA over [total, h, d] leaks between sequences and is a different
      mathematical operation from the one the kernel implements. So this loops over segments.
    * PADDING. `cu_seqlens_*` are the REAL lengths; `cu_seqlens_*_padded` are the offsets of each
      segment inside the buffer. On this workload 71.3% of the buffer is padding, and TE leaves
      those rows ZERO (fast_zero_fill). The output is therefore built as zeros_like(q) with only
      the live rows written -- which also makes the padded rows of dq/dk/dv exactly zero, as the
      fused kernel produces them.
    * GQA. q has 14 heads, k/v have 2. The kv heads are expanded by repeat_interleave (group g
      serves q-heads [7g, 7g+7)), NOT by repeat/tile -- the two differ in which q-head reads which
      kv-head and both produce a same-shaped, wrong answer.

    Everything is done in fp32; this is the truth the bf16 original is scored against.
    """
    qf, kf, vf = q.float(), k.float(), v.float()
    d = qf.shape[-1]
    scale = float(attn_scale) if attn_scale is not None else d ** -0.5
    hq, hkv = qf.shape[-2], kf.shape[-2]
    if hq % hkv:
        raise ValueError(f"GQA: {hq} q-heads is not a multiple of {hkv} kv-heads")
    rep = hq // hkv
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_kv.tolist()
    pq = cu_seqlens_q_padded.tolist() if cu_seqlens_q_padded is not None else cq
    pk = cu_seqlens_kv_padded.tolist() if cu_seqlens_kv_padded is not None else ck
    out = torch.zeros_like(qf)
    for i in range(len(cq) - 1):
        lq, lk = cq[i + 1] - cq[i], ck[i + 1] - ck[i]
        if lq <= 0 or lk <= 0:
            continue
        qs, ks = pq[i], pk[i]
        qi = qf[qs:qs + lq].transpose(0, 1)                       # [hq, lq, d]
        ki = kf[ks:ks + lk].repeat_interleave(rep, dim=-2).transpose(0, 1)
        vi = vf[ks:ks + lk].repeat_interleave(rep, dim=-2).transpose(0, 1)
        s = torch.matmul(qi, ki.transpose(-1, -2)) * scale        # [hq, lq, lk]
        if causal:
            # top-left aligned, which is what TE's "causal" means when lq == lk (it does here:
            # self-attention). Written as an explicit mask so the alignment is visible.
            rows = torch.arange(lq, device=s.device).unsqueeze(-1)
            cols = torch.arange(lk, device=s.device).unsqueeze(0)
            s = s.masked_fill(cols > rows, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out = out.index_copy(0, torch.arange(qs, qs + lq, device=out.device),
                             torch.matmul(p, vi).transpose(0, 1))
    return out


def lmhead_ce_reference(hidden_states, weight, labels, loss_mask):
    """Independent fp32 reference for the LM-head + cross-entropy COMPOSITE.

    Mirrors, in fp32 and out-of-place, exactly what Megatron does on this config:
        ColumnParallelLinear(no bias, TP1)  ->  vocab_parallel_cross_entropy  ->  * loss_mask

    Four things here are semantics, not style:

    * THE `* loss_mask` IS PART OF THE OP. The candidate elides loss-masked rows, which is exact
      only if those rows contribute zero. Writing the multiply into the op contract is what makes
      the oracle able to say so; leaving it out would score an exact kernel against garbage rows.
      It is safe in both directions because loss_mask is 0/1 and `pretrain_gpt.loss_func`
      multiplies by it again (x*m*m == x*m), so integration needs no change to the loss function.
    * LAYOUT. `labels`/`loss_mask` arrive as [b, s] and the logits as [s, b, V]; Megatron's
      `compute_language_model_loss` transposes labels in and the loss out. Getting this wrong
      transposes a [1, 4096] tensor onto itself at MBS=1 and is invisible until MBS>1.
    * OUT-OF-RANGE TARGETS. Megatron's CE masks targets outside [vocab_start, vocab_end) and
      zeroes their predicted logit. At TP1 that band is the whole vocabulary, but SFT labels can
      still carry ignore values, and a bare `gather` would RAISE on them.
    * NO IN-PLACE. The original subtracts the max and exponentiates in place to save memory; a
      differentiable reference cannot, and log-sum-exp is the same mathematics.
    """
    h = hidden_states.float()
    w = weight.float()
    logits = torch.nn.functional.linear(h, w)                 # [s, b, V]
    vocab = logits.shape[-1]
    tgt = labels.transpose(0, 1).contiguous()                 # [b, s] -> [s, b]
    bad = (tgt < 0) | (tgt >= vocab)
    safe = torch.where(bad, torch.zeros_like(tgt), tgt)
    lse = torch.logsumexp(logits, dim=-1)                     # [s, b]
    pred = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    pred = torch.where(bad, torch.zeros_like(pred), pred)
    loss_sb = lse - pred
    loss_bs = loss_sb.transpose(0, 1).contiguous()            # [s, b] -> [b, s]
    return loss_bs * loss_mask.float().reshape(loss_bs.shape)


def rope_qk_thd_reference(q, k, cu_seqlens_q, cu_seqlens_kv, freqs):
    """Independent fp32 reference for Megatron's UNFUSED thd RoPE applied to q AND k of one layer.

    Mirrors `rope_utils._apply_rotary_pos_emb_thd` -> `_apply_rotary_pos_emb_bshd` at CP=1, non-
    interleaved, no MLA, mscale 1.0, rot_dim == head_dim (all four recorded in meta, checked at
    capture), in fp32 and as ONE gather instead of a per-segment loop. Returns (q_rot, k_rot).

    Three things here are semantics, not style:

    * POSITIONS RESET AT EVERY SEGMENT. `freqs` holds max_seqlen rows (not total tokens), so Megatron
      takes CASE 2 and segment i uses freqs[0:len_i]. A kernel that indexes freqs by the GLOBAL token
      position applies the wrong rotation to every segment after the first and produces a
      same-shaped, plausible, wrong answer -- and attention would then see positions continuing
      across conversations, which is exactly the packing leak `pack_gate` exists to rule out.
    * NON-INTERLEAVED rotate-half: (x1, x2) = chunk(x, 2, -1); rotate = cat(-x2, x1). The interleaved
      (GPT-J) pairing is a different operation with the same shapes.
    * q and k use their OWN cu_seqlens (identical in self-attention, but they are separate inputs at
      the call site and the oracle keeps them separate).

    fp32 throughout, including cos/sin; the original casts cos/sin to bf16 and rounds after each
    multiply and the add. That is the error the original carries, not the truth.
    """
    f = freqs.float().reshape(freqs.shape[0], freqs.shape[-1])            # [max_s, d]

    def _one(t, cu):
        cu_l = [int(c) for c in cu.tolist()]
        lens = [b - a for a, b in zip(cu_l[:-1], cu_l[1:])]
        if sum(lens) != t.shape[0]:
            raise ValueError(f"cu_seqlens cover {sum(lens)} tokens, tensor has {t.shape[0]}")
        if max(lens) > f.shape[0]:
            raise ValueError(f"segment of {max(lens)} tokens exceeds freqs rows {f.shape[0]}")
        pos = torch.cat([torch.arange(n, device=t.device) for n in lens])
        ang = f.to(t.device)[pos].unsqueeze(1)                               # [t, 1, d]
        if ang.shape[-1] != t.shape[-1]:
            raise ValueError("partial rotary (rot_dim < head_dim) is not this op")
        tf = t.float()
        x1, x2 = torch.chunk(tf, 2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        return tf * torch.cos(ang) + rot * torch.sin(ang)

    return _one(q, cu_seqlens_q), _one(k, cu_seqlens_kv)


def swiglu_reference(x):
    """Independent fp32 reference for Megatron's bias-free SwiGLU (the `swiglu_bwd_fwd_bw` head).

    `megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction` reached through
    `megatron.core.transformer.mlp.bias_swiglu_impl(fc1_out, None, ...)`: forward `swiglu(y)`, backward
    `swiglu_back(g, y)`, both torch.compile'd (the `triton_poi_fused_mul_silu_split_0` /
    `triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0` kernels in the profile).

    * THE SPLIT IS `chunk(2, -1)`: the FIRST half of the last dim is the gate (SiLU'd), the SECOND is
      the up-projection. dgrad keeps that layout: [d_gate | d_up] concatenated on the last dim, which
      is exactly what fc1's backward consumes. Swapping halves (or interleaving) is same-shaped and
      wrong.
    * fp32 throughout. The original is an inductor kernel that upcasts bf16 loads to fp32 and rounds
      once at the store, so its error is ~0.5 bf16 ulp -- the limit is therefore tight, and a kernel
      that rounds intermediates to bf16 will exceed it. That is deliberate (fp32-internal is free for
      a memory-bound op).
    The gradient is taken by autograd of this function; it equals Megatron's closed form
    g*sig(a)*(1 + a*(1-sig(a)))*b , g*silu(a) mathematically.
    """
    gate, up = torch.chunk(x.float(), 2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def rmsnorm_reference(x, ln_weight, eps=1e-6):
    """Independent fp32 reference for a bare RMSNorm (head `rmsnorm_bwd_fc1_fused`): the norm INSIDE
    the MLP fc1 `_LayerNormLinear`, which on the accepted stack the `mlp_fc1_gemm` overlay computes
    with TE's `tex.rmsnorm_fwd` / `tex.rmsnorm_bwd` (zero_centered_gamma False, sm_margin 0).

        y = x * rsqrt(mean(x^2, -1) + eps) * gamma

    * eps IS A CONSTANT OF THE CAPTURED CALL (1e-5 here, recorded in meta.layout.const), not this
      default; a mismatched eps moves every output by a scale and is caught only by scale_bias.
    * gamma is applied as-is (NOT 1 + gamma): the call passes zero_centered_gamma=False.
    * fp32 throughout. dgamma is a reduction over all T rows (4096), so a candidate that accumulates it
      in bf16 loses ~log2(T) bits; the oracle's dln_weight limit is the original's (fp32-accumulated)
      error, so that shortcut fails here rather than in a trajectory.
    The gradients are autograd of this function onto the (bf16) leaves.
    """
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * rms * ln_weight.float()


def sdpa_gqa_causal_reference(q, k, v, scale=None, is_causal=True, dropout_p=0.0, enable_gqa=True,
                              attn_mask=None):
    """Independent fp32 reference for torch's DENSE `scaled_dot_product_attention` as HF's sdpa path
    calls it on Qwen2.5 (head `attn_fmha_ck_hf`, the hf_trl stack): BHSD layout, GQA via
    enable_gqa, causal, no mask, no dropout. q [B, Hq, S, D], k/v [B, Hkv, S, D].

    * GQA EXPANSION IS repeat_interleave on the head dim (kv head g serves q-heads [g*r, g*r + r)),
      which is what both torch's enable_gqa and HF's repeat_kv do. Tiling (repeat) is same-shaped and
      wrong.
    * CAUSAL IS TOP-LEFT (tril), torch's is_causal semantics; here S_q == S_k so it is unambiguous.
    * SCALE is the captured constant (HF passes module.scaling = D**-0.5 explicitly).
    * fp32 throughout; dropout must be 0 (Qwen2.5 attention_dropout 0.0) and the mask None -- the
      reference refuses anything else rather than computing a different operation.
    """
    if dropout_p:
        raise ValueError("sdpa_gqa_causal_reference is deterministic: dropout_p must be 0")
    if attn_mask is not None:
        raise ValueError("sdpa_gqa_causal_reference covers the mask-free (is_causal) path only")
    qf, kf, vf = q.float(), k.float(), v.float()
    hq, hkv = qf.shape[-3], kf.shape[-3]
    if hq % hkv:
        raise ValueError(f"GQA: {hq} q-heads is not a multiple of {hkv} kv-heads")
    if hq != hkv:
        if not enable_gqa:
            raise ValueError("head counts differ but enable_gqa is False")
        kf = kf.repeat_interleave(hq // hkv, dim=-3)
        vf = vf.repeat_interleave(hq // hkv, dim=-3)
    sc = float(scale) if scale is not None else qf.shape[-1] ** -0.5
    s = torch.matmul(qf, kf.transpose(-1, -2)) * sc
    if is_causal:
        lq, lk = s.shape[-2], s.shape[-1]
        mask = torch.ones(lq, lk, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), vf)


def gateup_swiglu_reference(x, weight_gate, weight_up):
    """Independent fp32 reference for HF `Qwen2MLP`'s gate/up projections + SwiGLU (head
    `hf_mlp_gateup_swiglu`, the hf_trl stack): h = silu(x @ Wg^T) * (x @ Wu^T), the tensor down_proj
    consumes. x [1, M, 3584], weight_gate / weight_up [18944, 3584] (nn.Linear layout [out, in]), no
    bias (Qwen2MLP builds both with bias=False).

    * TWO SEPARATE WEIGHTS, NOT ONE [37888, 3584]: HF stores gate_proj and up_proj as distinct
      Parameters, and the oracle's inputs are what the module hands over. A candidate may concatenate
      them (as a view when they are adjacent in FSDP's flat buffer, see task_notes), but the
      gradients are scored per weight: swapping dWg and dWu is same-shaped and wrong.
    * THE ACTIVATION IS ON THE GATE: silu(gate) * up. silu(up) * gate is same-shaped and wrong.
    * fp32 THROUGHOUT. The original rounds four times (gate GEMM, silu, up GEMM, product -- all bf16
      under autocast); this is the truth that error is measured against, not a re-run of it.
    """
    xf = x.float()
    gate = torch.nn.functional.linear(xf, weight_gate.float())
    up = torch.nn.functional.linear(xf, weight_up.float())
    return torch.nn.functional.silu(gate) * up


REFERENCES = {"rmsnorm_linear": rmsnorm_linear_reference,
              "rmsnorm_linear_bias": rmsnorm_linear_reference,
              "rmsnorm_linear_swiglu": rmsnorm_linear_swiglu_reference,
              "swiglu": swiglu_reference,
              "linear": linear_reference,
              "linear_pair": linear_pair_reference,
              "attention_varlen": attention_varlen_reference,
              "lmhead_ce": lmhead_ce_reference,
              "rope_qk_thd": rope_qk_thd_reference,
              "rmsnorm": rmsnorm_reference,
              "sdpa_gqa_causal": sdpa_gqa_causal_reference,
              "gateup_swiglu": gateup_swiglu_reference}

# Which captured tensor plays which role, per reference. The names are the ORDER the reference's
# signature expects, and they are also the labels the oracle's outputs carry.
LAYOUTS = {
    # A bare RMSNorm, fwd+bwd: the fc1 norm on the accepted stack (`rmsnorm_bwd_fc1_fused`). Two
    # differentiable inputs; outputs (out, dgrad, dln_weight). eps comes from the capture via const.
    "rmsnorm": {"args": ["x", "ln_weight"],
                "grad_labels": ["dgrad", "dln_weight"],
                "grad_of": ["x", "ln_weight"]},
    "rmsnorm_linear": {"args": ["x", "ln_weight", "weight", "bias"],
                       "grad_labels": ["dgrad", "dln_weight", "wgrad"],
                       "grad_of": ["x", "ln_weight", "weight"]},
    # The attention QKV projection: same op as rmsnorm_linear, but it HAS a bias ([4608] on Qwen2.5,
    # add_qkv_bias) and the bias is a learned parameter, so its gradient is an OUTPUT of the op. The
    # plain layout leaves `bias` out of grad_of (fc1 has none); scoring QKV with it would let a
    # candidate drop or mis-reduce the fused bias grad (the dgrad GEMM's BH_Bias epilogue) unseen.
    "rmsnorm_linear_bias": {"args": ["x", "ln_weight", "weight", "bias"],
                            "grad_labels": ["dgrad", "dln_weight", "wgrad", "dbias"],
                            "grad_of": ["x", "ln_weight", "weight", "bias"]},
    # Same argument layout as rmsnorm_linear -- the composite adds an ACTIVATION, not an input. On
    # this model fc1 has no bias, so the capture records three tensors and `bias` binds to None.
    "rmsnorm_linear_swiglu": {"args": ["x", "ln_weight", "weight", "bias"],
                              "grad_labels": ["dgrad", "dln_weight", "wgrad"],
                              "grad_of": ["x", "ln_weight", "weight"]},
    # SwiGLU alone: one differentiable input (fc1's [s, b, 2*ffn] output), one output, one gradient.
    "swiglu": {"args": ["x"],
               "grad_labels": ["dgrad"],
               "grad_of": ["x"]},
    "linear": {"args": ["weight", "x", "bias"],
               "grad_labels": ["wgrad", "dgrad"],
               "grad_of": ["weight", "x"]},
    # fc2 + o_proj of one layer frozen as ONE op (see linear_pair_reference): two forward outputs,
    # two grad_outputs, four gradients. Both weights are [out, in] and take TE's fp32 main_grad
    # under gradient_accumulation_fusion; the leaves (and so the scored wgrads) are bf16.
    "linear_pair": {"args": ["weight_fc2", "x_fc2", "weight_oproj", "x_oproj"],
                    "out_labels": ["out_fc2", "out_oproj"],
                    "grad_labels": ["wgrad_fc2", "dgrad_fc2", "wgrad_oproj", "dgrad_oproj"],
                    "grad_of": ["weight_fc2", "x_fc2", "weight_oproj", "x_oproj"]},
    # Attention has no weights: its three gradients are all dgrads. `grad_of` therefore lists only
    # q/k/v -- the cu_seqlens tensors are int32 INDEXING metadata, and calling requires_grad_ on
    # them is an error, not a no-op.
    "attention_varlen": {"args": ["cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded",
                                  "cu_seqlens_kv_padded", "q", "k", "v"],
                         "grad_labels": ["dq", "dk", "dv"],
                         "grad_of": ["q", "k", "v"]},
    # The LM head is a COMPOSITE: two call sites, one op. `labels` is int64 index metadata and
    # `loss_mask` is a 0/1 selector -- neither is differentiable, so `grad_of` lists only the two
    # tensors the head actually learns through.
    "lmhead_ce": {"args": ["hidden_states", "weight", "labels", "loss_mask"],
                  "grad_labels": ["dgrad", "wgrad"],
                  "grad_of": ["hidden_states", "weight"]},
    # RoPE on the (q, k) PAIR of one layer: TWO forward outputs and therefore TWO grad_outputs.
    # `out_labels` is the only multi-output layout so far; every other layout has one output named
    # "out" and is unaffected. cu_seqlens are int32 index metadata and freqs is a non-learned fp32
    # buffer, so only q and k are differentiable.
    "rope_qk_thd": {"args": ["q", "k", "cu_seqlens_q", "cu_seqlens_kv", "freqs"],
                    "out_labels": ["q_out", "k_out"],
                    "grad_labels": ["dq", "dk"],
                    "grad_of": ["q", "k"]},
    # Dense torch SDPA as HF calls it (BHSD, GQA via enable_gqa, causal, no mask, no dropout). The
    # constants (scale, is_causal, enable_gqa, dropout_p) travel in layout.const from the capture.
    "sdpa_gqa_causal": {"args": ["q", "k", "v"],
                        "grad_labels": ["dq", "dk", "dv"],
                        "grad_of": ["q", "k", "v"]},
    # HF Qwen2MLP gate/up + SwiGLU (head hf_mlp_gateup_swiglu): x and the two projection weights, all
    # differentiable; one output h, three gradients. dgrad = dx is the SUM of both projections'
    # dgrads (x feeds gate and up), i.e. exactly what one concatenated dgrad GEMM produces.
    "gateup_swiglu": {"args": ["x", "weight_gate", "weight_up"],
                      "grad_labels": ["dgrad", "wgrad_gate", "wgrad_up"],
                      "grad_of": ["x", "weight_gate", "weight_up"]},
}


def run(record_path, reference="rmsnorm_linear", arg_index=(0, 1, 2, 3)):
    rec = load_record(record_path)
    flat = []

    def _walk(n):
        if isinstance(n, dict) and n.get("__t__"):
            flat.append(n)
        elif isinstance(n, list):
            [_walk(v) for v in n]
        elif isinstance(n, dict):
            [_walk(v) for v in n.values()]
    _walk(rec["args"])
    tens = [_t(flat[i]) for i in arg_index if i < len(flat)]
    x, ln_w, w, b = (tens + [None] * 4)[:4]

    iso = rec.get("isolated") or {}
    ref_out = REFERENCES[reference](x, ln_w, w, b)
    # Prefer the ISOLATED re-run: same op, same frozen inputs, one call, no pairing ambiguity.
    captured_out = _t(iso.get("out")) if iso.get("out") is not None else None
    if captured_out is None:
        captured_out = _t(rec.get("output")) if rec.get("output") is not None else None
    if captured_out is None:
        # output may be a tuple snapshot
        out_node = rec.get("output")
        if isinstance(out_node, list) and out_node:
            captured_out = _t(out_node[0])
    if captured_out is None:
        return {"error": "no captured forward output in the record"}

    base = {"out": error_stats(captured_out, ref_out)}
    # Backward: compare the original's captured input grads against the reference's own grads.
    go = _t(rec.get("grad_output"))
    src_grads = iso.get("grads") or rec.get("input_grads") or {}
    if go is not None and src_grads:
        # Match captured gradients to inputs BY SHAPE, not by index. The backward's return tuple is
        # aligned with ALL of the forward's parameters (including the non-tensor ones), so its
        # indices do not line up with the tensor list -- observed directly: the captured keys were
        # ['0','1','3'], with no '2', because the weight gradient is absent from the returned tuple
        # under gradient_accumulation_fusion (it goes straight into main_grad).
        by_shape = {}
        for k, node in src_grads.items():
            if isinstance(node, dict) and node.get("__t__"):
                by_shape.setdefault(tuple(node["shape"]), []).append((k, node))
        xg = x.clone().requires_grad_(True)
        lg = ln_w.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        o = REFERENCES[reference](xg, lg, wg, b)
        grads = torch.autograd.grad(o, [xg, lg, wg], grad_outputs=go.float(), allow_unused=True)
        for label, tensor, g in (("dgrad", x, grads[0]), ("dln_weight", ln_w, grads[1]),
                                 ("wgrad", w, grads[2])):
            if g is None or tensor is None:
                continue
            match = by_shape.get(tuple(tensor.shape))
            if not match:
                base[label] = {"missing": True,
                               "why": (f"the op's backward returned no gradient of shape "
                                       f"{list(tensor.shape)}; under gradient_accumulation_fusion "
                                       f"the weight gradient goes into main_grad instead")}
                continue
            base[label] = error_stats(_t(match[0][1]), g)
    return {"reference": reference, "base_error": base,
            "note": ("这 is the error the ORIGINAL carries; a candidate is judged against it, "
                     "not against a hand-picked tolerance").replace("这 is", "this is")}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rmsnorm_linear"),
                     indent=2, default=float))
