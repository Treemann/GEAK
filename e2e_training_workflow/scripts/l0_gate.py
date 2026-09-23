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
    }


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


def linear_reference(w, x, b=None):
    """Independent reference for TE's _Linear. NOTE the argument order: `_Linear.apply` takes the
    WEIGHT first, unlike `_LayerNormLinear`. Reading the order off the capture instead of assuming it
    is the difference between an oracle and a plausible-looking pile of wrong numbers."""
    return torch.nn.functional.linear(x.float(), w.float(), b.float() if b is not None else None)


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


REFERENCES = {"rmsnorm_linear": rmsnorm_linear_reference, "linear": linear_reference,
              "attention_varlen": attention_varlen_reference,
              "lmhead_ce": lmhead_ce_reference}

# Which captured tensor plays which role, per reference. The names are the ORDER the reference's
# signature expects, and they are also the labels the oracle's outputs carry.
LAYOUTS = {
    "rmsnorm_linear": {"args": ["x", "ln_weight", "weight", "bias"],
                       "grad_labels": ["dgrad", "dln_weight", "wgrad"],
                       "grad_of": ["x", "ln_weight", "weight"]},
    "linear": {"args": ["weight", "x", "bias"],
               "grad_labels": ["wgrad", "dgrad"],
               "grad_of": ["weight", "x"]},
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
