"""Capture the UNFUSED RoPE applied to q AND k (one attention layer, one microbatch) fwd+bwd.

WHAT THE OP IS. The workload runs `--no-rope-fusion` (the TE fused thd RoPE path fails on this
TE/Megatron pair, knowledge/megatron_internals.md), so Megatron's `SelfAttention.forward` calls
`megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb` TWICE per layer -- once for
the query ([t, 28, 128]) and once for the key ([t, 4, 128]) -- and each call goes through
`_apply_rotary_pos_emb_thd` -> `_apply_rotary_pos_emb_bshd`: a per-segment `torch.cat` of freqs, the
cos/sin, `t*cos + rotate_half(t)*sin` (chunk, neg, cat, two muls, an add) and a final `cat` with the
empty pass-through slice. Those are the `elementwise_kernel_manual_unroll mul` + `CatArrayBatchedCopy`
+ add kernels the profile attributes 2.5% of the critical path to. The head wants ONE fused kernel for
q AND k, so the op frozen here is the PAIR, not a single call: inputs (q, k, cu_seqlens_q,
cu_seqlens_kv, freqs), outputs (q_rot, k_rot), gradients (dq, dk).

WHY THE CALL SITE IS `attention.apply_rotary_pos_emb`. `attention.py` does
`from ...rope_utils import apply_rotary_pos_emb`, so the name that is CALLED lives in the attention
module's namespace; patching `rope_utils.apply_rotary_pos_emb` would not intercept anything. The same
symbol is the integration bind point.

PAIRING q WITH k IS CHECKED, NOT ASSUMED. Within one `SelfAttention.forward` the query call is
immediately followed by the key call (attention.py: `query = apply_rotary_pos_emb(...)` then
`key = apply_rotary_pos_emb(...)`). The capture takes the first call of the capture step whose input
has the q head count, then requires the very NEXT call to have the kv head count, the same token
count and the same freqs tensor. Anything else aborts the capture loudly instead of freezing a
mismatched pair.

grad_output IS SOUND HERE. The rotated q and k are fresh tensors whose ONLY consumer is
`core_attention` (attention.py, "core attention computation"); value is not rotated. So a tensor hook
on each output is this op's exact grad_output, not a sum over consumers (the hazard named in
gate_convergence.md applies to tensors with several readers, which these are not).

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original `apply_rotary_pos_emb` on clones of the
frozen inputs, in the live process -- same doctrine as capture_train_shapes. The re-run must reproduce
the LIVE outputs; that check is recorded as composition evidence.

Env:
    GEAK_ROPE_CAPTURE_DIR=/work/tasks/rope_cap      (required; enables the capture)
    GEAK_ROPE_CAPTURE_STEP=13                      (capture on this step, after warmup)
    GEAK_ROPE_CAPTURE_RANKS=0                      (one rank: concurrent writers corrupt the file)
"""
from __future__ import annotations

import json
import os

import torch

_DIR = os.environ.get("GEAK_ROPE_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_ROPE_CAPTURE_STEP", "0") or 0)
_RANKS = {int(r) for r in os.environ.get("GEAK_ROPE_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)

_state = {"step": 0, "done": False, "q": None, "k": None, "grads": {}}


def note_step(step):
    _state["step"] = step


def _snap(x):
    """Same on-disk marker as capture_train_shapes, so build_task_dir reads this record unchanged."""
    if torch.is_tensor(x):
        return {"__t__": True, "data": x.detach().to("cpu", copy=True),
                "dtype": str(x.dtype), "shape": list(x.shape),
                "requires_grad": bool(x.requires_grad)}
    return x


def _layout(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "contiguous": t.is_contiguous(),
            "storage_offset": int(t.storage_offset()), "dtype": str(t.dtype)}


def install():
    if not ENABLED:
        return False
    try:
        from megatron.core.transformer import attention as _attn
    except Exception as e:
        print(f"[geak-rope] cannot import megatron.core.transformer.attention: {e!r}", flush=True)
        return False
    orig = getattr(_attn, "apply_rotary_pos_emb", None)
    if orig is None:
        print("[geak-rope] attention has no apply_rotary_pos_emb; not armed", flush=True)
        return False
    os.makedirs(_DIR, exist_ok=True)

    def armed():
        return (not _state["done"]) and _state["step"] + 1 == _STEP

    def abort(why):
        print(f"[geak-rope] NOT capturing: {why}", flush=True)
        _state["done"] = True

    def wrapped(t, freqs, *a, **kw):
        if not armed() or not torch.is_tensor(t) or not t.requires_grad:
            return orig(t, freqs, *a, **kw)
        out = orig(t, freqs, *a, **kw)
        if _state["q"] is None:
            # The first rotated tensor of the step is the query of layer 0, microbatch 0.
            _state["q"] = {"t": t, "freqs": freqs, "a": a, "kw": kw, "out": out}
            return out
        q = _state["q"]
        why = None
        if t.dim() != 3 or q["t"].dim() != 3:
            why = f"expected thd [t,h,d] inputs, got {list(q['t'].shape)} then {list(t.shape)}"
        elif t.shape[0] != q["t"].shape[0] or t.shape[-1] != q["t"].shape[-1]:
            why = f"second call {list(t.shape)} does not match the first {list(q['t'].shape)}"
        elif t.shape[1] >= q["t"].shape[1]:
            why = (f"second call has {t.shape[1]} heads, not fewer than the first's "
                   f"{q['t'].shape[1]}: this is not the (query, key) pair of one layer")
        if why:
            abort(why)
            return out
        _state["k"] = {"t": t, "freqs": freqs, "a": a, "kw": kw, "out": out}
        _state["done"] = True

        def mk_hook(name):
            def _hook(g):
                try:
                    _state["grads"][name] = g.detach().clone()
                    if len(_state["grads"]) == 2:
                        _capture(orig)
                except Exception as e:  # never kill an 8-rank job from a backward hook
                    print(f"[geak-rope] capture FAILED (run continues): {e!r}", flush=True)
                return g
            return _hook

        q["out"].register_hook(mk_hook("q"))
        out.register_hook(mk_hook("k"))
        print(f"[geak-rope] armed pair: q {list(q['t'].shape)} k {list(t.shape)}", flush=True)
        return out

    _attn.apply_rotary_pos_emb = wrapped
    print(f"[geak-rope] armed on megatron.core.transformer.attention.apply_rotary_pos_emb for step "
          f"{_STEP} (rank {_RANK})", flush=True)
    return True


def _cfg_summary(kw):
    cfg = kw.get("config")
    keys = ("apply_rope_fusion", "rotary_interleaved", "multi_latent_attention", "mrope_section",
            "context_parallel_size", "tensor_model_parallel_size", "sequence_parallel",
            "rotary_base", "rotary_percent", "params_dtype", "bf16", "fp16")
    return {k: repr(getattr(cfg, k, "<absent>")) for k in keys} if cfg is not None else {}


def _capture(orig):
    q, k, gq, gk = _state["q"], _state["k"], _state["grads"]["q"], _state["grads"]["k"]
    cu_q = q["kw"].get("cu_seqlens")
    cu_k = k["kw"].get("cu_seqlens")
    cp = q["kw"].get("cp_group")
    freqs = q["freqs"]

    qc = q["t"].detach().clone().requires_grad_(True)
    kc = k["t"].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        qo = orig(qc, freqs, *q["a"], **q["kw"])
        ko = orig(kc, k["freqs"], *k["a"], **k["kw"])
        dq, dk = torch.autograd.grad((qo, ko), (qc, kc), grad_outputs=(gq, gk))

    iso_vs_live = {"q": float((qo.detach().float() - q["out"].detach().float()).abs().max()),
                   "k": float((ko.detach().float() - k["out"].detach().float()).abs().max())}
    seg_q = (cu_q[1:] - cu_q[:-1]).tolist() if torch.is_tensor(cu_q) else None
    seg_k = (cu_k[1:] - cu_k[:-1]).tolist() if torch.is_tensor(cu_k) else None
    args = [q["t"], k["t"], cu_q, cu_k, freqs]
    evidence = {
        "isolated_vs_live_max_abs": iso_vs_live,
        "same_freqs_object_for_q_and_k": k["freqs"] is freqs,
        "freqs_equal_for_q_and_k": bool(torch.equal(k["freqs"], freqs)),
        "same_cu_seqlens_for_q_and_k": bool(torch.is_tensor(cu_q) and torch.is_tensor(cu_k)
                                            and torch.equal(cu_q, cu_k)),
        "cu_seqlens_q": cu_q.tolist() if torch.is_tensor(cu_q) else None,
        "segment_lengths_q": seg_q,
        "segment_lengths_k": seg_k,
        "freqs_rows_vs_total_tokens": [int(freqs.shape[0]),
                                       int(cu_q[-1]) if torch.is_tensor(cu_q) else None],
        "positions_reset_per_segment": (
            "yes: freqs has max_seqlen rows != cu_seqlens[-1], so _apply_rotary_pos_emb_thd takes "
            "CASE 2 (freqs[0:len] per segment)" if torch.is_tensor(cu_q)
            and int(freqs.shape[0]) != int(cu_q[-1]) else
            "CASE 1 (offset mapping) -- freqs rows == total tokens; check before trusting"),
        "input_layout": {"q": _layout(q["t"]), "k": _layout(k["t"]),
                         "q_out": _layout(q["out"]), "k_out": _layout(k["out"]),
                         "freqs": _layout(freqs)},
        "non_tensor_kwargs": {kk: repr(v)[:160] for kk, v in q["kw"].items()
                              if not torch.is_tensor(v) and kk != "config"},
        "cp_size": cp.size() if cp is not None else None,
        "config": _cfg_summary(q["kw"]),
        "grad_output_nonzero_frac": {"q": float((gq != 0).float().mean()),
                                     "k": float((gk != 0).float().mean())},
        "grad_output_abs_mean": {"q": float(gq.abs().mean()), "k": float(gk.abs().mean())},
        "grad_output_abs_max": {"q": float(gq.abs().max()), "k": float(gk.abs().max())},
        "note": ("grad_output is captured from the rotated q / k, whose sole consumer is "
                 "core_attention, so each hook value is this op's exact gradient."),
    }
    rec = {
        "target": "megatron.core.transformer.attention:apply_rotary_pos_emb"
                  " [query call + key call of one SelfAttention.forward, thd, unfused]",
        "step": _STEP, "rank": _RANK,
        "args": [_snap(t) for t in args],
        "kwargs": {},
        "grad_output": [_snap(gq), _snap(gk)],
        "output": [_snap(q["out"].detach()), _snap(k["out"].detach())],
        "isolated": {
            "out": [_snap(qo.detach()), _snap(ko.detach())],
            "grads": {"0": _snap(dq), "1": _snap(dk)},
            "grad_shapes": [list(qc.shape), list(kc.shape)],
        },
        "composition_evidence": evidence,
        "opaque_args": [{"__opaque__": type(v).__name__, "module": getattr(type(v), "__module__", "?"),
                         "repr": repr(v)[:200]} for kk, v in q["kw"].items()
                        if not torch.is_tensor(v) and v is not None and kk in ("config", "cp_group")],
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}.pt")
    tmp = path + ".partial"
    torch.save(rec, tmp)
    os.replace(tmp, path)
    with open(os.path.join(_DIR, "capture_meta.json"), "w") as f:
        json.dump({"target": rec["target"], "step": _STEP, "rank": _RANK,
                   "shapes": {"args": [list(t.shape) for t in args],
                              "grad_output": [list(gq.shape), list(gk.shape)]},
                   "dtypes": {"args": [str(t.dtype) for t in args],
                              "grad_output": [str(gq.dtype), str(gk.dtype)]},
                   "composition_evidence": evidence}, f, indent=2, default=str)
    print(f"[geak-rope] wrote {path}; iso_vs_live={iso_vs_live} segs={seg_q}", flush=True)
