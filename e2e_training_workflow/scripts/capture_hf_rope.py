"""Capture HF Qwen2 `apply_rotary_pos_emb` (q AND k of one layer, fwd + bwd) AS IT RUNS on the hf_trl
stack. Head `hf_rope_layout_fused`.

WHAT THE OP IS. transformers 4.55.0 `modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)`:

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                       # [1,1,S,128]
    q_embed  = q * cos + rotate_half(q) * sin    rotate_half(x) = cat(-x[..., 64:], x[..., :64])
    k_embed  = k * cos + rotate_half(k) * sin

q [1,28,S,128] and k [1,4,S,128] are bf16 TRANSPOSED VIEWS of the q_proj / k_proj outputs
(`.view(1,S,H,128).transpose(1,2)`, strides (S*H*128, 128, H*128, 1)); cos / sin [1,S,128] come from
Qwen2RotaryEmbedding (computed in fp32, cast to the hidden dtype) and are NOT differentiable (that
forward runs under no_grad). Outputs q_embed / k_embed; gradients dq, dk. One call per layer per
microbatch: 28 x 8 = 224 calls per rank per step. The live kernels: 2 x (mul, slice+neg, cat, mul,
add) forward and the mul / neg / slice-backward / add chain backward, plus -- outside this function
but caused by its output layout -- the `.contiguous()` copies sdpa_attention_forward makes of q / k
(BSHD-backed -> BHSD) and the copy ViewBackward makes of dq / dk on the way back to q_proj's output.

THE CALL SITE. Qwen2Attention.forward looks `apply_rotary_pos_emb` up in modeling_qwen2's module
globals at call time. On hf5 Qwen2Attention.forward is the hf_attn_qkvo_proj wrapper, but it calls
_PREV (the ORIGINAL body) which still resolves that global -- so rebinding
`transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb` intercepts every call on the stack,
and it is the same bind point an integration uses. Nothing in hf5 rebinds it (checked at install and
recorded).

grad_output IS SOUND HERE. Tensor hooks on q_embed and k_embed report d loss / d q_embed and
d loss / d k_embed -- the op's grad_outputs by definition (the consumer-sum hazard concerns INPUTS
with several consumers; no gradient is read from a hook on q or k: dq / dk come from the isolated
re-run). Their LAYOUT is recorded as well: it is what a candidate's backward will receive (the CK
attention backward returns dq / dk as BSHD-backed views).

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original function on leaves rebuilt with the LIVE
strides (empty_strided + copy_), in the live process, inside the second of the two grad_output hooks.
Recorded per frozen call: isolated outputs == live outputs (bitwise), a second isolated re-run ==
the first (bitwise), live / isolated output and grad strides, whether the downstream .contiguous()
of q_embed / k_embed is a copy, and cos / sin facts (same object across layers, max error against a
fresh fp32 rotary table at positions arange(S), i.e. position_ids == arange(S)).

Env:
    GEAK_HFROPE_CAPTURE_DIR=/work/tasks/<name>_cap   (required; enables the capture)
    GEAK_HFROPE_CAPTURE_STEP=12                      (1-based step to capture, after warmup)
    GEAK_HFROPE_CAPTURE_CALLS=0,14,27,101,210        (per-step call indices to freeze;
                                                      call c = layer c % 28 of microbatch c // 28)
    GEAK_HFROPE_CAPTURE_RANKS=0,4                    (one file per rank and call; no shared writer)
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFROPE_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFROPE_CAPTURE_STEP", "0") or 0)
_CALLS = {int(c) for c in os.environ.get("GEAK_HFROPE_CAPTURE_CALLS", "0,14,27,101,210").split(",")
          if c.strip().isdigit()}
_RANKS = {int(r) for r in os.environ.get("GEAK_HFROPE_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
TARGET = "transformers.models.qwen2.modeling_qwen2:apply_rotary_pos_emb"
ROLES = ["q", "k", "cos", "sin"]          # == layout `hf_rope_qk` args (l0_gate.LAYOUTS)

_state = {"step": 0, "calls": 0, "per_call": [], "written": [], "meta_done": False,
          "cos_ids": {}, "orig": None, "rope_theta": None, "install_facts": {}}


def note_step(step):
    """Called at every step end with the number of steps COMPLETED."""
    _state["step"] = step
    if step >= _STEP and not _state["meta_done"] and _state["per_call"]:
        _state["meta_done"] = True
        _write_capture_meta()


def _snap(t):
    # .to("cpu", copy=True) keeps the strides of a non-overlapping dense tensor: the frozen q / k are
    # the live BSHD-backed transposed views, not contiguous copies.
    return {"__t__": True, "data": t.detach().to("cpu", copy=True), "dtype": str(t.dtype),
            "shape": list(t.shape), "requires_grad": bool(t.requires_grad)}


def _lay(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "contiguous": t.is_contiguous(),
            "dtype": str(t.dtype), "storage_offset": int(t.storage_offset()),
            "is_view": t._base is not None,
            "base_shape": list(t._base.shape) if t._base is not None else None}


def _like(t, grad):
    """A leaf with the SAME strides as t (empty_strided + copy_), so the isolated re-run sees the
    live layout, not a contiguous clone."""
    c = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    c.copy_(t.detach())
    return c.requires_grad_(True) if grad else c


def _isolated(fr, gos):
    orig = _state["orig"]
    q, k = _like(fr["q"], True), _like(fr["k"], True)
    with torch.enable_grad():
        qe, ke = orig(q, k, fr["cos"], fr["sin"])
        dq, dk = torch.autograd.grad((qe, ke), (q, k), grad_outputs=gos)
    return [qe.detach(), ke.detach()], [dq.detach(), dk.detach()]


def _rope_table(S, dev):
    """A fresh fp32 cos/sin table at positions arange(S) (Qwen2RotaryEmbedding, default rope)."""
    th = _state["rope_theta"] or 10000.0
    inv = 1.0 / (th ** (torch.arange(0, 128, 2, dtype=torch.int64, device=dev).float() / 128))
    fr = torch.outer(torch.arange(S, device=dev, dtype=torch.float32), inv)
    emb = torch.cat((fr, fr), -1)
    return emb.cos(), emb.sin()


def install(model=None):
    if not ENABLED:
        return False
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfrope] cannot import modeling_qwen2: {e!r}", flush=True)
        return False
    orig = mq.apply_rotary_pos_emb
    _state["orig"] = orig
    try:
        _state["rope_theta"] = float(getattr(model.config, "rope_theta", 10000.0))
        _state["install_facts"]["rope_scaling"] = repr(getattr(model.config, "rope_scaling", None))
    except Exception:  # noqa: BLE001
        pass
    fw = mq.Qwen2Attention.forward
    _state["install_facts"].update({
        "apply_rotary_pos_emb_before_capture": f"{getattr(orig, '__module__', '?')}.{getattr(orig, '__qualname__', '?')}",
        "is_the_original": getattr(orig, "__module__", "") == mq.__name__,
        "Qwen2Attention.forward": f"{getattr(fw, '__module__', '?')}.{getattr(fw, '__qualname__', '?')}",
        "rope_theta": _state["rope_theta"]})
    os.makedirs(_DIR, exist_ok=True)

    def wrapped(q, k, cos, sin, *a, **kw):
        if _state["step"] + 1 != _STEP:
            return orig(q, k, cos, sin, *a, **kw)
        c = _state["calls"]
        _state["calls"] += 1
        S = int(q.shape[-2])
        cid = id(cos)
        first_use = cid not in _state["cos_ids"]
        _state["cos_ids"].setdefault(cid, c)
        row = {"call": c, "layer": c % 28, "microbatch": c // 28, "S": S,
               "q": _lay(q), "k": _lay(k), "cos": _lay(cos), "sin": _lay(sin),
               "extra_args": [repr(x)[:80] for x in a], "extra_kwargs": {kk: repr(v)[:80] for kk, v in kw.items()},
               "q_requires_grad": bool(q.requires_grad), "k_requires_grad": bool(k.requires_grad),
               "cos_requires_grad": bool(cos.requires_grad), "sin_requires_grad": bool(sin.requires_grad),
               "cos_first_seen_at_call": _state["cos_ids"][cid], "cos_new_object": first_use,
               "autocast": {"enabled": torch.is_autocast_enabled("cuda"),
                            "dtype": str(torch.get_autocast_dtype("cuda"))},
               "grad_enabled": torch.is_grad_enabled()}
        frz = c in _CALLS and q.requires_grad and k.requires_grad
        fr = ({"q": _like(q, False), "k": _like(k, False), "cos": cos.detach().clone(),
               "sin": sin.detach().clone()} if frz else None)
        qe, ke = orig(q, k, cos, sin, *a, **kw)
        row["q_embed"] = _lay(qe)
        row["k_embed"] = _lay(ke)
        _state["per_call"].append(row)
        if first_use and S > 0:
            try:
                rc, rs = _rope_table(S, cos.device)
                row["cos_vs_fresh_fp32_arange_max_abs"] = float((cos[0].float() - rc).abs().max())
                row["sin_vs_fresh_fp32_arange_max_abs"] = float((sin[0].float() - rs).abs().max())
                row["cos_equals_round_fresh"] = bool(torch.equal(cos[0], rc.to(cos.dtype)))
                row["sin_equals_round_fresh"] = bool(torch.equal(sin[0], rs.to(sin.dtype)))
            except Exception as e:  # noqa: BLE001
                row["rope_table_check_err"] = repr(e)[:200]
        if not frz:
            return qe, ke
        pend = {"c": c, "row": row, "fr": fr, "live": [qe.detach().clone(), ke.detach().clone()],
                "gos": {}, "done": False}

        def mk(slot):
            def _hook(g):
                try:
                    pend["gos"][slot] = g.detach().clone()
                    pend["row"].setdefault("grad_output_layout", {})[slot] = _lay(g)
                    if len(pend["gos"]) == 2 and not pend["done"]:
                        pend["done"] = True
                        _finish(pend)
                except Exception as e:  # noqa: BLE001  -- a capture must never kill the run
                    print(f"[geak-hfrope] capture of call {c} FAILED (run continues): {e!r}", flush=True)
                return g
            return _hook

        qe.register_hook(mk("q"))
        ke.register_hook(mk("k"))
        return qe, ke

    mq.apply_rotary_pos_emb = wrapped
    print(f"[geak-hfrope] armed on {TARGET} for step {_STEP}, calls {sorted(_CALLS)}, rank {_RANK}; "
          f"facts {json.dumps(_state['install_facts'])}", flush=True)
    return True


def _finish(pend):
    t0 = time.time()
    c, row, fr = pend["c"], pend["row"], pend["fr"]
    gos = [pend["gos"]["q"], pend["gos"]["k"]]
    live = pend["live"]
    o1, g1 = _isolated(fr, gos)
    o2, g2 = _isolated(fr, gos)
    rec = {"target": TARGET, "step": _STEP, "rank": _RANK, "call": c,
           "module": f"model.layers.{row['layer']}.self_attn (microbatch {row['microbatch']})",
           "args": [_snap(fr[r]) for r in ROLES], "kwargs": {}, "const": {},
           "arg_roles": ROLES, "autocast": row["autocast"],
           "output": [_snap(t) for t in live],
           "grad_output": [_snap(g) for g in gos]}
    rec["isolated"] = {"out": [_snap(t) for t in o1],
                       "grads": {str(i): _snap(g) for i, g in enumerate(g1)},
                       "grad_shapes": [list(g.shape) for g in g1]}
    S = row["S"]
    rc, rs = _rope_table(S, fr["cos"].device)
    rec["composition_evidence"] = {
        "iso_out_equals_live_out": [bool(torch.equal(a, b)) for a, b in zip(o1, live)],
        "iso_out_vs_live_max_abs": [float((a.float() - b.float()).abs().max()) for a, b in zip(o1, live)],
        "iso_rerun_bitwise_out": [bool(torch.equal(a, b)) for a, b in zip(o1, o2)],
        "iso_rerun_bitwise_grads": [bool(torch.equal(a, b)) for a, b in zip(g1, g2)],
        "layouts": {"q": row["q"], "k": row["k"], "cos": row["cos"], "sin": row["sin"],
                    "q_embed_live": row["q_embed"], "k_embed_live": row["k_embed"],
                    "grad_output": row.get("grad_output_layout"),
                    "q_embed_iso": _lay(o1[0]), "k_embed_iso": _lay(o1[1]),
                    "dq_iso": _lay(g1[0]), "dk_iso": _lay(g1[1])},
        "downstream_contiguous_is_copy": {"q": not row["q_embed"]["contiguous"],
                                          "k": not row["k_embed"]["contiguous"]},
        "cos_vs_fresh_fp32_arange_max_abs": float((fr["cos"][0].float() - rc).abs().max()),
        "sin_vs_fresh_fp32_arange_max_abs": float((fr["sin"][0].float() - rs).abs().max()),
        "cos_equals_round_fresh": bool(torch.equal(fr["cos"][0], rc.to(fr["cos"].dtype))),
        "sin_equals_round_fresh": bool(torch.equal(fr["sin"][0], rs.to(fr["sin"].dtype))),
        "cos_halves_equal": bool(torch.equal(fr["cos"][..., :64], fr["cos"][..., 64:])),
        "sin_halves_equal": bool(torch.equal(fr["sin"][..., :64], fr["sin"][..., 64:])),
        "out_dtypes": [str(t.dtype) for t in o1], "grad_dtypes": [str(g.dtype) for g in g1],
        "go_dtypes": [str(g.dtype) for g in gos],
        "go_abs_mean": [float(g.float().abs().mean()) for g in gos],
        "go_abs_max": [float(g.float().abs().max()) for g in gos],
        "go_zero_frac": [float((g == 0).float().mean()) for g in gos],
        "q_abs_max": float(fr["q"].float().abs().max()), "k_abs_max": float(fr["k"].float().abs().max()),
        "out_abs_max": [float(t.float().abs().max()) for t in live],
        "grad_abs_mean": [float(g.float().abs().mean()) for g in g1],
        "grad_abs_max": [float(g.float().abs().max()) for g in g1],
        "note": ("grad_output = tensor hooks on q_embed / k_embed (the op's outputs): d loss / d out by "
                 "definition. dq / dk = isolated re-run of the original on stride-preserving leaves."),
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}_call{c}.pt")
    torch.save(rec, path + ".partial")
    os.replace(path + ".partial", path)
    _state["written"].append(path)
    ev = rec["composition_evidence"]
    print(f"[geak-hfrope] wrote {path} {rec['module']} S={S} iso==live {ev['iso_out_equals_live_out']} "
          f"rerun {ev['iso_rerun_bitwise_out']}/{ev['iso_rerun_bitwise_grads']} "
          f"go_layout {[(v['stride'], v['contiguous']) for v in (row.get('grad_output_layout') or {}).values()]} "
          f"({time.time()-t0:.1f}s)", flush=True)


def _write_capture_meta():
    per = _state["per_call"]
    Ss = sorted({r["S"] for r in per})
    meta = {"target": TARGET, "step": _STEP, "rank": _RANK, "calls_in_step": len(per),
            "frozen_calls": sorted(_CALLS), "written": _state["written"], "distinct_S": Ss,
            "install_facts": _state["install_facts"],
            "q_layouts": sorted({json.dumps({kk: r["q"][kk] for kk in ("contiguous", "dtype", "is_view")})
                                 for r in per}),
            "q_embed_contiguous_any": any(r["q_embed"]["contiguous"] for r in per),
            "k_embed_contiguous_any": any(r["k_embed"]["contiguous"] for r in per),
            "cos_dtypes": sorted({r["cos"]["dtype"] for r in per}),
            "cos_requires_grad_any": any(r["cos_requires_grad"] for r in per),
            "distinct_cos_objects": len(_state["cos_ids"]),
            "autocast": sorted({json.dumps(r["autocast"]) for r in per}),
            "per_call": per}
    try:
        with open(os.path.join(_DIR, f"capture_meta_rank{_RANK}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[geak-hfrope] step {_STEP}: {len(per)} rope calls, S={Ss}, "
              f"{len(_state['cos_ids'])} distinct cos objects", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfrope] meta write failed: {e!r}", flush=True)
