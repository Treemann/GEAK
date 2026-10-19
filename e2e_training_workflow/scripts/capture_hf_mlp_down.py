"""Capture HF `Qwen2MLP.down_proj` (fwd + bwd) AS IT RUNS on the hf_trl stack. Head `hf_mlp_down_proj`.

WHAT THE OP IS. down_proj is a bias-free nn.Linear(18944 -> 3584):

    out = h @ Wd^T        h [1,M,18944] bf16 (SwiGLU output), Wd [3584,18944] bf16, out [1,M,3584]

forward: one GEMM [M,18944]x[18944,3584]; backward: dgrad dh = go @ Wd ([M,3584]x[3584,18944]) and
wgrad dWd = go^T @ h ([3584,M]x[M,18944]). 28 calls per sample, 224 per rank per step.

WHY MODULE HOOKS ON THE 28 down_proj INSTANCES (not a class rebind). On the accepted hf3 stack
Qwen2MLP.forward is ALREADY rebound by hf_mlp_gateup_swiglu, whose forward ends in
`self.down_proj(entry(x, wg, wu))` -- the module call. Forward pre/post hooks on the down_proj
instances therefore see exactly the live call on top of the stack, without touching the stack's
rebind (the candidate must CHAIN on it, see engagement notes). The live forward is not altered.

grad_output IS SOUND HERE. The tensor hook is on down_proj's OUTPUT, whose sole consumer is the
decoder layer's residual add; a tensor hook reports d loss / d out, which is this op's grad_output
by definition (the consumer-sum hazard is on INPUTS that feed several consumers, not on outputs).

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original (`nn.Linear.forward` of THIS module, called
unbound so the capture's own hooks do not re-fire) on clones of the frozen h and Wd, in the live
process, inside the grad_output hook. The weight is swapped for a cloned leaf for that one call
(the live dWd is summed over 8 microbatches into FSDP's flat grad and is unobservable). Clones are
taken at FORWARD time. Autocast state is recorded and re-entered (the hook runs on the autograd
thread, where autocast is off). Recorded per frozen call: isolated out == live out (bitwise), second
isolated re-run == first (out, dh, dWd bitwise), and the live weight's storage facts (what a
"wgrad accumulates into .grad with beta=1" lever could assume).

Env:
    GEAK_HFDOWN_CAPTURE_DIR=/work/tasks/<name>_cap   (required; enables the capture)
    GEAK_HFDOWN_CAPTURE_STEP=12                      (1-based step to capture, after warmup)
    GEAK_HFDOWN_CAPTURE_CALLS=0,14,27,210            (per-step down_proj call indices to freeze)
    GEAK_HFDOWN_CAPTURE_RANKS=0,4                    (one file per rank and call; no shared writer)
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFDOWN_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFDOWN_CAPTURE_STEP", "0") or 0)
_CALLS = {int(c) for c in os.environ.get("GEAK_HFDOWN_CAPTURE_CALLS", "0,14,27,210").split(",")
          if c.strip().isdigit()}
_RANKS = {int(r) for r in os.environ.get("GEAK_HFDOWN_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
TARGET = "transformers.models.qwen2.modeling_qwen2:Qwen2MLP.down_proj"

_state = {"step": 0, "calls": 0, "names": {}, "per_call": [], "written": [], "meta_done": False,
          "pending": {}}


def note_step(step):
    """Called at every step end with the number of steps COMPLETED."""
    _state["step"] = step
    if step >= _STEP and not _state["meta_done"] and _state["per_call"]:
        _state["meta_done"] = True
        _write_capture_meta()


def _snap(t):
    return {"__t__": True, "data": t.detach().to("cpu", copy=True), "dtype": str(t.dtype),
            "shape": list(t.shape), "requires_grad": bool(t.requires_grad)}


def _swap_weight(mod, w):
    """Point mod.weight at w for one call (FSDP use_orig_params keeps a plain-tensor view in the
    instance __dict__, shadowing `_parameters`; checked first, identity-verified after)."""
    if "weight" in mod.__dict__:
        tok = ("d", mod.__dict__["weight"])
        mod.__dict__["weight"] = w
    elif "weight" in mod._parameters:
        tok = ("p", mod._parameters["weight"])
        mod._parameters["weight"] = w
    else:
        raise RuntimeError(f"{type(mod).__name__} has no weight attribute to swap")
    if mod.weight is not w:
        _restore_weight(mod, tok)
        raise RuntimeError("weight swap did not take effect (attribute lookup bypasses it)")
    return tok


def _restore_weight(mod, tok):
    kind, old = tok
    if kind == "p":
        mod._parameters["weight"] = old
    else:
        mod.__dict__["weight"] = old


def _isolated(lin, h0, w0, go, ac):
    hc = h0.detach().clone().requires_grad_(True)
    wc = w0.detach().clone().requires_grad_(True)
    tok = _swap_weight(lin, wc)
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=ac["dtype"], enabled=ac["enabled"]):
            # The ORIGINAL op: nn.Linear.forward of this very module (unbound -> no hooks re-fire).
            out = type(lin).forward(lin, hc)
        dh, dw = torch.autograd.grad(out, (hc, wc), grad_outputs=go)
    finally:
        _restore_weight(lin, tok)
    return out.detach(), dh.detach(), dw.detach()


def _storage_facts(w):
    f = {"w_type": type(w).__name__, "w_is_param": isinstance(w, torch.nn.Parameter),
         "w_is_view": w._base is not None, "w_contig": w.is_contiguous(),
         "w_stride": list(w.stride()), "w_requires_grad": bool(w.requires_grad),
         "w_grad_is_none": w.grad is None if w.is_leaf else "non-leaf"}
    try:
        if w._base is not None:
            b = w._base
            f["base_shape"] = list(b.shape)
            f["base_dtype"] = str(b.dtype)
            f["offset_elems"] = (w.data_ptr() - b.data_ptr()) // w.element_size()
            f["base_grad_is_none"] = b.grad is None
    except Exception as e:  # noqa: BLE001
        f["err"] = repr(e)[:200]
    return f


def install(model=None):
    if not ENABLED:
        return False
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfdown] cannot import Qwen2MLP: {e!r}", flush=True)
        return False
    if model is None:
        print("[geak-hfdown] no model handed over; cannot hook down_proj instances", flush=True)
        return False
    os.makedirs(_DIR, exist_ok=True)
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, mq.Qwen2MLP):
            lin = m.down_proj
            _state["names"][id(lin)] = name + ".down_proj"
            lin.register_forward_pre_hook(_pre)
            lin.register_forward_hook(_post)
            n += 1
    print(f"[geak-hfdown] armed on {n} down_proj modules for step {_STEP}, calls {sorted(_CALLS)}, "
          f"rank {_RANK}; Qwen2MLP.forward currently = "
          f"{getattr(mq.Qwen2MLP.forward, '__module__', '?')}.{getattr(mq.Qwen2MLP.forward, '__qualname__', '?')}",
          flush=True)
    return n > 0


def _pre(mod, inp):
    if _state["step"] + 1 != _STEP:
        return None
    h = inp[0]
    k = _state["calls"]
    _state["calls"] += 1
    w = mod.weight
    ac = {"enabled": torch.is_autocast_enabled("cuda"), "dtype": torch.get_autocast_dtype("cuda")}
    row = {"call": k, "module": _state["names"].get(id(mod), "?"), "M": int(h.numel() // h.shape[-1]),
           "h_shape": list(h.shape), "h_dtype": str(h.dtype), "h_contig": h.is_contiguous(),
           "h_requires_grad": bool(h.requires_grad), "w_dtype": str(w.dtype), "w_shape": list(w.shape),
           "bias": mod.bias is not None, "autocast": {"enabled": ac["enabled"], "dtype": str(ac["dtype"])},
           "grad_enabled": torch.is_grad_enabled(), "storage": _storage_facts(w)}
    _state["per_call"].append(row)
    if k in _CALLS:
        _state["pending"][id(mod)] = (k, row, h.detach().clone(), w.detach().clone(), ac, inp)
    return None


def _post(mod, inp, out):
    p = _state["pending"].pop(id(mod), None)
    if p is None:
        return None
    k, row, h0, w0, ac, _ = p
    _state["per_call"][k]["out_dtype"] = str(out.dtype)
    if not out.requires_grad:
        print(f"[geak-hfdown] call {k}: output not differentiable, not frozen", flush=True)
        return None
    live_out = out.detach().clone()
    rec = {"target": TARGET, "step": _STEP, "rank": _RANK, "call": k, "module": row["module"],
           "args": [_snap(w0), _snap(h0)], "kwargs": {}, "const": {},
           "arg_roles": ["weight", "x"], "bias": None,
           "autocast": row["autocast"], "storage": row["storage"], "output": _snap(live_out)}

    def _hook(g):
        try:
            t0 = time.time()
            go = g.detach().clone()
            rec["grad_output"] = _snap(go)
            o1, dh1, dw1 = _isolated(mod, h0, w0, go, ac)
            o2, dh2, dw2 = _isolated(mod, h0, w0, go, ac)
            # layout `linear`: grad_labels [wgrad, dgrad] = grads of [weight, x], keyed by arg index.
            rec["isolated"] = {"out": _snap(o1), "grads": {"0": _snap(dw1), "1": _snap(dh1)},
                               "grad_shapes": [list(dw1.shape), list(dh1.shape)]}
            gz = go.reshape(-1, go.shape[-1]).abs().sum(-1) == 0
            hz = h0.reshape(-1, h0.shape[-1]).abs().sum(-1) == 0
            rec["composition_evidence"] = {
                "iso_out_equals_live_out": bool(torch.equal(o1, live_out)),
                "iso_out_vs_live_max_abs": float((o1.float() - live_out.float()).abs().max()),
                "iso_rerun_bitwise": [bool(torch.equal(o1, o2)), bool(torch.equal(dh1, dh2)),
                                      bool(torch.equal(dw1, dw2))],
                "dtypes": [str(o1.dtype), str(dh1.dtype), str(dw1.dtype)],
                "go_dtype": str(go.dtype), "go_abs_mean": float(go.float().abs().mean()),
                "go_abs_max": float(go.float().abs().max()), "go_zero_rows": int(gz.sum()),
                "h_zero_rows": int(hz.sum()),
                "h_abs_max": float(h0.float().abs().max()),
                "out_abs_max": float(live_out.float().abs().max()),
                "w_abs_max": float(w0.float().abs().max()),
                "dh_abs_mean": float(dh1.float().abs().mean()),
                "dw_abs_mean": float(dw1.float().abs().mean()),
                "dw_abs_max": float(dw1.float().abs().max()),
            }
            path = os.path.join(_DIR, f"reference_io_rank{_RANK}_call{k}.pt")
            torch.save(rec, path + ".partial")
            os.replace(path + ".partial", path)
            _state["written"].append(path)
            print(f"[geak-hfdown] wrote {path} module={row['module']} M={row['M']} "
                  f"evidence={json.dumps(rec['composition_evidence'])} ({time.time()-t0:.1f}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001  -- a capture must never kill the run
            print(f"[geak-hfdown] capture of call {k} FAILED (run continues): {e!r}", flush=True)
        return g

    out.register_hook(_hook)
    return None


def _write_capture_meta():
    per = _state["per_call"]
    Ms = sorted({r["M"] for r in per})
    meta = {"target": TARGET, "step": _STEP, "rank": _RANK, "calls_in_step": len(per),
            "frozen_calls": sorted(_CALLS), "distinct_M": Ms,
            "h_dtypes": sorted({r["h_dtype"] for r in per}),
            "w_dtypes": sorted({r["w_dtype"] for r in per}),
            "out_dtypes": sorted({r.get("out_dtype", "?") for r in per}),
            "bias": sorted({r["bias"] for r in per}),
            "autocast": sorted({json.dumps(r["autocast"]) for r in per}),
            "storage_kinds": sorted({json.dumps({k: v for k, v in r["storage"].items()
                                                 if k not in ("offset_elems",)}) for r in per}),
            "per_call": per}
    try:
        with open(os.path.join(_DIR, f"capture_meta_rank{_RANK}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[geak-hfdown] step {_STEP}: {len(per)} down_proj calls, M={Ms}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfdown] meta write failed: {e!r}", flush=True)
