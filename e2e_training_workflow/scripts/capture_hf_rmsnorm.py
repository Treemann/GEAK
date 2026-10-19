"""Capture HF `Qwen2RMSNorm` (fwd + bwd) AS IT RUNS on the hf_trl stack. Head `hf_rmsnorm_fused_fwdbwd`.

WHAT THE OP IS. transformers 4.55 `Qwen2RMSNorm.forward` is an unfused chain:

    h = x.to(float32); var = h.pow(2).mean(-1, keepdim=True); h = h * rsqrt(var + eps)
    return self.weight * h.to(x.dtype)

-- aten _to_copy / pow / mean / add / rsqrt / mul / _to_copy / mul in forward and their autograd
mirror in backward. 57 calls per sample: per decoder layer `input_layernorm` and
`post_attention_layernorm` (28 x 2), plus `model.norm` before the LM head. All share the class, the
shape [1, M, 3584] (M = the sample's unpadded length at MBS 1) and eps. The op frozen here is ONE call:
inputs (x [1,M,3584], weight [3584], eps), output out [1,M,3584], gradients dgrad (dx) and dln_weight
(dweight).

WHY THE HOOK IS ON `Qwen2RMSNorm.forward` (the CLASS attribute). The module's __call__ looks `forward`
up on the class at call time, so a class-level rebind reaches every instance including model.norm,
whenever it is installed. That is also the integration bind point.

WHICH CALL. Calls are counted per step in forward order. With gradient checkpointing off (hf_sft.py
sets gradient_checkpointing=False) and no recompute, call k of the step is microbatch k // 57,
position k % 57 -- but the position is NOT trusted: install(model) maps id(module) -> qualified name
before FSDP wraps the model (FSDP wraps decoder layers; it does not replace the norm modules), and
every frozen record carries the module's own name.

grad_output IS SOUND HERE. The hook is on this call's OUTPUT tensor. A tensor hook fires once with the
gradient SUMMED over the tensor's consumers -- for the output that sum IS the op's grad_output (input
norm -> q/k/v projections, post norm -> gate/up projections, final norm -> lm_head). The consumer-sum
hazard of gate_convergence.md is on the INPUT side (x also feeds the residual add), which is why dgrad
is never read from x's hook.

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the ORIGINAL forward (the unwrapped HF function object) on
clones of the frozen x AND weight, in the live process, inside the grad_output hook. The weight is
swapped for a cloned leaf only for that one call (module._parameters / __dict__, restored in finally),
so the re-run yields this call's OWN dweight -- the live one is summed over 8 microbatches into FSDP's
flat grad and is unobservable. The clones are taken at FORWARD time: by the time the backward hook fires
FSDP may have resharded and freed the unsharded weight. Autocast state of the live forward is recorded
and re-entered (the hook runs on the autograd thread, where autocast is off).
Checks recorded per frozen call: isolated out == live out (bitwise), and a second isolated re-run ==
the first (forward AND both gradients bitwise -- determinism of the golden).

Env:
    GEAK_HFRMSN_CAPTURE_DIR=/work/tasks/<name>_cap    (required; enables the capture)
    GEAK_HFRMSN_CAPTURE_STEP=12                       (1-based step to capture, after warmup)
    GEAK_HFRMSN_CAPTURE_CALLS=0,28,29,55,56           (per-step forward call indices to freeze)
    GEAK_HFRMSN_CAPTURE_RANKS=0,4                     (one file per rank and call; no shared writer)
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFRMSN_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFRMSN_CAPTURE_STEP", "0") or 0)
_CALLS = {int(c) for c in os.environ.get("GEAK_HFRMSN_CAPTURE_CALLS", "0,28,29,55,56").split(",")
          if c.strip().isdigit()}
_RANKS = {int(r) for r in os.environ.get("GEAK_HFRMSN_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
TARGET = "transformers.models.qwen2.modeling_qwen2:Qwen2RMSNorm.forward"

_state = {"step": 0, "calls": 0, "names": {}, "per_call": [], "pending": 0, "written": [],
          "meta_done": False, "orig": None}


def note_step(step):
    """Called at every step end (probe's step number). Writes the per-call table after the capture
    step; the frozen records themselves are written from the backward hooks."""
    _state["step"] = step
    if step >= _STEP and not _state["meta_done"] and _state["per_call"]:
        _state["meta_done"] = True
        _write_capture_meta()


def _snap(t):
    return {"__t__": True, "data": t.detach().to("cpu", copy=True), "dtype": str(t.dtype),
            "shape": list(t.shape), "requires_grad": bool(t.requires_grad)}


def _swap_weight(mod, w):
    if "weight" in mod._parameters:
        old = mod._parameters["weight"]
        mod._parameters["weight"] = w
        return ("p", old)
    old = mod.__dict__.get("weight")
    mod.__dict__["weight"] = w
    return ("d", old)


def _restore_weight(mod, tok):
    kind, old = tok
    if kind == "p":
        mod._parameters["weight"] = old
    else:
        mod.__dict__["weight"] = old


def _isolated(orig, mod, x0, w0, go, ac):
    xc = x0.detach().clone().requires_grad_(True)
    wc = w0.detach().clone().requires_grad_(True)
    tok = _swap_weight(mod, wc)
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=ac["dtype"], enabled=ac["enabled"]):
            out = orig(mod, xc)
        dx, dw = torch.autograd.grad(out, (xc, wc), grad_outputs=go)
    finally:
        _restore_weight(mod, tok)
    return out.detach(), dx.detach(), dw.detach()


def install(model=None):
    if not ENABLED:
        return False
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfrmsn] cannot import Qwen2RMSNorm: {e!r}", flush=True)
        return False
    if model is not None:
        _state["names"] = {id(m): n for n, m in model.named_modules() if isinstance(m, mq.Qwen2RMSNorm)}
    orig = mq.Qwen2RMSNorm.forward
    _state["orig"] = orig
    os.makedirs(_DIR, exist_ok=True)

    def wrapped(self, hidden_states):
        if _state["step"] + 1 != _STEP:
            return orig(self, hidden_states)
        k = _state["calls"]
        _state["calls"] += 1
        w = self.weight
        ac = {"enabled": torch.is_autocast_enabled("cuda"),
              "dtype": torch.get_autocast_dtype("cuda")}
        row = {"call": k, "module": _state["names"].get(id(self), "?"),
               "M": int(hidden_states.shape[-2]), "x_shape": list(hidden_states.shape),
               "x_dtype": str(hidden_states.dtype), "w_dtype": str(w.dtype),
               "w_type": type(w).__name__, "w_requires_grad": bool(w.requires_grad),
               "eps": float(self.variance_epsilon), "autocast": {"enabled": ac["enabled"],
                                                                  "dtype": str(ac["dtype"])},
               "grad_enabled": torch.is_grad_enabled()}
        _state["per_call"].append(row)
        if k not in _CALLS:
            return orig(self, hidden_states)
        x0 = hidden_states.detach().clone()
        w0 = w.detach().clone()
        out = orig(self, hidden_states)
        live_out = out.detach().clone()
        rec = {"target": TARGET, "step": _STEP, "rank": _RANK, "call": k, "module": row["module"],
               "args": [_snap(hidden_states), _snap(w)], "kwargs": {},
               "const": {"eps": float(self.variance_epsilon)}, "autocast": row["autocast"],
               "output": _snap(live_out)}
        _state["pending"] += 1

        def _hook(g):
            try:
                t0 = time.time()
                go = g.detach().clone()
                rec["grad_output"] = _snap(go)
                o1, dx1, dw1 = _isolated(orig, self, x0, w0, go, ac)
                o2, dx2, dw2 = _isolated(orig, self, x0, w0, go, ac)
                rec["isolated"] = {"out": _snap(o1), "grads": {"0": _snap(dx1), "1": _snap(dw1)},
                                   "grad_shapes": [list(dx1.shape), list(dw1.shape)]}
                rec["composition_evidence"] = {
                    "iso_out_equals_live_out": bool(torch.equal(o1, live_out)),
                    "iso_out_vs_live_max_abs": float((o1.float() - live_out.float()).abs().max()),
                    "iso_rerun_bitwise": [bool(torch.equal(o1, o2)), bool(torch.equal(dx1, dx2)),
                                          bool(torch.equal(dw1, dw2))],
                    "dx_dtype": str(dx1.dtype), "dw_dtype": str(dw1.dtype),
                    "go_dtype": str(go.dtype), "go_abs_mean": float(go.float().abs().mean()),
                    "go_abs_max": float(go.float().abs().max()),
                    "x_abs_max": float(x0.float().abs().max()),
                    "x_rms_min": float(x0.float().pow(2).mean(-1).sqrt().min()),
                    "x_rms_max": float(x0.float().pow(2).mean(-1).sqrt().max()),
                    "w_abs_max": float(w0.float().abs().max()),
                    "w_min": float(w0.float().min()), "w_max": float(w0.float().max()),
                }
                path = os.path.join(_DIR, f"reference_io_rank{_RANK}_call{k}.pt")
                torch.save(rec, path + ".partial")
                os.replace(path + ".partial", path)
                _state["written"].append(path)
                print(f"[geak-hfrmsn] wrote {path} module={row['module']} M={row['M']} "
                      f"evidence={json.dumps(rec['composition_evidence'])} ({time.time()-t0:.1f}s)",
                      flush=True)
            except Exception as e:  # noqa: BLE001  -- a capture must never kill the run
                print(f"[geak-hfrmsn] capture of call {k} FAILED (run continues): {e!r}", flush=True)
            return g

        if out.requires_grad:
            out.register_hook(_hook)
        else:
            print(f"[geak-hfrmsn] call {k}: output does not require grad, not frozen", flush=True)
        return out

    mq.Qwen2RMSNorm.forward = wrapped
    print(f"[geak-hfrmsn] armed on {TARGET} for step {_STEP}, calls {sorted(_CALLS)}, rank {_RANK}, "
          f"{len(_state['names'])} named norm modules", flush=True)
    return True


def _write_capture_meta():
    per = _state["per_call"]
    Ms = sorted({r["M"] for r in per})
    meta = {"target": TARGET, "step": _STEP, "rank": _RANK, "calls_in_step": len(per),
            "frozen_calls": sorted(_CALLS), "distinct_M": Ms,
            "eps_seen": sorted({r["eps"] for r in per}),
            "x_dtypes": sorted({r["x_dtype"] for r in per}),
            "w_dtypes": sorted({r["w_dtype"] for r in per}),
            "w_types": sorted({r["w_type"] for r in per}),
            "autocast": sorted({json.dumps(r["autocast"]) for r in per}),
            "per_call": per}
    try:
        with open(os.path.join(_DIR, f"capture_meta_rank{_RANK}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[geak-hfrmsn] step {_STEP}: {len(per)} norm calls, M={Ms}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfrmsn] meta write failed: {e!r}", flush=True)
