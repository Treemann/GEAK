"""Capture the HF `Qwen2MLP` gate/up projections + SwiGLU (fwd + bwd) AS THEY RUN on the hf_trl stack.
Head `hf_mlp_gateup_swiglu`.

WHAT THE OP IS. transformers 4.55 `Qwen2MLP.forward` is one line:

    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

The op frozen here is everything BEFORE down_proj, i.e. the tensor down_proj consumes:

    h = silu(x @ Wg^T) * (x @ Wu^T)        x [1,M,3584] bf16, Wg/Wu [18944,3584] bf16, h [1,M,18944]

forward: two GEMMs (gate, up) + silu + mul; backward: mul/silu backward, two dgrad GEMMs + add (dx),
two wgrad GEMMs (dWg, dWu). down_proj is NOT in the op (a separate head; folding it in would make a
gate/up win unattributable). 28 calls per sample (one per decoder layer), 224 per rank per step.

WHY THE HOOK IS ON `Qwen2MLP.forward` (the CLASS attribute) -- the module's __call__ looks `forward`
up on the class at call time, so a class-level rebind reaches all 28 instances; it is also the
integration bind point. The frozen call runs the ORIGINAL forward unchanged: `h` is taken from a
forward PRE-hook on `self.down_proj` (registered for that one call only), so nothing about the live
computation is re-implemented here.

grad_output IS SOUND HERE. The tensor hook is on h, whose SOLE consumer is down_proj; the hook value
is this op's exact grad_output. The consumer-sum hazard (gate_convergence.md) is on x, which also
feeds the residual add -- dgrad is therefore never read from x's hook.

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original expression (the module's own gate_proj /
up_proj / act_fn submodules, i.e. the HF code path) on clones of the frozen x, Wg and Wu, in the live
process, inside the grad_output hook. Both weights are swapped for cloned leaves for that one call so
the re-run returns this call's OWN dWg / dWu (the live ones are summed over 8 microbatches into FSDP's
flat grad and are unobservable). Clones are taken at FORWARD time. Autocast state is recorded and
re-entered (the hook runs on the autograd thread, where autocast is off).
Recorded per frozen call: isolated h == live h (bitwise), second isolated re-run == first (h and all
three gradients bitwise), and the storage facts a concatenated-GEMM candidate depends on (are Wg and
Wu adjacent views of one flat buffer?).

Env:
    GEAK_HFMLP_CAPTURE_DIR=/work/tasks/<name>_cap   (required; enables the capture)
    GEAK_HFMLP_CAPTURE_STEP=12                      (1-based step to capture, after warmup)
    GEAK_HFMLP_CAPTURE_CALLS=0,14,27                (per-step forward call indices to freeze)
    GEAK_HFMLP_CAPTURE_RANKS=0,4                    (one file per rank and call; no shared writer)
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFMLP_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFMLP_CAPTURE_STEP", "0") or 0)
_CALLS = {int(c) for c in os.environ.get("GEAK_HFMLP_CAPTURE_CALLS", "0,14,27").split(",")
          if c.strip().isdigit()}
_RANKS = {int(r) for r in os.environ.get("GEAK_HFMLP_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
TARGET = "transformers.models.qwen2.modeling_qwen2:Qwen2MLP.forward"

_state = {"step": 0, "calls": 0, "names": {}, "per_call": [], "written": [], "meta_done": False,
          "orig": None}


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
    """Point mod.weight at w for one call. Under FSDP use_orig_params the live weight is a PLAIN
    TENSOR view set in the instance __dict__ (shadowing any `_parameters` entry), so the __dict__ is
    checked FIRST -- the first live capture swapped `_parameters` only, the lookup still found the
    __dict__ view, and autograd.grad reported the cloned leaf unused. Verified by identity after."""
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


def _isolated(mlp, x0, wg0, wu0, go, ac):
    xc = x0.detach().clone().requires_grad_(True)
    wgc = wg0.detach().clone().requires_grad_(True)
    wuc = wu0.detach().clone().requires_grad_(True)
    tg = _swap_weight(mlp.gate_proj, wgc)
    tu = _swap_weight(mlp.up_proj, wuc)
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=ac["dtype"], enabled=ac["enabled"]):
            # The ORIGINAL expression of Qwen2MLP.forward, minus down_proj, on the module's own
            # submodules (same call order: gate, act, up, mul).
            h = mlp.act_fn(mlp.gate_proj(xc)) * mlp.up_proj(xc)
        dx, dwg, dwu = torch.autograd.grad(h, (xc, wgc, wuc), grad_outputs=go)
    finally:
        _restore_weight(mlp.up_proj, tu)
        _restore_weight(mlp.gate_proj, tg)
    return h.detach(), dx.detach(), dwg.detach(), dwu.detach()


def _storage_facts(wg, wu):
    """What a concatenated [37888,3584] gate|up GEMM can assume about the live weights."""
    f = {"wg_type": type(wg).__name__, "wu_type": type(wu).__name__,
         "wg_is_view": wg._base is not None, "wu_is_view": wu._base is not None,
         "wg_contig": wg.is_contiguous(), "wu_contig": wu.is_contiguous(),
         "wg_stride": list(wg.stride()), "same_storage": False, "up_follows_gate": False}
    try:
        f["same_storage"] = (wg.untyped_storage().data_ptr() == wu.untyped_storage().data_ptr())
        f["up_follows_gate"] = (wu.data_ptr() == wg.data_ptr() + wg.numel() * wg.element_size())
        f["base_shape"] = list(wg._base.shape) if wg._base is not None else None
    except Exception as e:  # noqa: BLE001
        f["err"] = repr(e)[:200]
    return f


def install(model=None):
    if not ENABLED:
        return False
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfmlp] cannot import Qwen2MLP: {e!r}", flush=True)
        return False
    if model is not None:
        _state["names"] = {id(m): n for n, m in model.named_modules() if isinstance(m, mq.Qwen2MLP)}
    orig = mq.Qwen2MLP.forward
    _state["orig"] = orig
    os.makedirs(_DIR, exist_ok=True)

    def wrapped(self, x):
        if _state["step"] + 1 != _STEP:
            return orig(self, x)
        k = _state["calls"]
        _state["calls"] += 1
        wg, wu = self.gate_proj.weight, self.up_proj.weight
        ac = {"enabled": torch.is_autocast_enabled("cuda"),
              "dtype": torch.get_autocast_dtype("cuda")}
        row = {"call": k, "module": _state["names"].get(id(self), "?"),
               "M": int(x.shape[-2]), "x_shape": list(x.shape), "x_dtype": str(x.dtype),
               "x_contig": x.is_contiguous(), "wg_dtype": str(wg.dtype), "wu_dtype": str(wu.dtype),
               "wg_shape": list(wg.shape), "wg_requires_grad": bool(wg.requires_grad),
               "act_fn": type(self.act_fn).__name__,
               "biases": [self.gate_proj.bias is not None, self.up_proj.bias is not None],
               "autocast": {"enabled": ac["enabled"], "dtype": str(ac["dtype"])},
               "grad_enabled": torch.is_grad_enabled(), "storage": _storage_facts(wg, wu)}
        _state["per_call"].append(row)
        if k not in _CALLS:
            return orig(self, x)
        x0 = x.detach().clone()
        wg0 = wg.detach().clone()
        wu0 = wu.detach().clone()
        holder = {}

        def _pre(mod, inp):
            holder["h"] = inp[0]
            return None

        hd = self.down_proj.register_forward_pre_hook(_pre)
        try:
            out = orig(self, x)
        finally:
            hd.remove()
        h = holder.get("h")
        if h is None or not h.requires_grad:
            print(f"[geak-hfmlp] call {k}: no differentiable h seen at down_proj, not frozen", flush=True)
            return out
        live_h = h.detach().clone()
        rec = {"target": TARGET, "step": _STEP, "rank": _RANK, "call": k, "module": row["module"],
               "args": [_snap(x), _snap(wg), _snap(wu)], "kwargs": {}, "const": {},
               "autocast": row["autocast"], "act_fn": row["act_fn"], "storage": row["storage"],
               "output": _snap(live_h)}

        def _hook(g):
            try:
                t0 = time.time()
                go = g.detach().clone()
                rec["grad_output"] = _snap(go)
                h1, dx1, dg1, du1 = _isolated(self, x0, wg0, wu0, go, ac)
                h2, dx2, dg2, du2 = _isolated(self, x0, wg0, wu0, go, ac)
                rec["isolated"] = {"out": _snap(h1),
                                   "grads": {"0": _snap(dx1), "1": _snap(dg1), "2": _snap(du1)},
                                   "grad_shapes": [list(dx1.shape), list(dg1.shape), list(du1.shape)]}
                gz = go.reshape(-1, go.shape[-1]).abs().sum(-1) == 0
                rec["composition_evidence"] = {
                    "iso_out_equals_live_out": bool(torch.equal(h1, live_h)),
                    "iso_out_vs_live_max_abs": float((h1.float() - live_h.float()).abs().max()),
                    "iso_rerun_bitwise": [bool(torch.equal(h1, h2)), bool(torch.equal(dx1, dx2)),
                                          bool(torch.equal(dg1, dg2)), bool(torch.equal(du1, du2))],
                    "dtypes": [str(h1.dtype), str(dx1.dtype), str(dg1.dtype), str(du1.dtype)],
                    "go_dtype": str(go.dtype), "go_abs_mean": float(go.float().abs().mean()),
                    "go_abs_max": float(go.float().abs().max()), "go_zero_rows": int(gz.sum()),
                    "x_abs_max": float(x0.float().abs().max()),
                    "h_abs_max": float(live_h.float().abs().max()),
                    "wg_abs_max": float(wg0.float().abs().max()),
                    "wu_abs_max": float(wu0.float().abs().max()),
                    "dwg_abs_mean": float(dg1.float().abs().mean()),
                    "dwu_abs_mean": float(du1.float().abs().mean()),
                }
                path = os.path.join(_DIR, f"reference_io_rank{_RANK}_call{k}.pt")
                torch.save(rec, path + ".partial")
                os.replace(path + ".partial", path)
                _state["written"].append(path)
                print(f"[geak-hfmlp] wrote {path} module={row['module']} M={row['M']} "
                      f"evidence={json.dumps(rec['composition_evidence'])} ({time.time()-t0:.1f}s)",
                      flush=True)
            except Exception as e:  # noqa: BLE001  -- a capture must never kill the run
                print(f"[geak-hfmlp] capture of call {k} FAILED (run continues): {e!r}", flush=True)
            return g

        h.register_hook(_hook)
        return out

    mq.Qwen2MLP.forward = wrapped
    print(f"[geak-hfmlp] armed on {TARGET} for step {_STEP}, calls {sorted(_CALLS)}, rank {_RANK}, "
          f"{len(_state['names'])} named MLP modules", flush=True)
    return True


def _write_capture_meta():
    per = _state["per_call"]
    Ms = sorted({r["M"] for r in per})
    meta = {"target": TARGET, "step": _STEP, "rank": _RANK, "calls_in_step": len(per),
            "frozen_calls": sorted(_CALLS), "distinct_M": Ms,
            "x_dtypes": sorted({r["x_dtype"] for r in per}),
            "w_dtypes": sorted({r["wg_dtype"] + "/" + r["wu_dtype"] for r in per}),
            "act_fns": sorted({r["act_fn"] for r in per}),
            "autocast": sorted({json.dumps(r["autocast"]) for r in per}),
            "storage_kinds": sorted({json.dumps({k: v for k, v in r["storage"].items()
                                                 if k not in ("base_shape",)}) for r in per}),
            "per_call": per}
    try:
        with open(os.path.join(_DIR, f"capture_meta_rank{_RANK}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[geak-hfmlp] step {_STEP}: {len(per)} MLP calls, M={Ms}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfmlp] meta write failed: {e!r}", flush=True)
