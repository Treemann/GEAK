"""Capture the RMSNorm of the MLP fc1 `_LayerNormLinear` (fwd + bwd) AS IT RUNS ON THE ACCEPTED STACK.

WHAT THE OP IS. On the accepted s7 stack, fc1 is driven by the `mlp_fc1_gemm` overlay, whose
`kernel._LNLinear` calls TE's C++ normalization directly:

    forward : ln_out, _, rstd = tex.rmsnorm_fwd(x2, ln_weight, eps, None, None, kBFloat16, 0, False)
    backward: dx, dgamma      = tex.rmsnorm_bwd(dln, x2, rstd, ln_weight, 0, False)

(`tex` = `transformer_engine_torch`, looked up as a MODULE ATTRIBUTE at call time.) The profile's
`rmsnorm_bwd_general` + `rmsnorm_bwd_finalize` kernels under `_FC1Backward` are the second call. The
op frozen here is that RMSNorm, fwd+bwd: inputs (x2 [T, H] bf16, ln_weight [H] bf16, eps), output
ln_out [T, H] bf16, gradients dx [T, H] bf16 and dgamma [H] bf16.

WHY THE CAPTURE HOOKS `transformer_engine_torch.rmsnorm_{fwd,bwd}` AND FILTERS BY CALLER. The same
two functions also serve other call sites with the SAME shapes -- the final RMSNorm before the LM head
([4096, 3584], via TE's operation fuser, `ops/basic/rmsnorm.py`, which binds them with
`from transformer_engine_torch import ...`, so whether it sees a rebind depends on import order).
Shape cannot tell them apart; the calling frame can. Only calls whose immediate caller is the fc1
overlay's `kernel.py` (a path containing `mlp_fc1_gemm`) are counted, timed or captured.

grad_output IS SOUND HERE. `dln` is the dgrad GEMM's output (grad_out @ W) computed inside fc1's own
backward, and ln_out's ONLY consumers are fc1's forward GEMM and its wgrad GEMM (which reads ln_out but
produces no gradient for it). So the tensor handed to rmsnorm_bwd IS this op's exact grad_output --
not a sum over consumers (the hazard named in gate_convergence.md does not apply).

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original TE functions on clones of the frozen inputs,
in the live process. Two checks make the pairing auditable rather than asserted:
  * fwd: re-running rmsnorm_fwd on the frozen (x2, ln_weight, eps) must reproduce the LIVE rstd that
    the backward received bit-for-bit (and the LIVE ln_out, paired by rstd storage);
  * bwd: re-running rmsnorm_bwd must reproduce the LIVE dx and dgamma.

THE SAME HOOK MEASURES THE HEAD'S TWO PREMISES, on every fc1 call of the capture step (rank 0):
  * DEAD ROWS: how many rows of dln (and of x2) are entirely zero -- the "slice to the live rows"
    lever only exists if they are;
  * OVERLAP: the fc1 overlay runs wgrad on a SIDE stream concurrently with dgrad -> rmsnorm_bwd on the
    main stream, then joins. An event recorded on that side stream at the moment rmsnorm_bwd is
    enqueued completes when all wgrad chunks have finished; comparing it with the end of rmsnorm_bwd
    says whether the norm backward is on the critical path at all (if wgrad finishes later, a faster
    norm only lengthens the main stream's wait).

Env:
    GEAK_RMSN_CAPTURE_DIR=/work/tasks/<name>_cap    (required; enables the capture)
    GEAK_RMSN_CAPTURE_STEP=13                       (capture on this step, after warmup)
    GEAK_RMSN_CAPTURE_CALL=14                       (which fc1 bwd call of the step to freeze, 0-based)
    GEAK_RMSN_CAPTURE_RANKS=0                       (one rank: concurrent writers corrupt the file)
"""
from __future__ import annotations

import json
import os
import sys

import torch

_DIR = os.environ.get("GEAK_RMSN_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_RMSN_CAPTURE_STEP", "0") or 0)
_CALL = int(os.environ.get("GEAK_RMSN_CAPTURE_CALL", "14") or 0)
_RANKS = {int(r) for r in os.environ.get("GEAK_RMSN_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
_CALLER_TAG = os.environ.get("GEAK_RMSN_CALLER_TAG", "mlp_fc1_gemm")

_state = {"step": 0, "done": False, "fwd_calls": 0, "bwd_calls": 0, "other_fwd": 0, "other_bwd": 0,
          "fwd_by_rstd": {}, "eps": set(), "per_call": [], "events": [], "t0": None,
          "frozen": None, "side_note": None}


def note_step(step):
    """Called at every step end. The capture is finalised at the end of the capture step, so the
    per-call statistics cover EVERY fc1 call of that step (both microbatches), not just the first."""
    _state["step"] = step
    if (not _state["done"]) and step >= _STEP and _state.get("orig") is not None \
            and _state["bwd_calls"] > 0:
        _state["done"] = True
        if _state["frozen"] is None:
            print(f"[geak-rmsn] NOT captured: call {_CALL} never happened "
                  f"({_state['bwd_calls']} fc1 bwd calls in step {_STEP})", flush=True)
            return
        try:
            _finish(*_state["orig"])
        except Exception as e:  # a capture failure must never kill an 8-rank job
            print(f"[geak-rmsn] capture FAILED (run continues): {e!r}", flush=True)


def _armed():
    return (not _state["done"]) and _state["step"] + 1 == _STEP


def _snap(x):
    if torch.is_tensor(x):
        return {"__t__": True, "data": x.detach().to("cpu", copy=True),
                "dtype": str(x.dtype), "shape": list(x.shape),
                "requires_grad": bool(x.requires_grad)}
    return x


def _from_fc1(depth=2):
    try:
        return _CALLER_TAG in sys._getframe(depth).f_code.co_filename
    except Exception:
        return False


def _fc1_side_stream(dev):
    """The fc1 overlay's wgrad side stream (bwd_overlap._STREAMS[(device, 0)]), if reachable."""
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", "") or ""
        if name.endswith("bwd_overlap") and _CALLER_TAG in f:
            s = getattr(mod, "_STREAMS", {}).get((dev, 0))
            if s is not None:
                return s, f
    mod = sys.modules.get("bwd_overlap")
    if mod is not None:
        s = getattr(mod, "_STREAMS", {}).get((dev, 0))
        if s is not None:
            return s, getattr(mod, "__file__", "?")
    return None, None


def install():
    if not ENABLED:
        return False
    try:
        import transformer_engine.pytorch  # noqa: F401  (registers transformer_engine_torch)
        import transformer_engine_torch as tex
    except Exception as e:
        print(f"[geak-rmsn] cannot import transformer_engine_torch: {e!r}", flush=True)
        return False
    os.makedirs(_DIR, exist_ok=True)
    orig_fwd, orig_bwd = tex.rmsnorm_fwd, tex.rmsnorm_bwd

    def wrapped_fwd(*a, **kw):
        out = orig_fwd(*a, **kw)
        if _armed():
            if _from_fc1():
                _state["fwd_calls"] += 1
                try:
                    _state["eps"].add(float(a[2] if len(a) > 2 else kw.get("eps")))
                    # Pair the live forward output with its backward by rstd storage: the backward
                    # receives the SAME saved rstd tensor.
                    _state["fwd_by_rstd"][out[2].data_ptr()] = (out[0], float(a[2]))
                except Exception as e:
                    print(f"[geak-rmsn] fwd bookkeeping failed: {e!r}", flush=True)
            else:
                _state["other_fwd"] += 1
        return out

    def wrapped_bwd(dz, x, rsigma, gamma, sm_margin, zcg, *rest):
        if not _armed():
            return orig_bwd(dz, x, rsigma, gamma, sm_margin, zcg, *rest)
        if not _from_fc1():
            _state["other_bwd"] += 1
            return orig_bwd(dz, x, rsigma, gamma, sm_margin, zcg, *rest)
        idx = _state["bwd_calls"]
        _state["bwd_calls"] += 1
        dev = dz.device
        main = torch.cuda.current_stream(dev)
        if _state["t0"] is None:
            _state["t0"] = torch.cuda.Event(enable_timing=True)
            _state["t0"].record(main)
        side, side_file = _fc1_side_stream(dev)
        if _state["side_note"] is None:
            _state["side_note"] = side_file or "fc1 side stream NOT found (overlap not measured)"
        ev_side = None
        if side is not None:
            ev_side = torch.cuda.Event(enable_timing=True)
            ev_side.record(side)          # completes when every wgrad chunk enqueued so far is done
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(main)
        out = orig_bwd(dz, x, rsigma, gamma, sm_margin, zcg, *rest)
        e1.record(main)
        _state["events"].append((idx, e0, e1, ev_side))
        try:
            dz2 = dz.reshape(-1, dz.shape[-1])
            x2 = x.reshape(-1, x.shape[-1])
            zr_dz = (dz2 == 0).all(dim=1)
            zr_x = (x2 == 0).all(dim=1)
            nz = (~zr_dz).nonzero()
            _state["per_call"].append({
                "call": idx, "dz_shape": list(dz.shape), "x_shape": list(x.shape),
                "dz_zero_rows": zr_dz.sum(), "x_zero_rows": zr_x.sum(),
                "dz_last_live_row": nz.max() if nz.numel() else None,
                "dz_zero_frac": (dz2 == 0).float().mean(),
                "dz_abs_mean": dz2.float().abs().mean(),
                "contig": [bool(dz.is_contiguous()), bool(x.is_contiguous())],
                "dtypes": [str(dz.dtype), str(x.dtype), str(rsigma.dtype), str(gamma.dtype)],
                "sm_margin": int(sm_margin), "zero_centered_gamma": bool(zcg)})
        except Exception as e:
            print(f"[geak-rmsn] stats failed on call {idx}: {e!r}", flush=True)
        # Pop (not get): holding every layer's ln_out past its backward would pin ~1.6 GB.
        fw = _state["fwd_by_rstd"].pop(rsigma.data_ptr(), None)
        if idx == _CALL:
            # Freeze clones NOW: dz is a temporary that dies with this backward.
            _state["frozen"] = {
                "dz": dz.detach().clone(), "x": x.detach().clone(),
                "rstd": rsigma.detach().clone(), "gamma": gamma.detach().clone(),
                "dx_live": out[0].detach().clone(), "dg_live": out[1].detach().clone(),
                "y_live": fw[0].detach().clone() if fw is not None else None,
                "eps": fw[1] if fw is not None else None,
                "sm_margin": int(sm_margin), "zcg": bool(zcg)}
        return out

    _state["orig"] = (orig_fwd, orig_bwd, tex)
    tex.rmsnorm_fwd = wrapped_fwd
    tex.rmsnorm_bwd = wrapped_bwd
    print(f"[geak-rmsn] armed on transformer_engine_torch.rmsnorm_fwd/bwd (caller *{_CALLER_TAG}*) for "
          f"step {_STEP}, freezing fc1 bwd call {_CALL} (rank {_RANK})", flush=True)
    return True


def _finish(orig_fwd, orig_bwd, tex):
    torch.cuda.synchronize()
    fr = _state["frozen"]
    t0 = _state["t0"]
    timing = []
    for idx, e0, e1, es in _state["events"]:
        rec = {"call": idx, "rmsnorm_bwd_us": e0.elapsed_time(e1) * 1e3,
               "start_ms_from_first": t0.elapsed_time(e0), "end_ms_from_first": t0.elapsed_time(e1)}
        if es is not None:
            rec["wgrad_done_ms_from_first"] = t0.elapsed_time(es)
            # > 0: wgrad (side) finishes AFTER the norm backward -> the norm is hidden under wgrad
            rec["wgrad_after_norm_end_us"] = (t0.elapsed_time(es) - t0.elapsed_time(e1)) * 1e3
        timing.append(rec)
    per_call = []
    for d in _state["per_call"]:
        per_call.append({k: (v.item() if torch.is_tensor(v) else v) for k, v in d.items()})

    # ---- isolated re-run of the ORIGINAL on clones of the frozen inputs ----
    x, g, dz, eps = fr["x"], fr["gamma"], fr["dz"], fr["eps"]
    if eps is None:
        eps_set = sorted(_state["eps"])
        eps = eps_set[0] if len(eps_set) == 1 else None
    y_iso, _mu, rstd_iso = orig_fwd(x.clone(), g.clone(), eps, None, None, tex.DType.kBFloat16, 0,
                                    False)
    dx_iso, dg_iso = orig_bwd(dz.clone(), x.clone(), rstd_iso, g.clone(), 0, False)
    torch.cuda.synchronize()
    ev = {
        "rstd_iso_equals_live": bool(torch.equal(rstd_iso, fr["rstd"])),
        "rstd_iso_vs_live_max_abs": float((rstd_iso - fr["rstd"]).abs().max()),
        "y_iso_equals_live": (bool(torch.equal(y_iso, fr["y_live"]))
                              if fr["y_live"] is not None else "live ln_out not paired"),
        "dx_iso_equals_live": bool(torch.equal(dx_iso, fr["dx_live"])),
        "dgamma_iso_equals_live": bool(torch.equal(dg_iso, fr["dg_live"])),
        "dgamma_iso_vs_live_max_abs": float((dg_iso.float() - fr["dg_live"].float()).abs().max()),
        "eps": eps, "eps_seen_on_fc1_calls": sorted(_state["eps"]),
        "fc1_fwd_calls_in_step": _state["fwd_calls"], "fc1_bwd_calls_in_step": _state["bwd_calls"],
        "other_callsite_fwd_calls_in_step": _state["other_fwd"],
        "other_callsite_bwd_calls_in_step": _state["other_bwd"],
        "sm_margin": fr["sm_margin"], "zero_centered_gamma": fr["zcg"],
        "side_stream": _state["side_note"],
    }
    # Standalone (uncontended) timing of the original on the frozen inputs, for comparison with the
    # in-situ timing above (which runs concurrently with fc1's side-stream wgrad).
    def _t(fn, n=50):
        for _ in range(5):
            fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / n * 1e3
    standalone = {
        "rmsnorm_fwd_us": _t(lambda: orig_fwd(x, g, eps, None, None, tex.DType.kBFloat16, 0, False)),
        "rmsnorm_bwd_us": _t(lambda: orig_bwd(dz, x, rstd_iso, g, 0, False)),
        "note": "TE original on the frozen inputs, alone on the device (rank 0, during the capture "
                "step; other ranks idle-wait at the next collective)."}

    rec = {
        "target": "transformer_engine_torch:rmsnorm_fwd+rmsnorm_bwd [caller: mlp_fc1_gemm/kernel.py]",
        "step": _STEP, "rank": _RANK,
        "args": [_snap(x), _snap(g)],
        "kwargs": {"eps": eps},
        "grad_output": _snap(dz),
        "output": _snap(fr["y_live"]) if fr["y_live"] is not None else None,
        "isolated": {"out": _snap(y_iso), "grads": {"0": _snap(dx_iso), "1": _snap(dg_iso)},
                     "grad_shapes": [list(x.shape), list(g.shape)],
                     "rstd": _snap(rstd_iso)},
        "live": {"dx": _snap(fr["dx_live"]), "dgamma": _snap(fr["dg_live"]),
                 "rstd": _snap(fr["rstd"])},
        "composition_evidence": ev,
        "opaque_args": [],
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}.pt")
    torch.save(rec, path + ".partial")
    os.replace(path + ".partial", path)
    side = {"captured_call": _CALL, "composition_evidence": ev, "standalone_timing": standalone,
            "in_situ_timing": timing, "per_call_rows": per_call,
            "shapes": {"x": list(x.shape), "ln_weight": list(g.shape), "grad_output": list(dz.shape),
                       "out": list(y_iso.shape), "rstd": list(rstd_iso.shape)},
            "dtypes": {"x": str(x.dtype), "ln_weight": str(g.dtype), "grad_output": str(dz.dtype),
                       "out": str(y_iso.dtype), "dx": str(dx_iso.dtype), "dgamma": str(dg_iso.dtype),
                       "rstd": str(rstd_iso.dtype)}}
    with open(os.path.join(_DIR, "capture_meta.json"), "w") as f:
        json.dump(side, f, indent=2, default=str)
    _state["fwd_by_rstd"].clear()
    _state["frozen"] = None
    print(f"[geak-rmsn] wrote {path}; evidence={ev}", flush=True)
