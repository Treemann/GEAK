"""Capture a training op's fwd+bwd I/O into an IMMUTABLE task dir (the L0 oracle).

WHAT MAKES A TRAINING TASK DIFFERENT. The kernel layer's inference tasks are forward-only. Here the
objective is fwd+bwd COMBINED, because the measured top head on this workload is itself a backward
kernel (`FusedAttnFuncBackward`, 21.5% of the critical path). A forward-only task would not even see
the biggest candidate, and would happily report a win for a kernel that is 2x faster forward and 3x
slower backward.

HOW THE TOLERANCE IS SET, and why not by hand. "Within rtol=1e-2" is arbitrary: too tight and every
honest bf16 rewrite fails, too loose and a real bug passes. Instead the oracle records THREE things
on frozen inputs:

    ref_fp32     the original implementation run in fp32   -- the best available truth
    base_lowp    the original implementation as it runs    -- what the baseline actually produces
    err_base     |base_lowp - ref_fp32|                    -- the error the baseline ALREADY carries

and the acceptance rule becomes: **the candidate must be no further from ref_fp32 than the original
is** (within a small margin). That is not arbitrary, it cannot be argued with, and it does not
penalise a candidate for rounding the baseline also does.

Plus the sign rule: the error distribution must not be one-sided. A small systematic bias passes
every magnitude check and still destroys convergence over thousands of steps -- which is exactly the
failure L0 exists to catch and L1 only catches later and more expensively.

Usage (inside the training process, installed by rocm_compat.py):
    GEAK_CAPTURE_TARGET="transformer_engine.pytorch.attention.dot_product_attention:DotProductAttention"
    GEAK_CAPTURE_STEP=12          # capture on this step (after warmup)
    GEAK_CAPTURE_DIR=/work/tasks/attn_bwd
"""
from __future__ import annotations

import importlib
import json
import os

import torch

_TARGET = os.environ.get("GEAK_CAPTURE_TARGET", "")
_STEP = int(os.environ.get("GEAK_CAPTURE_STEP", "0") or 0)
_DIR = os.environ.get("GEAK_CAPTURE_DIR", "")
# Capture on ONE rank by default. Every rank writing the same path corrupts it (observed: a 689-byte
# truncated reference_io.pt from two concurrent writers), and the extra copies carry no information
# at TP1 -- the ranks see the same op with the same shapes.
_RANKS = {int(r) for r in os.environ.get("GEAK_CAPTURE_RANKS", "0").split(",") if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_TARGET and _DIR and _STEP > 0 and _RANK in _RANKS)

_state = {"step": 0, "captured": False}


def note_step(step):
    _state["step"] = step


# Primitives that survive a round trip. Anything else is a RUNTIME HANDLE.
_PRIM = (int, float, bool, str, bytes, type(None))


def _snap(x):
    if torch.is_tensor(x):
        return {"__t__": True, "data": x.detach().to("cpu", copy=True),
                "dtype": str(x.dtype), "shape": list(x.shape), "requires_grad": bool(x.requires_grad)}
    if isinstance(x, _PRIM):
        return x
    if isinstance(x, (list, tuple)):
        return [_snap(v) for v in x]
    if isinstance(x, dict):
        return {k: _snap(v) for k, v in x.items()}
    # A TRAINING-SPECIFIC CONSTRAINT, not an edge case. Training ops take runtime handles -- process
    # groups, FP8 recipes, module state, autocast contexts -- and they are neither picklable nor
    # replayable. (Found the hard way: TE's _LayerNormLinear.apply carries a ProcessGroup, and
    # torch.save died inside a backward hook, killing the training run.) They are recorded as OPAQUE
    # so the task dir states what must be RECONSTRUCTED rather than pretending the call can be
    # replayed verbatim. A standalone training-kernel task is therefore never a pure replay.
    return {"__opaque__": type(x).__name__,
            "module": getattr(type(x), "__module__", "?"),
            "repr": repr(x)[:200]}


def _resolve(spec):
    """Resolve "module.path:attr" to (owner, name, original).

    Training ops are overwhelmingly `torch.autograd.Function` subclasses, and those are INVOKED
    through `.apply`, not by calling the class. Naming the class and silently wrapping something that
    is never called is the failure this branch exists to avoid (it cost one smoke run: `torch:matmul`
    armed on both ranks and never fired, because TE never routes through it).
    """
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    parts = attr.split(".")
    owner = mod
    for p in parts[:-1]:
        owner = getattr(owner, p)
    name = parts[-1]
    target = getattr(owner, name)
    if isinstance(target, type) and issubclass(target, torch.autograd.Function):
        return target, "apply", target.apply
    return owner, name, target


def install():
    """Wrap the target callable; capture once, then get out of the way."""
    if not ENABLED:
        return False
    try:
        owner, name, orig = _resolve(_TARGET)
    except Exception as e:
        print(f"[geak-capture] cannot resolve {_TARGET}: {e!r}", flush=True)
        return False

    os.makedirs(_DIR, exist_ok=True)

    def wrapped(*a, **kw):
        if _state["captured"] or _state["step"] + 1 != _STEP:
            return orig(*a, **kw)
        _state["captured"] = True
        rec = {"target": _TARGET, "step": _STEP, "args": _snap(a), "kwargs": _snap(kw)}
        out = orig(*a, **kw)
        # grad_output is captured from the LIVE run: a synthetic one of the right shape would have
        # the wrong scale, and backward kernels are scale-sensitive.
        tgt = out[0] if isinstance(out, (list, tuple)) and torch.is_tensor(out[0]) else out
        if torch.is_tensor(tgt) and tgt.requires_grad:
            def _hook(g):
                # A capture failure must NEVER kill the training run -- same doctrine as the probe.
                # This is not hypothetical: an unguarded torch.save raised inside this hook and took
                # down an 8-rank job mid-backward.
                try:
                    rec["grad_output"] = _snap(g)
                    path = os.path.join(_DIR, f"reference_io_rank{_RANK}.pt")
                    tmp = path + ".partial"
                    torch.save(rec, tmp)
                    os.replace(tmp, path)          # atomic: a reader never sees a half-written file
                    _write_meta(rec)
                    print(f"[geak-capture] wrote {path} "
                          f"(fwd inputs + grad_output for {_TARGET})", flush=True)
                except Exception as e:
                    print(f"[geak-capture] capture FAILED (run continues): {e!r}", flush=True)
                return g
            tgt.register_hook(_hook)
        rec["output"] = _snap(out)
        return out

    if name == "apply" and isinstance(owner, type):
        # staticmethod on the class: bind it explicitly or python re-wraps it as a bound method.
        setattr(owner, name, staticmethod(wrapped))
    else:
        setattr(owner, name, wrapped)
    print(f"[geak-capture] armed on {_TARGET} (via {owner.__name__ if isinstance(owner, type) else owner.__name__}.{name}) "
          f"for step {_STEP}", flush=True)
    return True


def _describe(node):
    if isinstance(node, dict) and node.get("__t__"):
        return {"shape": node["shape"], "dtype": node["dtype"], "requires_grad": node["requires_grad"]}
    if isinstance(node, list):
        return [_describe(v) for v in node]
    if isinstance(node, dict):
        return {k: _describe(v) for k, v in node.items()}
    return node


def _collect_opaque(node, out=None):
    out = [] if out is None else out
    if isinstance(node, dict) and node.get("__opaque__"):
        out.append({"type": node["__opaque__"], "module": node.get("module")})
    elif isinstance(node, list):
        for v in node:
            _collect_opaque(v, out)
    elif isinstance(node, dict):
        for v in node.values():
            _collect_opaque(v, out)
    return out


def _write_meta(rec):
    opaque = _collect_opaque(rec["args"]) + _collect_opaque(rec["kwargs"])
    meta = {
        "target": rec["target"], "step": rec["step"], "rank": _RANK,
        "opaque_args": opaque,
        "opaque_note": ("these arguments are runtime handles (process groups, recipes, module state). "
                        "They are NOT captured and NOT replayable: the standalone task must "
                        "RECONSTRUCT them. A training-kernel task is never a pure replay of the call."),
        "args": _describe(rec["args"]), "kwargs": _describe(rec["kwargs"]),
        "grad_output": _describe(rec.get("grad_output")),
        "objective": "fwd+bwd combined latency",
        "oracle_rule": ("the candidate must be no further from the fp32 reference than the ORIGINAL "
                        "implementation is, and its error distribution must not be one-sided"),
        "immutable": ("reference_io.pt is frozen input. Nothing downstream may regenerate it: the "
                      "kernel layer must be scored against inputs it cannot influence."),
    }
    with open(os.path.join(_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Oracle construction (run OFFLINE, after capture, on the frozen record)
# ---------------------------------------------------------------------------
def build_oracle(task_dir, fn):
    """Compute ref_fp32 / base_lowp / err_base for out, dgrad and wgrad on the frozen inputs.

    `fn` MUST be a pure-PyTorch reference implementation, not the original fused op. That is forced
    by the capture itself: the real op's arguments include runtime handles (see `opaque_args` in
    meta.json -- on TE's _LayerNormLinear they are a ProcessGroup, a WeightGradStore, and the owning
    module), which cannot be serialized and cannot be rebuilt offline. So the oracle is
    "frozen I/O + an independent reference implementation", exactly as the inference kernel tasks
    are -- not a replay of the captured call.

    For the captured example that reference is two lines (`F.layer_norm` then `F.linear` in fp32);
    the extractor role is responsible for writing it and for stating what it assumes.
    """
    rec = torch.load(os.path.join(task_dir, "reference_io.pt"), weights_only=False)

    def _load(node, dtype=None, dev="cuda"):
        if isinstance(node, dict) and node.get("__t__"):
            t = node["data"].to(dev)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            return t.requires_grad_(node["requires_grad"])
        if isinstance(node, list):
            return [_load(v, dtype, dev) for v in node]
        if isinstance(node, dict):
            return {k: _load(v, dtype, dev) for k, v in node.items()}
        return node

    results = {}
    for tag, dtype in (("ref_fp32", torch.float32), ("base_lowp", None)):
        a = _load(rec["args"], dtype)
        kw = _load(rec["kwargs"], dtype)
        go = _load(rec.get("grad_output"), dtype)
        out = fn(*a, **kw)
        tgt = out[0] if isinstance(out, (list, tuple)) else out
        grads = torch.autograd.grad(tgt, [t for t in _flat(a) if torch.is_tensor(t) and t.requires_grad],
                                    grad_outputs=go, allow_unused=True, retain_graph=False)
        results[tag] = {"out": tgt.detach().float().cpu(),
                        "grads": [g.detach().float().cpu() if g is not None else None for g in grads]}

    err = {"out": (results["base_lowp"]["out"] - results["ref_fp32"]["out"]).abs().max().item()}
    for i, (b, r) in enumerate(zip(results["base_lowp"]["grads"], results["ref_fp32"]["grads"])):
        if b is not None and r is not None:
            err[f"grad{i}"] = (b - r).abs().max().item()
    torch.save({"results": results, "err_base": err},
               os.path.join(task_dir, "oracle.pt"))
    return err


def _flat(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _flat(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _flat(v)
