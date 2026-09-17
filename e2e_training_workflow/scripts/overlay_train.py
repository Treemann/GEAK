"""Reversible rebind of a training op + PROOF that both halves engaged.

WHY ENGAGEMENT NEEDS PROVING, SEPARATELY, FOR EACH HALF. An overlay that binds the forward and
silently leaves the backward on the original path still produces a plausible throughput number and a
plausible loss curve. It is the exact failure L1 exists to catch -- and it is far cheaper to catch
here, before spending a gate. "The number moved" is not proof; a counter that incremented is.

WHY NOT EDIT THE TREE. The rebind travels as an install-time monkeypatch driven from the same
PYTHONPATH entry point as the probe, so nothing under site-packages or the Megatron checkout is ever
modified and removing the env var removes the change. On a single node `torchrun` inherits the
parent environment, so one overlay reaches every rank -- an assumption that does NOT survive a move
to multi-node, where the overlay must be materialised per node.

Manifest (GEAK_OVERLAY=<path/to/manifest.json>):
    {"target": "transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear",
     "replacement": "/work/overlays/my_lnlinear.py:MyLayerNormLinear",   # omit for identity rebind
     "engagement_out": "/work/runs/<run>/engagement.json"}

An omitted `replacement` installs an IDENTITY rebind: same numerics, same kernels, counters only.
That is the honest way to test the mechanism -- it separates "did the rebind engage?" from "is the
new kernel correct?", so a failure has one meaning instead of two.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os

# torch is needed to INSTALL the rebind, not to READ the proof back. Keeping the import soft lets
# the orchestrator call verdict() from outside the training container.
try:
    import torch
except ImportError:  # pragma: no cover - reader-side use
    torch = None

_MANIFEST = os.environ.get("GEAK_OVERLAY", "")
ENABLED = bool(_MANIFEST)

COUNTS = {"forward": 0, "backward": 0}
_cfg = {}


def _resolve(spec):
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    parts = attr.split(".")
    owner = mod
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1], getattr(owner, parts[-1])


def _load_replacement(spec):
    path, _, attr = spec.partition(":")
    sp = importlib.util.spec_from_file_location("_geak_overlay_impl", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return getattr(mod, attr)


def _flush():
    out = _cfg.get("engagement_out")
    if not out:
        return
    try:
        rank = int(os.environ.get("RANK", "0"))
        tmp = f"{out}.rank{rank}.partial"
        with open(tmp, "w") as f:
            json.dump({"rank": rank, "target": _cfg.get("target"),
                       "replacement": _cfg.get("replacement"), **COUNTS}, f)
        os.replace(tmp, f"{out}.rank{rank}")
    except Exception as e:
        print(f"[geak-overlay] could not write engagement proof: {e!r}", flush=True)


if torch is not None:

    class _Counter(torch.autograd.Function):
        """Counts the backward separately from the forward. A single counter cannot distinguish
        'both halves ran' from 'the forward ran twice' -- which is exactly the distinction that
        matters when an overlay binds one half and leaves the other on the original path."""

        @staticmethod
        def forward(ctx, x):
            COUNTS["forward"] += 1
            return x

        @staticmethod
        def backward(ctx, g):
            COUNTS["backward"] += 1
            # Flush early (so a crashed run still carries proof) and then periodically, so the
            # recorded counts describe the whole run rather than a snapshot at an arbitrary point.
            if COUNTS["backward"] < 8 or COUNTS["backward"] % 32 == 0:
                _flush()
            return g


def install():
    if not ENABLED:
        return False
    try:
        with open(_MANIFEST) as f:
            _cfg.update(json.load(f))
        owner, name, orig = _resolve(_cfg["target"])
        # Test the RESOLVED TARGET, not its owner. Getting this backwards replaces the class itself
        # with a plain function, and the next `_LayerNormLinear.apply(...)` inside the stack dies with
        # "'function' object has no attribute 'apply'" -- which is what happened the first time.
        if isinstance(orig, type) and issubclass(orig, torch.autograd.Function):
            owner, name, orig = orig, "apply", orig.apply
        impl = _load_replacement(_cfg["replacement"]) if _cfg.get("replacement") else None
        call = (impl.apply if isinstance(impl, type) and issubclass(impl, torch.autograd.Function)
                else impl) if impl is not None else orig

        def wrapped(*a, **kw):
            out = call(*a, **kw)
            tgt = out[0] if isinstance(out, (list, tuple)) and torch.is_tensor(out[0]) else out
            if torch.is_tensor(tgt) and tgt.requires_grad:
                tagged = _Counter.apply(tgt)
                if isinstance(out, tuple):
                    return (tagged,) + tuple(out[1:])
                if isinstance(out, list):
                    return [tagged] + list(out[1:])
                return tagged
            COUNTS["forward"] += 1      # inference-mode call: no backward will ever come
            return out

        if name == "apply" and isinstance(owner, type):
            setattr(owner, name, staticmethod(wrapped))
        else:
            setattr(owner, name, wrapped)
        mode = "identity rebind (counters only)" if impl is None else f"-> {_cfg['replacement']}"
        print(f"[geak-overlay] {_cfg['target']} {mode}", flush=True)
        return True
    except Exception as e:
        # An overlay that fails to install must SAY SO. Falling through silently would leave the run
        # measuring the original implementation while the report claims a candidate was tested.
        print(f"[geak-overlay] INSTALL FAILED ({e!r}) — the run is measuring the ORIGINAL "
              f"implementation; do not report it as a candidate leg.", flush=True)
        return False


def verdict(engagement_dir_or_prefix):
    """Read the proofs back: both halves must have engaged, on every rank that wrote one."""
    import glob
    files = sorted(glob.glob(engagement_dir_or_prefix + ".rank*"))
    if not files:
        return {"forward": False, "backward": False, "evidence": "no engagement proof written",
                "ranks": 0}
    fwd = bwd = True
    per_rank = {}
    for p in files:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        per_rank[d.get("rank")] = {"forward": d.get("forward", 0), "backward": d.get("backward", 0)}
        fwd = fwd and d.get("forward", 0) > 0
        bwd = bwd and d.get("backward", 0) > 0
    return {"forward": fwd, "backward": bwd, "ranks": len(per_rank), "per_rank": per_rank,
            "evidence": f"counters from {len(per_rank)} rank(s)"}


if __name__ == "__main__":
    import sys
    print(json.dumps(verdict(sys.argv[1]), indent=2))
