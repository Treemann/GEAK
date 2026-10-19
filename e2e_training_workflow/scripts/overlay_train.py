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
import inspect
import json
import os

# torch is needed to INSTALL the rebind, not to READ the proof back. Keeping the import soft lets
# the orchestrator call verdict() from outside the training container.
try:
    import torch
except ImportError:  # pragma: no cover - reader-side use
    torch = None

# GEAK_OVERLAY takes ONE manifest or a COMMA-SEPARATED LIST, installed left to right.
#
# Why a list. Accepted kernels rebind different symbols -- the LM-head candidate targets
# GPTModel._postprocess, the FC2 one targets transformer_engine...linear:_Linear -- and with a single
# manifest there was no way to run two accepted kernels TOGETHER. The workflow could accept N
# candidates and then deliver none of them as a stack, which makes the combined figure (the only one
# a user actually cares about) unmeasurable. Stacking on the SAME symbol still has to be handled
# inside one overlay's impl.py by delegating the other call site; this list is for distinct symbols.
#
# Each manifest keeps its OWN counters and its OWN engagement_out. That is not tidiness: a shared
# counter cannot distinguish "both overlays engaged" from "one engaged twice", which is the same
# distinction the forward/backward split exists to make.
_MANIFESTS = [m.strip() for m in os.environ.get("GEAK_OVERLAY", "").split(",") if m.strip()]
_MANIFEST = _MANIFESTS[0] if _MANIFESTS else ""      # kept for back-compat with single-overlay reads
ENABLED = bool(_MANIFESTS)

COUNTS = {"forward": 0, "backward": 0}   # aggregate across overlays; per-overlay counts live in _SLOTS
_SLOTS = []                              # [{"cfg": {...}, "counts": {...}}]
_cfg = {}                                # first slot's cfg, for back-compat


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


def _flush_slot(slot):
    cfg, counts = slot["cfg"], slot["counts"]
    out = cfg.get("engagement_out")
    if not out:
        return
    try:
        rank = int(os.environ.get("RANK", "0"))
        tmp = f"{out}.rank{rank}.partial"
        with open(tmp, "w") as f:
            json.dump({"rank": rank, "target": cfg.get("target"),
                       "replacement": cfg.get("replacement"), **counts}, f)
        os.replace(tmp, f"{out}.rank{rank}")
    except Exception as e:
        print(f"[geak-overlay] could not write engagement proof: {e!r}", flush=True)


def _flush():
    for slot in _SLOTS:
        _flush_slot(slot)


if torch is not None:

    class _Counter(torch.autograd.Function):
        """Counts the backward separately from the forward. A single counter cannot distinguish
        'both halves ran' from 'the forward ran twice' -- which is exactly the distinction that
        matters when an overlay binds one half and leaves the other on the original path."""

        @staticmethod
        def forward(ctx, x, slot):
            ctx.slot = slot
            slot["counts"]["forward"] += 1
            COUNTS["forward"] += 1
            return x

        @staticmethod
        def backward(ctx, g):
            c = ctx.slot["counts"]
            c["backward"] += 1
            COUNTS["backward"] += 1
            # Flush early (so a crashed run still carries proof) and then periodically, so the
            # recorded counts describe the whole run rather than a snapshot at an arbitrary point.
            if c["backward"] < 8 or c["backward"] % 32 == 0:
                _flush_slot(ctx.slot)
            return g, None


def _install_one(manifest_path):
    """Install ONE manifest. Returns True on success; prints and returns False on failure."""
    cfg = {}
    try:
        with open(manifest_path) as f:
            cfg.update(json.load(f))
        slot = {"cfg": cfg, "counts": {"forward": 0, "backward": 0}}
        owner, name, orig = _resolve(cfg["target"])
        # Test the RESOLVED TARGET, not its owner. Getting this backwards replaces the class itself
        # with a plain function, and the next `_LayerNormLinear.apply(...)` inside the stack dies with
        # "'function' object has no attribute 'apply'" -- which is what happened the first time.
        if isinstance(orig, type) and issubclass(orig, torch.autograd.Function):
            owner, name, orig = orig, "apply", orig.apply
        impl = _load_replacement(cfg["replacement"]) if cfg.get("replacement") else None
        call = (impl.apply if isinstance(impl, type) and issubclass(impl, torch.autograd.Function)
                else impl) if impl is not None else orig

        def wrapped(*a, **kw):
            out = call(*a, **kw)
            # A model-level rebind (e.g. HF *ForCausalLM.forward) returns a ModelOutput -- a dict, not
            # a tuple -- whose differentiable result is its "loss". Tag that, and CLONE the tagged
            # value: _Counter returns its input as-is, i.e. a view, and HF Trainer.compute_loss then
            # does `loss *= num_processes` IN PLACE, which autograd forbids on a custom-Function view.
            # (A 0-d clone; no other output type takes this branch, so existing overlays are unchanged.)
            if (not torch.is_tensor(out) and not isinstance(out, (list, tuple))
                    and hasattr(out, "keys") and "loss" in out.keys()
                    and torch.is_tensor(out["loss"]) and out["loss"].requires_grad):
                out["loss"] = _Counter.apply(out["loss"], slot).clone()
                return out
            tgt = out[0] if isinstance(out, (list, tuple)) and torch.is_tensor(out[0]) else out
            if torch.is_tensor(tgt) and tgt.requires_grad:
                tagged = _Counter.apply(tgt, slot)
                if isinstance(out, tuple):
                    return (tagged,) + tuple(out[1:])
                if isinstance(out, list):
                    return [tagged] + list(out[1:])
                return tagged
            slot["counts"]["forward"] += 1   # inference-mode call: no backward will ever come
            COUNTS["forward"] += 1
            return out

        # Keep the ORIGINAL's signature visible. HF Trainer introspects the model class's forward
        # (find_labels -> label_names, can_return_loss); behind a bare (*a, **kw) it finds no
        # `labels`, so evaluate() never computes a loss and returns no eval_loss. Hit 2026-10-19 by the
        # first model-level rebind (Qwen2ForCausalLM.forward, P0-4j): training was unaffected, every
        # held-out evaluation was lost. Introspection only; the call path is unchanged.
        if name != "apply":
            try:
                wrapped.__signature__ = inspect.signature(orig)
            except (TypeError, ValueError):
                pass
        if name == "apply" and isinstance(owner, type):
            setattr(owner, name, staticmethod(wrapped))
        else:
            setattr(owner, name, wrapped)
        _SLOTS.append(slot)
        if len(_SLOTS) == 1:
            _cfg.update(cfg)             # back-compat: _cfg mirrors the first overlay
        mode = "identity rebind (counters only)" if impl is None else f"-> {cfg['replacement']}"
        print(f"[geak-overlay] {cfg['target']} {mode}", flush=True)
        return True
    except Exception as e:
        # An overlay that fails to install must SAY SO. Falling through silently would leave the run
        # measuring the original implementation while the report claims a candidate was tested.
        print(f"[geak-overlay] INSTALL FAILED for {manifest_path!r} ({e!r}) — the run is measuring "
              f"the ORIGINAL implementation for this target; do not report it as a candidate leg.",
              flush=True)
        return False


def install():
    """Install every manifest in GEAK_OVERLAY, left to right.

    ALL-OR-NOTHING by design. A partially installed stack is the worst possible state: the run looks
    like a candidate leg, carries engagement proof for the overlays that did load, and silently
    measures the original implementation for the one that did not. The combined number would then be
    attributed to a stack that never existed.
    """
    if not ENABLED:
        return False
    if torch is None:
        # torch is soft-imported so verdict() can be read from outside the container. Installing
        # without it would rebind the target to a wrapper whose first line calls torch.is_tensor,
        # i.e. it would replace a working op with one that raises on the first call.
        print("[geak-overlay] REFUSING to install: torch is not importable here. The rebind would "
              "replace a working op with a wrapper that raises on its first call.", flush=True)
        return False
    ok = [_install_one(m) for m in _MANIFESTS]
    if all(ok):
        if len(_MANIFESTS) > 1:
            print(f"[geak-overlay] STACK of {len(_MANIFESTS)} overlays installed: "
                  f"{[s['cfg'].get('target') for s in _SLOTS]}", flush=True)
        return True
    print(f"[geak-overlay] STACK INCOMPLETE: {sum(ok)}/{len(_MANIFESTS)} overlays installed. "
          f"This leg is NOT a measurement of the intended stack — discard it.", flush=True)
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
