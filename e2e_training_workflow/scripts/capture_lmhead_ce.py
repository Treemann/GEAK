"""Capture the LM-head + cross-entropy COMPOSITE (output projection -> CE -> loss-mask) fwd+bwd.

WHY THIS IS NOT `capture_train_shapes.py`. That module wraps ONE callable. The candidate here is a
FUSED output-projection + cross-entropy that gathers only the loss-unmasked rows, so the op whose
latency and numerics must be frozen spans two call sites that no single `autograd.Function`
covers:

    logits = self.output_layer(hidden_states, weight=W)      # ColumnParallelLinear, TP1, no bias
    loss   = self.compute_language_model_loss(labels, logits) # vocab_parallel_cross_entropy
    out    = loss * loss_mask                                 # absorbed from pretrain_gpt.loss_func

Wrapping only the linear would freeze an op whose fusion partner is invisible; wrapping only the CE
would hide both GEMMs, which are 5.8 of the head's 6.4 critical-path points.

WHY `* loss_mask` IS PART OF THE OP, not a liberty taken with it. The candidate's whole premise is
that loss-masked rows contribute exactly zero. That premise has to be written into the OP CONTRACT,
otherwise the oracle compares the candidate against a reference that still computes garbage on rows
the candidate elides, and a mathematically exact kernel fails its own gate. Moving the multiply
inside is safe in BOTH directions because `loss_mask` is 0/1: `pretrain_gpt.loss_func` multiplies by
it again, and x*m*m == x*m, so the integration needs no change to the loss function.

WHAT THE CAPTURE PROVES, rather than assumes: it records the live `grad_output` of the loss tensor
and checks that its support is exactly the support of `loss_mask`. That is the measured evidence
that eliding masked rows is exact and not an approximation.

The golden comes from an ISOLATED RE-RUN of the original chain, in the live process, on the frozen
inputs -- the same doctrine as capture_train_shapes: the real backward cannot be paired to the
captured forward, and the call carries runtime handles that cannot be replayed offline.

Env:
    GEAK_LMHEAD_CAPTURE_DIR=/work/tasks/lmhead_ce     (required; enables the capture)
    GEAK_LMHEAD_CAPTURE_STEP=13                       (capture on this step, after warmup)
    GEAK_LMHEAD_CAPTURE_RANKS=0                       (one rank: concurrent writers corrupt the file)
"""
from __future__ import annotations

import json
import os

import torch


def _tp_world_size():
    """Tensor-parallel world size, or 1 if Megatron's parallel state is unavailable.

    Deliberately fails SAFE-LOUD in the caller rather than here: if the import works we get the real
    number; if it does not, we return 1 and the capture proceeds, because on a stack without
    megatron.core there is no vocab sharding to worry about. Any exception is swallowed because this
    runs inside a forward hook on an 8-rank job.
    """
    try:
        from megatron.core import parallel_state
        return int(parallel_state.get_tensor_model_parallel_world_size())
    except Exception:
        return 1

_DIR = os.environ.get("GEAK_LMHEAD_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_LMHEAD_CAPTURE_STEP", "0") or 0)
_RANKS = {int(r) for r in os.environ.get("GEAK_LMHEAD_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)

_state = {"step": 0, "captured": False, "rec": None}


def note_step(step):
    _state["step"] = step


def _snap(x):
    """Same on-disk marker as capture_train_shapes, so build_task_dir reads this record unchanged."""
    if torch.is_tensor(x):
        return {"__t__": True, "data": x.detach().to("cpu", copy=True),
                "dtype": str(x.dtype), "shape": list(x.shape),
                "requires_grad": bool(x.requires_grad)}
    return x


def install():
    if not ENABLED:
        return False
    try:
        from megatron.core.models.gpt.gpt_model import GPTModel
    except Exception as e:
        print(f"[geak-lmhead] cannot import GPTModel: {e!r}", flush=True)
        return False

    orig = GPTModel._postprocess
    os.makedirs(_DIR, exist_ok=True)

    def wrapped(self, *a, **kw):
        if _state["captured"] or _state["step"] + 1 != _STEP:
            return orig(self, *a, **kw)
        hidden = kw.get("hidden_states")
        labels = kw.get("labels")
        loss_mask = kw.get("loss_mask")
        # Refuse LOUDLY rather than capture a silently different op. Each of these would change what
        # the frozen oracle means, and each is cheap to detect and impossible to notice later.
        why = None
        if hidden is None or labels is None:
            why = "hidden_states/labels absent (not the training loss path)"
        elif loss_mask is None:
            why = ("loss_mask is None at GPTModel._postprocess -- the unmasked-gather candidate has "
                   "nothing to gather on and would have to be plumbed from forward_step")
        elif getattr(self, "mtp_process", False) or getattr(self.config, "mtp_num_layers", None):
            why = "MTP is active; the head is not the plain output-projection + CE chain"
        elif getattr(self, "share_embeddings_and_output_weights", False):
            why = "tied embeddings: the head weight is shared and wgrad is not this op's alone"
        elif getattr(self.config, "cross_entropy_loss_fusion", False):
            why = "cross_entropy_loss_fusion is on; the CE baseline is a different implementation"
        elif _tp_world_size() > 1:
            # THE SILENT ONE. l0_gate.lmhead_ce_reference takes logsumexp over weight.shape[0] and
            # so assumes it holds the WHOLE vocabulary. At TP>1 the output_layer holds a vocab SHARD
            # and vocab_parallel_cross_entropy all-reduces the max and the sum across ranks, so a
            # captured oracle would freeze a shard-local logsumexp: wrong, self-consistent, and
            # indistinguishable from a correct one once frozen. Every leg measured so far was TP1,
            # which is exactly why this could sit here unnoticed -- refuse rather than wait for the
            # first TP>1 run to produce a confident wrong number.
            why = (f"tensor_model_parallel_world_size={_tp_world_size()}: the head holds a vocab "
                   f"SHARD and CE is reduced across ranks, so a single-rank capture would freeze a "
                   f"shard-local logsumexp as if it were the full-vocabulary reference")
        if why:
            print(f"[geak-lmhead] NOT capturing: {why}", flush=True)
            _state["captured"] = True
            return orig(self, *a, **kw)

        _state["captured"] = True
        out = orig(self, *a, **kw)          # [b, s] loss, unmasked
        if not (torch.is_tensor(out) and out.requires_grad):
            print("[geak-lmhead] _postprocess did not return a differentiable loss; skipped", flush=True)
            return out

        def _hook(g):
            # A capture failure must NEVER kill an 8-rank training job.
            try:
                _capture(self, hidden, labels, loss_mask, out, g)
            except Exception as e:
                print(f"[geak-lmhead] capture FAILED (run continues): {e!r}", flush=True)
            return g

        out.register_hook(_hook)
        return out

    GPTModel._postprocess = wrapped
    print(f"[geak-lmhead] armed on GPTModel._postprocess for step {_STEP} (rank {_RANK})", flush=True)
    return True


def _capture(model, hidden, labels, loss_mask, live_out, grad_out):
    W = model.output_layer.weight

    # THE FACT THE WHOLE CANDIDATE RESTS ON, measured rather than assumed: the gradient arriving at
    # the loss tensor is supported exactly on the loss-unmasked positions. If that were not true,
    # eliding masked rows would be an approximation and this task would be dishonest.
    go = grad_out.detach().float()
    m = loss_mask.detach().float().reshape(go.shape)
    go_nz = (go != 0)
    m_nz = (m != 0)
    mask_evidence = {
        "grad_output_nonzero": int(go_nz.sum()),
        "loss_mask_nonzero": int(m_nz.sum()),
        "total_positions": int(go.numel()),
        "grad_support_equals_mask_support": bool(torch.equal(go_nz, m_nz)),
        "grad_output_on_masked_max_abs": float(go[~m_nz].abs().max()) if int((~m_nz).sum()) else 0.0,
        "unmasked_fraction": float(m_nz.float().mean()),
    }

    # ---- isolated re-run of the ORIGINAL chain, here, synchronously, on the frozen inputs -------
    h = hidden.detach().clone().requires_grad_(True)
    w = W.detach().clone().requires_grad_(True)
    mg = getattr(W, "main_grad", None)
    if mg is not None:
        # Under gradient_accumulation_fusion the linear writes wgrad straight into main_grad and
        # returns None for it. A private zeroed buffer both satisfies the unconditional dereference
        # and hands us exactly this op's weight-gradient contribution.
        w.main_grad = torch.zeros_like(mg)
    with torch.enable_grad():
        logits, _ = model.output_layer(h, weight=w)
        loss_sb = model.compute_language_model_loss(labels, logits)      # [b, s]
        masked = loss_sb * loss_mask.to(loss_sb.dtype).reshape(loss_sb.shape)
        grads = torch.autograd.grad(masked, [h, w], grad_outputs=grad_out.detach().clone(),
                                    allow_unused=True, retain_graph=False)
    dgrad = grads[0]
    wgrad = grads[1]
    if wgrad is None:
        wgrad = getattr(w, "main_grad", None)
    if wgrad is None:
        raise RuntimeError("no weight gradient from the isolated re-run (neither autograd nor main_grad)")

    rec = {
        "target": "megatron.core.models.gpt.gpt_model:GPTModel._postprocess"
                  " [output_layer -> compute_language_model_loss -> *loss_mask]",
        "step": _STEP, "rank": _RANK,
        # ORDER IS THE CONTRACT: build_task_dir zips these with layout["args"].
        "args": [_snap(hidden.detach()), _snap(W.detach()),
                 _snap(labels.detach()), _snap(loss_mask.detach())],
        "kwargs": {},
        "grad_output": _snap(grad_out.detach()),
        "output": _snap(masked.detach()),
        "isolated": {
            "out": _snap(masked.detach()),
            "grads": {"0": _snap(dgrad), "1": _snap(wgrad)},
            "grad_shapes": [list(h.shape), list(w.shape)],
            "main_grad_shapes": [list(w.shape)] if mg is not None else [],
        },
        "mask_evidence": mask_evidence,
        "live_out_matches_isolated_on_unmasked": float(
            (live_out.detach().float() * m - masked.detach().float()).abs().max()),
        "context": {
            "gradient_accumulation_fusion": bool(model.output_layer.gradient_accumulation_fusion),
            "tp_world_size": int(getattr(model.output_layer, "explicit_expert_comm", False) or 0) or None,
            "vocab_size": int(W.shape[0]),
            "hidden_size": int(W.shape[1]),
            "bias": model.output_layer.bias is not None,
            "cross_entropy_loss_fusion": bool(getattr(model.config, "cross_entropy_loss_fusion", False)),
        },
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}.pt")
    tmp = path + ".partial"
    torch.save(rec, tmp)
    os.replace(tmp, path)
    with open(os.path.join(_DIR, "capture_meta.json"), "w") as f:
        json.dump({"target": rec["target"], "step": _STEP, "rank": _RANK,
                   "shapes": {"hidden_states": list(hidden.shape), "weight": list(W.shape),
                              "labels": list(labels.shape), "loss_mask": list(loss_mask.shape),
                              "out": list(masked.shape), "grad_output": list(grad_out.shape),
                              "dgrad": list(dgrad.shape), "wgrad": list(wgrad.shape)},
                   "dtypes": {"hidden_states": str(hidden.dtype), "weight": str(W.dtype),
                              "labels": str(labels.dtype), "loss_mask": str(loss_mask.dtype),
                              "out": str(masked.dtype), "grad_output": str(grad_out.dtype),
                              "dgrad": str(dgrad.dtype), "wgrad": str(wgrad.dtype)},
                   "mask_evidence": mask_evidence,
                   "context": rec["context"]}, f, indent=2)
    print(f"[geak-lmhead] wrote {path}; mask_evidence={mask_evidence}", flush=True)
