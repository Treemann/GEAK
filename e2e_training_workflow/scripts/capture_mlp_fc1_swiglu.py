"""Capture the MLP fc1 + SwiGLU COMPOSITE (RMSNorm -> fc1 gate/up GEMM -> SiLU(gate)*up) fwd+bwd.

WHY A COMPOSITE, AND NOT `capture_train_shapes.py` ON `_LayerNormLinear`. The head this task serves
is `mlp_fc1_swiglu`: fc1 forward (11.286%) + the two fc1 backward halves (12.447% + 10.551%) + the
SwiGLU pointwise that sits between fc1 and fc2 (2.730%) = 37.01% of the critical path. The stated
first win is to make the activation an fc1 EPILOGUE -- i.e. to delete the pointwise pass, not to
speed it up. An oracle that froze only `_LayerNormLinear` could not score that at all: the candidate
would have to return a [4096, 1, 37888] pre-activation tensor it is trying not to materialise, and
the 776 MB/call the pointwise moves would sit outside the measured latency. So the op is the
composite, and its output is the POST-activation [4096, 1, 18944] tensor that fc2 consumes.

WHY THE BOUNDARY IS EXACTLY HERE, on both sides:
  * upstream -- fc1's input is the block's hidden state, which ALSO feeds the residual add. That is
    precisely the tensor `gate_convergence.md` warns about: a `register_hook` on it returns the SUM
    over consumers. We never hook it; the op's gradient w.r.t. it comes from an isolated re-run.
  * downstream -- the activation output is a fresh tensor whose ONLY consumer is `linear_fc2`
    (megatron/core/transformer/mlp.py: `intermediate_parallel` is produced by the activation and
    passed to `self.linear_fc2`, and nothing else reads it). A tensor hook there is therefore this
    op's true grad_output and not a sum, which is what makes capturing a real `grad_output` sound
    for this op where it was not for `_LayerNormLinear`'s input.

WHY fc2 IS NOT IN THE OP. fc2 is a separate head (ranks 5/6/7 of the profile) with its own
candidate; including it would freeze two heads into one task and make the attribution of any win
unreadable. The cut is at the activation output because that is where the epilogue fusion ends.

WHAT IS CAPTURED AS THE ARGUMENT LIST is the `_LayerNormLinear.apply` tensor arguments in their
call order -- `x`, `ln_weight`, `weight` (and `bias` if the model has one) -- so
`build_task_dir.py`'s existing `rmsnorm_linear`-style layout reads it unchanged. The SwiGLU is not
an argument: it is part of the OPERATION, recorded in meta as the reference
`rmsnorm_linear_swiglu`.

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original chain (`_LayerNormLinear.apply` ->
`bias_swiglu_impl`), in the live process, on the frozen inputs. Same doctrine as
capture_train_shapes: the real backward cannot be paired to the captured forward (one `Function`
class serves QKV and fc1), and the call carries runtime handles (ProcessGroup, quantizers,
wgrad_store) that cannot be rebuilt offline.

Env:
    GEAK_MLP_CAPTURE_DIR=/work/tasks/mlp_fc1_swiglu   (required; enables the capture)
    GEAK_MLP_CAPTURE_STEP=13                          (capture on this step, after warmup)
    GEAK_MLP_CAPTURE_RANKS=0                          (one rank: concurrent writers corrupt the file)
"""
from __future__ import annotations

import json
import os

import torch

_DIR = os.environ.get("GEAK_MLP_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_MLP_CAPTURE_STEP", "0") or 0)
_RANKS = {int(r) for r in os.environ.get("GEAK_MLP_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)

_state = {"step": 0, "captured": False, "fc1": None}


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
        from megatron.core.transformer import mlp as _mlp
        from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
    except Exception as e:
        print(f"[geak-mlp] cannot import the MLP / TE call sites: {e!r}", flush=True)
        return False

    os.makedirs(_DIR, exist_ok=True)

    orig_apply = _LayerNormLinear.apply
    orig_swiglu = getattr(_mlp, "bias_swiglu_impl", None)
    if orig_swiglu is None:
        print("[geak-mlp] megatron.core.transformer.mlp has no bias_swiglu_impl; not armed",
              flush=True)
        return False

    def armed():
        return (not _state["captured"]) and _state["step"] + 1 == _STEP

    def wrapped_apply(*a, **kw):
        # Keep a REFERENCE to the most recent call's arguments while armed. This is how the fc1 call
        # site is identified without guessing: `_LayerNormLinear` serves both the QKV projection and
        # fc1, and the call immediately preceding the activation, inside the same MLP.forward, is
        # fc1 by construction. The shape assertion below turns "by construction" into "checked".
        if armed():
            _state["fc1"] = (a, kw)
        return orig_apply(*a, **kw)

    def wrapped_swiglu(inp, bias, *a, **kw):
        if not armed():
            return orig_swiglu(inp, bias, *a, **kw)
        fc1 = _state["fc1"]
        why = None
        if fc1 is None:
            why = "no _LayerNormLinear call preceded the activation (fc1 is not the TE fused path)"
        elif not torch.is_tensor(inp):
            why = "activation input is not a tensor"
        if why is None:
            tens = [t for t in fc1[0] if torch.is_tensor(t)]
            w = next((t for t in tens if t.dim() == 2 and t.shape[0] == inp.shape[-1]), None)
            if w is None:
                why = (f"the preceding _LayerNormLinear has no [{inp.shape[-1]}, k] weight, so it is "
                       f"not the fc1 that produced this activation input")
            elif inp.shape[-1] % 2:
                why = f"activation input width {inp.shape[-1]} is not even; not a gated unit"
        if why:
            print(f"[geak-mlp] NOT capturing: {why}", flush=True)
            _state["captured"] = True
            return orig_swiglu(inp, bias, *a, **kw)

        _state["captured"] = True
        out = orig_swiglu(inp, bias, *a, **kw)
        if not (torch.is_tensor(out) and out.requires_grad):
            print("[geak-mlp] activation output is not differentiable; skipped", flush=True)
            return out

        def _hook(g):
            # A capture failure must NEVER kill an 8-rank training job (this is not hypothetical:
            # an unguarded torch.save once raised inside a backward hook and took down the job).
            try:
                _capture(orig_apply, orig_swiglu, fc1, inp, bias, a, kw, out, g.detach().clone())
            except Exception as e:
                print(f"[geak-mlp] capture FAILED (run continues): {e!r}", flush=True)
            return g

        out.register_hook(_hook)
        return out

    _LayerNormLinear.apply = staticmethod(wrapped_apply)
    _mlp.bias_swiglu_impl = wrapped_swiglu
    print(f"[geak-mlp] armed on _LayerNormLinear.apply + mlp.bias_swiglu_impl for step {_STEP} "
          f"(rank {_RANK})", flush=True)
    return True


def _capture(orig_apply, orig_swiglu, fc1, fc1_out_live, act_bias, act_a, act_kw, live_out, grad_out):
    args, kwargs = fc1
    leaves, tensor_nodes, main_grads = [], [], []

    def _clone(node):
        if torch.is_tensor(node):
            t = node.detach().clone()
            if node.requires_grad:
                t.requires_grad_(True)
                leaves.append(t)
            # Under gradient_accumulation_fusion TE writes the weight gradient into `main_grad` and
            # returns None for it. A private zeroed buffer both satisfies the unconditional
            # dereference and leaves us holding exactly this op's weight-gradient contribution.
            mg = getattr(node, "main_grad", None)
            if mg is not None:
                t.main_grad = torch.zeros_like(mg)
                main_grads.append((tuple(t.shape), t))
            tensor_nodes.append(node)
            return t
        if isinstance(node, (list, tuple)):
            return type(node)(_clone(v) for v in node)
        return node

    new_args = [_clone(v) for v in args]
    new_kwargs = {k: _clone(v) for k, v in kwargs.items()}
    bias_clone = _clone(act_bias) if torch.is_tensor(act_bias) else act_bias

    with torch.enable_grad():
        fc1_out = orig_apply(*new_args, **new_kwargs)
        if isinstance(fc1_out, (list, tuple)) and torch.is_tensor(fc1_out[0]):
            fc1_out = fc1_out[0]
        act = orig_swiglu(fc1_out, bias_clone, *act_a, **act_kw)
        grads = torch.autograd.grad(act, leaves, grad_outputs=grad_out, allow_unused=True,
                                    retain_graph=False)

    iso = {str(i): _snap(g) for i, g in enumerate(grads) if torch.is_tensor(g)}
    for shape, t in main_grads:
        mg = getattr(t, "main_grad", None)
        if mg is not None and float(mg.abs().sum()) > 0:
            iso[f"main_grad{list(shape)}"] = _snap(mg)

    # THE CHECK THAT MAKES THE COMPOSITION AUDITABLE rather than asserted: the isolated chain, run on
    # clones of the frozen inputs, must reproduce the LIVE activation output. A mis-assembled
    # composite (wrong call site, wrong activation variant, gate/up swapped) fails here loudly
    # instead of freezing a plausible wrong oracle.
    iso_vs_live = float((act.detach().float() - live_out.detach().float()).abs().max())

    # Only the tensors the reference layout names, in call order. A zero-element `bias` is TE's
    # "no bias" sentinel and is DROPPED: passed through it would make the frozen op a bias-add on an
    # empty tensor, which neither the reference nor a candidate can reproduce.
    arg_tensors = [t for t in args if torch.is_tensor(t) and t.numel() > 0]

    rec = {
        "target": "megatron.core.transformer.mlp:MLP.forward"
                  " [_LayerNormLinear (RMSNorm+fc1) -> bias_swiglu_impl]",
        "step": _STEP, "rank": _RANK,
        "args": [_snap(t) for t in arg_tensors],
        "kwargs": {},
        "grad_output": _snap(grad_out),
        "output": _snap(live_out.detach()),
        "isolated": {
            "out": _snap(act.detach()),
            "grads": iso,
            "grad_shapes": [list(t.shape) for t in leaves],
        },
        "composition_evidence": {
            "isolated_vs_live_max_abs": iso_vs_live,
            "fc1_out_shape": list(fc1_out_live.shape),
            "act_out_shape": list(live_out.shape),
            "gate_up_split": [int(fc1_out_live.shape[-1]) // 2, int(fc1_out_live.shape[-1]) // 2],
            "activation_bias": bool(torch.is_tensor(act_bias) and act_bias.numel() > 0),
            "dropped_empty_bias": bool(any(torch.is_tensor(t) and t.numel() == 0 for t in args)),
            "grad_output_nonzero_frac": float((grad_out != 0).float().mean()),
            "grad_output_abs_mean": float(grad_out.abs().mean()),
            "note": ("grad_output is captured from the ACTIVATION OUTPUT, whose sole consumer is "
                     "linear_fc2 -- so a tensor hook is this op's gradient, not a sum over "
                     "consumers. The hazard called out in gate_convergence.md applies to the "
                     "block's hidden state (also read by the residual add), which is an INPUT here "
                     "and is never hooked."),
        },
        "opaque_args": [{"__opaque__": type(v).__name__,
                         "module": getattr(type(v), "__module__", "?"), "repr": repr(v)[:200]}
                        for v in args if not torch.is_tensor(v)],
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}.pt")
    tmp = path + ".partial"
    torch.save(rec, tmp)
    os.replace(tmp, path)
    with open(os.path.join(_DIR, "capture_meta.json"), "w") as f:
        json.dump({"target": rec["target"], "step": _STEP, "rank": _RANK,
                   "shapes": {"args": [list(t.shape) for t in arg_tensors],
                              "out": list(live_out.shape), "grad_output": list(grad_out.shape),
                              "grads": rec["isolated"]["grad_shapes"]},
                   "dtypes": {"args": [str(t.dtype) for t in arg_tensors],
                              "out": str(live_out.dtype), "grad_output": str(grad_out.dtype)},
                   "composition_evidence": rec["composition_evidence"],
                   "isolated_grad_keys": sorted(iso)}, f, indent=2)
    print(f"[geak-mlp] wrote {path}; iso_vs_live={iso_vs_live:.3g} grads={sorted(iso)}", flush=True)
