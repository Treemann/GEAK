"""Capture HF `Qwen2ForCausalLM`'s LM head + causal-LM loss (fwd + bwd) AS IT RUNS on the hf_trl stack.
Head `hf_lmhead_fused_ce`.

WHAT THE OP IS. The tail of Qwen2ForCausalLM.forward, frozen as ONE op (layout `hf_lmhead_ce`):

    logits = hidden_states @ W^T                    hidden [1,S,3584] bf16 (model.norm output),
                                                    W = lm_head.weight [152064,3584], no bias
    loss   = ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch)
           = sum_{t: shift_label[t] != -100} CE(float(logits[t]), shift_label[t]) / num_items_in_batch

    shift_label = pad(labels, (0,1), -100)[..., 1:]   (HF's own shift: the LAST position never
                                                       carries loss; ignore_index -100)

Gradients: dgrad = d loss / d hidden [1,S,3584], wgrad = d loss / d W [152064,3584]. One call per
microbatch: 8 calls per rank per step (GBS 64 / 8 ranks / MBS 1).

WHY THE WHOLE TAIL AND NOT THE GEMM. The lever is a chunked fused lm_head + CE that never materialises
the bf16 [S,152064] logits nor their fp32 upcast; only an oracle whose inputs are (hidden, W, labels,
num_items_in_batch) and whose outputs are (loss, dhidden, dW) can score it. num_items_in_batch is an
INPUT, not a constant: it is the global count of labels != -100 over the step's microbatches on all
ranks (Trainer.get_batch_samples + gather), different every step.

HOOKS. A forward pre-hook WITH KWARGS on the Qwen2ForCausalLM instance (labels, num_items_in_batch
and any other loss kwargs pass through it), forward pre/post hooks on its lm_head (the live input,
weight, autocast state and logits), and a forward hook on the instance whose tensor hook on the
returned `loss` captures grad_output (d total / d loss: the Trainer multiplies by num_processes
before backward). No hf5 overlay rebinds Qwen2ForCausalLM.forward, lm_head or the loss function, so
the hooks see the original path exactly; the live forward is not altered.

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original (nn.Linear.forward of THIS lm_head, called
unbound so the capture hooks do not re-fire, then THIS model's own `loss_function` with the SAME
kwargs) on clones of hidden and W taken at FORWARD time, in the live process, inside the loss's
grad_output hook. W is swapped for a cloned leaf for that one call (identity-verified) so the re-run
yields this call's OWN dW. Recorded per frozen call: isolated loss / logits == live (bitwise), a
second isolated re-run == the first (bitwise), and the weight's storage facts.

A MASKED VARIANT of one call (GEAK_HFLMCE_CAPTURE_MASKED) is also re-run with a prompt-like prefix
plus scattered positions of the labels set to -100: on this workload every row carries loss, so
without it the oracle could not see a kernel that ignores ignore_index. Same hidden, W, grad_output
and num_items_in_batch; its golden is the ORIGINAL on those labels.

Env:
    GEAK_HFLMCE_CAPTURE_DIR=/work/tasks/<name>_cap   (required; enables the capture)
    GEAK_HFLMCE_CAPTURE_STEP=12                      (1-based step to capture, after warmup)
    GEAK_HFLMCE_CAPTURE_CALLS=0,3,7                  (microbatch indices within the step to freeze)
    GEAK_HFLMCE_CAPTURE_RANKS=0,4                    (one file per rank and call; no shared writer)
    GEAK_HFLMCE_CAPTURE_MASKED=0                     (call that also gets the masked-label variant;
                                                      -1 = none)
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFLMCE_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFLMCE_CAPTURE_STEP", "0") or 0)
_CALLS = {int(c) for c in os.environ.get("GEAK_HFLMCE_CAPTURE_CALLS", "0,3,7").split(",")
          if c.strip().isdigit()}
_RANKS = {int(r) for r in os.environ.get("GEAK_HFLMCE_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_MASKED = int(os.environ.get("GEAK_HFLMCE_CAPTURE_MASKED", "0") or 0)
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
TARGET = ("transformers.models.qwen2.modeling_qwen2:Qwen2ForCausalLM.forward"
          "[lm_head -> loss_function(ForCausalLMLoss)]")
# Record argument order == layout `hf_lmhead_ce` args (l0_gate.LAYOUTS).
ROLES = ["hidden_states", "weight", "labels", "num_items_in_batch"]
IGNORE = -100

_state = {"step": 0, "calls": 0, "per_call": [], "written": [], "meta_done": False, "cur": None,
          "model": None}


def note_step(step):
    """Called at every step end with the number of steps COMPLETED."""
    _state["step"] = step
    if step >= _STEP and not _state["meta_done"] and _state["per_call"]:
        _state["meta_done"] = True
        _write_capture_meta()


def _snap(t):
    return {"__t__": True, "data": t.detach().to("cpu", copy=True), "dtype": str(t.dtype),
            "shape": list(t.shape), "requires_grad": bool(t.requires_grad)}


def _swap(mod, attr, t):
    """Point mod.<attr> at t for one call (FSDP use_orig_params keeps a plain-tensor view in the
    instance __dict__, shadowing `_parameters`; checked first, identity-verified after)."""
    if attr in mod.__dict__:
        tok = (attr, "d", mod.__dict__[attr])
        mod.__dict__[attr] = t
    elif attr in mod._parameters:
        tok = (attr, "p", mod._parameters[attr])
        mod._parameters[attr] = t
    else:
        raise RuntimeError(f"{type(mod).__name__} has no {attr} attribute to swap")
    if getattr(mod, attr) is not t:
        _restore(mod, tok)
        raise RuntimeError(f"{attr} swap did not take effect (attribute lookup bypasses it)")
    return tok


def _restore(mod, tok):
    attr, kind, old = tok
    if kind == "p":
        mod._parameters[attr] = old
    else:
        mod.__dict__[attr] = old


def _storage_facts(w):
    f = {"type": type(w).__name__, "is_param": isinstance(w, torch.nn.Parameter),
         "is_view": w._base is not None, "contig": w.is_contiguous(),
         "stride": list(w.stride()), "requires_grad": bool(w.requires_grad),
         "dtype": str(w.dtype), "shape": list(w.shape)}
    try:
        if w._base is not None:
            b = w._base
            f["base_shape"] = list(b.shape)
            f["base_dtype"] = str(b.dtype)
            f["offset_elems"] = (w.data_ptr() - b.data_ptr()) // w.element_size()
    except Exception as e:  # noqa: BLE001
        f["err"] = repr(e)[:200]
    return f


def _masked_labels(labels):
    """Prompt-like prefix (first 40%) plus every 7th position of the rest -> ignore_index."""
    lm = labels.clone()
    s = lm.shape[-1]
    p = int(0.4 * s)
    lm[..., :p] = IGNORE
    lm[..., p::7] = IGNORE
    return lm


def _isolated(model, frozen, go, ac, labels, loss_kw):
    """The ORIGINAL lm_head + loss_function on clones of the frozen tensors. Returns
    (loss, logits, [dhidden, dW])."""
    lm = model.lm_head
    h = frozen["hidden_states"].detach().clone().requires_grad_(True)
    w = frozen["weight"].detach().clone().requires_grad_(True)
    tok = _swap(lm, "weight", w)
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=ac["dtype"], enabled=ac["enabled"]):
            logits = torch.nn.Linear.forward(lm, h)     # unbound -> the capture's hooks do not fire
            loss = model.loss_function(logits=logits, labels=labels,
                                       vocab_size=model.config.vocab_size, **loss_kw)
        grads = torch.autograd.grad(loss, [h, w], grad_outputs=go)
    finally:
        _restore(lm, tok)
    return loss.detach(), logits.detach(), [g.detach() for g in grads]


def install(model=None):
    if not ENABLED:
        return False
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hflmce] cannot import Qwen2ForCausalLM: {e!r}", flush=True)
        return False
    if model is None or not isinstance(model, mq.Qwen2ForCausalLM):
        print(f"[geak-hflmce] model is {type(model).__name__}, not Qwen2ForCausalLM; not armed", flush=True)
        return False
    os.makedirs(_DIR, exist_ok=True)
    _state["model"] = model
    model.register_forward_pre_hook(_top_pre, with_kwargs=True)
    model.register_forward_hook(_top_post)
    model.lm_head.register_forward_pre_hook(_lm_pre)
    model.lm_head.register_forward_hook(_lm_post)
    fw = mq.Qwen2ForCausalLM.forward
    print(f"[geak-hflmce] armed on Qwen2ForCausalLM for step {_STEP}, calls {sorted(_CALLS)}, masked "
          f"{_MASKED}, rank {_RANK}; Qwen2ForCausalLM.forward = {getattr(fw, '__module__', '?')}."
          f"{getattr(fw, '__qualname__', '?')}; loss_function = "
          f"{getattr(model.loss_function, '__qualname__', '?')}", flush=True)
    return True


def _active():
    return _state["step"] + 1 == _STEP


def _top_pre(mod, args, kwargs):
    if not _active():
        return None
    k = _state["calls"]
    _state["calls"] += 1
    labels = kwargs.get("labels")
    nib = kwargs.get("num_items_in_batch")
    ids = kwargs.get("input_ids", args[0] if args else None)
    loss_kw = {kk: v for kk, v in kwargs.items()
               if kk not in ("input_ids", "attention_mask", "position_ids", "past_key_values",
                             "inputs_embeds", "labels", "use_cache", "cache_position", "logits_to_keep")}
    row = {"call": k, "kwargs_keys": sorted(kwargs.keys()), "loss_kwargs_keys": sorted(loss_kw.keys()),
           "S": int(ids.shape[-1]) if ids is not None else None,
           "labels_shape": list(labels.shape) if labels is not None else None,
           "labels_ignored": int((labels == IGNORE).sum()) if labels is not None else None,
           "labels_last_is_ignored": bool(labels[..., -1].eq(IGNORE).all()) if labels is not None else None,
           "labels_eq_input_ids": (bool(torch.equal(labels, ids)) if labels is not None and ids is not None
                                   and labels.shape == ids.shape else None),
           "num_items_in_batch": (int(nib) if torch.is_tensor(nib) and nib.numel() == 1
                                  else (nib if isinstance(nib, (int, float)) else None)),
           "num_items_type": (f"tensor {nib.dtype} {list(nib.shape)} {nib.device}" if torch.is_tensor(nib)
                              else type(nib).__name__),
           "attention_mask_zeros": (int((kwargs["attention_mask"] == 0).sum())
                                    if torch.is_tensor(kwargs.get("attention_mask")) else None),
           "logits_to_keep": kwargs.get("logits_to_keep", 0) if not torch.is_tensor(
               kwargs.get("logits_to_keep")) else "tensor",
           "grad_enabled": torch.is_grad_enabled()}
    _state["per_call"].append(row)
    cur = {"k": k, "row": row, "labels": labels, "nib": nib, "loss_kw": loss_kw, "frozen": None,
           "live_logits": None, "done": False}
    _state["cur"] = cur
    return None


def _lm_pre(mod, inp):
    cur = _state["cur"]
    if not _active() or cur is None:
        return None
    x = inp[0]
    ac = {"enabled": torch.is_autocast_enabled("cuda"), "dtype": torch.get_autocast_dtype("cuda")}
    row = cur["row"]
    row.update({"hidden_shape": list(x.shape), "hidden_dtype": str(x.dtype), "hidden_contig": x.is_contiguous(),
                "hidden_requires_grad": bool(x.requires_grad),
                "autocast": {"enabled": ac["enabled"], "dtype": str(ac["dtype"])},
                "weight_storage": _storage_facts(mod.weight),
                "bias": mod.bias is not None})
    cur["ac"] = ac
    if cur["k"] in _CALLS:
        cur["frozen"] = {"hidden_states": x.detach().clone(), "weight": mod.weight.detach().clone()}
    return None


def _lm_post(mod, inp, out):
    cur = _state["cur"]
    if not _active() or cur is None:
        return None
    cur["row"]["logits_dtype"] = str(out.dtype)
    cur["row"]["logits_shape"] = list(out.shape)
    if cur["frozen"] is not None:
        cur["live_logits"] = out          # alive anyway until the loss is formed; kept to compare
    return None


def _top_post(mod, inp, out):
    cur = _state["cur"]
    if not _active() or cur is None:
        return None
    loss = out.get("loss") if hasattr(out, "get") else None
    cur["row"]["loss_dtype"] = str(loss.dtype) if loss is not None else None
    cur["row"]["loss_shape"] = list(loss.shape) if loss is not None else None
    if cur["frozen"] is None:
        _state["cur"] = None
        return None
    if loss is None or not loss.requires_grad:
        print(f"[geak-hflmce] call {cur['k']}: loss missing or not differentiable, not frozen", flush=True)
        _state["cur"] = None
        return None
    cur["live_loss"] = loss.detach().clone()

    def _hook(g, cur=cur):
        try:
            if not cur["done"]:
                cur["done"] = True
                _finish(cur, g.detach().clone())
        except Exception as e:  # noqa: BLE001  -- a capture must never kill the run
            print(f"[geak-hflmce] capture of call {cur['k']} FAILED (run continues): {e!r}", flush=True)
        finally:
            cur["live_logits"] = None
            cur["frozen"] = None
        return g

    loss.register_hook(_hook)
    _state["cur"] = None
    return None


def _nib_tensor(nib, dev):
    if torch.is_tensor(nib):
        return nib.detach().clone()
    return torch.tensor(nib, device=dev)


def _record(cur, go, labels, tag, model):
    t0 = time.time()
    k, row, ac, fr = cur["k"], cur["row"], cur["ac"], cur["frozen"]
    loss_kw = dict(cur["loss_kw"])
    l1, lg1, g1 = _isolated(model, fr, go, ac, labels, loss_kw)
    live_eq = {}
    if tag == "":
        live_eq = {"iso_loss_equals_live": bool(torch.equal(l1, cur["live_loss"])),
                   "iso_loss_vs_live_abs": float((l1.float() - cur["live_loss"].float()).abs()),
                   "iso_logits_equals_live": (bool(torch.equal(lg1, cur["live_logits"]))
                                              if cur["live_logits"] is not None else None)}
    del lg1
    l2, _lg2, g2 = _isolated(model, fr, go, ac, labels, loss_kw)
    del _lg2
    nib = _nib_tensor(cur["nib"], fr["hidden_states"].device)
    frozen = {"hidden_states": fr["hidden_states"], "weight": fr["weight"], "labels": labels,
              "num_items_in_batch": nib}
    rec = {"target": TARGET, "step": _STEP, "rank": _RANK, "call": k, "variant": tag or "live",
           "module": "model (Qwen2ForCausalLM) lm_head + loss_function",
           "args": [_snap(frozen[r]) for r in ROLES], "kwargs": {}, "const": {},
           "loss_kwargs_keys": sorted(loss_kw.keys()),
           "arg_roles": ROLES, "autocast": row.get("autocast"), "storage": row.get("weight_storage"),
           "output": [_snap(cur["live_loss"] if tag == "" else l1)],
           "grad_output": [_snap(go)]}
    rec["isolated"] = {"out": [_snap(l1)],
                       "grads": {str(i): _snap(g) for i, g in enumerate(g1)},
                       "grad_shapes": [list(g.shape) for g in g1]}
    h = fr["hidden_states"]
    rec["composition_evidence"] = {
        **live_eq,
        "iso_rerun_bitwise_loss": bool(torch.equal(l1, l2)),
        "iso_rerun_bitwise_grads": [bool(torch.equal(a, b)) for a, b in zip(g1, g2)],
        "loss": float(l1), "go": float(go), "go_dtype": str(go.dtype),
        "num_items_in_batch": int(nib), "labels_ignored": int((labels == IGNORE).sum()),
        "S": int(h.shape[-2]), "grad_dtypes": [str(g.dtype) for g in g1],
        "dgrad_zero_rows": int((g1[0].reshape(-1, g1[0].shape[-1]).abs().sum(-1) == 0).sum()),
        "wgrad_zero_rows": int((g1[1].abs().sum(-1) == 0).sum()),
        "hidden_abs_max": float(h.float().abs().max()),
        "grad_abs_mean": [float(g.float().abs().mean()) for g in g1],
        "grad_abs_max": [float(g.float().abs().max()) for g in g1],
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}_call{k}{tag}.pt")
    torch.save(rec, path + ".partial")
    os.replace(path + ".partial", path)
    _state["written"].append(path)
    print(f"[geak-hflmce] wrote {path} S={int(h.shape[-2])} evidence="
          f"{json.dumps({kk: rec['composition_evidence'][kk] for kk in list(live_eq) + ['iso_rerun_bitwise_loss', 'iso_rerun_bitwise_grads', 'loss', 'go', 'num_items_in_batch', 'labels_ignored', 'dgrad_zero_rows']})} "
          f"({time.time()-t0:.1f}s)", flush=True)


def _finish(cur, go):
    model = _state["model"]
    labels = cur["labels"].detach().clone()
    _record(cur, go, labels, "", model)
    torch.cuda.empty_cache()
    if cur["k"] == _MASKED:
        _record(cur, go, _masked_labels(labels), "_masked", model)
        torch.cuda.empty_cache()


def _write_capture_meta():
    per = _state["per_call"]
    meta = {"target": TARGET, "step": _STEP, "rank": _RANK, "calls_in_step": len(per),
            "frozen_calls": sorted(_CALLS), "masked_call": _MASKED, "written": _state["written"],
            "distinct_S": sorted({r["S"] for r in per if r.get("S") is not None}),
            "per_call": per}
    try:
        with open(os.path.join(_DIR, f"capture_meta_rank{_RANK}.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        print(f"[geak-hflmce] step {_STEP}: {len(per)} calls, S={meta['distinct_S']}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hflmce] meta write failed: {e!r}", flush=True)
