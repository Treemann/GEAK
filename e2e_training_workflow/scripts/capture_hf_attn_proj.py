"""Capture HF `Qwen2Attention`'s four projections (fwd + bwd) AS THEY RUN on the hf_trl stack.
Head `hf_attn_qkvo_proj`.

WHAT THE OP IS. One layer's q/k/v/o projections, frozen as ONE op (layout `attn_qkvo_proj`):

    q = x @ Wq^T + bq    x [1,M,3584] bf16 (input_layernorm output), Wq [3584,3584], bq [3584]
    k = x @ Wk^T + bk    Wk [512,3584], bk [512]
    v = x @ Wv^T + bv    Wv [512,3584], bv [512]
    o = a @ Wo^T         a [1,M,3584] bf16 (the attention output, reshaped), Wo [3584,3584], no bias

Gradients: dx = dq@Wq + dk@Wk + dv@Wv (x feeds all three, so the op's dgrad is the SUM -- exactly what
one fused [4608,3584] QKV dgrad GEMM produces), dWq/dWk/dWv, dbq/dbk/dbv, da = do@Wo, dWo.
28 layers x 8 microbatches = 224 calls of each projection per rank per step.

WHY ONE OP AND NOT FOUR TASKS. q/k/v share their input, so a fused QKV GEMM (one [M,3584]x[3584,4608]
forward with a fused bias, one dgrad, one wgrad, one dbias reduction) is the natural lever and only a
composite oracle can score it: per-projection oracles would hand it three separate dx's it never
forms. o_proj is in the same module and shares the [3584,3584] shape class with q, so the head's
fwd+bwd objective is the four together. o does not interact with q/k/v numerically (attention sits
between them); it is scored on its own outputs.

WHY MODULE HOOKS ON THE 4x28 Linear INSTANCES. On the accepted hf4 stack Qwen2Attention.forward is NOT
rebound (attn_fmha_ck_hf rebinds torch.nn.functional.scaled_dot_product_attention; rmsnorm and the
two MLP overlays rebind Qwen2RMSNorm / Qwen2MLP). The projections are reached as the module calls
self.q_proj(hidden_states) ... self.o_proj(attn_output). Forward pre/post hooks on those instances see
exactly the live calls on top of the stack; the live forward is not altered.

grad_output IS SOUND HERE. The tensor hooks are on the four projections' OUTPUTS. q/k/v outputs are
consumed only by `.view(hidden_shape).transpose(1, 2)`; o's only by the decoder layer's residual add.
A tensor hook on an output reports d loss / d out, this op's grad_output by definition. (The
consumer-sum hazard concerns INPUTS with several consumers -- x here -- and no gradient is read from a
hook on x: dx comes from the isolated re-run.)

THE GOLDEN COMES FROM AN ISOLATED RE-RUN of the original (nn.Linear.forward of THESE four modules,
called unbound so the capture hooks do not re-fire) on clones of x, a and all seven weight/bias
tensors taken at FORWARD time, in the live process, inside the LAST of the four grad_output hooks.
Each weight/bias is swapped for a cloned leaf for that one call (identity-verified) so the re-run yields
this call's OWN dW / db (the live ones are summed over 8 microbatches into FSDP's flat grad). Autocast
state is recorded and re-entered (the hook runs on the autograd thread, where autocast is off).
Recorded per frozen call: isolated outputs == live outputs (bitwise), a second isolated re-run == the
first (bitwise, every output), that q/k/v received the SAME input tensor object, and the live
weight/bias storage facts.

Env:
    GEAK_HFATTN_CAPTURE_DIR=/work/tasks/<name>_cap   (required; enables the capture)
    GEAK_HFATTN_CAPTURE_STEP=12                      (1-based step to capture, after warmup)
    GEAK_HFATTN_CAPTURE_CALLS=0,14,27,210            (per-step attention call indices to freeze;
                                                      call k = layer k % 28 of microbatch k // 28)
    GEAK_HFATTN_CAPTURE_RANKS=0,4                    (one file per rank and call; no shared writer)
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFATTN_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFATTN_CAPTURE_STEP", "0") or 0)
_CALLS = {int(c) for c in os.environ.get("GEAK_HFATTN_CAPTURE_CALLS", "0,14,27,210").split(",")
          if c.strip().isdigit()}
_RANKS = {int(r) for r in os.environ.get("GEAK_HFATTN_CAPTURE_RANKS", "0").split(",")
          if r.strip().isdigit()}
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0 and _RANK in _RANKS)
TARGET = "transformers.models.qwen2.modeling_qwen2:Qwen2Attention.{q_proj,k_proj,v_proj,o_proj}"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
# Record argument order == layout `attn_qkvo_proj` args (l0_gate.LAYOUTS).
ROLES = ["x", "weight_q", "bias_q", "weight_k", "bias_k", "weight_v", "bias_v", "x_o", "weight_o"]

_state = {"step": 0, "calls": 0, "attn_of": {}, "names": {}, "per_call": [], "written": [],
          "meta_done": False, "cur": {}, "pending": {}}


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


def _isolated(att, frozen, gos, ac):
    """The ORIGINAL four module calls on clones of the frozen tensors. Returns (outs, grads in
    ROLES order)."""
    leaves = {r: frozen[r].detach().clone().requires_grad_(True) for r in ROLES}
    toks = []
    try:
        for p, wr, br in (("q_proj", "weight_q", "bias_q"), ("k_proj", "weight_k", "bias_k"),
                          ("v_proj", "weight_v", "bias_v"), ("o_proj", "weight_o", None)):
            lin = getattr(att, p)
            toks.append((lin, _swap(lin, "weight", leaves[wr])))
            if br is not None:
                toks.append((lin, _swap(lin, "bias", leaves[br])))
        with torch.enable_grad(), torch.autocast("cuda", dtype=ac["dtype"], enabled=ac["enabled"]):
            lf = torch.nn.Linear.forward          # unbound -> the capture's module hooks do not fire
            q = lf(att.q_proj, leaves["x"])
            k = lf(att.k_proj, leaves["x"])
            v = lf(att.v_proj, leaves["x"])
            o = lf(att.o_proj, leaves["x_o"])
        grads = torch.autograd.grad((q, k, v, o), [leaves[r] for r in ROLES], grad_outputs=gos)
    finally:
        for lin, tok in reversed(toks):
            _restore(lin, tok)
    return [t.detach() for t in (q, k, v, o)], [g.detach() for g in grads]


def install(model=None):
    if not ENABLED:
        return False
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfattn] cannot import Qwen2Attention: {e!r}", flush=True)
        return False
    if model is None:
        print("[geak-hfattn] no model handed over; cannot hook the projection instances", flush=True)
        return False
    os.makedirs(_DIR, exist_ok=True)
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, mq.Qwen2Attention):
            _state["names"][id(m)] = name
            for p in PROJ:
                lin = getattr(m, p)
                _state["attn_of"][id(lin)] = (m, p)
                lin.register_forward_pre_hook(_pre)
                lin.register_forward_hook(_post)
            n += 1
    fw = mq.Qwen2Attention.forward
    print(f"[geak-hfattn] armed on {n} Qwen2Attention modules (4 projections each) for step {_STEP}, "
          f"calls {sorted(_CALLS)}, rank {_RANK}; Qwen2Attention.forward = "
          f"{getattr(fw, '__module__', '?')}.{getattr(fw, '__qualname__', '?')}", flush=True)
    return n > 0


def _pre(mod, inp):
    if _state["step"] + 1 != _STEP:
        return None
    att, p = _state["attn_of"][id(mod)]
    x = inp[0]
    ac = {"enabled": torch.is_autocast_enabled("cuda"), "dtype": torch.get_autocast_dtype("cuda")}
    if p == "q_proj":
        k = _state["calls"]
        _state["calls"] += 1
        row = {"call": k, "module": _state["names"].get(id(att), "?"), "M": int(x.numel() // x.shape[-1]),
               "x_shape": list(x.shape), "x_dtype": str(x.dtype), "x_contig": x.is_contiguous(),
               "x_requires_grad": bool(x.requires_grad),
               "autocast": {"enabled": ac["enabled"], "dtype": str(ac["dtype"])},
               "grad_enabled": torch.is_grad_enabled(), "storage": {}, "out_dtype": {},
               "qkv_same_input": True, "x_o_contig": None}
        _state["per_call"].append(row)
        _state["cur"][id(att)] = {"k": k, "row": row, "x": x}
        if k in _CALLS:
            _state["pending"][id(att)] = {"k": k, "row": row, "ac": ac, "frozen": {"x": x.detach().clone()},
                                          "live": {}, "gos": {}, "done": False}
    cur = _state["cur"].get(id(att))
    if cur is None:
        return None
    row = cur["row"]
    row["storage"][p] = {"weight": _storage_facts(mod.weight),
                         "bias": _storage_facts(mod.bias) if mod.bias is not None else None}
    if p in ("k_proj", "v_proj") and x is not cur["x"]:
        row["qkv_same_input"] = False
    if p == "o_proj":
        row["x_o_contig"] = x.is_contiguous()
        row["x_o_shape"] = list(x.shape)
    pend = _state["pending"].get(id(att))
    if pend is not None and pend["k"] == cur["k"]:
        sfx = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}[p]
        pend["frozen"][f"weight_{sfx}"] = mod.weight.detach().clone()
        if sfx != "o":
            pend["frozen"][f"bias_{sfx}"] = mod.bias.detach().clone()
        else:
            pend["frozen"]["x_o"] = x.detach().clone()
    return None


def _post(mod, inp, out):
    if _state["step"] + 1 != _STEP:
        return None
    att, p = _state["attn_of"][id(mod)]
    cur = _state["cur"].get(id(att))
    if cur is not None:
        cur["row"]["out_dtype"][p] = str(out.dtype)
    pend = _state["pending"].get(id(att))
    if pend is None or cur is None or pend["k"] != cur["k"]:
        return None
    if not out.requires_grad:
        print(f"[geak-hfattn] call {pend['k']} {p}: output not differentiable, not frozen", flush=True)
        _state["pending"].pop(id(att), None)
        return None
    pend["live"][p] = out.detach().clone()

    def _hook(g, p=p):
        try:
            pend["gos"][p] = g.detach().clone()
            if len(pend["gos"]) == 4 and not pend["done"]:
                pend["done"] = True
                _state["pending"].pop(id(att), None)
                _finish(att, pend)
        except Exception as e:  # noqa: BLE001  -- a capture must never kill the run
            print(f"[geak-hfattn] capture of call {pend['k']} FAILED (run continues): {e!r}", flush=True)
        return g

    out.register_hook(_hook)
    return None


def _finish(att, pend):
    t0 = time.time()
    k, row, ac, fr = pend["k"], pend["row"], pend["ac"], pend["frozen"]
    missing = [r for r in ROLES if r not in fr]
    if missing:
        raise RuntimeError(f"frozen roles missing: {missing}")
    gos = [pend["gos"][p] for p in PROJ]
    live = [pend["live"][p] for p in PROJ]
    o1, g1 = _isolated(att, fr, gos, ac)
    o2, g2 = _isolated(att, fr, gos, ac)
    rec = {"target": TARGET, "step": _STEP, "rank": _RANK, "call": k, "module": row["module"],
           "args": [_snap(fr[r]) for r in ROLES], "kwargs": {}, "const": {},
           "arg_roles": ROLES, "autocast": row["autocast"], "storage": row["storage"],
           "output": [_snap(t) for t in live],
           "grad_output": [_snap(g) for g in gos]}
    # layout `attn_qkvo_proj`: grads keyed by position in grad_of (== ROLES).
    rec["isolated"] = {"out": [_snap(t) for t in o1],
                       "grads": {str(i): _snap(g) for i, g in enumerate(g1)},
                       "grad_shapes": [list(g.shape) for g in g1]}
    x0 = fr["x"]
    rec["composition_evidence"] = {
        "iso_out_equals_live_out": [bool(torch.equal(a, b)) for a, b in zip(o1, live)],
        "iso_out_vs_live_max_abs": [float((a.float() - b.float()).abs().max()) for a, b in zip(o1, live)],
        "iso_rerun_bitwise_out": [bool(torch.equal(a, b)) for a, b in zip(o1, o2)],
        "iso_rerun_bitwise_grads": [bool(torch.equal(a, b)) for a, b in zip(g1, g2)],
        "qkv_same_input_object": row["qkv_same_input"],
        "out_dtypes": [str(t.dtype) for t in o1], "grad_dtypes": [str(g.dtype) for g in g1],
        "go_dtypes": [str(g.dtype) for g in gos],
        "go_abs_mean": [float(g.float().abs().mean()) for g in gos],
        "go_abs_max": [float(g.float().abs().max()) for g in gos],
        "go_zero_rows": [int((g.reshape(-1, g.shape[-1]).abs().sum(-1) == 0).sum()) for g in gos],
        "x_zero_rows": int((x0.reshape(-1, x0.shape[-1]).abs().sum(-1) == 0).sum()),
        "x_abs_max": float(x0.float().abs().max()),
        "x_o_abs_max": float(fr["x_o"].float().abs().max()),
        "out_abs_max": [float(t.float().abs().max()) for t in live],
        "grad_abs_mean": {r: float(g.float().abs().mean()) for r, g in zip(ROLES, g1)},
        "grad_abs_max": {r: float(g.float().abs().max()) for r, g in zip(ROLES, g1)},
    }
    path = os.path.join(_DIR, f"reference_io_rank{_RANK}_call{k}.pt")
    torch.save(rec, path + ".partial")
    os.replace(path + ".partial", path)
    _state["written"].append(path)
    print(f"[geak-hfattn] wrote {path} module={row['module']} M={row['M']} "
          f"evidence={json.dumps({kk: rec['composition_evidence'][kk] for kk in ('iso_out_equals_live_out', 'iso_rerun_bitwise_out', 'iso_rerun_bitwise_grads', 'qkv_same_input_object', 'go_zero_rows')})} "
          f"({time.time()-t0:.1f}s)", flush=True)


def _write_capture_meta():
    per = _state["per_call"]
    Ms = sorted({r["M"] for r in per})
    meta = {"target": TARGET, "step": _STEP, "rank": _RANK, "calls_in_step": len(per),
            "frozen_calls": sorted(_CALLS), "written": _state["written"], "distinct_M": Ms,
            "x_dtypes": sorted({r["x_dtype"] for r in per}),
            "out_dtypes": sorted({json.dumps(r["out_dtype"], sort_keys=True) for r in per}),
            "autocast": sorted({json.dumps(r["autocast"]) for r in per}),
            "qkv_same_input_all": all(r["qkv_same_input"] for r in per),
            "x_contig_all": all(r["x_contig"] for r in per),
            "x_o_contig_all": all(bool(r["x_o_contig"]) for r in per),
            "storage_kinds": sorted({json.dumps({p: {w: ({kk: vv for kk, vv in s.items() if kk != "offset_elems"}
                                                         if s else None) for w, s in d.items()}
                                                 for p, d in r["storage"].items()}, sort_keys=True)
                                     for r in per}),
            "per_call": per}
    try:
        with open(os.path.join(_DIR, f"capture_meta_rank{_RANK}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[geak-hfattn] step {_STEP}: {len(per)} attention calls, M={Ms}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfattn] meta write failed: {e!r}", flush=True)
