"""Capture FSDP's per-micro-batch FLAT-GRAD ACCUMULATION as it runs on the hf_trl stack.
Head `hf_fsdp_flatgrad_accum`.

WHAT THE OP IS. HF Trainer wraps the model in FSDP1 (accelerate, SHARD_GRAD_OP, use_orig_params=True,
bf16 param mixed precision; one FlatParameter per Qwen2DecoderLayer + one root handle holding
embed_tokens / norm / lm_head). In every forward FSDP exposes the original parameters as VIEWS of the
unsharded bf16 flat parameter:

    views = torch.split(flat_param, flat_param._numels_with_padding)   # FlatParamHandle._get_unflat_views_aligned
    module.<name> = views[i].view(shape_i)                              # FlatParamHandle._use_unsharded_views

so in backward, per handle and per MICRO-BATCH, autograd runs
    SplitWithSizesBackward0 :  new = torch.cat([g_0.view(-1), ..., g_{k-1}.view(-1)])   (zeros for None)
    AccumulateGrad(flat_param): flat_param.grad is None ? steal(new) : flat_param.grad.add_(new)
The cat is a full write of the layer's 233,057,792-element bf16 flat grad (root: 1,089,998,336) that
exists only to be read once more by the add_. Under no_sync (micro-batches 0..6 of 8) the add_
accumulates IN bf16 in flat_param.grad; on the sync micro-batch FSDP's post-backward hook (a post
hook on the same AccumulateGrad node) then reduce-scatters it.

WHAT THIS RECORDS, on EVERY rank, at one step (GEAK_HFFLAT_CAPTURE_STEP):
  * census: per handle and micro-batch -- number of split nodes reached from the views (must be 1),
    the incoming grads' None / stride / storage_offset / storage-sharing facts, the CUDA stream the
    node runs on, whether flat_param.grad existed (steal vs add_), whether the result is in place
    (data_ptr unchanged), and an IN-SITU BITWISE check of the live result against an independent
    reference computed in the node's pre-hook:  expected = bf16( fp32(prev) + fp32(cat(grads)) ).
    The check runs in a post-ACCUMULATE tensor hook, which fires BEFORE FSDP's node post hook, so
    the sync micro-batch is checked too (FSDP has not reduce-scattered yet).
  * frozen cases (GEAK_HFFLAT_CAPTURE_FREEZE = "rank:handle:mb,..."): the incoming grads and the
    prior flat grad copied to CPU, plus the sha256 of the live result -- the oracle's inputs.

Nothing in the live computation is altered: the pre-hook only reads, the reference is computed on
side tensors, and the hooks are removed after the step.

Env:
    GEAK_HFFLAT_CAPTURE_DIR=/work/tasks/<name>_cap      (required; enables the capture)
    GEAK_HFFLAT_CAPTURE_STEP=4                          (1-based step to capture, after warmup)
    GEAK_HFFLAT_CAPTURE_FREEZE=0:L14:0,0:L14:2,0:root:2 (rank:handle:micro-batch to freeze; handle
                                                         = L<layer index> or root)
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_HFFLAT_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_HFFLAT_CAPTURE_STEP", "0") or 0)
_RANK = int(os.environ.get("RANK", "0"))
_FREEZE = set()
for _tok in os.environ.get("GEAK_HFFLAT_CAPTURE_FREEZE", "").split(","):
    _p = _tok.strip().split(":")
    if len(_p) == 3 and _p[0].isdigit() and _p[2].isdigit():
        _FREEZE.add((int(_p[0]), _p[1], int(_p[2])))
ENABLED = bool(_DIR and _STEP > 0)
TARGET = ("torch.distributed.fsdp._flat_param:FlatParamHandle._get_unflat_views_aligned / "
          "_use_unsharded_views  [backward: SplitWithSizesBackward0 cat + AccumulateGrad add_]")

_state = {"step": 0, "mb": {}, "records": [], "written": False, "names": {}, "post_hooked": set(),
          "pending": {}, "t0": None}


def note_step(step):
    """Called at every step end with the number of steps COMPLETED."""
    _state["step"] = step
    if step >= _STEP and not _state["written"]:
        _state["written"] = True
        _write()


def _active():
    return _state["step"] == _STEP - 1


def _label(handle):
    fp = handle.flat_param
    names = [_state["names"].get(id(pi.module), "?") + "." + pi.param_name for pi in fp._param_infos]
    if any("embed_tokens" in n or "lm_head" in n for n in names):
        return "root", names
    for n in names:
        parts = n.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"L{parts[i + 1]}", names
    return f"h{id(handle) % 100000}", names


def _sha(t):
    h = hashlib.sha256()
    b = t.detach().contiguous().view(torch.int16).cpu().numpy().tobytes() if t.dtype == torch.bfloat16 \
        else t.detach().contiguous().cpu().numpy().tobytes()
    h.update(b)
    return h.hexdigest()


def _bitdiff(a, b):
    """Bitwise difference count of two bf16 tensors, NaN == NaN (any payload)."""
    na, nb = torch.isnan(a), torch.isnan(b)
    nan_mismatch = int((na != nb).sum())
    m = ~(na | nb)
    d = int((a.view(torch.int16)[m] != b.view(torch.int16)[m]).sum())
    return nan_mismatch + d


def _grad_facts(gs):
    out, by_storage = [], {}
    for i, g in enumerate(gs):
        if g is None:
            out.append(None)
            continue
        sp = g.untyped_storage().data_ptr()
        by_storage.setdefault(sp, []).append(i)
        out.append({"numel": g.numel(), "shape": list(g.shape), "stride": list(g.stride()),
                    "storage_offset": int(g.storage_offset()), "contig": g.is_contiguous(),
                    "dtype": str(g.dtype), "ptr_mod_16": g.data_ptr() % 16,
                    "storage_nbytes": g.untyped_storage().nbytes()})
    shared = [v for v in by_storage.values() if len(v) > 1]
    return out, shared


def _find_split(handle):
    """The ONE SplitWithSizesBackward0 node all this forward's views of flat_param hang off."""
    nodes, chains = set(), []
    node = None
    for t in handle.flat_param._tensors or []:
        if t is None or t.grad_fn is None:
            continue
        n, chain = t.grad_fn, []
        while n is not None and "SplitWithSizes" not in type(n).__name__ and len(chain) < 4:
            chain.append(type(n).__name__)
            n = n.next_functions[0][0] if n.next_functions else None
        if n is not None:
            nodes.add(id(n))
            node = n
        chains.append(chain)
    return node, len(nodes), sorted({"/".join(c) for c in chains})


def _install_post(handle, label):
    fp = handle.flat_param
    if id(fp) in _state["post_hooked"]:
        return
    _state["post_hooked"].add(id(fp))

    def post(p):
        pend = _state["pending"].pop(id(fp), None)
        if pend is None or not _active():
            return
        rec, exp, prev_ptr, frz = pend
        res = p.grad
        torch.cuda.current_stream().synchronize()
        rec["result_dtype"] = str(res.dtype) if res is not None else None
        rec["result_numel"] = res.numel() if res is not None else None
        rec["in_place"] = (prev_ptr is not None and res is not None and res.data_ptr() == prev_ptr)
        rec["post_stream"] = torch.cuda.current_stream().stream_id
        if res is not None and exp is not None and res.dtype == exp.dtype and res.numel() == exp.numel():
            rec["bitwise_ndiff_vs_ref"] = _bitdiff(res.reshape(-1), exp)
        else:
            rec["bitwise_ndiff_vs_ref"] = None
        if frz is not None:
            frz["live_result_sha256"] = _sha(res)
            frz["live_in_place"] = rec["in_place"]
            frz["live_bitwise_ndiff_vs_ref"] = rec["bitwise_ndiff_vs_ref"]
            frz["census"] = dict(rec)
            path = os.path.join(_DIR, f"case_rank{_RANK}_{rec['handle']}_mb{rec['mb']}.pt")
            t = time.time()
            torch.save(frz, path)
            rec["frozen_path"] = path
            rec["save_s"] = round(time.time() - t, 1)
            print(f"[geak-hfflat] froze {path} ({rec['save_s']} s)", flush=True)

    fp.register_post_accumulate_grad_hook(post)


def _wrap(orig):
    def _use_unsharded_views(self, as_params):
        orig(self, as_params)
        try:
            from torch.distributed.fsdp._flat_param import HandleTrainingState
            if (not _active() or as_params or not torch.is_grad_enabled()
                    or self._training_state != HandleTrainingState.FORWARD):
                return
            label, names = _label(self)
            mb = _state["mb"].get(label, 0)
            _state["mb"][label] = mb + 1
            node, n_nodes, chains = _find_split(self)
            fp = self.flat_param
            rec = {"rank": _RANK, "step": _STEP, "handle": label, "mb": mb,
                   "flat_numel": fp.numel(), "flat_dtype": str(fp.dtype),
                   "numels_with_padding": list(fp._numels_with_padding),
                   "is_padding_mask": list(fp._is_padding_mask),
                   "n_split_nodes": n_nodes, "view_chains": chains,
                   "split_node": type(node).__name__ if node is not None else None}
            if node is None:
                _state["records"].append(rec)
                return
            acc = node.next_functions[0][0]
            rec["next_node"] = type(acc).__name__
            rec["acc_is_flat_param"] = getattr(acc, "variable", None) is fp
            _install_post(self, label)
            frz_key = (_RANK, label, mb)

            def pre(grad_outputs):
                try:
                    gs = list(grad_outputs)
                    rec["n_grads"] = len(gs)
                    rec["n_none"] = sum(g is None for g in gs)
                    rec["grads"], rec["shared_storage_groups"] = _grad_facts(gs)
                    rec["node_stream"] = torch.cuda.current_stream().stream_id
                    rec["default_stream"] = torch.cuda.default_stream().stream_id
                    rec["grad_mode"] = torch.is_grad_enabled()
                    prev = fp.grad
                    rec["prev_is_none"] = prev is None
                    rec["prev_dtype"] = str(prev.dtype) if prev is not None else None
                    rec["prev_numel"] = prev.numel() if prev is not None else None
                    with torch.no_grad():
                        parts = [g.reshape(-1) if g is not None else
                                 torch.zeros(n, dtype=fp.dtype, device=fp.device)
                                 for g, n in zip(gs, fp._numels_with_padding)]
                        cat = torch.cat(parts)
                        exp = cat.clone() if prev is None else \
                            (prev.reshape(-1).float() + cat.float()).to(prev.dtype)
                        del cat
                    frz = None
                    if frz_key in _FREEZE:
                        frz = {"target": TARGET, "rank": _RANK, "step": _STEP, "handle": label,
                               "mb": mb, "fqns": names, "numels_with_padding": list(fp._numels_with_padding),
                               "is_padding_mask": list(fp._is_padding_mask),
                               "shapes": [list(s) for s in fp._shapes],
                               "grads": [g.detach().to("cpu", copy=True) if g is not None else None
                                         for g in gs],
                               "grad_facts": rec["grads"],
                               "prev": prev.detach().reshape(-1).to("cpu", copy=True) if prev is not None else None,
                               "torch": str(torch.__version__)}
                    _state["pending"][id(fp)] = (rec, exp, prev.data_ptr() if prev is not None else None, frz)
                except Exception as e:  # noqa: BLE001
                    rec["pre_err"] = repr(e)[:300]
                return None

            node.register_prehook(pre)
            _state["records"].append(rec)
        except Exception as e:  # noqa: BLE001
            print(f"[geak-hfflat] wrapper error (live path unaffected): {e!r}", flush=True)
    _use_unsharded_views.__wrapped__ = orig
    return _use_unsharded_views


def install(model=None):
    if not ENABLED:
        return False
    import torch.distributed.fsdp._flat_param as fpm
    os.makedirs(_DIR, exist_ok=True)
    if model is not None:
        _state["names"] = {id(m): n for n, m in model.named_modules()}
    cls = fpm.FlatParamHandle
    cls._use_unsharded_views = _wrap(cls._use_unsharded_views)
    print(f"[geak-hfflat] capture installed: step {_STEP}, rank {_RANK}, freeze "
          f"{sorted(k for k in _FREEZE if k[0] == _RANK)}", flush=True)
    return True


def _write():
    recs = _state["records"]
    summ = {}
    for r in recs:
        s = summ.setdefault(r["handle"], {"mbs": 0, "split_nodes_ok": 0, "steal": 0, "add": 0,
                                          "in_place_adds": 0, "bitwise_ok": 0, "bitwise_checked": 0,
                                          "streams": set(), "none_grads": 0, "shared_groups": 0})
        s["mbs"] += 1
        s["split_nodes_ok"] += int(r.get("n_split_nodes") == 1)
        if r.get("prev_is_none") is True:
            s["steal"] += 1
        elif r.get("prev_is_none") is False:
            s["add"] += 1
            s["in_place_adds"] += int(bool(r.get("in_place")))
        if r.get("bitwise_ndiff_vs_ref") is not None:
            s["bitwise_checked"] += 1
            s["bitwise_ok"] += int(r["bitwise_ndiff_vs_ref"] == 0)
        if "node_stream" in r:
            s["streams"].add(r["node_stream"])
        s["none_grads"] += r.get("n_none", 0) or 0
        s["shared_groups"] += len(r.get("shared_storage_groups") or [])
    for s in summ.values():
        s["streams"] = sorted(s["streams"])
    out = {"target": TARGET, "rank": _RANK, "step": _STEP, "summary": summ, "records": recs}
    with open(os.path.join(_DIR, f"census_rank{_RANK}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[geak-hfflat] census written: {len(recs)} records, "
          f"handles {len(summ)}", flush=True)
