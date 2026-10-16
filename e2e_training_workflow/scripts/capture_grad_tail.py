"""Capture the dist-opt GRAD TAIL (pre-reduce-scatter norm check + grad scale) AS IT RUNS ON THE STACK.

WHAT THE OP IS. With --use-distributed-optimizer, overlap_grad_reduce off and no bucket size, the
whole model's fp32 main-grad buffer is ONE bucket, and once per optimizer step
`megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync` runs:

    check_grads():  grad_norm = bucket.grad_data.norm(p=2)       # aten::linalg_vector_norm
                    validate_result(grad_norm, torch.isnan) / (torch.isinf)   (host syncs)
    scale:          bucket.grad_data *= bucket.gradient_scaling_factor        # aten::mul_
    reduce_scatter(local_shard_view, bucket.grad_data, op=SUM)              # ncclDevKernel_Generic

The profile's `reduce_kernel<512,1> linalg_vector_norm [7612864000]` and
`vectorized_elementwise_kernel mul_ [7612864000]` are the first two. There is no autograd here: the
op's INPUT is the backward's product (the accumulated main grads), so there is no grad_output to
capture -- the frozen input is the real grad buffer itself.

WHAT IS CAPTURED (capture step, before the original start_grad_sync runs):
  * every rank: buffer facts (numel, dtype, contiguity, storage offset, n buckets/groups, the scaling
    factor, ddp_config flags, reduce op), full-buffer statistics computed in fp64 (sum of squares,
    zeros, non-finite count, the count of values the scale would push into the fp32 subnormal range
    -- the ONLY place where 'scale before the sum' and 'scale after the sum' can differ for a
    power-of-two factor), and Megatron's own torch norm value;
  * rank 0 only: K real contiguous chunks of the unscaled buffer at evenly spaced offsets (the oracle
    inputs) -- the 30 GB buffer itself cannot be frozen.

WHAT IS MEASURED IN SITU (all ranks; collectives need every rank): the head's 'fold the scale into
the reduce-scatter' lever. The unscaled buffer is cloned, the original start_grad_sync runs, and its
reduced local shard R is then compared BITWISE against
    (a) RS(SUM) of the clone, then * factor          -- 'scale after the sum'
    (b) RS(AVG) of the clone                           -- --ddp-average-in-collective
    (c) RS(PREMUL_SUM(factor)) of the clone            -- torch.distributed._make_nccl_premul_sum
with each RS timed by CUDA events, so the record says whether the fold is bit-exact on this RCCL AND
what it costs, not just whether it should be.

Env:
    GEAK_GRADTAIL_CAPTURE_DIR=/work/tasks/<name>_cap   (required; enables the capture)
    GEAK_GRADTAIL_CAPTURE_STEP=13                      (1-based step whose grad sync is captured)
    GEAK_GRADTAIL_CHUNKS=16  GEAK_GRADTAIL_CHUNK_NUMEL=2097152
    GEAK_GRADTAIL_INSITU=1                             (0 disables the RS-fold comparison)
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

_DIR = os.environ.get("GEAK_GRADTAIL_CAPTURE_DIR", "")
_STEP = int(os.environ.get("GEAK_GRADTAIL_CAPTURE_STEP", "0") or 0)
_K = int(os.environ.get("GEAK_GRADTAIL_CHUNKS", "16") or 16)
_CN = int(os.environ.get("GEAK_GRADTAIL_CHUNK_NUMEL", str(2 * 1024 * 1024)) or 2097152)
_INSITU = os.environ.get("GEAK_GRADTAIL_INSITU", "1") != "0"
_RANK = int(os.environ.get("RANK", "0"))
ENABLED = bool(_DIR and _STEP > 0)

_state = {"step": 0, "done": False, "calls": {}, "scale_calls": {}, "zero_calls": {}}


def note_step(step):
    """Called after the probe's step_end with the 1-based number of the step just finished."""
    _state["step"] = step
    if step == _STEP + 1 and not _state.get("flushed"):
        _state["flushed"] = True
        _write_json(f"calls_rank{_RANK}.json", {
            "rank": _RANK, "start_grad_sync_calls_per_step": _state["calls"],
            "ddp_scale_gradients_calls_per_step": _state["scale_calls"],
            "zero_grad_buffer_calls_per_step": _state["zero_calls"]})


def _write_json(name, obj):
    os.makedirs(_DIR, exist_ok=True)
    p = os.path.join(_DIR, name)
    with open(p + ".partial", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(p + ".partial", p)


def _cur_step():
    return _state["step"] + 1          # the step currently running (1-based)


def _stats(g, factor):
    """Full-buffer statistics, chunked so no fp64 copy of 30 GB is ever made."""
    n = g.numel()
    step = 1 << 28
    ss = 0.0; zeros = 0; nonfinite = 0; tiny = 0; amax = 0.0; amin_nz = math.inf; sub_in = 0
    # |x * f| < 2^-126  <=>  |x| < 2^-126 / f : the product is subnormal (or underflows), the only
    # range where scaling before vs after an fp32 sum can round differently for f = 2^-k.
    thr = (2.0 ** -126) / factor if factor not in (0.0,) else 0.0
    for s in range(0, n, step):
        c = g[s:s + step]
        cd = c.double()
        ss += float((cd * cd).sum())
        a = c.abs()
        zeros += int((c == 0).sum())
        nonfinite += int((~torch.isfinite(c)).sum())
        tiny += int(((a < thr) & (a > 0)).sum())
        sub_in += int(((a < 2.0 ** -126) & (a > 0)).sum())
        amax = max(amax, float(a.max()))
        nz = a[a > 0]
        if nz.numel():
            amin_nz = min(amin_nz, float(nz.min()))
        del cd, a, nz
    return {"numel": n, "sumsq_fp64": ss, "norm_fp64": math.sqrt(ss), "zeros": zeros,
            "zero_frac": zeros / n, "nonfinite": nonfinite, "abs_max": amax,
            "abs_min_nonzero": amin_nz, "subnormal_inputs": sub_in,
            "would_be_subnormal_after_scale": tiny, "subnormal_threshold_in": thr}


def _ev_time(fn):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); out = fn(); e.record(); torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def _cmp(a, b):
    d = (a != b)
    nd = int(d.sum())
    rec = {"bitwise_equal": nd == 0, "n_diff": nd}
    if nd:
        rec["max_abs_diff"] = float((a.double() - b.double()).abs().max())
        rec["max_rel_diff"] = float(((a.double() - b.double()).abs()
                                     / b.double().abs().clamp_min(1e-30)).max())
    return rec


def install():
    if not ENABLED:
        return False
    import megatron.core.distributed.param_and_grad_buffer as pgb
    import megatron.core.distributed.distributed_data_parallel as ddpm
    os.makedirs(_DIR, exist_ok=True)

    BG = pgb._ParamAndGradBucketGroup
    orig_sgs = BG.start_grad_sync
    orig_scale = ddpm.DistributedDataParallel.scale_gradients
    orig_zero = ddpm.DistributedDataParallel.zero_grad_buffer

    def scale_gradients(self, scaling_factor):
        k = str(_cur_step()); _state["scale_calls"][k] = _state["scale_calls"].get(k, 0) + 1
        return orig_scale(self, scaling_factor)

    def zero_grad_buffer(self, *a, **kw):
        k = str(_cur_step()); _state["zero_calls"][k] = _state["zero_calls"].get(k, 0) + 1
        return orig_zero(self, *a, **kw)

    def start_grad_sync(self, force_all_reduce=False):
        k = str(_cur_step()); _state["calls"][k] = _state["calls"].get(k, 0) + 1
        if _state["done"] or _cur_step() != _STEP:
            return orig_sgs(self, force_all_reduce)
        _state["done"] = True
        try:
            return _capture(self, force_all_reduce)
        except Exception as e:   # a capture failure must never kill an 8-rank job
            import traceback
            print(f"[geak-gradtail] capture FAILED rank{_RANK}: {e!r}\n{traceback.format_exc()}",
                  flush=True)
            _write_json(f"error_rank{_RANK}.json", {"error": repr(e), "tb": traceback.format_exc()})
            raise

    def _capture(self, force_all_reduce):
        import torch.distributed as dist
        cfg = self.ddp_config
        b0 = self.buckets[0]
        g = b0.grad_data
        factor = float(b0.gradient_scaling_factor)
        rec = {"rank": _RANK, "step": _STEP, "n_buckets": len(self.buckets),
               "force_all_reduce": bool(force_all_reduce),
               "buckets": [{"numel": bk.grad_data.numel(), "dtype": str(bk.grad_data.dtype),
                            "contiguous": bk.grad_data.is_contiguous(),
                            "storage_offset": bk.grad_data.storage_offset(),
                            "storage_numel": bk.grad_data.untyped_storage().nbytes()
                            // bk.grad_data.element_size(),
                            "data_ptr_mod_256": bk.grad_data.data_ptr() % 256,
                            "offset": getattr(bk, "offset", None),
                            "numel_unpadded": getattr(bk, "numel_unpadded", None),
                            "gradient_scaling_factor": float(bk.gradient_scaling_factor),
                            "n_params": len(bk.params)} for bk in self.buckets],
               "gradient_scaling_factor": factor,
               "factor_is_power_of_two": factor > 0 and math.frexp(factor)[0] == 0.5,
               "ddp_config": {k2: getattr(cfg, k2, None) for k2 in (
                   "grad_reduce_in_fp32", "overlap_grad_reduce", "overlap_param_gather",
                   "use_distributed_optimizer", "check_for_nan_in_grad", "check_for_large_grads",
                   "bucket_size", "average_in_collective", "num_distributed_optimizer_instances",
                   "reduce_scatter_with_fp32_accumulation")},
               "dp_size": self.intra_distributed_optimizer_instance_size,
               "dp_rank": self.intra_distributed_optimizer_instance_rank,
               "reduce_op_used": "AVG" if cfg.average_in_collective else "SUM"}
        try:
            from megatron.core.rerun_state_machine import get_rerun_state_machine
            rec["rerun_mode"] = str(get_rerun_state_machine().mode)
        except Exception as e:
            rec["rerun_mode"] = repr(e)
        torch.cuda.synchronize()
        rec["stats_unscaled"] = _stats(g, factor)
        nv, t_norm = _ev_time(lambda: g.norm(p=2))
        rec["megatron_norm_fp32"] = float(nv)
        rec["megatron_norm_rel_err_vs_fp64"] = (abs(float(nv) - rec["stats_unscaled"]["norm_fp64"])
                                                / max(rec["stats_unscaled"]["norm_fp64"], 1e-30))
        rec["t_norm_ms_isolated"] = t_norm

        # Real chunks for the oracle (rank 0): evenly spaced, chunk-aligned to 256 elements.
        if _RANK == 0:
            n = g.numel()
            span = max(n - _CN, 1)
            offs = sorted({(int(span * i / max(_K - 1, 1)) // 256) * 256 for i in range(_K)})
            chunks = torch.stack([g[o:o + _CN].detach().to("cpu", copy=True) for o in offs])
            torch.save({"offsets": offs, "chunk_numel": _CN, "chunks": chunks,
                        "buffer_numel": n, "dtype": str(g.dtype), "factor": factor,
                        "megatron_norm_fp32": float(nv),
                        "stats_unscaled": rec["stats_unscaled"]},
                       os.path.join(_DIR, "grad_chunks_rank0.pt"))
            rec["chunk_offsets"] = offs

        g0 = None
        if _INSITU and len(self.buckets) == 1 and cfg.use_distributed_optimizer \
                and not force_all_reduce:
            free, total = torch.cuda.mem_get_info()
            need = g.numel() * g.element_size() * 1.2
            rec["insitu_mem"] = {"free_gb": free / 2**30, "need_gb": need / 2**30}
            if free > need:
                g0 = g.clone()
            else:
                rec["insitu_skipped"] = "not enough free memory for a clone of the buffer"

        # The ORIGINAL op, timed as a whole (norm + scale + RS).
        torch.cuda.synchronize()
        _, t_orig = _ev_time(lambda: orig_sgs(self, force_all_reduce))
        rec["t_original_start_grad_sync_ms"] = t_orig
        dp, r = self.intra_distributed_optimizer_instance_size, \
            self.intra_distributed_optimizer_instance_rank
        shard = g.numel() // dp
        R = g[r * shard:(r + 1) * shard]

        if g0 is not None:
            # The non-local shards hold the scaled-but-unreduced data: the scale itself, bitwise.
            nd = 0; mx = 0.0; step_ = 1 << 27
            for lo, hi in ((0, r * shard), ((r + 1) * shard, g.numel())):
                for s0 in range(lo, hi, step_):
                    s1 = min(s0 + step_, hi)
                    a, bb = g[s0:s1], g0[s0:s1] * factor
                    d = a != bb
                    k_ = int(d.sum())
                    if k_:
                        nd += k_
                        mx = max(mx, float((a.double() - bb.double()).abs().max()))
                    del a, bb, d
            ins = {"scale_vs_ieee_mul_nonlocal": {"bitwise_equal": nd == 0, "n_diff": nd,
                                                  "max_abs_diff": mx,
                                                  "numel_checked": g.numel() - shard}}
            grp = self.intra_distributed_optimizer_instance_group
            # time the scale alone on the clone's copy, for the record
            tmp = g0[r * shard:(r + 1) * shard].clone()
            _, ins["t_mul_local_shard_ms"] = _ev_time(lambda: tmp.mul_(factor))
            del tmp
            outs = {}
            o = torch.empty(shard, dtype=g.dtype, device=g.device)
            _, outs["t_rs_sum_ms"] = _ev_time(
                lambda: dist.reduce_scatter_tensor(o, g0, op=dist.ReduceOp.SUM, group=grp))
            o.mul_(factor)
            ins["sum_then_scale"] = _cmp(o, R)
            try:
                _, outs["t_rs_avg_ms"] = _ev_time(
                    lambda: dist.reduce_scatter_tensor(o, g0, op=dist.ReduceOp.AVG, group=grp))
                ins["rs_avg"] = _cmp(o, R)
            except Exception as e:
                ins["rs_avg"] = {"error": repr(e)}
            try:
                op = dist._make_nccl_premul_sum(factor)
                _, outs["t_rs_premul_sum_ms"] = _ev_time(
                    lambda: dist.reduce_scatter_tensor(o, g0, op=op, group=grp))
                ins["rs_premul_sum"] = _cmp(o, R)
            except Exception as e:
                ins["rs_premul_sum"] = {"error": repr(e)}
            # The reference RS (scaled input, SUM) re-timed on the same footing as the variants.
            g0.mul_(factor)
            _, outs["t_rs_sum_scaled_input_ms"] = _ev_time(
                lambda: dist.reduce_scatter_tensor(o, g0, op=dist.ReduceOp.SUM, group=grp))
            ins["rs_sum_scaled_input_repeat"] = _cmp(o, R)
            ins["timings"] = outs
            rec["insitu_rs_fold"] = ins
            del g0, o
            torch.cuda.empty_cache()
        rec["stats_local_shard_after"] = {"numel": R.numel(),
                                          "norm_fp64": float(R.double().norm())}
        _write_json(f"grad_tail_rank{_RANK}.json", rec)
        print(f"[geak-gradtail] captured rank{_RANK} step {_STEP}: factor={factor} "
              f"numel={g.numel()} norm={float(nv):.6g}", flush=True)
        return None

    BG.start_grad_sync = start_grad_sync
    ddpm.DistributedDataParallel.scale_gradients = scale_gradients
    ddpm.DistributedDataParallel.zero_grad_buffer = zero_grad_buffer
    print(f"[geak-gradtail] installed rank{_RANK}: capture step {_STEP} -> {_DIR}", flush=True)
    return True
