"""Step decomposition + Top-N + the read-only config diagnosis, from probe-driven traces.

WHY NOT %GPU TIME. The inference layer ranks candidates by `pct_gpu_time x achievable_speedup`.
That formula MISLEADS in training: a collective that is fully overlapped with compute can own 30% of
GPU time and be worth exactly zero, because removing it removes nothing from the critical path. So
the step is decomposed into four buckets and candidates are ranked on the part that is actually
recoverable:

    compute        GPU busy with non-communication kernels
    exposed_comm   communication NOT hidden behind compute -- the recoverable part of comms
    idle           no kernel resident: host/dataloader stalls, launch-bound stretches, PP bubble
    (straggler)    cross-rank skew, reported separately from the probe's per-rank medians

THE READ-ONLY CONFIG DIAGNOSIS. `fast` mode has no ConfigSweep, so it jumps straight to the kernel
track -- which in the inference doctrine is the expensive lever, normally pulled only after the cheap
ones. The specific hazard: if the step is dominated by exposed comm, idle, or padding, NO kernel win
can show up end to end and the budget burns on isolated speedups the gate then correctly rejects.
This module therefore names the offending knob and says whether kernel work can pay -- and changes
nothing. Diagnosis, not treatment; it is also the requirements list for ConfigSweep when deep lands.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

COMM_PAT = re.compile(r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|broadcast|send|recv",
                      re.IGNORECASE)
# Coarse op families, used for routing and for the Top-N's "what kind of thing is this" column.
FAMILY = (
    ("attention", re.compile(r"attn|attention|flash|fmha|softmax.*mask", re.I)),
    ("gemm", re.compile(r"gemm|matmul|Cijk|hipblaslt|rocblas|linear", re.I)),
    ("norm", re.compile(r"norm|rms|layer_?norm", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized_elementwise|add|mul|copy|cast|fill", re.I)),
    ("optimizer", re.compile(r"adam|optimizer|clip_?grad|multi_tensor", re.I)),
    ("moe", re.compile(r"moe|topk|permute|grouped", re.I)),
)


def _family(name):
    for fam, pat in FAMILY:
        if pat.search(name):
            return fam
    return "other"


def _union_ms(intervals):
    """Total wall time covered by a set of [start, end] intervals (overlaps counted once)."""
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total / 1000.0   # chrome traces are in microseconds


def _subtract_ms(a, b):
    """Time covered by intervals `a` but NOT by intervals `b` -- i.e. EXPOSED comm."""
    if not a:
        return 0.0
    bs = sorted(b)
    total = 0.0
    for s, e in sorted(a):
        cur = s
        for x, y in bs:
            if y <= cur:
                continue
            if x >= e:
                break
            if x > cur:
                total += min(x, e) - cur
            cur = max(cur, y)
            if cur >= e:
                break
        if cur < e:
            total += e - cur
    return total / 1000.0


def load_trace(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def decompose(trace_path):
    events = load_trace(trace_path)
    kernels, shapes_by_ext = [], {}
    for e in events:
        cat = (e.get("cat") or "").lower()
        if cat in ("cpu_op", "user_annotation"):
            dims = (e.get("args") or {}).get("Input Dims")
            ext = (e.get("args") or {}).get("External id")
            if dims and ext is not None and ext not in shapes_by_ext:
                shapes_by_ext[ext] = {"op": e.get("name"), "dims": dims}
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ts, dur = e.get("ts"), e.get("dur")
            if ts is None or dur is None:
                continue
            kernels.append({
                "name": e.get("name", "?"), "ts": ts, "dur": dur,
                "stream": (e.get("args") or {}).get("stream"),
                "ext": (e.get("args") or {}).get("External id"),
                "cat": cat,
            })
    if not kernels:
        return {"error": f"no GPU kernel events in {trace_path}"}

    comm = [(k["ts"], k["ts"] + k["dur"]) for k in kernels if COMM_PAT.search(k["name"])]
    comp = [(k["ts"], k["ts"] + k["dur"]) for k in kernels if not COMM_PAT.search(k["name"])]
    span_start = min(k["ts"] for k in kernels)
    span_end = max(k["ts"] + k["dur"] for k in kernels)
    span_ms = (span_end - span_start) / 1000.0

    compute_ms = _union_ms(comp)
    comm_ms = _union_ms(comm)
    exposed_ms = _subtract_ms(comm, comp)
    busy_ms = _union_ms(comp + comm)
    idle_ms = max(0.0, span_ms - busy_ms)

    agg = defaultdict(lambda: {"ms": 0.0, "calls": 0, "family": None, "shapes": None})
    for k in kernels:
        a = agg[k["name"]]
        a["ms"] += k["dur"] / 1000.0
        a["calls"] += 1
        if a["family"] is None:
            a["family"] = "comm" if COMM_PAT.search(k["name"]) else _family(k["name"])
        if a["shapes"] is None and k.get("ext") in shapes_by_ext:
            a["shapes"] = shapes_by_ext[k["ext"]]

    topn = sorted(
        ({"kernel": n, "ms": round(v["ms"], 3), "calls": v["calls"], "family": v["family"],
          "pct_span": round(100 * v["ms"] / span_ms, 3) if span_ms else 0.0,
          # THE ranking input: only the part on the critical path is recoverable. Compute kernels
          # sit on it; a communication kernel only contributes its EXPOSED fraction.
          "critical_path_share_pct": round(
              100 * v["ms"] / span_ms * (exposed_ms / comm_ms if v["family"] == "comm" and comm_ms else 1.0), 3)
          if span_ms else 0.0,
          "op": (v["shapes"] or {}).get("op"), "input_dims": (v["shapes"] or {}).get("dims")}
         for n, v in agg.items()),
        key=lambda d: -d["critical_path_share_pct"])

    return {
        "trace": trace_path,
        "span_ms": round(span_ms, 3),
        "buckets": {
            "compute_ms": round(compute_ms, 3),
            "comm_total_ms": round(comm_ms, 3),
            "exposed_comm_ms": round(exposed_ms, 3),
            "idle_ms": round(idle_ms, 3),
            "compute_pct": round(100 * compute_ms / span_ms, 2) if span_ms else 0,
            "exposed_comm_pct": round(100 * exposed_ms / span_ms, 2) if span_ms else 0,
            "idle_pct": round(100 * idle_ms / span_ms, 2) if span_ms else 0,
            "comm_overlap_pct": round(100 * (1 - exposed_ms / comm_ms), 2) if comm_ms else None,
        },
        "kernels_total": len(kernels),
        "top_n": topn[:25],
    }


def config_diagnosis(buckets, bench_summary=None, rank_skew_pct=None):
    """Read-only verdict: can kernel work pay at this configuration, and if not, which knob is why."""
    findings, blocking = [], []
    exposed = buckets.get("exposed_comm_pct") or 0
    idle = buckets.get("idle_pct") or 0
    pad = (bench_summary or {}).get("pad_fraction_pct") or 0
    skew = rank_skew_pct or 0

    if exposed > 15:
        blocking.append({
            "finding": f"{exposed:.1f}% of the step is EXPOSED communication",
            "knob": "comm.overlap_grad_reduce / comm.overlap_param_gather / bucket size",
            "why": "a kernel win shortens compute, which exposes MORE comm; the step barely moves"})
    elif exposed > 5:
        findings.append({"finding": f"{exposed:.1f}% exposed comm", "knob": "comm.overlap_*",
                         "why": "material but not dominant"})
    if idle > 15:
        blocking.append({
            "finding": f"{idle:.1f}% of the step has NO kernel resident",
            "knob": "data.num_workers / data.packing / launch overhead",
            "why": "the GPU is waiting on the host; making GPU kernels faster changes nothing"})
    elif idle > 5:
        findings.append({"finding": f"{idle:.1f}% idle", "knob": "data.num_workers", "why": "host-side stalls"})
    if pad > 40:
        blocking.append({
            "finding": f"{pad:.1f}% of tokens are padding",
            "knob": "data.packing (with attention isolation)",
            "why": ("most of the compute is spent on padding, so a kernel win is diluted by the same "
                    "factor; and the nominal tokens/s a stack reports hides this entirely")})
    if skew > 5:
        blocking.append({"finding": f"{skew:.1f}% rank skew", "knob": "parallel mapping / placement",
                         "why": "the step is as slow as its slowest rank; a straggler caps any win"})

    verdict = "kernel_work_can_pay" if not blocking else "config_first"
    return {
        "verdict": verdict,
        "blocking": blocking,
        "findings": findings,
        "note": ("READ-ONLY. fast mode does not fix these; it reports them so a rejected kernel win "
                 "is attributable to the configuration rather than mistaken for a bad kernel. This "
                 "list is also ConfigSweep's requirements list when deep mode lands."),
    }


def analyse(probe_dir, bench_summary_path=None):
    traces = sorted(glob.glob(os.path.join(probe_dir, "trace_rank*.json")))
    if not traces:
        return {"error": f"no traces in {probe_dir}; run with GEAK_PROFILE_STEPS=start:end"}
    summary = None
    if bench_summary_path and os.path.exists(bench_summary_path):
        summary = json.load(open(bench_summary_path))
    per_rank = {}
    for t in traces:
        m = re.search(r"trace_rank(\d+)\.json", t)
        per_rank[int(m.group(1)) if m else len(per_rank)] = decompose(t)
    primary = per_rank[sorted(per_rank)[0]]
    if "error" in primary:
        return primary
    out = {
        "probe_dir": probe_dir,
        "ranks_profiled": sorted(per_rank),
        "buckets": primary["buckets"],
        "span_ms": primary["span_ms"],
        "top_n": primary["top_n"],
        "per_rank_buckets": {r: v.get("buckets") for r, v in per_rank.items()},
    }
    out["config_diagnosis"] = config_diagnosis(
        primary["buckets"], summary, (summary or {}).get("rank_skew_pct"))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parse_train_profile.py <probe_dir> [bench_summary.json]")
        raise SystemExit(2)
    print(json.dumps(analyse(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=2))
