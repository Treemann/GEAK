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


def _family(name, op=None):
    """Kernel family from the kernel name, falling back to the PARENT OP when the name says nothing.

    Names are vendor vocabulary. On the HF stack, ROCm's AOTriton flash-attention backward kernels
    are `bwd_kernel_dk_dv`, `bwd_kernel_fuse`, `bwd_kernel_dq` -- nothing a name pattern written for
    TE/CK kernels recognises -- and 13.8% of the profiled step was filed under "other" while the
    forward kernel, `attn_fwd`, was classed correctly. The parent op (`aten::_flash_attention_
    backward`) is stack-neutral vocabulary, so it decides when the name cannot. Name first, because
    a name that does match is the more specific evidence.
    """
    for fam, pat in FAMILY:
        if pat.search(name):
            return fam
    if op:
        for fam, pat in FAMILY:
            if pat.search(op):
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


def _main_stream(kernels):
    """The compute stream with the most busy time -- the step's critical path. None if unknown."""
    by = defaultdict(list)
    for k in kernels:
        if k.get("stream") is not None and not COMM_PAT.search(k["name"]):
            by[k["stream"]].append((k["ts"], k["ts"] + k["dur"]))
    return max(by, key=lambda s: _union_ms(by[s])) if by else None


def _side_stream_exposed_ms(kernels, main):
    """Per kernel name: ms of its SIDE-stream calls not covered by any main-stream kernel.

    A kernel on a side stream (an overlapped wgrad, a two-stream lm-head) only lengthens the step
    where the main stream has nothing running -- the same rule as exposed comm. Summing its full
    duration double-counts: on the 7-kernel packed stack the side-stream fc1 wgrad read 28% of the
    span while the main stream was busy for all of it."""
    main_iv = [(k["ts"], k["ts"] + k["dur"]) for k in kernels if k.get("stream") == main]
    side = defaultdict(list)
    for k in kernels:
        if main is not None and k.get("stream") not in (None, main) and not COMM_PAT.search(k["name"]):
            side[k["name"]].append((k["ts"], k["ts"] + k["dur"]))
    return {n: _subtract_ms(iv, main_iv) for n, iv in side.items()}


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

    main = _main_stream(kernels)
    side_exposed = _side_stream_exposed_ms(kernels, main)
    agg = defaultdict(lambda: {"ms": 0.0, "calls": 0, "family": None, "shapes": None, "side_ms": 0.0})
    for k in kernels:
        a = agg[k["name"]]
        a["ms"] += k["dur"] / 1000.0
        a["calls"] += 1
        if main is not None and k.get("stream") not in (None, main) and not COMM_PAT.search(k["name"]):
            a["side_ms"] += k["dur"] / 1000.0
        if a["family"] is None:
            a["family"] = "comm" if COMM_PAT.search(k["name"]) else _family(k["name"])
        if a["shapes"] is None and k.get("ext") in shapes_by_ext:
            a["shapes"] = shapes_by_ext[k["ext"]]
            if a["family"] == "other":
                a["family"] = _family(k["name"], (a["shapes"] or {}).get("op"))

    topn = sorted(
        ({"kernel": n, "ms": round(v["ms"], 3), "calls": v["calls"], "family": v["family"],
          "pct_span": round(100 * v["ms"] / span_ms, 3) if span_ms else 0.0,
          # THE ranking input: only the part on the critical path is recoverable. Main-stream compute
          # kernels sit on it; a communication kernel only contributes its EXPOSED fraction, and so
          # does a side-stream compute kernel (its time not covered by the main stream).
          "critical_path_share_pct": round(100 * (
              v["ms"] * exposed_ms / comm_ms if v["family"] == "comm" and comm_ms
              else v["ms"] - v["side_ms"] + side_exposed.get(n, 0.0)) / span_ms, 3)
          if span_ms else 0.0,
          "side_stream_ms": round(v["side_ms"], 3),
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
        "main_stream": main,
        "top_n": topn[:25],
    }


def config_diagnosis(buckets, bench_summary=None, rank_skew_pct=None):
    """Read-only verdict: can kernel work pay at this configuration, and if not, which knob is why."""
    findings, blocking, opportunity = [], [], []
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
            "knob": "data.num_workers / launch overhead",
            "why": ("the step is host-bound: shortening a kernel converts into MORE idle rather "
                    "than into a shorter step, so the win does not reach the headline")})
    elif idle > 5:
        findings.append({"finding": f"{idle:.1f}% idle", "knob": "data.num_workers", "why": "host-side stalls"})
    if pad > 40:
        # NOT a blocker, and the earlier claim that it was is arithmetically wrong. The headline is
        # useful_tokens / step_time; a kernel win shortens step_time and leaves useful_tokens alone,
        # so the RELATIVE gain is independent of the padding fraction (a 20% win on a head that is
        # 19% of the step is +3.95% at 70% padding and +3.95% at 0% padding). What padding really
        # says is that a MUCH larger lever exists elsewhere -- packing raises useful_tokens directly
        # -- so the kernel budget carries a high opportunity cost. That is a priority argument, not
        # an impossibility argument, and calling it blocking would have told the first real run to
        # give up on work that would have paid.
        opportunity.append({
            "finding": f"{pad:.1f}% of tokens are padding",
            "knob": "data.packing (with attention isolation)",
            "why": ("kernel wins still pay in full -- the padding fraction cancels out of "
                    "useful_tokens/step_time -- but packing would raise useful_tokens directly and "
                    "is therefore a much larger lever. High opportunity cost, not a blocker."),
            "available_in": "deep (ConfigSweep)"})
    if skew > 5:
        blocking.append({"finding": f"{skew:.1f}% rank skew", "knob": "parallel mapping / placement",
                         "why": "the step is as slow as its slowest rank; a straggler caps any win"})

    verdict = "kernel_work_can_pay" if not blocking else "config_first"
    return {
        "verdict": verdict,
        "blocking": blocking,
        # Levers that would pay MORE than the kernel track, without making the kernel track pointless.
        # Kept separate from `blocking` because the two demand different answers: a blocker says the
        # measurement cannot show a win, an opportunity says the budget could be better spent.
        "opportunity": opportunity,
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
