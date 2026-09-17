"""Summarize per-rank probe sidecars: warmup detection, steady window, useful tok/s, rank skew.

POLICY LIVES HERE, not in the probe. Measured on the 7B baseline, the SAME run reports a 12.75% or a
0.18% spread depending only on where the warmup is declared to end -- so the rule is explicit and
re-runnable against the stored series rather than baked into the collector.

Warmup rule: walk forward and take the first index from which every subsequent step is within
`tol` of the median of the remaining tail. That detects the real settling point (autotune, allocator,
comm bring-up) instead of trusting a hard-coded skip count.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import statistics
import sys


def load_rank(path):
    header, steps = None, []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("record") == "header":
                header = r
            elif r.get("record") == "step":
                steps.append(r)
    return header, steps


def detect_warmup_end(durs, tol=0.02, patience=3):
    """First index from which the run has SETTLED to its steady level.

    Warmup is a monotone settling phenomenon (autotune, allocator, comm bring-up); later isolated
    spikes (a dataloader stall, another tenant) are a DIFFERENT phenomenon and must not push the
    warmup boundary forward. So the reference level is the median of the back half, and the boundary
    is the first index that stays at or below ref*(1+tol) for `patience` consecutive steps. Outliers
    after that point are counted separately rather than absorbed into the warmup.
    """
    n = len(durs)
    if n < patience + 1:
        return 0
    ref = statistics.median(durs[max(0, n // 2):])
    limit = ref * (1 + tol)
    for i in range(n - patience + 1):
        if all(d <= limit for d in durs[i:i + patience]):
            return i
    return 0


def dispersion(vals):
    """Report spread AND a robust spread, so 'wide because of warmup' is distinguishable from
    'wide because of periodic stalls'. Never collapse them into one number."""
    med = statistics.median(vals)
    out = {
        "median_ms": round(med, 2),
        "spread_pct": round(100 * (max(vals) - min(vals)) / med, 3),
        "stdev_ms": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
    }
    s = sorted(vals)
    if len(s) >= 4:
        q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
        out["iqr_spread_pct"] = round(100 * (q3 - q1) / med, 3)
    outliers = [v for v in vals if v > med * 1.02]
    out["outlier_steps"] = len(outliers)
    out["outlier_max_pct"] = round(100 * (max(outliers) - med) / med, 2) if outliers else 0.0
    return out


def summarize(dirpath, tol=0.02):
    ranks = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "probe_rank*.jsonl"))):
        h, s = load_rank(p)
        if h is None or not s:
            continue
        ranks[h["rank"]] = (h, s)
    if not ranks:
        return {"error": f"no probe sidecars in {dirpath}"}

    per_rank, world = {}, max(len(ranks), int(next(iter(ranks.values()))[0].get("world_size", 1)))
    for rank, (h, steps) in sorted(ranks.items()):
        durs = [s["dur_ms"] for s in steps]
        i0 = detect_warmup_end(durs, tol=tol)
        steady = steps[i0:]
        sd = [s["dur_ms"] for s in steady]
        disp = dispersion(sd)
        med = disp["median_ms"]
        useful = statistics.median([s.get("useful_tokens", 0) for s in steady])
        total = statistics.median([s.get("total_tokens", 0) for s in steady])
        per_rank[rank] = {
            "steps_total": len(steps), "warmup_steps": i0, "steady_steps": len(sd),
            **disp,
            # Per-GPU rate: this rank's own tokens over its own step time.
            "useful_tokens_per_step": useful, "total_tokens_per_step": total,
            "pad_fraction_pct": round(100 * (1 - useful / total), 2) if total else None,
            "useful_tok_s_per_gpu": round(useful / (med / 1000.0), 1) if med else None,
            "nominal_tok_s_per_gpu": round(total / (med / 1000.0), 1) if med else None,
            "peak_reserved_mb": max((s.get("peak_reserved_mb", 0) for s in steady), default=0),
            "peak_alloc_mb": max((s.get("peak_alloc_mb", 0) for s in steady), default=0),
        }

    # Realized config, as OBSERVED by the probe in each rank's process (not as intended by the
    # caller). This is what makes "did these two legs measure the same thing?" answerable, and it is
    # the pinned-config verification input (Seam 1).
    headers = [h for h, _ in ranks.values()]
    cfg = dict(headers[0].get("config", {}))
    cfg["world_size"] = str(len(ranks))
    argv_hashes = sorted({h.get("argv_hash") for h in headers})
    cfg_digest = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

    meds = [v["median_ms"] for v in per_rank.values()]
    slowest = max(per_rank.items(), key=lambda kv: kv[1]["median_ms"])[0]
    out = {
        "dir": dirpath, "ranks": len(per_rank), "world_size": world,
        "per_rank": per_rank,
        # The step is as slow as its slowest rank, so the run-level number takes the max, and the
        # skew is reported instead of being averaged away -- it is the straggler bucket's input.
        "step_median_ms": round(max(meds), 2),
        "rank_skew_pct": round(100 * (max(meds) - min(meds)) / statistics.median(meds), 3),
        "slowest_rank": slowest,
        "useful_tok_s_per_gpu": round(statistics.median(
            [v["useful_tok_s_per_gpu"] or 0 for v in per_rank.values()]), 1),
        "nominal_tok_s_per_gpu": round(statistics.median(
            [v["nominal_tok_s_per_gpu"] or 0 for v in per_rank.values()]), 1),
        "peak_reserved_mb": max(v["peak_reserved_mb"] for v in per_rank.values()),
        "warmup_steps_max": max(v["warmup_steps"] for v in per_rank.values()),
        "config": cfg,
        "config_digest": cfg_digest,
        # Ranks running different argv would silently mix two configurations inside ONE leg.
        "ranks_agree_on_argv": len(argv_hashes) == 1,
        "worst_spread_pct": max(v["spread_pct"] for v in per_rank.values()),
        "worst_iqr_spread_pct": max(v.get("iqr_spread_pct", 0) for v in per_rank.values()),
        "outlier_steps_max": max(v["outlier_steps"] for v in per_rank.values()),
        "pad_fraction_pct": statistics.median([v["pad_fraction_pct"] or 0 for v in per_rank.values()]),
    }
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    print(json.dumps(summarize(d, tol), indent=2))
