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


# Below this many steady steps the IQR is dominated by where the quartiles happen to land, not by
# the machine. Measured: 7 steady steps gave 1.243%, 24 gave 0.072% on the same workload.
MIN_STEADY_FOR_DISPERSION = 12


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


def summarize(dirpath, tol=0.02, force_warmup=None, max_steps=None):
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
        # force_warmup/max_steps exist so two legs can be summarised on a COMMON window. The headline
        # divides median(useful_tokens) by median(step_ms), and useful_tokens swings +-25% step to
        # step on SFT data (7389..11945 on the reference workload). So a window that starts one or
        # two steps later contains DIFFERENT SAMPLES and the headline moves -- measured at 1.1%
        # across three runs of the same config, i.e. 9x the machine's own 0.118% reproducibility.
        # Data order is deterministic, so aligning the window makes that term cancel exactly.
        i0 = int(force_warmup) if force_warmup is not None else detect_warmup_end(durs, tol=tol)
        steady = steps[i0:]
        if max_steps:
            steady = steady[:int(max_steps)]
        sd = [s["dur_ms"] for s in steady]
        disp = dispersion(sd)
        med = disp["median_ms"]
        useful = statistics.median([s.get("useful_tokens", 0) for s in steady])
        total = statistics.median([s.get("total_tokens", 0) for s in steady])
        # A dispersion estimate needs samples. With the README smoke settings (--iters 8) the steady
        # window is 7 steps, and the quartiles then straddle the near-warmup value: measured IQR
        # 1.243% against 0.078% on a 30-step run of the same workload. A band built on that is ~8x
        # too loose, so the number is flagged rather than quietly used.
        reliable = len(sd) >= MIN_STEADY_FOR_DISPERSION
        per_rank[rank] = {
            "steps_total": len(steps), "warmup_steps": i0, "steady_steps": len(sd),
            "dispersion_reliable": reliable,
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

    # Two different things live in this snapshot and conflating them breaks the gate in both
    # directions. WORKLOAD IDENTITY (what is being trained) must match or a ratio is fiction.
    # IMPLEMENTATION (how it is computed) is exactly what a candidate changes -- requiring it to match
    # would refuse every config-level candidate as "incomparable", which is what happened the first
    # time an attention-backend pair was compared.
    # GEAK_OVERLAY_SHA is IMPLEMENTATION, and listing it here is mandatory, not cosmetic: `identity`
    # is the COMPLEMENT of `impl`, so an unlisted new key silently becomes WORKLOAD IDENTITY and
    # every overlay candidate would be refused as "a different workload" before it could be gated.
    IMPL_KEYS = ("attention-backend", "GEAK_OVERLAY", "GEAK_OVERLAY_SHA")
    # Snapshot-format metadata is NEITHER half: it describes the record, not the run. It is excluded
    # from both digests and surfaces as digest_schema below.
    META_KEYS = ("snapshot_keys",)
    impl = {k: v for k, v in cfg.items() if k in IMPL_KEYS or k.startswith("NVTE_")}
    identity = {k: v for k, v in cfg.items() if k not in impl and k not in META_KEYS}
    cfg_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    impl_digest = hashlib.sha256(json.dumps(impl, sort_keys=True).encode()).hexdigest()[:16]

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
        "workload_identity": identity,
        "implementation": impl,
        # Bumped whenever the digest DEFINITION changes (e.g. when implementation keys were split
        # out of identity). Without it, a summary written under an older definition looks like a
        # different workload, and the gate refuses a comparison that is actually fine after a
        # re-summarise -- which is free, since the probe sidecars are on disk.
        #
        # READ FROM THE HEADER, not hardcoded, and that distinction is load-bearing. Schema 3 widened
        # workload identity from 5 realized fields to the regularization/optimizer/architecture set,
        # after attention_dropout 0.0 vs 0.1 digested identically. A v2 sidecar PHYSICALLY DOES NOT
        # CONTAIN attention_dropout, so unlike every previous bump this one cannot be repaired by
        # re-summarising: hardcoding 3 here would stamp v3 on a v2 header and produce a digest that
        # CLAIMS to cover dropout and does not -- precisely the fiction the guard exists to prevent.
        # A v2-vs-v3 pair must be RE-MEASURED, not re-summarised.
        "digest_schema": 3 if cfg.get("snapshot_keys") == "v3" else 2,
        "config_digest": cfg_digest,      # identity only: what is being trained
        "impl_digest": impl_digest,       # how it is computed: a candidate is EXPECTED to differ here
        # Ranks running different argv would silently mix two configurations inside ONE leg.
        "ranks_agree_on_argv": len(argv_hashes) == 1,
        "worst_spread_pct": max(v["spread_pct"] for v in per_rank.values()),
        "worst_iqr_spread_pct": max(v.get("iqr_spread_pct", 0) for v in per_rank.values()),
        "dispersion_reliable": all(v.get("dispersion_reliable") for v in per_rank.values()),
        "min_steady_steps": min(v["steady_steps"] for v in per_rank.values()),
        "outlier_steps_max": max(v["outlier_steps"] for v in per_rank.values()),
        "pad_fraction_pct": statistics.median([v["pad_fraction_pct"] or 0 for v in per_rank.values()]),
    }
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    fw = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else None
    print(json.dumps(summarize(d, tol, fw, ms), indent=2))
