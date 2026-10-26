"""Compare two legs on a COMMON window — the only way the headline delta means what it says.

WHY THIS IS NOT OPTIONAL. `useful_tok_s_per_gpu` is median(useful_tokens) / median(step_ms), and on
SFT data `useful_tokens` swings +-25% step to step (7389..11945 on the reference workload). Each leg
detects its own warmup end, legitimately landing on 8, 10 or 11. A window that starts two steps later
contains DIFFERENT SAMPLES, so the headline moves even when nothing about the machine changed:

    three runs of the SAME config, each on its own window : 863.2, 862.6, 872.2  (spread 1.1%)
    the same three runs on a COMMON window               : 872.4, 871.9, 872.2  (spread 0.06%)

1.1% is 3.7x the gate's 0.3% band floor, so an unaligned comparison can accept or reject on nothing
but window placement. Data order is deterministic, so aligning the window makes the sample-composition
term cancel exactly -- there is no statistical correction needed, just the same steps on both sides.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_summarize import MIN_STEADY_FOR_DISPERSION, summarize  # noqa: E402


def _per_step(probe_dir, warm, steps):
    """(dur_ms, useful_tokens) per step for the SLOWEST rank's view, over the aligned window.

    The step is as slow as its slowest rank, so per step the max duration across ranks is taken.
    useful_tokens is the step's GLOBAL count (summed over ranks): `same_batches` asks whether both legs
    trained the same samples in step i, not whether rank 0 got the same ones -- a token-balanced leg
    (--hf-balance-tokens, DECISIONS 4.61) moves samples between ranks inside a step by design. Until
    2026-10-23 this was rank 0's count, which no unbalanced pair can tell apart from the global one.
    """
    import glob
    from probe_summarize import load_rank
    per_rank = []
    for f in sorted(glob.glob(os.path.join(probe_dir, "probe_rank*.jsonl"))):
        try:
            _, st = load_rank(f)
        except BaseException:
            return None
        per_rank.append(st[warm:warm + steps])
    if not per_rank or any(len(r) != steps for r in per_rank):
        return None
    out = []
    for i in range(steps):
        durs = [r[i].get("dur_ms") for r in per_rank]
        if any(d is None for d in durs):
            return None
        toks = [r[i].get("useful_tokens") for r in per_rank]
        out.append((max(durs), None if any(t is None for t in toks) else sum(toks)))
    return out


def _quantile(v, q):
    """Linearly interpolated quantile of a SORTED list. The old ratios[n // 4] index pick jumped by a
    whole order statistic when n moved by one -- the quartile-cliff effect windowMatchedLegs documents."""
    x = q * (len(v) - 1)
    i = int(x)
    return v[i] if i + 1 >= len(v) else v[i] + (x - i) * (v[i + 1] - v[i])


# A step slower than 1.04x its own leg's median is a stall (GC, recompile, a neighbour's burst), not
# kernel time: fixed-shape Megatron steps sit within ~0.3% of the median otherwise.
STALL_FACTOR = 1.04


def _stall_trimmed(pa, pb, fixed):
    """Sorted per-step ratios with the stalled steps of EITHER leg removed, and how many were removed.

    WHY (DECISIONS 4.54). The band is SE_SAFETY x 0.929 x IQR/sqrt(n) of the paired ratio, and one or
    two stalls per ~25-step leg land on a quartile and inflate that IQR 3-10x. On 525 Megatron bare
    true-null pairs the raw IQR left 16% of them `inconclusive_noise` and found a +0.5% effect 65% of
    the time; the trimmed IQR: 6.7% and 81%, with the same false-win count. Only for FIXED step shape:
    on dynamically padded legs step time follows content (+-25%), so a duration threshold would drop
    long batches, not stalls -- and the paired ratio there is already tight (HF: same power either way).
    Falls back to every step when the trim would leave fewer than max(12, n/2)."""
    allr = sorted(y[0] / x[0] for x, y in zip(pa, pb) if x[0] > 0 and y[0] > 0)
    if not fixed:
        return allr, 0
    ma = _quantile(sorted(x[0] for x in pa), 0.5)
    mb = _quantile(sorted(y[0] for y in pb), 0.5)
    keep = sorted(y[0] / x[0] for x, y in zip(pa, pb)
                  if 0 < x[0] <= STALL_FACTOR * ma and 0 < y[0] <= STALL_FACTOR * mb)
    if len(keep) < max(12, len(allr) // 2):
        return allr, 0
    return keep, len(allr) - len(keep)


TRIM_FRAC = 0.10


def _tmean_delta(ratios):
    """Delta from the 10%-per-side TRIMMED MEAN of the per-step ratios (audit 2026-10-25 #6). The median is
    blind to a kernel that is slower on SOME steps: 4 of 22 steps 10% slower, the rest 1% faster reads +1.01%
    by median and -1.0% in total. A stall-threshold trim cannot help (STALL_FACTOR 1.04 would drop exactly
    those steps), but a trimmed mean keeps a systematic minority of slow steps while the 10% cut on each
    side removes the one or two host stalls a 25-step leg carries in either leg."""
    r = sorted(ratios)
    k = int(len(r) * TRIM_FRAC)
    core = r[k:len(r) - k] if len(r) - 2 * k >= 5 else r
    m = sum(core) / len(core)
    return round(100 * (1.0 / m - 1.0), 4) if m > 0 else None


def compare(ref_probe, cand_probe, tol=0.02):
    a, b = summarize(ref_probe, tol), summarize(cand_probe, tol)
    for x, name in ((a, "ref"), (b, "cand")):
        if "error" in x:
            return {"error": f"{name}: {x['error']}"}

    # Start both at the LATER warmup end: the earlier one may still be settling in the other leg.
    warm = max(a["warmup_steps_max"], b["warmup_steps_max"])
    steps = min(a["per_rank"][sorted(a["per_rank"])[0]]["steps_total"],
                b["per_rank"][sorted(b["per_rank"])[0]]["steps_total"]) - warm
    if steps < 1:
        return {"error": "no overlapping steady window"}
    A = summarize(ref_probe, tol, force_warmup=warm, max_steps=steps)
    B = summarize(cand_probe, tol, force_warmup=warm, max_steps=steps)

    ra, rb = A["useful_tok_s_per_gpu"], B["useful_tok_s_per_gpu"]
    out = {
        "window": {"warmup_skipped": warm, "steps_compared": steps,
                   "own_warmup": {"ref": a["warmup_steps_max"], "cand": b["warmup_steps_max"]}},
        "ref": {"useful_tok_s_per_gpu": ra, "step_median_ms": A["step_median_ms"],
                "iqr_spread_pct": A["worst_iqr_spread_pct"], "config_digest": A["config_digest"],
                "impl_digest": A.get("impl_digest")},
        "cand": {"useful_tok_s_per_gpu": rb, "step_median_ms": B["step_median_ms"],
                 "iqr_spread_pct": B["worst_iqr_spread_pct"], "config_digest": B["config_digest"],
                 "impl_digest": B.get("impl_digest")},
        "delta_pct": round(100 * (rb - ra) / ra, 4) if ra else None,
        "unaligned_delta_pct": round(100 * (b["useful_tok_s_per_gpu"] - a["useful_tok_s_per_gpu"])
                                     / a["useful_tok_s_per_gpu"], 4) if a["useful_tok_s_per_gpu"] else None,
        "dispersion_reliable": bool(A.get("dispersion_reliable") and B.get("dispersion_reliable")),
        "min_steady_steps": min(A.get("min_steady_steps", 0), B.get("min_steady_steps", 0)),
    }
    # REPORT-ONLY wall-time delta on the SAME aligned window (DECISIONS 4.41). Per-leg bench_summary
    # wall rates use each leg's own window, and with 1-2 stalls per ~20 steps the window start alone
    # moves them by about the size of the effect, so a wall delta is only quoted from here.
    wa, wb = A.get("wall_useful_tok_s_per_gpu"), B.get("wall_useful_tok_s_per_gpu")
    out["wall_delta_pct"] = round(100 * (wb - wa) / wa, 4) if wa and wb else None
    out["ref"]["stall_pct"], out["cand"]["stall_pct"] = A.get("stall_pct"), B.get("stall_pct")
    # ---- PAIRED estimator (added 2026-10-09; what the gate reads when same_batches holds) ----
    # delta_pct above is median(useful)/median(dur) per leg, each median taken independently. That is
    # robust only when every step has the same SHAPE -- true on Megatron, which pads every sample to
    # SEQ (step-time IQR 0.30%). On a dynamically-padded stack step time follows content (HF TRL:
    # IQR ~25%), the two legs' medians land on different steps, and two IDENTICAL HF runs read as
    # +1.1523% apart. With a shared data order step i is the same batch on both legs, so the per-step
    # ratio cancels content exactly; its median is also robust to the stall outliers that a
    # sum-over-sum rate would let through. On that same HF null pair it reads -0.15%.
    pa = _per_step(ref_probe, warm, steps)
    pb = _per_step(cand_probe, warm, steps)
    if pa and pb and len(pa) == len(pb):
        same_batches = all(x[1] == y[1] for x, y in zip(pa, pb))
        ratios = sorted(y[0] / x[0] for x, y in zip(pa, pb) if x[0] > 0 and y[0] > 0)
        if ratios:
            n = len(ratios)
            med_r = _quantile(ratios, 0.5)
            disp, dropped = _stall_trimmed(pa, pb, fixed=(A.get("step_shape") == "fixed"
                                                         and B.get("step_shape") == "fixed"))
            out["paired"] = {
                # The DELTA stays the untrimmed median: trimming changes which steps it is the
                # median of, and on true-null pairs that manufactured a 14% delta (DECISIONS 4.54).
                "delta_pct": round(100 * (1.0 / med_r - 1.0), 4),
                "ratio_iqr_pct": round(100 * (_quantile(disp, 0.75) - _quantile(disp, 0.25))
                                       / _quantile(disp, 0.5), 4),
                "ratio_iqr_raw_pct": round(100 * (_quantile(ratios, 0.75) - _quantile(ratios, 0.25)) / med_r, 4),
                # n of the sample the dispersion came from -- what the gate divides by.
                "steps": len(disp),
                "steps_total": n,
                "steps_dropped": dropped,
                # The estimator is only valid if step i really is the same batch on both legs.
                "same_batches": same_batches,
                # total time over the same (stall-trimmed) steps: the gate's veto on heterogeneous effects
                "tmean_delta_pct": _tmean_delta(ratios),
                # report-only: total time over EVERY aligned step, stalls included
                "sum_delta_raw_pct": round(100 * (sum(x[0] for x in pa) / sum(y[0] for y in pb) - 1.0), 4),
            }

    notes = []
    if A["config_digest"] != B["config_digest"]:
        notes.append("WORKLOAD IDENTITY DIFFERS — the ratio is not about one configuration")
    if not out["dispersion_reliable"]:
        notes.append(f"steady window is {out['min_steady_steps']} steps (< {MIN_STEADY_FOR_DISPERSION}): "
                     "the IQR is a sample-size artifact, not a noise band — re-run with more iters")
    if out["unaligned_delta_pct"] is not None and out["delta_pct"] is not None:
        drift = abs(out["unaligned_delta_pct"] - out["delta_pct"])
        out["window_artifact_pct"] = round(drift, 4)
        if drift > 0.3:
            notes.append(f"window placement alone accounted for {drift:.2f}% — reporting the "
                         "unaligned number would have been mostly artifact")
    out["notes"] = notes
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: compare_legs.py <ref_probe_dir> <cand_probe_dir>")
        raise SystemExit(2)
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
