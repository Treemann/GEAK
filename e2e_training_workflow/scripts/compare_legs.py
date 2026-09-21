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
