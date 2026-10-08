#!/usr/bin/env python3
"""Compare two legs on EQUAL USEFUL WORK, for config changes that alter what a step contains.

WHY A SECOND COMPARATOR EXISTS. `compare_legs.py` aligns two legs by STEP INDEX, and every
trajectory instrument in this layer rests on the same premise, stated in trajectory_gate.py:38 --
"both legs must be run with the SAME seed and the SAME data order; otherwise step i is not the same
batch on both sides and none of this means anything."

A kernel change honours that premise: it changes HOW a step is computed, not WHAT is in it. Several
config changes do not. Packing, GBS, MBS and sequence length all change how many samples land in a
step, so step i holds different data on the two legs and step-index alignment compares unlike things.
The one lever worth the most on this workload -- 70.1% of tokens are padding -- is in exactly that
class.

WHAT IS COMPARABLE ACROSS SUCH A CHANGE. Not steps, and not wall time per step. The invariant is the
USEFUL WORK itself: `useful_tokens` counts non-pad, loss-unmasked tokens, so packing changes how fast
they are consumed and not how many there are. This comparator therefore asks:

    how long did each leg take to consume the SAME NUMBER of useful tokens?

which is well defined whatever a step contains.

WHAT THIS DOES NOT RESTORE. The step-1 forward probe and the per-step envelope stay invalid for this
class -- they compare step i to step i and no amount of re-windowing fixes that. A config change of
this kind therefore has NO trajectory gate, which is a real safety gap and must be stated wherever
such a result is reported, not discovered later. See knowledge/gate_convergence.md.

Usage:
  compare_legs_tokens.py <ref_probe_dir> <cand_probe_dir> [--tokens N]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_summarize import detect_warmup_end, load_rank  # noqa: E402


def _steps_raw(probe_dir, rank=0):
    """ALL per-step (dur_ms, useful_tokens) for one rank, plus the DETECTED warmup end.

    Returns every step rather than pre-trimming, so the caller can align both legs on a common
    warmup -- which it must, or the two windows start at different places in the data stream."""
    path = os.path.join(probe_dir, f"probe_rank{rank}.jsonl")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(probe_dir) if f.startswith("probe_rank")]
        if not cands:
            raise FileNotFoundError(f"no probe sidecars in {probe_dir}")
        path = os.path.join(probe_dir, sorted(cands)[0])
    _, steps = load_rank(path)
    durs = [s.get("dur_ms") for s in steps]
    warm = detect_warmup_end(durs)
    out = []
    for s in steps:
        d, u = s.get("dur_ms"), s.get("useful_tokens")
        if d is None or u is None:
            continue
        out.append((float(d), int(u)))
    return out, warm


def compare_on_tokens(ref_probe, cand_probe, target_tokens=None):
    """Time taken by each leg to consume the same number of useful tokens."""
    # START BOTH AT THE LATER WARMUP END, exactly as compare_legs.py does. Letting each leg use its
    # own warmup starts them at DIFFERENT POSITIONS IN THE DATA STREAM, so the two windows hold
    # different samples and the equal-token comparison re-acquires the sample-composition artifact it
    # exists to avoid. Measured before this was fixed: 1.6-2.9% worst-case error on pairs whose true
    # delta is zero, against 0.035-0.071% for the step-aligned comparator.
    ra_all, warm_a0 = _steps_raw(ref_probe)
    rb_all, warm_b0 = _steps_raw(cand_probe)
    warm = max(warm_a0, warm_b0)
    ra, rb = ra_all[warm:], rb_all[warm:]
    warm_a = warm_b = warm
    if not ra or not rb:
        return {"error": "no steady steps on one of the legs"}

    tot_a = sum(u for _, u in ra)
    tot_b = sum(u for _, u in rb)
    # The budget is what BOTH legs actually completed; asking for more would extrapolate.
    budget = min(tot_a, tot_b) if target_tokens is None else min(target_tokens, tot_a, tot_b)
    if budget <= 0:
        return {"error": "no useful tokens recorded"}

    def consume(rows):
        """Wall time to reach `budget` useful tokens; the final step is prorated, not rounded.

        Rounding to a whole step would quantise the comparison by one step -- ~11.6 s at 7B against
        signals of a few percent -- and the quantisation would differ between legs whose steps hold
        different amounts of work, which is precisely the bias this comparator exists to avoid.
        """
        acc_t, acc_u, n = 0.0, 0, 0
        for d, u in rows:
            if acc_u + u >= budget:
                frac = (budget - acc_u) / u if u else 1.0
                return acc_t + d * frac, n + frac
            acc_t += d
            acc_u += u
            n += 1
        return acc_t, n

    ta, na = consume(ra)
    tb, nb = consume(rb)
    rate_a = budget / ta if ta else None          # useful tokens per ms
    rate_b = budget / tb if tb else None
    return {
        "basis": "equal useful tokens",
        "useful_tokens_compared": budget,
        "ref": {"ms": round(ta, 1), "steps_used": round(na, 2), "total_useful": tot_a,
                "warmup_skipped": warm_a, "tok_per_ms": round(rate_a, 6) if rate_a else None},
        "cand": {"ms": round(tb, 1), "steps_used": round(nb, 2), "total_useful": tot_b,
                 "warmup_skipped": warm_b, "tok_per_ms": round(rate_b, 6) if rate_b else None},
        "delta_pct": round(100 * (rate_b - rate_a) / rate_a, 4) if rate_a and rate_b else None,
        "coverage": {"ref_frac_of_leg": round(budget / tot_a, 4), "cand_frac_of_leg": round(budget / tot_b, 4)},
        "note": ("Equal-useful-work basis. Valid across a change in what a step contains (packing, "
                 "GBS, MBS, seq). It does NOT restore the step-1 probe or the per-step envelope: "
                 "those compare step i to step i and remain invalid for this class of change, so a "
                 "result from here carries NO trajectory gate."),
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    target = None
    if "--tokens" in sys.argv:
        target = int(sys.argv[sys.argv.index("--tokens") + 1])
    for tok in sys.argv[1:]:
        if tok.startswith("--") and tok != "--tokens":
            sys.exit(f"unknown flag {tok!r}; known: --tokens")
    print(json.dumps(compare_on_tokens(args[0], args[1], target), indent=2))


if __name__ == "__main__":
    main()
