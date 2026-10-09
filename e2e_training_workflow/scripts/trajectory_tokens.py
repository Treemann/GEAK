#!/usr/bin/env python3
"""Compare two legs' LOSS TRAJECTORY on equal useful work, not on step index.

NOT A GATE YET. It is an instrument with one control measured and one missing; see the bottom of
knowledge/gate_convergence.md. `roles/config_tuner.md` keeps refusing Class B until both exist.

WHY. Every trajectory instrument in this layer rests on `trajectory_gate.py:38` -- both legs must run
the same seed and data order, or step i is not the same batch on both sides. A kernel change honours
that; a Class-B config change does not. Packing is the example that matters: Megatron's
`--global-batch-size` counts SEQUENCES (`scripts/adapters/megatron.sh:54`), so packing four samples
into one buffer makes a step of 64 sequences hold ~256 original samples. Step i is a different batch,
the step-1 probe and the per-step envelope compare unlike things, and L0 does not apply either
because no kernel changed. Hence "Class B has no trajectory gate at all".

That is true of the instruments as built. It is not an argument that no instrument exists.

THE SECOND ABSCISSA. Both legs read the same dataset in the same order, so after consuming the same
number of USEFUL (non-pad, loss-unmasked) tokens they have seen the same samples, whatever the step
boundaries did in between. `loss(cumulative_useful_tokens)` is therefore defined on both legs and
comparable by interpolation onto a common grid -- the move `compare_legs_tokens.py` already makes for
throughput, applied to the trajectory instead.

WHERE IT IS VALID, AND WHERE IT IS NOT. The legs must be doing the same OPTIMIZATION, not merely
reading the same data:

  packing / sequence length   same samples per optimizer update  -> comparable
  GBS / MBS                   a different NUMBER of updates over the same tokens -> NOT comparable,
                              and the difference is the intended effect, not a defect

So this covers the lever worth the most on this workload -- 70.1% of tokens are padding -- and
honestly declines the one it cannot judge. It does NOT restore the step-1 probe: that is a forward on
identical weights at one specific step, and no re-windowing recovers it.

Usage:
  trajectory_tokens.py <ref_probe_dir> <cand_probe_dir> [--key loss|grad_norm] [--points N]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_summarize import detect_warmup_end, load_rank  # noqa: E402


def _series(probe_dir, rank=0):
    """(cumulative_useful_tokens, value) per step, plus the detected warmup end."""
    path = os.path.join(probe_dir, f"probe_rank{rank}.jsonl")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(probe_dir) if f.startswith("probe_rank")]
        if not cands:
            raise FileNotFoundError(f"no probe sidecars in {probe_dir}")
        path = os.path.join(probe_dir, sorted(cands)[0])
    _, steps = load_rank(path)
    return steps, detect_warmup_end([s.get("dur_ms") for s in steps])


def _curve(steps, warm, key):
    """Cumulative useful tokens -> value, from a COMMON warmup index.

    Both legs must start at the same place in the data stream. Letting each use its own warmup is the
    bug compare_legs_tokens.py documents: the windows then hold different samples and the comparison
    re-acquires the composition artifact it exists to remove.
    """
    cum, out = 0, []
    for i, s in enumerate(steps):
        u, v = s.get("useful_tokens"), s.get(key)
        if u is None:
            continue
        cum += int(u)
        if i >= warm and v is not None:
            out.append((cum, float(v)))
    return out


def _interp(curve, x):
    """Linear interpolation; returns None outside the curve's support rather than extrapolating."""
    if not curve or x < curve[0][0] or x > curve[-1][0]:
        return None
    lo, hi = 0, len(curve) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if curve[mid][0] <= x:
            lo = mid
        else:
            hi = mid
    (x0, y0), (x1, y1) = curve[lo], curve[hi]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def compare(ref_probe, cand_probe, key="loss", points=50):
    ra, warm_a = _series(ref_probe)
    rb, warm_b = _series(cand_probe)
    warm = max(warm_a, warm_b)
    ca, cb = _curve(ra, warm, key), _curve(rb, warm, key)
    # REFUSE a short window rather than return a number from it. MEASURED, not guessed: on a clean
    # 30-step pair this comparator reports median +9.49% with sign balance 0.96, while a leg carrying
    # a known defect (cross-step staleness, alpha = 1.0) reports -7.58% with 0.06. The CLEAN pair
    # deviates more than the broken one, and both are one-sided, so the sign statistic does not
    # rescue it either. At 300 steps the same comparator puts three clean pairs at medians of +0.45,
    # -0.54 and -0.90%. Anything in between has not been characterised, so the threshold sits where
    # the evidence does.
    #
    # A number returned here would be indistinguishable from a measurement, which is exactly how the
    # soak analyzer came to report a terminal loss of 0.00000 for every leg off a truncated run.
    MIN_STEPS = 200
    if len(ca) < MIN_STEPS or len(cb) < MIN_STEPS:
        return {"error": (f"refusing to compare: {len(ca)} / {len(cb)} steady steps, minimum "
                          f"{MIN_STEPS}. This comparator has NO demonstrated sensitivity on short "
                          "windows -- at 30 steps a clean pair separates further than a known-bad "
                          "one. Do not lower this to get an answer; the answer would be noise.")}

    # The grid spans only the tokens BOTH legs consumed, so nothing is extrapolated.
    lo = max(ca[0][0], cb[0][0])
    hi = min(ca[-1][0], cb[-1][0])
    if hi <= lo:
        return {"error": "the legs' useful-token ranges do not overlap"}
    grid = [lo + (hi - lo) * i / (points - 1) for i in range(points)]

    devs = []
    for x in grid:
        ya, yb = _interp(ca, x), _interp(cb, x)
        if ya is None or yb is None or ya == 0:
            continue
        devs.append(100.0 * (yb - ya) / abs(ya))
    if not devs:
        return {"error": "no comparable grid points"}

    # TERMINAL statistic, pre-registered 2026-10-09 BEFORE the leg that tests it existed.
    #
    # Why it exists: on the alpha = 1.0 stale leg the whole-trajectory median sat inside the null
    # range while terminal loss was +6.25 sd -- a late-onset, accumulating divergence that the ~250
    # steps where the legs agree diluted away. That observation is what SUGGESTED this statistic, so
    # that leg is excluded from judging it; a fresh alpha = 1.0 leg does.
    #
    # Why THIS window and not a tuned one: "the last 10%" is the soak spec's existing definition of
    # terminal loss (l2_soak_analyze2.terminal), fixed long before this. Re-using it is the one choice
    # that cannot have been fitted to the leg that prompted it. Mean, not median, for the same
    # reason the soak uses a mean: an accumulating defect shows as a shift of the late values, and
    # there is no outlier-robustness to buy back in a window this short.
    k = max(1, int(0.10 * len(devs)))
    terminal = devs[-k:]
    terminal_mean = sum(terminal) / len(terminal)

    devs_sorted = sorted(devs)
    n = len(devs_sorted)
    med = devs_sorted[n // 2] if n % 2 else 0.5 * (devs_sorted[n // 2 - 1] + devs_sorted[n // 2])
    return {
        "basis": "equal cumulative useful tokens",
        "key": key,
        "warmup_skipped": warm,
        "grid_points": n,
        "tokens_compared": int(hi - lo),
        "median_rel_pct": round(med, 4),
        "terminal_mean_rel_pct": round(terminal_mean, 4),
        "terminal_points": k,
        "mean_rel_pct": round(sum(devs) / n, 4),
        "max_abs_rel_pct": round(max(abs(d) for d in devs), 4),
        "p90_abs_rel_pct": round(sorted(abs(d) for d in devs)[int(0.9 * (n - 1))], 4),
        # Direction matters independently of magnitude: a one-sided trajectory offset is a different
        # finding from a noisy one of the same size. Same reasoning as L0's sign_balance.
        "sign_balance": round(sum(1 for d in devs if d > 0) / n, 4),
        "note": ("Valid across a change in what a step CONTAINS (packing, sequence length) but NOT "
                 "across a change in the NUMBER of optimizer updates (GBS, MBS). Does not restore "
                 "the step-1 probe. NOT A GATE: see knowledge/gate_convergence.md for which controls "
                 "it still owes."),
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    key, points = "loss", 50
    for i, tok in enumerate(sys.argv):
        if tok == "--key":
            key = sys.argv[i + 1]
        elif tok == "--points":
            points = int(sys.argv[i + 1])
        elif tok.startswith("--") and tok not in ("--key", "--points"):
            # Unknown flags are REFUSED, not ignored. `--kind` was silently ignored here once and
            # applied a 40x tighter floor for a day.
            sys.exit(f"unknown flag {tok!r}; known: --key, --points")
    print(json.dumps(compare(args[0], args[1], key, points), indent=2))


if __name__ == "__main__":
    main()
