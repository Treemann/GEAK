#!/usr/bin/env python3
"""P0-4e verdicts (knowledge/gate_convergence.md, "P0-4e PRE-REGISTRATION").

Under the standard LR schedule (warmup + cosine decay) the end-of-training held-out loss is the primary
quality metric: with the LR decayed, the final weights settle and REF's seed spread is ~0.005, while a
x0.7 schedule moved it +0.0206 (P0-4d, all five seeds positive). That +0.0206 is the FIXED yardstick.

  quality   PASS iff |mean d| <= 0.25 x 0.0206 (= 0.00515) and NOT (mean d > 0 and t >= 2.776)
            FAIL iff mean d > 0.00515 and t >= 2.776;  otherwise INCONCLUSIVE
            d_k = held-out(S8)_k - held-out(REF)_k, paired by seed, n = 5
  stability FAIL iff any S8 leg has max10 > 1.0 (catastrophic)
            PASS iff no catastrophic and at most 1 S8 leg with max10 > 0.25 (severe)
            otherwise INCONCLUSIVE.  max10: largest 10-step mean of loss(S8) - loss(REF) after step 150.
  CLEARED iff both PASS.

Usage: quality_ab_final.py <runs/<name>_STAMP prefix> [--cand s8]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

YARDSTICK = 0.0206
MARGIN = 0.25 * YARDSTICK
T_CRIT = 2.776
SEVERE, CATASTROPHIC, AFTER, WIN = 0.25, 1.0, 150, 10
VAL_RE = re.compile(r"validation loss at .*? lm loss value: ([0-9.Ee+-]+)")


def steps(d):
    p = os.path.join(d, "probe", "probe_rank0.jsonl")
    return [r for r in (json.loads(l) for l in open(p)) if r.get("record") == "step"] if os.path.exists(p) else []


def heldout(d):
    try:
        v = VAL_RE.findall(open(os.path.join(d, "train.log"), errors="replace").read())
        return float(v[-1]) if v else None
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--cand", default="s8")
    a = ap.parse_args()
    P = a.prefix.rstrip("/")
    seeds = sorted({int(m.group(1)) for f in glob.glob(P + "_ref_seed*") for m in [re.search(r"seed(\d+)$", f)] if m})
    rows, d, sev = [], [], []
    for s in seeds:
        r, c = f"{P}_ref_seed{s}", f"{P}_{a.cand}_seed{s}"
        sr, sc = steps(r), steps(c)
        hr, hc = heldout(r), heldout(c)
        if len(sr) < 358 or len(sc) < 358 or hr is None or hc is None:
            print(f"INCOMPLETE: seed {s} ref {len(sr)} steps / {hr}, cand {len(sc)} steps / {hc}")
            return
        diff = [x["loss"] - y["loss"] for x, y in zip(sc, sr)]
        m10 = max(statistics.fmean(diff[i:i + WIN]) for i in range(AFTER, len(diff) - WIN + 1))
        d.append(hc - hr)
        sev.append(m10)
        print(f"seed {s}: held-out REF {hr:.6f} {a.cand} {hc:.6f} d {hc - hr:+.5f} | max10 {m10:.3f}")
    m, sd = statistics.fmean(d), statistics.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else math.copysign(math.inf, m)
    if m > MARGIN and t >= T_CRIT:
        qual = f"FAIL-quality: mean {m:+.5f} > margin {MARGIN:.5f}, t {t:.2f}"
    elif abs(m) <= MARGIN and not (m > 0 and t >= T_CRIT):
        qual = f"PASS-quality: |mean {m:+.5f}| <= {MARGIN:.5f} (0.25 x yardstick {YARDSTICK}), t {t:.2f}"
    else:
        qual = f"INCONCLUSIVE-quality: mean {m:+.5f}, t {t:.2f}, margin {MARGIN:.5f}"
    n_cat, n_sev = sum(x > CATASTROPHIC for x in sev), sum(x > SEVERE for x in sev)
    if n_cat:
        stab = f"FAIL-stability: {n_cat} catastrophic leg(s)"
    elif n_sev <= 1:
        stab = f"PASS-stability: {n_sev} severe, 0 catastrophic"
    else:
        stab = f"INCONCLUSIVE-stability: {n_sev} severe, 0 catastrophic"
    cleared = qual.startswith("PASS") and stab.startswith("PASS")
    print("QUALITY:  ", qual)
    print("STABILITY:", stab)
    print("OVERALL:  ", "CLEARED" if cleared else "NOT CLEARED")
    print(json.dumps({"seeds": seeds, "d": [round(x, 6) for x in d], "mean": round(m, 6), "t": round(t, 3),
                      "max10": [round(x, 3) for x in sev], "quality": qual, "stability": stab, "cleared": cleared}))


if __name__ == "__main__":
    main()
