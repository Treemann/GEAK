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
    ap.add_argument("--base", default="ref", help="the comparison arm (e.g. s8 when testing a lever ON the stack)")
    ap.add_argument("--ctrl", default=None,
                    help="measure the yardstick IN THIS RUN from a positive-control arm (e.g. ctrl70) instead of "
                         "the fixed P0-4d value; the control must then be DETECTED (mean > 0, t >= 2.776)")
    a = ap.parse_args()
    P = a.prefix.rstrip("/")
    seeds = sorted({int(m.group(1)) for f in glob.glob(P + f"_{a.base}_seed*") for m in [re.search(r"seed(\d+)$", f)] if m})
    rows, d, sev = [], [], []
    for s in seeds:
        r, c = f"{P}_{a.base}_seed{s}", f"{P}_{a.cand}_seed{s}"
        sr, sc = steps(r), steps(c)
        hr, hc = heldout(r), heldout(c)
        # Complete = both legs ran the same number of steps, enough to cover the episode window after
        # step AFTER (358 on the Megatron protocol, 296 on HF's one-epoch protocol), and both evaluated.
        if len(sr) != len(sc) or len(sr) < AFTER + WIN + 1 or hr is None or hc is None:
            print(f"INCOMPLETE: seed {s} ref {len(sr)} steps / {hr}, cand {len(sc)} steps / {hc}")
            return
        diff = [x["loss"] - y["loss"] for x, y in zip(sc, sr)]
        m10 = max(statistics.fmean(diff[i:i + WIN]) for i in range(AFTER, len(diff) - WIN + 1))
        d.append(hc - hr)
        sev.append(m10)
        print(f"seed {s}: held-out {a.base} {hr:.6f} {a.cand} {hc:.6f} d {hc - hr:+.5f} | max10 {m10:.3f}")
    yard, ctrl_note = YARDSTICK, "fixed (P0-4d)"
    if a.ctrl:
        cd = []
        for s in seeds:
            hb, hk = heldout(f"{P}_{a.base}_seed{s}"), heldout(f"{P}_{a.ctrl}_seed{s}")
            if hk is None:
                print(f"INCOMPLETE: control seed {s} has no held-out loss")
                return
            cd.append(hk - hb)
        cm, csd = statistics.fmean(cd), statistics.stdev(cd)
        ct = cm / (csd / math.sqrt(len(cd))) if csd > 0 else math.copysign(math.inf, cm)
        print(f"control {a.ctrl} - {a.base}: d {[round(x, 5) for x in cd]} mean {cm:+.5f} t {ct:.2f}")
        if not (cm > 0 and ct >= T_CRIT):
            print("QUALITY:   INCONCLUSIVE-quality: the positive control was not detected")
            print("OVERALL:   NOT CLEARED")
            return
        yard, ctrl_note = cm, f"measured in-run from {a.ctrl} (t {ct:.2f})"
    margin = 0.25 * yard
    m, sd = statistics.fmean(d), statistics.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else math.copysign(math.inf, m)
    if m > margin and t >= T_CRIT:
        qual = f"FAIL-quality: mean {m:+.5f} > margin {margin:.5f}, t {t:.2f}"
    elif abs(m) <= margin and not (m > 0 and t >= T_CRIT):
        qual = f"PASS-quality: |mean {m:+.5f}| <= {margin:.5f} (0.25 x yardstick {yard:.5f}, {ctrl_note}), t {t:.2f}"
    else:
        qual = f"INCONCLUSIVE-quality: mean {m:+.5f}, t {t:.2f}, margin {margin:.5f} ({ctrl_note})"
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
