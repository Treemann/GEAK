#!/usr/bin/env python3
"""Apply the PRE-REGISTERED L2 soak criteria (knowledge/gate_convergence.md).

The criteria were written down before any soak leg ran. This file only implements them; it does not
get to choose them. Order of reporting is fixed and deliberate:

    1. the NULL CONTROL   -- do the criteria pass a clean leg?
    2. the POSITIVE CONTROL -- do they reject a known-bad one?
    3. only then, the candidate.

If either control fails, the soak is ADVISORY and the candidate verdict is reported as such rather
than quietly re-interpreted. A criterion that rejects a clean leg is measuring chaos (that is exactly
how the L1 per-step envelope failed); a criterion that passes a known-bad kernel has no demonstrated
sensitivity, so a candidate PASS would mean nothing.

Usage:
  l2_soak_analyze.py --ref A B C --cand D [--pos E] [--k 3] [--warmup 12]
"""
import argparse
import json
import os
import statistics as st
import sys


def series(run_dir, key="loss"):
    """Loss/grad_norm are DP-reduced, so rank 0 carries the trajectory."""
    p = os.path.join(run_dir, "probe", "probe_rank0.jsonl")
    out = []
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("record") == "step":
                out.append(r.get(key))
    return out


def ols_slope(y):
    n = len(y)
    if n < 3:
        return 0.0
    xm = (n - 1) / 2.0
    ym = sum(y) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(y))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


def stats_vs(x, ref, tail_frac=0.10):
    """The three numeric criteria, all as x-relative-to-ref."""
    n = min(len(x), len(ref))
    x, ref = x[:n], ref[:n]
    d = [a - b for a, b in zip(x, ref)]
    t = max(1, int(tail_frac * n))
    return {
        "terminal_level": (sum(x[-t:]) / t) - (sum(ref[-t:]) / t),
        "drift_slope": ols_slope(d),
        "sign_balance": sum(1 for v in d if v > 0) / len(d),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--pos")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=12)
    a = ap.parse_args()

    if len(a.ref) < 3:
        print(f"WARNING: {len(a.ref)} reference legs; the spec asks for >=3. The spread S is then a "
              f"single sample and the criteria are not calibrated -- treat everything below as "
              f"ADVISORY regardless of the verdicts.\n")

    w = a.warmup
    refs = [series(d)[w:] for d in a.ref]
    cand = series(a.cand)[w:]
    pos = series(a.pos)[w:] if a.pos else None
    n = min([len(s) for s in refs] + [len(cand)] + ([len(pos)] if pos else []))
    refs = [s[:n] for s in refs]
    cand, pos = cand[:n], (pos[:n] if pos else None)
    ref_mean = [sum(col) / len(col) for col in zip(*refs)]
    print(f"steady window: {n} steps (warmup {w} skipped), {len(refs)} reference legs\n")

    # S = the spread of each statistic when NOTHING changed. Every threshold is K x this.
    pair = [stats_vs(refs[i], refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))]
    S = {
        "terminal_level": max(abs(p["terminal_level"]) for p in pair),
        "drift_slope": max(abs(p["drift_slope"]) for p in pair),
        "sign_balance": max(abs(p["sign_balance"] - 0.5) for p in pair),
    }
    print(f"reference-vs-reference spread S over {len(pair)} pair(s):")
    for k2, v in S.items():
        print(f"  {k2:16s} {v:.6g}")
    print(f"thresholds = K x S with K={a.k}\n")

    def judge(label, x):
        s = stats_vs(x, ref_mean)
        rows, failed = [], []
        for key, val, dev in (("terminal_level", s["terminal_level"], abs(s["terminal_level"])),
                              ("drift_slope", s["drift_slope"], abs(s["drift_slope"])),
                              ("sign_balance", s["sign_balance"], abs(s["sign_balance"] - 0.5))):
            lim = a.k * S[key]
            bad = dev > lim
            if bad:
                failed.append(key)
            rows.append(f"  {key:16s} {val:+.6g}   |dev|={dev:.6g}  limit={lim:.6g}  "
                        f"{'FAIL' if bad else 'pass'}")
        print(f"{label}:")
        print("\n".join(rows))
        print(f"  -> {'FAIL on ' + ', '.join(failed) if failed else 'PASS on all numeric criteria'}\n")
        return failed

    # ---- 1. NULL CONTROL: each clean ref judged against the mean of the others -------------------
    print("=" * 72)
    print("1. NULL CONTROL -- the criteria must PASS a leg where nothing changed")
    print("=" * 72)
    null_bad = []
    for i, r in enumerate(refs):
        others = [s for j, s in enumerate(refs) if j != i]
        om = [sum(c) / len(c) for c in zip(*others)]
        s = stats_vs(r, om)
        f = []
        for key, dev in (("terminal_level", abs(s["terminal_level"])),
                         ("drift_slope", abs(s["drift_slope"])),
                         ("sign_balance", abs(s["sign_balance"] - 0.5))):
            if dev > a.k * S[key]:
                f.append(key)
        print(f"  {os.path.basename(a.ref[i]):28s} {'FAIL: ' + ','.join(f) if f else 'pass'}")
        null_bad += f
    print(f"\n  null control: {'FAILED' if null_bad else 'PASSED'}"
          f"{' -- the criteria reject a clean leg, so they measure chaos' if null_bad else ''}")
    print("  NOTE: mildly self-fitting -- each held-out leg also contributed to S. Passing is "
          "necessary, not sufficient.\n")

    # ---- 2. POSITIVE CONTROL --------------------------------------------------------------------
    pos_bad = None
    if pos is not None:
        print("=" * 72)
        print("2. POSITIVE CONTROL -- the criteria must REJECT a known-bad kernel")
        print("=" * 72)
        pos_bad = judge("  positive control (deliberate gradient bias)", pos)
        print(f"  positive control: {'REJECTED as required' if pos_bad else 'NOT REJECTED'}"
              f"{'' if pos_bad else ' -- the soak has demonstrated NO sensitivity'}\n")

    # ---- 3. the candidate, last ----------------------------------------------------------------
    print("=" * 72)
    print("3. CANDIDATE")
    print("=" * 72)
    cand_bad = judge("  candidate (accepted overlay)", cand)

    advisory = bool(null_bad) or (pos is not None and not pos_bad) or pos is None
    print("=" * 72)
    if advisory:
        why = []
        if null_bad:
            why.append("the null control failed")
        if pos is None:
            why.append("no positive control was run")
        elif not pos_bad:
            why.append("the positive control was not rejected")
        print(f"VERDICT: ADVISORY ({'; '.join(why)}). The candidate result below is reported, "
              f"not relied on.")
    else:
        print("VERDICT: the soak is ENTITLED TO SPEAK (null control passed, positive control "
              "rejected).")
    print(f"CANDIDATE: {'FAIL on ' + ', '.join(cand_bad) if cand_bad else 'PASS'}")
    print("=" * 72)
    return 1 if (cand_bad and not advisory) else 0


if __name__ == "__main__":
    sys.exit(main())
