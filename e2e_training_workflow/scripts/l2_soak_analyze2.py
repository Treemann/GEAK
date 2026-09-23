#!/usr/bin/env python3
"""L2 soak spec v2: two-sample comparison of MEANS (knowledge/gate_convergence.md).

v1 compared ONE candidate leg against the mean of the references and demanded 3x the max pairwise
reference spread. It failed its positive control, and the reason is structural rather than a bad
threshold: with a single candidate leg the comparison carries one full between-leg sigma no matter
how many reference legs are added, so the x1.004 bias -- worth 2.50 sigma -- was unreachable.

v2 replicates the candidate and tests means. Reporting order is fixed and unchanged:
    1. null control   (split the references in half; the test must NOT fire)
    2. positive control (must fire, or v2 has no demonstrated sensitivity and is advisory)
    3. the candidate.

Usage:
  l2_soak_analyze2.py --ref A B C --cand D E F G [--pos H I J K] [--alpha 0.05] [--warmup 12]
"""
import argparse
import itertools
import json
import math
import os
import statistics as st
import sys

# Two-sided Student t critical values at alpha = 0.05 / 0.01, by degrees of freedom. Table rather
# than scipy: this box's container has scipy but the analysis must run outside it too.
T_CRIT = {
    0.05: {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
           10: 2.23, 12: 2.18, 14: 2.14, 16: 2.12, 20: 2.09, 30: 2.04},
    0.01: {1: 63.66, 2: 9.92, 3: 5.84, 4: 4.60, 5: 4.03, 6: 3.71, 7: 3.50, 8: 3.36, 9: 3.25,
           10: 3.17, 12: 3.05, 14: 2.98, 16: 2.92, 20: 2.85, 30: 2.75},
}


def t_crit(df, alpha):
    tab = T_CRIT[alpha]
    for k in sorted(tab):
        if df <= k:
            return tab[k]
    return tab[max(tab)]


def series(run_dir, key="loss"):
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


def welch(a, b, alpha):
    """Welch t-test on two small samples. Returns (t, df, crit, fires)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return (float("inf"), na + nb - 2, t_crit(na + nb - 2, alpha), True)
    t = (st.mean(a) - st.mean(b)) / se
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den else na + nb - 2
    c = t_crit(int(round(df)), alpha)
    return (t, df, c, abs(t) > c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--pos", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=12)
    a = ap.parse_args()

    def terminal(d, n):
        s = series(d)[a.warmup:][:n]
        t = max(1, int(0.10 * len(s)))
        return sum(s[-t:]) / t

    def full_offset(d, n, base):
        s = series(d)[a.warmup:][:n]
        return sum(x - b for x, b in zip(s, base)) / len(s)

    allp = a.ref + a.cand + list(a.pos)
    n = min(len(series(d)[a.warmup:]) for d in allp)
    refs_s = [series(d)[a.warmup:][:n] for d in a.ref]
    base = [sum(c) / len(c) for c in zip(*refs_s)]
    print(f"steady window {n} steps (warmup {a.warmup} skipped); "
          f"{len(a.ref)} ref / {len(a.cand)} cand / {len(a.pos)} pos legs; alpha={a.alpha}\n")

    R = [terminal(d, n) for d in a.ref]
    C = [terminal(d, n) for d in a.cand]
    P = [terminal(d, n) for d in a.pos] if a.pos else []
    sigma = st.stdev(R) if len(R) > 1 else float("nan")
    print(f"terminal loss per leg (mean of last 10% of steps):")
    for lbl, vals, dirs in (("ref", R, a.ref), ("cand", C, a.cand), ("pos", P, a.pos)):
        for v, d in zip(vals, dirs):
            print(f"  {lbl:5s} {os.path.basename(d):28s} {v:.5f}")
    print(f"\n  between-leg sigma (references) = {sigma:.5f}\n")

    # ---- 1. NULL CONTROL: split the references and test them against each other -----------------
    print("=" * 74)
    print("1. NULL CONTROL -- splitting the references must NOT fire the test")
    print("=" * 74)
    null_fired = []
    for k in range(1, len(R)):
        for idx in itertools.combinations(range(len(R)), k):
            g1 = [R[i] for i in idx]
            g2 = [R[i] for i in range(len(R)) if i not in idx]
            if len(g1) < 2 or len(g2) < 2:
                continue
            res = welch(g1, g2, a.alpha)
            if res and res[3]:
                null_fired.append(idx)
    if not any(len([1 for i in range(len(R)) if i not in idx]) >= 2
               for k in range(1, len(R)) for idx in itertools.combinations(range(len(R)), k)):
        print(f"  not enough reference legs ({len(R)}) for a 2-vs-2 split; the reference spread is")
        print(f"  instead reported directly: max pairwise |diff| = "
              f"{max(abs(x - y) for x, y in itertools.combinations(R, 2)):.5f} "
              f"({max(abs(x - y) for x, y in itertools.combinations(R, 2)) / sigma:.2f} sigma)")
    print(f"  null control: {'FIRED -- criteria reject clean legs' if null_fired else 'did not fire (pass)'}\n")

    # ---- 2. POSITIVE CONTROL -------------------------------------------------------------------
    pos_fires = None
    if P:
        print("=" * 74)
        print("2. POSITIVE CONTROL -- the test must FIRE on a known-bad kernel")
        print("=" * 74)
        r = welch(P, R, a.alpha)
        pos_fires = r[3]
        print(f"  terminal loss: pos mean {st.mean(P):.5f} vs ref mean {st.mean(R):.5f}  "
              f"delta {st.mean(P) - st.mean(R):+.5f} ({abs(st.mean(P) - st.mean(R)) / sigma:.2f} sigma)")
        print(f"  Welch t = {r[0]:.3f}, df = {r[1]:.1f}, crit = {r[2]:.3f} -> "
              f"{'REJECTED as required' if pos_fires else 'NOT REJECTED -- no demonstrated sensitivity'}\n")

    # ---- 3. CANDIDATE --------------------------------------------------------------------------
    print("=" * 74)
    print("3. CANDIDATE")
    print("=" * 74)
    rc = welch(C, R, a.alpha)
    print(f"  terminal loss: cand mean {st.mean(C):.5f} vs ref mean {st.mean(R):.5f}  "
          f"delta {st.mean(C) - st.mean(R):+.5f} ({abs(st.mean(C) - st.mean(R)) / sigma:.2f} sigma)")
    print(f"  Welch t = {rc[0]:.3f}, df = {rc[1]:.1f}, crit = {rc[2]:.3f} -> "
          f"{'FAIL (differs from reference)' if rc[3] else 'PASS (not distinguishable from reference)'}")
    # secondary, reported only
    fo_c = [full_offset(d, n, base) for d in a.cand]
    fo_r = [full_offset(d, n, base) for d in a.ref]
    rs = welch(fo_c, fo_r, a.alpha)
    if rs:
        print(f"  [secondary, non-gating] full-window mean offset: t = {rs[0]:.3f}, "
              f"crit = {rs[2]:.3f} -> {'differs' if rs[3] else 'not distinguishable'}\n")

    print("=" * 74)
    advisory = bool(null_fired) or not P or not pos_fires
    if advisory:
        why = ("the null control fired" if null_fired else
               "no positive control was run" if not P else
               "the positive control was not rejected")
        print(f"VERDICT: ADVISORY ({why}). The candidate result is reported, not relied on.")
    else:
        print("VERDICT: v2 is ENTITLED TO SPEAK -- null control silent, positive control rejected.")
    print(f"CANDIDATE: {'FAIL' if rc[3] else 'PASS'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
