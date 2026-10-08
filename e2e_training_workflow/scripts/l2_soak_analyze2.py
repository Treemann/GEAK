#!/usr/bin/env python3
"""L2 soak spec v3: two-sample comparison of MEANS, plus the power the design actually has.

v3 is v2's arithmetic with three things v2 lacked, all of which cost GPU hours by their absence:

  * the MINIMUM DETECTABLE EFFECT is printed BEFORE the results. v1 spent ~5 h on a design whose
    MDE was 0.0532 against an effect of 0.03742 -- underpowered by 42%, and no threshold could have
    rescued it. The number takes a line of arithmetic and was never computed. Now a run states its
    own power first, so an underpowered design is visible before its output is read.
  * a DISPERSION criterion, declared WEAKER ON PURPOSE. v2's control shifted spread (sd ratio 2.75,
    variance ratio 7.6) while the means coincided, so a location-only test could not see it. Since
    this criterion was motivated by looking at that data, it is not independent evidence when it is
    the only thing that fires, and the output says so rather than leaving the reader to know it.
  * --cand is OPTIONAL. A sensitivity experiment has no candidate: it runs references and positive
    controls to find out what the instrument can see. v2 could not express that and the question
    went unasked for two runs.

Original v2 header follows.

L2 soak spec v2: two-sample comparison of MEANS (knowledge/gate_convergence.md).

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
import math
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


# Upper-tail F at 0.025 (two-sided 0.05), indexed [df_num][df_den]. A table, not scipy: this script
# runs wherever a soak's output lands, and a missing dependency must not be the reason a 4-hour
# result cannot be read.
_F025 = {2: {2: 39.00, 3: 16.04, 4: 10.65, 5: 8.43, 6: 7.26},
         3: {2: 39.17, 3: 15.44, 4: 9.98, 5: 7.76, 6: 6.60},
         4: {2: 39.25, 3: 15.10, 4: 9.60, 5: 7.39, 6: 6.23},
         5: {2: 39.30, 3: 14.88, 4: 9.36, 5: 7.15, 6: 5.99}}


def fvar(p_grp, r_grp):
    """Variance-ratio test on terminal loss. SECONDARY and WEAKER BY DECLARATION.

    This criterion exists because v2's positive control moved DISPERSION, not location -- sd ratio
    2.75, variance ratio 7.6, with the means coinciding. It was therefore chosen after looking at
    the data it is meant to judge, which is exactly the move that invalidates a threshold. It is
    kept because a one-sided instrument that cannot see a real failure mode is worse, and it is
    labelled everywhere it appears so that a reader never mistakes it for independent evidence.
    """
    if len(p_grp) < 3 or len(r_grp) < 3:
        return None
    vp, vr = st.variance(p_grp), st.variance(r_grp)
    if vr <= 0:
        return None
    dn, dd = len(p_grp) - 1, len(r_grp) - 1
    crit = _F025.get(dn, {}).get(dd)
    if crit is None:
        return None
    f = vp / vr
    return f, dn, dd, crit, f > crit


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
    # Optional: a SENSITIVITY run has only references and positive controls. Requiring a candidate
    # is what made "what can this instrument actually see?" unaskable for two soaks.
    ap.add_argument("--cand", nargs="*", default=[])
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
    lens = {d: len(series(d)[a.warmup:]) for d in allp}
    n = min(lens.values())
    # REFUSE on a short or empty leg instead of computing from it. Found by pointing this script at
    # l2soak_0923_pos1, a truncated leg: the common window collapsed to 0 steps and it reported a
    # terminal loss of 0.00000 for every leg, an MDE of 0.00000, and only then died in a later
    # division. Every one of those numbers looked like a measurement. A soak costs hours; a soak
    # whose analyzer invents numbers from no data costs the hours AND the conclusion.
    MIN_STEPS = 50
    if n < MIN_STEPS:
        short = sorted((v, os.path.basename(k)) for k, v in lens.items() if v < MIN_STEPS)
        sys.exit(f"refusing to analyse: the common steady window is {n} steps (minimum {MIN_STEPS}).\n"
                 "  short or empty legs: " + ", ".join(f"{nm} ({v} steps)" for v, nm in short) +
                 "\n  Drop them from the arguments, or re-run them. Do NOT lower MIN_STEPS to make\n"
                 "  this pass: the window is shared by all legs, so a truncated leg silently\n"
                 "  truncates every comparison in the run.")
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
    print(f"\n  between-leg sigma (references) = {sigma:.5f}")

    # ---- POWER, STATED BEFORE THE RESULTS ------------------------------------------------------
    # v1 spent ~5 GPU-hours on a design that could not have detected the effect it was looking for.
    # Printing this first makes that visible without having to read the verdict and work backwards.
    def mde(n, m):
        if n < 1 or m < 2 or sigma != sigma:
            return float("nan")
        return t_crit(n + m - 2, a.alpha) * sigma * math.sqrt(1.0 / n + 1.0 / m)
    for lbl, grp in (("cand", C), ("pos", P)):
        if grp:
            d = mde(len(grp), len(R))
            print(f"  MDE ({lbl} n={len(grp)} vs ref m={len(R)}) = {d:.5f} terminal loss "
                  f"-- an effect smaller than this CANNOT be detected by this design")
    print()

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
        if r is None:
            # One positive leg carries no within-group variance, so no two-sample test exists. This
            # is not a tooling limitation to work around -- it IS v1's design, and saying so is the
            # point. v1 reported a number from this shape and the number was noise.
            print(f"  terminal loss: pos mean {st.mean(P):.5f} vs ref mean {st.mean(R):.5f}  "
                  f"delta {st.mean(P) - st.mean(R):+.5f} "
                  f"({abs(st.mean(P) - st.mean(R)) / sigma:.2f} sigma)")
            print(f"  NO TEST POSSIBLE: {len(P)} positive leg(s) gives no within-group variance.")
            print(f"  Quote the delta above as a single draw, never as an effect size.\n")
            pos_fires = False
        else:
            pos_fires = r[3]
            print(f"  terminal loss: pos mean {st.mean(P):.5f} vs ref mean {st.mean(R):.5f}  "
                  f"delta {st.mean(P) - st.mean(R):+.5f} "
                  f"({abs(st.mean(P) - st.mean(R)) / sigma:.2f} sigma)")
            print(f"  Welch t = {r[0]:.3f}, df = {r[1]:.1f}, crit = {r[2]:.3f} -> "
                  f"{'REJECTED as required' if pos_fires else 'NOT REJECTED -- no demonstrated sensitivity'}\n")

    # dispersion, reported alongside the positive control it was motivated by
    if P:
        fv = fvar(P, R)
        if fv:
            print(f"  [secondary, WEAKER BY DECLARATION -- motivated by v2's data, not independent] "
                  f"dispersion F = {fv[0]:.2f} ({fv[1]},{fv[2]}), crit {fv[3]:.2f} -> "
                  f"{'fires' if fv[4] else 'silent'}\n")

    # ---- 3. CANDIDATE --------------------------------------------------------------------------
    if not C:
        # A SENSITIVITY run: references and positive controls only, asking what the soak can see.
        print("=" * 74)
        print("3. CANDIDATE -- none supplied; this is a SENSITIVITY run, not a gating run")
        print("=" * 74)
        licensed = (not null_fired) and bool(P) and bool(pos_fires)
        print(f"  null control: {'FIRED' if null_fired else 'silent'}; "
              f"positive control: {'REJECTED' if pos_fires else 'not rejected'}")
        print("=" * 74)
        if licensed:
            print("VERDICT: the soak HAS demonstrated sensitivity to this control at this alpha.")
            print("         It may now be used to gate a candidate at this design and no other.")
        else:
            print("VERDICT: the soak has NO demonstrated sensitivity to this control at this alpha.")
            print("         Thresholds are NOT to be retuned to make it fire -- lower alpha or add")
            print("         legs, and re-register. See knowledge/gate_convergence.md, spec v3.")
        print("=" * 74)
        return 0
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
        print("VERDICT: v3 is ENTITLED TO SPEAK -- null control silent, positive control rejected.")
    print(f"CANDIDATE: {'FAIL' if rc[3] else 'PASS'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
