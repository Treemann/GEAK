#!/usr/bin/env python3
"""P0-4e verdicts (knowledge/gate_convergence.md, "P0-4e PRE-REGISTRATION").

Under the standard LR schedule (warmup + cosine decay) the end-of-training held-out loss is the primary
quality metric: with the LR decayed, the final weights settle and REF's seed spread is ~0.005, while a
x0.7 schedule moved it +0.0206 (P0-4d, all five seeds positive). That +0.0206 is the FIXED yardstick.

  quality   PASS iff |mean d| <= 0.25 x 0.0206 (= 0.00515) and NOT (mean d > 0 and t >= 2.776)
            FAIL iff mean d > 0.00515 and t >= 2.776;  otherwise INCONCLUSIVE
            d_k = held-out(S8)_k - held-out(REF)_k, paired by seed, n = 5
  stability FAIL iff any S8 leg is catastrophic: max10 > 1.0, or its held-out d > 5 x yardstick
            PASS iff no catastrophic and at most 1 S8 leg severe: max10 > 0.25, or held-out d > 1 x yardstick
            otherwise INCONCLUSIVE.  max10: largest 10-step mean of loss(S8) - loss(REF) after step 150.
            (the held-out arm of both thresholds added 2026-10-24, DECISIONS 4.66)
  CLEARED iff both PASS.

RULES. `--rule point` (above) is the PRE-REGISTERED rule of every outcome test up to 2026-10-22 and
reproduces their verdicts exactly. It is a point-estimate check: a stack that truly degrades held-out
loss by the full margin still clears ~25-49% of the time (audit 2026-10-22 G3). `--rule noninf`, the
DEFAULT from 2026-10-22 (DECISIONS 4.44), is a one-sided non-inferiority test on the same paired d:
  quality   PASS iff mean d + t(0.95, n-1) x sd/sqrt(n) <= margin   (the upper 95% bound is inside)
            FAIL iff mean d > margin and t >= t(0.975, n-1)          (as before)
            otherwise INCONCLUSIVE
It caps the chance of clearing a stack that is exactly one margin worse at ~5%, at the price of more
INCONCLUSIVE verdicts for noisy runs (add seeds). The output prints both rules' verdicts.

Usage: quality_ab_final.py <prefix> [<prefix> ...] [--cand s8] [--rule noninf|point]
  The LAST stdout line is always one JSON object with quality / stability / cleared (cleared derived from
  the two strings); a cand leg that stopped on a non-finite loss is FAIL-stability, not INCOMPLETE.
  Several prefixes are judged as ONE sample (a pre-registered seed extension: the first run's prefix
  plus the extension's). A seed found under two prefixes is refused -- it would be counted twice.
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
# Per-seed held-out thresholds, in units of the yardstick (DECISIONS 4.66).
HELDOUT_SEVERE_Y, HELDOUT_CATASTROPHIC_Y = 1.0, 5.0
T_CRIT = 2.776   # two-sided 95% t, df 4: the pre-registered 5-seed protocol
# The critical value for OTHER seed counts (review 2026-10-20). It was a constant: a 3-seed run judged
# with 2.776 instead of 4.303 is anti-conservative, and nothing checked the count. df -> t(0.975, df).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086}


_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
        10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740,
        18: 1.734, 19: 1.729, 20: 1.725}


def t_one_sided(n):
    """One-sided 95% critical t for n paired seeds (df = n - 1); the normal 1.645 past the table."""
    return _T95.get(n - 1, 1.645)


def t_crit(n):
    """Two-sided 95% critical t for n paired seeds (df = n - 1); the normal 1.96 past the table."""
    return _T975.get(n - 1, 1.96)
SEVERE, CATASTROPHIC, AFTER, WIN = 0.25, 1.0, 150, 10
VAL_RE = re.compile(r"validation loss at .*? lm loss value: ([0-9.Ee+-]+)")


def steps(d):
    """Step records of rank 0. A leg killed mid-write (a crash, a divergence exit) leaves a truncated last
    line: skip unparsable lines instead of crashing with no verdict (found on real legs, 2026-10-25)."""
    p = os.path.join(d, "probe", "probe_rank0.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("record") == "step":
            out.append(r)
    return out


# A cand leg that STOPPED early because training diverged (non-finite loss; Megatron's NaN check /
# rerun state machine exiting) is the worst stability outcome, not an incomplete run (audit 2026-10-25
# G-7: it printed INCOMPLETE and no JSON, so a caller could not tell it from a leg still running).
DIVERGED_RE = re.compile(r"(lm loss: (nan|inf|-inf)\b|loss is (nan|inf)|NaN detected|found NaN|"
                         r"Unexpected result nan|rerun_state_machine.*(nan|inf|NaN|Inf))", re.IGNORECASE)


def diverged(d, st):
    if any(not math.isfinite(float(r.get("loss", 0.0) or 0.0)) for r in st):
        return True
    try:
        return bool(DIVERGED_RE.search(open(os.path.join(d, "train.log"), errors="replace").read()))
    except FileNotFoundError:
        return False


def _host(p):
    """A container /work/... prefix (agents mix them) -> the host path under GEAK_WORK_DIR."""
    return os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"), p[len("/work/"):]) \
        if p.startswith("/work/") else p


def emit(quality, stability, **kw):
    """EVERY outcome ends in one JSON line (the orchestrator recomputes this verdict, ORCH-A8), and
    `cleared` is DERIVED from the two verdict strings -- never set on its own."""
    cleared = quality.startswith("PASS") and stability.startswith("PASS")
    print("QUALITY:  ", quality)
    print("STABILITY:", stability)
    print("OVERALL:  ", "CLEARED" if cleared else "NOT CLEARED")
    print(json.dumps({**kw, "quality": quality, "stability": stability, "cleared": cleared}))


def heldout(d):
    try:
        v = VAL_RE.findall(open(os.path.join(d, "train.log"), errors="replace").read())
        return float(v[-1]) if v else None
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="+")
    ap.add_argument("--cand", default="s8")
    ap.add_argument("--base", default="ref", help="the comparison arm (e.g. s8 when testing a lever ON the stack)")
    ap.add_argument("--ctrl", default=None,
                    help="measure the yardstick IN THIS RUN from a positive-control arm (e.g. ctrl70) instead of "
                         "the fixed P0-4d value; the control must then be DETECTED (mean > 0, t >= 2.776)")
    ap.add_argument("--rule", choices=("noninf", "point"), default="noninf",
                    help="noninf (default since 2026-10-22): upper one-sided 95%% bound of mean d <= margin; "
                         "point: the pre-registered rule of the outcome tests up to 2026-10-22")
    a = ap.parse_args()
    prefixes = [_host(p.rstrip("/")) for p in a.prefix]
    where = {}
    for P in prefixes:
        for f in glob.glob(P + f"_{a.base}_seed*"):
            m = re.search(r"seed(\d+)$", f)
            if m:
                s_ = int(m.group(1))
                if s_ in where and where[s_] != P:
                    emit(f"REFUSED-quality: seed {s_} appears under {where[s_]} and {P}; a seed may be counted once",
                         "REFUSED-stability: not judged")
                    return
                where[s_] = P
    seeds = sorted(where)
    if len(seeds) < 3:
        # An empty glob used to crash in statistics.fmean; one or two seeds cannot carry a t-test.
        emit(f"INCOMPLETE-quality: {len(seeds)} seed(s) found for {prefixes}_{a.base}_seed* -- the protocol "
             f"needs >= 3 (pre-registered: 5)", "INCOMPLETE-stability: not judged", seeds=seeds)
        return
    tc = t_crit(len(seeds))
    if len(seeds) != 5:
        print(f"NOTE: {len(seeds)} seeds, not the pre-registered 5 -- t critical = {tc} (df {len(seeds) - 1})")
    rows, d, sev = [], [], []
    for s in seeds:
        P = where[s]
        r, c = f"{P}_{a.base}_seed{s}", f"{P}_{a.cand}_seed{s}"
        sr, sc = steps(r), steps(c)
        hr, hc = heldout(r), heldout(c)
        # Complete = both legs ran the same number of steps, enough to cover the episode window after
        # step AFTER (358 on the Megatron protocol, 296 on HF's one-epoch protocol), and both evaluated.
        if len(sr) != len(sc) or len(sr) < AFTER + WIN + 1 or hr is None or hc is None:
            if len(sc) < len(sr) and hc is None and diverged(c, sc):
                emit(f"INCONCLUSIVE-quality: seed {s} {a.cand} did not finish",
                     f"FAIL-stability: seed {s} {a.cand} DIVERGED at step {len(sc)} (non-finite loss / NaN exit)",
                     seeds=seeds, diverged_seed=s)
                return
            emit(f"INCOMPLETE-quality: seed {s} ref {len(sr)} steps / {hr}, cand {len(sc)} steps / {hc}",
                 "INCOMPLETE-stability: not judged", seeds=seeds)
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
            hb, hk = heldout(f"{where[s]}_{a.base}_seed{s}"), heldout(f"{where[s]}_{a.ctrl}_seed{s}")
            if hk is None:
                emit(f"INCOMPLETE-quality: control seed {s} has no held-out loss", "INCOMPLETE-stability: not judged",
                     seeds=seeds)
                return
            cd.append(hk - hb)
        cm, csd = statistics.fmean(cd), statistics.stdev(cd)
        ct = cm / (csd / math.sqrt(len(cd))) if csd > 0 else math.copysign(math.inf, cm)
        print(f"control {a.ctrl} - {a.base}: d {[round(x, 5) for x in cd]} mean {cm:+.5f} t {ct:.2f}")
        if not (cm > 0 and ct >= tc):
            emit("INCONCLUSIVE-quality: the positive control was not detected", "NOT-JUDGED-stability: the "
                 "control failed first", seeds=seeds, ctrl_d=[round(x, 6) for x in cd])
            return
        yard, ctrl_note = cm, f"measured in-run from {a.ctrl} (t {ct:.2f})"
    margin = 0.25 * yard
    m, sd = statistics.fmean(d), statistics.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else math.copysign(math.inf, m)
    upper = m + t_one_sided(len(d)) * sd / math.sqrt(len(d))
    def _point():
        if m > margin and t >= tc:
            return f"FAIL-quality: mean {m:+.5f} > margin {margin:.5f}, t {t:.2f}"
        if abs(m) <= margin and not (m > 0 and t >= tc):
            return f"PASS-quality: |mean {m:+.5f}| <= {margin:.5f} (0.25 x yardstick {yard:.5f}, {ctrl_note}), t {t:.2f}"
        return f"INCONCLUSIVE-quality: mean {m:+.5f}, t {t:.2f}, margin {margin:.5f} ({ctrl_note})"
    def _noninf():
        if m > margin and t >= tc:
            return f"FAIL-quality: mean {m:+.5f} > margin {margin:.5f}, t {t:.2f}"
        if upper <= margin:
            return (f"PASS-quality: upper 95% bound {upper:+.5f} <= margin {margin:.5f} "
                    f"(mean {m:+.5f}, 0.25 x yardstick {yard:.5f}, {ctrl_note})")
        return (f"INCONCLUSIVE-quality: upper 95% bound {upper:+.5f} > margin {margin:.5f} "
                f"(mean {m:+.5f}, t {t:.2f}, {ctrl_note}) -- add seeds")
    qual = _noninf() if a.rule == "noninf" else _point()
    other = _point() if a.rule == "noninf" else _noninf()
    print(f"RULE: {a.rule}   (the other rule would say: {other.split(':')[0]})")
    # A seed is also severe / catastrophic by its HELD-OUT loss alone (DECISIONS 4.66). max10 looks for
    # SPIKES in the training-loss difference; a sustained moderate divergence slips under it: P0-4n seed 4680
    # trained to a held-out +0.273 (13x the yardstick) at max10 0.145. Thresholds in yardstick units, so
    # they hold for HF's in-run yardstick too: severe = one seed worse than the whole x0.7-LR recipe
    # change; catastrophic = 5x that. Calibrated on every outcome test on disk: no cleared test has a seed
    # above +0.0112 (0.54 yardstick); every seed above 1 yardstick belongs to a test already NOT CLEARED.
    d_sev = [x > HELDOUT_SEVERE_Y * yard for x in d]
    d_cat = [x > HELDOUT_CATASTROPHIC_Y * yard for x in d]
    n_cat = sum((x > CATASTROPHIC) or c for x, c in zip(sev, d_cat))
    n_sev = sum((x > SEVERE) or ds for x, ds in zip(sev, d_sev))
    if n_cat:
        stab = f"FAIL-stability: {n_cat} catastrophic leg(s)"
    elif n_sev <= 1:
        stab = f"PASS-stability: {n_sev} severe, 0 catastrophic"
    else:
        stab = f"INCONCLUSIVE-stability: {n_sev} severe, 0 catastrophic"
    emit(qual, stab, rule=a.rule, upper95=round(upper, 6), seeds=seeds, d=[round(x, 6) for x in d],
         mean=round(m, 6), t=round(t, 3), max10=[round(x, 3) for x in sev], heldout_severe=d_sev,
         heldout_catastrophic=d_cat, yardstick=round(yard, 6))


if __name__ == "__main__":
    main()
