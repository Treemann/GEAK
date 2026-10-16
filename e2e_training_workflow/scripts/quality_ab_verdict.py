#!/usr/bin/env python3
"""P0-4d verdicts (knowledge/gate_convergence.md, "P0-4d PRE-REGISTRATION"): quality and stability,
decided SEPARATELY. This fixes two flaws found in the P0-4b rules:
  * the instability rule was gated on the quality control, which calibrates a different thing;
  * the episode threshold (0.10) sat at the same-code null (0.109).

PV (progressive validation) = mean training loss over steps 201-358 of a single epoch, paired by seed
(arms of one seed see identical batches). Severity of a leg X against its seed's REF leg:
  max10 = max over 10-step windows after step 150 of mean(loss_X - loss_REF)
  severe if max10 > 0.25 (2.3x the same-code null), catastrophic if max10 > 1.0.
REF legs are scored against the S8 leg of the same seed with the sign reversed (symmetric).

Usage: quality_ab_verdict.py <runs/<name>_STAMP prefix> [--cand s8] [--ctrl ctrl70]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

PV_FROM, PV_TO = 201, 358
T_CRIT = 2.776                       # df 4
PASS_FRACTION = 0.25
SEVERE, CATASTROPHIC, AFTER, WIN = 0.25, 1.0, 150, 10


def steps(d):
    p = os.path.join(d, "probe", "probe_rank0.jsonl")
    return [r for r in (json.loads(l) for l in open(p)) if r.get("record") == "step"] if os.path.exists(p) else []


def max10(a, b):
    d = [x["loss"] - y["loss"] for x, y in zip(a, b)]
    return max(statistics.fmean(d[i:i + WIN]) for i in range(AFTER, len(d) - WIN + 1))


def paired(d):
    m, sd = statistics.fmean(d), statistics.stdev(d)
    return {"d": [round(x, 5) for x in d], "mean": round(m, 5), "sd": round(sd, 5),
            "t": round(m / (sd / math.sqrt(len(d))), 3) if sd > 0 else math.copysign(math.inf, m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--cand", default="s8")
    ap.add_argument("--ctrl", default="ctrl70")
    a = ap.parse_args()
    P = a.prefix.rstrip("/")
    seeds = sorted({int(m.group(1)) for f in glob.glob(P + "_ref_seed*") for m in [re.search(r"seed(\d+)$", f)] if m})
    S = {(arm, s): steps(f"{P}_{arm}_seed{s}") for arm in ("ref", a.cand, a.ctrl) for s in seeds}
    short = [f"{k}: {len(v)}" for k, v in S.items() if len(v) < PV_TO]
    if short:
        print("INCOMPLETE:", short)
        return
    pv = {k: statistics.fmean(x["loss"] for x in v[PV_FROM - 1:PV_TO]) for k, v in S.items()}
    sev = {}
    for s in seeds:
        r, c, k = S[("ref", s)], S[(a.cand, s)], S[(a.ctrl, s)]
        sev[(a.cand, s)], sev[("ref", s)], sev[(a.ctrl, s)] = max10(c, r), max10(r, c), max10(k, r)
        print(f"seed {s}: PV ref {pv[('ref', s)]:.4f} {a.cand} {pv[(a.cand, s)]:.4f} {a.ctrl} {pv[(a.ctrl, s)]:.4f} | "
              f"max10 {a.cand} {sev[(a.cand, s)]:.3f} ref(rev) {sev[('ref', s)]:.3f} {a.ctrl} {sev[(a.ctrl, s)]:.3f}")
    count = lambda arm, thr: sum(sev[(arm, s)] > thr for s in seeds)
    n_c, n_r, n_k = count(a.cand, SEVERE), count("ref", SEVERE), count(a.ctrl, SEVERE)
    cat_c, cat_null = count(a.cand, CATASTROPHIC), count("ref", CATASTROPHIC) + count(a.ctrl, CATASTROPHIC)
    if n_c == 0:
        stab = "PASS-stability: no severe leg"
    elif (n_c >= 2 and n_c - n_r >= 2) or (cat_c >= 1 and cat_null == 0):
        stab = f"FAIL-stability: {a.cand} severe {n_c}/{len(seeds)} (catastrophic {cat_c}) vs REF {n_r}, {a.ctrl} {n_k}"
    else:
        stab = f"INCONCLUSIVE-stability: {a.cand} severe {n_c} vs REF {n_r}, {a.ctrl} {n_k}"
    ctrl = paired([pv[(a.ctrl, s)] - pv[("ref", s)] for s in seeds])
    cand = paired([pv[(a.cand, s)] - pv[("ref", s)] for s in seeds])
    if ctrl["t"] < T_CRIT:
        qual = f"INCONCLUSIVE-quality: control not detected (t {ctrl['t']})"
    elif cand["mean"] > 0 and cand["t"] >= T_CRIT:
        qual = "FAIL-quality: significantly higher PV"
    elif abs(cand["mean"]) <= PASS_FRACTION * abs(ctrl["mean"]):
        qual = f"PASS-quality: |{cand['mean']}| <= {PASS_FRACTION} x |{ctrl['mean']}|"
    else:
        qual = f"INCONCLUSIVE-quality: {cand['mean']} vs control {ctrl['mean']}"
    cleared = stab.startswith("PASS") and qual.startswith("PASS")
    print("CTRL - REF (PV):", ctrl)
    print("CAND - REF (PV):", cand)
    print("STABILITY:", stab)
    print("QUALITY:  ", qual)
    print("OVERALL:  ", "CLEARED" if cleared else "NOT CLEARED")
    print(json.dumps({"seeds": seeds, "ctrl": ctrl, "cand": cand, "stability": stab, "quality": qual,
                      "cleared": cleared}))


if __name__ == "__main__":
    main()
