#!/usr/bin/env python3
"""Apply the P0-4b pre-registered verdicts (knowledge/gate_convergence.md) to quality_ab.sh legs.

PRIMARY METRIC -- progressive validation (PV): the mean training loss over steps 201-358. In a single
epoch every pack is seen once, so each step's loss (taken before that step's update) is a loss on
unseen data. Arms of one seed see identical batches, so PV is paired by seed.

Usage: quality_ab_pv.py <runs/qab_STAMP prefix>
"""
import glob
import json
import math
import os
import re
import statistics
import sys

PV_FROM, PV_TO = 201, 358          # inclusive, 1-based steps
T_CRIT = 2.776                      # two-sided 95%, df 4 (five paired seeds)
PASS_FRACTION = 0.25
EPISODE_AFTER, EPISODE_WIN, EPISODE_DELTA = 150, 10, 0.10
INSTABILITY_EXCESS = 4
VAL_RE = re.compile(r"validation loss at .*? lm loss value: ([0-9.Ee+-]+)")


def steps(d):
    p = os.path.join(d, "probe", "probe_rank0.jsonl")
    if not os.path.exists(p):
        return []
    return [r for r in (json.loads(l) for l in open(p)) if r.get("record") == "step"]


def training_path_ok(d, n):
    """Every overlay proof shows fwd > 0 and no fallback in training buckets 1..n (eval bucket exempt)."""
    proofs = glob.glob(os.path.join(d, "*", "engagement_fused*.json.rank*"))
    for f in proofs:
        for k, v in (json.load(open(f)).get("per_step") or {}).items():
            if isinstance(v, dict) and int(k) <= n:
                fb = sum(x for kk, x in v.items() if kk.startswith("fallback") and isinstance(x, int))
                fw = sum(x for kk, x in v.items() if kk.startswith("fwd") and isinstance(x, int))
                if fb or not fw:
                    return False, len(proofs)
    return True, len(proofs)


def has_episode(a, b):
    """Any EPISODE_WIN-step window after EPISODE_AFTER where mean(a - b) > EPISODE_DELTA."""
    d = [x["loss"] - y["loss"] for x, y in zip(a, b)]
    for i in range(EPISODE_AFTER, len(d) - EPISODE_WIN + 1):
        if statistics.fmean(d[i:i + EPISODE_WIN]) > EPISODE_DELTA:
            return i + 1
    return None


def paired(d):
    m = statistics.fmean(d)
    sd = statistics.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else math.copysign(math.inf, m)
    return {"d": [round(x, 5) for x in d], "mean": round(m, 5), "sd": round(sd, 5), "t": round(t, 3)}


def main():
    prefix = sys.argv[1].rstrip("/")
    seeds = sorted({int(m.group(1)) for f in glob.glob(prefix + "_*_seed*")
                    for m in [re.search(r"_seed(\d+)$", f)] if m})
    legs = {(a, s): f"{prefix}_{a}_seed{s}" for a in ("ref", "s8", "ctrl") for s in seeds}
    S = {k: steps(v) for k, v in legs.items()}
    short = [f"{a}/{s}: {len(v)} steps" for (a, s), v in S.items() if len(v) < PV_TO]
    if short:
        print("VERDICT: INCOMPLETE --", "; ".join(short))
        return
    for s in seeds:
        ok, n = training_path_ok(legs[("s8", s)], PV_TO)
        if not ok or n == 0:
            print(f"VERDICT: VOID -- S8 seed {s} did not run the stack in training ({n} proofs)")
            return
    pv = {k: statistics.fmean(x["loss"] for x in v[PV_FROM - 1:PV_TO]) for k, v in S.items()}
    for s in seeds:
        val = {}
        for a in ("ref", "s8", "ctrl"):
            m = VAL_RE.findall(open(os.path.join(legs[(a, s)], "train.log"), errors="replace").read())
            val[a] = float(m[-1]) if m else None
        gn = {a: sum(x["grad_norm"] > 10 for x in S[(a, s)][EPISODE_AFTER:]) for a in ("ref", "s8", "ctrl")}
        print(f"seed {s}: PV ref {pv[('ref', s)]:.4f} s8 {pv[('s8', s)]:.4f} ctrl {pv[('ctrl', s)]:.4f} | "
              f"held-out {val} | gn>10 after {EPISODE_AFTER}: {gn}")
    ctrl = paired([pv[("ctrl", s)] - pv[("ref", s)] for s in seeds])
    s8 = paired([pv[("s8", s)] - pv[("ref", s)] for s in seeds])
    ep_s8 = {s: has_episode(S[("s8", s)], S[("ref", s)]) for s in seeds}
    ep_ref = {s: has_episode(S[("ref", s)], S[("s8", s)]) for s in seeds}
    n_s8, n_ref = sum(v is not None for v in ep_s8.values()), sum(v is not None for v in ep_ref.values())
    print("CTRL - REF (PV):", ctrl)
    print("S8   - REF (PV):", s8)
    print(f"episodes (first step of a >{EPISODE_DELTA} {EPISODE_WIN}-step window): S8 {ep_s8}  REF {ep_ref}")
    if ctrl["t"] < T_CRIT:
        verdict = f"INCONCLUSIVE (underpowered): control t {ctrl['t']} < {T_CRIT}"
    elif s8["mean"] > 0 and s8["t"] >= T_CRIT:
        verdict = "S8 FAIL (quality): significantly higher progressive-validation loss"
    elif n_s8 - n_ref >= INSTABILITY_EXCESS:
        verdict = f"S8 FAIL (instability): {n_s8} S8 episodes vs {n_ref} REF"
    elif abs(s8["mean"]) <= PASS_FRACTION * abs(ctrl["mean"]):
        verdict = (f"S8 PASS: |{s8['mean']}| <= {PASS_FRACTION} x |{ctrl['mean']}|; "
                   f"episodes S8 {n_s8} vs REF {n_ref}")
    else:
        verdict = f"INCONCLUSIVE at the observed effect: S8 {s8['mean']} vs control {ctrl['mean']}"
    print("VERDICT:", verdict)
    print(json.dumps({"seeds": seeds, "ctrl": ctrl, "s8": s8, "episodes_s8": n_s8,
                      "episodes_ref": n_ref, "verdict": verdict}))


if __name__ == "__main__":
    main()
