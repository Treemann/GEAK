#!/usr/bin/env python3
"""NULL CONTROL for L1's strict early sign rule on IMPLEMENTATION-CHANGE pairs (P1-2, 2026-10-15).

WHY. The strict rule (trajectory_gate.py, check 3) blocks when all >= 5 non-identical early diffs of
loss (or grad_norm) have one sign. Its false-reject rate, 5.1% (loss) / 0.6% (grad_norm), was
measured on 693 SAME-implementation pairs: two runs of one stack. But a gate never compares a stack
with itself. It compares an implementation with a DIFFERENT, correct implementation, and different
bf16 numerics decorrelate the chaotic trajectory faster than re-running the same code does. On
2026-10-13 cross-condition pairs of the packed stack fired 3/18 against 0/12 within condition
(knowledge/gate_convergence.md, "Stacked correct kernels"). This script measures that rate properly.

WHAT COUNTS AS A CORRECT IMPLEMENTATION. A leg qualifies when:
  * it is the packed 7B workload (config_digest 1ca93b7828f56395, argv hash cd0703efbd26ce3a);
  * it is a clean, unprofiled, uncaptured 30-step leg;
  * every overlay it binds is one of the ACCEPTED heads: each impl.py hash in its GEAK_OVERLAY_SHA
    must appear in the accepted s8 stack's GEAK_OVERLAY_SHA (--accepted-leg). The probe records
    manifest BASENAMES only, so overlay names cannot be read from a leg; hashes can. Those heads
    passed L0 and L1 and were re-measured stacked, so any pair of them is a correct-vs-correct
    comparison: a "true null" for a correctness detector.
A leg's implementation identity is GEAK_OVERLAY_SHA (the impl.py hashes). It is independent of
manifest basenames, which differ per leg in the older stack measurements.
Pairs with equal identity are SAME-implementation nulls. Pairs with different identities are
IMPLEMENTATION-CHANGE nulls.

THE RULE IS THE SHIPPED ONE. The diff construction and the strict test are copied from
trajectory_gate.verdict (check 3): EARLY_STEPS diffs, identical steps excluded, >= 5 required. The
L0 waiver is not applied, because none of these pairs carries an L0 "strictly better" verdict.

Usage: python3 scripts/controls/sign_rule_impl_null.py [--runs /home/treemann/geak_train/runs]
"""
import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trajectory_gate import EARLY_STEPS, _series  # noqa: E402

CONFIG_DIGEST = "1ca93b7828f56395"
ARGV_HASH = "cd0703efbd26ce3a"
def leg_identity(run, accepted):
    """(identity, overlay names) for a qualifying leg, or None."""
    try:
        b = json.load(open(os.path.join(run, "bench_summary.json")))
    except Exception:
        return None
    if (b.get("status") != "ok" or b.get("config_digest") != CONFIG_DIGEST
            or b.get("profiled_run") or b.get("captured_run")):
        return None
    occ = b.get("occupancy_verdict")
    if isinstance(occ, dict):
        occ = occ.get("verdict")
    if occ not in (None, "clean"):
        return None
    probe = os.path.join(run, "probe")
    try:
        s = _series(probe)
    except Exception:
        return None
    if s["argv_hash"] != ARGV_HASH or len(s["loss"]) != 30:
        return None
    cfg = s["config"] or {}
    shas = [x for x in str(cfg.get("GEAK_OVERLAY_SHA") or "").split("+") if x]
    if any(x not in accepted for x in shas):
        return None              # binds something that is not an accepted head
    return ("+".join(shas) or "bare"), tuple(shas), s


def strict_one_sided(rs, cs, key):
    d = [cs[key][i] - rs[key][i] for i in range(min(len(rs[key]), len(cs[key]), EARLY_STEPS))
         if rs[key][i] is not None and cs[key][i] is not None]
    if len(d) < 5:
        return None
    nz = [v for v in d if v != 0]
    return len(nz) >= 5 and (all(v > 0 for v in nz) or all(v < 0 for v in nz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/treemann/geak_train/runs")
    ap.add_argument("--accepted-leg", default="/home/treemann/geak_train/runs/baseline_td1015s8_1015_083212_leg1",
                    help="a leg of the full accepted stack; its GEAK_OVERLAY_SHA defines 'accepted'")
    a = ap.parse_args()
    acc_cfg = _series(os.path.join(a.accepted_leg, "probe"))["config"] or {}
    accepted = set(x for x in str(acc_cfg.get("GEAK_OVERLAY_SHA", "")).split("+") if x)
    if not accepted:
        sys.exit(f"no GEAK_OVERLAY_SHA on {a.accepted_leg}")
    print(f"accepted impl.py hashes ({len(accepted)}) from {os.path.basename(a.accepted_leg)}")
    legs = []
    for run in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if not os.path.isdir(run):
            continue
        r = leg_identity(run, accepted)
        if r:
            legs.append((os.path.basename(run),) + r)
    ids = {}
    for name, ident, names, _ in legs:
        ids.setdefault(ident, (len(ids), names))
    print(f"{len(legs)} qualifying legs, {len(ids)} distinct implementations:")
    for ident, (k, names) in sorted(ids.items(), key=lambda x: x[1][0]):
        n = sum(1 for l in legs if l[1] == ident)
        print(f"  impl#{k}: {n} legs, {len(names)} overlays")

    tally = {c: {"loss": [0, 0], "grad_norm": [0, 0]} for c in ("same", "change", "change_one")}
    fired = []
    # Ordered pairs (ref, cand): the rule is direction-symmetric, so unordered pairs suffice.
    for (n1, i1, h1, s1), (n2, i2, h2, s2) in itertools.combinations(legs, 2):
        cls = "same" if i1 == i2 else "change"
        # The gate's own shape: base stack vs base stack + ONE overlay (set difference of exactly 1).
        one = cls == "change" and len(set(h1) ^ set(h2)) == 1
        for key in ("loss", "grad_norm"):
            v = strict_one_sided(s1, s2, key)
            if v is None:
                continue
            for c in ((cls, "change_one") if one else (cls,)):
                tally[c][key][1] += 1
                if v:
                    tally[c][key][0] += 1
            if v and cls == "change":
                fired.append((key, n1, n2))
    print()
    for cls, label in (("same", "SAME-implementation pairs"), ("change", "IMPLEMENTATION-CHANGE pairs"),
                       ("change_one", "  of which differ by ONE overlay")):
        for key in ("loss", "grad_norm"):
            f, n = tally[cls][key]
            print(f"  {label:30s} {key:9s}: {f:4d}/{n:5d} fired = {100.0 * f / n if n else 0:5.1f}%")
    by_leg = {}
    for _, n1, n2 in fired:
        for n in (n1, n2):
            by_leg[n] = by_leg.get(n, 0) + 1
    if by_leg:
        print("\n  legs most often involved in an implementation-change firing:")
        for n, c in sorted(by_leg.items(), key=lambda x: -x[1])[:8]:
            print(f"    {c:3d}  {n}")
    print(json.dumps({"legs": len(legs), "implementations": len(ids), "tally": tally}))


if __name__ == "__main__":
    main()
