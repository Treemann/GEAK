#!/usr/bin/env python3
"""verify_gate.py -- recompute a gate from the two leg DIRECTORIES, trusting no agent's numbers.

The orchestrator used to decide on the numbers the Integrator agent reported about legs it chose
itself: the summaries, the window-aligned comparison and the engagement claim (audit 2026-10-22,
ORCH-1). An agent that trimmed a field, picked the friendlier of two pairs, or misread a proof
produced the verdict. This script is the independent half: given ref and cand leg dirs it

  * re-summarises both probes with the CURRENT probe_summarize (merging the dispatcher-only fields
    occupancy / hang / status from each bench_summary.json),
  * runs compare_legs.compare() on them (window-aligned + paired),
  * runs the workflow's OWN gateVerdict on the result (scripts/gate_eval.js), and
  * reads the engagement proofs of the overlay(s) the candidate ADDS (proof subdirs present in the
    cand leg and absent from the ref leg): every rank must show forward AND backward calls, and every
    `fallback*` counter must be 0.

  verify_gate.py <ref_leg_dir> <cand_leg_dir> [--nproc 8] [--floor 0.25] [--kind kernel|config]
                 [--lever-flags "--flag value ..."]

Prints one JSON object. Exit 0 always when it could judge (the verdict is in the JSON); exit 2 when it
could not (missing dirs or probes) -- the caller must then fail closed.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_legs import compare  # noqa: E402
from probe_summarize import summarize  # noqa: E402

DISPATCHER_FIELDS = ("occupancy_verdict", "hang_detected", "status", "contamination_exhausted", "attempts",
                     "job_returncode", "overlay_install_failed", "profiled_run", "captured_run")


def _node():
    n = os.environ.get("NODE") or shutil.which("node")
    if n:
        return n
    c = sorted(glob.glob(os.path.expanduser("~/.cursor-server/bin/*/node")) +
               glob.glob(os.path.expanduser("~/.cursor-server/bin/*/*/node")))
    return c[0] if c else None


def _leg(d):
    s = summarize(os.path.join(d, "probe"))
    if s.get("error"):
        raise SystemExit(json.dumps({"error": f"{d}: {s['error']}"}))
    try:
        bs = json.load(open(os.path.join(d, "bench_summary.json")))
    except (OSError, ValueError):
        bs = {}
    for k in DISPATCHER_FIELDS:
        if k in bs:
            s[k] = bs[k]
    s["dir"], s["probe_dir"] = d, os.path.join(d, "probe")
    return s


def _proof_dirs(d):
    """Overlay proof subdirs of a leg: those holding engagement*.json.rank* files."""
    out = set()
    for f in glob.glob(os.path.join(d, "*", "engagement*.json.rank*")):
        out.add(os.path.basename(os.path.dirname(f)))
    return out


def _engagement(ref_dir, cand_dir, nproc):
    added = sorted(_proof_dirs(cand_dir) - _proof_dirs(ref_dir))
    res = {"overlays": {}, "added": added}
    if not added:
        res.update(forward=False, backward=False,
                   why="the candidate leg carries no overlay proof that the reference lacks")
        return res
    fwd_all = bwd_all = True
    for sub in added:
        o = {"ranks_generic": 0, "ranks_fused": 0, "fallback_total": 0, "forward": True, "backward": True}
        per_rank_f, per_rank_b = {}, {}
        for f in glob.glob(os.path.join(cand_dir, sub, "engagement.json.rank*")):
            r = json.load(open(f))
            rk = r.get("rank")
            o["ranks_generic"] += 1
            per_rank_f[rk] = per_rank_f.get(rk, 0) + int(r.get("forward") or 0)
            per_rank_b[rk] = per_rank_b.get(rk, 0) + int(r.get("backward") or 0)
        for f in glob.glob(os.path.join(cand_dir, sub, "engagement_fused.json.rank*")):
            r = json.load(open(f))
            rk = r.get("rank")
            o["ranks_fused"] += 1
            # the overlay's own proof: candidate_fwd/_bwd (or fwd/bwd) counts
            per_rank_f[rk] = per_rank_f.get(rk, 0) + int(r.get("candidate_fwd") or r.get("fwd") or 0)
            per_rank_b[rk] = per_rank_b.get(rk, 0) + int(r.get("candidate_bwd") or r.get("bwd") or 0)
            for k, v in r.items():
                if k.startswith("fallback") and isinstance(v, (int, float)) and v:
                    o["fallback_total"] += v
        o["forward"] = len(per_rank_f) >= nproc and all(v > 0 for v in per_rank_f.values())
        o["backward"] = len(per_rank_b) >= nproc and all(v > 0 for v in per_rank_b.values())
        if o["fallback_total"]:
            o["forward"] = o["backward"] = False
        res["overlays"][sub] = o
        fwd_all &= o["forward"]
        bwd_all &= o["backward"]
    res.update(forward=bool(fwd_all), backward=bool(bwd_all), ranks=nproc)
    return res


def _leg_dir(p):
    """Accept a leg dir or its probe/ dir, as a host path or a container /work/... path."""
    p = str(p).rstrip("/")
    if p.startswith("/work/"):
        p = os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"), p[len("/work/"):])
    if os.path.basename(p) == "probe":
        p = os.path.dirname(p)
    return os.path.abspath(p)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    ref_dir, cand_dir = _leg_dir(args[0]), _leg_dir(args[1])
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d   # noqa: E731
    nproc, floor = int(opt("--nproc", 8)), float(opt("--floor", 0.25))
    kind, lever = opt("--kind", "kernel"), opt("--lever-flags", "")
    for d in (ref_dir, cand_dir):
        if not os.path.isdir(os.path.join(d, "probe")):
            print(json.dumps({"error": f"no probe dir under {d}"}))
            sys.exit(2)
    ref, cand = _leg(ref_dir), _leg(cand_dir)
    cmp = compare(ref["probe_dir"], cand["probe_dir"])
    if kind == "kernel":
        allowed = ["GEAK_OVERLAY", "GEAK_OVERLAY_SHA"]
    else:
        allowed = [t.lstrip("-").split("=")[0] for t in lever.split() if t.startswith("--")]
        allowed += [k for k in {**ref.get("implementation", {}), **cand.get("implementation", {})} if k.startswith("NVTE_")]
    out = {"ref": ref, "cand": cand, "comparison": cmp, "kind": kind, "verdict": None}
    node = _node()
    if node and isinstance(cmp, dict) and not cmp.get("error"):
        spec = {"ref": ref, "cand": cand, "comparison": cmp, "mode": "fast", "allowedImpl": allowed,
                "args": {"across_job_floor_pct": floor}}
        r = subprocess.run([node, os.path.join(HERE, "gate_eval.js")], input=json.dumps(spec),
                           capture_output=True, text=True, timeout=60)
        out["verdict"] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[:400]}
    else:
        out["verdict"] = {"error": "no node to run gate_eval.js, or no comparison"}
    if kind == "kernel":
        out["engagement"] = _engagement(ref_dir, cand_dir, nproc)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
