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
    `fallback*` counter must be 0;
  * checks that every overlay the REF leg runs is still engaged in the cand leg (`carried`), for
    kernel AND config gates: a lever that silently switches an accepted overlay off is refused.

  verify_gate.py <ref_leg_dir> <cand_leg_dir> [--nproc 8] [--floor PCT (default: per-stack)] [--kind kernel|config]
                 [--lever-flags "--flag value ..."] [--lever-env KNOB1,KNOB2]

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


def _proofs(d):
    """{slot: {rank: [proof dicts]}} for every engagement*.json.rank* under <leg>/<slot>/. Covers both
    layouts: Megatron (engagement.json.rank* = overlay_train wrapper, engagement_fused.json.rank* = the
    kernel's own proof) and HF (engagement_<slot>_cand.json.rank*, either kind)."""
    out = {}
    for f in glob.glob(os.path.join(d, "*", "engagement*.json.rank*")):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        rk = r.get("rank", f.rsplit("rank", 1)[-1])
        out.setdefault(os.path.basename(os.path.dirname(f)), {}).setdefault(str(rk), []).append(r)
    return out


def _identity(proofs_by_rank):
    """What overlay a slot runs: its replacement / manifest / kernel hashes, from any rank's proofs."""
    ids = set()
    for plist in proofs_by_rank.values():
        for r in plist:
            for k in ("replacement", "manifest", "kernel_sha256", "impl_sha256"):
                if k in r:
                    v = r[k]
                    if k in ("replacement", "manifest"):
                        v = os.path.basename(os.path.dirname(str(v).split(":")[0]))   # overlay dir name, not install path
                    elif isinstance(v, dict):
                        # Source hashes only: compiled artifacts (*.so) are rebuilt per install and their
                        # names carry a build key, so the SAME gated attention overlay read as 'changed'
                        # between the s7 and s8eps installs (2026-10-23).
                        v = {f: h for f, h in v.items() if not f.endswith(".so")}
                    ids.add((k, json.dumps(v, sort_keys=True)))
        break
    return ids


def _slot_engaged(plist_by_rank, nproc):
    """A slot is engaged iff on >= nproc ranks forward AND backward ran on the CANDIDATE path and no
    fallback counter is non-zero. The kernel's own counters (candidate_fwd/_bwd) are the evidence when
    present; the wrapper's forward/backward count only when they are the ONLY proof (an overlay whose
    impl IS the entry). Summing the two (2026-10-22 review MEG-M1) let a wrapper that delegated every
    call read as engaged."""
    o = {"ranks": len(plist_by_rank), "fallback_total": 0, "evidence": None}
    fwd, bwd = {}, {}
    for rk, plist in plist_by_rank.items():
        fused = [r for r in plist if "candidate_fwd" in r or "candidate_bwd" in r]
        use = fused or [r for r in plist if "forward" in r or "backward" in r]
        o["evidence"] = "candidate counters" if fused else "wrapper counters"
        fwd[rk] = sum(int(r.get("candidate_fwd", r.get("forward", 0)) or 0) for r in use)
        bwd[rk] = sum(int(r.get("candidate_bwd", r.get("backward", 0)) or 0) for r in use)
        for r in plist:
            for k, v in r.items():
                if k.startswith("fallback") and isinstance(v, (int, float)) and v:
                    o["fallback_total"] += v
    o["forward"] = len(fwd) >= nproc and all(v > 0 for v in fwd.values()) and not o["fallback_total"]
    o["backward"] = len(bwd) >= nproc and all(v > 0 for v in bwd.values()) and not o["fallback_total"]
    return o


def _engagement(ref_dir, cand_dir, nproc):
    rp, cp = _proofs(ref_dir), _proofs(cand_dir)
    # Judge every slot that is NEW in the cand, or whose overlay identity CHANGED (a candidate that
    # replaces an incumbent overlay in the same slot was always refused before).
    judged = sorted(s for s in cp if s not in rp or _identity(cp[s]) != _identity(rp[s]))
    res = {"overlays": {}, "added": judged}
    if not judged:
        res.update(forward=False, backward=False,
                   why="the candidate leg carries no overlay proof that is new or changed vs the reference "
                       "(expected <leg>/<slot>/engagement*.json.rank*)")
        return res
    for sub in judged:
        res["overlays"][sub] = _slot_engaged(cp[sub], nproc)
    res.update(forward=all(o["forward"] for o in res["overlays"].values()),
               backward=all(o["backward"] for o in res["overlays"].values()), ranks=nproc)
    return res


def _carried(ref_dir, cand_dir, nproc):
    """Every overlay slot the REFERENCE leg proves engaged must still be engaged in the candidate leg.

    WHY (DECISIONS 4.55). A config lever can switch an accepted overlay off without touching its bytes:
    `--rope-fusion` on s8 makes Megatron call the fused RoPE path, and the gated rope overlay then
    delegated all 3,360 calls per rank (fallback_fwd 3360, candidate_fwd 0). The pair still measured,
    and only the throughput sign showed something was wrong. A slot engaged in the ref and not in the
    cand is a silently removed accepted kernel, whatever the delta says."""
    rp, cp = _proofs(ref_dir), _proofs(cand_dir)
    out = {}
    for slot in sorted(rp):
        if not _slot_engaged(rp[slot], nproc)["forward"]:
            continue                       # the ref itself does not run it: nothing is carried
        e = _slot_engaged(cp.get(slot, {}), nproc)
        out[slot] = {"engaged": bool(e["forward"] and e["backward"]), "fallback_total": e["fallback_total"],
                     "ranks": e["ranks"]}
    lost = [k for k, v in out.items() if not v["engaged"]]
    return {"slots": out, "lost": lost, "ok": not lost}


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
    nproc, floor = int(opt("--nproc", 8)), opt("--floor", None)
    kind, lever = opt("--kind", "kernel"), opt("--lever-flags", "")
    # A lever that is a flag PLUS an env knob (e.g. --no-check-for-nan-in-loss-and-grad with
    # GEAK_NAN_FROM_GRAD_NORM=1, DECISIONS 4.62): the knob's names, comma-separated.
    lever_env = [k.strip() for k in opt("--lever-env", "").split(",") if k.strip()]
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
        allowed += ["env:" + k for k in lever_env]
        allowed += [k for k in {**ref.get("implementation", {}), **cand.get("implementation", {})} if k.startswith("NVTE_")]
    out = {"ref": ref, "cand": cand, "comparison": cmp, "kind": kind, "verdict": None}
    node = _node()
    if node and isinstance(cmp, dict) and not cmp.get("error"):
        # Without --floor the gate's own PER-STACK default applies (DECISIONS 4.54); the workflow
        # always passes the run's floor explicitly.
        stack = "hf_trl" if "variable" in (ref.get("step_shape"), cand.get("step_shape")) or \
            (ref.get("config") or {}).get("geak-stack") == "hf_trl" else "megatron"
        gargs = {"stack": stack}
        if floor is not None:
            gargs["across_job_floor_pct"] = float(floor)
        spec = {"ref": ref, "cand": cand, "comparison": cmp, "mode": "fast", "allowedImpl": allowed,
                "args": gargs}
        r = subprocess.run([node, os.path.join(HERE, "gate_eval.js")], input=json.dumps(spec),
                           capture_output=True, text=True, timeout=60)
        out["verdict"] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[:400]}
    else:
        out["verdict"] = {"error": "no node to run gate_eval.js, or no comparison"}
    if kind == "kernel":
        out["engagement"] = _engagement(ref_dir, cand_dir, nproc)
    out["carried"] = _carried(ref_dir, cand_dir, nproc)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
