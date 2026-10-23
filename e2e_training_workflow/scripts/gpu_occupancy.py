"""Occupancy guard — the training analogue of the inference layer's isolation discipline.

The inference layer gets isolation from measuring both legs sequentially in one session on a box it
owns. A training node is routinely shared (this one carries ~20 unrelated containers), and a
training A/B needs EVERY GPU, so a neighbour that starts mid-leg silently drags one leg down and
inflates or destroys the delta. There is no way to prevent that from inside the workflow; what is
possible is to make it a DETECTED FACT rather than a silent bias. Sample before and after each leg;
a leg whose occupancy changed is marked contaminated and must be re-measured, not quietly reported.

`rocm-smi --showpids` lists KFD processes. Ours are identified by the PIDs we launched, so anything
else holding a GPU is a neighbour.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys


def kfd_pids():
    try:
        out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception as e:
        return {"available": False, "error": repr(e), "pids": []}
    if "No KFD PIDs currently running" in out:
        return {"available": True, "pids": []}
    pids = []
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)\s+\S+", line)
        if m and m.group(1) != "0":
            pids.append(int(m.group(1)))
    return {"available": True, "pids": sorted(set(pids))}


def sample(tag: str):
    s = kfd_pids()
    s["tag"] = tag
    return s


def mid_foreign(mid, nproc):
    """Foreign PIDs among samples taken WHILE our job held the GPUs. rocm-smi reports host PIDs and the
    job runs in a container, so our ranks cannot be named; they are recognised by persistence instead.
    A PID in fewer than half of the samples came and went inside the leg -> foreign. If more PIDs than
    our `nproc` ranks + 1 persist, a neighbour sat on the box the whole leg -> all persistent ones are
    reported (we cannot tell which), and the leg is contaminated either way. The +1: an 8-rank HF job
    was measured holding 9 KFD processes (its launcher opens the device too); a neighbour present the
    whole leg is in any case usually also caught by the before/after samples."""
    if not mid:
        return []
    seen = {}
    for s in mid:
        for p in set(s.get("pids", [])):
            seen[p] = seen.get(p, 0) + 1
    persistent = {p for p, c in seen.items() if 2 * c >= len(mid)}
    transient = set(seen) - persistent
    return sorted(transient | (persistent if nproc and len(persistent) > nproc + 1 else set()))


def compare(before: dict, after: dict, own_pids=(), mid=(), nproc=0):
    """`mid`: samples taken WHILE the job ran (bench_train, every 60 s). Before/after alone miss a
    neighbour that comes and goes inside a leg (review: occupancy guard sampled only at the ends)."""
    own = set(int(p) for p in own_pids)
    b = set(before.get("pids", [])) - own
    a = set(after.get("pids", [])) - own
    m = set(mid_foreign(list(mid), int(nproc or 0))) - own
    verdict = "clean"
    if not before.get("available") or not after.get("available"):
        verdict = "unknown"
    elif b or a or m:
        verdict = "contaminated"
    return {
        "verdict": verdict,
        "foreign_pids_before": sorted(b),
        "foreign_pids_after": sorted(a),
        "foreign_pids_during": sorted(m),
        "mid_samples": len(mid),
        "note": {
            "clean": "no foreign GPU process observed around this leg",
            "contaminated": "another process held a GPU during this leg; the number is not trustworthy",
            "unknown": "rocm-smi unavailable; isolation could not be established",
        }[verdict],
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        before, after = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
        own = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []
        mid = []
        if len(sys.argv) > 5 and sys.argv[5]:
            try:
                mid = [json.loads(l) for l in open(sys.argv[5]) if l.strip()]
            except OSError:
                mid = []
        nproc = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else 0
        print(json.dumps(compare(before, after, own, mid, nproc), indent=2))
    else:
        out = sample(sys.argv[1] if len(sys.argv) > 1 else "sample")
        # `mid` samples are appended one per line by bench_train's in-leg sampler
        print(json.dumps(out) if out.get("tag") == "mid" else json.dumps(out, indent=2))
