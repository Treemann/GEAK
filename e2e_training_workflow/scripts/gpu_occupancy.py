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


def compare(before: dict, after: dict, own_pids=()):
    own = set(int(p) for p in own_pids)
    b = set(before.get("pids", [])) - own
    a = set(after.get("pids", [])) - own
    verdict = "clean"
    if not before.get("available") or not after.get("available"):
        verdict = "unknown"
    elif b or a:
        verdict = "contaminated"
    return {
        "verdict": verdict,
        "foreign_pids_before": sorted(b),
        "foreign_pids_after": sorted(a),
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
        print(json.dumps(compare(before, after, own), indent=2))
    else:
        print(json.dumps(sample(sys.argv[1] if len(sys.argv) > 1 else "sample"), indent=2))
