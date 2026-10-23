#!/usr/bin/env python3
"""quality_reuse.py -- reuse an earlier outcome test's legs as an arm of a new one, and PROVE it is valid.

WHY (review P3, DECISIONS 4.61/4.63). An outcome test of "cleared stack + one change" re-runs the cleared
stack's legs (and the positive control's) on the same seeds, recipe and data -- ~2/3 of its GPU time
spent re-measuring what an earlier outcome test already measured. Reusing them is sound only if the
reused legs trained exactly the recipe the new cand legs train, apart from the change under test. This
tool makes that a checked fact rather than an assertion:

  link   --old-prefix P --old-arm A --new-prefix N --new-arm B --seeds 1111,2222,...
         symlinks P_A_seed<s> -> N_B_seed<s> (refuses a missing leg or an existing target)
  check  --prefix N --base B --cand C --seeds ... [--allow key1,key2]
         per seed, the framework's REALIZED argument table (train.log) of the base leg and the cand leg
         must be equal except for paths / run names and the --allow keys (the change under test); both
         legs must carry a held-out evaluation. Exit 1 on any other difference.

Megatron and HF (hf_sft.py prints the same '  key ....... value' dump) are both covered.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

RUNS = os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"), "runs")
DUMP = re.compile(r"^\s{2}([a-z0-9_]+) \.{3,}\s(.*)$")
# Per-leg paths and names: they differ between any two legs and say nothing about the recipe.
PATHISH = {"output_dir", "logging_dir", "run_name", "save", "load_dir", "tensorboard_dir", "wandb_exp_name"}
# Arguments ADDED to a launcher after older legs were run, with the value those older legs effectively
# ran (the launcher's default then and now). Absent on one side == this value on the other is no
# difference. Anything else absent on one side IS a difference.
ADDED_LATER = {"hf_average_tokens_across_devices": "true", "hf_balance_tokens": "off"}
EVAL = re.compile(r"validation loss at .*? lm loss value: ([0-9.Ee+-]+)")


def _leg(prefix, arm, seed):
    return os.path.join(RUNS, f"{os.path.basename(prefix)}_{arm}_seed{seed}")


def realized(d):
    t = {}
    with open(os.path.join(d, "train.log"), errors="replace") as f:
        for i, line in enumerate(f):
            if i > 8000:
                break
            m = DUMP.match(line.rstrip("\n"))
            if m:
                t[m.group(1)] = m.group(2).strip()
    return t


def heldout(d):
    try:
        v = EVAL.findall(open(os.path.join(d, "train.log"), errors="replace").read())
        return float(v[-1]) if v else None
    except OSError:
        return None


def cmd_link(a):
    out = []
    for s in a.seeds.split(","):
        src, dst = _leg(a.old_prefix, a.old_arm, s), _leg(a.new_prefix, a.new_arm, s)
        if not os.path.isdir(src):
            sys.exit(f"missing reused leg {src}")
        if os.path.lexists(dst):
            sys.exit(f"{dst} exists; refusing to overwrite")
        os.symlink(os.path.realpath(src), dst)
        out.append({"seed": s, "link": dst, "to": os.path.realpath(src)})
    print(json.dumps({"linked": out}, indent=1))


def cmd_check(a):
    allow = {k.strip().replace("-", "_") for k in (a.allow or "").split(",") if k.strip()}
    rep, ok = [], True
    for s in a.seeds.split(","):
        b, c = _leg(a.prefix, a.base, s), _leg(a.prefix, a.cand, s)
        tb, tc = realized(b), realized(c)
        if len(tb) < 10 or len(tc) < 10:
            rep.append({"seed": s, "ok": False, "why": "no realized argument table in a train.log"})
            ok = False
            continue
        val = lambda t, k: t.get(k, ADDED_LATER.get(k))  # noqa: E731
        diff = {k: (val(tb, k), val(tc, k)) for k in sorted(set(tb) | set(tc))
                if val(tb, k) != val(tc, k) and k not in PATHISH}
        bad = {k: v for k, v in diff.items() if k not in allow}
        hb, hc = heldout(b), heldout(c)
        r = {"seed": s, "differs_allowed": {k: v for k, v in diff.items() if k in allow}, "differs_other": bad,
             "heldout": [hb, hc], "ok": not bad and hb is not None and hc is not None}
        if hb is None or hc is None:
            r["why"] = "a leg has no held-out evaluation"
        ok = ok and r["ok"]
        rep.append(r)
    print(json.dumps({"valid": ok, "seeds": rep}, indent=1))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("link")
    l.add_argument("--old-prefix", required=True); l.add_argument("--old-arm", required=True)
    l.add_argument("--new-prefix", required=True); l.add_argument("--new-arm", required=True)
    l.add_argument("--seeds", required=True)
    c = sub.add_parser("check")
    c.add_argument("--prefix", required=True); c.add_argument("--base", required=True)
    c.add_argument("--cand", required=True); c.add_argument("--seeds", required=True)
    c.add_argument("--allow", default="")
    a = ap.parse_args()
    if a.cmd == "link":
        cmd_link(a)
        return 0
    return cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
