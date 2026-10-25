#!/usr/bin/env python3
"""quality_reuse.py -- reuse an earlier outcome test's legs as an arm of a new one, and PROVE it is valid.

WHY (review P3, DECISIONS 4.61/4.63). An outcome test of "cleared stack + one change" re-runs the cleared
stack's legs (and the positive control's) on the same seeds, recipe and data -- ~2/3 of its GPU time
spent re-measuring what an earlier outcome test already measured. Reusing them is sound only if the
reused legs trained exactly the recipe the new cand legs train, apart from the change under test. This
tool makes that a checked fact rather than an assertion:

  link   --old-prefix P --old-arm A --new-prefix N --new-arm B --seeds 1111,2222,...
         symlinks P_A_seed<s> -> N_B_seed<s> (refuses a missing leg or an existing target)
  check  --prefix N --base B --cand C --seeds ... [--allow k1,k2] [--allow-impl K1,K2] [--ctrl C]
         [--ctrl-allow lr,min_lr] [--accept-unknown-harness]
         per seed (audit 2026-10-25 ID-3 / HFC-1: the realized table alone cannot see overlays or env):
           * REALIZED argument table (train.log, normalised as probe_summarize does) equal except paths
             and --allow keys;
           * IMPLEMENTATION from the probe header -- overlay content hash (GEAK_OVERLAY_SHA), NVTE_* and
             env: knobs -- equal except --allow-impl keys;
           * HARNESS (rocm_compat / overlay_train / hf_sft / image) equal; a leg without a harness record
             fails unless --accept-unknown-harness (recorded);
           * both legs carry a held-out evaluation; with --ctrl, the control arm passes the same checks
             against the base with --ctrl-allow. Exit 1 on any failure.

Megatron and HF (hf_sft.py prints the same '  key ....... value' dump) are both covered.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_summarize import _norm, is_impl_key, snapshot_schema  # noqa: E402  (the ONE realized-value normaliser / impl classifier)

RUNS = os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"), "runs")
# Runtime code + image that must match between a reused leg and the new one (as comparable(), 4.59 / 4.67).
HARNESS_KEYS = ("rocm_compat.py", "overlay_train.py", "hf_sft.py", "image_id")
DUMP = re.compile(r"^\s{2}([a-z0-9_]+) \.{3,}\s(.*)$")
# Per-leg paths and names: they differ between any two legs and say nothing about the recipe.
PATHISH = {"output_dir", "logging_dir", "run_name", "save", "load_dir", "tensorboard_dir", "wandb_exp_name"}
# Arguments ADDED to a launcher after older legs were run, with the value those older legs effectively
# ran (the launcher's default then and now). Absent on one side == this value on the other is no
# difference. Anything else absent on one side IS a difference.
ADDED_LATER = {"hf_average_tokens_across_devices": "true", "hf_balance_tokens": "off",
               "hf_train_metrics": "on"}   # before 2026-10-18 TRL's per-micro-batch metrics always ran
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
                t[m.group(1)] = str(_norm(m.group(2)))   # strips spliced log tags, normalises numbers
    return t


def header(d):
    """rank-0 probe header: the captured config (workload + implementation) and the harness record."""
    try:
        with open(os.path.join(d, "probe", "probe_rank0.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("record") == "header":
                    return r
    except (OSError, ValueError):
        return None
    return None


def impl_of(h):
    """The implementation half that the realized table CANNOT see: overlay identity and env knobs."""
    cfg = (h or {}).get("config") or {}
    return {k: v for k, v in cfg.items() if is_impl_key(k) and k != "GEAK_OVERLAY"}   # label, not content


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


def _compare(b, c, allow, allow_impl, accept_unknown_harness):
    """One pair of legs: realized table, implementation (overlay / env), harness, held-out present."""
    tb, tc = realized(b), realized(c)
    r = {"base": os.path.basename(b), "cand": os.path.basename(c), "why": []}
    if len(tb) < 10 or len(tc) < 10:
        r["why"].append("no realized argument table in a train.log")
    val = lambda t, k: t.get(k, ADDED_LATER.get(k))  # noqa: E731
    diff = {k: (val(tb, k), val(tc, k)) for k in sorted(set(tb) | set(tc))
            if val(tb, k) != val(tc, k) and k not in PATHISH}
    r["differs_allowed"] = {k: v for k, v in diff.items() if k in allow}
    r["differs_other"] = {k: v for k, v in diff.items() if k not in allow}
    hb, hc = header(b), header(c)
    if not hb or not hc:
        r["why"].append("a leg has no probe header")
    ib, ic = impl_of(hb), impl_of(hc)
    # SCHEMA-AWARE: a key the OLDER snapshot did not record yet (v5 / v6 additions such as
    # ddp-average-in-collective, rope-fusion, hf-train-metrics) is not a difference -- its value is in
    # the realized table, compared above. Env knobs (env:*) and the overlay hash are NOT in the realized
    # table: absent on an older leg they are unverifiable, accepted only with --accept-unknown-harness.
    sb, sc = snapshot_schema((hb or {}).get("config")), snapshot_schema((hc or {}).get("config"))
    older = ib if sb < sc else ic if sc < sb else None
    unrecorded, unverifiable = [], []
    idiff = {}
    for k in sorted(set(ib) | set(ic)):
        if ib.get(k) == ic.get(k):
            continue
        if k.startswith("GEAK_OVERLAY"):
            pass   # recorded on every schema >= 2: absent = NO overlay, a real difference (bare vs stacked)
        elif k.startswith("env:"):
            # env knobs are recorded from snapshot v5 (audit OVL-2); on an older leg they are unknown
            if older is not None and k not in older and min(sb, sc) < 5:
                unverifiable.append(k)
        elif older is not None and k not in older:
            unrecorded.append(k)   # an argv key the older snapshot predates: the realized table decides
            continue
        idiff[k] = (ib.get(k), ic.get(k))
    r["schemas"], r["impl_unrecorded_on_older"] = [sb, sc], unrecorded
    if unverifiable:
        r["impl_unverifiable"] = unverifiable
        if accept_unknown_harness:
            idiff = {k: v for k, v in idiff.items() if k not in unverifiable}
    r["impl_differs_allowed"] = {k: v for k, v in idiff.items() if k in allow_impl}
    r["impl_differs_other"] = {k: v for k, v in idiff.items() if k not in allow_impl}
    gb, gc = (hb or {}).get("harness"), (hc or {}).get("harness")
    if gb and gc:
        r["harness_differs"] = {k: (gb.get(k), gc.get(k)) for k in HARNESS_KEYS if gb.get(k) != gc.get(k)}
        r["harness_unverified"] = False
    else:
        r["harness_differs"], r["harness_unverified"] = {}, True
        if not accept_unknown_harness:
            r["why"].append("a leg predates the harness record (runtime code / image unknown); "
                            "pass --accept-unknown-harness to accept that explicitly")
    r["heldout"] = [heldout(b), heldout(c)]
    if None in r["heldout"]:
        r["why"].append("a leg has no held-out evaluation")
    if r["differs_other"]:
        r["why"].append("realized arguments differ beyond --allow")
    if r["impl_differs_other"]:
        r["why"].append("implementation (overlay / env knobs) differs beyond --allow-impl")
    if r["harness_differs"]:
        r["why"].append("runtime code / image differ")
    r["ok"] = not r["why"]
    return r


def cmd_check(a):
    norm = lambda x: {k.strip().replace("-", "_") for k in (x or "").split(",") if k.strip()}  # noqa: E731
    allow, ctrl_allow = norm(a.allow), norm(a.ctrl_allow)
    allow_impl = {k.strip() for k in (a.allow_impl or "").split(",") if k.strip()}
    rep, ok = [], True
    for s in a.seeds.split(","):
        r = _compare(_leg(a.prefix, a.base, s), _leg(a.prefix, a.cand, s), allow, allow_impl,
                     a.accept_unknown_harness)
        r["seed"] = s
        # The positive control is reused too: it must be the base's recipe apart from ITS declared change
        # (e.g. lr / min_lr), and run the base's implementation (review: the ctrl arm was never checked).
        if a.ctrl:
            rc = _compare(_leg(a.prefix, a.base, s), _leg(a.prefix, a.ctrl, s), ctrl_allow, set(),
                          a.accept_unknown_harness)
            r["ctrl"] = rc
            if not rc["ok"]:
                r["ok"] = False
                r["why"] = r["why"] + [f"ctrl arm: {w}" for w in rc["why"]]
        ok = ok and r["ok"]
        rep.append(r)
    print(json.dumps({"valid": ok, "seeds": rep, "allow": sorted(allow), "allow_impl": sorted(allow_impl),
                      "ctrl": a.ctrl, "ctrl_allow": sorted(ctrl_allow),
                      "accept_unknown_harness": bool(a.accept_unknown_harness)}, indent=1))
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
    c.add_argument("--allow", default="", help="realized-argument keys the change under test may move")
    c.add_argument("--allow-impl", default="", help="implementation keys it may move, e.g. GEAK_OVERLAY_SHA, "
                   "env:GEAK_NAN_FROM_GRAD_NORM, no-check-for-nan-in-loss-and-grad")
    c.add_argument("--ctrl", default=None, help="also check this control arm against the base")
    c.add_argument("--ctrl-allow", default="lr,min_lr", help="realized keys the control may move")
    c.add_argument("--accept-unknown-harness", action="store_true",
                   help="accept legs that predate the harness record (recorded in the output)")
    a = ap.parse_args()
    if a.cmd == "link":
        cmd_link(a)
        return 0
    return cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
