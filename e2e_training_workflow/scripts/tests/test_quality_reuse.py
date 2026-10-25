#!/usr/bin/env python3
"""quality_reuse.py check must see what the realized argument table cannot (audit 2026-10-25 ID-3 / HFC-1).

The first version compared train.log's realized table only. A bare REF reused for an hf8-stacked cand
differs in the overlay content and env knobs, neither of which is an argv value -- and it passed. These
synthetic legs pin the rules: the overlay hash is always compared (absent = no overlay); an env knob an
older snapshot could not record is unverifiable and needs --accept-unknown-harness; a leg without a
harness record fails unless that is accepted; a different harness always fails; a ctrl arm is checked
against the base.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "quality_reuse.py")
BASE_ARGS = {f"arg_{i}": str(i) for i in range(12)}
HARNESS = {"rocm_compat.py": "a", "overlay_train.py": "b", "hf_sft.py": "c", "image_id": "d"}


def leg(root, name, args=None, cfg=None, harness=HARNESS, heldout=True, env=None):
    d = os.path.join(root, "runs", name)
    os.makedirs(os.path.join(d, "probe"))
    with open(os.path.join(d, "train.log"), "w") as f:
        for k, v in {**BASE_ARGS, **(args or {})}.items():
            f.write(f"  {k} ....... {v}\n")
        if heldout:
            f.write(" validation loss at iteration 358 on validation set | lm loss value: 1.29E+00 |\n")
    hdr = {"record": "header", "config": {"snapshot_keys": "v6", **(cfg or {})}}
    if harness is not None:
        hdr["harness"] = harness
    if env is not False:
        hdr["coverage"] = {"v": 1, "argv": {}, "env": env or {}}
    with open(os.path.join(d, "probe", "probe_rank0.jsonl"), "w") as f:
        f.write(json.dumps(hdr) + "\n")


def check(root, *extra):
    p = subprocess.run([sys.executable, "-B", TOOL, "check", "--prefix", "t", "--base", "ref", "--cand", "cand",
                        "--seeds", "1", *extra], capture_output=True, text=True,
                       env={**os.environ, "GEAK_WORK_DIR": root})
    return json.loads(p.stdout)


def case(label, ref, cand, extra=(), expect=True, ctrl=None):
    with tempfile.TemporaryDirectory() as root:
        leg(root, "t_ref_seed1", **ref)
        leg(root, "t_cand_seed1", **cand)
        if ctrl is not None:
            leg(root, "t_ctrl70_seed1", **ctrl)
        r = check(root, *extra)
    ok = r["valid"] is expect
    print(f"  {'ok' if ok else 'FAIL'}: {label}" + ("" if ok else f" -> {r['seeds'][0]['why']}"))
    return ok


SHA = {"GEAK_OVERLAY_SHA": "abc"}
OLD = {"snapshot_keys": "v4"}
results = [
    case("identical legs are valid", {}, {}),
    case("bare REF vs stacked cand is INVALID (the audit counterexample)", {}, {"cfg": SHA}, expect=False),
    case("... even with --accept-unknown-harness", {"cfg": OLD}, {"cfg": SHA},
         ("--accept-unknown-harness",), expect=False),
    case("... and valid only when the overlay is the declared change", {}, {"cfg": SHA},
         ("--allow-impl", "GEAK_OVERLAY_SHA")),
    case("an env knob differing on same-schema legs is invalid", {"cfg": {"env:X": "0"}}, {"cfg": {"env:X": "1"}},
         expect=False),
    case("an env knob an older (v4) snapshot could not record is unverifiable without the flag",
         {"cfg": OLD}, {"cfg": {"env:X": "1"}}, expect=False),
    case("... and accepted with --accept-unknown-harness", {"cfg": OLD}, {"cfg": {"env:X": "1"}},
         ("--accept-unknown-harness",)),
    case("a realized argument outside --allow is invalid", {}, {"args": {"arg_3": "9"}}, expect=False),
    case("... and valid when allowed", {}, {"args": {"arg_3": "9"}}, ("--allow", "arg_3")),
    case("a leg without a harness record fails by default", {"harness": None}, {}, expect=False),
    case("... and passes with --accept-unknown-harness", {"harness": None}, {}, ("--accept-unknown-harness",)),
    case("a different harness always fails", {}, {"harness": {**HARNESS, "image_id": "z"}},
         ("--accept-unknown-harness",), expect=False),
    case("a leg without a held-out evaluation fails", {}, {"heldout": False}, expect=False),
    case("an unclassified env knob differing (NCCL_ALGO Ring vs Tree) is INVALID", {"env": {"NCCL_ALGO": "Ring"}},
         {"env": {"NCCL_ALGO": "Tree"}}, expect=False),
    case("... and valid when pre-declared", {"env": {"NCCL_ALGO": "Ring"}}, {"env": {"NCCL_ALGO": "Tree"}},
         ("--allow-impl", "env NCCL_ALGO")),
    case("a leg without the full env capture fails by default", {"env": False}, {}, expect=False),
    case("... and passes with --accept-unknown-harness", {"env": False}, {}, ("--accept-unknown-harness",)),
    case("a ctrl arm that is the base + lr is valid", {}, {}, ("--ctrl", "ctrl70"),
         ctrl={"args": {"lr": "7e-6"}}),
    case("a ctrl arm on another implementation is INVALID", {}, {}, ("--ctrl", "ctrl70"),
         ctrl={"args": {"lr": "7e-6"}, "cfg": SHA}, expect=False),
]
if not all(results):
    print(f"{results.count(False)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
