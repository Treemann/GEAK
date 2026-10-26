#!/usr/bin/env python3
"""rebuild_oracle.py -- rebuild a pre-v2 template oracle with the CURRENT build_task_dir, provably the same.

WHY (DECISIONS 4.60). A task dir built by build_task_dir before template_version 2 silently skips a
candidate's missing trailing gradient (the output-count / coverage / shape gate did not exist) and has
no 1-ulp rule. roles/kernel_extractor.md step 1b says "rebuild it before reuse"; this makes that one
deterministic command instead of a by-hand reconstruction of the original build arguments.

What is carried over from the OLD meta.json (nothing is guessed):
  reference      <- reference_implementation "baseline_src/reference_impl.py:<name>"
  const          <- layout.const
  name, notes    <- name, task_notes
  ulp_labels / ulp_floor / abs_floor_max_ulp / exact_labels <- acceptance
  holdouts       <- every holdouts[].tag, read from <capture_dir>/reference_io_<tag>.pt
  primary        <- --capture, or found in <capture_dir>: the capture whose rebuilt base_error equals
                    the old meta's (every label, every statistic, to 1e-9 relative)
PROOF OF SAMENESS: the rebuilt primary AND every holdout must reproduce the old base_error exactly,
or the rebuild is refused and the new dir is left as evidence. The old dir is never touched.

  rebuild_oracle.py <old_task_dir> <new_task_dir> --capture-dir <dir> [--capture <primary.pt>]
Runs where build_task_dir runs (inside the training image). Prints a JSON report; exit 0 = verified.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12 + 1e-9 * abs(b)
    return a == b


def build_args(old):
    ref = str(old.get("reference_implementation") or "")
    if ":" not in ref:
        raise SystemExit(f"old meta has no template reference_implementation ({ref!r}): a custom oracle, "
                         "not rebuildable by template -- confirm it scores every output instead")
    acc = old.get("acceptance") or {}
    return {"reference": ref.rsplit(":", 1)[1], "name": old.get("name"),
            "const": (old.get("layout") or {}).get("const"), "notes": old.get("task_notes"),
            "ulp_labels": acc.get("ulp_labels") or [], "ulp_floor": acc.get("ulp_floor") or None,
            "abs_floor_max_ulp": acc.get("abs_floor_max_ulp") or None,
            "exact_labels": acc.get("exact_labels") or []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_task")
    ap.add_argument("new_task")
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--capture", default=None)
    a = ap.parse_args()
    from build_task_dir import build
    old = json.load(open(os.path.join(a.old_task, "meta.json")))
    if (old.get("template_version") or 1) >= 2:
        print(json.dumps({"verified": True, "skipped": "already template_version >= 2"}))
        return 0
    if os.path.exists(a.new_task):
        raise SystemExit(f"{a.new_task} exists: rebuild into a FRESH dir")
    kw = build_args(old)
    hold = [(h["tag"], os.path.join(a.capture_dir, f"reference_io_{h['tag']}.pt")) for h in old.get("holdouts") or []]
    missing = [p for _, p in hold if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"holdout captures not found: {missing}")
    held = {p for _, p in hold}
    cands = ([a.capture] if a.capture else
             sorted(p for p in glob.glob(os.path.join(a.capture_dir, "reference_io*.pt")) if p not in held))
    report = {"old": os.path.abspath(a.old_task), "new": os.path.abspath(a.new_task), "args": kw,
              "holdouts": [t for t, _ in hold], "tried": []}
    for i, cap in enumerate(cands):
        tmp = a.new_task if i == len(cands) - 1 or a.capture else a.new_task + f".try{i}"
        # candidate_bind (how the candidate is bound / proven engaged) is carried over: it was dropped (audit G-10)
        build(cap, tmp, kw["reference"], kw["name"], kw["const"], kw["notes"], old.get("candidate_bind"), kw["ulp_labels"],
              hold, kw["ulp_floor"], kw["abs_floor_max_ulp"], kw["exact_labels"])
        new = json.load(open(os.path.join(tmp, "meta.json")))
        prim_ok = _close(new.get("base_error"), old.get("base_error"))
        report["tried"].append({"capture": cap, "primary_base_error_equal": prim_ok})
        if prim_ok:
            if tmp != a.new_task:
                os.rename(tmp, a.new_task)
            break
        if tmp != a.new_task:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        report["verified"] = False
        report["why"] = "no capture reproduces the old primary base_error"
        print(json.dumps(report, indent=1))
        return 1
    new = json.load(open(os.path.join(a.new_task, "meta.json")))
    oh = {h["tag"]: h.get("base_error") for h in old.get("holdouts") or []}
    nh = {h["tag"]: h.get("base_error") for h in new.get("holdouts") or []}
    report["holdout_base_error_equal"] = {t: _close(nh.get(t), oh[t]) for t in oh}
    # Files the old dir carried that the template does not generate (e.g. a harness_lib.py copied in
    # by the original builder): copied verbatim, and listed, so nothing the old oracle needed is lost.
    gen = set(os.listdir(a.new_task))
    extra = [f for f in os.listdir(a.old_task)
             if f not in gen and not f.endswith(".pt") and f != "__pycache__"]
    for f in extra:
        src = os.path.join(a.old_task, f)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(a.new_task, f))
    report["copied_from_old"] = extra
    report["template_version"] = new.get("template_version")
    report["verified"] = bool(all(report["holdout_base_error_equal"].values()))
    if not report["verified"]:
        report["why"] = "a holdout's rebuilt base_error differs from the old one"
    new["rebuilt_from"] = {"old_task": report["old"], "verified_same_base_error": report["verified"],
                           "note": "rebuilt by scripts/rebuild_oracle.py (DECISIONS 4.60)"}
    mp = os.path.join(a.new_task, "meta.json")
    os.chmod(mp, 0o644)
    json.dump(new, open(mp, "w"), indent=2)
    os.chmod(mp, 0o444)   # an oracle's meta is read-only, as build_task_dir leaves it (audit G-10)
    print(json.dumps(report, indent=1))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
