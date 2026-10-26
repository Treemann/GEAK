"""l0_verify.py -- the gate verifier's OWN L0: re-run the frozen oracle on the shipped overlay, then derive the
L1-plan evidence from the outputs, not from an agent's transcript.

WHY (audit 2026-10-25 residual of G-3 / DECISIONS 4.77). L0 pass, `out_bit_identical` and `per_output` were
copied by the Convergence Auditor from runs it chose. They decide whether a kernel is accepted and which L1
floor applies (`overlay` 0.05% vs `kernel_swap` 0.25% vs strictly-better 2.0%). This recomputes all three:

  * correct     -- the task's own unittest.py (template or custom), run with the overlay's kernel files as
                   kernel_src, last stdout JSON line;
  * per_output  -- template oracles: {label: {cand_err: its max_abs vs the fp32 reference, base_err: the
                   ORIGINAL op's max_abs from meta.base_error}} on the primary case;
  * out_bit_identical -- template oracles: every FORWARD output (layout.out_labels, default ["out"]) bitwise
                   equal to the captured ORIGINAL op's output (reference_io `ref`, the golden) on the primary
                   AND every holdout. Custom oracles report null for both evidence fields (the orchestrator
                   then takes the conservative plan).

  l0_verify.py --task <task dir> --overlay <overlay dir> --out result.json
Runs inside the training image on one GPU (l0_verify.sh). exit 0 = correct, 1 = not correct, 2 = could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

SKIP = {"__pycache__", "build", ".torch_ext", "mgc_ext"}


def stage(task, overlay):
    """A private run dir: the task's files linked, the overlay's kernel files copied in as kernel_src."""
    run = tempfile.mkdtemp(prefix="l0v_")
    for f in os.listdir(task):
        if f in ("kernel_src",) or f in SKIP:
            continue
        os.symlink(os.path.join(task, f), os.path.join(run, f))
    src = os.path.join(overlay, "kernel_src") if os.path.isdir(os.path.join(overlay, "kernel_src")) else overlay
    shutil.copytree(src, os.path.join(run, "kernel_src"),
                    ignore=lambda d, names: [n for n in names if n in SKIP or (n.startswith("m_") and n.endswith(".json"))
                                             or n.startswith("engagement") or n == "impl.py"])
    return run


def run_oracle(run):
    """The oracle's own verdict. Run from the PARENT dir by path: a script named unittest.py in sys.path[0]
    (or the cwd for -c) shadows the stdlib module torch imports."""
    code = ("import runpy,sys; sys.argv=[{p!r}]; runpy.run_path({p!r}, run_name='__main__')"
            .format(p=os.path.join(run, "unittest.py")))
    p = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(run), capture_output=True, text=True,
                       timeout=3600)
    last = None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith("{") and '"correct"' in s:
            try:
                last = json.loads(s)
            except ValueError:
                pass
    return last, p.returncode, (p.stdout[-3000:] + p.stderr[-2000:])


PAIRED = r'''
import importlib.util, json, os, sys
import torch
run = sys.argv[1]
spec = importlib.util.spec_from_file_location("l0v_ut", os.path.join(run, "unittest.py"))
ut = importlib.util.module_from_spec(spec); spec.loader.exec_module(ut)
sys.path.insert(0, os.path.join(run, "kernel_src"))
import kernel
call = ut.make_call(kernel.entry)
n_fwd = len((ut.META.get("layout") or {}).get("out_labels") or ["out"])
labels = ut.META["signature"]["outputs"]
files = ["reference_io.pt"] + [h["file"] for h in ut.META.get("holdouts", [])]
res = {}
for f in files:
    case = next(iter(ut.h.iter_eager_cases_from_oracle(os.path.join(run, f), device="cuda")))
    gold = case.get("ref")
    if gold is None:
        res[f] = None; continue
    got = call(case["args"])
    eq = []
    for a, g in list(zip(got, gold))[:n_fwd]:
        eq.append(bool(a is not None and g is not None and tuple(a.shape) == tuple(g.shape)
                       and torch.equal(a.detach().to(g.dtype), g.detach().to(a.device))))
    res[f] = {"forward_labels": labels[:n_fwd], "bit_identical": all(eq) if eq else None}
print("L0V_PAIRED " + json.dumps(res))
'''


def paired(run):
    p = subprocess.run([sys.executable, "-c", PAIRED, run], cwd=os.path.dirname(run), capture_output=True,
                       text=True, timeout=3600)
    for line in p.stdout.splitlines():
        if line.startswith("L0V_PAIRED "):
            return json.loads(line[len("L0V_PAIRED "):]), None
    return None, (p.stdout[-1500:] + p.stderr[-1500:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.task, "meta.json")))
    template = (meta.get("template_version") or 0) >= 2 and "make_call" in open(os.path.join(a.task, "unittest.py")).read()
    res = {"task": os.path.abspath(a.task), "overlay": os.path.abspath(a.overlay),
           "oracle": {"kind": "template" if template else "custom", "template_version": meta.get("template_version"),
                      "task_dir": os.path.abspath(a.task)},
           "correct": None, "out_bit_identical": None, "per_output": None}
    run = stage(a.task, a.overlay)
    try:
        verdict, rc, tail = run_oracle(run)
        if verdict is None:
            res.update(error="the oracle printed no verdict", tail=tail)
            json.dump(res, open(a.out, "w"), indent=1)
            return 2
        res["correct"] = bool(verdict.get("correct"))
        if template:
            rep, base = verdict.get("l0") or {}, meta.get("base_error") or {}
            po = {}
            for label in meta["signature"]["outputs"]:
                r, b = rep.get(label) or {}, base.get(label) or {}
                if isinstance(r.get("max_abs"), (int, float)) and isinstance(b.get("max_abs"), (int, float)):
                    po[label] = {"cand_err": r["max_abs"], "base_err": b["max_abs"]}
            res["per_output"] = po or None
            pr, perr = paired(run)
            res["paired"] = pr
            if pr and all(v is not None for v in pr.values()):
                res["out_bit_identical"] = all(v["bit_identical"] is True for v in pr.values())
            elif perr:
                res["paired_error"] = perr
    finally:
        shutil.rmtree(run, ignore_errors=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
