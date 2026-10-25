#!/usr/bin/env python3
"""quality_ab_final stability (DECISIONS 4.66): a seed whose HELD-OUT loss is worse by more than the yardstick is
severe, by more than 5 yardsticks catastrophic -- even when its training-loss max10 never spikes (P0-4n seed 4680:
held-out +0.273 at max10 0.145). Synthetic legs; no GPU."""
import json, os, subprocess, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
QF = os.path.join(HERE, "..", "quality_ab_final.py")
fails = 0
def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1
def leg(root, name, held, bump=0.0):
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "probe"))
    with open(os.path.join(d, "probe", "probe_rank0.jsonl"), "w") as f:
        for i in range(358):
            f.write(json.dumps({"record": "step", "step": i + 1, "loss": 1.5 + (bump if i > 300 else 0.0)}) + "\n")
    open(os.path.join(d, "train.log"), "w").write(f" validation loss at iteration 358 | lm loss value: {held:.6E} |\n")
def verdict(root, held_c, bump=0.0):
    for i, (s, hc) in enumerate(zip((11, 22, 33, 44, 55), held_c)):
        leg(root, f"q_ref_seed{s}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s}", 1.30 + 0.001 * i + hc, bump if s == 44 else 0.0)
    out = subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand"],
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])
for name, ds, bump, want in (
        ("clean", [0.0001, -0.0002, 0.0, 0.0001, -0.0001], 0.0, "PASS-stability: 0 severe"),
        ("one seed held-out +0.273, max10 0.145", [0.0001, -0.0002, 0.0, 0.273, -0.0001], 0.145, "FAIL-stability"),
        ("one seed held-out +0.03 (1.5 yardsticks)", [0.0001, -0.0002, 0.0, 0.03, -0.0001], 0.0, "PASS-stability: 1 severe"),
        ("two seeds held-out +0.03", [0.03, -0.0002, 0.0, 0.03, -0.0001], 0.0, "INCONCLUSIVE-stability: 2 severe")):
    root = tempfile.mkdtemp(prefix="qstab_")
    try:
        v = verdict(root, ds, bump)
        ok(v["stability"].startswith(want), f"{name}: {v['stability']}")
    finally:
        shutil.rmtree(root)
# Audit 2026-10-25 G-7 / ORCH-A8: EVERY outcome prints one JSON line with cleared derived from the strings, and a
# cand leg that stopped on a non-finite loss is FAIL-stability, not INCOMPLETE.
def run(root, *extra):
    out = subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand", *extra],
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])
root = tempfile.mkdtemp(prefix="qstab_")
try:
    for i, s in enumerate((11, 22, 33, 44, 55)):
        leg(root, f"q_ref_seed{s}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s}", 1.30 + 0.001 * i)
    d = os.path.join(root, "q_cand_seed33")
    with open(os.path.join(d, "probe", "probe_rank0.jsonl"), "w") as f:
        for i in range(120):
            f.write(json.dumps({"record": "step", "step": i + 1, "loss": 1.5 if i < 119 else float("nan")}) + "\n")
    open(os.path.join(d, "train.log"), "w").write(" iteration 120 | lm loss: nan |\n")
    with open(os.path.join(d, "probe", "probe_rank0.jsonl"), "a") as f:
        f.write('{"record": "step", "step": 121, "lo')          # killed mid-write
    v = run(root)
    ok(v["stability"].startswith("FAIL-stability") and "DIVERGED" in v["stability"] and v["cleared"] is False,
       f"a cand leg that stopped on NaN (probe line truncated) is FAIL-stability with JSON: {v['stability']}")
    open(os.path.join(d, "train.log"), "w").write(" iteration 120 | lm loss: 1.5 |\n")
    with open(os.path.join(d, "probe", "probe_rank0.jsonl"), "w") as f:
        for i in range(120):
            f.write(json.dumps({"record": "step", "step": i + 1, "loss": 1.5}) + "\n")
    v = run(root)
    ok(v["quality"].startswith("INCOMPLETE") and v["cleared"] is False,
       f"a cand leg still short of the end WITHOUT a non-finite loss stays INCOMPLETE, as JSON: {v['quality'][:40]}")
finally:
    shutil.rmtree(root)
root = tempfile.mkdtemp(prefix="qstab_")
try:
    v = run(root)
    ok(v["quality"].startswith("INCOMPLETE") and v["cleared"] is False, "no legs at all -> INCOMPLETE as JSON")
    for i, s in enumerate((11, 22, 33, 44, 55)):
        leg(root, f"q_ref_seed{s}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s}", 1.30 + 0.001 * i)
        leg(root, f"q_ctrl70_seed{s}", 1.30 + 0.001 * i + (0.001 if i % 2 else -0.001))
    v = run(root, "--ctrl", "ctrl70")
    ok(v["quality"].startswith("INCONCLUSIVE-quality: the positive control") and v["cleared"] is False,
       "an undetected control -> INCONCLUSIVE as JSON")
finally:
    shutil.rmtree(root)
print(f"\n{fails} FAILURE(S)" if fails else "\nall checks passed"); sys.exit(1 if fails else 0)
