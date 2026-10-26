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
def leg(root, name, held, bump=0.0, cfg=None, content=None, nsteps=358, nan_exit=False):
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "probe"))
    seed = name.rsplit("seed", 1)[-1]
    with open(os.path.join(d, "probe", "probe_rank0.jsonl"), "w") as f:
        f.write(json.dumps({"record": "header", "config": {"snapshot_keys": "v6", "seed": seed, "lr": "1.0e-5",
                                                           **(cfg or {})}, **({"content": content} if content else {})}) + "\n")
        for i in range(nsteps):
            f.write(json.dumps({"record": "step", "step": i + 1, "loss": 1.5 + (bump if i > 300 else 0.0)}) + "\n")
    if nan_exit:
        open(os.path.join(d, "train.log"), "w").write(" iteration 200 | lm loss: nan |\n")
    else:
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
        # DECISIONS 4.86: ONE catastrophic leg is within the recipe's background rate -> INCONCLUSIVE (extend)
        ("one seed held-out +0.273, max10 0.145", [0.0001, -0.0002, 0.0, 0.273, -0.0001], 0.145, "INCONCLUSIVE-stability"),
        ("two seeds held-out +0.273", [0.273, -0.0002, 0.0, 0.273, -0.0001], 0.0, "FAIL-stability"),
        ("one seed held-out +0.03 (1.5 yardsticks)", [0.0001, -0.0002, 0.0, 0.03, -0.0001], 0.0, "PASS-stability: 1 severe"),
        ("two seeds held-out +0.03", [0.03, -0.0002, 0.0, 0.03, -0.0001], 0.0, "INCONCLUSIVE-stability: 2 severe")):
    root = tempfile.mkdtemp(prefix="qstab_")
    try:
        v = verdict(root, ds, bump)
        ok(v["stability"].startswith(want), f"{name}: {v['stability']}")
    finally:
        shutil.rmtree(root)
# DECISIONS 4.86: a catastrophic BASE leg is flagged and its seed leaves the quality statistic; Pocock bound.
root = tempfile.mkdtemp(prefix="qstab_")
try:
    for i, s in enumerate((11, 22, 33, 44, 55)):
        leg(root, f"q_ref_seed{s}", 1.30 + 0.001 * i + (0.3 if s == 33 else 0.0))   # seed 33: the BASE leg is bad
        leg(root, f"q_cand_seed{s}", 1.30 + 0.001 * i)
    out = subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand"],
                         capture_output=True, text=True).stdout
    v = json.loads(out.strip().splitlines()[-1])
    ok(v["base_arm_catastrophic"] == [33] and "base arm: 1 catastrophic" in v["stability"],
       f"a catastrophic base leg is flagged and dropped from quality ({v['base_arm_catastrophic']})")
    ok(v["alpha"] == 0.0294 and "97.06%" in v["quality"], "two looks by default: Pocock 0.0294 one-sided")
    out1 = subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand", "--looks", "1"],
                          capture_output=True, text=True).stdout
    ok(json.loads(out1.strip().splitlines()[-1])["alpha"] == 0.05, "--looks 1: a single-look test at 0.05")
finally:
    shutil.rmtree(root)

# Audit 2026-10-25 G-6: legs paired by NAME must be one experiment.
root = tempfile.mkdtemp(prefix="qstab_")
try:
    for i, s in enumerate((11, 22, 33, 44, 55)):
        leg(root, f"q_ref_seed{s}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s}", 1.30 + 0.001 * i, cfg={"lr": "2.0e-5"} if s == 33 else None)
    v = run_raw = json.loads(subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand"],
                                            capture_output=True, text=True).stdout.strip().splitlines()[-1])
    ok(v["quality"].startswith("REFUSED") and any("lr" in p for p in v["consistency"]),
       f"a cand leg on another recipe (lr) is refused: {v['quality'][:60]}")
    v = json.loads(subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand",
                                   "--allow-workload", "lr"], capture_output=True, text=True).stdout.strip().splitlines()[-1])
    ok(not v["quality"].startswith("REFUSED"), "...unless the test declares lr as the change under test")
finally:
    shutil.rmtree(root)
root = tempfile.mkdtemp(prefix="qstab_")
try:
    for i, s in enumerate((11, 22, 33, 44, 55)):
        leg(root, f"q_ref_seed{s}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s}", 1.30 + 0.001 * i, cfg={"GEAK_OVERLAY_SHA": "aaa" if s != 44 else "bbb"})
    v = json.loads(subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[-1])
    ok(v["quality"].startswith("REFUSED") and any("another implementation" in p for p in v["consistency"]),
       "a cand arm that runs another overlay stack on one seed is refused")
finally:
    shutil.rmtree(root)

# Review 2026-10-26 fixes.
def judge(root, *extra):
    out = subprocess.run([sys.executable, QF, os.path.join(root, "q"), "--base", "ref", "--cand", "cand", *extra],
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])
SEEDS = (11, 22, 33, 44, 55)
root = tempfile.mkdtemp(prefix="qstab_")
try:   # a BASE leg that NaN-exits: seed dropped as base-arm catastrophic, the test judged on the rest
    for i, s_ in enumerate(SEEDS):
        leg(root, f"q_ref_seed{s_}", 1.30 + 0.001 * i, nsteps=200 if s_ == 33 else 358, nan_exit=(s_ == 33))
        leg(root, f"q_cand_seed{s_}", 1.30 + 0.001 * i)
    v = judge(root)
    ok(not v["quality"].startswith("INCOMPLETE") and 33 in v.get("base_arm_catastrophic", []),
       f"a base leg that NaN-exits is base-arm catastrophic, not INCOMPLETE ({v['quality'][:40]})")
finally:
    shutil.rmtree(root)
root = tempfile.mkdtemp(prefix="qstab_")
try:   # with --ctrl: a diverged base leg no longer wrecks the positive control
    for i, s_ in enumerate(SEEDS):
        leg(root, f"q_ref_seed{s_}", 1.30 + 0.001 * i + (0.3 if s_ == 33 else 0.0))
        leg(root, f"q_cand_seed{s_}", 1.30 + 0.001 * i)
        leg(root, f"q_ctrl70_seed{s_}", 1.30 + 0.001 * i + 0.0065 + 0.0002 * (i % 2), cfg={"lr": "7.0e-6"})
    v = judge(root, "--ctrl", "ctrl70")
    ok("positive control was not detected" not in v["quality"] and 33 in v.get("base_arm_catastrophic", []),
       f"--ctrl: the control is measured without the diverged base seed ({v['quality'][:50]})")
finally:
    shutil.rmtree(root)
root = tempfile.mkdtemp(prefix="qstab_")
try:   # the control arm must be the base recipe with ONLY the LR scaled
    for i, s_ in enumerate(SEEDS):
        leg(root, f"q_ref_seed{s_}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s_}", 1.30 + 0.001 * i)
        leg(root, f"q_ctrl70_seed{s_}", 1.30 + 0.001 * i + 0.0065 + 0.0002 * (i % 2),
            cfg={"lr": "7.0e-6", "weight-decay": "0.0" if s_ == 22 else "0.1"})
    v = judge(root, "--ctrl", "ctrl70")
    ok(v["quality"].startswith("REFUSED") and any("ctrl70 differs" in p for p in v["consistency"]),
       "a control leg that moved more than the LR is refused")
finally:
    shutil.rmtree(root)
root = tempfile.mkdtemp(prefix="qstab_")
try:   # presence of an env knob on one seed's cand only, across snapshot versions, is another implementation
    for i, s_ in enumerate(SEEDS):
        leg(root, f"q_ref_seed{s_}", 1.30 + 0.001 * i)
        leg(root, f"q_cand_seed{s_}", 1.30 + 0.001 * i,
            cfg={"snapshot_keys": "v5"} if s_ != 44 else {"snapshot_keys": "v6", "env:FC1_BWD_MODE": "serial"})
    v = judge(root)
    ok(v["quality"].startswith("REFUSED") and any("FC1_BWD_MODE" in p for p in v["consistency"]),
       "an env knob present on one cand seed only (v5 vs v6) is caught")
finally:
    shutil.rmtree(root)
root = tempfile.mkdtemp(prefix="qstab_")
try:   # content: the arms read different input bytes under the same path
    for i, s_ in enumerate(SEEDS):
        leg(root, f"q_ref_seed{s_}", 1.30 + 0.001 * i, content={"v": 1, "inputs": {"train-data-path": "100:aa"}})
        leg(root, f"q_cand_seed{s_}", 1.30 + 0.001 * i,
            content={"v": 1, "inputs": {"train-data-path": "100:bb" if s_ == 55 else "100:aa"}})
    v = judge(root)
    ok(v["quality"].startswith("REFUSED") and any("input contents" in p for p in v["consistency"]),
       "arms that read different input contents are refused")
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
