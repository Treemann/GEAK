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
print(f"\n{fails} FAILURE(S)" if fails else "\nall checks passed"); sys.exit(1 if fails else 0)
