#!/usr/bin/env python3
"""bench_train.sh guards that need no GPU (audit 2026-10-22 ORCH-8, OVL-7).

* one --out is one leg: a dir that already holds a measured leg is refused BEFORE anything runs
  (agent-chosen leg names collided across sessions and the second run overwrote the first summary);
* --gpus is accepted (the host-side re-exec maps it to HIP_VISIBLE_DEVICES).
Runs the script with GEAK_IN_CONTAINER=1 so it never re-executes into docker.
"""
import os, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BT = os.path.join(ROOT, "scripts", "bench_train.sh")
WL = os.path.join(ROOT, "workloads", "qwen2_5_0_5b.env")
fails = 0
def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1

tmp = tempfile.mkdtemp(prefix="btguard_")
try:
    out = os.path.join(tmp, "leg")
    os.makedirs(os.path.join(out, "probe"))
    open(os.path.join(out, "bench_summary.json"), "w").write("{}")
    env = dict(os.environ, GEAK_IN_CONTAINER="1", TRITON_CACHE_DIR=os.path.join(tmp, "tc"))
    r = subprocess.run(["bash", BT, "--workload", WL, "--stack", "megatron", "--nproc", "8", "--iters", "2",
                        "--gpus", "0,1,2,3,4,5,6,7", "--out", out], env=env, capture_output=True, text=True, timeout=60)
    ok(r.returncode == 3 and "already holds a measured leg" in r.stderr,
       f"an --out that already holds a leg is refused up front (rc={r.returncode})")
    ok("unknown arg" not in r.stderr, "--gpus is an accepted argument")
    ok(open(os.path.join(out, "bench_summary.json")).read() == "{}", "the existing leg's summary is untouched")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
src = open(BT).read()
ok('_VIS="${HIP_VISIBLE_DEVICES:-${_GPUS:-0,1,2,3,4,5,6,7}}"' in src and '-e HIP_VISIBLE_DEVICES="$_VIS"' in src,
   "the re-exec maps --gpus to HIP_VISIBLE_DEVICES")
ok("--init --name" in src and "--label geak=1" in src, "the container is named, labelled and has an init")
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
