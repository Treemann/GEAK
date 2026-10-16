"""Guards the Megatron adapter's launch argv (no GPU).

WHY. The argv is workload identity: `argv_hash` and `config_digest` are computed from it, and six
bundles plus every baseline and band were measured under the argv the OLDER descriptors produce.
Twice in one day an adapter change put that at risk:
  * 2026-10-16 (e29ca475): the LR schedule became descriptor-driven. It was argv-identical for old
    descriptors, but that was proven only by a hand diff.
  * 2026-10-16 (c3edee3b): `$DDP_EXTRA_ARGS` was referenced bare. bench_train.sh runs `set -u`, so
    every leg on a descriptor that does not define it died at launch. The hand diff ran in a `set +u`
    subshell and hid it.
This test runs `adapter_launch` with a fake torchrun under `set -euo pipefail`, exactly as
bench_train.sh does, and pins the default argv by hash.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# GEAK_ADAPTER_UNDER_TEST lets the same checks run against a historical adapter, e.g. the pre-schedule
# one (`git show e29ca475~1:e2e_training_workflow/scripts/adapters/megatron.sh`), which is where the
# pinned hash below comes from.
ADAPTER = os.environ.get("GEAK_ADAPTER_UNDER_TEST", os.path.join(HERE, "..", "adapters", "megatron.sh"))

fails = []


def ok(c, m):
    print(("  ok: " if c else "  FAIL: ") + m)
    if not c:
        fails.append(m)


def flag(argv, name):
    """Value after `name`, or None -- never raises, so a broken launch reports as FAIL lines."""
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def launch(extra_env):
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "bin"))
        fake = os.path.join(t, "bin", "torchrun")
        with open(fake, "w") as f:
            f.write('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        os.chmod(fake, 0o755)
        env_lines = " ".join(f"{k}={v!r}" for k, v in extra_env.items())
        script = f"""
set -euo pipefail
export PATH={t}/bin:$PATH OUT_DIR={t}/o PROBE_DIR={t}/o/p MLM_DIR=/m GEAK_SCRIPTS=/s NPROC=8 SEQ=4096 \\
  ROPE_BASE=10000 TOKENIZER_DIR=/t MODEL_CKPT=/c DATA_PATH=/d MBS=1 GBS=16 ITERS=30 \\
  ARCH_ARGS="--num-layers 28" EXTRA_TRAIN_ARGS="" EXTRA_ENV="" {env_lines}
source {ADAPTER}
adapter_launch
wait
cat {t}/o/train.log
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return r.returncode, r.stdout.splitlines(), r.stderr


rc, argv, err = launch({})
ok(rc == 0 and "unbound variable" not in err, f"older descriptor (no LR/DDP vars) launches under set -u ({err.strip()[:120]})")
sha = hashlib.sha256("\n".join(argv).encode()).hexdigest()[:16]
PINNED = "e44b022c2e4bae65"   # == the pre-schedule adapter (e29ca475~1), measured 2026-10-16
ok(len(argv) == 80, f"older descriptor argv has 80 tokens (got {len(argv)})")
ok(sha == PINNED, f"older descriptor argv is byte-identical to the pinned one ({sha} vs {PINNED})")
ok(flag(argv, "--lr-decay-style") == "constant",
   "older descriptor keeps the historical constant LR")

rc, argv, err = launch({"LR": "1.0e-5", "MIN_LR": "1.0e-6", "LR_DECAY_STYLE": "cosine",
                        "LR_WARMUP_ITERS": "11", "LR_DECAY_ITERS": "358"})
ok(rc == 0, "scheduled descriptor launches under set -u")
ok(flag(argv, "--lr-decay-style") == "cosine" and "--lr-decay-iters" in argv,
   "scheduled descriptor carries cosine + decay iters")
ok("--ddp-average-in-collective" not in argv, "and NOT the DDP lever unless asked")

rc, argv, err = launch({"DDP_EXTRA_ARGS": "--ddp-average-in-collective"})
ok(rc == 0 and argv.count("--ddp-average-in-collective") == 1, "DDP_EXTRA_ARGS reaches the argv once")

print(f"\n{len(fails)} FAILURE(S)" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
