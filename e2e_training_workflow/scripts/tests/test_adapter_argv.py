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


def launch(extra_env, adapter=None, gbs=16):
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
  ROPE_BASE=10000 TOKENIZER_DIR=/t MODEL_CKPT=/c DATA_PATH=/d MBS=1 GBS={gbs} ITERS=30 \\
  ARCH_ARGS="--num-layers 28" EXTRA_TRAIN_ARGS="" EXTRA_ENV="" {env_lines}
source {adapter or ADAPTER}
adapter_launch
wait
cat {t}/o/train.log
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        # The temp dir appears in path arguments (e.g. HF's --output-dir <OUT_DIR>/hf_out); normalise
        # it so the hash pins the ARGV, not a random directory name.
        return r.returncode, [x.replace(t, "<TMP>") for x in r.stdout.splitlines()], r.stderr


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

# ---- v6 recipe knobs (DECISIONS 4.55) ----------------------------------------------------------------
rc, argv, err = launch({"NORM_EPS": "1e-6"})
ok(rc == 0 and flag(argv, "--norm-epsilon") == "1e-6", "NORM_EPS reaches the argv as --norm-epsilon")
rc, argv, err = launch({"EXTRA_TRAIN_ARGS": "--rope-fusion --manual-gc"})
ok(rc == 0 and "--no-rope-fusion" not in argv and "--rope-fusion" not in argv and "--manual-gc" in argv,
   "the --rope-fusion pseudo-flag drops --no-rope-fusion, is itself stripped, and keeps the other extras")


def preflight(norm_eps, model_eps, extra=""):
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "tok")); os.makedirs(os.path.join(t, "o"))
        if model_eps is not None:
            open(os.path.join(t, "tok", "config.json"), "w").write('{"rms_norm_eps": %s}' % model_eps)
        env = f"TOKENIZER_DIR={t}/tok OUT_DIR={t}/o EXTRA_TRAIN_ARGS='{extra}'" + (f" NORM_EPS={norm_eps}" if norm_eps else "")
        r = subprocess.run(["bash", "-c", f"set -euo pipefail; export {env}; source {ADAPTER}; adapter_preflight"],
                           capture_output=True, text=True)
        w = os.path.join(t, "o", "recipe_warnings.json")
        return r.returncode, r.stderr, open(w).read() if os.path.exists(w) else None


rc, err, w = preflight("1e-6", "1e-06")
ok(rc == 0 and w is None, "preflight: NORM_EPS matching the model config passes silently")
rc, err, w = preflight("1e-5", "1e-06")
ok(rc == 0 and "RECIPE NOTE" in err and w and '"note": "descriptor NORM_EPS"' in w,
   "preflight: a descriptor eps differing from the model is a recipe CHOICE (P0-4l): runs, printed, recorded")
rc, err, w = preflight(None, "1e-06")
ok(rc == 0 and "RECIPE NOTE" in err and w and '"model_config": 1e-06' in w and "Megatron default" in w,
   "preflight: no NORM_EPS on a 1e-6 model runs Megatron's 1e-5 and records it")
rc, err, w = preflight("1e-6", "1e-06", extra="--seed 1 --norm-epsilon 1e-5")
ok(rc == 0 and "experiment arm" in err and w and '"run": 1e-5' in w,
   "preflight: a per-leg --norm-epsilon arm is allowed, warned and recorded (the descriptor stays the recipe)")
rc, err, w = preflight(None, None)
ok(rc == 0 and w is None, "preflight: no config.json -> nothing to check")

# ---- the HF TRL adapter: same contract -------------------------------------------------------------
HF = os.environ.get("GEAK_HF_ADAPTER_UNDER_TEST", os.path.join(HERE, "..", "adapters", "hf_trl.sh"))
rc, argv, err = launch({}, adapter=HF, gbs=64)
sha = hashlib.sha256("\n".join(argv).encode()).hexdigest()[:16]
HF_PINNED = "b9bbfd2a6dac626e"   # == hf_trl.sh before the schedule/FSDP became descriptor-driven (2026-10-16)
ok(rc == 0 and len(argv) == 49, f"HF: older descriptor launches under set -u, 49 tokens (got {len(argv)}; {err.strip()[:100]})")
ok(sha == HF_PINNED, f"HF: older descriptor argv is byte-identical to the pinned one ({sha} vs {HF_PINNED})")
rc, argv, err = launch({"LR": "1.0e-5", "MIN_LR": "1.0e-6", "LR_DECAY_STYLE": "cosine",
                        "LR_WARMUP_ITERS": "11", "LR_DECAY_ITERS": "358", "HF_FSDP": "full_shard"},
                       adapter=HF, gbs=64)
ok(rc == 0 and flag(argv, "--lr-decay-style") == "cosine" and flag(argv, "--lr-decay-iters") == "358"
   and flag(argv, "--hf-fsdp") == "full_shard", "HF: scheduled + FSDP descriptor reaches the argv")

# ---- the HF schedule equals Megatron's, at the LRs Megatron actually logged ----------------------
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("hf_sft", os.path.join(HERE, "..", "adapters", "hf_sft.py"))
_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hf)
# runs/lrsmoke_1016_cos/train.log: Megatron "learning rate:" at iterations 1, 11, 30 (0-based step = it - 1)
for it, logged in ((1, 9.090909e-07), (11, 1.000000e-05), (30, 9.933586e-06)):
    got = 1.0e-5 * _hf.lr_factor(it - 1, 1.0e-5, 1.0e-6, 11, 358)
    ok(abs(got - logged) <= 1e-12 + 1e-6 * logged, f"HF lr at iteration {it} = {got:.6e} == Megatron's {logged:.6e}")
ok(abs(1.0e-5 * _hf.lr_factor(357, 1.0e-5, 1.0e-6, 11, 358) - 1.0e-6) < 1e-9,
   "HF lr reaches min_lr at the decay horizon")

print(f"\n{len(fails)} FAILURE(S)" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
