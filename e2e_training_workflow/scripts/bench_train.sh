#!/bin/bash
# bench_train.sh -- stack-neutral steady-state training measurement dispatcher.
#
# Owns: job lifecycle, the occupancy guard, the hang watchdog, and the canonical summary.
# Owns NOT: anything framework-specific -- that is scripts/adapters/<stack>.sh.
#
# The headline number comes from the PROBE (scripts/train_probe.py), not from the stack's log:
# upstream Megatron reports no tokens/s at all, and stacks that do report it count nominal
# (padding-inclusive) tokens, which for SFT overstates useful work by ~3.5x on this workload.
# The stack's own figure is recorded beside ours as a cross-check.
#
# Usage:
#   bench_train.sh --workload <file.env> --iters N [--stack megatron] [--nproc 8]
#                  [--out DIR] [--data PATH] [--mbs N] [--gbs N] [--seq N]
#                  [--extra-train-args "..."] [--extra-env "..."] [--tag NAME]
set -uo pipefail

GEAK_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(dirname "$GEAK_SCRIPTS")"

STACK=megatron; NPROC=8; ITERS=30; TAG=""
WORKLOAD=""; OUT_DIR=""; DATA_PATH="${DATA_PATH:-/work/data/sft_ultrachat20k.jsonl}"
MLM_DIR="${MLM_DIR:-/workspace/Primus/third_party/Megatron-LM}"
EXTRA_TRAIN_ARGS=""; EXTRA_ENV=""; MBS_OVR=""; GBS_OVR=""; SEQ_OVR=""
HANG_TIMEOUT="${HANG_TIMEOUT:-900}"     # no step progress for this long => hung collective

while [ $# -gt 0 ]; do
  case "$1" in
    --workload) WORKLOAD="$2"; shift 2 ;;
    --stack) STACK="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --data) DATA_PATH="$2"; shift 2 ;;
    --mbs) MBS_OVR="$2"; shift 2 ;;
    --gbs) GBS_OVR="$2"; shift 2 ;;
    --seq) SEQ_OVR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --extra-train-args) EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    --extra-env) EXTRA_ENV="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$WORKLOAD" ] || { echo "--workload is required (no default: benching the wrong target silently is worse than failing)" >&2; exit 2; }
# shellcheck disable=SC1090
source "$WORKLOAD"
[ -n "$MBS_OVR" ] && MBS="$MBS_OVR"; [ -n "$GBS_OVR" ] && GBS="$GBS_OVR"; [ -n "$SEQ_OVR" ] && SEQ="$SEQ_OVR"
OUT_DIR="${OUT_DIR:-/work/runs/bench_${WORKLOAD_NAME}${TAG:+_$TAG}_$(date +%m%d_%H%M%S)}"
PROBE_DIR="$OUT_DIR/probe"
mkdir -p "$PROBE_DIR"
export MLM_DIR MODEL_CKPT TOKENIZER_DIR DATA_PATH OUT_DIR PROBE_DIR NPROC ITERS MBS GBS SEQ \
       ARCH_ARGS ROPE_BASE EXTRA_TRAIN_ARGS EXTRA_ENV GEAK_SCRIPTS

# shellcheck disable=SC1090
source "$GEAK_SCRIPTS/adapters/${STACK}.sh"
adapter_capabilities > "$OUT_DIR/capabilities.json"

echo ">>> bench_train: stack=$STACK workload=$WORKLOAD_NAME nproc=$NPROC iters=$ITERS mbs=$MBS gbs=$GBS seq=$SEQ"
echo ">>> out=$OUT_DIR"

python3 "$GEAK_SCRIPTS/gpu_occupancy.py" before > "$OUT_DIR/occupancy_before.json" 2>/dev/null

adapter_launch
echo ">>> job pid $JOB_PID"

# ---- hang watchdog -------------------------------------------------------------------------
# A hung collective leaves every rank ALIVE while nobody advances, so process liveness reports
# "healthy". Progress is the only honest signal: watch the probe heartbeats.
( last_progress=$(date +%s); last_sig=""
  while kill -0 "$JOB_PID" 2>/dev/null; do
    sleep 30
    sig=$(cat "$PROBE_DIR"/heartbeat_rank*.json 2>/dev/null | md5sum | cut -d' ' -f1)
    now=$(date +%s)
    if [ "$sig" != "$last_sig" ]; then last_sig="$sig"; last_progress=$now; fi
    if [ $((now - last_progress)) -gt "$HANG_TIMEOUT" ]; then
      echo "!!! no step progress for ${HANG_TIMEOUT}s -> treating as hung collective; tearing down" \
        | tee -a "$OUT_DIR/train.log"
      echo '{"hang_detected": true}' > "$OUT_DIR/hang.json"
      adapter_teardown "$JOB_PID"
      break
    fi
  done ) &
WATCHDOG=$!

wait "$JOB_PID"; JOB_RC=$?
kill "$WATCHDOG" 2>/dev/null
adapter_teardown "$JOB_PID" >/dev/null 2>&1

python3 "$GEAK_SCRIPTS/gpu_occupancy.py" after > "$OUT_DIR/occupancy_after.json" 2>/dev/null
python3 "$GEAK_SCRIPTS/gpu_occupancy.py" compare "$OUT_DIR/occupancy_before.json" \
        "$OUT_DIR/occupancy_after.json" "" > "$OUT_DIR/occupancy_verdict.json" 2>/dev/null

STACK_TPUT="$(adapter_metrics 2>/dev/null)"
python3 - "$PROBE_DIR" "$OUT_DIR" "$JOB_RC" "${STACK_TPUT:-}" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.environ["GEAK_SCRIPTS"] + "/x")))
from probe_summarize import summarize

probe_dir, out_dir, rc, stack_tput = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
s = summarize(probe_dir)
s["job_returncode"] = rc
s["hang_detected"] = os.path.exists(os.path.join(out_dir, "hang.json"))
for name in ("occupancy_verdict", "capabilities"):
    p = os.path.join(out_dir, name + ".json")
    if os.path.exists(p):
        try:
            s[name] = json.load(open(p))
        except Exception:
            pass
# The stack's own number, kept beside ours and never substituted for it.
s["stack_reported"] = {"tflop_s_per_gpu_median": float(stack_tput) if stack_tput else None,
                       "note": "cross-check only; the gate uses useful_tok_s_per_gpu from the probe"}
ok = (rc == 0 and not s["hang_detected"] and "error" not in s
      and s.get("occupancy_verdict", {}).get("verdict") != "contaminated")
s["status"] = "ok" if ok else "unusable"
json.dump(s, open(os.path.join(out_dir, "bench_summary.json"), "w"), indent=2)
print(json.dumps({k: s.get(k) for k in (
    "status", "useful_tok_s_per_gpu", "nominal_tok_s_per_gpu", "pad_fraction_pct",
    "step_median_ms", "worst_spread_pct", "worst_iqr_spread_pct", "outlier_steps_max",
    "warmup_steps_max", "rank_skew_pct", "peak_reserved_mb")}, indent=2))
PY
echo ">>> summary: $OUT_DIR/bench_summary.json"
