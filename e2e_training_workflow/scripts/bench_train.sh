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

# ---- host / container transparency -------------------------------------------------------------
# The orchestrator runs wherever Claude Code runs; the training stack only exists inside the image.
# Rather than making every role know which side it is on (and getting it wrong once), this script
# detects that it is outside the container and re-executes itself inside, with the same arguments.
# The same command line therefore works from both sides, which is the only version of this that
# survives contact with an agent reading a prompt.
if [ ! -d "${MLM_DIR:-/workspace/Primus/third_party/Megatron-LM}" ] && [ -z "${GEAK_IN_CONTAINER:-}" ]; then
  IMAGE="${GEAK_IMAGE:-rocm/primus:v26.4}"
  MODELS="${GEAK_MODELS_DIR:-/home/treemann/models}"
  WORK="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
  REPO="$(cd "$GEAK_ROOT/.." && pwd)"
  # Translate host paths to their mount points, so a caller may pass either and be right. Without
  # this the re-exec succeeds and then dies on a path that exists only outside -- exactly the
  # confusing failure this mechanism exists to prevent.
  # Resolve RELATIVE paths first: the container starts in / , so a path that was valid in the
  # caller's shell resolves to nothing inside. (Observed: `--workload workloads/x.env` run from the
  # workflow directory re-executed fine and then died on a missing file.) Anything that names an
  # existing file or directory becomes absolute before the mount translation.
  _ARGS=()
  for _a in "$@"; do
    if [ -e "$_a" ]; then _a="$(cd "$(dirname "$_a")" && pwd)/$(basename "$_a")"; fi
    case "$_a" in
      "$REPO"*|"$WORK"*|"$MODELS"*) ;;
      /*) if [ -e "$_a" ]; then
            echo "!!! $_a is outside every mount ($REPO, $WORK, $MODELS) and will NOT exist inside" \
                 "the container. Move it under one of them, or set GEAK_WORK_DIR/GEAK_MODELS_DIR." >&2
            exit 2
          fi ;;
    esac
    _a="${_a//$REPO//repo}"
    _a="${_a//$WORK//work}"
    _a="${_a//$MODELS//models}"
    _ARGS+=("$_a")
  done
  echo ">>> not inside the training image; re-executing in $IMAGE"
  exec docker run --rm --network host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size 32G \
    -v "$MODELS":/models:ro -v "$WORK":/work -v "$REPO":/repo \
    -e GEAK_IN_CONTAINER=1 -e GEAK_HOST_WORK="$WORK" \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    ${ATTENTION_BACKEND:+-e ATTENTION_BACKEND="$ATTENTION_BACKEND"} \
    ${RETRY_CONTAMINATED:+-e RETRY_CONTAMINATED="$RETRY_CONTAMINATED"} \
    --entrypoint bash "$IMAGE" \
    /repo/e2e_training_workflow/scripts/bench_train.sh "${_ARGS[@]}"
fi

STACK=megatron; NPROC=8; ITERS=30; TAG=""
WORKLOAD=""; OUT_DIR=""; DATA_PATH="${DATA_PATH:-/work/data/sft_ultrachat20k.jsonl}"
MLM_DIR="${MLM_DIR:-/workspace/Primus/third_party/Megatron-LM}"
EXTRA_TRAIN_ARGS=""; EXTRA_ENV=""; MBS_OVR=""; GBS_OVR=""; SEQ_OVR=""
# Probe-driven profiler window (upstream Megatron has no profiler args at all; see the adapter's
# capability manifest). Empty => no trace, and the run is byte-identical to an unprofiled one.
PROFILE_STEPS=""; PROFILE_RANKS="0"
HANG_TIMEOUT="${HANG_TIMEOUT:-900}"     # no step progress for this long => hung collective
# Contamination policy (locked: re-measure). A leg that shared the box with a foreign GPU process is
# not a measurement, so it is RE-RUN rather than reported. Retries are bounded: past this, the node
# is busy in a way the workflow cannot fix, and burning gate budget on it is worse than stopping and
# saying so. Every attempt is recorded, because a leg that needed three tries is itself a finding.
RETRY_CONTAMINATED="${RETRY_CONTAMINATED:-2}"

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
    --retry-contaminated) RETRY_CONTAMINATED="$2"; shift 2 ;;
    --profile-steps) PROFILE_STEPS="$2"; shift 2 ;;
    --profile-ranks) PROFILE_RANKS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$WORKLOAD" ] || { echo "--workload is required (no default: benching the wrong target silently is worse than failing)" >&2; exit 2; }
[ -r "$WORKLOAD" ] || { echo "workload descriptor not readable: $WORKLOAD" >&2; exit 2; }
# shellcheck disable=SC1090
source "$WORKLOAD"
: "${WORKLOAD_NAME:?the workload descriptor must define WORKLOAD_NAME (see workloads/*.env)}"
[ -n "$MBS_OVR" ] && MBS="$MBS_OVR"; [ -n "$GBS_OVR" ] && GBS="$GBS_OVR"; [ -n "$SEQ_OVR" ] && SEQ="$SEQ_OVR"
OUT_DIR="${OUT_DIR:-/work/runs/bench_${WORKLOAD_NAME}${TAG:+_$TAG}_$(date +%m%d_%H%M%S)}"
PROBE_DIR="$OUT_DIR/probe"
mkdir -p "$PROBE_DIR"
export MLM_DIR MODEL_CKPT TOKENIZER_DIR DATA_PATH OUT_DIR PROBE_DIR NPROC ITERS MBS GBS SEQ \
       ARCH_ARGS ROPE_BASE EXTRA_TRAIN_ARGS EXTRA_ENV GEAK_SCRIPTS PROFILE_STEPS PROFILE_RANKS \
       LR MIN_LR LR_DECAY_STYLE LR_WARMUP_ITERS LR_DECAY_ITERS

# shellcheck disable=SC1090
source "$GEAK_SCRIPTS/adapters/${STACK}.sh"
adapter_capabilities > "$OUT_DIR/capabilities.json"

echo ">>> bench_train: stack=$STACK workload=$WORKLOAD_NAME nproc=$NPROC iters=$ITERS mbs=$MBS gbs=$GBS seq=$SEQ"
echo ">>> out=$OUT_DIR"

# ---- one attempt: occupancy sample -> job -> occupancy sample -> verdict -------------------
run_attempt() {
  local n="$1"
  python3 "$GEAK_SCRIPTS/gpu_occupancy.py" before > "$OUT_DIR/occupancy_before.json" 2>/dev/null

  adapter_launch
  echo ">>> attempt $n: job pid $JOB_PID"

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
  # run_attempt is consumed through a pipe, so it executes in a SUBSHELL and any variable it sets is
  # lost to the parent. The exit code has to travel through the filesystem, not through a variable.
  echo "$JOB_RC" > "$OUT_DIR/.job_rc"
  kill "$WATCHDOG" 2>/dev/null
  adapter_teardown "$JOB_PID" >/dev/null 2>&1

  python3 "$GEAK_SCRIPTS/gpu_occupancy.py" after > "$OUT_DIR/occupancy_after.json" 2>/dev/null
  python3 "$GEAK_SCRIPTS/gpu_occupancy.py" compare "$OUT_DIR/occupancy_before.json" \
          "$OUT_DIR/occupancy_after.json" "" > "$OUT_DIR/occupancy_verdict.json" 2>/dev/null
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict','unknown'))" \
          "$OUT_DIR/occupancy_verdict.json" 2>/dev/null || echo unknown
}

ATTEMPTS=0
while :; do
  ATTEMPTS=$((ATTEMPTS + 1))
  VERDICT="$(run_attempt "$ATTEMPTS" | tail -1)"
  JOB_RC="$(cat "$OUT_DIR/.job_rc" 2>/dev/null || echo 1)"
  [ "$VERDICT" = "contaminated" ] || break
  if [ "$ATTEMPTS" -gt "$RETRY_CONTAMINATED" ]; then
    echo "!!! leg contaminated on all $ATTEMPTS attempts — NOT reporting a number. Another process " \
         "held a GPU each time; this node is busy in a way the workflow cannot fix." | tee -a "$OUT_DIR/train.log"
    echo "{\"contamination_exhausted\": true, \"attempts\": $ATTEMPTS}" > "$OUT_DIR/blocked.json"
    break
  fi
  echo ">>> leg contaminated (a foreign process held a GPU); re-measuring (attempt $((ATTEMPTS + 1)))" \
    | tee -a "$OUT_DIR/train.log"
  mv "$OUT_DIR/probe" "$OUT_DIR/probe.contaminated.$ATTEMPTS" 2>/dev/null || true
  mkdir -p "$PROBE_DIR"
done

# A capture run writes the frozen record (tens of MB) from inside a backward hook, which costs real
# time: measured 4043.9 vs ~4755 useful tok/s/GPU on the same config, a 15% dent. So a capture run is
# no more a measurement leg than a profiled one is.
# Matches EVERY capture entry point, not just the generic one: the LM-head composite capture
# (GEAK_LMHEAD_CAPTURE_DIR) writes a multi-GB record from a backward hook and would otherwise have
# produced a throughput number that looks like a measurement and is not.
if printf '%s' "${EXTRA_ENV:-}" | grep -qE 'GEAK_CAPTURE_TARGET|GEAK_LMHEAD_CAPTURE_DIR|GEAK_MLP_CAPTURE_DIR|GEAK_ROPE_CAPTURE_DIR|GEAK_RMSN_CAPTURE_DIR|GEAK_GRADTAIL_CAPTURE_DIR'; then
  echo ">>> NOTE: shape/oracle capture was active; this run's throughput is NOT a valid measurement." \
    | tee -a "$OUT_DIR/train.log"
  echo '{"captured_run": true}' > "$OUT_DIR/captured.json"
fi

# A profiled window perturbs the steps it covers, so the profile run is NOT a measurement run:
# keep them separate rather than reporting a throughput taken under the profiler.
if [ -n "$PROFILE_STEPS" ]; then
  echo ">>> NOTE: profiler window ${PROFILE_STEPS} was active; this run's throughput is NOT a valid measurement." \
    | tee -a "$OUT_DIR/train.log"
  echo '{"profiled_run": true}' > "$OUT_DIR/profiled.json"
fi

STACK_TPUT="$(adapter_metrics 2>/dev/null)"
python3 - "$PROBE_DIR" "$OUT_DIR" "$JOB_RC" "${STACK_TPUT:-}" "$ATTEMPTS" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.environ["GEAK_SCRIPTS"] + "/x")))
from probe_summarize import summarize

probe_dir, out_dir, rc, stack_tput = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
attempts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
s = summarize(probe_dir)
s["job_returncode"] = rc
s["hang_detected"] = os.path.exists(os.path.join(out_dir, "hang.json"))
s["profiled_run"] = os.path.exists(os.path.join(out_dir, "profiled.json"))
s["captured_run"] = os.path.exists(os.path.join(out_dir, "captured.json"))
# A leg that needed retries is itself a finding: it says the node was not quiet.
s["attempts"] = attempts
s["contamination_retries"] = attempts - 1
s["contamination_exhausted"] = os.path.exists(os.path.join(out_dir, "blocked.json"))
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
# A profiled run can never be a gate leg: the window it traced is not the workload's real timing.
ok = (rc == 0 and not s["hang_detected"] and not s["profiled_run"] and not s["captured_run"]
      and not s["contamination_exhausted"] and "error" not in s
      and s.get("occupancy_verdict", {}).get("verdict") != "contaminated")
s["status"] = "ok" if ok else "unusable"
json.dump(s, open(os.path.join(out_dir, "bench_summary.json"), "w"), indent=2)
print(json.dumps({k: s.get(k) for k in (
    "status", "useful_tok_s_per_gpu", "nominal_tok_s_per_gpu", "pad_fraction_pct",
    "step_median_ms", "worst_spread_pct", "worst_iqr_spread_pct", "outlier_steps_max",
    "warmup_steps_max", "rank_skew_pct", "peak_reserved_mb")}, indent=2))
PY
echo ">>> summary: $OUT_DIR/bench_summary.json"
# The orchestrator may be reading from OUTSIDE the container, where /work is mounted elsewhere.
# Printing only the inside path sends an agent looking for a file that is not there.
if [ -n "${GEAK_HOST_WORK:-}" ]; then
  echo ">>> summary (host path): ${OUT_DIR/\/work/$GEAK_HOST_WORK}/bench_summary.json"
fi
