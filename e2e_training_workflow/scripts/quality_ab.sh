#!/bin/bash
# quality_ab.sh -- P0-4: does training on the accepted stack produce an equally good model?
# Pre-registered in knowledge/gate_convergence.md ("P0-4 PRE-REGISTRATION", 2026-10-15); this script
# runs exactly that protocol and quality_ab_analyze.py applies its verdicts.
#
# Arms (run interleaved per seed, in this order):
#   ref   no overlay
#   s8    the accepted stack (stack_overlay_env.sh --cond s8); s2..s7 accepted as arms for bisection
#   ctrl  POSITIVE CONTROL: ref with --lr 7.0e-6 (x0.7), a known modest recipe change the metric must see
# Each leg: one epoch (358 iters at GBS 16) on the train19k pack, then Megatron's end-of-training
# evaluation over 19 x 16 = 304 packs of the held-out pack.
#
# Usage: quality_ab.sh [--seeds 1234,2345,3456] [--arms ref,s8,ctrl] [--iters 358] [--name qab]
#                      [--workload workloads/qwen2_5_math_7b_packed_cosine.env]   (default: the packed
#                      constant-LR descriptor of P0-4/P0-4b)
set -euo pipefail
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
SEEDS="1234,2345,3456"; ARMS="ref,s8,ctrl"; ITERS=""; NAME=qab; NPROC=8; GPUS=""
# Default = the CLEARED production recipe (audit 2026-10-22 G4: it was the constant-LR packed.env, which
# is NOT cleared and which no outcome test since P0-4d should run on).
WORKLOAD="$GEAK_ROOT/workloads/qwen2_5_math_7b_packed_cosine_avg.env"; CAND_CMD=""; CAND_TARGS=""
STACK=megatron; TRAIN=""; HELDOUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --nproc) NPROC="$2"; shift 2;;    # the run's process count (was hard-coded 8; review QAB-GPUS)
    --gpus) GPUS="$2"; shift 2;;      # the run's gpu_ids, passed to bench_train --gpus    # default: the descriptor's LR_DECAY_ITERS (its planned run = one epoch)
    --name) NAME="$2"; shift 2;;
    --workload) WORKLOAD="$2"; shift 2;;
    --cand-cmd) CAND_CMD="$2"; shift 2;;
    --cand-train-args) CAND_TARGS="$2"; shift 2;;   # e.g. a config lever under test, cand arm only
    # Second stack (HF TRL): unpacked data -- HF pads dynamically and does not take Megatron packs.
    --stack) STACK="$2"; shift 2;;
    --train) TRAIN="$2"; shift 2;;
    --heldout) HELDOUT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
# One epoch = the descriptor's planned run length, not a hard-coded 358 (which was GBS 16 on train19k only).
if [ -z "$ITERS" ]; then
  ITERS="$(bash -c 'source "$1" >/dev/null 2>&1; echo "${LR_DECAY_ITERS:-}"' _ "$WORKLOAD")"
  [ -n "$ITERS" ] || { echo "!!! $WORKLOAD sets no LR_DECAY_ITERS (not a scheduled recipe?): pass --iters explicitly" >&2; exit 2; }
fi
if [ "$STACK" = hf_trl ]; then
  TRAIN="${TRAIN:-/work/data/sft_ultrachat20k_train19k.jsonl}"
  HELDOUT="${HELDOUT:-/work/data/sft_ultrachat20k_heldout1k.jsonl}"
  EVAL_ARGS="--valid-data-path $HELDOUT"          # hf_sft.py evaluates once, after training
else
  TRAIN="${TRAIN:-/work/data/sft_ultrachat20k_train19k_packed_7b.jsonl}"
  HELDOUT="${HELDOUT:-/work/data/sft_ultrachat20k_heldout1k_packed_7b.jsonl}"
  EVAL_ARGS="--valid-data-path $HELDOUT --eval-interval 1000000 --eval-iters 19"
fi
# The pre-registered data, by content: a different file is a different experiment.
( cd "$WORK_HOST/data" && sha256sum "$(basename "$TRAIN")" "$(basename "$HELDOUT")" ) \
  | awk '{print substr($1,1,16), $2}' | while read -r h f; do
    case "$f:$h" in
      sft_ultrachat20k_train19k_packed_7b.jsonl:e3bb6e7b7194d07a|sft_ultrachat20k_heldout1k_packed_7b.jsonl:54fad019ad78a761) ;;
      sft_ultrachat20k_train19k.jsonl:89f8aa068165b0ff|sft_ultrachat20k_heldout1k.jsonl:6409b13001becc78) ;;
      *) echo "!!! $f sha $h is not a pre-registered data file -- refusing" >&2; exit 3;;
    esac
  done
STAMP="$(date +%m%d_%H%M%S)"
OVL_HOST="$WORK_HOST/overlays/${NAME}_$STAMP"
for SEED in ${SEEDS//,/ }; do
  for ARM in ${ARMS//,/ }; do
    LEG="${ARM}_seed${SEED}"
    OUT_HOST="$WORK_HOST/runs/${NAME}_${STAMP}_${LEG}"
    EXTRA=""; TARGS="--seed $SEED $EVAL_ARGS"
    case "$ARM" in
      ref) ;;
      s[2-8]) EXTRA="$("$GEAK_ROOT/scripts/stack_overlay_env.sh" --cond "$ARM" --stack "$STACK" --out "$OUT_HOST" --install "$OVL_HOST")" ;;
      # The accepted stack with every side stream disabled (fc1/qkv wgrad overlap -> serial, lm-head
      # single stream): same kernel bytes, single-stream scheduling. Tests the side-stream race
      # hypothesis for the late-training instability (2026-10-16).
      s8serial) EXTRA="$("$GEAK_ROOT/scripts/stack_overlay_env.sh" --cond s8 --stack "$STACK" --out "$OUT_HOST" --install "$OVL_HOST")"
                EXTRA="$EXTRA FC1_BWD_MODE=serial QKV_BWD_MODE=serial LMHEAD_CE_TWO_STREAM=0" ;;
      # A candidate stack given as a COMMAND with the BASE_OVERLAY_CMD contract (append `--out <leg>`,
      # print one `--extra-env` line). This is how the workflow's QualityGate tests a stack that is not
      # (yet) a stack_overlay_env.sh condition, e.g. the base stack plus this run's accepted overlays.
      cand) [ -n "$CAND_CMD" ] || { echo "arm cand needs --cand-cmd" >&2; exit 2; }
            EXTRA="$(eval "$CAND_CMD --out \"$OUT_HOST\"")"; TARGS="$TARGS $CAND_TARGS" ;;
      only-*) EXTRA="$("$GEAK_ROOT/scripts/stack_overlay_env.sh" --cond "only:${ARM#only-}" --stack "$STACK" --out "$OUT_HOST" --install "$OVL_HOST/${ARM}")" ;;
      ctrl) TARGS="$TARGS --lr 7.0e-6" ;;
      # Positive control for a SCHEDULED recipe: the whole schedule scaled x0.7 (peak and floor).
      ctrl70) TARGS="$TARGS --lr 7.0e-6 --min-lr 7.0e-7" ;;
      *) echo "unknown arm: $ARM" >&2; exit 2;;
    esac
    echo ">>> $(date +%H:%M:%S) leg $LEG -> $OUT_HOST"
    "$GEAK_ROOT/scripts/bench_train.sh" --workload "$WORKLOAD" --stack "$STACK" --nproc "$NPROC" ${GPUS:+--gpus "$GPUS"} \
      --iters "$ITERS" --data "$TRAIN" --out "$OUT_HOST" --tag "${NAME}_${LEG}" \
      --extra-train-args "$TARGS" ${EXTRA:+--extra-env "$EXTRA"} || echo "!!! leg $LEG returned non-zero"
  done
done
echo ">>> legs under $WORK_HOST/runs/${NAME}_${STAMP}_*"
echo ">>> analyse: python3 $GEAK_ROOT/scripts/quality_ab_analyze.py $WORK_HOST/runs/${NAME}_${STAMP}"
