#!/bin/bash
# measure_packed_stack.sh -- interleaved, cumulative re-measure of the kernels accepted on the packed
# 7B workload. Each kernel was gated against the BARE packed reference (or, since 2026-10-14, on top
# of a `baseline_overlay` stack); this asks what it adds on top of the ones before it.
#
# Conditions are cumulative and defined in ONE place, scripts/stack_overlay_env.sh (which also holds
# the coexistence rules: stacking order, fc1's proof-path patch, module isolation):
#   ref, s2 (rope+swiglu), s3 (+fc1), s4 (+lmhead), s5 (+fc2_oproj), s6 (+attn), s7 (+qkv), s8 (+rmsnorm)
# s<k> vs s<k-1> is that kernel's increment on the stack; s<k> vs ref is the total. Legs of one rep run
# back to back in the --conds order, so node drift cannot align with condition.
#
# Results so far (knowledge/sft_measurement.md): s3 +9.1%, s5 +13.8%, s7 +20.7% over ref.
#
# Usage: measure_packed_stack.sh [--conds ref,s5,s6,s7] [--reps 3] [--iters 30] [--name pstack]
#                                [--profile 12:13]   (profiler window on ranks 0,4: NOT a throughput leg)
#                                [--workload workloads/qwen2_5_math_7b_packed_cosine.env]
#                                [--gbs-list 16,24,64]  (GBS in PACKS; every condition at every GBS,
#                                 interleaved within a rep; legs are named <cond>g<gbs><rep>. Legs at
#                                 different GBS do not see the same batches, so compare them per leg,
#                                 and pair conditions only WITHIN one GBS)
set -euo pipefail
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
CONDS="ref,s5,s6,s7"; REPS=3; ITERS=30; NAME=pstack; PROFILE=""; GBS_LIST=""
WORKLOAD="$GEAK_ROOT/workloads/qwen2_5_math_7b_packed.env"
while [ $# -gt 0 ]; do
  case "$1" in
    --conds) CONDS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --gbs-list) GBS_LIST="$2"; shift 2;;
    --workload) WORKLOAD="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
for c in ${CONDS//,/ }; do
  case "$c" in ref|s2|s3|s4|s5|s6|s7|s8) ;; *) echo "unknown condition: $c" >&2; exit 2;; esac
done
STAMP="$(date +%m%d_%H%M%S)"
OVL_HOST="$WORK_HOST/overlays/${NAME}_$STAMP"     # fresh per measurement run, never rm -rf'd
[ -e "$OVL_HOST" ] && { echo "!!! $OVL_HOST exists" >&2; exit 3; }

for G in ${GBS_LIST//,/ }; do
  case "$G" in *[!0-9]*|"") echo "bad --gbs-list entry: $G" >&2; exit 2;; esac
  [ $((G % 8)) -eq 0 ] || { echo "GBS $G is not a multiple of DP 8 x MBS 1" >&2; exit 2; }
done
G_LOOP="${GBS_LIST//,/ }"; G_LOOP="${G_LOOP:-_}"   # "_" = the descriptor's own GBS, legacy leg names
for i in $(seq 1 "$REPS"); do
 for G in $G_LOOP; do
  for MODE in ${CONDS//,/ }; do
    if [ "$G" = _ ]; then LEG="${MODE}${i}"; GARG=(); else LEG="${MODE}g${G}${i}"; GARG=(--gbs "$G"); fi
    OUT_HOST="$WORK_HOST/runs/${NAME}_${STAMP}_${LEG}"
    # Installs on first use and re-verifies the gated bytes on every leg; refuses on any mismatch.
    EXTRA="$("$GEAK_ROOT/scripts/stack_overlay_env.sh" --cond "$MODE" --out "$OUT_HOST" --install "$OVL_HOST")"
    echo ">>> $(date +%H:%M:%S) leg $LEG -> $OUT_HOST"
    "$GEAK_ROOT/scripts/bench_train.sh" --workload "$WORKLOAD" --stack megatron --nproc 8 \
      --iters "$ITERS" --out "$OUT_HOST" --tag "${NAME}_${LEG}" ${EXTRA:+--extra-env "$EXTRA"} \
      "${GARG[@]}" ${PROFILE:+--profile-steps "$PROFILE" --profile-ranks 0,4}
  done
 done
done
echo ">>> legs under $WORK_HOST/runs/${NAME}_${STAMP}_*  (overlays: $OVL_HOST)"
