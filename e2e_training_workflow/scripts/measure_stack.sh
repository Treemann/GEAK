#!/bin/bash
# measure_stack.sh -- measure N accepted overlays running TOGETHER against the bare stack.
#
# WHY THIS EXISTS. Each accepted kernel is gated against the stack carried at the time, and separate
# runs each gate against the BARE stack. Those deltas do not add: lmhead measured +4.66% and
# mlp_fc2_down +7.34%, both against bare, in different runs. The combined figure -- the only one a
# user actually cares about -- was never measured, and until GEAK_OVERLAY took a list it could not be.
#
# WHAT IT DOES NOT DO. It does not re-gate anything. L0 and L1 judged each kernel when it was
# accepted; this answers one question only: what does the whole stack deliver on the headline metric?
# Engagement IS re-proved, per overlay, because a stack that silently drops one overlay would report
# a smaller number and look like an honest disappointment.
#
# Usage:
#   measure_stack.sh --bundles <dir1>,<dir2>,... --out-prefix <dir> [--iters 30] [--reps 2]
set -euo pipefail

GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
BUNDLES="" ; OUT_PREFIX="" ; ITERS=30 ; REPS=2
while [ $# -gt 0 ]; do
  case "$1" in
    --bundles) BUNDLES="$2"; shift 2;;
    --out-prefix) OUT_PREFIX="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$BUNDLES" ] && [ -n "$OUT_PREFIX" ] || { echo "need --bundles and --out-prefix" >&2; exit 2; }

IFS=',' read -ra BLIST <<< "$BUNDLES"
STAMP="$(date +%m%d_%H%M%S)"
SPECS=()

# ---- install every overlay, verifying the bytes that were gated --------------------------------
for B in "${BLIST[@]}"; do
  NAME="$(ls "$B/overlays" | head -1)"
  [ -n "$NAME" ] || { echo "!!! $B has no overlays/ payload" >&2; exit 3; }
  OVL_HOST="$WORK_HOST/overlays/stack_${STAMP}_${NAME}"
  mkdir -p "$OVL_HOST"
  cp "$B/overlays/$NAME/"* "$OVL_HOST/" 2>/dev/null || true
  # L0 passed on THESE bytes. A stack assembled from anything else is ungated.
  ( cd "$OVL_HOST" && sha256sum -c --quiet "$B/SHA256SUMS.overlay" ) || {
    echo "!!! overlay bytes in $B differ from the gated ones -- refusing" >&2; exit 3; }
  TARGET="$(grep -ho '"target": "[^"]*"' "$B/launch.sh" | head -1 | sed 's/.*: "//;s/"//')"
  REPL="$(grep -ho '"replacement": "[^"]*"' "$B/launch.sh" | head -1 | sed 's/.*: "//;s/"//')"
  # Re-point the replacement at THIS run's copy: the bundle's path names its own install location.
  REPL_FILE="${REPL%%:*}"; REPL_ATTR="${REPL##*:}"
  REPL_NEW="/work/overlays/stack_${STAMP}_${NAME}/$(basename "$REPL_FILE"):${REPL_ATTR}"
  echo ">>> $NAME : $TARGET -> $REPL_NEW"
  # Record what this overlay needs; the MANIFEST ITSELF is written per leg below, because
  # engagement_out must name the leg that is about to run. A manifest generated once, outside the
  # loop, writes every leg's proof into one directory -- or, as happened on the first attempt here,
  # into a directory that does not exist, and 3528 silent "could not write engagement proof" lines
  # later the stack has no proof at all. The bundle's own launch.sh already warns about exactly this.
  SPECS+=("$NAME|$TARGET|$REPL_NEW|$OVL_HOST")
done


# ---- paired legs, interleaved ------------------------------------------------------------------
# ref, cand, ref, cand ... so node drift cannot align with condition. The pairing is what the
# headline rests on; a block of refs followed by a block of cands would confound the two.
for i in $(seq 1 "$REPS"); do
  for MODE in ref cand; do
    OUT="${OUT_PREFIX}_${MODE}${i}"
    [ -e "$OUT" ] && { echo "!!! $OUT exists -- need a fresh dir" >&2; exit 3; }
    unset GEAK_OVERLAY
    EXTRA=""
    if [ "$MODE" = cand ]; then
      mkdir -p "$OUT"
      MANIFESTS=""
      for SPEC in "${SPECS[@]}"; do
        IFS='|' read -r NAME TARGET REPL_NEW OVL_HOST <<< "$SPEC"
        MFILE="$OVL_HOST/manifest_${MODE}${i}.json"
        cat > "$MFILE" <<EOF
{"target": "$TARGET",
 "replacement": "$REPL_NEW",
 "engagement_out": "/work/runs/$(basename "$OUT")/engagement_${NAME}.json"}
EOF
      done
      # Container paths, in bundle order. Built from SPECS rather than accumulated above so there is
      # exactly one place that decides what GEAK_OVERLAY says.
      MANIFESTS="$(for SPEC in "${SPECS[@]}"; do IFS='|' read -r N T R O <<< "$SPEC"; \
        printf '%s,' "/work/overlays/$(basename "$O")/manifest_${MODE}${i}.json"; done | sed 's/,$//')"
      echo ">>> GEAK_OVERLAY=$MANIFESTS"
      EXTRA="GEAK_OVERLAY=$MANIFESTS"
    fi
    echo ">>> leg $MODE$i -> $OUT"
    "$GEAK_ROOT/scripts/bench_train.sh" \
      --workload "$GEAK_ROOT/workloads/qwen2_5_math_7b.env" \
      --stack megatron --nproc 8 --iters "$ITERS" \
      --out "$OUT" --tag "stack_$MODE$i" \
      ${EXTRA:+--extra-env "$EXTRA"}
  done
done

# ---- engagement, per overlay -------------------------------------------------------------------
echo ">>> engagement per overlay (each must show forward>0 AND backward>0 on ALL 8 ranks):"
python3 - "${OUT_PREFIX}_cand1" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "engagement_*.json.rank*")))
if not files:
    print("  NO ENGAGEMENT PROOF -- this leg did not measure the stack it claims to."); sys.exit(4)
by = {}
for p in files:
    name = os.path.basename(p).split(".json.rank")[0].replace("engagement_", "")
    by.setdefault(name, []).append(json.load(open(p)))
ok = True
for name, recs in sorted(by.items()):
    bad = [r for r in recs if r["forward"] == 0 or r["backward"] == 0 or r.get("fallback_fwd", 0)]
    ok &= not bad and len(recs) == 8
    print(f"  {name:36s} ranks={len(recs)} fwd={sum(r['forward'] for r in recs)} "
          f"bwd={sum(r['backward'] for r in recs)}{'   <-- BAD' if bad or len(recs) != 8 else ''}")
if not ok:
    print("  A stack that silently drops an overlay reports a smaller number and looks like an "
          "honest disappointment. Discard this leg.")
    sys.exit(4)
PY
echo ">>> legs written under ${OUT_PREFIX}_*; compare with scripts/compare_legs.py"
