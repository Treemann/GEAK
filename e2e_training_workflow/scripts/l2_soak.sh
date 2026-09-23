#!/bin/bash
# l2_soak.sh -- the L2 convergence soak: a LONG leg, run at the pinned measurement identity.
#
# WHY THIS EXISTS. L1 cannot see slow convergence damage, and that is measured, not assumed. On a
# band calibrated from three identical 7B legs, injecting perturbations into a real leg gives:
#     step-1 loss +0.2% as `overlay`      -> CAUGHT (0.05% limit)
#     loss +25% across the whole steady window, step 1 clean -> MISSED BY EVERYTHING
#     grad_norm x3 across the steady window -> caught
# By step 12-30 three IDENTICAL legs already disagree by ~8.9% median relative, so the per-step
# envelope is structurally a coarse net for gross deviations. A 30-step gate leg invokes the swapped
# kernel ~224 times; one epoch invokes it ~19200 times. The soak is the only instrument that looks at
# what L1 is named after, and `risk_class: backward` candidates auto-trigger it.
#
# WHAT IT IS NOT. It is not a throughput measurement -- do not quote tok/s from a soak leg. The soak
# asks one question: does the accepted stack train the same way as the reference over a long run?
#
# Usage:
#   l2_soak.sh --mode ref|cand|pos --bundle <finalize_bundle_dir> --steps N --out <dir> [--pos-kernel F]
#
#     ref   reference leg, overlay OFF. >=3 of these: they calibrate the criteria AND supply the
#           null control (hold one out; if the criteria reject a held-out CLEAN leg, they are junk).
#     cand  the accepted stack, overlay ON, bytes verified against the bundle's SHA256SUMS.overlay.
#     pos   POSITIVE CONTROL: the same overlay with a known-bad kernel substituted. If the soak does
#           not reject this, the soak proves nothing about the candidate. A negative test is worth
#           more than a passing one -- that doctrine is what produced scale_bias (DECISIONS 3.2).
set -euo pipefail

GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
MODE="" ; BUNDLE="" ; STEPS=300 ; OUT_HOST="" ; POS_KERNEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --bundle) BUNDLE="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --out) OUT_HOST="$2"; shift 2;;
    --pos-kernel) POS_KERNEL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$MODE" ] && [ -n "$BUNDLE" ] && [ -n "$OUT_HOST" ] || { echo "need --mode --bundle --out" >&2; exit 2; }

# One probe dir is ONE leg. train_probe.py refuses to append to an existing sidecar, but failing here
# costs a second instead of an hour of GPU.
[ -e "$OUT_HOST" ] && { echo "!!! $OUT_HOST exists -- a soak leg needs a FRESH dir" >&2; exit 3; }
mkdir -p "$OUT_HOST"
OUT_CTR="${OUT_HOST/#$WORK_HOST//work}"

# Always unset first: an inherited GEAK_OVERLAY would make a "ref" leg silently measure the candidate.
unset GEAK_OVERLAY
EXTRA_ENV=""

if [ "$MODE" = cand ] || [ "$MODE" = pos ]; then
  OVL_HOST="$WORK_HOST/overlays/l2soak_$MODE"
  rm -rf "$OVL_HOST"; mkdir -p "$OVL_HOST"
  cp "$BUNDLE/overlays/lmhead_ce_unmasked_gather/"{impl.py,candidate_kernel.py,lmhead_ce_triton.py} "$OVL_HOST/"
  if [ "$MODE" = cand ]; then
    # Fail loudly if the bytes are not the ones that were gated. L0 passed on THESE sha256s.
    ( cd "$OVL_HOST" && sha256sum -c --quiet "$BUNDLE/SHA256SUMS.overlay" ) || {
      echo "!!! overlay bytes differ from the gated ones -- refusing to launch" >&2; exit 3; }
  else
    [ -f "$POS_KERNEL" ] || { echo "!!! --pos-kernel required for mode=pos" >&2; exit 2; }
    # Substitute the known-bad kernel. The checksum check is deliberately SKIPPED here and the
    # substitution is printed: this leg is SUPPOSED to differ from the gated bytes, and hiding that
    # would make the control indistinguishable from the candidate.
    cp "$POS_KERNEL" "$OVL_HOST/candidate_kernel.py"
    echo ">>> POSITIVE CONTROL: candidate_kernel.py replaced by $POS_KERNEL"
    echo ">>> sha256 $(sha256sum "$OVL_HOST/candidate_kernel.py" | cut -d' ' -f1)"
    grep -n "bias\|1\.00[0-9]" "$OVL_HOST/candidate_kernel.py" | head -3 || true
  fi
  # Manifest generated PER RUN: a checked-in one names an absolute engagement_out and would write
  # this run's proof into an old run's directory, where stale counters read as a fresh pass.
  MANIFEST_HOST="$OUT_HOST/geak_overlay_manifest.json"
  cat > "$MANIFEST_HOST" <<EOF
{"target": "megatron.core.models.gpt.gpt_model:GPTModel._postprocess",
 "replacement": "/work/overlays/l2soak_$MODE/impl.py:postprocess",
 "engagement_out": "$OUT_CTR/engagement.json"}
EOF
  EXTRA_ENV="GEAK_OVERLAY=${MANIFEST_HOST/#$WORK_HOST//work}"
fi

echo ">>> L2 soak leg: mode=$MODE steps=$STEPS out=$OUT_HOST"
"$GEAK_ROOT/scripts/bench_train.sh" \
  --workload "$GEAK_ROOT/workloads/qwen2_5_math_7b.env" \
  --stack megatron \
  --nproc 8 \
  --iters "$STEPS" \
  --out "$OUT_HOST" \
  --tag "l2soak_$MODE" \
  ${EXTRA_ENV:+--extra-env "$EXTRA_ENV"}

if [ "$MODE" = cand ] || [ "$MODE" = pos ]; then
  echo ">>> engagement (forward>0 AND backward>0 on ALL 8 ranks, fallback_fwd == 0):"
  python3 - "$OUT_HOST" <<'PY'
import glob, json, sys
ok = True
for p in sorted(glob.glob(sys.argv[1] + "/engagement_fused.json.rank*")):
    d = json.load(open(p))
    bad = d["forward"] == 0 or d["backward"] == 0 or d.get("fallback_fwd", 0) != 0
    ok &= not bad
    print(f"  {p.split('.')[-1]}: fwd={d['forward']} bwd={d['backward']} "
          f"fallback_fwd={d.get('fallback_fwd', 0)}{'   <-- BAD' if bad else ''}")
if not ok:
    print("  ENGAGEMENT FAILED: this leg did not measure the overlay it claims to.")
    sys.exit(4)
PY
fi
echo ">>> leg complete: $OUT_HOST"
