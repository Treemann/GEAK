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
#   l2_soak.sh --mode ref|cand|pos --bundle <bundle> --steps N --out <dir>
#              [--control bias|stale] [--pos-bias 1.004] [--alpha 1.0]
#
#     ref   reference leg, overlay OFF. >=3 of these: they calibrate the criteria AND supply the
#           null control (hold one out; if the criteria reject a held-out CLEAN leg, they are junk).
#     cand  the accepted stack, overlay ON, bytes verified against the bundle's SHA256SUMS.overlay.
#     pos   POSITIVE CONTROL. If the soak does not reject this, the soak proves nothing about the
#           candidate. A negative test is worth more than a passing one -- that doctrine is what
#           produced scale_bias (DECISIONS 3.2).
#
#           THE CLASS IS NOW EXPLICIT (--control), because the first one was the wrong class and
#           nothing in the interface said so. Two soaks, ~10 h, failed to reject `bias` -- and
#           DECISIONS 3.6c then measured WHY: L0 rejects that same defect in 11.5 s, so it never
#           reaches the soak at all. Naming the class at the call site makes "which control did this
#           soak actually run?" answerable from the command line instead of from the source.
#
#             bias   x1.004 on the returned GRADIENT, every invocation. OP-LOCAL NUMERICS.
#                    Kept for the record, and it is the one that must NOT be used to license the
#                    soak: L0 owns this class. Running it answers a question already answered.
#             stale  cross-step staleness -- blend in the PREVIOUS invocation's gscale (--alpha).
#                    L0 CANNOT SEE THIS BY CONSTRUCTION: it evaluates one frozen microbatch in one
#                    call, where there is no previous invocation and a repeated identical input
#                    makes the blend bit-exactly the identity for every alpha. This is the class
#                    DECISIONS 3.6c reserves for the soak -- cross-step state accumulating over
#                    ~19 200 invocations. See scripts/controls/stale_state_control.py.
set -euo pipefail

GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
MODE="" ; BUNDLE="" ; STEPS=300 ; OUT_HOST="" ; POS_KERNEL="" ; BIAS=""
CONTROL="stale" ; ALPHA="1.0"   # default to the class L0 cannot see; `bias` must be asked for
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --bundle) BUNDLE="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --out) OUT_HOST="$2"; shift 2;;
    --pos-bias|--pos-kernel) POS_KERNEL="$2"; shift 2;;
    --control) CONTROL="$2"; shift 2;;
    --alpha) ALPHA="$2"; shift 2;;
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
  # A dir PER LEG, named after the output dir. Two reasons: a shared one cannot be cleaned (the
  # container writes __pycache__ as root, and `rm -rf` then fails under `set -e`), and a stale file
  # from a previous leg silently becomes part of the next one's measurement.
  OVL_HOST="$WORK_HOST/overlays/l2soak_$(basename "$OUT_HOST")"
  mkdir -p "$OVL_HOST"
  cp "$BUNDLE/overlays/lmhead_ce_unmasked_gather/"{impl.py,candidate_kernel.py,lmhead_ce_triton.py} "$OVL_HOST/"
  if [ "$MODE" = cand ]; then
    # Fail loudly if the bytes are not the ones that were gated. L0 passed on THESE sha256s.
    ( cd "$OVL_HOST" && sha256sum -c --quiet "$BUNDLE/SHA256SUMS.overlay" ) || {
      echo "!!! overlay bytes differ from the gated ones -- refusing to launch" >&2; exit 3; }
  else
    # POSITIVE CONTROL: inject a multiplicative bias into the GRADIENT the overlay returns.
    #
    # Injection point matters, and the obvious choices are wrong. Substituting a foreign "known-bad"
    # kernel does not even load: the overlay's impl.py requires candidate_kernel's internals
    # (_prep/_scatter_loss), so a task-harness kernel dies with AttributeError before step 1 and
    # controls nothing. Biasing the forward loss is subtly worse -- it is inside a custom
    # autograd.Function whose backward is computed independently, so the REPORTED loss shifts while
    # training proceeds identically. That would test the criteria against a cosmetic shift and would
    # look like a working control.
    #
    # `gscale` is the real entry: it feeds dlogits -> dh -> dgrad AND wgrad, so a bias here actually
    # changes what the optimizer sees, which is what a convergence soak must be able to detect. The
    # checksum check is deliberately SKIPPED and the patch printed -- this leg is SUPPOSED to differ
    # from the gated bytes, and hiding that would make the control indistinguishable from the
    # candidate.
    case "$CONTROL" in
      stale)
        # The class the soak exists for. Invisibility to L0 is algebraic; the self-test proves it.
        python3 "$GEAK_ROOT/scripts/controls/stale_state_control.py" "$OVL_HOST/impl.py" "$ALPHA"
        ;;
      bias)
        echo ">>> WARNING: --control bias is OP-LOCAL NUMERICS, which L0 rejects in 11.5 s." >&2
        echo ">>> Two soaks (~10 h) failed to reject it for that reason (DECISIONS 3.6c)." >&2
        echo ">>> A soak licensed by this control demonstrates nothing. Use --control stale." >&2
        BIAS="${POS_KERNEL:-1.004}"
        python3 - "$OVL_HOST/impl.py" "$BIAS" <<'PY'
import re, sys
p, bias = sys.argv[1], sys.argv[2]
src = open(p).read()
old = "gscale = (g_f.index_select(0, idx) * mg).contiguous()"
if old not in src:
    sys.exit("POSITIVE CONTROL: gradient injection point not found in impl.py -- refusing to run a "
             "control that would silently be a second candidate leg")
src = src.replace(old, f"gscale = (g_f.index_select(0, idx) * mg * {bias}).contiguous()"
                       f"   # POSITIVE CONTROL: deliberate gradient bias")
open(p, "w").write(src)
print(f">>> POSITIVE CONTROL: gradient scaled by {bias} in impl.py backward")
PY
        ;;
      *) echo "unknown --control $CONTROL (known: stale, bias)" >&2; exit 2;;
    esac
    grep -n "POSITIVE CONTROL" "$OVL_HOST/impl.py" | head -2
    echo ">>> impl.py sha256 $(sha256sum "$OVL_HOST/impl.py" | cut -d' ' -f1) (differs from gated, by design)"
  fi
  # Manifest generated PER RUN: a checked-in one names an absolute engagement_out and would write
  # this run's proof into an old run's directory, where stale counters read as a fresh pass.
  MANIFEST_HOST="$OUT_HOST/geak_overlay_manifest.json"
  cat > "$MANIFEST_HOST" <<EOF
{"target": "megatron.core.models.gpt.gpt_model:GPTModel._postprocess",
 "replacement": "/work/overlays/l2soak_$(basename "$OUT_HOST")/impl.py:postprocess",
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
