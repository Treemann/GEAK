#!/bin/bash
# measure_packed_stack.sh -- interleaved, cumulative re-measure of the kernels accepted on the packed
# 7B workload. Each kernel was gated against the BARE packed reference; this asks what it still adds
# on top of the ones before it.
#
# Conditions are cumulative (each adds one overlay to the previous):
#   ref -- packed reference, no overlay
#   s2  -- rope_fused_qk + swiglu_bwd_fwd_bw      (bundle finalize_bundle_1010_rope_swiglu)
#   s3  -- + mlp_fc1_gemm                          (gated 2026-10-10)
#   s4  -- + lmhead_ce_dense_gemm                  (bundle finalize_bundle_1013_lmhead_dense)
#   s5  -- + mlp_fc2_oproj_gemm                    (2026-10-13, inconclusive_noise: +1.78% in a 2.0% band)
# s<k> vs s<k-1> is that kernel's increment on the stack; s<k> vs ref is the total. Legs of one rep run
# back to back in the --conds order, so node drift cannot align with condition.
#
# Overlay coexistence (checked 2026-10-13): targets are disjoint (rope: attention.apply_rotary_pos_emb,
# swiglu: mlp.bias_swiglu_impl, fc1: TE _LayerNormLinear, lmhead: GPTModel._postprocess, fc2/o_proj:
# TE _Linear). Each writes its proof next to its OWN manifest's engagement_out (fc1 needs the patch
# below; the others match by __file__ or by their directory name in "replacement"). fc1 and fc2 both
# ship gemm_select.py / lt_gemm.cpp: fc2 pins its gemm_select into sys.modules only while its kernel.py
# loads, and the extensions are named fc1_lt_gemm / fc2oproj_lt_gemm.
#
# WHY fc1's impl.py IS PATCHED (and why that does not ungate it): it writes its proof next to the FIRST
# manifest in GEAK_OVERLAY, which stacked is rope's. Only impl.py -- the training-side rebind --
# changes; the kernel bytes L0 judged are copied verbatim and their sha256 printed.
#
# Usage: measure_packed_stack.sh [--conds ref,s3,s4,s5] [--reps 3] [--iters 30] [--name pstack]
set -euo pipefail
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
CONDS="ref,s3,s4,s5"; REPS=3; ITERS=30; NAME=pstack
while [ $# -gt 0 ]; do
  case "$1" in
    --conds) CONDS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
for c in ${CONDS//,/ }; do
  case "$c" in ref|s2|s3|s4|s5) ;; *) echo "unknown condition: $c" >&2; exit 2;; esac
done
RS_BUNDLE="$WORK_HOST/runs/finalize_bundle_1010_rope_swiglu"
LM_BUNDLE="$WORK_HOST/runs/finalize_bundle_1013_lmhead_dense"
FC1_SRC="$WORK_HOST/overlays/mlp_fc1_gemm_1010_054626"
FC2_SRC="$WORK_HOST/overlays/mlp_fc2_oproj_gemm_1013_072624"
WORKLOAD="$GEAK_ROOT/workloads/qwen2_5_math_7b_packed.env"
STAMP="$(date +%m%d_%H%M%S)"
OVL_HOST="$WORK_HOST/overlays/${NAME}_$STAMP"
OVL_CTR="/work/overlays/${NAME}_$STAMP"
[ -e "$OVL_HOST" ] && { echo "!!! $OVL_HOST exists" >&2; exit 3; }
mkdir -p "$OVL_HOST/build"

# ---- bundles: copy, then verify the installed bytes against the gated sums ---------------------
cp -r "$RS_BUNDLE/overlays/rope_fused_qk" "$RS_BUNDLE/overlays/swiglu_bwd_fwd_bw" \
      "$LM_BUNDLE/overlays/lmhead_ce_dense_gemm" "$OVL_HOST/"
for B in "$RS_BUNDLE" "$LM_BUNDLE"; do
  ( cd "$OVL_HOST" && sha256sum -c --quiet "$B/SHA256SUMS.overlay" ) || {
    echo "!!! installed bytes differ from $B/SHA256SUMS.overlay -- refusing" >&2; exit 3; }
  echo ">>> bytes verified against $B/SHA256SUMS.overlay"
done

# ---- gate-dir overlays: kernel bytes verbatim, manifests and build caches left behind ----------
copy_overlay() {  # $1 src, $2 dest name
  mkdir -p "$OVL_HOST/$2"
  for f in "$1"/*; do
    case "$(basename "$f")" in manifest_*|__pycache__|.torch_ext) continue;; esac
    cp -r "$f" "$OVL_HOST/$2/"
  done
  echo ">>> $2 kernel bytes (L0 judged these):"
  ( cd "$OVL_HOST/$2" && sha256sum kernel.py gemm_select.py lt_gemm.cpp lt_solutions.csv | sed 's/^/      /' )
}
copy_overlay "$FC1_SRC" mlp_fc1_gemm
copy_overlay "$FC2_SRC" mlp_fc2_oproj_gemm
python3 - "$OVL_HOST/mlp_fc1_gemm/impl.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = '''_OUT = ""
try:
    _OUT = json.load(open(os.environ["GEAK_OVERLAY"].split(",")[0]))["engagement_out"]
    _OUT = os.path.join(os.path.dirname(_OUT), "engagement_fused.json")
except Exception:
    _OUT = os.environ.get("GEAK_FC1_ENGAGEMENT", "")'''
new = '''_OUT = ""
try:
    # Find OUR manifest (the one whose replacement lives in this file's directory), not the FIRST one:
    # stacked behind rope, the first manifest is rope's and fc1's proof would overwrite rope's.
    _here = os.path.dirname(os.path.abspath(__file__))
    _mans = [json.load(open(m)) for m in os.environ["GEAK_OVERLAY"].split(",") if m.strip()]
    _mine = [m for m in _mans if os.path.dirname(m.get("replacement", "").split(":")[0]) == _here]
    _OUT = (_mine or _mans)[0]["engagement_out"]
    _OUT = os.path.join(os.path.dirname(_OUT), "engagement_fused.json")
except Exception:
    _OUT = os.environ.get("GEAK_FC1_ENGAGEMENT", "")'''
assert old in s, "fc1 impl.py proof-path block not found -- refusing to run an unpatched stack"
open(p, "w").write(s.replace(old, new, 1))
print(">>> fc1 impl.py patched: proof path follows its own manifest")
PY

# name|target|replacement attr|proof subdir, in stacking order
OVERLAYS=(
  "rope_fused_qk|megatron.core.transformer.attention:apply_rotary_pos_emb|apply_rotary_pos_emb|rope"
  "swiglu_bwd_fwd_bw|megatron.core.transformer.mlp:bias_swiglu_impl|bias_swiglu_impl|swiglu"
  "mlp_fc1_gemm|transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear|lnlinear_apply|fc1"
  "lmhead_ce_dense_gemm|megatron.core.models.gpt.gpt_model:GPTModel._postprocess|postprocess|lmhead"
  "mlp_fc2_oproj_gemm|transformer_engine.pytorch.module.linear:_Linear|linear_apply|fc2op"
)
n_overlays() { case "$1" in ref) echo 0;; s2) echo 2;; s3) echo 3;; s4) echo 4;; s5) echo 5;; esac; }

for i in $(seq 1 "$REPS"); do
  for MODE in ${CONDS//,/ }; do
    LEG="${MODE}${i}"
    OUT_HOST="$WORK_HOST/runs/${NAME}_${STAMP}_${LEG}"
    OUT_CTR="/work/runs/${NAME}_${STAMP}_${LEG}"
    EXTRA=""; LIST=""
    for ((k = 0; k < $(n_overlays "$MODE"); k++)); do
      IFS='|' read -r DIR TARGET ATTR SUB <<< "${OVERLAYS[$k]}"
      mkdir -p "$OUT_HOST/$SUB"
      cat > "$OVL_HOST/$DIR/m_${LEG}.json" <<EOF
{"target": "$TARGET",
 "replacement": "$OVL_CTR/$DIR/impl.py:$ATTR",
 "engagement_out": "$OUT_CTR/$SUB/engagement.json"}
EOF
      LIST="${LIST:+$LIST,}$OVL_CTR/$DIR/m_${LEG}.json"
    done
    [ -n "$LIST" ] && EXTRA="GEAK_OVERLAY=$LIST TORCH_EXTENSIONS_DIR=$OVL_CTR/build"
    echo ">>> $(date +%H:%M:%S) leg $LEG -> $OUT_HOST"
    "$GEAK_ROOT/scripts/bench_train.sh" --workload "$WORKLOAD" --stack megatron --nproc 8 \
      --iters "$ITERS" --out "$OUT_HOST" --tag "${NAME}_${LEG}" ${EXTRA:+--extra-env "$EXTRA"}
  done
done
echo ">>> legs under $WORK_HOST/runs/${NAME}_${STAMP}_*  (overlays: $OVL_HOST)"
