#!/bin/bash
# measure_stack3.sh -- does mlp_fc1_gemm (+1.60%, re-gated 2026-10-10) add on top of the accepted
# rope_fused_qk + swiglu_bwd_fwd_bw stack (+7.19% validated) on the packed 7B workload?
#
# Three conditions, interleaved so node drift cannot align with condition:
#   ref  -- packed reference, no overlay
#   s2   -- the accepted bundle: rope, swiglu
#   s3   -- rope, swiglu, fc1
# s3 vs s2 is fc1's increment on top of the accepted stack; s3 vs ref is the total.
#
# WHY fc1's impl.py IS PATCHED (and why that does not ungate it). rope's and fc1's impl.py both write
# their per-call proof, `engagement_fused.json.rank*`, next to the FIRST manifest in GEAK_OVERLAY.
# Stacked, fc1 would overwrite rope's proof and the engagement check would read one overlay's counters
# as the other's. swiglu already finds its own manifest by its own __file__; fc1 is patched to do the
# same. Only impl.py -- the training-side rebind -- changes. L0 judged kernel_src/ (kernel.py,
# gemm_select.py, bwd_overlap.py, lt_gemm.cpp, lt_solutions.csv), whose bytes are copied verbatim and
# whose sha256 the script prints.
#
# Usage: measure_stack3.sh [--reps 2] [--iters 30]
set -euo pipefail
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
REPS=2; ITERS=30
while [ $# -gt 0 ]; do
  case "$1" in
    --reps) REPS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
BUNDLE="$WORK_HOST/runs/finalize_bundle_1010_rope_swiglu"
FC1_SRC="$WORK_HOST/overlays/mlp_fc1_gemm_1010_054626"
WORKLOAD="$GEAK_ROOT/workloads/qwen2_5_math_7b_packed.env"
STAMP="$(date +%m%d_%H%M%S)"
OVL_HOST="$WORK_HOST/overlays/stack3_$STAMP"
OVL_CTR="/work/overlays/stack3_$STAMP"
[ -e "$OVL_HOST" ] && { echo "!!! $OVL_HOST exists" >&2; exit 3; }
mkdir -p "$OVL_HOST/build"

# ---- install rope + swiglu exactly as the bundle does, verifying the gated bytes ---------------
cp -r "$BUNDLE/overlays/rope_fused_qk" "$BUNDLE/overlays/swiglu_bwd_fwd_bw" "$OVL_HOST/"
( cd "$OVL_HOST" && sha256sum -c --quiet "$BUNDLE/SHA256SUMS.overlay" ) || {
  echo "!!! installed rope/swiglu bytes differ from the gated ones -- refusing" >&2; exit 3; }
echo ">>> rope + swiglu bytes verified against $BUNDLE/SHA256SUMS.overlay"

# ---- install fc1: kernel bytes verbatim, impl.py patched for its proof path only ---------------
mkdir -p "$OVL_HOST/mlp_fc1_gemm"
for f in "$FC1_SRC"/*; do
  case "$(basename "$f")" in manifest_*|__pycache__|.torch_ext) continue;; esac
  cp -r "$f" "$OVL_HOST/mlp_fc1_gemm/"
done
echo ">>> fc1 kernel bytes (L0 judged these):"
( cd "$OVL_HOST/mlp_fc1_gemm" && sha256sum kernel.py gemm_select.py bwd_overlap.py lt_gemm.cpp lt_solutions.csv | sed 's/^/      /' )
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

manifest() {  # $1 dir-name, $2 target, $3 replacement attr, $4 out-ctr, $5 subdir, $6 file
  cat > "$OVL_HOST/$1/$6" <<EOF
{"target": "$2",
 "replacement": "$OVL_CTR/$1/impl.py:$3",
 "engagement_out": "$4/$5/engagement.json"}
EOF
}

for i in $(seq 1 "$REPS"); do
  for MODE in ref s2 s3; do
    OUT_HOST="$WORK_HOST/runs/stack3_${STAMP}_${MODE}${i}"
    OUT_CTR="/work/runs/stack3_${STAMP}_${MODE}${i}"
    EXTRA=""
    if [ "$MODE" != ref ]; then
      mkdir -p "$OUT_HOST/rope" "$OUT_HOST/swiglu" "$OUT_HOST/fc1"
      manifest rope_fused_qk "megatron.core.transformer.attention:apply_rotary_pos_emb" apply_rotary_pos_emb "$OUT_CTR" rope "m_${MODE}${i}.json"
      manifest swiglu_bwd_fwd_bw "megatron.core.transformer.mlp:bias_swiglu_impl" bias_swiglu_impl "$OUT_CTR" swiglu "m_${MODE}${i}.json"
      LIST="$OVL_CTR/rope_fused_qk/m_${MODE}${i}.json,$OVL_CTR/swiglu_bwd_fwd_bw/m_${MODE}${i}.json"
      if [ "$MODE" = s3 ]; then
        manifest mlp_fc1_gemm "transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear" lnlinear_apply "$OUT_CTR" fc1 "m_${MODE}${i}.json"
        LIST="$LIST,$OVL_CTR/mlp_fc1_gemm/m_${MODE}${i}.json"
      fi
      EXTRA="GEAK_OVERLAY=$LIST TORCH_EXTENSIONS_DIR=$OVL_CTR/build"
    fi
    echo ">>> $(date +%H:%M:%S) leg ${MODE}${i} -> $OUT_HOST"
    "$GEAK_ROOT/scripts/bench_train.sh" --workload "$WORKLOAD" --stack megatron --nproc 8 \
      --iters "$ITERS" --out "$OUT_HOST" --tag "stack3_${MODE}${i}" ${EXTRA:+--extra-env "$EXTRA"}
  done
done
echo ">>> legs under $WORK_HOST/runs/stack3_${STAMP}_*  (overlays: $OVL_HOST)"
