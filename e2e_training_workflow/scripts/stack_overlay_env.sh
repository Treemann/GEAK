#!/bin/bash
# stack_overlay_env.sh -- the accepted packed-7B kernel stack as ONE reusable reference.
#
# Installs (or re-verifies) the overlays of a cumulative stack condition, writes their manifests so
# that each overlay's per-call proof lands under the given leg directory, and prints the ONE line a
# bench_train.sh leg needs:   GEAK_OVERLAY=<m1,m2,...> TORCH_EXTENSIONS_DIR=<dir>
# (stdout carries nothing else; every log line goes to stderr). Condition `ref` prints an empty line.
#
#   EXTRA="$(scripts/stack_overlay_env.sh --cond s7 --out "$OUT_HOST")"
#   scripts/bench_train.sh ... --out "$OUT_HOST" ${EXTRA:+--extra-env "$EXTRA"}
#
# This is the single definition of the stack. `measure_packed_stack.sh` uses it per leg, and the
# workflow's `baseline_overlay` arg names it, so that a new head is gated ON TOP of the accepted stack
# rather than against the bare reference (increments change when stacked: attn +4.27% bare vs +5.25%
# on the five-kernel stack).
#
# Conditions are cumulative, in stacking order:
#   ref -- no overlay
#   s2  -- rope_fused_qk + swiglu_bwd_fwd_bw      (bundle finalize_bundle_1010_rope_swiglu)
#   s3  -- + mlp_fc1_gemm                          (gate dir; impl.py proof-path patch, see below)
#   s4  -- + lmhead_ce_dense_gemm                  (bundle finalize_bundle_1013_lmhead_dense)
#   s5  -- + mlp_fc2_oproj_gemm                    (gate dir)
#   s6  -- + attn_fmha_ck_varlen                   (bundle finalize_bundle_1013_attn_ck_varlen)
#   s7  -- + qkv_lnlinear_gemm                     (gate dir; chains after fc1)
#   s8  -- + rmsnorm_bwd_fc1_fused                 (bundle finalize_bundle_1015_s7_rmsn; gated ON s7
#                                                    2026-10-15, +0.26-0.38%; rebinds TWO symbols,
#                                                    transformer_engine_torch:rmsnorm_fwd/_bwd)
#
# Coexistence rules this script enforces (measured 2026-10-13, knowledge/sft_measurement.md):
#  * Targets are disjoint except qkv and fc1, which both rebind TE `_LayerNormLinear.apply`. qkv
#    captures the binding current at its import as _PREV and delegates non-QKV calls to it, so it must
#    come AFTER fc1 -- the order below is the stacking order and is not negotiable.
#  * fc1's impl.py writes its proof next to the FIRST manifest in GEAK_OVERLAY (rope's, when stacked).
#    It is patched to find its own manifest; only that training-side rebind changes, the kernel bytes
#    L0 judged are verified unchanged. The others already locate their own manifest.
#  * fc1, fc2_oproj and qkv all ship `gemm_select.py` / `lt_gemm.cpp`; fc2_oproj and qkv pin their own
#    modules only while loading, and the extensions are named fc1_lt_gemm / fc2oproj_lt_gemm /
#    qkv_lt_gemm.
#
# Install directory. Default: a CONTENT-KEYED persistent directory
# ($WORK/overlays/stackenv_<cond>_<hash>), so repeated legs reuse one install and one kernel build
# (the CK attention instances take minutes to compile) -- and it is RE-VERIFIED against the gated
# bytes on every call, never trusted. --install DIR uses DIR instead (a fresh dir per measurement run).
# Nothing is ever rm -rf'd: the container leaves root-owned build caches behind.
#
# Manifests are named m_base.json in every overlay, so GEAK_OVERLAY -- part of impl_digest -- is the
# same string on every leg of the same condition. They are rewritten per leg (engagement_out points
# into that leg), which is safe because legs on this box never run concurrently.
set -euo pipefail
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_HOST="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
COND=""; OUT_HOST=""; OVL_HOST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cond) COND="$2"; shift 2;;
    --out) OUT_HOST="$2"; shift 2;;
    --install) OVL_HOST="$2"; shift 2;;
    -h|--help) sed -n '2,45p' "$0" >&2; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
case "$COND" in
  ref) echo ""; exit 0;;
  s2) N=2;; s3) N=3;; s4) N=4;; s5) N=5;; s6) N=6;; s7) N=7;; s8) N=8;;
  # The SECOND STACK (HF TRL, 7B): its own accepted overlays (wf_4385c413-a19, DECISIONS 4.27).
  hf1) N=1; HFSTACK=1;; hf2) N=2; HFSTACK=1;; hf3) N=3; HFSTACK=1;; hf4) N=4; HFSTACK=1;;
  # ONE overlay on the bare workload, for isolating a defect (2026-10-16 instability bisection).
  # Not a shippable condition: qkv alone is invalid (it chains after fc1) and is refused below.
  only:*) N=1; ONLY="${COND#only:}";;
  *) echo "--cond must be one of ref,s2..s8,hf1,hf2,hf3,hf4,only:<overlay> (got '$COND')" >&2; exit 2;;
esac
[ -n "$OUT_HOST" ] || { echo "--out <leg dir on the host> is required" >&2; exit 2; }
case "$OUT_HOST" in "$WORK_HOST"/*) ;; *) echo "--out must live under $WORK_HOST (mounted at /work)" >&2; exit 2;; esac
OUT_CTR="/work${OUT_HOST#$WORK_HOST}"

RS_BUNDLE="$WORK_HOST/runs/finalize_bundle_1010_rope_swiglu"
LM_BUNDLE="$WORK_HOST/runs/finalize_bundle_1013_lmhead_dense"
AT_BUNDLE="$WORK_HOST/runs/finalize_bundle_1013_attn_ck_varlen"
FC1_SRC="$WORK_HOST/overlays/mlp_fc1_gemm_1010_054626"
FC2_SRC="$WORK_HOST/overlays/mlp_fc2_oproj_gemm_1013_072624"
QKV_SRC="$WORK_HOST/overlays/qkv_lnlinear_gemm_1013_145001"
RN_BUNDLE="$WORK_HOST/runs/finalize_bundle_1015_s7_rmsn"

# name|kind|source|target|replacement attr|proof subdir[|manifest names]   -- in stacking order
# An overlay that rebinds several symbols lists ';'-separated targets, attrs and manifest names
# (m_<x>.json -> proof engagement_<x>.json). The names are the ones it was GATED with, because the
# manifest basenames are part of impl_digest. A single-symbol overlay uses m_base.json.
OVERLAYS=(
  "rope_fused_qk|bundle|$RS_BUNDLE|megatron.core.transformer.attention:apply_rotary_pos_emb|apply_rotary_pos_emb|rope"
  "swiglu_bwd_fwd_bw|bundle|$RS_BUNDLE|megatron.core.transformer.mlp:bias_swiglu_impl|bias_swiglu_impl|swiglu"
  "mlp_fc1_gemm|gate|$FC1_SRC|transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear|lnlinear_apply|fc1"
  "lmhead_ce_dense_gemm|bundle|$LM_BUNDLE|megatron.core.models.gpt.gpt_model:GPTModel._postprocess|postprocess|lmhead"
  "mlp_fc2_oproj_gemm|gate|$FC2_SRC|transformer_engine.pytorch.module.linear:_Linear|linear_apply|fc2op"
  "attn_fmha_ck_varlen|bundle|$AT_BUNDLE|transformer_engine.pytorch.attention.dot_product_attention.backends:FusedAttnFunc|fused_attn_apply|attn"
  "qkv_lnlinear_gemm|gate|$QKV_SRC|transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear|lnlinear_apply|qkv"
  "rmsnorm_bwd_fc1_fused|bundle|$RN_BUNDLE|transformer_engine_torch:rmsnorm_fwd;transformer_engine_torch:rmsnorm_bwd|rmsnorm_fwd;rmsnorm_bwd|rmsn|m_fwd.json;m_bwd.json"
)

# HF TRL stack: attn (CK/aiter flash fwd+bwd replacing SDPA), then the fused Qwen2RMSNorm. Manifest
# basenames are the gated ones (m_attn_cand.json / m_rmsn_cand.json), so impl_digest reproduces.
HF_BUNDLE="$WORK_HOST/runs/finalize_bundle_1017_hf_attn_rmsn"
HF_MLP_BUNDLE="$WORK_HOST/runs/finalize_bundle_1017_hf2_mlp"     # hf3: gated ON hf2 (wf_bda82b0e-152)
# hf4: down_proj, gated ON hf3 (wf_499a73c3-cd2), re-adjudicated after the warmup fix (DECISIONS 4.34).
# It rebinds Qwen2MLP.forward too and CHAINS: _PREV is the gate/up wrapper, so it must come after hf3.
HF_DOWN_SRC="$WORK_HOST/overlays/hf_mlp_down_proj_1018_023501/hf_mlp_down_proj"
if [ -n "${HFSTACK:-}" ]; then
  OVERLAYS=(
    "attn_fmha_ck_hf|bundle|$HF_BUNDLE|torch.nn.functional:scaled_dot_product_attention|entry|attn|m_attn_cand.json"
    "hf_rmsnorm_fused_fwdbwd|bundle|$HF_BUNDLE|transformers.models.qwen2.modeling_qwen2:Qwen2RMSNorm.forward|entry|rmsn|m_rmsn_cand.json"
    "hf_mlp_gateup_swiglu|bundle|$HF_MLP_BUNDLE|transformers.models.qwen2.modeling_qwen2:Qwen2MLP.forward|entry|mlp|m_mlp_cand.json"
    "hf_mlp_down_proj|gate|$HF_DOWN_SRC|transformers.models.qwen2.modeling_qwen2:Qwen2MLP.forward|entry|down|m_down_cand.json"
  )
fi
if [ -n "${ONLY:-}" ]; then
  [ "$ONLY" = qkv_lnlinear_gemm ] && { echo "only:qkv_lnlinear_gemm is invalid: qkv chains after fc1" >&2; exit 2; }
  SEL=()
  for o in "${OVERLAYS[@]}"; do [ "${o%%|*}" = "$ONLY" ] && SEL+=("$o"); done
  [ "${#SEL[@]}" -eq 1 ] || { echo "only: unknown overlay '$ONLY'" >&2; exit 2; }
  OVERLAYS=("${SEL[@]}")
  COND="only_$ONLY"
fi
if [ -z "$OVL_HOST" ]; then
  KEY="$( { echo "$COND"; for ((k = 0; k < N; k++)); do
             IFS='|' read -r _ KIND SRC _ <<< "${OVERLAYS[$k]}"
             if [ "$KIND" = bundle ]; then cat "$SRC/SHA256SUMS.overlay"; else echo "$SRC"; fi
           done; } | sha256sum | cut -c1-12 )"
  OVL_HOST="$WORK_HOST/overlays/stackenv_${COND}_$KEY"
fi
OVL_CTR="/work${OVL_HOST#$WORK_HOST}"
mkdir -p "$OVL_HOST/build"

# Install what is missing, then verify EVERYTHING (new or reused) against the gated bytes.
python3 - "$OVL_HOST" "$N" "${OVERLAYS[@]}" <<'PY' >&2
import hashlib, os, shutil, sys
ovl, n, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
SKIP = {"__pycache__", ".torch_ext", "build"}
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
FC1_OLD = '''_OUT = ""
try:
    _OUT = json.load(open(os.environ["GEAK_OVERLAY"].split(",")[0]))["engagement_out"]
    _OUT = os.path.join(os.path.dirname(_OUT), "engagement_fused.json")
except Exception:
    _OUT = os.environ.get("GEAK_FC1_ENGAGEMENT", "")'''
FC1_NEW = '''_OUT = ""
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
bad = []
for spec in specs[:n]:
    name, kind, src = spec.split("|")[:3]
    dst = os.path.join(ovl, name)
    if kind == "bundle":
        sums = [l.split(None, 1) for l in open(os.path.join(src, "SHA256SUMS.overlay")) if l.strip()]
        want = {os.path.normpath(p.strip()): h for h, p in sums if os.path.normpath(p.strip()).split(os.sep)[0] == name}
        if not want:
            sys.exit(f"!!! {src}/SHA256SUMS.overlay lists no files for {name}")
        if not os.path.isdir(dst):
            shutil.copytree(os.path.join(src, "overlays", name), dst,
                            ignore=shutil.ignore_patterns(*SKIP, "manifest_*", "m_*.json"))
    else:   # a gate directory: everything but manifests and build caches, verified file by file
        want = {}
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for f in files:
                if f.startswith("manifest_") or f.startswith("m_") or f.endswith(".pyc"):
                    continue
                rel = os.path.relpath(os.path.join(root, f), src)
                want[os.path.join(name, rel)] = sha(os.path.join(root, f))
        if not os.path.isdir(dst):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP, "manifest_*", "m_*.json", "*.pyc"))
            if name == "mlp_fc1_gemm":
                p = os.path.join(dst, "impl.py"); s = open(p).read()
                if FC1_OLD not in s:
                    sys.exit("!!! fc1 impl.py proof-path block not found -- refusing an unpatched stack")
                open(p, "w").write(s.replace(FC1_OLD, FC1_NEW, 1))
        if name == "mlp_fc1_gemm":   # the one sanctioned difference: the patched impl.py
            p = os.path.join(dst, "impl.py")
            if FC1_NEW not in open(p).read():
                bad.append(f"{p}: fc1 proof-path patch missing")
            want.pop(os.path.join(name, "impl.py"), None)
    for rel, h in want.items():
        p = os.path.join(ovl, rel)
        if not os.path.isfile(p):
            bad.append(f"{p}: missing")
        elif sha(p) != h:
            bad.append(f"{p}: sha256 differs from the gated bytes")
    print(f">>> {name}: {len(want)} gated files verified in {dst}")
if bad:
    sys.exit("!!! installed stack differs from the gated one -- refusing:\n  " + "\n  ".join(bad))
PY

LIST=""
for ((k = 0; k < N; k++)); do
  IFS='|' read -r NAME _ _ TARGETS ATTRS SUB MANS <<< "${OVERLAYS[$k]}"
  mkdir -p "$OUT_HOST/$SUB" "$OVL_HOST/$NAME/build"
  IFS=';' read -r -a T_ARR <<< "$TARGETS"; IFS=';' read -r -a A_ARR <<< "$ATTRS"
  IFS=';' read -r -a M_ARR <<< "${MANS:-m_base.json}"
  [ "${#T_ARR[@]}" -eq "${#A_ARR[@]}" ] && [ "${#T_ARR[@]}" -eq "${#M_ARR[@]}" ] || {
    echo "!!! $NAME: targets/attrs/manifests counts differ" >&2; exit 3; }
  for ((m = 0; m < ${#T_ARR[@]}; m++)); do
    MAN="${M_ARR[$m]}"; TAG="${MAN#m_}"; TAG="${TAG%.json}"
    if [ "$MAN" = m_base.json ]; then PROOF=engagement.json; else PROOF="engagement_$TAG.json"; fi
    cat > "$OVL_HOST/$NAME/$MAN" <<EOF
{"target": "${T_ARR[$m]}",
 "replacement": "$OVL_CTR/$NAME/impl.py:${A_ARR[$m]}",
 "engagement_out": "$OUT_CTR/$SUB/$PROOF"}
EOF
    LIST="${LIST:+$LIST,}$OVL_CTR/$NAME/$MAN"
  done
done
echo ">>> stack $COND: $N overlays from $OVL_HOST, proofs under $OUT_HOST/{$(for ((k = 0; k < N; k++)); do IFS='|' read -r _ _ _ _ _ SUB _ <<< "${OVERLAYS[$k]}"; printf '%s,' "$SUB"; done | sed 's/,$//')}" >&2
echo "GEAK_OVERLAY=$LIST TORCH_EXTENSIONS_DIR=$OVL_CTR/build"
