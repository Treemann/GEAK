# Bare Megatron-LM SFT adapter for bench_train.sh. Sourced, not executed.
#
# Reads from the dispatcher: MLM_DIR MODEL_CKPT TOKENIZER_DIR DATA_PATH OUT_DIR PROBE_DIR
#   NPROC ITERS MBS GBS SEQ ARCH_ARGS ROPE_BASE EXTRA_TRAIN_ARGS EXTRA_ENV GEAK_SCRIPTS
# Sets JOB_PID. Everything Megatron-specific lives here; the dispatcher stays stack-neutral.
#
# ATTENTION_BACKEND is overridable because it is a CONFIG choice, not an adapter fact: the image
# ships NVTE_FUSED_ATTN=1 / NVTE_FLASH_ATTN=0, and Megatron asserts if the CLI backend disagrees with
# that env. Hardcoding `fused` here made an entire measurement leg unusable the first time a
# different backend was needed. It belongs to the config IR once ConfigSweep exists.
#
# ROCm/upstream compatibility is NOT optional here (see scripts/rocm_compat.py): upstream Megatron
# cannot initialize on ROCm, and its --sft THD path mis-shapes cu_seqlens. Both are applied by the
# launcher shim, which is also where the measurement probe is installed.

adapter_capabilities() {
  cat <<'CAP'
{
  "step_ranged_profiler": false,
  "_step_ranged_profiler_note": "upstream Megatron ships NO profiler arguments; the --profile family in Primus recipes is Primus's own. Profiling is probe-driven instead (GEAK_PROFILE_STEPS), which works on any stack.",
  "probe_driven_profiler": true,
  "reports_tokens_per_s": false,
  "reports_grad_norm_per_step": true,
  "reports_peak_memory": true,
  "in_job_reconfiguration": false,
  "deterministic_mode": true,
  "config_form": "cli_args",
  "max_micro_batch_size_packed_sft": 1
}
CAP
}

# RMSNorm epsilon preflight (DECISIONS 4.55). Megatron never reads the epsilon from the checkpoint
# (common.pt args carry norm_epsilon None) and defaults to 1e-5; the HF config next to the tokenizer
# says what the model was pretrained with (Qwen2.5: 1e-6). The two are DIFFERENT RECIPES and the choice
# is measured, not assumed: P0-4l found the model-matched 1e-6 WORSE on held-out loss (+0.0067, 5/5
# seeds) and less stable, so the production recipe keeps 1e-5. This check therefore never refuses: a
# mismatch is printed and recorded in $OUT_DIR/recipe_warnings.json so no leg runs it silently.
adapter_preflight() {
  local cfg="$TOKENIZER_DIR/config.json" want
  [ -r "$cfg" ] || return 0
  want=$(python3 -c "import json,sys; v=json.load(open(sys.argv[1])).get('rms_norm_eps'); print('' if v is None else repr(float(v)))" "$cfg" 2>/dev/null) || return 0
  [ -n "$want" ] || return 0
  local got="${NORM_EPS:-1e-05}" arm="" prev=""
  # A per-leg --norm-epsilon in EXTRA_TRAIN_ARGS (argparse: last wins) is a deliberate experiment ARM
  # (e.g. an outcome test of the epsilon itself): recorded and printed under its own note.
  for t in $EXTRA_TRAIN_ARGS; do
    [ "$prev" = "--norm-epsilon" ] && arm="$t"
    case "$t" in --norm-epsilon=*) arm="${t#--norm-epsilon=}";; esac
    prev="$t"
  done
  if [ -n "$arm" ]; then
    got="$arm"
    if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)" "$got" "$want"; then
      echo ">>> RECIPE WARNING: this leg's --norm-epsilon $arm (extra train args) differs from the model's rms_norm_eps $want: an experiment arm, recorded in recipe_warnings.json" >&2
      printf '{"norm_epsilon": {"run": %s, "model_config": %s, "source": "%s", "note": "per-leg --norm-epsilon in extra train args (experiment arm)"}}\n' "$arm" "$want" "$cfg" > "$OUT_DIR/recipe_warnings.json"
    fi
    return 0
  fi
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)" "$got" "$want"; then
    return 0
  fi
  local src="Megatron default (descriptor sets no NORM_EPS)"
  [ -n "${NORM_EPS:-}" ] && src="descriptor NORM_EPS"
  echo ">>> RECIPE NOTE: Megatron runs RMSNorm eps $got ($src) while the model's config says rms_norm_eps $want ($cfg). Deliberate for the production recipe: 1e-6 measured worse (P0-4l, DECISIONS 4.55)." >&2
  printf '{"norm_epsilon": {"run": %s, "model_config": %s, "source": "%s", "note": "%s"}}\n' "$got" "$want" "$cfg" "$src" > "$OUT_DIR/recipe_warnings.json"
}

adapter_launch() {
  mkdir -p "$OUT_DIR" "$PROBE_DIR"
  # RoPE fusion is a Class-A IMPLEMENTATION lever (same rotation, TE/apex fused kernel). Megatron
  # only has the negative flag, so a lever cannot ADD it: `--rope-fusion` in EXTRA_TRAIN_ARGS is a
  # GEAK pseudo-flag, stripped here, that drops the recipe's --no-rope-fusion. The probe records the
  # realized state as `rope-fusion` (identity v6, implementation half).
  local rope_flag="--no-rope-fusion" extra=""
  for t in $EXTRA_TRAIN_ARGS; do
    if [ "$t" = "--rope-fusion" ]; then rope_flag=""; else extra="$extra $t"; fi
  done
  EXTRA_TRAIN_ARGS="${extra# }"
  # shellcheck disable=SC2086
  env $EXTRA_ENV \
    PYTHONPATH="$MLM_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    GEAK_PROBE_DIR="$PROBE_DIR" \
    ${PROFILE_STEPS:+GEAK_PROFILE_STEPS="$PROFILE_STEPS"} \
    ${PROFILE_STEPS:+GEAK_PROFILE_RANKS="${PROFILE_RANKS:-0}"} \
    torchrun --nproc_per_node="$NPROC" --nnodes=1 \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29511}" \
      "$GEAK_SCRIPTS/rocm_compat.py" "$MLM_DIR/pretrain_gpt.py" \
      $ARCH_ARGS \
      --seq-length "$SEQ" --max-position-embeddings "$SEQ" \
      --normalization RMSNorm --swiglu --position-embedding-type rope --rotary-base "$ROPE_BASE" \
      --disable-bias-linear --add-qkv-bias \
      --transformer-impl transformer_engine --attention-backend "${ATTENTION_BACKEND:-fused}" \
      --tokenizer-type SFTTokenizer --tokenizer-model "$TOKENIZER_DIR" \
      --load "$MODEL_CKPT" --finetune --ckpt-format torch_dist --no-load-optim --no-load-rng \
      --sft --sft-tokenizer-prompt-format "${PROMPT_FORMAT:-default}" \
      --no-create-attention-mask-in-dataloader \
      --train-data-path "$DATA_PATH" \
      --micro-batch-size "$MBS" --global-batch-size "$GBS" \
      --train-iters "$ITERS" \
      --lr "${LR:-1.0e-5}" --min-lr "${MIN_LR:-0.0}" --lr-decay-style "${LR_DECAY_STYLE:-constant}" \
      --lr-warmup-iters "${LR_WARMUP_ITERS:-0}" ${LR_DECAY_ITERS:+--lr-decay-iters "$LR_DECAY_ITERS"} \
      ${DDP_EXTRA_ARGS:-} \
      --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
      --bf16 --use-distributed-optimizer $rope_flag ${NORM_EPS:+--norm-epsilon "$NORM_EPS"} \
      --log-interval 1 --log-throughput \
      --eval-iters 0 --eval-interval 1000000 --save-interval 1000000 \
      --seed "${SEED:-1234}" --distributed-timeout-minutes "${DIST_TIMEOUT_MIN:-10}" \
      $EXTRA_TRAIN_ARGS \
      > "$OUT_DIR/train.log" 2>&1 &
  JOB_PID=$!
}

# The stack's OWN reported numbers -- cross-check only, never the gate (upstream reports no tokens/s
# at all, and stacks that do report it use a nominal, padding-inclusive definition).
adapter_metrics() {
  grep -oE "throughput per GPU \(TFLOP/s/GPU\): +[0-9.]+" "$OUT_DIR/train.log" 2>/dev/null \
    | awk '{print $NF}' | tail -20 | sort -n | awk '{v[NR]=$1} END {if(NR) print v[int((NR+1)/2)]}'
}

adapter_teardown() {
  local pid="${1:-$JOB_PID}"
  [ -n "$pid" ] || return 0
  kill -TERM "-$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  sleep 5
  kill -KILL "$pid" 2>/dev/null || true
  # A dead rank can leave siblings blocked in a collective; they hold GPU memory and will poison the
  # next leg, so sweep by name rather than trusting the parent to reap them.
  pkill -KILL -f "pretrain_gpt.py" 2>/dev/null || true
  sleep 2
}
