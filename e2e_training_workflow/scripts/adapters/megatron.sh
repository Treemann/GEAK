# Bare Megatron-LM SFT adapter for bench_train.sh. Sourced, not executed.
#
# Reads from the dispatcher: MLM_DIR MODEL_CKPT TOKENIZER_DIR DATA_PATH OUT_DIR PROBE_DIR
#   NPROC ITERS MBS GBS SEQ ARCH_ARGS ROPE_BASE EXTRA_TRAIN_ARGS EXTRA_ENV GEAK_SCRIPTS
# Sets JOB_PID. Everything Megatron-specific lives here; the dispatcher stays stack-neutral.
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

adapter_launch() {
  mkdir -p "$OUT_DIR" "$PROBE_DIR"
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
      --transformer-impl transformer_engine --attention-backend fused \
      --tokenizer-type SFTTokenizer --tokenizer-model "$TOKENIZER_DIR" \
      --load "$MODEL_CKPT" --finetune --ckpt-format torch_dist --no-load-optim --no-load-rng \
      --sft --sft-tokenizer-prompt-format "${PROMPT_FORMAT:-default}" \
      --no-create-attention-mask-in-dataloader \
      --train-data-path "$DATA_PATH" \
      --micro-batch-size "$MBS" --global-batch-size "$GBS" \
      --train-iters "$ITERS" \
      --lr 1.0e-5 --min-lr 0.0 --lr-decay-style constant --lr-warmup-iters 0 \
      --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
      --bf16 --use-distributed-optimizer --no-rope-fusion \
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
