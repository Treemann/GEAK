# HF transformers + TRL SFT adapter for bench_train.sh. Sourced, not executed.
#
# The second implementation of the adapter seam. Same contract as megatron.sh: reads the dispatcher's
# variables, sets JOB_PID, keeps everything stack-specific in this file and in hf_sft.py.
#
# Reads: TOKENIZER_DIR DATA_PATH OUT_DIR PROBE_DIR NPROC ITERS MBS GBS SEQ EXTRA_TRAIN_ARGS EXTRA_ENV
#        GEAK_SCRIPTS, and optionally HF_MODEL_DIR.
#
# WEIGHTS: MODEL_CKPT in a workload descriptor names Megatron's torch_dist checkpoint, which HF cannot
# load. The HF-format weights are the tokenizer directory itself on every workload in this repo
# (qwen2_5_0_5b, qwen2_5_math_7b), so that is the default; HF_MODEL_DIR overrides it. Recorded as an
# adapter-seam finding: the workload descriptor's MODEL_CKPT is Megatron vocabulary, and a stack-
# neutral descriptor would name the weights by FORMAT.
#
# RECIPE KNOBS ARE ARGUMENTS, NOT ENVIRONMENT. bench_train.sh re-executes inside the training image
# and forwards only a short list of env vars (ATTENTION_BACKEND, RETRY_CONTAMINATED). An adapter env
# knob set on the host therefore reaches the container as nothing at all -- measured: HF_FSDP=
# full_shard on the host produced a leg whose captured argv says `hf-fsdp none`, and it OOMed exactly
# like the DDP leg. Defaults are written out here; override with
#   bench_train.sh ... --extra-train-args "--hf-fsdp full_shard"
# which is forwarded verbatim and, being appended last, wins the argparse duplicate.
#
# ATTENTION_BACKEND here is HF's attn_implementation (sdpa | flash_attention_2 | eager), not
# Megatron's fused/flash. Same knob name, different value set -- a second place the config IR is
# shown to be Megatron's argv rather than an IR.

adapter_capabilities() {
  cat <<'CAP'
{
  "step_ranged_profiler": false,
  "probe_driven_profiler": true,
  "reports_tokens_per_s": true,
  "_reports_tokens_per_s_note": "TRL logs cumulative num_tokens and the Trainer reports train_runtime; the ratio is a NOMINAL (padding-inclusive) cross-check of the probe, never the gate.",
  "reports_grad_norm_per_step": true,
  "reports_peak_memory": true,
  "in_job_reconfiguration": false,
  "deterministic_mode": false,
  "config_form": "python_trainingarguments (translated from the Megatron CLI vocabulary by hf_sft.py)",
  "max_micro_batch_size_packed_sft": null,
  "padding": "dynamic (pad to longest in micro-batch) -- Megatron pads every sample to SEQ"
}
CAP
}

adapter_launch() {
  mkdir -p "$OUT_DIR" "$PROBE_DIR"
  local model_dir="${HF_MODEL_DIR:-$TOKENIZER_DIR}"
  # shellcheck disable=SC2086
  env $EXTRA_ENV \
    PYTHONPATH="$GEAK_SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" \
    GEAK_PROBE_DIR="$PROBE_DIR" \
    ${PROFILE_STEPS:+GEAK_PROFILE_STEPS="$PROFILE_STEPS"} \
    ${PROFILE_STEPS:+GEAK_PROFILE_RANKS="${PROFILE_RANKS:-0}"} \
    TOKENIZERS_PARALLELISM=false \
    torchrun --nproc_per_node="$NPROC" --nnodes=1 \
      --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29521}" \
      "$GEAK_SCRIPTS/adapters/hf_sft.py" \
      --geak-stack hf_trl \
      --load "$model_dir" --tokenizer-model "$TOKENIZER_DIR" \
      --train-data-path "$DATA_PATH" \
      --micro-batch-size "$MBS" --global-batch-size "$GBS" --seq-length "$SEQ" \
      --train-iters "$ITERS" \
      --lr "${LR:-1.0e-5}" --min-lr "${MIN_LR:-0.0}" --lr-decay-style "${LR_DECAY_STYLE:-constant}" \
      --lr-warmup-iters "${LR_WARMUP_ITERS:-0}" ${LR_DECAY_ITERS:+--lr-decay-iters "$LR_DECAY_ITERS"} \
      --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
      --bf16 --sft --seed "${SEED:-1234}" --master-weights fp32 --hf-fsdp "${HF_FSDP:-none}" \
      --attention-backend "${ATTENTION_BACKEND:-sdpa}" \
      --output-dir "$OUT_DIR/hf_out" \
      $EXTRA_TRAIN_ARGS \
      > "$OUT_DIR/train.log" 2>&1 &
  JOB_PID=$!
}

# The stack's OWN figure: nominal tokens/s/GPU from TRL's cumulative num_tokens over train_runtime.
# Cross-check only. It includes warmup and counts every token the collator kept, not loss tokens.
adapter_metrics() {
  python3 - "$OUT_DIR/train.log" "$NPROC" <<'PY' 2>/dev/null
import re, sys
t = open(sys.argv[1], errors="replace").read()
nt = re.findall(r"'num_tokens': '?([0-9.e+]+)", t)
rt = re.findall(r"'train_runtime': '?([0-9.]+)", t)
if nt and rt:
    print(round(float(nt[-1]) / float(rt[-1]) / int(sys.argv[2]), 1))
PY
}

adapter_teardown() {
  local pid="${1:-$JOB_PID}"
  [ -n "$pid" ] || return 0
  kill -TERM "-$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  sleep 5
  kill -KILL "$pid" 2>/dev/null || true
  pkill -KILL -f "adapters/hf_sft.py" 2>/dev/null || true
  sleep 2
}
