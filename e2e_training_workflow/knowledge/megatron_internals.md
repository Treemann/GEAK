# Bare Megatron-LM on ROCm: what it takes to run SFT at all

Target: `pretrain_gpt.py --sft` from `/workspace/Primus/third_party/Megatron-LM` inside
`rocm/primus:v26.4`. The image is an **environment**, not a dependency: Primus's trainer and config
layers are bypassed entirely, and nothing outside `scripts/adapters/` may reference a Primus path.

## Three things that do not work out of the box

### 1. Upstream Megatron cannot initialize on ROCm
`initialize_megatron` -> `_compile_dependencies()` -> `megatron.legacy.fused_kernels.load()` ->
`_get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)`. `CUDA_HOME` is `None` on ROCm, so the path
dies on `None + "/bin/nvcc"`. It is a CUDA-only path (it builds `.cu` fused softmax kernels).
`scripts/rocm_compat.py` no-ops it, exactly as Primus does
(`primus/backends/megatron/patches/runtime_hooks_patches.py`, `disable_compile_dependencies: true`).

So "bare Megatron" really means **upstream + a small set of ROCm compatibility patches**. The set is
small and well-bounded (1 of Primus's 23 registered patches is of this kind), and those patches
belong in our own shim/overlay, not in a dependency on Primus.

### 2. Upstream `--sft` THD packing is self-inconsistent on this commit
`SFTDataset.__getitem__` returns a **per-sample 1D** `cu_seqlens`; the default collate stacks it to
`[MBS, n+1]`; `get_thd_batch_on_this_cp_rank` forwards that 2D tensor into `PackedSeqParams`; and
Megatron's own rope helpers then assume 1D (`cu_seqlens[-1]`):
- fused path -> `RuntimeError: expected 1D tensor`
- non-fused path -> `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`

Not a ROCm issue; Primus never hits it because it ships its own SFT dataset. `rocm_compat.py`
squeezes the vacuous batch dim at **MBS=1** and **refuses MBS>1 loudly** — merging rows needs offset
re-basing plus flattening tokens into one sequence, which is a semantics change, not a workaround.
Raise the global batch size with gradient accumulation instead.

### 3. Upstream ships no profiler arguments at all
`grep -c profile megatron/training/arguments.py` -> `0`. The `--profile` family in Primus recipes is
Primus's own addition. Profiling is therefore **probe-driven** (`GEAK_PROFILE_STEPS=start:end`),
which is stack-neutral and costs the adapter nothing.

## The working argument set (see `scripts/adapters/megatron.sh`)
Non-obvious ones and why:
- `--tokenizer-type SFTTokenizer` — `HuggingFaceTokenizer` raises
  `NotImplementedError: This method is supported only for SFTTokenizer` at the first batch.
- `--attention-backend fused` — the image sets `NVTE_FLASH_ATTN=0`, `NVTE_FUSED_ATTN=1` (TE's CK
  fused path on ROCm); `auto` asserts against that env.
- `--no-create-attention-mask-in-dataloader` — the SFT dataset asserts it is off.
- `--no-rope-fusion` — the TE fused thd RoPE path fails on this TE/Megatron pair.
- `--eval-interval` must be set even with `--eval-iters 0`: `get_train_valid_test_num_samples()`
  computes `train_iters // eval_interval` unconditionally.
- `--sft-tokenizer-prompt-format default` — the other three formats (`nemotron-h-aligned` is the
  **default**, plus `nemotron-nano-v2`, `identity`) hard-code Nemotron chat templates and pad tokens
  (`<SPECIAL_233>`, `<unk>`) that do not exist in Qwen's vocabulary. `default` uses the tokenizer's
  own chat template and pad token, but returns early before the masking loop, so **prompt tokens are
  not masked**. That does not touch the compute graph, so throughput and L1 stay valid; it only costs
  SFT semantic realism. The cheap upgrade is a `qwen` branch: same `PromptConfig` as `default`
  without the early return, `assistant_prefix_len=3` for `<|im_start|>assistant\n`; the masking loop
  itself is format-agnostic, so it is ~5 lines and it never touches the performance path.
- `--disable-last-saving` does NOT exist upstream (it is a Primus argument).

## Model arguments come from the checkpoint, never from guesswork
`release/common.pt` carries the full `args` namespace used at conversion. Dump it with
`scripts/dump_ckpt_args.py`-style loading (`PYTHONPATH` must include the Megatron tree, since the
pickle references `megatron.*` classes) and copy `num_layers / hidden_size / ffn_hidden_size /
num_attention_heads / num_query_groups / kv_channels / rotary_base /
untie_embeddings_and_output_weights` into a `workloads/*.env` descriptor.
