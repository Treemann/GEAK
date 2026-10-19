#!/usr/bin/env python3
"""HF transformers + TRL SFT launcher -- the second training stack. Launched by adapters/hf_trl.sh.

WHY THIS STACK. Month 2's first item is to validate the adapter seam with a second implementation,
because a seam with one implementation is a claim, not a seam. HF TRL was chosen over TorchTitan
(also present in the image) for reasons that are about what the experiment can attribute:

  * SAME image, SAME weights (HF format already on disk), SAME dataset. Any difference measured is
    the stack, not the environment.
  * It REPORTS its own token counts, which is what lets us check whether the probe's useful-token
    figure is genuinely stack-neutral rather than Megatron's definition under another name.
  * Its configuration vocabulary (TrainingArguments) is nothing like Megatron's CLI, so it tests
    whether the config IR is an IR or Megatron's argv.
  * TorchTitan is pretraining-oriented with no native SFT path; measuring it would mean measuring a
    data pipeline written here, which confounds the stack with our own code.

ONE THING DELIBERATELY CARRIED OVER: THE CLI VOCABULARY. This launcher accepts Megatron's argument
names (--micro-batch-size, --global-batch-size, --seq-length, --lr, --seed, ...) and translates them.
That is not convenience. `train_probe._config_snapshot` captures identity from argv by those names,
and speaking them here means the identity machinery works unchanged. It is ALSO the first concrete
answer to the roadmap's question "is the config IR an IR, or Megatron vocabulary in an IR's
clothes?" -- it is the latter, and this file is where the translation lives. Recorded, not hidden.

`--geak-stack` is added to the argv for exactly one reason: so that a Megatron leg and an HF leg
with otherwise identical arguments CANNOT share a config_digest. Without it they would, and the gate
would happily compare a Megatron reference against an HF candidate as "the same workload".

LOSS SEMANTICS MATCH MEGATRON'S `--sft-tokenizer-prompt-format default`, which masks NOTHING: the
whole conversation, rendered by the tokenizer's own chat template, is the target, truncated to SEQ.
Megatron's loss_mask therefore counts (non-pad length - 1) per sample because of the shift; the HF
probe binding counts `labels[:, 1:] != -100` for the same reason. Counting `labels != -100` instead
would over-count by one per sample -- small, and exactly the kind of small that makes two stacks'
useful tok/s quietly incomparable.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # scripts/, for train_probe


def _decoder_layer_cls(model_dir):
    """FSDP wraps per decoder layer; the class name comes from the model's own config, not a guess."""
    import json
    arch = json.load(open(os.path.join(model_dir, "config.json")))["architectures"][0]
    return arch.replace("ForCausalLM", "DecoderLayer")       # Qwen2ForCausalLM -> Qwen2DecoderLayer


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Megatron vocabulary, translated below. Names must match train_probe._IDENTITY_PAIR_KEYS.
    ap.add_argument("--load", required=True, help="HF model directory")
    ap.add_argument("--tokenizer-model", required=True)
    ap.add_argument("--train-data-path", required=True)
    ap.add_argument("--micro-batch-size", type=int, required=True)
    ap.add_argument("--global-batch-size", type=int, required=True)
    ap.add_argument("--seq-length", type=int, required=True)
    ap.add_argument("--train-iters", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1.0e-5)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--lr-decay-style", default="constant")
    ap.add_argument("--lr-warmup-iters", type=int, default=0)
    ap.add_argument("--lr-decay-iters", type=int, default=None)
    # Held-out evaluation ONCE, after training (the outcome test, quality_ab.sh --stack hf_trl). The
    # result is printed in Megatron's format so the same analysis (quality_ab_final.py) reads both.
    ap.add_argument("--valid-data-path", default=None)
    # TRL's SFTTrainer.compute_loss adds, for EVERY micro-batch, two logging-only metrics: num_tokens
    # and mean_token_accuracy. That is 3 gather_for_metrics + 2 .item() host syncs plus an argmax over
    # the full [seq, vocab] logits (and a contiguous copy of them). The loss itself comes from
    # Trainer.compute_loss and is untouched. The gathers make every micro-batch a cross-rank barrier:
    # 32.9% of the 7B ZeRO-2 step was ranks waiting there (wf_bda82b0e-152). `off` skips them.
    ap.add_argument("--hf-train-metrics", choices=("on", "off"), default="on")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--attention-backend", default="sdpa",
                    help="HF attn_implementation: sdpa | flash_attention_2 | eager")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--sft", action="store_true")
    ap.add_argument("--geak-stack", default="hf_trl")
    # Every recipe choice this launcher makes must be VISIBLE IN ARGV, because identity is captured
    # from argv. The first version switched pure-bf16 -> fp32-master in code with argv unchanged, and
    # two legs training different recipes carried identical config_digest AND impl_digest.
    # Parallel strategy. DDP replicates fp32 master weights, grads and AdamW state on every GPU and
    # OOMs at 7B (183.5 GB allocated at step 3 on a 192 GB device). full_shard is FSDP with fp32
    # master shards and bf16 compute -- the counterpart of Megatron's --use-distributed-optimizer.
    # shard_grad_op = ZeRO-2: params stay gathered for the whole optimizer step (no per-micro-batch
    # re-gather), grads + optimizer state sharded. Added 2026-10-17: full_shard exposed 30% of the 7B
    # step as un-hidden all-gathers (8 micro-batches each re-gathering every layer).
    ap.add_argument("--hf-fsdp", choices=("none", "full_shard", "shard_grad_op"), default="none")
    ap.add_argument("--master-weights", choices=("fp32", "bf16"), default="fp32",
                    help="fp32 = fp32 master weights + bf16 autocast (Megatron --bf16); bf16 = pure bf16")
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main():
    a = parse()
    # REALIZED ARGUMENT TABLE, in Megatron's own dump format ("  key ....... value"), so that
    # probe_summarize.identity_vs_realized checks this stack exactly as it checks Megatron. argparse
    # resolves duplicates (last wins) before this prints, so this is what actually runs. Without it
    # an HF leg launched with `--extra-train-args "--hf-fsdp full_shard"` was captured as
    # `hf-fsdp none` and nothing could notice.
    if int(os.environ.get("RANK", "0")) == 0:
        print("------------------------ arguments ------------------------", flush=True)
        for k, v in sorted(vars(a).items()):
            print(f"  {k} {'.' * max(3, 48 - len(k))} {v}", flush=True)
        print("-------------------- end of arguments ---------------------", flush=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    denom = a.micro_batch_size * world
    if a.global_batch_size % denom:
        sys.exit(f"--global-batch-size {a.global_batch_size} is not divisible by "
                 f"micro-batch-size x world ({denom}); Megatron would refuse this too")
    grad_accum = a.global_batch_size // denom
    # Only schedules translated EXACTLY are accepted: an HF run whose LR trajectory differs from the
    # Megatron one while its argv claims the same schedule would have the identity digest vouch for a
    # config that did not run. Translated: constant (no warmup), and Megatron's linear-warmup + cosine
    # decay to min-lr over --lr-decay-iters (which may exceed --train-iters, as in 30-iter throughput
    # legs of a 358-iter recipe -- HF's own cosine can only decay over max_steps, hence lr_factor).
    if a.lr_decay_style not in ("constant", "cosine"):
        sys.exit(f"--lr-decay-style {a.lr_decay_style} is not translated (constant, cosine are)")
    if a.lr_decay_style == "constant" and a.lr_warmup_iters != 0:
        sys.exit("constant LR with warmup is not translated")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    import train_probe

    # OVERLAYS. On Megatron the launcher shim (rocm_compat.py) installs them; HF has no shim, so the
    # launcher does it, BEFORE the model is built. overlay_train rebinds an import path
    # ("module:Class.attr"), which is stack-neutral by construction -- what is NOT neutral is the
    # target string, which names a stack's own module (Megatron's GPTModel._postprocess has no HF
    # counterpart by that name). A failed install EXITS rather than continuing: a leg that silently
    # measures the original implementation while its env names an overlay is the worst outcome the
    # overlay layer exists to prevent.
    if os.environ.get("GEAK_OVERLAY"):
        import overlay_train
        if not overlay_train.install():
            sys.exit("[hf_sft] GEAK_OVERLAY is set but the overlay stack did not install completely; "
                     "refusing to run a leg that would measure the original implementation")

    tok = AutoTokenizer.from_pretrained(a.tokenizer_model)
    # FP32 MASTER WEIGHTS, bf16 compute -- the recipe Megatron's --bf16 actually runs. Loading the
    # model in bf16 (as this launcher first did) makes HF's Trainer train in PURE bf16: weights and
    # AdamW states in bf16, no fp32 master copy. Step-1 loss cannot tell the difference (it is a
    # forward on identical weights), so the parity checks passed; the optimizer is what differs,
    # and it shows up over hundreds of steps as a different training run under the same name.
    # With fp32 weights, `bf16=True` in TrainingArguments enables bf16 autocast for the compute.
    model = AutoModelForCausalLM.from_pretrained(
        a.load, torch_dtype=torch.float32 if a.master_weights == "fp32" else torch.bfloat16,
        attn_implementation=a.attention_backend)

    ds = load_dataset("json", data_files=a.train_data_path, split="train")

    cfg = SFTConfig(
        output_dir=a.output_dir,
        per_device_train_batch_size=a.micro_batch_size,
        per_device_eval_batch_size=a.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=a.train_iters,
        learning_rate=a.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=a.weight_decay,
        max_grad_norm=a.clip_grad,
        adam_beta1=a.adam_beta1,
        adam_beta2=a.adam_beta2,
        bf16=a.bf16,
        seed=a.seed,
        data_seed=a.seed,
        max_length=a.seq_length,
        # Match Megatron `default` prompt format: no masking at all.
        assistant_only_loss=False,
        completion_only_loss=False,
        packing=False,
        logging_steps=1,                 # the probe takes loss/grad_norm from each step's log
        save_strategy="no",
        report_to=[],
        dataloader_drop_last=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        **({"fsdp": f"{a.hf_fsdp} auto_wrap",
            "fsdp_config": {"transformer_layer_cls_to_wrap": [_decoder_layer_cls(a.load)]}}
           if a.hf_fsdp in ("full_shard", "shard_grad_op") else {}),
    )
    # The held-out set goes through the SAME SFTTrainer preprocessing as the training set (chat
    # template, tokenisation, labels); handing a raw dataset to evaluate() later fails on columns.
    ev = load_dataset("json", data_files=a.valid_data_path, split="train") if a.valid_data_path else None
    trainer_cls = SFTTrainer
    if a.hf_train_metrics == "off":
        from transformers import Trainer

        class _SFTTrainerNoStepMetrics(SFTTrainer):
            """SFTTrainer whose compute_loss is exactly Trainer.compute_loss (the loss SFTTrainer itself
            computes first), without the per-micro-batch logging metrics and their cross-rank gathers."""
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                return Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
        trainer_cls = _SFTTrainerNoStepMetrics
    trainer = trainer_cls(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                          **({"eval_dataset": ev} if ev is not None else {}))
    if a.lr_decay_style == "cosine":
        from torch.optim.lr_scheduler import LambdaLR

        def _create_scheduler(num_training_steps, optimizer=None):
            opt = optimizer if optimizer is not None else trainer.optimizer
            trainer.lr_scheduler = LambdaLR(opt, lambda step: lr_factor(
                step, a.lr, a.min_lr, a.lr_warmup_iters, a.lr_decay_iters or a.train_iters))
            trainer._created_lr_scheduler = True
            return trainer.lr_scheduler
        trainer.create_scheduler = _create_scheduler
    train_probe.install_hf(trainer)
    # Shape/oracle capture on THIS stack, at the same step boundary the Megatron shim uses
    # (rocm_compat.py). No-op unless GEAK_CAPTURE_TARGET/DIR/STEP are set; never breaks a run.
    try:
        import capture_train_shapes as _cap
        if _cap.ENABLED:
            _cap.install()
            _orig_step_end = train_probe._Probe.step_end

            def _step_end(self, *aa, **kw):
                _cap.note_step(self.step)
                return _orig_step_end(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end
    except Exception as e:  # noqa: BLE001
        print(f"[geak-capture] not installed: {e!r}", flush=True)
    # HF Qwen2RMSNorm fwd+bwd capture (head hf_rmsnorm_fused_fwdbwd). Separate module: the op's weight
    # is module state, not an argument, so the generic capture cannot give its dweight. No-op unless
    # GEAK_HFRMSN_CAPTURE_DIR/STEP are set; never breaks a run.
    try:
        import capture_hf_rmsnorm as _hfrmsn
        if _hfrmsn.ENABLED and _hfrmsn.install(model):
            _orig_step_end2 = train_probe._Probe.step_end

            def _step_end2(self, *aa, **kw):
                _hfrmsn.note_step(self.step + 1)   # = steps COMPLETED (step_end increments after)
                return _orig_step_end2(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end2
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfrmsn] not installed: {e!r}", flush=True)
    # HF Qwen2MLP gate/up + SwiGLU fwd+bwd capture (head hf_mlp_gateup_swiglu). Separate module for
    # the same reason as the norm: both weights are module state, and h (down_proj's input) is not a
    # return value of any call the generic capture can see. No-op unless GEAK_HFMLP_CAPTURE_DIR/STEP
    # are set; never breaks a run.
    try:
        import capture_hf_mlp_gateup as _hfmlp
        if _hfmlp.ENABLED and _hfmlp.install(model):
            _orig_step_end3 = train_probe._Probe.step_end

            def _step_end3(self, *aa, **kw):
                _hfmlp.note_step(self.step + 1)    # = steps COMPLETED (step_end increments after)
                return _orig_step_end3(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end3
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfmlp] not installed: {e!r}", flush=True)
    # HF Qwen2MLP.down_proj fwd+bwd capture (head hf_mlp_down_proj). Module hooks on the 28 down_proj
    # instances, so it sees the live call whether or not Qwen2MLP.forward is rebound by the stack.
    # No-op unless GEAK_HFDOWN_CAPTURE_DIR/STEP are set; never breaks a run.
    try:
        import capture_hf_mlp_down as _hfdown
        if _hfdown.ENABLED and _hfdown.install(model):
            _orig_step_end4 = train_probe._Probe.step_end

            def _step_end4(self, *aa, **kw):
                _hfdown.note_step(self.step + 1)   # = steps COMPLETED (step_end increments after)
                return _orig_step_end4(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end4
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfdown] not installed: {e!r}", flush=True)
    # HF Qwen2Attention q/k/v/o projections fwd+bwd capture (head hf_attn_qkvo_proj). Module hooks on the
    # 4 x 28 projection instances; the golden is an isolated re-run of all four inside the last of their
    # grad_output hooks. No-op unless GEAK_HFATTN_CAPTURE_DIR/STEP are set; never breaks a run.
    try:
        import capture_hf_attn_proj as _hfattn
        if _hfattn.ENABLED and _hfattn.install(model):
            _orig_step_end5 = train_probe._Probe.step_end

            def _step_end5(self, *aa, **kw):
                _hfattn.note_step(self.step + 1)   # = steps COMPLETED (step_end increments after)
                return _orig_step_end5(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end5
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfattn] not installed: {e!r}", flush=True)
    # HF Qwen2ForCausalLM lm_head + causal-LM loss fwd+bwd capture (head hf_lmhead_fused_ce). Hooks on the
    # model instance (labels / num_items_in_batch kwargs, the loss's grad_output) and its lm_head; the
    # golden is an isolated re-run of lm_head + model.loss_function inside the loss's grad_output hook.
    # No-op unless GEAK_HFLMCE_CAPTURE_DIR/STEP are set; never breaks a run.
    try:
        import capture_hf_lmhead_ce as _hflmce
        if _hflmce.ENABLED and _hflmce.install(model):
            _orig_step_end6 = train_probe._Probe.step_end

            def _step_end6(self, *aa, **kw):
                _hflmce.note_step(self.step + 1)   # = steps COMPLETED (step_end increments after)
                return _orig_step_end6(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end6
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hflmce] not installed: {e!r}", flush=True)
    # HF Qwen2 apply_rotary_pos_emb (q AND k of one layer) fwd+bwd capture (head hf_rope_layout_fused).
    # Rebinds the modeling_qwen2 module global the ORIGINAL Qwen2Attention body looks up (also under the
    # hf5 qkvo wrapper); the golden is an isolated re-run inside the second grad_output hook. No-op
    # unless GEAK_HFROPE_CAPTURE_DIR/STEP are set; never breaks a run.
    try:
        import capture_hf_rope as _hfrope
        if _hfrope.ENABLED and _hfrope.install(model):
            _orig_step_end7 = train_probe._Probe.step_end

            def _step_end7(self, *aa, **kw):
                _hfrope.note_step(self.step + 1)   # = steps COMPLETED (step_end increments after)
                return _orig_step_end7(self, *aa, **kw)
            train_probe._Probe.step_end = _step_end7
    except Exception as e:  # noqa: BLE001
        print(f"[geak-hfrope] not installed: {e!r}", flush=True)
    trainer.train()
    if ev is not None:
        m = trainer.evaluate()
        if int(os.environ.get("RANK", "0")) == 0:
            print(f" validation loss at iteration {a.train_iters} on validation set | "
                  f"lm loss value: {m['eval_loss']:.6E} | [hf_sft eval over {len(ev)} conversations]",
                  flush=True)


def lr_factor(step, lr, min_lr, warmup, decay_iters):
    """Megatron's schedule as a LambdaLR factor. `step` is 0 for the first optimizer step, which
    Megatron calls iteration 1: warmup lr*i/warmup for i <= warmup, then cosine from lr to min_lr over
    (decay_iters - warmup), then min_lr. Verified against Megatron's logged LRs in
    tests/test_adapter_argv.py (9.090909e-07 at 1, 1e-05 at 11, 9.933586e-06 at 30)."""
    import math
    i = step + 1
    if warmup > 0 and i <= warmup:
        return (lr * i / warmup) / lr
    if i > decay_iters:
        return min_lr / lr
    ratio = (i - warmup) / max(1, decay_iters - warmup)
    return (min_lr + 0.5 * (math.cos(math.pi * ratio) + 1.0) * (lr - min_lr)) / lr


if __name__ == "__main__":
    main()
