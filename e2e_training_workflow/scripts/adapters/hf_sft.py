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
    ap.add_argument("--master-weights", choices=("fp32", "bf16"), default="fp32",
                    help="fp32 = fp32 master weights + bf16 autocast (Megatron --bf16); bf16 = pure bf16")
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main():
    a = parse()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    denom = a.micro_batch_size * world
    if a.global_batch_size % denom:
        sys.exit(f"--global-batch-size {a.global_batch_size} is not divisible by "
                 f"micro-batch-size x world ({denom}); Megatron would refuse this too")
    grad_accum = a.global_batch_size // denom
    if a.lr_decay_style != "constant" or a.lr_warmup_iters != 0:
        # Only the schedule the Megatron adapter actually uses is translated. Accepting others
        # silently would produce an HF run whose LR trajectory differs from the reference while its
        # argv claims otherwise -- the identity digest would vouch for a config that did not run.
        sys.exit("only --lr-decay-style constant with --lr-warmup-iters 0 is translated")

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
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    train_probe.install_hf(trainer)
    trainer.train()


if __name__ == "__main__":
    main()
