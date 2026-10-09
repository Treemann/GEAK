# Paused 2026-10-09 (second pause) — GPUs released for another task

GPUs free (`rocm-smi --showpids`: no KFD PIDs). No container of this project is running; the other
32 containers on the box were not touched. Nothing is queued and nothing restarts on its own.

## What was running

`measure_stack.sh` — does the accepted +12.36% kernel stack (lmhead_ce_unmasked_gather +
mlp_fc2_down) still pay ON TOP OF PACKING? Two arms × 2 reps on the packed dataset. Stopped after
**1 of 4 legs**: `runs/stackPK1009_ref1` (packed, no overlays) completed cleanly; the other three
never started. The partial result proves nothing on its own — re-run all four.

Why it matters: several accepted kernels earn their speed by skipping padding, and packing removes
97% of it. Kernel work after packing should be prioritised on a PACKED profile, and this run says
whether the old wins survive.

## To resume

```
R=/home/treemann/geak_train/runs
bash scripts/measure_stack.sh --bundles $R/finalize_bundle_lmhead_0922,$R/finalize_bundle_fc2_0924 \
  --out-prefix $R/stackPK1010 --iters 30 --reps 2 \
  --data /work/data/sft_ultrachat20k_packed_7b_cfg1009.jsonl
```
(~45 min, 8 GPUs). Use a NEW --out-prefix: one probe dir is one leg, and the old prefix already has
a ref1.

Then, by value:
1. Re-profile the PACKED stack and re-rank kernel heads on it (the current queue — mlp_fc1_rowslice
   +14.7%, attn_fmha_true_seqlen +8.9% — was ranked on padded data).
2. Choose GBS in packs (16 or 24) to keep samples per optimizer step; a recipe decision for the
   workload owner.
3. Flash attention backend: root-cause the 2.7x gradient (the varlen kernel itself is exact).
4. Second stack exit criterion: one kernel accepted on HF TRL through the same gates.

## State at pause (all committed; last commit 5af40231)

* Packing ACCEPTED by the workflow on its own (run wf_01fab37a-6da): +258.11%, Validate +256.75%
  (3,042.3 vs 839.2 useful tok/s/GPU at 7B). Bundle: `runs/finalize_bundle_1009_pack`.
* Tests all pass, preflight GO.
* Still uncommitted and NOT mine: the 09-24 agent changes to bench_train.sh, build_task_dir.py,
  capture_train_shapes.py, l0_gate.py, rocm_compat.py, and the untracked capture_mlp_fc1_swiglu.py.
