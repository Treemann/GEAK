# Resume notes — updated 2026-10-10

## State (2026-10-10, after run wf_cc06b402-312 completed)
Run `wf_cc06b402-312` finished: `rope_fused_qk` +3.89% and `swiglu_bwd_fwd_bw` +3.15% ACCEPTED,
validated together at +7.19% (2,874.2 -> 3,081.8 useful tok/s/GPU); bundle
`runs/finalize_bundle_1010_rope_swiglu`. Cumulative vs the unpacked reference at GBS 16: 3.67x.

**Done 2026-10-13:** `measure_stack3.sh` -- rope + swiglu + fc1 stacked = **+9.1%** over the packed
reference (fc1 keeps its full +1.6% on top), 3,132 useful tok/s/GPU, **3.73x** the unpacked
reference. Legs `runs/stack3_1013_031326_*`. (A first attempt on 10-12 was killed at step 11 of its
first leg when the session's permission mode switched; it produced nothing and was re-run.)

**Next:**
1. Commit the doc edits for the stack3 result if not yet committed (`knowledge/sft_measurement.md`,
   `ROADMAP.md`, `ROADMAP.zh-CN.md`, this file).
2. Remaining packed-profile heads: `mlp_fc1_wgrad` (+1.52% expected), `mlp_fc2_gemm` -- a fast-mode
   run on `workloads/qwen2_5_math_7b_packed.env` with `skip_heads: mlp_fc1_gemm,rope_fused_qk,swiglu_bwd_fwd_bw`.
   Note: a fresh run measures heads against the BARE packed reference, so their deltas then need a
   stacked re-measure like this one.

**Where files go (user instruction, 2026-10-13):** development files in this repo; the run workspace
stays at `/home/treemann/geak_train`. One-off debugging scripts are not added to the repo unless
needed.

Stopping a run: `TaskStop` the workflow, then stop ONLY containers whose command is
`/repo/e2e_training_workflow/scripts/bench_train.sh` (by ID), and never `pkill -f` a pattern that
also matches the calling shell. Kernel-track runs need `kernel_workflow/` in the session's scope.

## Done since the last pause (all committed)
* Kernel stack on packed data: +0.08-0.28%, no_win -> kernel wins were padding wins.
* Packed profile: GEMM ~78%, attention bwd 6.3%, exposed comm 3.7% (0% overlap).
* GBS trade-off measured: 64 / 24 / 16 packs -> 3.64x / 3.50x / 3.43x unpacked.
* TE flash attention is correct at op level; Megatron's flash-path 2.7x gradient gap is elsewhere
  (time-boxed, flash quarantined).
* identity_vs_realized false positive from interleaved rank logs fixed.
* `workloads/qwen2_5_math_7b_packed.env` (GBS 16 PENDING owner confirmation, floor 0.25%).

## Decisions waiting on the user / workload owner
1. GBS after packing: 16 (as in the descriptor), 24, or 64 packs.
2. Whether the reference recipe should drop dropout to the model's own 0.0.
3. Review of the uncommitted agent changes (09-24 batch + knowledge/critical_path.md from 10-09).
