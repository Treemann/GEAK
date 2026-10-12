# Resume notes — updated 2026-10-10

## State (2026-10-10, after run wf_cc06b402-312 completed)
Run `wf_cc06b402-312` finished: `rope_fused_qk` +3.89% and `swiglu_bwd_fwd_bw` +3.15% ACCEPTED,
validated together at +7.19% (2,874.2 -> 3,081.8 useful tok/s/GPU); bundle
`runs/finalize_bundle_1010_rope_swiglu`. Cumulative vs the unpacked reference at GBS 16: 3.67x.

**Pending when Bash came back (the auto-mode safety classifier was unavailable for a stretch):**
1. Commit the doc edits already on disk: `knowledge/sft_measurement.md`, `ROADMAP.md`,
   `ROADMAP.zh-CN.md` (the two-kernel result), plus this file and `scripts/measure_stack3.sh`.
2. Run `bash scripts/measure_stack3.sh` (~6 legs x ~4 min): ref / rope+swiglu / rope+swiglu+fc1,
   interleaved, 2 reps -- does fc1 (+1.60% alone) add on top of the accepted stack? The script patches
   ONLY fc1's impl.py proof path (rope and fc1 both wrote their proof next to the FIRST manifest and
   would overwrite each other when stacked); kernel_src bytes are copied verbatim.
3. Then the remaining packed-profile heads: `mlp_fc1_wgrad` (+1.52% expected), `mlp_fc2_gemm`.

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
