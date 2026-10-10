# Resume notes — updated 2026-10-10

## Running now
Fast-mode kernel track on the PACKED 7B descriptor, remaining heads -- run `wf_cc06b402-312`,
`skip_heads: mlp_fc1_gemm` (already measured: +1.601% paired, L0/L1 PASS after re-gate), budget 2,
max_gates 4, kernel_budget 3, same baseline_dirs as before, t0_epoch_ms 1791615479170.
Requires `kernel_workflow/` in the session's directory scope (see RUN.md).

If it must be stopped: `TaskStop` the workflow, then stop ONLY containers whose command is
`/repo/e2e_training_workflow/scripts/bench_train.sh` (by ID), and never `pkill -f` a pattern that
also matches the calling shell. Resume with `resumeFromRunId: "wf_cc06b402-312"` and the same args.

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
