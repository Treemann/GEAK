# Resume notes — updated 2026-10-10

## State (2026-10-10, after run wf_cc06b402-312 completed)
Run `wf_cc06b402-312` finished: `rope_fused_qk` +3.89% and `swiglu_bwd_fwd_bw` +3.15% ACCEPTED,
validated together at +7.19% (2,874.2 -> 3,081.8 useful tok/s/GPU); bundle
`runs/finalize_bundle_1010_rope_swiglu`. Cumulative vs the unpacked reference at GBS 16: 3.67x.

**Done 2026-10-13:** `measure_stack3.sh` -- rope + swiglu + fc1 stacked = **+9.1%** over the packed
reference (fc1 keeps its full +1.6% on top), 3,132 useful tok/s/GPU, **3.73x** the unpacked
reference. Legs `runs/stack3_1013_031326_*`. (A first attempt on 10-12 was killed at step 11 of its
first leg when the session's permission mode switched; it produced nothing and was re-run.)

**Done 2026-10-13:** fast-mode run `wf_6d27fb7e-e69` (packed 7B, skip rope/swiglu/fc1, 2 gates, 5.0 h):
* `lmhead_ce_dense_gemm` **ACCEPTED**: gate +2.25% paired (band 0.40%), Director-validated +2.27%
  (2,874 -> 2,937 vs the BARE packed reference), peak reserved -4.65%. L0 equal-to-original; L1 pass as
  kernel_swap and overlay. Bundle `runs/finalize_bundle_1013_lmhead_dense`.
* `mlp_fc2_oproj_gemm` **inconclusive_noise**: +1.78% paired inside a 2.01% band (both legs noisy,
  IQR 1.7% / 2.6%); L0 and L1 pass. Not a no_win -- needs a quiet re-measure.
* Unexplored (task budget): `lmhead_wgrad_reselect` (exp. +1.51%), `attn_fmha_ck_varlen` (+0.98%),
  `qkv_lnlinear_gemm` (+0.39%). Profile: exposed comm 6.7%, 0% grad all-reduce overlap.

**Done 2026-10-13:** `scripts/measure_packed_stack.sh --conds ref,s3,s4,s5 --reps 3` (legs
`runs/pstack_1013_111120_*`, 40 min): lmhead +2.5% on rope+swiglu+fc1, fc2_oproj +1.74% on top
(IQR 0.13-0.18% -> its inconclusive_noise resolves as a win), all five **+13.8%**, 3,264 useful
tok/s/GPU, **3.89x** unpacked. L1 passes on every stacked pair. Details: `knowledge/sft_measurement.md`.

**Next:**
1. Unexplored heads: fast mode on the packed descriptor with `skip_heads:
   mlp_fc1_gemm,rope_fused_qk,swiglu_bwd_fwd_bw,lmhead_ce_dense_gemm,mlp_fc2_oproj_gemm`
   (candidates `lmhead_wgrad_reselect`, `attn_fmha_ck_varlen`, `qkv_lnlinear_gemm`). Note
   `lmhead_wgrad_reselect` may overlap the accepted lmhead overlay (same `_postprocess` site): it
   must be measured on top of it, not against the bare reference.
2. No combined five-kernel bundle exists yet; `measure_packed_stack.sh` is the reproducible recipe.

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
