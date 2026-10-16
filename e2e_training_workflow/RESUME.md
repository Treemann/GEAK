# Resume notes — updated 2026-10-14

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

**Done 2026-10-13:** fast-mode run `wf_def62c9e-e34` (skip the five + `lmhead_wgrad_reselect`, queue
exhausted, 3.9 h): `attn_fmha_ck_varlen` ACCEPTED (+4.27% validated, bundle
`runs/finalize_bundle_1013_attn_ck_varlen`); `qkv_lnlinear_gemm` inconclusive_noise (+0.88% in 1.22%).
Stacked re-measure `measure_packed_stack.sh --conds ref,s5,s6,s7 --reps 3` (legs
`runs/pstack_1013_154204_*`): attn +5.25% on the five, qkv +1.05% on top -> **seven kernels +20.7%,
3,460 useful tok/s/GPU, 4.12x unpacked**. One L1 false reject (s7 vs s6 rep 3, strict sign rule, one
outlier leg) -- analysed in `knowledge/gate_convergence.md` "Stacked correct kernels".

**Done 2026-10-15:** `wf_1529a068-555`, the first real run of `baseline_overlay`.
* What it validated:
  * Setup measured three STACKED legs. All 7 overlays were engaged with fallback 0, and
    `impl_digest` 18aa6b781217a343 was identical on all three (the fixed manifest names work).
  * The baseline was 3,443–3,481 useful tok/s/GPU, matching the 10-13 s7 legs.
  * The profile was taken on the stack: ~95% of the critical path is already in accepted overlays.
  * Finalize produced a seven-kernel bundle, `runs/finalize_bundle_1015_s7_stack` (launch.sh with
    `--ref` and `--bare`). Two cold runs passed every check and reproduced the baseline within
    0.044% paired. This largely covers P0-3; it still depends on the gated sources under
    `$GEAK_WORK_DIR`.
* What it found: the stop rule refused every head because Setup priced the floor at 0.93%. The
  across-job spread had been measured with the aligned median ratio, which host stalls inflate; the
  paired ratio read 0.07%. Fixed in DECISIONS 4.18 (ae01511f).

**Done 2026-10-15:** `wf_eac2c468-e66` (stacked, reusing the three s7 legs, paired floor 0.25%).
* `rmsnorm_bwd_fc1_fused` was ACCEPTED on the stack, but only at floor level: gate +0.257%,
  validation +0.29%, re-measure +0.36/+0.38%. Bundle `runs/finalize_bundle_1015_s7_rmsn`, now
  `stack_overlay_env.sh --cond s8`.
* The carried re-profile is verified.
* `grad_tail_fused_scale_check` was dropped by the forward-only rule although it is outside autograd.
  Fixed in DECISIONS 4.19.
* Details: `knowledge/sft_measurement.md`, "Gating ON the stack".

**Done 2026-10-15:** `wf_99be8541-d9f` (base s8).
* `grad_tail_fused_scale_check` was admitted via `autograd: false`. Op-level 1.64x, but **no_win** in
  the step: +0.13% paired against a 0.39% band.
* The kernel queue is exhausted on s8. Bundle `runs/finalize_bundle_1015_s8` (8 overlays).
* Cumulative: s8 is about +21% over the packed reference, ~4.15x unpacked.

**Done 2026-10-15 (no GPU):** P1-2. `scripts/controls/sign_rule_impl_null.py` measured the strict
sign rule on 96 legs:
* one-overlay correct changes: 3.4% (loss) / 1.0% (grad_norm);
* all correct changes: 0.8% / 0.6%;
* same-code re-runs: 6.1% / 1.1%.
The rule stands. The 10-13 "~17%" was a small correlated sample (DECISIONS 4.20).

**Done 2026-10-15:** user decision plus measurement settled the recipe: **GBS 16 packs, dropout
default 0.1 untuned** (DECISIONS 4.21).
* On s8, GBS 16 / 24 / 64 gives 3,488 / 3,582 / 3,718; the stack's gain is +21.5% at all three
  (`runs/gbsx_1015_133312_*`).
* Headline: 3,488 useful tok/s/GPU, 4.16x unpacked.
* The descriptor file is unchanged, because bundles pin its sha256.

**Done 2026-10-15:** P0-4 (`runs/qab_1015_151502_*`).
* INCONCLUSIVE by its pre-registered rule: the control was not detected.
* The diagnostics showed the end-of-epoch held-out loss is dominated by recipe-level late
  instability. A same-seed REF rerun moved 1.298 -> 1.401.
* Progressive validation was chosen as the instrument. See `knowledge/gate_convergence.md`.
* The stack is NOT yet cleared for long training.

**Done 2026-10-16:** P0-4b, the bisection, isolation and the side-stream test.
* **The accepted 8-kernel stack FAILS on late-training instability.** s8-family legs are
  severe-or-worse on 9/16, against 0/9 null legs. Catastrophic episodes (loss 5–9) occur only with
  ≥ 4 overlays.
* rope and swiglu are cleared at bisection resolution. fc1 and lmhead are each mildly implicated
  alone.
* Ruled out as the main cause: wgrad precision (probed in situ, fp32-exact) and side streams (s8
  serial still has a catastrophic episode).
* The mechanism is open. The leading lead is per-parameter gradient differences on identical
  weights (the step-1 grad norm moves 20–85% between stacks).
* All stacked bundles are NOT cleared for production training (DECISIONS 4.22).
* Full log: `knowledge/gate_convergence.md`.

**User clarification (2026-10-16):** "no tuning" means general default training practice (warmup
allowed), kept fixed during optimisation, not constant LR.
* New descriptor `workloads/qwen2_5_math_7b_packed_cosine.env`: warmup 11, cosine to 1e-6 over 358.
* The older descriptors are argv-identical, so no pinned bundle is affected.
* P0-4c (s2 under constant LR) was stopped during its first leg and never analysed. When stopping
  it, `TaskStop` left the container running; it was stopped by ID after `docker inspect`.

**Done 2026-10-16:** P0-4d (`runs/qabcos_1016_092855_*`).
* NOT CLEARED by its rules: both INCONCLUSIVE.
* Under the standard schedule the catastrophic episodes are gone (S8 0/5). The decayed held-out
  loss puts S8 at +0.0005 against the control's +0.0206.

**Done 2026-10-16: P0-4e CLEARED** (`runs/qabfin_1016_140340_*`).
* The s8 stack is cleared for training under the standard schedule: held-out d −0.0023, t −1.54;
  0 severe / 0 catastrophic.
* It is NOT cleared under constant LR (DECISIONS 4.24).
* Plan steps 3 (P0-2) and 4 (QualityGate) were done while it ran.

**Done 2026-10-16 (later):**
* Headline on the cleared recipe: 3,491, 4.16x (`runs/pcos_1016_170258_*`).
* GEMM microbench: sustained peak 199 TFLOP/s; training GEMMs at 85–99% of it in isolation; step
  MFU about 76%. P2-1 de-prioritised.

**Running:** P2-3 DP-communication sweep on s8 + the scheduled recipe, one-off driver
`geak_train/scratch/comm_sweep_1016.sh`. Arms:
* A: baseline;
* B: `--overlap-grad-reduce`;
* C: B + `--overlap-param-gather`;
* D: B + `--ddp-bucket-size 2e8`;
* E: `--ddp-average-in-collective`.
2 reps each; legs `runs/comm_<stamp>_*`. A winner still needs L1 and the outcome gate before
acceptance.

**AUTONOMOUS PLAN (user offline 2026-10-16 → 10-18; instruction: never stop, decide professionally, record
everything).** In order:
1. **P0-4e verdict** (`quality_ab_final.py`).
   * CLEARED: write DECISIONS 4.23 (the stack is cleared for training under the standard schedule;
     under the constant-LR recipe it is NOT). Update bundle status in the docs, ROADMAP and README.
   * NOT CLEARED: debug. Compare per-parameter gradients REF vs S8 on identical weights and batch
     (which parameters carry the step-1 grad-norm gap), localise, fix, re-test. Do not move on
     until resolved.
2. **Headline on the recommended recipe.** Interleaved ref vs s8 on the cosine descriptor, 3 reps
   (30 iters), so the published numbers carry that recipe's config_digest. Throughput is expected
   to be unchanged.
3. **P0-2, review the uncommitted agent changes.** No GPU. Read each diff and run the tests.
   * Commit what is correct and useful, with a message that says so.
   * Leave anything doubtful in place and document why.
   * Never discard work silently.
4. **Fold the outcome test into the workflow** as an optional final gate (`quality_gate` arg:
   REF vs the final stack, 5 seeds, `quality_ab_final.py` rules), so a stack is never shipped on
   30-step gates alone (DECISIONS 4.22). Dry-run test it.
5. Then the P2 items in ROADMAP order (GEMM efficiency analysis first, then comm bucket levers,
   then HF TRL).

**Next:** the prioritised list is `ROADMAP.md` "Near-term priorities (2026-10-14)". In short:
* P0-1: GBS + dropout decision (owner).
* P0-2: review the uncommitted agent changes.
* P0-3: seven-kernel bundle.
* P0-4: long-horizon quality by outcome (held-out eval loss, 3 seeds, positive control).
* P1: L1 null for implementation-change pairs; GBS on the stack. The stacked baseline in the workflow
  is BUILT (2026-10-14, DECISIONS 4.17: `baseline_overlay`, `scripts/stack_overlay_env.sh`, carried
  re-profile, queue ranked in code). It is dry-run tested and still needs one real run.

Done 2026-10-13 and recorded: re-profile ON the seven-kernel stack (`runs/pprof_1013_162241_s71`),
side-stream profiler fix (DECISIONS 4.16). Kernel queue exhausted.

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
1. ~~GBS after packing~~ DECIDED 2026-10-15: GBS 16 packs (DECISIONS 4.21).
2. ~~Whether the reference recipe should drop dropout to the model's own 0.0.~~ DECIDED 2026-10-15: keep the default 0.1, no tuning.
3. ~~Review of the uncommitted agent changes~~ DONE 2026-10-16 (060347f9): all committed after review.
