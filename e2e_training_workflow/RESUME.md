# Resume notes — updated 2026-10-23

## OPEN AFTER THE 2026-10-23 REVIEW (owner: do all, in this order; status kept current)

Order: (8) stale records -> (3) >= 3 baseline legs so the L1 envelope votes -> (1) main_grad contract in L0 ->
(2) 30-step headline with the unstable first steps excluded (owner: NOT a full-epoch headline) -> P0-4 HF
attention mid-depth oracles -> P1 (5) harness/rocm_compat identity, (6) template-oracle output hole, (7) eps
cross-stack note -> P2 HF token-balanced ranks, 1022 attention into s8, NaN-check/zero_ passes, --gc default ->
P3 reuse HF QualityGate ref/ctrl70 legs, mid-leg occupancy sampling, HF L1 backward check.
* (8) DONE 2026-10-23: comm-sweep arm C record corrected (crash, AssertionError in start_param_sync, not
  "degenerate"); README open list refreshed (P1-C and in-workflow QualityGate were already done).
* (3) RESOLVED DIFFERENTLY 2026-10-23: the premise was wrong -- Setup already takes REPEAT 3, and the per-step
  envelope was RETIRED from voting on evidence (DECISIONS 3.4b), so no leg count makes it vote. Fixed the
  stale text that said ">=5 legs restore a blocking envelope" and the band label "BLOCKING" that
  contradicted verdict().
* (1) DONE 2026-10-23 (DECISIONS 4.56): scripts/main_grad_contract.{py,sh} gate every kernel candidate's fp32
  main_grad accumulation (fail closed); reproduces G5 on the four s8 GEMM overlays.
* (2) DONE 2026-10-23 (DECISIONS 4.57): 30-step legs kept; the headline window starts at max(detected, step index 8)
  (lowest pairwise p90 of K = 0..15 on 1,080 same-identity pairs; worst pair 3.29% -> 1.89%).
* P1(5) DONE 2026-10-23 (DECISIONS 4.59): comparable() refuses a pair whose rocm_compat / overlay_train / image differ.
* P0-4 DONE 2026-10-23 (DECISIONS 4.58): layer-14 holdouts for the six per-layer s8 kernels: all pass, all controls fail.
* P1(6) DONE 2026-10-23 (DECISIONS 4.60): L0 must name its oracle; pre-v2 template oracles refused; 8 HF oracles
  rebuilt into tasks_v2/ (base_error reproduced exactly); the six shipped HF kernels pass them.
* P1(7) DONE 2026-10-23: README states the cross-stack recipe difference (eps, packing) next to the HF headline.
* P2 HF balance DONE 2026-10-23 (DECISIONS 4.61): +15.2..+15.9% paired on hf8, P0-4m CLEARED; recommended HF config
  = hf8 + --hf-balance-tokens on. P3 QualityGate reuse DONE (4.63), mid-leg occupancy DONE, HF unrounded loss DONE.

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

**Done:** P2-3 comm sweep (`runs/comm_1016_172452_*`).
* `--ddp-average-in-collective` gives +0.86 / +0.92%, L1 PASS.
* Overlap variants are worse or unusable.

**Done 2026-10-16 (evening):**
* **P0-4f CLEARED.** `--ddp-average-in-collective` is accepted (+0.88%). Recommended descriptor
  `qwen2_5_math_7b_packed_cosine_avg.env` (DECISIONS 4.25).
* **Final headline: 3,504 useful tok/s/GPU, 4.18x** (`runs/hl_1016_205202_*`).
* A regression from the adapter change (`DDP_EXTRA_ARGS` unbound under `set -u`) was found by the
  headline run, fixed in 07ce1c5d, and is guarded by `test_adapter_argv.py`.

**Done:** HF infrastructure, DECISIONS 4.26 (commit 65044881).
* Descriptor-driven schedule and FSDP; exact Megatron schedule translation; capture hook.
* Descriptor `qwen2_5_math_7b_hf_cosine.env`.
* 7B HF smoke: 1,152 useful tok/s/GPU.

**Done 2026-10-17:** **P2-2 exit criterion MET** (`wf_4385c413-a19`, DECISIONS 4.27).
* `attn_fmha_ck_hf` +6.64% and `hf_rmsnorm_fused_fwdbwd` +2.15%, validated together +9.33%.
* HF ZeRO-2 (`shard_grad_op`): +21%, bitwise-identical trajectory (4.28). Recommended HF descriptor
  `qwen2_5_math_7b_hf_cosine_zero2.env`.

**Done 2026-10-17:**
* HF headline: 1,096 → 1,474 useful tok/s/GPU (1.35x). [superseded by 4.33: 1,083 → 1,433, 1.32x]
* **P0-4g CLEARED: the HF stack is safe to train with** (4.29).

**Done 2026-10-17 (evening):**
* `hf_mlp_gateup_swiglu` accepted, +22.3% on hf2 (4.30); condition `hf3`.
* TRL step metrics off: +32% bare, +22% on the stack, bitwise identical (4.31).
* **HF 2.07x** (1,096 → 2,269). Recommended descriptor `qwen2_5_math_7b_hf_cosine_zero2_nm.env` + hf3.

**Done 2026-10-18: P0-4h CLEARED.** The final HF configuration (2.07x) is safe to train with
(4.32).

**Done 2026-10-18:**
* `wf_499a73c3-cd2`: `hf_mlp_down_proj` was returned `unusable_ref` because of the warmup defect.
  Fixed (DECISIONS 4.33), and the HF figures were CORRECTED to 1,083 → 2,178, 2.01x. The head was
  then re-adjudicated as a win, +7.18% (4.34); condition `hf4`.

**Done 2026-10-18: P0-4i CLEARED.** hf4 is the recommended HF stack: **2,290, 2.11x** (4.35).

**Done 2026-10-19:** `wf_65926511-100` accepted `hf_attn_qkvo_proj` on hf4, +2.63% (2,291 to 2,352): condition **hf5**, DECISIONS 4.36, pending the outcome test. Commit 09577e34 put the HF capture/oracle tooling in the tree.

**Done 2026-10-19:** `wf_39ac0dc6-211`: `hf_lmhead_fused_ce` accepted on hf5, +2.60% (2,344 to 2,402) = **hf6** (4.37). `hf_rope_layout_fused` rejected by the L1 direction rule; remedy = bf16-rounding-exact variant.

**Done 2026-10-19: P0-4j CLEARED.** hf6 is the recommended HF stack: **2,406, 2.22x** (4.38).
* hf6 vs hf4: +5.23% paired.
* The first outcome attempt lost eval_loss to the overlay signature bug (fixed 410207cb, re-run `qabhf6b`).

**Done 2026-10-20:** `wf_02793feb-78a` accepted `hf_fsdp_flatgrad_accum` (+5.81%) and `hf_rope_bf16_exact` (+2.94%), both bitwise to the original. Validated +8.99%, 2,400 to 2,618 = **hf8** (4.39).

**Done 2026-10-20: P0-4k CLEARED.** hf8 is the recommended HF stack: **2,618, 2.42x** (4.40).

**Pre-manual-test fixes DONE 2026-10-20** (commits cd75d661, e5468c9f), user-approved list:
* Disk: deleted 9 finished HF `exp/team_*` rounds plus `tasks_scratch`; free 29G to 171G.
* Overlay install failure: the leg is now unusable (rocm_compat exits, bench_train scans train.log).
* `--stack` guard on stack conditions.
* Capture detection by pattern.
* quality_ab_final: t from n, >= 3 seeds.
* integration-test threshold.
* RUN.md manual-test section; QualityGate default `_cosine_avg`.
Still open below: P1-A (bundle on `_cosine_avg`), lr-decay-iters identity, Megatron `sources`, ConfigSweep merge, docs, P2.

**Done 2026-10-20/21: manual test M1 PASSED** (`wf_becee9a0-c6f`, base commit e771d3c7, fast mode, bare
`_cosine_avg` reference, budget 1). Setup 3 baseline legs 2,894 useful tok/s/GPU, across-job 0.022%;
profile `kernel_work_can_pay`; `attn_fmha_ck_varlen` 1020 ACCEPTED: gate +3.80% (band 1.61%), bundle
re-check +3.73/+3.89%, Director +4.08% (band 0.55%); L0 pass (paired dq scale_bias -1.5e-5), L1 pass
(`--kind kernel_swap`, step-1 loss +0.0117%; step-1 grad_norm -5..-6% reported only). Bundle
`runs/finalize_bundle_1020_attn_ck_varlen`, report `runs/architect_report_1020_attn_ck_varlen_cosine.md`.
NOT exercised: QualityGate, ConfigSweep, baseline_overlay stacking, more than one head. The 1020 kernel is
gated against the BARE reference only; s8 still carries the 1013 attention overlay.
Commits since: be6c7f6c (agents group under meta.phases titles; per-lever ConfigSweep label kept),
9fbf1b33 (QualityGate also runs when only a non-packing config lever was accepted), 372fdf43 (L1 `--kind`
documented for auditor/tuner; DECISIONS 3.2b paired-dq red line), and snapshot v5 (DECISIONS 1.10b).

**Done 2026-10-21 (autonomous, user-approved order):**
* Snapshot v5 + implementation check in the gate (092d111d, DECISIONS 1.10b).
* M2 NOT run for kernels: cost-benefit (wf_6d834cc5-cc7, skeptic-checked) priced every head M2 could
  open on s8 at 0.05-0.5% e2e, at or below the 0.25% floor / 0.39% s8 band; the s8 queue was already
  exhausted on 10-15. The 1013 -> 1020 attention swap is worth ~+0.12% on s8: fold it in only at the
  next outcome test that is forced for another reason. M2 remains optional as a pipeline regression.
* Host stalls (CPU-only survey of 48 legs) -> synchronized-GC A/B -> DECISIONS 4.41: aligned wall-time
  +0.75 / +1.12% on the A/B pairs but +0.03% on the bundle's cold pair, and some stalls survive with
  automatic GC off, so GC is a partial, unconfirmed fix (opt-in). L1 step-1 0.0%. Kept: report-only
  `wall_useful_tok_s_per_gpu` / `stall_pct` in bench_summary and aligned `wall_delta_pct` in compare_legs.
* **P1-A DONE:** `runs/finalize_bundle_1021_s8_cosine_avg` (default = cleared s8 + `_cosine_avg`;
  `--gc`; `--bare`). 4 cold runs from /tmp: stack x2 and gc pass at launch; bare passes after pinning its
  own v5 impl_digest (unknown until it ran). bare -> s8 +20.88 / +21.79% paired (2,881 -> 3,479 / 3,526);
  the two cold stack medians differ by 1.4% aligned (paired -0.01%). README points to it.
* Still open: locate the non-GC stalls; full-epoch host-memory check before `--gc` could become default; P1-B(3) sources; P1-C
  ConfigSweep merge / config_flags carry; QualityGate never run in a workflow; P2; P3 (MoE).

**RUN 2026-10-22 `wf_b4b35fd3-d93` + seed extension: QualityGate EXERCISED in a workflow for the first time
(DECISIONS 4.45).** 1022 attention accepted (+4.42% gate, +3.97% Director, wall +1.5%); outcome test INCONCLUSIVE at
n=5 under the new noninf rule (one outlier seed), CLEARED at the pre-registered n=10 (upper bound +0.00242 vs 0.00515).
Open from it: the workflow's QualityGate cannot itself extend seeds -- the extension was run by hand.

**RUN 2026-10-22 `wf_292a2e25-eb6`** (deep, bare `_cosine`, config_levers ddp_average_in_collective, budget 0,
quality_gate on; first real run on the fail-closed code): every phase ran with agents_error 0; Setup legs passed the
strict legUsable (v5 digests 3dc12e7c / 72932ab7); the lever measured +0.54% paired (+0.73% aligned, L1 pass as
`overlay`, step-1 loss 0.0) but the band was 1.68% -> `inconclusive_noise`, NOT accepted; so QualityGate correctly
skipped ("nothing accepted") and Finalize returned null proofs; status ok. QualityGate is therefore STILL unexercised
inside a workflow (it needs an accepted change). Finding: the band came from a paired-ratio IQR of 2.17% over 23 steps;
today's BARE null pairs show ratio IQR 0.54-1.49% vs 0.18% on s8 pairs and 0.44% on the 10-20 bare pair -- bare
legs are currently a noisy reference (cause unknown; not the Triton cache, which bare legs only read). Synchronized
GC on BOTH legs of every gate would remove the GC tail; consider it as a measurement setting.

**AUDIT 2026-10-22** (`wf_3be44e2c-a6b`, 6 lenses x adversarial verify, DECISIONS 4.42). FIXED: the critical
fail-open (kernel accepted with no L0/L1) and every high item (engagement/finalize/legUsable fail closed, ref must be
the carried stack, ConfigSweep re-gate, overlay carry composed, status, L1 NaN, L0 template output count for new
oracles, --hf-train-metrics + overlay env knobs in the implementation digest, wall rate from t_wall, bundle --bare
PermissionError). OPEN, by priority:
* **RESOLVED 2026-10-22 (DECISIONS 4.43): the ~900 ms 'external' class was rope_fused_qk's Triton recompile per
  segment-count bucket, paid every run because each container had an empty cache** (not telemetry: the user's
  rocm-smi watch was paused for one leg and resumed; same stall steps). bench_train now defaults to a persistent
  TRITON_CACHE_DIR. Remaining overlay-induced stall: extra Python GC churn on overlay legs (synchronized GC,
  `--gc`, removes it). Still open: the 1020 attention's higher stall rate in its 30-step gate legs (cand
  stall_pct 2-7%) -- re-measure its wall delta with the warm cache before folding it into s8.
* DONE 2026-10-22 (4.44): outcome tests default to a non-inferiority rule; every historical clearance also passes it.
  DONE: quality_ab defaults to the cleared recipe and the descriptor's LR_DECAY_ITERS; the workflow runs the one
  pre-registered seed extension itself; `quality_ctrl: ctrl70` adds an in-run positive control (opt-in, +50% legs).
* Agent trust: the orchestrator sees only the Integrator's chosen pair (ORCH-1); Validate's estimator and band are
  agent-chosen (ORCH-2: DONE, the card now fixes the estimator and band formula); probe paths (ORCH-3: DONE, schema);
  Validate as a bare string (ORCH-7: DONE, schema requires numbers + L1 + legs, status needs l1.pass); leg-dir
  collisions (ORCH-8: DONE, bench_train refuses an --out that holds a leg). ORCH-1 DONE: a `gate_verifier` role
  runs scripts/verify_gate.py on EVERY measured ref x cand pairing (the Integrator must list legs_measured); the
  reported pair must win on the recomputed numbers with engagement proven from the proof files, and any regressing
  pairing vetoes. Its first use found the accepted 1020 gate's first pair inconclusive (cand leg stalled 7%; band
  12%) -- the win came from the re-measured pair; every pairing agreed on +3.6..+4.2%, so the accept stands.
* Oracles: G6 DONE 2026-10-22 (DECISIONS 4.47): 6/8 s8 kernels pass every holdout; qkv and rmsnorm each fail one
  holdout on a one-ulp rounding flip (frozen rule = FAIL; not a defect; proposal: ulp floor for ~0-error labels).
  G5 DONE (4.50: main_grad accumulation correct on fc1/fc2op/qkv/lmhead; contract checks recommended for future
  GEMM oracles). G8 (4.49): the HF attention overlay FAILS the paired-dq rule (+1.0e-2) while its stacks are
  outcome-cleared -- owner decided (4.51): cross-family rule; validated (4.53): it passes at L14/L27 and fails
  only at the near-one-hot layer 0 -> stays cleared, known deviation narrowed to that regime. 4.52 (1-ulp rule)
  validated: qkv/rmsnorm holdouts pass under the new oracle rules, every control still fails. Still open: HF attention oracle has the same dq blind band with no paired
  check (G8); training-path fp32 main_grad wgrad of the GEMM overlays is never exercised by a voting gate (G5);
  G7 DONE (4.48: kernel_swap floor 0.25%, 2.0% only with --l0-strictly-better); the 28 frozen template oracles keep the
  missing-output hole (re-generate when a task is reused).
* DONE 2026-10-23 (review step 3, DECISIONS 4.55): identity snapshot v6 (norm-epsilon = workload identity,
  rope-fusion = implementation, manifest labels out of impl_digest); verify_gate `carried` check (a lever
  may not switch off an accepted overlay); eps 1e-6 recipe + s8eps built, gated (L0 on a new eps-1e-6
  oracle, L1, +0.34/+0.31%) and outcome-tested (P0-4l, runs/qabeps6_1023_073707_*): s8eps CLEARED on
  eps 1e-6, but eps 1e-6 LOSES to 1e-5 (held-out +0.0067, 5/5 seeds, 1 catastrophic) -> production keeps
  _cosine_avg / eps 1e-5. Fused RoPE: +3.7% on BARE (L1 kernel_swap PASS), -0.33% on s8 (disables the rope
  overlay) -> not adopted. finalize_bundle_1021_s8_cosine_avg re-pinned to v6 (runs/repin1021_v6_*, all
  checks pass; harness git 3337c13e). finalize_bundle_1023_s8eps_eps6 kept, marked NOT the deliverable.
* DONE 2026-10-23 (review step 2, DECISIONS 4.54): stall-trimmed paired dispersion (delta untrimmed), per-stack floor
  (Megatron 0.30 / HF 0.25), inconclusive at an absolute 1.0% band, banked_memory by step shape (2% / 6%),
  `batch_ms` -> `batch_site_ms`, calibrate_band per stratum with injected-effect power over 2,072 true nulls
  (realpath-deduped, realized-arg-equal): Megatron bare 0/1290 false, 1.1% inconclusive (was 16%), power +0.5% 83%.
* DONE 2026-10-22: calibrate_band.py now judges null pairs with the gate itself (scripts/gate_eval.js): 400 pairs ->
  0 false wins, 1 floor-level false regression, and 24 false `banked_memory` -> threshold raised 2% -> 5% (null
  peak-memory spread: p99 4.0%). Open measurement hygiene: the unpacked-workload
  floor is breached by a true null pair; 30-step window sits ~1.2% above the full-epoch median (TP-5) and within-leg
  drift over an epoch is -3.4% (M3); packing's GBS-in-packs changes samples per update (TP-7); HF 4.33 correction
  left two HF lever deltas on 12/4-step windows (F2); HF "step-1 loss 0.0" claims are at 4-decimal resolution (F6).
* Reproducibility: rocm_compat changes the loss on packed data and is not in leg identity (OVL-6); kernels depend on
  the unpinned image (OVL-5) and live repo code (OVL-4): MITIGATED -- legs record image id + harness shas; bundles
  pin both (SHA256SUMS.harness, digest-pinned image; deliverable re-checked cold). DONE: OVL-6 (harness shas recorded per leg,
  report-only), OVL-7 (--gpus -> HIP_VISIBLE_DEVICES, named/labelled container with --init, knobs forwarded).
* Low: DONE stop_reason labels, classOfLever by flag name. Open: impl_digest includes manifest names (F8: a digest
  break, do it with the next schema bump), stale records (F10).

**MEGATRON REVIEW 2026-10-20** (two read-only audits plus spot checks). None of the HF-era fixes (4.33
warmup, 410207cb signature, 4.37, 4.39) changes a Megatron number. The fixed-shape detector path was
re-verified on the headline legs, and 3,504 / 4.18x stands. Open TODO, in priority order:

* **P1-A, the deliverable. DONE 2026-10-21 (`finalize_bundle_1021_s8_cosine_avg`).** Was: `finalize_bundle_1015_s8/launch.sh:51` runs `packed.env`: constant LR, NOT cleared
  (4.22), and no `--ddp-average-in-collective`. No bundle runs the cleared recipe `_cosine_avg`.
  * Re-finalize an s8 bundle on `_cosine_avg`, with two cold runs.
  * The pinned stack_overlay_env sha is stale: 67a6959e vs 87193aa3. Only a warning, but re-verify.
* **P1-B, measurement integrity (code).**
  * (1) DONE e5468c9f: a failed overlay install makes the leg unusable; s-/hf-conditions refuse the
    wrong `--stack`.
  * (2) DONE 2026-10-21, snapshot v5 (DECISIONS 1.10b): `--lr-decay-iters` captured, and the review's NEW
    finding fixed with it: `--ddp-average-in-collective` was captured nowhere, so `_cosine` and
    `_cosine_avg` legs shared both digests; the gate now also refuses a ref/cand whose implementation
    differs beyond the change under test. Pairs across a capture boundary must be re-measured; most
    bundle post-checks will report a digest mismatch on a new leg until re-finalized (P1-A, 1.10b).
    Audit still owed: any past comparison that paired a `_cosine` (pcos_*/hl_*) leg against a
    `_cosine_avg` leg carried a silent ~0.9%.
  * (3) The Megatron overlays' impl hash covers impl.py only. Add `sources` (a deliberate impl_digest break).
  * (4) DONE e5468c9f: capture detection by pattern.
* **P1-C, workflow paths never run on Megatron.**
  * QualityGate: default is `_cosine_avg` (cd75d661) and it now fires on an accepted config lever
    (9fbf1b33), but it has never run inside a workflow, and its default is not the run's WORKLOAD
    (matters for HF; the missing `--stack` is HF-only too).
  * quality_ab hardcodes ITERS=358 and nproc 8. quality_ab_final's T_CRIT 2.776 assumes n=5 unchecked.
  * ConfigSweep overwrites `config_flags` instead of merging (`curConfig = { ...curConfig, config_flags: sweep.flags_added }`, twice),
    reports the first lever, and never refreshes the digest. NEW (2026-10-21 audit): no role card reads
    `CONFIG.config_flags`, so an accepted lever is not carried into later gate/Finalize/Validate legs.
  * measure_packed_stack / quality_ab default to `packed.env`.
* **P1-D, docs.** README "no HF kernel" / `ref,s2..s7` / `--cond s7`, and the deployable-bundle line. RUN.md
  workload and skip_heads (missing rmsnorm_bwd_fc1_fused, grad_tail). ROADMAP P2-3 not marked done, gate table
  (soak/step-1 retired), dropout "open question". RESUME's old next-lists. The PENDING-descriptor rationale
  (launch.sh only warns).
* **P2.**
  * Megatron `rope_fused_qk` is the same "more accurate RoPE, loss +0.044%" case HF rejected in 4.37.
    It was accepted only because dropout masks the sign test. A bf16-exact Megatron rope (4.39 remedy)
    would make the policy consistent.
  * Why s8 diverges under constant LR is still not localized. The per-parameter grad comparison was never
    run, so the clearance is recipe-specific.
  * Megatron L0 oracles have no holdouts.
  * New task dirs get sign_min_z 4.0, validated on HF only; run one Megatron negative control.
  * measure_stack3.sh duplicates the stack definition. calibrate_band.py hardcodes the work dir.
  * Tests don't cover stack_overlay_env mapping, overlay_train install/failure, or quality_ab verdicts.
  * JS suites: there is no system node, but `~/.cursor-server/bin/linux-x64/*/node` works (RUNBOOK:148).
    2026-10-20: test_training_gate and test_workflow_dryrun pass. test_pipeline_integration on a same-config
    Megatron pair (hl_1016_205202_ref1 vs ref2) fails one check: `window_artifact_pct > 0.05` when warmups
    differ. That threshold was calibrated on unpacked legs; packed fixed-shape legs give 0.014%. Recalibrate
    it as a test fix, not a code fix.
  * The flash backend's 2.7x grad gap is unexplained (quarantined).

**PAUSED 2026-10-20 (user request):** `wf_7c87df08-799` (hf8, gate/up bwd GEMM lane) was stopped mid-run; its leftover container was stopped by ID. HF/TRL optimization is paused while the Megatron flow is reviewed.

**Next (autonomous), after hf8 (re-profile of wf_02793feb-78a):** accepted overlays cover 74% of the span.
* Kernel: `hf_gateup_bwd_gemm`, re-tuning inside the accepted gate/up overlay. Its bwd is 26.8% of the span, exp. +2.3%. Minor: `hf_flatgrad_side_tail` 0.6%, `hf_attn_gqa_kv_nocopy` 0.6%.
* Config: exposed FSDP comm 8.9% on rank 0, 56% overlapped. That needs adapter support for fsdp_config knobs (backward_prefetch, forward_prefetch, limit_all_gathers).
* DP token imbalance: a workload decision, not a lever.

**Older next-list (kept for history):** the remaining HF heads on top of hf4 with `_nm`: lm_head ~7.9%, q/o_proj ~7.2%,
k/v_proj ~2%. Also the engine-side ZeRO flat-grad `cat` on side streams (~4.8% of the span).

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
