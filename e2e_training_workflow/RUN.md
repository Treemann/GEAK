# Launching a run

## Read this first if you ran this workflow before 2026-10-26 (DECISIONS 4.86-4.103)

* **Outcome tests** use two looks (Pocock 0.0294 at each; `quality_ab_final.py --looks 1` only for a pre-registered
  single-look study). Both arms are checked; FAIL-stability needs >= 2 catastrophic cand legs; one catastrophic leg,
  or INCONCLUSIVE quality, runs the pre-registered extension. A base leg that NaN-exits drops its seed. The legs a
  verdict pairs by name must be one experiment (workload, seed, implementation, input contents; the control arm =
  base recipe with only lr / min-lr moved) -- a recipe test declares its key with `--allow-workload`.
* **quality_ab.sh** `--ctrl-base <cond>` runs the positive control ON a stack (e.g. `hf8`); `hf1..hf8` are arms.
* **Gates**: when the per-step median and the 10%-trimmed mean disagree (`estimators_disagree`) the orchestrator
  re-gates ONCE at >= 60 iterations (kernel: the Integrator, config: the Tuner). Every gate ref must carry the
  carried stack's config / impl digests; the Tuner returns `legs_measured` (all crossed by the verifier).
* **Kernel L0** is re-run by the gate verifier (`scripts/l0_verify.sh <task> <overlay> <out> --gpu N`, one GPU);
  its `correct` and evidence decide. The packed-attention head uses `tasks_v2/attn_fmha_ck_varlen_1022` (oracle v2).
* **Identity** also covers content (overlay directory trees, data / tokenizer / checkpoint fingerprints): legs
  measured before 2026-10-26 have no `content` record and must be re-measured to be references.
* **HF legs** cross-check the probe's token count against the trainer's num_items_in_batch every step; a leg
  that disagrees is unusable. HF headlines are POOLED tokens / time (rank-median kept as a field).
* **Bundles**: check_leg fails a leg whose recorded harness (rocm_compat / overlay_train / train_probe / hf_sft)
  or image differs from the pin; HF `bal` legs must show token balancing. Re-validate and re-pin after a harness change.
* **Ops**: bench_train runs from a private copy (editing it mid-leg is safe), refuses a leg below
  `GEAK_MIN_FREE_GB` (20) free disk, and the job runs in its own process group.

## Read this first if you ran this workflow before 2026-10-25

The 2026-10-25 audit fixes (DECISIONS 4.73-4.79) change what a run trusts and what it refuses:
* **Old legs are not references any more.** The probe now records every flag and kernel / collective env
  var; a pair in which only one leg carries that capture is refused as stale. Do not pass `baseline_dirs`
  or `quality_reuse` legs measured before 2026-10-25 -- Setup re-measures them anyway.
* **Agents decide nothing.** The gate verifier recomputes every gate pairing, L1, the main_grad applicability,
  Validate's headline pair and the outcome-test verdict; `status` / `cleared` are computed. A run that ends
  `unconfirmed` reports why in `status_reason`.
* **Flag+env levers work**: the Config Tuner returns `env_added`, e.g. `nan_from_grad_norm`
  (`--no-check-for-nan-in-loss-and-grad` + `GEAK_NAN_FROM_GRAD_NORM=1`, now in the Tier-0 pack).
* `quality_reuse: {prefix, ref_arm[, ctrl_arm][, allow][, allow_impl][, accept_unknown_harness]}` -- the
  check is built from what the run accepted; `allow` / `allow_impl` pre-declare anything else, and legs
  without the harness / env capture need `accept_unknown_harness: true` (recorded). A reused ctrl arm must be
  the same base stack (4.75).
* `quality_ab.sh` gained `--all-train-args` / `--all-env` (settings of every arm, e.g. `fast` on both arms of
  a kernel A/B) and the `s8a22` arm.
Verified in agent-driven runs wf_7300e31a-c76, wf_3e85bddf-a53, wf_e0ebcf87-87f, wf_e849d951-2cc (4.81-4.84).

## Read this first if you ran this workflow before 2026-09-23

Two changes alter what a run costs and what it will accept. Neither is optional.

**1. Existing baselines must be RE-MEASURED, not re-summarised.** `config_digest` moved to schema 3
(it covered five fields; `attention_dropout` 0.0 and 0.1 digested identically). The version lives in
the probe header, so legs recorded earlier stay schema 2 forever and a 2-vs-3 pair is refused with
`stale_summary`. Re-summarising cannot fix it — a v2 sidecar never recorded the new fields. Concretely:
the 7B `+4.54%` baseline needs one fresh leg pair before the next candidate can be gated.

**2. Setup now takes three baseline legs, not two** (`REPEAT: 3`). The L1 envelope needs more than
one sample of the noise per step; with two legs it collapsed wherever those legs happened to cross,
and both earlier runs had to hand-recalibrate mid-flight. The band now self-tests (see below), so if
three is not enough it will say so rather than quietly under-gating. Cost: one extra leg per run,
which also feeds the across-job throughput band.

**What to check in the result.** `validation` is still the official number. Additionally:
`accepted[].verdict.delta_source` must read `window-aligned` — a raw ratio can no longer be accepted
at all (`klass: no_aligned_delta`), because two identical 7B legs measured +4.5925% unaligned against
−0.0588% aligned. `accepted[].validation.superseded` flags any gate figure the Director's independent
re-measurement contradicts; quote `validated_delta_pct` when it is true. The band file's
`envelope_discriminating` / `envelope_status` say whether L1's per-step envelope was allowed to vote
or was demoted to advisory. `unexplored_heads` and `stop_reason` say what the run never got to —
Stage 2 stopped on the wall clock with 3 of 4 heads untried, and nothing used to record that.

## Manual test of the Megatron workflow (fast mode, no ConfigSweep) — 2026-10-20

**Recipe:** `workloads/qwen2_5_math_7b_packed_cosine_avg.env`. That is GBS 16 packs, dropout 0.1, warmup
11 + cosine to 10%, and `--ddp-average-in-collective`. It is the recipe the s8 stack is CLEARED on
(DECISIONS 4.24/4.25). Do not use `packed.env`: it is constant LR, no warmup, and NOT cleared (4.22).

**0. Before you start**
* The 8 GPUs are free: `rocm-smi --showpids` shows no KFD process, and `docker ps` shows no leftover
  `rocm/primus` container of yours.
* At least 80 GB free under `/home/treemann/geak_train`: `df -h /home/treemann/geak_train`.
* `bash scripts/preflight.sh`. This is the full check, including one ~10 min measurement leg; it must
  say GO.

**1. A session that has the Workflow tool** (see "Getting a session that HAS the Workflow tool" below)
```bash
cd <REPO>/e2e_training_workflow && bash scripts/launch_workflow_session.sh
# in that session ask: "do you have a Workflow tool?"  -- if no, stop here
date +%s%3N        # paste as t0_epoch_ms
```

**2a. Test M1: the whole pipeline on the BARE reference.** About 4-6 h. This is the main test.
It has a known ground truth: on the bare packed reference these heads were accepted before, gated
alone — `attn_fmha_ck_varlen` +4.27%, `rope_fused_qk` +3.89%, `swiglu_bwd_fwd_bw` +3.15%,
`lmhead_ce_dense_gemm` +2.27%, `mlp_fc1_gemm` +1.60%.
```
Workflow({ scriptPath: "<REPO>/e2e_training_workflow/training_workflow.js", args: {
  workflow_dir: "<REPO>/e2e_training_workflow",
  workload: "<REPO>/e2e_training_workflow/workloads/qwen2_5_math_7b_packed_cosine_avg.env",
  stack: "megatron", nproc: 8, gpu_ids: "0,1,2,3,4,5,6,7", iters: 30,
  budget: 1, max_gates: 2, kernel_budget: 2,   // floor: per-stack default (Megatron 0.30%, HF 0.25%; DECISIONS 4.54)
  time_budget_ms: 21600000, t0_epoch_ms: "<date +%s%3N>" }})
```
**2b. Test M2: gating ON the accepted stack (s8), stop rule included.** About 1.5-3 h.
The queue on s8 was exhausted on 2026-10-15. The expected outcome is "no head pays", or one small
head; both are valid results.
```
  ... same args, plus
  baseline_overlay: "<REPO>/e2e_training_workflow/scripts/stack_overlay_env.sh --cond s8",
  skip_heads: "mlp_fc1_gemm,rope_fused_qk,swiglu_bwd_fwd_bw,lmhead_ce_dense_gemm,mlp_fc2_oproj_gemm,attn_fmha_ck_varlen,qkv_lnlinear_gemm,lmhead_wgrad_reselect,rmsnorm_bwd_fc1_fused,grad_tail_fused_scale_check"
```
* Optional on either test: `quality_gate: "true"`. It adds about 3.3 h: REF and the final stack, 5 seeds,
  one epoch each. The default `quality_workload` is `_cosine_avg` as of 2026-10-20.
* It runs only if something was accepted.

**3. While it runs**
* Watch `/workflows` in the session.
* Legs appear under `/home/treemann/geak_train/runs/` (`baseline_td*`, `gate_*`, `validate_*`).
* Kernel lanes appear under `<REPO>/exp/team_*`.
* To stop: use TaskStop in the session. Then check `docker ps` and stop your leftover
  `rocm/primus` container by ID, after `docker inspect`. TaskStop does not kill it.

**4. What to check in the result**

| field | expected |
|---|---|
| `status` | `ok` |
| `baseline.useful_tok_s_per_gpu` | M1 about 2,870–2,900; M2 about 3,500 |
| `baseline.occupancy_verdict` | `clean` |
| `bands.floor_pct` | 0.25 |
| `config_diagnosis.verdict` | `kernel_work_can_pay` |
| `accepted[].verdict.delta_source` | `window-aligned` |
| `accepted[].verdict.estimator` | paired |
| `audit.l0.pass` / `audit.l1.pass` | `true`, with the evidence paths in the text |
| engagement | 8/8 ranks, fallback 0 |
| `validation` | the official number; quote `validated_delta_pct` |
| `stop_reason`, `unexplored_heads` | what was not tried |
| M1 deltas | the accepted head's delta is near the ground truth above |

**5. Re-measure by hand, outside the workflow:**
`scripts/measure_packed_stack.sh --workload workloads/qwen2_5_math_7b_packed_cosine_avg.env --conds ref,s8 --reps 2`.
Then run `compare_legs.py <ref>/probe <s8>/probe` per pair. Expected s8 vs ref: about +21%.

**Known, not blocking (RESUME "MEGATRON REVIEW 2026-10-20"):**
* Identity snapshot v5 (2026-10-21, DECISIONS 1.10b): legs and bands recorded before it cannot be paired
  with new legs (the gate says RE-MEASURE). Existing bundles' `launch.sh` post-checks will report a digest
  mismatch on a new leg (cosine/HF: config_digest; packed constant-LR: impl_digest; the oldest bundles
  check nothing that moved). The training itself is unaffected; re-pin when re-finalizing.
* The deliverable is `runs/finalize_bundle_1021_s8_cosine_avg` (s8 on `_cosine_avg`, v5 digests pinned).
  `finalize_bundle_1015_s8` runs `packed.env` (not cleared); do not use it as the recipe.
* ConfigSweep merges levers wrongly. It is not exercised here.

## The accepted Megatron stack (s8)

The recipe is `qwen2_5_math_7b_packed_cosine_avg.env` (above).
* GBS 16 packs and dropout 0.1 are CONFIRMED (DECISIONS 4.21).
* `packed.env` keeps its "PENDING" note because six bundles pin its sha256. A mismatch only makes
  `launch.sh` print a WARNING; it does not refuse.

**Accepted** (each gated against the BARE packed reference; increments measured stacked, in order):

| head | gated | stacked increment |
|---|---|---|
| `rope_fused_qk` + `swiglu_bwd_fwd_bw` | +3.89% / +3.15% | +7.2% together |
| `mlp_fc1_gemm` | +1.60% | +1.64% |
| `lmhead_ce_dense_gemm` | +2.27% | +2.5% |
| `mlp_fc2_oproj_gemm` | inconclusive_noise | +1.74% |
| `attn_fmha_ck_varlen` | +4.27% | +5.25% |
| `qkv_lnlinear_gemm` (chains after fc1) | inconclusive_noise | +1.05% |

All seven stacked: **+20.7%** (2,866 -> 3,460 useful tok/s/GPU).

**Queue: exhausted.** The re-profile on the stack shows main-stream GEMMs at ~75%, exposed
all-reduce at 7.4%, and nothing else above 5.3%. A follow-up `fast` run on s8 needs the `skip_heads` of Test M2 above.
`lmhead_wgrad_reselect` is subsumed by the accepted lm-head overlay, which binds the same site.

**Gate the next head ON the stack (2026-10-14).** Pass
`baseline_overlay: "<REPO>/e2e_training_workflow/scripts/stack_overlay_env.sh --cond s8"` (s8 = the
seven + `rmsnorm_bwd_fc1_fused`, accepted ON s7 on 2026-10-15; use the latest condition).
* Setup, Profile, every gate leg, Finalize and Validate then run on top of the seven kernels.
* The deltas the run reports are increments on that stack.
* Do NOT pass bare `baseline_dirs` with it. A reused leg must have run the same stack (same
  `GEAK_OVERLAY` basenames and `GEAK_OVERLAY_SHA`). Otherwise Setup measures three fresh stacked
  legs.

`stack_overlay_env.sh` reuses one content-keyed install (`overlays/stackenv_<cond>_<hash>`) and
re-verifies it on every call, so the CK attention build is paid once.

**Before calling a stack safe to train with, run the outcome gate** (DECISIONS 4.22/4.23).
* Pass `quality_gate: "true"`. Optional: `quality_seeds`, and `quality_workload` (default
  `qwen2_5_math_7b_packed_cosine_avg.env` since 2026-10-20).
* It adds about 3.3 h: REF and the final stack, 5 seeds × one epoch (measured 2026-10-25: ~3.3 h for 10 legs).
* An INCONCLUSIVE first verdict runs ONE pre-registered seed extension; a FAIL-stability (one chaotic seed is
  enough, 4.82) is final and is NOT extended. Do not re-run a NOT CLEARED test until it passes: decide a
  doubt with a fresh pre-registered study (P0-4q, 4.85).
* Outside the workflow: `scripts/quality_ab.sh --workload ... --arms ref,cand --cand-cmd "<cmd>"`,
  then `scripts/quality_ab_final.py <prefix> --cand cand`.

**Without `baseline_overlay`, re-measure on the stack after any acceptance.** The workflow gates against the bare reference.
`scripts/measure_packed_stack.sh --conds ref,s5,s6,s7 --reps 3` is the interleaved, cumulative
re-measure: it verifies sha256 for every overlay and fixes the stacking order. Add a condition there
for a new head. `--profile 12:13` turns one leg into a profiler leg.

## The 7B queue as of 2026-10-08 (five runs in) — UNPACKED workload, historical

Superseded: on packed data the +12.36% stack below measured +0.08–0.28% (padding wins).

Expected gain is computed the way the orchestrator does it, `cp_share × (1 − 1/attainable)`, which
has now been validated twice: predicted 3.94% → measured **+4.66%** (lmhead), and the fc2 head
predicted from a 36.17%-share GEMM that the Architect correctly ranked *below* smaller heads with
more headroom.

**Decided:**

| head | result |
|---|---|
| `lmhead_ce_unmasked_gather` | **accepted, +4.66% validated** |
| `mlp_fc2_down` | **accepted, +7.34% validated** — wgrad 13× closer to fp32 than the original |
| `attn_fmha_fwdbwd_gqa_d128_causal` | **rejected** — backward defect, step-1 grad_norm −88.50% |
| `mlp_fc1_swiglu` | **rejected** — inf gradient at iteration #2, reproduced twice |
| `attn_dense_wgrad_rowslice` | **+1.147% on top of fc2, real** — `inconclusive_noise`, not a no_win: its own runtime row-slice makes step time data-dependent (IQR 2.2–2.9%), so its band exceeds its signal |
| `attn_fmha_bwd_retry` | **+0.31%** (2026-10-08, four pairings spanning 0.014 pts, both null controls exactly 0.0000%, step-median cross-check +0.304%). `inconclusive_noise` only because the *identity-rebind reference* legs carried 2.9–3.1% IQR. Its value is correctness, not speed: L0 rates it strictly more accurate, the incumbent failing an analytic `dq` invariant by 1355.79× |

**Combined, measured** (not derived — `lmhead + mlp_fc2_down` via `measure_stack.sh`): **+12.36%**,
four pairings spanning 0.12 points, null control +0.0706%, `step_median_ms` cross-check +12.324%.
The two **compound** rather than add.

**Open**, from the re-profile after `mlp_fc2_down` was accepted:

| head | cp_share | expected gain |
|---|---|---|
| `mlp_fc1_swiglu_rowslice` | 17.45% | **5.82%** |
| `attn_outproj_rowslice` | 16.34% | 3.57% |
| `lmhead_ce_gather_restack` | 4.4% | 2.93% |
| `attn_fmha_bwd_retry` | 19.35% | 2.52% |
| `qkv_lnlinear_rowslice` | 3.97% | 0.87% |

Two of those are retries informed by a rejection: `mlp_fc1_swiglu_rowslice` follows the inf-gradient
candidate, and `attn_fmha_bwd_retry` carries the backward diagnostic forward. Note also that the two
accepted heads were measured against the BARE stack in separate runs, so their deltas do **not**
simply add — a run that carries both is what would settle the combined figure.

## Making a follow-up run cheaper

Measured on the 2026-09-24 run (10.6 h, 2 heads), the time went:

| block | cost | share |
|---|---|---|
| kernel_lane (the actual optimisation) | 4.9 h | 46% |
| gate legs | 1.6 h | 15% |
| profile / re-profile (5×) | 1.2 h | 11% |
| Setup baseline legs | 0.7 h | 7% |
| finalize + validate | 0.6 h | 6% |

Two of those are avoidable on a follow-up run, and neither is the 46%:

**`baseline_dirs`** — pass clean legs of the same identity and Setup reuses them instead of
re-measuring. The Director still verifies each one (`config_digest`, `digest_schema`, occupancy,
dispersion, not profiled/captured) and re-measures any that fails. Safe only since schema 3: under
schema 2 the digest covered five fields and two different workloads could share one.

**`skip_heads`** — comma-separated head short-names whose verdict is already known. Applied at both
queue-construction sites, so a re-profile after an accept cannot re-introduce them. Without it, every
run re-profiles the bare stack, rebuilds the same queue, and spends the budget re-deriving decided
answers — which is what happened twice before it existed.

One trap that is *not* a knob: do **not** raise `iters` just for the gate legs. The L1 band is
calibrated at `iters`, and a per-step envelope is fitted to a particular argv *and* leg length.
Running 60-step gate legs against a 30-step band cost about an hour of extra control pairs on
2026-09-24 and produced a `pass=false` that the Auditor had to disprove by hand. If the legs are too
short, raise `iters` for the whole run.

## Before you spend hours

```bash
cd <REPO>/e2e_training_workflow
bash scripts/preflight.sh          # ~2.5 min, includes a real measurement leg
bash scripts/preflight.sh --no-bench   # ~10 s, everything except the leg
```
It checks GPUs and whether the box is quiet, the image, both workload descriptors and the
checkpoints/tokenizers/dataset they name, every script and all six role prompts, the kernel layer it
recurses into, `harness_lib.py` (task dirs need it), the test suites, disk, and finally one real
measurement leg. `GO` / `NO-GO` at the end. Every check corresponds to something that actually went
wrong during bring-up.

## Getting a session that HAS the Workflow tool

**This is the step that silently blocks everything.** A normal session on this box does not have the
`Workflow` tool at all — it is simply absent from the toolset, and an agent asked to "run the
workflow" will improvise something else instead of saying it cannot.

```bash
bash scripts/launch_workflow_session.sh
```

Two causes, both verified on `claude 2.1.273`:

1. `CLAUDE_CODE_SIMPLE=1` in `~/.claude/settings.json` strips the toolset to exactly
   `Bash, Edit, Read`. Clearing it yields `Agent, Bash, ..., Workflow, Write`.
2. Public builds (≥2.1.x) **reject** `--effort ultracode`; the Workflow/parallel/phase primitives are
   gated behind the `enableWorkflows` + `ultracode` **settings keys**. Having the tool present is not
   the same as it *executing* the JS pipeline rather than backgrounding it, so both are injected —
   mirroring `interface/run_e2e.py`.

Equivalent one-liner if you prefer:
```bash
cd <REPO>/e2e_training_workflow
IS_SANDBOX=1 CLAUDE_CODE_SIMPLE= claude \
  --settings '{"enableWorkflows":true,"ultracode":true,"env":{"CLAUDE_CODE_SIMPLE":""}}' \
  --dangerously-skip-permissions
```

**Verify before launching a run** — ask the new session: *"do you have a Workflow tool?"* If the
answer is no, nothing below will work.

### Why a plain session says "Not logged in · Please run /login"

Not an auth failure, and there is no prompt to answer. `~/.claude/settings.json` sets
`ANTHROPIC_API_KEY` to the literal string `dummy` — the real credential is
`ANTHROPIC_CUSTOM_HEADERS` against a gateway — and `~/.claude.json` lists that key under
`customApiKeyResponses.rejected`. A rejected key counts as **no credential**, so the session goes
straight to the login screen, and `--dangerously-skip-permissions` suppresses the approval dialog, so
nothing asks you to choose.

Verified headlessly, on this box:

| configuration | result |
|---|---|
| `ANTHROPIC_API_KEY` unset, header present | `Not logged in · Please run /login` |
| `ANTHROPIC_AUTH_TOKEN` set instead, same header | authenticates |
| the file's header, headless | authenticates |
| the running session's header, headless | authenticates |

So nothing is expired and no credential is missing. The launcher swaps the API key for
`ANTHROPIC_AUTH_TOKEN`, whose path does **not** go through the approval gate — no edit to
`~/.claude.json` is needed. The token is a placeholder; the gateway authenticates on the header.

## Stage 1 — shake the orchestration out on 0.5B (~20 min, cheap)

The point is **not** a speedup. It is the first time the six role prompts are executed by an agent;
the deterministic half has 50 unit checks, the agent half has none. Expect to edit prompts.

```
Workflow({
  scriptPath: "<REPO>/e2e_training_workflow/training_workflow.js",
  args: {
    workflow_dir: "<REPO>/e2e_training_workflow",
    workload:     "<REPO>/e2e_training_workflow/workloads/qwen2_5_0_5b.env",
    nproc: 2, gpu_ids: "0,1",
    iters: 30,          // 30 leaves ~29 steady steps at 0.5B; below 12 the gate refuses the leg
    budget: 1,          // one kernel task
    max_gates: 2,
    kernel_budget: 2,
    t0_epoch_ms: <paste `date +%s%3N`>   // see "Why the launch args carry a timestamp" below
  }
})
```

## Stage 2 — the real run on 7B (~2-3 h)

```
Workflow({
  scriptPath: "<REPO>/e2e_training_workflow/training_workflow.js",
  args: {
    workflow_dir: "<REPO>/e2e_training_workflow",
    workload:     "<REPO>/e2e_training_workflow/workloads/qwen2_5_math_7b.env",
    nproc: 8, gpu_ids: "0,1,2,3,4,5,6,7",
    iters: 30,          // 11-step warmup at 7B leaves 19 steady steps; below ~25 iters it is refused
    budget: 2,
    max_gates: 4,
    kernel_budget: 3,
    time_budget_ms: 18000000,
    t0_epoch_ms: <paste `date +%s%3N`>   // REQUIRED, or time_budget_ms is not enforced
  }
})
```

### Why the launch args carry a timestamp

The workflow sandbox has **no clock**: `Date.now`, argless `new Date`, `performance` and `process`
are all absent or throw, so a resumed run replays identically. Probed on this box, only
`new Date(<explicit ms>)` still works. So the script cannot time itself — it takes the start from
`t0_epoch_ms` and reads "now" through an agent, which runs bash and can call `date`. On resume a
cached stamp replays its original value, which is what we want: hours parked between sessions must
not count against a budget meant to bound GPU work. Omit `t0_epoch_ms` and the run proceeds with
`time_budget_ms` unenforced, logging a warning instead of implying a bound it is not applying.

## What should happen

* Setup measures the baseline **twice** and derives this run's bands from those two legs, using
  `compare_legs.py` — never a raw ratio, which carries a sample-composition term larger than the
  signal.
* Profile reports the step decomposition and a **read-only** diagnosis. On this workload it returns
  **`kernel_work_can_pay`**, with 70.6% padding listed as an *opportunity* (packing is a bigger lever)
  rather than a blocker — a kernel win is not diluted by padding.
* The head at 7B is a GEMM at **19% of the critical path**, so a 20% kernel win is worth ~4% on the
  headline, well clear of the 0.15% aligned band.
* Kernel lanes use the whole GPU pool; each gate **drains them first**.
* `deep_mode: true` adds ConfigSweep. See "Deep mode" below; it has run end to end since 2026-10-08.

## Where it is most likely to fail first

1. **Role prompts** — path and command details, first execution ever.
2. **The recursive `kernel_workflow` call** — the task dir matches its documented contract and its
   own unittest runs, but the kernel layer has never consumed one.
3. **Overlay engagement at 8 ranks** — proven at 2.

## Reading the result

`wfReturn.validation` is the Director's independent re-measurement and is the official number.
Per-candidate deltas are measured against different references and do **not** sum to it.
Check `delta_source` on each verdict: `window-aligned` is trustworthy, the raw-ratio fallback is
labelled as carrying the artifact.

## Deep mode: ConfigSweep (as of 2026-10-09)

```
Workflow({scriptPath: ".../training_workflow.js", args: {
  ..., "deep_mode": "true", "phases": "setup,profile,strategize,config",
  "config_levers": "packing",            // or "recompute_granularity,attention_backend", or omit
  "budget": 0, "max_gates": 2, "t0_epoch_ms": "<date +%s%3N>" }})
```

* `config_levers` names the levers to measure, in ONE run so they share a baseline and a band.
  Omit it and the Config Tuner picks one Tier-0 lever itself.
* Every Class-A lever must return an L1 result (`trajectory_gate.py compare`) or it cannot be
  accepted. This is what stops Megatron's **flash** attention backend on this ROCm stack, whose
  gradient is 2.7x off under a correct forward.
* **The Tier-0 Class-A pack is measured and none of it pays on the 7B reference**: comm overlap
  +0.118% (costs 780 MB), recompute selective −3.40% (costs 838 MB), attention flash −0.73% and
  unsafe. Do not re-run them on this workload expecting a different answer.
* **Packing is the lever that pays: 833 → 3,048 useful tok/s/GPU (3.66x).** It is the one Class-B
  lever with a gate (`scripts/pack_gate.py`); GBS/MBS/seq stay refused. After packing,
  `--global-batch-size` counts packs — divide it by the packing ratio (~3.3) to keep samples per
  optimizer step.

To pack by hand:
```
docker run ... rocm/primus:v26.4 python /repo/e2e_training_workflow/scripts/pack_sft.py \
    /work/data/sft_ultrachat20k.jsonl /work/data/sft_ultrachat20k_packed_7b.jsonl \
    --tokenizer /models/qwen2_5_math_7b --seq 4096
python3 scripts/pack_gate.py --workload workloads/qwen2_5_math_7b.env \
    --source /work/data/sft_ultrachat20k.jsonl --packed /work/data/sft_ultrachat20k_packed_7b.jsonl \
    --model /models/qwen2_5_math_7b --pack-index 1
```

## A second stack: HF TRL (as of 2026-10-09)

```
bash scripts/bench_train.sh --stack hf_trl --workload workloads/qwen2_5_0_5b.env --nproc 2 --iters 30
bash scripts/bench_train.sh --stack hf_trl --workload workloads/qwen2_5_math_7b.env --nproc 8 --iters 30 \
    --extra-train-args "--hf-fsdp full_shard"         # 7B: DDP with fp32 master weights OOMs
```

* Recipe knobs are ARGUMENTS (`--extra-train-args`), never environment: the container re-exec
  forwards only ATTENTION_BACKEND and RETRY_CONTAMINATED, so an env knob silently does nothing.
* HF pads dynamically, so its step time follows content. The gate reads a PAIRED per-step delta for
  exactly this reason; per-leg medians put two identical HF runs 1.15% apart.
* Compare stacks only same-code, same-day, same recipe (dropout, master-weight precision). An old
  Megatron leg once made HF look 1.8x "better" in loss for reasons that had nothing to do with HF.

## The session must be able to read `kernel_workflow/` (found 2026-10-10)

The kernel track calls `kernel_workflow/kernel_lane.js` as a child workflow, and the runtime only loads
a child script from the session's working directory or a directory added to it. A session started IN
`e2e_training_workflow/` cannot see its sibling, and the run dies at the first lane call -- after an
hour of Setup, Profile and Strategize. Either start the session one level up (`GEAK-claude/`), or add
the directory: `/add-dir ../kernel_workflow`, or in `.claude/settings.local.json`:

```json
{"permissions": {"additionalDirectories": ["/abs/path/to/GEAK-claude/kernel_workflow"]}}
```

(The first attempt to fix this normalised the `..` out of the path, commit 8ffdaa71. That was a wrong
diagnosis: the normalised path was refused identically. The normalisation is kept because it is
harmless and makes the path readable in logs; the fix is the directory.)
