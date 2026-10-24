# e2e_training_workflow — End-to-End LLM SFT Training-Throughput Optimizer (AMD Instinct MI GPUs)

A second system layer over the **UNCHANGED** single-kernel `kernel_workflow`, aimed at **SFT training
throughput** instead of serving throughput.

**Where it stands (2026-10-22), Megatron, Qwen2.5-Math-7B, 8x MI308X:**
* Useful throughput went **839 -> 3,504 tok/s/GPU (4.18x)** on the recommended production recipe,
  `workloads/qwen2_5_math_7b_packed_cosine_avg.env`: GBS 16 packs, default dropout, standard LR
  schedule, `--ddp-average-in-collective`. These are 30-step MEDIANS. **Production wall time over a full
  epoch is 3,396 tok/s/GPU (~4.12x)**: host stalls the median excludes cost ~1.5-3%, more on overlay
  legs (DECISIONS 4.41, 4.42).
* Packing gives 3.42x. It was accepted by `deep` mode through a structural gate.
* Eight kernels stacked on top give **+21.6%** median (**~+19.5% wall**, 19 one-epoch pairs), and one
  DP-comm lever +0.9% (`knowledge/sft_measurement.md`).

Every number is useful (non-pad, loss-unmasked) tokens. Per-change deltas are paired against an interleaved
reference; the cumulative multipliers (4.18x, 2.42x) are cross-session medians (audit 2026-10-22, F7).

Second stack (HF TRL, 7B, unpacked): **1,083 -> 2,618 useful tok/s/GPU (2.42x)**, cleared for
training (DECISIONS 4.40); with token-balanced rank assignment (`--hf-balance-tokens on`, same samples per
step) **+15.2..+15.9% paired on top**, cleared by outcome test P0-4m (DECISIONS 4.61). These are corrected figures (4.33): an earlier "2,269 / 2.07x" for hf3
came from a warmup detector that over-trimmed variable-shape legs. The two stacks do NOT run the
same recipe and their tok/s are not a framework comparison: HF is unpacked with dynamic padding and
RMSNorm eps 1e-6 (the model's); Megatron is packed with eps 1e-5 (measured better there, P0-4l,
DECISIONS 4.55). Since 2026-10-23 every Megatron leg records the eps mismatch in recipe_warnings.json.
* Two levers with a bitwise-identical trajectory: ZeRO-2 +19%, and skipping TRL's per-micro-batch
  metric gathers (+28% bare, +21% on the stack); paired deltas corrected in DECISIONS 4.33c.
* Eight overlays accepted through the same workflow and gates:
  * attention, RMSNorm, MLP gate/up + SwiGLU, MLP down_proj;
  * the fused q/k/v/o projections and the fused lm_head + loss;
  * FSDP flat-grad accumulation, bitwise to the original;
  * RoPE, a bf16-exact variant. The first, more accurate variant was rejected by the L1 direction rule.
* The final configuration (hf8) is **cleared for training** by the one-epoch, 5-seed outcome test
  (P0-4k): held-out −0.00002 vs a control of +0.0065, no episodes.
* Recipe: `workloads/qwen2_5_math_7b_hf_cosine_zero2_nm.env` + `stack_overlay_env.sh --cond hf8`.

Training safety: the 8-kernel stack is **cleared for training under a standard LR schedule**
(warmup + cosine; production descriptor `workloads/qwen2_5_math_7b_packed_cosine_avg.env`). That rests
on pre-registered one-epoch, multi-seed outcome tests: held-out loss −0.0023 vs REF, no instability
episodes (DECISIONS 4.24), and again with `--ddp-average-in-collective` (4.25). It is NOT cleared for
constant-LR, no-warmup training (4.22). **Deliverable bundle (from 2026-10-24): `finalize_bundle_1024_s8a22`** =
s8 with the 1022 attention kernel, default mode `fast` (+ synchronized GC + the NaN check moved to the grad norm):
3,567 useful tok/s/GPU vs bare 2,892 on the re-run; cleared by outcome tests P0-4o (`fast`, 4.70) and P0-4p (the 1022
kernel, 4.72). `finalize_bundle_1021_s8_cosine_avg` (the same with the 1013 kernel) stays pinned for reproducing
earlier results.
`finalize_bundle_1015_s8` runs the constant-LR `packed.env` and is NOT the deliverable.
The recipe keeps Megatron's RMSNorm eps 1e-5 although the model was pretrained with 1e-6: the
model-matched 1e-6 was outcome-tested and lost (held-out +0.0067, 5/5 seeds; P0-4l, DECISIONS 4.55).

What is NOT established yet:
* MoE, TP/PP > 1, multi-node and fp8 (Megatron is validated single-node, TP1 x DP8, dense, bf16).

Priorities: `ROADMAP.md`, "Near-term priorities". Same fractal two-altitude doctrine as `../e2e_workflow/`,
but the metric, the bottleneck model and the correctness gate are different enough that it is a
separate workflow rather than a mode of the inference one.

`../e2e_workflow/` is **not edited** by this project; anything shared is consumed read-only.

## Why it is not just "e2e_workflow with a training adapter"

| | inference (`e2e_workflow`) | training (this) |
|---|---|---|
| headline | serving tok/s | **useful** (non-pad, loss-unmasked) tok/s/GPU |
| bottleneck model | `pct_gpu_time x achievable_speedup` | **critical-path share** — a fully overlapped collective is worth zero |
| correctness | byte-identical greedy output | **L0** (fwd+bwd op numerics vs a frozen oracle) + **L1** (step-1 forward probe + strict early sign rule; the per-step envelope and step-1 grad_norm are advisory) |
| concurrency | lanes and the serving slot coexist | **a gate needs the whole box** — lanes drain first |
| scarce resource | agent time | **gates** (one A/B = two job launches, exclusive) |
| noise band | 2-3% | **paired per-step ratio, stall-trimmed dispersion; across-job floor 0.30% Megatron / 0.25% HF** (2,072 null pairs, DECISIONS 4.54) |

**`DECISIONS.md` records every framework-level choice with the measurement that forced it.**
`PLAN.md` carries the narrative; `RUNBOOK.md` the commands.

## Measured facts this design rests on
All from 8x MI308X (gfx942), `rocm/primus:v26.4`, bare Megatron `pretrain_gpt.py --sft`. These are
the facts the design was built on (September). Later ones that changed it are summarised at the end of
this list and recorded in `DECISIONS.md`.

- **71.6% of tokens are padding** on the reference workload: the stack-conventional nominal tokens/s
  overstates useful work by **3.5x**. Hence the headline counts `loss_mask.sum()`.
- **Warmup detection is worth 70x in gate sensitivity.** Same run: skip-2 gives a 12.75% spread,
  skipping to the measured warmup end gives 0.18% — with an unchanged median.
- **Raw spread vs robust spread differ 500x** on a clean run (38.6% vs 0.072%): two isolated stalls
  carry all of it. Both are always reported; the band uses the robust one.
- **Across-job reproducibility, window-aligned, is 0.0589% worst over six 7B pairs spanning four
  days** (0.31-0.42% at 0.5B, where host jitter is a far larger share of the step). The earlier
  0.118% figure was unaligned and mostly window artifact. Bands are derived per pair, and the floor
  depends on whether the delta was window-aligned: 0.15% if it was, 0.5% if it was not.
- **Training is chaotic**: two identical runs are bit-identical at step 1, 0.76% apart by step 5, and
  **2.04 apart in absolute loss by step 21**. L1 is therefore a step-indexed envelope, anchored on
  the one bit-reproducible quantity available (step-1 loss).
- **The top head by critical path is a BACKWARD kernel** (`FusedAttnFuncBackward`, 21.5%), which is
  why an extracted task is fwd+bwd or it is nothing.

Later measurements that changed the design:
- **Per-leg medians are not a valid estimator.** Window placement alone made +4.59%, and two
  identical HF legs read 1.15% apart. The gate reads a PAIRED per-step ratio over the same batches.
- **The per-step envelope cannot discriminate** under bf16 chaos: out-of-sample null legs fail it.
  It is advisory. Step-1 grad_norm moves up to 85% between correct implementations and is advisory
  too.
- **Every kernel win on the padded workload was a padding win.** Packing (3.42x, 1% padding) erased
  the old +12.36% stack. The packed profile is ~75% dense GEMM.
- **Overlays use side streams.** A side-stream kernel is on the critical path only where the main
  stream is idle (DECISIONS 4.16).

## Layout
```
training_workflow.js          orchestration: fast/deep modes, lane/gate barrier, gate arithmetic,
                              the ConfigSweep seam, recursive dispatch into ../kernel_workflow/
roles/                        6 agent prompts (director, architect, profiler, extractor,
                              convergence auditor, integrator)
knowledge/                    the doctrine, carrying measured numbers rather than advice
scripts/bench_train.sh        stack-neutral dispatcher: lifecycle, occupancy guard, hang watchdog
scripts/adapters/megatron.sh  framework adapter: Megatron-LM --sft + TE (+ capability manifest)
scripts/adapters/hf_trl.sh    framework adapter: HF Transformers + TRL (hf_sft.py; FSDP at 7B)
scripts/train_probe.py        torch-level probe: steps, useful tokens, memory, grad norm, profiler
scripts/probe_summarize.py    policy: warmup detection, steady window, dispersion, config digest
scripts/parse_train_profile.py  step decomposition + Top-N + the read-only config diagnosis
scripts/compare_legs.py       throughput delta: window-aligned + paired per-step ratio
scripts/calibrate_band.py     across-job floor from null pairs
scripts/trajectory_gate.py    L1: step-1 forward probe + strict sign rule (envelope advisory)
scripts/pack_sft.py           packing (pure data transform) / pack_gate.py: its structural gate
scripts/stack_overlay_env.sh  THE definition of the accepted stacks (Megatron ref, s2..s8; HF hf1..hf8): install,
                              verify the gated bytes, print a leg's --extra-env
scripts/measure_packed_stack.sh  interleaved cumulative re-measure of the accepted packed stack
scripts/capture_train_shapes.py  fwd+bwd capture -> immutable L0 oracle
scripts/overlay_train.py      reversible rebind + per-half engagement proof
scripts/gpu_occupancy.py      isolation guard for a shared node
scripts/rocm_compat.py        ROCm/upstream patches + probe/capture install point
workloads/*.env               pinned workload descriptors (config IR, descriptor role, floor);
                              qwen2_5_math_7b_packed.env is the current reference
```

## Modes
`fast` (default) = fixed config, kernel track only, bounded by wall clock **and** gate count.
`deep` = adds ConfigSweep (`deep_mode: "true"`, `config_levers`).
* Class-A levers (overlap, recompute, attention backend) must pass L1.
* Packing is the one Class-B lever with a gate (`pack_gate.py`).
* GBS/MBS/seq stay refused in code.

See `RUN.md` for both.

## Running the pieces today
```bash
# one measurement (inside the container; --workload is required, there is no default)
bash scripts/bench_train.sh --workload workloads/qwen2_5_math_7b.env --nproc 8 --iters 30

# a profiled window (never a gate leg: status is forced to unusable)
bash scripts/bench_train.sh --workload ... --profile-steps 12:13 --profile-ranks 0,4
python3 scripts/parse_train_profile.py <run>/probe <run>/bench_summary.json

# L1: calibrate from two baseline runs, then judge
python3 scripts/trajectory_gate.py calibrate <probeA> <probeB> -o band.json
python3 scripts/trajectory_gate.py compare <ref_probe> <cand_probe> --band band.json

# the accepted packed stack vs the bare reference, interleaved, on the production recipe
bash scripts/measure_packed_stack.sh --workload workloads/qwen2_5_math_7b_packed_cosine_avg.env --conds ref,s8 --reps 2

# tests (no GPU; the .js ones need node)
node scripts/tests/test_training_gate.js
node scripts/tests/test_workflow_dryrun.js
for t in scripts/tests/test_*.py; do python3 $t; done
```

## Status
Implemented and exercised end to end on real runs (Megatron, 0.5B and 7B; more than ten Workflow
runs with `agents_error: 0`): measurement, profiling, extraction, the recursive kernel lane, overlays
(stackable, with per-rank engagement proof), L0, L1, the packing gate, ConfigSweep, independent
validation, and bundle delivery.

The HF TRL adapter runs the same workflow and gates unchanged: eight overlays accepted and the final
configuration cleared for training (DECISIONS 4.27-4.40).

Open, in priority order (2026-10-23; full list with rationale in `RESUME.md`, "OPEN AFTER THE 2026-10-23 REVIEW"):
* correctness coverage: the GEMM overlays' fp32 main_grad wgrad path in L0 (G5); mid-depth oracle
  captures (all are layer 0); HF L1 backward check;
* identity: rocm_compat / harness version outside the digest (OVL-6);
* throughput: HF token-balanced rank assignment; the 1022 attention and the NaN-check / zero_ passes on s8;
* MoE / TP > 1 (P3, deferred by the owner).
Done since the last version of this list: ConfigSweep merges and carries accepted levers (P1-C); the
QualityGate runs inside a workflow (wf_b4b35fd3, 2026-10-22), including its seed extension.

## Launching a full `fast` run from Claude Code

The deterministic half runs anywhere; the training stack only exists inside the image, and
`scripts/bench_train.sh` bridges that automatically (it re-executes itself in the container and
translates paths both ways), so Claude Code can be started on the host.

**1. Prerequisites** (all already true on the reference box)
```bash
docker images | grep rocm/primus              # the image
ls ~/models/qwen2_5_math_7b_torch_dist        # a Megatron torch_dist checkpoint
ls ~/geak_train/data/sft_ultrachat20k.jsonl   # the frozen dataset
rocm-smi --showpids                           # "No KFD PIDs" = the box is quiet
```

**2. Smoke the measurement path first.** One clean leg before spending a run:
```bash
cd <REPO>/e2e_training_workflow
bash scripts/bench_train.sh --workload workloads/qwen2_5_0_5b.env --nproc 2 --iters 8
# expect status: ok, occupancy_verdict: clean
```
Relative paths are fine: the script resolves them before it re-enters the container (the container
starts in `/`, so an unresolved relative path would be looked up there and not found — changing the
caller's directory does not help, because the lookup happens on the other side). A path that resolves
outside every mount is rejected with the mounts listed, rather than failing obscurely inside.

**3. Start Claude Code with permissions auto-approved** (the workflow spawns many agents and runs
benchmarks):
```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

**4. Invoke the Workflow.** `workflow_dir` must be this folder — a JS workflow cannot read its own
path.
```
Workflow({
  scriptPath: "<REPO>/e2e_training_workflow/training_workflow.js",
  args: {
    workflow_dir: "<REPO>/e2e_training_workflow",   // REQUIRED
    workload:     "<REPO>/e2e_training_workflow/workloads/qwen2_5_math_7b_packed.env",  // REQUIRED, no default (the packed descriptor is the current reference)
    nproc: 8, gpu_ids: "0,1,2,3,4,5,6,7",
    iters: 30,             // steady-state steps per leg (warmup is detected, not assumed)
    budget: 2,             // kernel tasks for a first run
    max_gates: 4,          // ONE A/B = two job launches and it blocks every lane
    kernel_budget: 3,      // passed down to each recursive kernel_workflow run
    time_budget_ms: 18000000,
    t0_epoch_ms: <paste `date +%s%3N`>   // REQUIRED for time_budget_ms to be enforced
  }
})
```
`t0_epoch_ms` is not optional noise. The workflow sandbox has **no clock** — `Date.now`, argless
`new Date`, `performance` and `process` are all absent or throw, so that a resumed run replays
identically. The script therefore cannot time itself: it takes the start from `t0_epoch_ms` and
reads "now" through an agent (agents run bash, so they can call `date`). Omit it and the run still
works, but `time_budget_ms` is **not applied** — the lane loop falls back to its `budget` and
`max_gates` bounds, and says so in the log rather than reporting a deadline it is not enforcing.
Start with `workloads/qwen2_5_0_5b.env`, `nproc: 2`, `budget: 1` to shake the orchestration out
cheaply — a 0.5B leg is ~1 s/step against ~11.6 s at 7B.

**What it will do, and where it will stop**
* Setup measures the TRUE baseline **twice** and derives this run's bands from those two legs.
* Profile emits the step decomposition plus a **read-only** config diagnosis. On the unpacked
  workload it flags padding (70.6%) as the opportunity that `fast` mode cannot take. On the packed
  descriptor it returns `kernel_work_can_pay`.
* The kernel loop runs lanes on the GPU pool, then **drains them** for each gate: a training A/B
  needs every GPU.
* `deep_mode: "true"` adds ConfigSweep after Strategize. See `RUN.md` "Deep mode".
* `skip_heads` excludes heads already accepted; `baseline_dirs` reuses measured null legs.
* `quality_gate: "true"` adds the one-epoch, multi-seed outcome test of the final stack (DECISIONS
  4.23). Without it the result states that the stack is not cleared for long training.
* `baseline_overlay: "<REPO>/e2e_training_workflow/scripts/stack_overlay_env.sh --cond s8"` gates
  every head ON TOP of the accepted eight-kernel stack instead of the bare reference. Every leg of
  the run carries it, and every delta is then an increment on that stack. Without it, re-measure an
  accepted head on the stack with `scripts/measure_packed_stack.sh`, because increments change when
  stacked.
* A leg contaminated by a foreign GPU process is re-measured (bounded); exhausting the retries stops
  the run rather than quietly spending the remaining gates on a busy node.

**Reading the result:** `wfReturn.validation` is the Director's own re-measurement and is the
official number. Per-candidate deltas are measured against different references and do NOT sum to it.
