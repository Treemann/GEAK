# e2e_training_workflow — End-to-End LLM SFT Training-Throughput Optimizer (AMD Instinct MI GPUs)

A second system layer over the **UNCHANGED** single-kernel `kernel_workflow`, aimed at **SFT training
throughput** instead of serving throughput. Same fractal two-altitude doctrine as `../e2e_workflow/`,
but the metric, the bottleneck model and the correctness gate are different enough that it is a
separate workflow rather than a mode of the inference one.

`../e2e_workflow/` is **not edited** by this project; anything shared is consumed read-only.

## Why it is not just "e2e_workflow with a training adapter"

| | inference (`e2e_workflow`) | training (this) |
|---|---|---|
| headline | serving tok/s | **useful** (non-pad, loss-unmasked) tok/s/GPU |
| bottleneck model | `pct_gpu_time x achievable_speedup` | **critical-path share** — a fully overlapped collective is worth zero |
| correctness | byte-identical greedy output | **L0** (fwd+bwd numerics) + **L1** (trajectory envelope) |
| concurrency | lanes and the serving slot coexist | **a gate needs the whole box** — lanes drain first |
| scarce resource | agent time | **gates** (one A/B = two job launches, exclusive) |
| noise band | 2-3% | **measured 0.12-0.13%** |

**`DECISIONS.md` records every framework-level choice with the measurement that forced it.**
`PLAN.md` carries the narrative; `RUNBOOK.md` the commands.

## Measured facts this design rests on
All from 8x MI308X (gfx942), `rocm/primus:v26.4`, bare Megatron `pretrain_gpt.py --sft`.

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

## Layout
```
training_workflow.js          orchestration: fast/deep modes, lane/gate barrier, gate arithmetic,
                              the ConfigSweep seam, recursive dispatch into ../kernel_workflow/
roles/                        6 agent prompts (director, architect, profiler, extractor,
                              convergence auditor, integrator)
knowledge/                    the doctrine, carrying measured numbers rather than advice
scripts/bench_train.sh        stack-neutral dispatcher: lifecycle, occupancy guard, hang watchdog
scripts/adapters/megatron.sh  the ONLY framework-specific layer (+ capability manifest)
scripts/train_probe.py        torch-level probe: steps, useful tokens, memory, grad norm, profiler
scripts/probe_summarize.py    policy: warmup detection, steady window, dispersion, config digest
scripts/parse_train_profile.py  step decomposition + Top-N + the read-only config diagnosis
scripts/trajectory_gate.py    L1: calibrate a step-indexed envelope, then judge against it
scripts/capture_train_shapes.py  fwd+bwd capture -> immutable L0 oracle
scripts/overlay_train.py      reversible rebind + per-half engagement proof
scripts/gpu_occupancy.py      isolation guard for a shared node
scripts/rocm_compat.py        ROCm/upstream patches + probe/capture install point
workloads/*.env               pinned workload descriptors (config IR, descriptor role)
```

## Modes
`fast` (default) = fixed config, kernel track only, bounded by wall clock **and** gate count.
`deep` = adds ConfigSweep; **not implemented**, and it refuses to run rather than reporting a
fast-mode result under a deep label. Its six integration hooks already exist in fast mode, so it is
additive rather than a rewrite.

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

# tests (no GPU)
node scripts/tests/test_training_gate.js
python3 scripts/tests/test_parse_train_profile.py
```

## Status
Measurement, profiling, extraction, the overlay, both gates, and the deterministic orchestration are
implemented and validated on real runs — including an identity-rebind overlay that proves forward and
backward engagement separately at 0.109% throughput cost (inside the measured band). The agent phases are wired to the role prompts but have not been executed
end to end: that needs the Workflow/agent runtime, not `node`. `PLAN.md` records what is measured,
what is assumed, and the open gaps — including the one in L1's step-1 floor.

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
    workload:     "<REPO>/e2e_training_workflow/workloads/qwen2_5_math_7b.env",  // REQUIRED, no default
    nproc: 8, gpu_ids: "0,1,2,3,4,5,6,7",
    iters: 30,             // steady-state steps per leg (warmup is detected, not assumed)
    budget: 2,             // kernel tasks for a first run
    max_gates: 4,          // ONE A/B = two job launches and it blocks every lane
    kernel_budget: 3,      // passed down to each recursive kernel_workflow run
    time_budget_ms: 18000000
  }
})
```
Start with `workloads/qwen2_5_0_5b.env`, `nproc: 2`, `budget: 1` to shake the orchestration out
cheaply — a 0.5B leg is ~1 s/step against ~11.6 s at 7B.

**What it will do, and where it will stop**
* Setup measures the TRUE baseline **twice** and derives this run's bands from those two legs.
* Profile emits the step decomposition plus a **read-only** config diagnosis. On this workload it
  returns `config_first` (70.6% padding), and `fast` mode cannot fix that — it says so and proceeds.
* The kernel loop runs lanes on the GPU pool, then **drains them** for each gate: a training A/B
  needs every GPU.
* `deep_mode: true` **refuses to run** — ConfigSweep is not implemented, and reporting a fast-mode
  result under a deep label would be worse than failing.
* A leg contaminated by a foreign GPU process is re-measured (bounded); exhausting the retries stops
  the run rather than quietly spending the remaining gates on a busy node.

**Reading the result:** `wfReturn.validation` is the Director's own re-measurement and is the
official number. Per-candidate deltas are measured against different references and do NOT sum to it.
