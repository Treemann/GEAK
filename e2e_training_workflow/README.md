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

## Measured facts this design rests on
All from 8x MI308X (gfx942), `rocm/primus:v26.4`, bare Megatron `pretrain_gpt.py --sft`.

- **71.6% of tokens are padding** on the reference workload: the stack-conventional nominal tokens/s
  overstates useful work by **3.5x**. Hence the headline counts `loss_mask.sum()`.
- **Warmup detection is worth 70x in gate sensitivity.** Same run: skip-2 gives a 12.75% spread,
  skipping to the measured warmup end gives 0.18% — with an unchanged median.
- **Raw spread vs robust spread differ 500x** on a clean run (38.6% vs 0.072%): two isolated stalls
  carry all of it. Both are always reported; the band uses the robust one.
- **Across-job reproducibility is 0.118% (7B/8GPU) and 0.128% (0.5B/2GPU)** — so sub-1% kernel wins
  are gateable, and the band is derived per-pair rather than assumed.
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
