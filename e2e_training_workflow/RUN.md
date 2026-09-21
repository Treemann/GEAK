# Launching a run

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
    kernel_budget: 2
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
    time_budget_ms: 18000000
  }
})
```

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
* `deep_mode: true` refuses to run. ConfigSweep does not exist yet.

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
