# Launching a run

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

## The 7B run's unexplored heads (Stage 2 stopped on the wall clock)

Stage 2 gated 1 of 4 queued heads and stopped at `3.38h of a 3.00h lane deadline`. The Architect's
own numbers, with expected gain computed the way the orchestrator does it
(`cp_share × (1 − 1/attainable)` — validated on head 1: predicted 3.94%, measured **+4.54%**):

| # | head | cp_share | attainable | expected gain | status |
|---|---|---|---|---|---|
| 1 | `lmhead_ce_unmasked_gather` | 6.41% | 2.6× | 3.94% | **accepted, +4.54%** |
| 2 | `attn_fmha_fwdbwd_gqa_d128_causal` | **18.2%** | 1.25× | **3.64%** | not attempted |
| 3 | `swiglu_fwd_bwd_fused` | 3.47% | 1.5× | 1.16% | not attempted |
| 4 | `rope_fused_fwd_bwd` | 1.89% | 2.5× | 1.13% | not attempted |

Two things make this list more actionable than it looks. Head 2 is the largest single item on the
critical path by a wide margin — but note the 0.5B run rejected an attention candidate on L1
(step-1 loss 0.4274% against a 0.05% overlay limit), so budget for a rejection there. Heads 3 and 4
are worth ~1.1% each, which under the **old** band (0.914% at 7B) would have been marginal-to-
ungateable; under the recalibrated band (floor 0.15% for this workload, `2×R_cal` measured at
0.259%) they are comfortably gateable. The band fix is what makes the tail of this list worth running.

A follow-up run now records this itself: `wfReturn.unexplored_heads` and `wfReturn.stop_reason`.

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
