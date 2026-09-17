# e2e_training_workflow — Design & Build Plan

## Goal
An end-to-end **LLM SFT training-throughput** optimizer for AMD Instinct MI GPUs, built as a second
system layer over the UNCHANGED single-kernel `kernel_workflow`. Same fractal two-altitude doctrine as
`../e2e_workflow/`, but the metric, the bottleneck model, and the correctness gate differ enough that
it is a separate workflow, not a mode of the inference one.

**`../e2e_workflow/` is not edited by this project.** Anything shared is consumed read-only by
relative path, preserving the inference workflow's byte-identical guarantees.

## Locked design decisions (2026-09-16, via interactive Q&A)
1. **Scope = SFT first.** Pretraining is a near neighbour (same loop, different data pipeline).
   **RL/RLHF is out of v1** — it is a rollout(inference)+train hybrid, to be composed later from
   `e2e_workflow` + this one rather than fused into either.
2. **Target stack = bare Megatron-LM (`pretrain_gpt.py --sft`), NOT Primus.** Primus implements a
   large part of what this workflow needs (throughput logging, config overrides, profiler windows,
   steady-state windowing) — which is exactly the reason not to bring up on it. Building against
   those conveniences would couple the design to the least portable layer and the coupling would
   only surface when the second framework arrives. Bringing up on bare Megatron forces every seam to
   be real from day one. **Primus becomes the second adapter**, where it serves as a validation
   target instead of a crutch.
   - `rocm/primus:v26.4` is still the container — as an **environment**, not a dependency. The
     Megatron tree it ships (`/workspace/Primus/third_party/Megatron-LM`, ROCm-validated, TE 2.14)
     is used directly via `PYTHONPATH`; Primus's trainer/config layers are bypassed entirely.
   - Upstream Megatron has a first-class SFT path: `pretrain_gpt.py --sft` →
     `megatron/training/datasets/sft_dataset.py`, jsonl with OpenAI-format `messages`, loss mask via
     `IGNORE_INDEX = -100`.
3. **Scale = single node, 8 GPUs** (box on hand: MI308X / gfx942, ROCm 6.4.3). No multi-node
   launcher, no scheduler dependency in v1. Several v1 shortcuts depend on this: `torchrun` inherits
   the parent env, so one PYTHONPATH overlay reaches all 8 ranks; and a 50-step A/B costs minutes, so
   no proxy-fidelity machinery is needed.
4. **Payoff is two-dimensional: (throughput, memory), throughput primary.** Memory is a *currency,
   not a metric* — see "The memory rule".
5. **Correctness gate = L0 + L1** (op-level fwd+bwd numerics; fixed-seed loss/grad-norm trajectory
   match). L2 soak is opt-in / risk-triggered, not in the default path.
6. **New top-level dir `e2e_training_workflow/`**, sibling to `e2e_workflow/` and `kernel_workflow/`.
7. **The whole workflow runs INSIDE the training container.** Consequence, and it is a hard
   constraint rather than a detail: **the workflow must assume nothing about the image.** Different
   frameworks generally mean different images, so "support torchtitan/HF/verl later" means this
   workflow has to be liftable into *their* image unchanged. Its own runtime footprint stays minimal
   and self-contained, and nothing outside `scripts/adapters/` may reference a Primus path.
8. **Framework portability is a v1 design constraint, not a v2 feature.** The three
   stack-neutral seams below (config IR, self-measuring probe, capability manifest) are built now,
   even though only one adapter exists — retrofitting them after the KB has filled up with
   framework-specific strings is the expensive order.
9. **Two modes, mirroring the inference layer: `fast` and `deep`.** `fast` = fixed config, kernel
   track only, time-boxed — **implemented and debugged first**. `deep` = adds ConfigSweep (and the
   editable-kernel tail) — its seam is designed now, built later. Unlike the inference layer there is
   no third "default" mode: training's cost per gate is high enough that the useful distinction is
   just "fixed config" vs "search the config too".
10. **ConfigSweep lives in `deep` only, but its integration seam is built in v1.** Four consequences
   that change `fast` rather than merely postponing work:
   - **The memory axis is dormant in `fast`.** Memory is only spendable *through* a config change
     (raise MBS, drop a recompute level), and `fast` has none. So `fast` accepts on **throughput
     only**; `peak_memory` is measured and recorded every time (cheap now, and it is exactly the
     input the future sweep consumes) but cannot make or break an accept.
   - **The config IR ships as a descriptor** (Seam 1): pin, hash, reproduce, key the KB. Being
     *searched* is a `deep` concern.
   - **Config drift becomes a correctness issue.** Once config is an input rather than a variable,
     silently running something other than what was pinned corrupts every comparison. The probe
     verifies the realized config against the pinned IR at step 0 and fails loudly on drift.
   - **A read-only config diagnosis substitutes for the sweep** in `fast` — see below.

11. **The pinned reference workload (v1).** Chosen so Step 0 costs nothing to stand up: the
   checkpoints already exist on-box in Megatron `torch_dist` form, and the dataset needs zero
   conversion.
   - **Model ladder** — `qwen2_5_0_5b_torch_dist` for plumbing debug (seconds per iteration),
     `qwen2_5_math_7b_torch_dist` as the **v1 baseline** (dense 7B; TP1×PP1×DP8 is roomy on
     8×192GiB gfx942), `qwen3_30b_a3b_torch_dist` as the later MoE workload. Tokenizer comes from
     the sibling HF dir. Provenance checked: `release/common.pt` carries recent upstream Megatron
     args (`mtp_num_layers`, `use_megatron_fsdp`, `rl_inference_*`) and no fork markers, so these
     were converted on-box with a Megatron close to the image's.
   - **Dataset** — `HuggingFaceH4/ultrachat_200k`, a **frozen local jsonl subset** (~20k
     conversations). Native `messages` field ⇒ no conversion; multi-turn with a spread of lengths ⇒
     padding waste is real, which is what makes packing a genuine lever and makes the dataset the
     validation fixture for Seam 2. Frozen locally so data order is pinned and runs need no network.
   - **Prompt format** — `default`. The other three (`nemotron-h-aligned` — Megatron's own default —
     plus `nemotron-nano-v2`, `identity`) hard-code Nemotron chat templates and pad tokens
     (`<SPECIAL_233>`, `<unk>`) that do not exist in Qwen's vocabulary. `default` is the only branch
     that uses the tokenizer's own chat template and pad token.
     **Accepted consequence:** `default` returns early before the masking loop, so prompt tokens are
     not masked (user turns contribute loss). This does not touch the compute graph, so every
     throughput number and the L1 comparison stay valid — both legs share the setting. It only costs
     SFT semantic realism, and `useful_tokens` still excludes padding (see below).
     **The cheap upgrade, when realism is wanted:** add a `qwen` branch — same `PromptConfig` as
     `default` but without the early return, with `assistant_prefix_len=3` for Qwen's
     `<|im_start|>assistant\n`. The masking loop itself is format-agnostic, so this is ~5 lines. It
     is a *data-side* change that never touches the performance path, and it is upstreamable.
     Rejected: adopting Pai-Megatron-Patch for its SFT path — that is a third Megatron variant and
     re-introduces exactly the framework coupling decision 2 exists to avoid.
12. **`useful_tokens` reads `loss_mask`, not `labels != -100`.** Megatron builds
   `loss_mask[labels == pad] = 0` (padding, by pad-token-id) and
   `loss_mask[labels == IGNORE_INDEX] = 0` (prompts, by -100) — so `-100` alone misses padding,
   which is the bigger term. The probe reads `loss_mask.sum()` where present and falls back to
   `(labels != -100).sum()` for stacks that only produce labels (HF Trainer / TRL / ms-swift).
   Padding stays masked under `default`, so packing remains measurable as a win.

## What the environment offers, and what v1 deliberately refuses to lean on
Inspected in-image (`rocm/primus:v26.4`). Primus supplies a lot of what this workflow needs — and that
is precisely the hazard. Each row below is a place where leaning on Primus would have looked like
progress in v1 and surfaced as a coupling bug when the second framework arrived. The v1 column is what
bare Megatron forces us to build honestly.

| Capability | Primus ships | v1 on bare Megatron | Why the harder path is right |
|---|---|---|---|
| throughput reporting | `log_throughput`, `log_avg_skip_iterations` | the probe measures it (Seam 2) | Primus's `tokens/s` is nominal-token-based and wrong for SFT anyway (see the SFT trap) |
| config steering | YAML `overrides:` + `${PRIMUS_TP:1}` env interpolation | CLI args, pinned + hashed as config IR | the IR is what makes knowledge portable; a YAML-overrides habit would not transfer |
| profiler window | `profile_step_start/end` | torch profiler driven by the probe at step boundaries | probe-driven windows work on any stack, including ones with no profiler integration |
| steady-state windowing | `log_avg_skip_iterations` | probe detects warmup end from step-time stability | a measured warmup end beats a hard-coded skip count on every stack |
| per-step loss / grad_norm | logged by the trainer | probe reads them at the torch level | L1 then works identically on a stack that logs nothing |
| multi-backend abstraction | `backends/{megatron,torchtitan,megatron_bridge,maxtext}` | one adapter (megatron) | Primus's abstraction is Primus-shaped; ours has to be ours |

**Kept as a reference target, not a dependency.** Primus is the planned **second adapter** and the
cheapest way to prove Seams 1-3 hold: the same Megatron backend underneath means any divergence in the
measured numbers is a bug in our seams rather than a difference in the workload. That is a sharper test
than a framework that differs in everything at once.

**Deferred with ConfigSweep, not discarded:** the Primus-Turbo knobs
(`use_turbo_attention/gemm/grouped_gemm/rms_norm`, fp8/fp4 autocast) and `examples/offline_tune`
(hipBLASLt-bench) are a ready-made Tier-0 lever pack for later. But `/workspace/Primus-Turbo` ships
`primus_turbo/triton/` + `csrc/` — editable kernels — so it stays a **Tier-1 target in v1** even with
its config knobs out of scope. That holds regardless of which framework launches the job: the kernel
track targets shared libraries, not the trainer.

## The config diagnosis (the `fast`-mode substitute for ConfigSweep)
With the sweep deferred, the workflow goes straight to the kernel track — which in the inference
doctrine is the *expensive* lever, normally pulled only after the cheap ones. That creates one specific
failure mode worth guarding: **if the baseline step is dominated by exposed comm, bubble, or host/
dataloader stalls, no kernel win can show up end-to-end**, and the budget burns on isolated speedups
that the gate then correctly rejects.

So the Profile phase emits a **read-only verdict**: given the step decomposition, is this baseline in a
regime where kernel work can pay? When it is not, the verdict names the offending knob (e.g. "62% of
the step is exposed grad-reduce; `overlap_grad_reduce` is off") but **changes nothing** — diagnosis,
not treatment. It is cheap, it keeps the Amdahl discipline intact while the sweep is missing, and it
doubles as the requirements list for ConfigSweep when that phase is built.

## The headline metric (and the SFT trap)
`useful tokens/s/GPU` over a **steady-state step window**, at a fixed (dataset, GBS, seq len).

Every training stack reports throughput as **nominal** tokens —
`seq_length × batch_size / elapsed_time / world_size` (Primus logs exactly this; Megatron reports the
same shape via `--log-throughput` in TFLOP/s). For SFT that is wrong in both directions:
- with padding, it credits work that was never done;
- with sequence packing, real useful work per step rises while the nominal number does not move at
  all — the metric *under*-credits the single biggest SFT lever.

So the headline counts **non-pad, loss-unmasked label tokens per step**. The definition is clean and,
usefully, **portable across the whole torch ecosystem**: Megatron's `sft_dataset.py` masks with
`IGNORE_INDEX = -100`, and HF Trainer / TRL / ms-swift all use the same `-100` convention. So
`useful_tokens = (labels != -100).sum()` is one probe implementation for every stack, not one per
adapter. The stack's own reported number is kept alongside as a cross-check, never as the gate.

Secondary: `peak_memory` (allocated **and** reserved, so fragmentation is visible).
Diagnostic only: `TFLOP/s/GPU` — never a gate, because recompute policy changes its denominator.

### The invariance envelope (what may and may not change)
| Fixed (changing it = cheating) | Free (that's the search space) |
|---|---|
| dataset + sample order, global batch size, seq len | micro-batch size, grad-accum steps |
| optimizer + hyperparameters, LR schedule | TP / DP / CP / EP / SP mapping |
| **attention isolation semantics** | recompute granularity + method |
| numerics within the L0/L1 tolerance | kernel backends (turbo knobs), fusions, comm knobs, dataloader |

**Packing is the boundary case and it resolves cleanly.** Packing is only inside the envelope when
attention stays isolated across packed sub-samples (thd layout + `cu_seqlens`). Without that
isolation, packed sub-samples attend each other — that changes the mathematics, not just the
schedule, and is **outside the envelope**: it must never be accepted as a throughput win. This is the
training analogue of "speculative decoding is not an optimization" in the inference doctrine.
Primus exposes the distinction as two separate switches (`enable_packed_sequences` /
`use_packed_attention`), which is a good model for how the IR should name it.

### The memory rule (DORMANT in `fast` — see locked decision 10)
Stated here because it is the target behaviour once ConfigSweep exists (`deep`). In `fast`, with the
config fixed, there is no way to convert memory into throughput, so accepts are decided on
**throughput alone** and `peak_memory` is recorded-but-not-gating. Hook 5 of the ConfigSweep seam is
where it re-arms.

A throughput-neutral, memory-saving change is **banked headroom, not a win**. It is accepted only
when the Architect converts it — raise MBS, drop a recompute level — and the *converted* config
clears the throughput gate. Otherwise it is recorded as `banked_memory_pct` and carried so a later
change can spend it. Keeps the headline honest without discarding the lever that unlocks the next
configuration.

## Bottleneck model: critical path, not %GPU time
The inference layer ranks by `pct_gpu_time × achievable_speedup`. **That misleads in training**: a
collective fully overlapped with compute can own 30% of GPU time and be worth zero. The Profiler
splits each step, per rank, into

`compute | exposed comm (not overlapped) | pipeline bubble | straggler / sync wait`

and ranks candidates by `critical_path_share × (1 − 1/attainable)`. On single-node TP+DP the bubble
bucket is usually near-empty and exposed comm is dominated by grad-reduce / param-gather — which
`overlap_grad_reduce` / `overlap_param_gather` already address, i.e. Tier-0 config, not kernel work.
Sample at least one rank per TP group plus DP rank 0 so skew is visible instead of averaged away.

## The correctness gate (L0 + L1)
**L0 — op level.** Same oracle idea as `e2e_workflow`'s `reference_io.pt`, extended so the record
includes `grad_output` and the gate covers `out`, `dgrad` **and** `wgrad` against an fp32 reference.
Two rules: tolerances are per-op and tied to accumulation depth/precision (one global `rtol` either
hides bugs or fails healthy kernels); and the gate checks the **sign distribution** of the error, not
only its magnitude — a small systematic bias passes every magnitude check and still destroys
convergence over thousands of steps.

**L1 — trajectory level. REDESIGNED after measurement (2026-09-17); the original sketch was the
wrong instrument.** The plan said "fixed seed, fixed data order, compare loss[i] and grad_norm[i]
over ~100 steps". Measured on two runs at identical config and identical data (the useful-token
sequences match step for step, so data order IS deterministic):

| step | loss delta | grad_norm delta |
|---|---|---|
| 1 | **0.0 — bit-identical** | **0.87%** |
| 5 | 0.76% | 22.7% |
| 12 | 4.40% | 40.2% |
| 21 | **2.04 absolute** (on a loss of ~3) | — |

Training is a chaotic system: identical runs diverge, and the divergence grows fast. Two consequences:

1. **A scalar aggregate over a long window is useless.** Calibrating `max |dloss|` across 25 steps
   produced a limit of **6.1 on a loss of 3** — a gate that passes any bug. The band must therefore be
   a **function of step index**, comparing the candidate's drift at step i against how far a baseline
   pair had drifted *by that same step*.
2. **Step 1 is special and precious.** It is a pure forward on identical weights, so its loss is
   bit-reproducible — one zero-noise probe of the forward path. `grad_norm` at step 1 is **not**
   reproducible (0.87%: backward atomics / reduction order), which is exactly why backward
   nondeterminism has to be calibrated rather than assumed away.

So L1 runs three independent detectors (`scripts/trajectory_gate.py`):
- **step-1 forward probe** — zero natural noise, the sharpest single check available;
- **per-step envelope** — `|cand - ref|` at step i vs `safety x` the baseline pair's disagreement at
  step i, failing on a *pattern* of exceedances rather than one isolated chaotic step;
- **early-window sign balance** — a one-sided delta is a systematic bias even when every magnitude
  check passes (same doctrine as the L0 sign rule).

Validated against a synthetic candidate carrying a +0.3% systematic loss bias — the signature of a
kernel that is within tolerance on every op and still walks the run off course. All three detectors
fire (step-1 0.30% vs 0.05% limit; envelope exceeded on 5 steps at up to 6.0x; sign balance 1.0/8),
while the real baseline pair passes.

**Known gap, not yet closed:** the step-1 floor (0.05% relative) is a *policy guess*, not a
calibration — our baseline pair runs identical kernels, so its measured step-1 delta is exactly 0 and
provides no floor. A legitimate kernel swap perturbs rounding and will move step-1 loss by some
amount; until that is calibrated (by pairing two known-equivalent kernel backends), the floor risks
false rejects. Calibrating it needs a baseline pair that differs in kernels but is known-correct.

**The band is measured, never assumed.** Training is natively nondeterministic (reduction order,
backward atomics, RCCL algorithm selection), so Setup runs the baseline **twice** and derives the
tolerance band from the observed run-to-run spread — same doctrine as the inference noise band.
`tests/trainer/test_megatron_trainer_deterministic.yaml` gives a starting point for how tight that
band can be made. L1 also catches the distributed-specific failure L0 structurally cannot: an overlay
that engaged on only some ranks (a small trajectory offset, not a crash).

**L2 — soak.** Opt-in, plus auto-triggered for high-risk classes: anything touching backward, the
optimizer, a reduction order, or precision/scaling. Not in the default per-candidate path.

## Levers
### Tier 0 — config only — **`deep` mode only (locked decisions 9-10)**
Not built in `fast`; kept here as the lever inventory ConfigSweep will consume, and as the vocabulary the
config diagnosis uses when it names an offending knob. Ordered by expected ROI:
1. **Kernel-backend knobs** — on Primus, `use_turbo_attention` / `use_turbo_gemm` /
   `use_turbo_grouped_gemm` / `use_turbo_rms_norm` / fp8 paths; on bare Megatron the TE/FA backend
   selection. Highest ROI per unit cost; a pure config sweep.
2. **Recompute** — granularity (selective|full), method (uniform|block), num layers. Usually the
   biggest SFT knob; trades directly against the memory axis.
3. **Parallel mapping** — TP × DP (× CP for long seq, EP for MoE), sequence parallel.
4. **Comm** — `overlap_grad_reduce`, `overlap_param_gather`, `use_distributed_optimizer`, bucket
   sizes, RCCL env.
5. **SFT data path** — packing (with attention isolation), dataloader workers. Host-side stalls are
   real and invisible in a GPU-only profile.
6. **Fusions** — grad-accum fusion, rope / masked-softmax / swiglu, MoE permute.
7. **Precision** — bf16 vs fp8 (TE 2.14). A numerics change ⇒ an L1 customer.
8. **MBS × grad-accum** at fixed GBS — the memory↔throughput conversion knob.
9. **Offline GEMM tuning** — hipBLASLt-bench, for the exact SFT shapes.

### Tier 1 — editable hot kernels (→ recursive `kernel_workflow`) — **the v1 track**
Source trees available in-image: TE, aiter, and `/workspace/Primus-Turbo`
(`primus_turbo/triton/`, `csrc/`). Extract with real shapes **and** `grad_output`; the task objective
is **fwd+bwd combined** latency. Optimizing forward alone is how you ship a kernel 2× faster forward
and 3× slower backward. Integration rebinds a `torch.autograd.Function`; engagement must be proven
**separately for forward and backward**. This track is framework-agnostic by construction — it
targets libraries the trainers share.

### Tier 2 — overlap / scheduling / host
Kill exposed comm, fix launch-bound steps, native layouts. Partly in v1 (host/launch work is a kernel
-layer concern); the comm-overlap half needs config and therefore waits.

## Roles (system layer)
| role | phases | owns |
|---|---|---|
| Training Director | setup, validate | isolated eval dir, TRUE baseline (throughput + memory + 2× trajectory for band calibration), final independent re-measure |
| System Architect | strategize, plan, update_experience, report | critical-path routing, budget, stop rule, persistent `framework_playbook.md` |
| Profiler | baseline, reprofile | multi-rank trace → step decomposition + Top-N with shapes + the **read-only config diagnosis** |
| Kernel Extractor | extract | fwd+bwd shapes + oracle → immutable standalone task dir |
| Convergence Auditor | gate | L0 + L1; owns the measured band and the risk classification that triggers L2 |
| Integrator | integrate, finalize | reversible overlay, throughput gate (memory recorded), deliverable bundle |
| (kernel squad) | — | UNCHANGED `../kernel_workflow/kernel_workflow.js`, recursive |
| ~~Config Tuner~~ | ~~sweep~~ | **`deep` mode only**; its scope is the Tier-0 inventory above, entering via the ConfigSweep seam |

The Convergence Auditor is the one genuinely new role vs the inference layer — the accept/reject
decision now needs evidence no other phase naturally owns.

## Modes: fast · deep
Mirrors `e2e_workflow`'s mode mechanics (`fast_mode` / `deep_mode` args, mutually exclusive, deep wins
if both are set), with one structural difference forced by the hardware — see "GPU exclusivity".

| | `fast` (v1, built first) | `deep` (seam now, built later) |
|---|---|---|
| config | **fixed**, pinned + hashed as IR | ConfigSweep searches it |
| tracks | Tier-1 head kernels by critical-path share | + ConfigSweep, + the editable-kernel tail |
| kernel depth | 1 pass per head, ≤2 authored languages | multi-round, cross-backend lanes, reseeding |
| gate | throughput only (memory recorded) | throughput + the memory rule re-armed |
| budget | wall-clock **and** gate count (see cost model) | larger wall-clock, more gates |
| answers | "best kernel wins at this config in N hours" | "best achievable number on this box" |

**Mode-knob discipline, inherited verbatim from the inference layer:** every `deep`-only knob is
selected by a `DEEP_MODE ? deep : fast` ternary and every `deep`-only phase is gated by `want(...)`,
so **adding `deep` later must leave a `fast` run byte-identical** — same prompts, same budgets, same
control flow. That is the property that makes it safe to ship `fast` first and grow into `deep`.

### GPU exclusivity (the one thing that does NOT carry over from inference)
Inference `fast` fans kernel lanes out across spare GPUs *while* the e2e gate runs on the serving
slot: the server occupies TP GPUs, the lanes take the rest. **Training has no spare GPUs** — a
TP1×DP8 job occupies all 8. So the concurrency model is **phase-exclusive**:

- while kernel lanes run, no training job is needed → lanes may use **all 8 GPUs** in parallel;
- a gate needs **all 8** → every lane must be drained first.

This replaces the inference layer's `makeSem`-style "lanes + serving slot coexist" with a **barrier**:
`drain lanes → run gate → resume lanes`. The rejected alternative — reserve 6 GPUs for the job and 2
for lanes — fails on measurement basis, not on performance: the parallel mapping is part of the
workload identity, so a DP6 measurement is not a number about the DP8 configuration anyone asked
about.

### Gate cost model (why `fast` counts gates, not only hours)
One training A/B = **two job launches** (checkpoint load + warmup + N steady-state steps), and it
blocks every lane while it runs. That makes gates, not agent time, the scarce resource — the opposite
of the inference layer, where the gate is comparatively cheap. Two consequences:
- `fast` is bounded by **both** `fast_budget_ms` and a gate count (`fast_max_gates`).
- A two-tier gate ladder keeps the count down: a **screen** reuses a recent reference leg (with a
  periodic drift re-measure), while a **confirm** runs a fully paired same-session A/B. Only a
  confirm can accept. This is the training analogue of the inference layer's `validation_level`
  ladder (`integrator` → `config_sweep` → `final_pair`).

## Pipeline
```
# fast (v1)
Setup(preflight + PIN config IR + TRUE baseline: useful-tok/s, peak mem, 2x trajectory -> band)
  -> Profile(multi-rank -> step decomposition + Top-N + READ-ONLY config diagnosis)
  -> Strategize(critical-path routing)
  -> LOOP[ lanes: Extract(fwd+bwd) -> recursive kernel_workflow -> L0        (parallel, 8 GPUs)
           barrier: drain lanes -> Overlay -> L1 -> throughput gate           (exclusive, 8 GPUs) ]
  -> Re-profile -> Finalize(overlay bundle + optional L2) -> Report -> Director Validation

# deep (later) — additive, inserted at the marked points
  ... Strategize -> [+ ConfigSweep -> Re-profile -> Re-strategize] -> LOOP[...] -> [+ tail loop] ...
```

## The ConfigSweep seam (what `fast` must already provide)
`deep` must be additive, not a rewrite. Six hooks are therefore built in v1 even though nothing
exercises them, each with a trivial `fast` behaviour:

| # | Hook | `fast` behaviour | What `deep` does with it |
|---|---|---|---|
| 1 | `curConfig` — a carried config-IR value threaded through every measurement | stays == the pinned baseline | mutates as sweeps are accepted |
| 2 | every bench / profile / gate call takes config as an **explicit argument** (never reads a pinned global) | always passed the same value | passed the evolving stack |
| 3 | `reprofile()` + `restrategize()` exist as independently callable units | called once after the kernel loop | called after every accepted config win — a config change moves which kernels dominate |
| 4 | the config diagnosis emits a **ranked list of offending knobs** | reported, then ignored | becomes ConfigSweep's input queue (`config_directions`, exactly as in the inference layer) |
| 5 | `proposeMemoryConversion()` — the memory rule's converter | no-op; records `banked_memory_pct` | emits a config delta that spends banked memory, then re-gates |
| 6 | KB identity includes the **config IR hash** | one config, so one bucket | records from different configs never silently mix |

Hooks 1-3 are the ones that are expensive to retrofit: threading a config value through a measurement
path after the fact touches every call site, and that is precisely where a silently-stale reference
leg would hide.

## Step 0 results (2026-09-17, measured — not projected)
A bare-Megatron SFT baseline runs. `pretrain_gpt.py --sft`, PYTHONPATH at
`/workspace/Primus/third_party/Megatron-LM`, inside `rocm/primus:v26.4`, on 8×MI308X (gfx942).
Artifacts: `/home/treemann/geak_train/{scripts,data,runs}`.

### Three things that changed the design
1. **Upstream Megatron does not run on ROCm unpatched.** `_compile_dependencies()` →
   `megatron.legacy.fused_kernels.load()` → `_get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)`;
   `CUDA_HOME` is None on ROCm → `None + "/bin/nvcc"`. Primus no-ops the whole function
   ("CUDA kernels are not compatible with ROCm", `disable_compile_dependencies: true`). So "bare
   Megatron" is really "upstream + a small set of ROCm compatibility patches". The set is small and
   well-bounded (1 of Primus's 23 registered patches is of this kind), and those patches belong in
   *our* overlay, not in a Primus dependency.
2. **Upstream `--sft` THD packing is self-inconsistent on this commit.** `SFTDataset` returns a
   per-sample 1D `cu_seqlens`; the default collate stacks it to `[MBS, n+1]`;
   `get_thd_batch_on_this_cp_rank` forwards that 2D tensor into `PackedSeqParams`, and Megatron's own
   rope helpers then assume 1D (`cu_seqlens[-1]`) — **both** the fused and the non-fused path crash.
   Not a ROCm issue; Primus never hits it because it ships its own SFT dataset. v1 squeezes the
   vacuous batch dim at MBS=1 and **refuses loudly** for MBS>1 (merging rows needs offset re-basing
   plus flattening tokens into one sequence — a semantics change, not a workaround).
3. **Upstream logs no tokens/s at all** (only `TFLOP/s/GPU`; the tokens/s line is a Primus addition).
   Seam 2 was justified as portability insurance; it turns out to be required on framework #1.

### The measurement finding that sets gate sensitivity
7B, TP1×DP8, seq 4096, GBS 64, MBS 1, bf16, 30 iterations:

| steady-state window | n | median (ms) | spread | stdev |
|---|---|---|---|---|
| skip 2 (the common default) | 28 | 11648.9 | **12.75%** | 329.7 |
| skip 4 | 26 | 11648.9 | 12.70% | 311.4 |
| **skip 10 (measured warmup end)** | 20 | 11647.5 | **0.18%** | 4.7 |
| skip 15 | 15 | 11647.4 | 0.14% | 4.4 |

The **median is unchanged** across windows (11648.9 → 11647.5) while the **spread differs 70×**.
Skipping the conventional 2 warmup steps yields a 12.7% noise band, under which no kernel win below
~12% is detectable; detecting the real warmup end (iteration ~11) yields 0.18%, where sub-1% deltas
are measurable. Iterations 3-10 sit +3.5% to +12.6% above steady state — autotune / allocator / comm
still settling, not noise. **This is why the probe derives the warmup end from step-time stability
instead of taking a hard-coded skip count** (Seam 2), and it is measured evidence, not a principle.

Reference numbers: 126.3 TFLOP/s/GPU; nominal 2812 tokens/s/GPU (padding included — the useful-token
figure awaits the probe); peak memory 78 GB reserved of 192 GB, i.e. a large MBS/recompute headroom
for the future ConfigSweep to spend; 0 skipped and 0 NaN iterations.

### Probe + dispatcher results (measured)
`scripts/{train_probe.py,probe_summarize.py,bench_train.sh,gpu_occupancy.py}` +
`scripts/adapters/megatron.sh` + `workloads/*.env` now produce a canonical `bench_summary.json`.

**Noise bands, measured.** 0.5B / 2 GPUs / 25 steps, two independent jobs at identical config:

| quantity | run A | run B | delta |
|---|---|---|---|
| `useful_tok_s_per_gpu` | 4753.8 | 4759.9 | **0.128%** |
| `step_median_ms` | 979.55 | 978.25 | 0.133% |
| outlier steps | 2 | 0 | sporadic |

So `across_job` ≈ **0.13%** and `within_job` (IQR) ≈ **0.07-0.13%** — an order of magnitude tighter
than the inference layer's 2-3% band. Sub-1% kernel deltas are therefore gateable, **provided** the
dispersion is reported the right way.

**Report spread AND robust spread; never collapse them.** The same 25-step run reports 38.6% raw
spread and 0.072% IQR spread: 22 of 24 steady steps sit inside 0.07%, and two isolated stalls
(+38.5%, +17.4%) carry the rest. Using raw spread as the noise band would make the gate useless;
hiding the stalls would discard a real host-side signal (they appear in one job and not the next).
`dispersion()` emits `spread_pct`, `iqr_spread_pct`, `outlier_steps` and `outlier_max_pct` together.

**Warmup length is not a constant.** The 7B/8-GPU run settles after ~10 steps; this 0.5B run settles
after 1 (dataset/tokenizer caches already warm). A fixed skip count is wrong in both directions,
which is why `detect_warmup_end()` derives it from the series — and why warmup detection must NOT
absorb later stalls (they are a different phenomenon, so the reference level is the back-half median
and the boundary needs `patience` consecutive in-band steps).

**The SFT trap, quantified.** `pad_fraction_pct = 71.6%`: nominal 16748 vs useful 4760
tokens/s/GPU — the stack-conventional number overstates useful work by **3.5x** on this workload.
Optimizing against it would mean optimizing a number that is 71% padding. It also says the data path
(packing) is a large Tier-0 lever, which `fast` cannot pull — so the read-only config diagnosis must
report it.

**Rank skew is ~0.002-0.013%**, so the straggler bucket is empty at this scale — worth re-checking at
8 GPUs and with TP>1, where it is the input to the critical-path model.

### New risk: the box is shared
~20 unrelated containers are up on this node (idle at the time of measurement, but not ours). The
inference layer gets isolation from "same-session sequential A/B"; training needs an explicit
**occupancy check around every leg**, so contention becomes a detected fact rather than a silent bias
in a reference leg.

## Framework portability — the three stack-neutral seams
A framework's conveniences (YAML config overrides, built-in `tokens/s` and per-step `grad_norm`
logging, step-ranged profiler hooks, vendor kernel knobs) are **exactly the parts other frameworks
least agree on**. Designing against them pours the foundation on the least portable ground — which is
why v1 brings up on bare Megatron and treats Primus as the second adapter (locked decision 2).
Three seams keep the core stack-neutral; a fourth observation explains why the cost is bounded.

### Seam 1 — a canonical config IR (the biggest structural change)
Without it, the Config Tuner learns `recompute_granularity` / `use_turbo_attention` literals and the
KB fills with Megatron/Primus strings — and the cross-run knowledge, which is this system's most
valuable asset, becomes non-transferable. So the core speaks only a **stack-neutral knob vocabulary**
(`micro_batch`, `recompute.granularity`, `comm.overlap_grad_reduce`, `data.packing`,
`attn.backend`, `precision.format`, …). The adapter maps IR ↔ native form, and **"unsupported" is a
first-class return value** — never an error, never a silent no-op (the same "never accept a silent
degrade" discipline the inference roles already enforce). The playbook then records "selective
recompute is worth X% on this model class", not a field name.

### Seam 2 — measurement is self-evidencing, not log-parsed
Some stacks (Primus) log everything per step, which makes a log parser look sufficient. It is a trap:
another framework may log `grad_norm` rarely or never, may not report peak memory, and **its numbers
are defined differently, so cross-framework results are not comparable**. Every new framework would
also need a new parser.

Instead the primary instrument is a **framework-agnostic torch-level probe** — step boundaries,
useful (non-pad, unmasked) tokens, peak memory, grad norm, all captured at the torch level. The
mechanism already exists: the probe *is* a PYTHONPATH overlay (`overlay_setup.py`), the same
mechanism used for kernel reintegration. The framework's own log degrades to a **cross-check**, and a
disagreement between the two is a signal worth surfacing rather than averaging away (same doctrine as
the inference layer's two profile orderings).

This also settles the useful-token counter: it belongs in the probe as `(labels != -100).sum()`, a
definition Megatron, HF Trainer, TRL and ms-swift all share — one implementation, every stack, no
framework patch.

### Seam 3 — a capability manifest; route on capability, never on framework name
Each adapter declares what it supports; orchestration routes and **degrades** on that, so the core
contains no `if framework == "primus"`. Same pattern as the inference layer's `env_report.json`.

| Primus capability | Degradation path when a stack lacks it |
|---|---|
| step-ranged profiler window | fall back to a time-ranged window (the inference layer's approach) |
| built-in `tokens/s` + warmup-skip | the probe measures it anyway — it was the primary all along |
| per-step `grad_norm` | probe reads it off the optimizer; failing that L1 degrades to loss-only **and records that the gate got weaker** |
| YAML `overrides:` + env interpolation | fall back to CLI overrides; failing that, a *generated derived recipe* — never an in-place edit of the user's recipe |
| a deterministic reference config | band comes purely from the baseline-self-repeat measurement (which is the rule anyway) |
| multi-backend abstraction | one adapter per framework — this is the only place real code is owed |

### Why the total cost stays bounded
Portability cost is not uniform across the three tracks:
- **Tier-1 kernel track is inherently portable.** It targets libraries the frameworks *share* (TE,
  aiter, Primus-Turbo, FA), rebinds a `torch.autograd.Function`, and ships as a PYTHONPATH overlay.
  None of that sees the framework.
- **Config track is the least portable** — hence Seam 1 and the lever packs below.
- **Measurement track is portable or not depending on this one decision** — hence Seam 2.

### Lever packs (keep stack-specific levers out of the core)
Primus-Turbo's knobs are stack-specific and must not become a core phase. They live in
`levers/<stack>.yaml`, each entry declaring: `native knob → IR name → risk class (numerics-changing?
memory-affecting?) → required gate level (L0/L1/L2)`. Adding a framework = one adapter + one lever
pack, nothing else. Because risk class is attached to the IR name, knowledge like "fp8 requires L1"
transfers across stacks for free.

## Adapter contract (`scripts/adapters/<stack>.sh`, sourced by `scripts/bench_train.sh`)
```
adapter_capabilities  # declare supported features (Seam 3) — consulted before anything else
adapter_launch        # start the job (torchrun, single node, in-container)
adapter_wait_steady   # block past warmup/JIT/capture steps
adapter_measure       # N steady-state steps; the PROBE supplies the numbers, not the log
adapter_metrics       # cross-check only: the stack's own loss / grad_norm / throughput reporting
adapter_profile_window# step-ranged window where supported, else time-ranged
adapter_apply_cfg     # canonical config IR delta -> this stack's config form
adapter_teardown      # kill ALL ranks; verify no residual GPU memory / stale RCCL state
```
`adapter_apply_cfg` is the addition vs the inference contract: inference normalized everything to
"flags + env"; training stacks disagree on config form (Primus YAML overrides vs DeepSpeed JSON vs
torchtitan TOML). For Primus it is a generated `overrides:` layer + `PRIMUS_*` env. Whatever the
form, the rule is constant: **derive a new artifact, never edit the caller's recipe in place.**

`adapter_teardown` is harder than the inference `server_teardown.sh` because of the training-specific
failure mode: a **hung collective** (one rank slow or dead, the rest silently blocked). Detection is a
heartbeat on step progress — supplied by the probe, so it is stack-neutral — not a process liveness
check, since every rank is alive.

## Measurement discipline
- Steady state only: drop the first N steps (JIT/autotune, allocator warmup, RCCL bringup). Exclude
  periodic events (checkpoint save, eval) from the timed window. The probe derives the warmup end
  from step-time stability rather than a hard-coded skip count (stacks that ship one, e.g. Primus's
  `log_avg_skip_iterations`, are cross-checked against the measured value, not trusted).
- Prefer **in-job interleaved A/B** where the stack allows it — cheaper and tighter than the inference
  layer's sequential same-session A/B, since a step is seconds, not minutes.
- Two noise bands, reported separately: `within_job` and `across_job`. Never ratio across bands.
- Every candidate reports `(tput_delta, peak_mem_delta)` plus its L0/L1 verdicts. Accept on throughput
  above band + L0 + L1; memory-only results are banked, not accepted.
- OOM is data: record the feasibility boundary — it is what makes the memory currency spendable.

## Reuse from `e2e_workflow` (read-only, by relative path)
- `scripts/overlay_setup.py` — reversible PYTHONPATH overlay. Works unmodified single-node: torchrun
  inherits the parent env, so one overlay reaches all 8 ranks. (Multi-node later needs per-node
  materialization — deliberately deferred.)
- `scripts/e2e_store.py` — the KB. New identity keys: workload identity becomes
  `(model, GBS, MBS, seqlen/packing, TP/DP/CP/EP, recompute, precision, #gpus, turbo knobs)`.
  Records are NOT comparable across parallel mappings and the store must say so rather than silently
  rank them.
- Doctrine assets: measured noise bands, engagement proof, reversible overlay, phase-by-phase
  checkpointing (`phases=` + validation checkpoints), Amdahl-style routing (re-based on critical path).
- NOT reused: `bench_e2e.sh` (server lifecycle), the output-parity gate, the vendored tuning skillset
  (its tuners are inference-op specific).

## Build order
- [ ] **Step 0 (manual, before any code): a runnable bare-Megatron SFT baseline.**
      `torchrun --nproc_per_node 8 pretrain_gpt.py --sft ...` with `PYTHONPATH` at
      `/workspace/Primus/third_party/Megatron-LM`, a pinned checkpoint + jsonl chat dataset, running
      stably for ~100 steps on gfx942. Nothing downstream means anything until this exists, and it
      must be done by hand — the workflow should not be discovering whether the recipe runs at all.
- [ ] `scripts/train_probe.py` — the stack-neutral measurement probe (Seam 2), shipped as an overlay:
      step boundaries, useful tokens (`labels != -100`), peak memory, grad norm, progress heartbeat,
      pinned-config verification. **Built first among the code** — everything downstream consumes its
      numbers, and it is what makes the second framework cheap.
- [ ] `scripts/config_ir.py` — the canonical knob vocabulary + IR↔native mapping (Seam 1). v1 scope
      is descriptor-only: pin, hash, reproduce, key the KB.
- [ ] `scripts/bench_train.sh` + `adapters/megatron.sh` — steady-state measurement on top of the
      probe. Validate standalone against the Step-0 baseline (reproduce its tokens/s) before anything
      is built on it.
- [ ] `scripts/trajectory_gate.py` — L1, including baseline-self-repeat band calibration.
- [ ] `scripts/parse_train_profile.py` — step decomposition + Top-N + the read-only config diagnosis.
- [ ] `scripts/capture_train_shapes.py` — fwd+bwd shape + oracle capture (L0 input).
- [ ] `knowledge/` — training_optimization, sft_measurement, critical_path, gate_convergence,
      megatron_internals, framework_playbook (persistent, grown per run).
- [ ] `roles/` — the 6 active roles above.
- [ ] `training_workflow.js` — orchestration + recursive kernel-layer dispatch: `fast`/`deep` mode
      args, the **lane/gate barrier** (drain → gate → resume), the gate-count budget, the two-tier
      gate ladder, and the six ConfigSweep seam hooks wired with their `fast` no-op behaviour.
- [ ] `interface/run_training.py` — external contract, mirroring `interface/run_e2e.py`.
- [ ] **Debug the whole `fast` flow end to end on the Step-0 baseline** — the actual v1 exit
      criterion: one complete run that produces a gated, L0+L1-verified kernel result and a
      deliverable bundle. Everything above is a means to this.
- [ ] *(deep)* ConfigSweep + Config Tuner role + `levers/<stack>.yaml`, entering through the six
      seam hooks; `adapters/primus.sh` as the second adapter.

## Open questions (not yet locked)
1. **The SFT recipe itself** — the one blocking item. Needs: model + pretrained checkpoint path, a
   jsonl chat dataset (`messages` format), seq len, and the parallel mapping to pin
   (suggest llama3-8B / bf16 / seq 4096 / TP1×DP8 on the 8×192GiB gfx942 box). Full SFT, **not
   LoRA** — LoRA freezes the base weights, which removes most wgrad and shrinks the optimizer, i.e.
   it under-exercises exactly the tracks being built.
2. **`--sft-tokenizer-prompt-format`** — Megatron defaults to `nemotron-h-aligned`; which format
   matches the chosen model/dataset? A wrong prompt format still trains, it just trains on the wrong
   thing, and it would silently poison the L1 reference trajectory.
3. **fp8 (TE 2.14) on gfx942** — relevant only once ConfigSweep exists; bf16 for v1.
4. **Pipeline parallel on a single node** — include later, or exclude to keep the bubble bucket empty?
5. **Second adapter timing** — Primus first (same Megatron backend underneath ⇒ any divergence is a
   seam bug, the sharpest possible test), then ms-swift (`ms-swift-rocm` image is already on the box)
   as the genuine cross-ecosystem check. How early is a schedule call.

*(Resolved: process boundary → in-container, locked decision 7. Useful-token counter → the probe,
Seam 2, framework-neutral via the `-100` convention. Bring-up stack → bare Megatron, locked decision
2. ConfigSweep → deferred, locked decision 9.)*
