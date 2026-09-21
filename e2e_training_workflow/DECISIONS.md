# Design decisions and the evidence for each

Every framework-level choice in this workflow, what it replaced, and the measurement that forced it.
Nothing here is reasoning from first principles alone: each row names a number and where it came
from. Measurements are on 8×MI308X (gfx942), `rocm/primus:v26.4`, bare Megatron `pretrain_gpt.py
--sft`, Qwen2.5 0.5B/2GPU and Qwen2.5-Math 7B/8GPU, seq 4096.

`PLAN.md` carries the narrative; this file is the index. Raw artifacts live in
`/home/treemann/geak_train/runs/` (probe sidecars per rank, one directory per run).

---

## 1. The measurement contract

| # | Decision | What it replaced | Evidence |
|---|---|---|---|
| 1.1 | Headline is **useful** (non-pad, loss-unmasked) tokens/s/GPU, read from `loss_mask.sum()` | the stack's nominal `seq_len × batch / time / world` | **71.6%** of tokens are padding: nominal 16748 vs useful 4760 tok/s/GPU, a **3.5×** overstatement. Upstream Megatron reports no tokens/s at all. |
| 1.2 | Warmup end is **detected from step-time stability**, not a fixed skip count | `log_avg_skip_iterations`-style constants | Same 7B run: skip-2 → spread **12.75%**; skip to the measured end (~10) → **0.18%**. Median unchanged (11648.9 → 11647.5). **70× difference in gate sensitivity.** Warmup is also not constant: 10-11 steps at 7B, 1 at 0.5B with warm caches. |
| 1.3 | Report `spread` **and** `iqr_spread` **and** `outlier_steps` — never one number | a single dispersion figure | One clean 25-step run: raw **38.6%**, IQR **0.072%** — 22 of 24 steps inside 0.07%, two isolated stalls carry the rest. **500×.** Raw as a band gates nothing; hiding the stalls discards a real host-side signal. |
| 1.4 | Legs are compared on a **common window** (`compare_legs.py`), never as a ratio of two summaries | each leg on its own window | `useful_tokens` swings **±25%** step to step (7389..11945). Three runs of one config: own windows **863.2/862.6/872.2** (spread 1.1%), common window **872.4/871.9/872.2** (spread 0.06%). On one real pair: unaligned **+0.3294%** — a *win* against the band — vs aligned **+0.0235%**; **93% was window placement.** Data order is deterministic, so alignment cancels the term exactly. |
| 1.5 | A leg with **< 12 steady steps** is refused (`dispersion_reliable`) | using whatever IQR came out | `--iters 8` leaves 7 steady steps and the quartiles straddle the near-warmup value: IQR **1.243%** vs **0.078%** on a 30-step run of the same workload. Small *n* errs both ways; an artificially tight band is the dangerous one. |
| 1.6 | The band floor is **regime-dependent**: aligned **0.15%**, unaligned **0.5%** | one 0.3% floor | Six pairwise comparisons of one 7B config across four days: aligned worst \|Δ\| **0.0589%**, unaligned worst **0.4456%** (artifact up to 0.3867%). A single 0.3% floor is ~5× too loose for aligned *and* too tight for unaligned — 0.4456% > 0.3% means it manufactures wins. |
| 1.7 | Reproducibility is **measured per workload**, the floor is only a floor | a global constant | Same six-pair method: **0.06%** at 7B, **0.31-0.42%** at 0.5B. Consistent with the step decomposition — idle is **2.35%** at 7B and **14.17%** at 0.5B, and host jitter does not cancel. |
| 1.8 | A **profiled** or **captured** run can never be a gate leg (`status: unusable`) | treating them as ordinary runs | Capture writes tens of MB from inside a backward hook: **4043.9** vs ~4755 useful tok/s/GPU on the same config, a **15%** dent. A profiler window perturbs the steps it traces by construction. |
| 1.9 | Occupancy sampled around every leg; contamination → **bounded re-measure**, then stop | assuming the box is ours | The node carries ~20 unrelated containers. Policy is re-measure (default 2 retries): re-measuring costs one leg, killing a neighbour could cost days of someone else's work, and the workflow cannot tell which it is. Exhausted retries are **fatal**, not a "no win". |
| 1.10 | `config_digest` covers **workload identity only**; `impl_digest` covers implementation; `digest_schema` versions the definition | one digest over the whole snapshot | Requiring the whole snapshot to match refused an attention-backend comparison outright, before looking at the trajectory. Separately, two runs of one workload disagreed only because the digest *definition* had changed — re-summarising from the sidecars gave identical digests, so that case is `stale_summary`, not "different workload". |

## 2. The bottleneck model

| # | Decision | Evidence |
|---|---|---|
| 2.1 | Rank by **critical-path share**, not `pct_gpu_time`; a communication kernel is scaled by its **exposed** fraction | A fully overlapped collective can own 30% of GPU time and be worth zero. The inference layer's `pct_gpu_time × achievable_speedup` would send the budget there. |
| 2.2 | **Profile at the real scale.** A toy profile routes the budget to the wrong kernel | Head op changes *class*: 0.5B → attention backward 21.5%; 7B → GEMM `_Linear [[3584,3584]]` 19.0%, with `aten::mm` vocab-projection backward 18.6% second and attention third at 14.7%. Buckets also move: compute 82.8% → 96.08%, idle 14.17% → 2.35%. |
| 2.3 | Profiling is **probe-driven**, not a stack feature | `grep -c profile megatron/training/arguments.py` → **0**. Upstream ships no profiler arguments; the `--profile` family in Primus recipes is Primus's own. The capability manifest was corrected to say so. |
| 2.4 | A **read-only** config diagnosis substitutes for ConfigSweep in `fast` | Without it the run spends its budget on kernels whose wins cannot appear. On this workload it returns `config_first` on **70.6% padding**, which `fast` cannot fix — saying so makes a later rejection attributable to the configuration. |
| 2.5 | Multi-rank profiling; skew reported, never averaged | Rank 0 vs rank 4 at 7B agree to 0.4 points; `rank_skew_pct` 0.001. The straggler bucket is empty at TP1×DP8 — worth re-checking at TP>1. |

## 3. The correctness gates

| # | Decision | Evidence |
|---|---|---|
| 3.1 | L0 judges a candidate against **the error the original already carries**, not a chosen rtol | The untouched original scores `max_rel_err` **11.7** against its own fp32 reference, because `dgrad` is ~77% zeros (padding carries no gradient). Any fixed rtol is either blind or permanently red. |
| 3.2 | L0 checks **`scale_bias` = ⟨err,ref⟩/⟨ref,ref⟩**, not only magnitude and sign balance | A `×1.005` candidate passed **both** the magnitude check (the original's own bf16 error, 0.764, is *larger* than the bias) and the sign test (0.528). `sign_balance` detects an *additive* bias; a multiplicative one leaves it at 0.5. With `scale_bias`: **FAIL at +5.00e-03**, the injected value recovered exactly. |
| 3.3 | Gate on **all** outputs, not the forward | Sensitivity moves with scale: at 0.5B a `×1.005` bias is caught on `out`; at 7B a `×1.002` bias passes `out` on magnitude and is caught on `wgrad`/`dgrad` (+2.01e-03 / +2.00e-03). |
| 3.4 | L1 is a **step-indexed envelope** anchored on step 1, not a scalar over a window | Training is chaotic: two identical runs are bit-identical at step 1, 0.76% apart by step 5, **2.04 apart in absolute loss by step 21** (loss ≈ 3). Calibrating `max\|Δloss\|` over 25 steps yields a limit of **6.1 on a loss of 3** — a gate that passes anything. Step 1 is a pure forward on identical weights, so it is bit-reproducible (verified at 0.5B/2GPU *and* 7B/8GPU); `grad_norm` at step 1 is not (0.87% at 0.5B, 0.165% at 7B — more DP ranks average more of it away). |
| 3.5 | The step-1 floor is **two-tier**: `overlay` 0.05%, `kernel_swap` 2.0% | Measured against an **L0-verified** equivalent pair (TE `_LayerNormLinear` vs a pure-PyTorch reimplementation, swapped in through the overlay): same kernels **0.000000%**, verified swap **1.748837%** — **35×** the original guess. The reference computes in fp32, so 1.75% is an *upper* bound for a same-precision swap. |
| 3.6 | `calibrate_swap` emits **`pair_usable`** from a 2×2 (two configs × two repeats) | Three "obvious" calibration pairs were rejected **by measurement**: `NVTE_USE_HIPBLASLT` on/off and `gradient_accumulation_fusion` on/off are **bit-identical** at step 1 (neither changes the kernels), and `--transformer-impl local` cannot run packed sequences at all. Two of them would have read as "kernel swaps are numerically free". |
| 3.7 | L0/L1 can only **reject**; acceptance needs throughput above the band | A faster wrong trainer is a regression, and neither gate can see throughput. |

## 4. Extraction and reintegration

| # | Decision | Evidence |
|---|---|---|
| 4.1 | The extracted task's objective is **fwd+bwd combined** | The top head by critical path is a *backward* kernel at both scales (0.5B `FusedAttnFuncBackward` 21.5%; 7B has three backward kernels in the top four). A forward-only task would not see the biggest candidate, and would call a kernel 2× faster forward and 3× slower backward a win. |
| 4.2 | fwd+bwd is wrapped into **one callable** returning `(out, dgrad, …, wgrad)` | The kernel layer's harness compares multi-tensor returns component-wise and times whatever closure it is given, so one wrapper buys L0 on every output *and* the combined latency — with **no change to the kernel layer**. |
| 4.3 | Runtime handles are recorded as **opaque context**; a task is never a replay | TE's `_LayerNormLinear.apply` carries a `ProcessGroup`, a `WeightGradStore` and the owning module. An unguarded `torch.save` on them **killed an 8-rank job** mid-backward. They are not picklable and not reconstructable offline, so the task is the pure-tensor operation and the fp32 reference must be independent. |
| 4.4 | The golden comes from an **isolated re-run of the original in the live process** | Three cheaper routes fail, each plausibly: `register_hook` on an input returns the **sum over all consumers** (the hidden state also feeds the residual add); intercepting the op's own `backward` **cannot be paired** to the captured forward (backward runs in reverse and one `Function` class serves several call sites — TE's `_LayerNormLinear` is both the QKV projection 896→1152 and the MLP's first layer 896→9728); pairing those by shape does not discriminate, because they share every shape except the weight. Mis-pairing shows up as `rel_mean_abs` ≈ **1.000**, the signature of comparing uncorrelated tensors. |
| 4.5 | Attach a zeroed **`main_grad`** buffer to cloned weights | TE dereferences it unconditionally under grad-accum fusion (`'Tensor' object has no attribute 'main_grad'`), and afterwards the buffer holds exactly this op's weight-gradient contribution — the quantity the real backward never returns. |
| 4.6 | Argument layout is **read per op**, carried in `meta.json` | `_Linear.apply` takes the **weight** first; `_LayerNormLinear.apply` takes the input first. A template hardcoding either one unpacks the other silently wrong. |
| 4.7 | Engagement is proven **per half**, with independent counters | An overlay binding only the forward produces a plausible throughput number and a plausible loss curve — the exact failure L1 exists to catch, far cheaper to catch here. A single counter cannot distinguish "both halves ran" from "the forward ran twice". Validated by identity rebind: 96 fwd / 64 bwd on both ranks, **0.109%** throughput cost (inside the band). |
| 4.8 | `GEAK_OVERLAY` belongs to the **implementation** half of the snapshot | Otherwise two legs differing only by an overlay look byte-identical in the record and the report cannot say what was swapped. |

## 5. Orchestration

| # | Decision | Evidence / rationale |
|---|---|---|
| 5.1 | **Two** modes (`fast`, `deep`), not three | A training gate is two job launches and blocks every lane, so the useful distinction is "fixed config" vs "search the config too". |
| 5.2 | A gate takes the **whole box**: lanes drain first (`makeBarrier`) | A TP1×DP8 job occupies all 8 GPUs, so the inference layer's "lanes and the serving slot coexist" has no analogue. Reserving a subset fails on *measurement basis*, not performance: the parallel mapping is part of the workload identity, so a DP6 number is not about the DP8 configuration. |
| 5.3 | Budget counts **gates**, not only wall clock | One A/B ≈ two job launches (checkpoint load + warmup + N steps) and it blocks every lane: at 7B a leg is ~7 minutes. Gates are the scarce resource — the opposite of the inference layer. |
| 5.4 | Six **ConfigSweep seam hooks** exist in `fast` with trivial behaviour | Hooks 1-3 (carried config, config as an explicit argument, independently callable reprofile/restrategize) are the expensive ones to retrofit: threading a config through every measurement afterwards touches every call site, and that is where a silently stale reference leg would hide. |
| 5.5 | `deep_mode` **refuses to run**; a phase with no role **fails loudly** | Reporting a fast-mode result under a deep label, or silently skipping a gate, produces a result that looks complete and is not. |
| 5.6 | An accepted candidate becomes the **reference** for the next gate, and triggers a re-profile | Deltas compound on the carried stack, and an accept moves which kernels dominate. |
| 5.7 | Memory is a **currency, not a metric**; dormant in `fast` | It is only spendable through a config change (raise MBS, drop recompute), which `fast` does not have. Recorded as `banked_memory_pct`. Headroom measured: peak **78 GB of 192 GB** at 7B. |
| 5.8 | `bench_train.sh` works identically from **host or container** | The orchestrator runs wherever Claude Code runs; the stack only exists in the image. Relative paths are resolved before re-entry (the container starts in `/`), and a path outside every mount is rejected with the mounts listed. Both failure modes were hit in testing. |

## 6. Framework portability (built in v1, exercised in v2)

| # | Seam | Why it is not deferrable |
|---|---|---|
| 6.1 | Canonical config **IR** | Otherwise the KB fills with Megatron/Primus literals and the cross-run knowledge — the most valuable asset here — does not transfer. |
| 6.2 | **Self-evidencing measurement** (torch-level probe) | Justified as portability insurance; turned out to be *required* on framework #1, since upstream Megatron logs no tokens/s. A log parser would also cost one parser per stack and yield numbers that are not comparable across stacks. |
| 6.3 | **Capability manifest**, route on capability not framework name | Already caught one over-claim: `step_ranged_profiler` was declared true and corrected to false when upstream turned out to have no profiler arguments. |

## 7. Bare Megatron on ROCm — compatibility facts

| Symptom | Cause | Handling |
|---|---|---|
| `TypeError: NoneType + str` in `_compile_dependencies` | `cpp_extension.CUDA_HOME` is None on ROCm; the path builds `.cu` fused softmax | `rocm_compat.py` no-ops it, as Primus does |
| `expected 1D tensor` / `Boolean value of Tensor ... ambiguous` in both rope paths | upstream `--sft` THD packing returns a per-sample 1D `cu_seqlens`, the collate stacks it to `[MBS, n+1]`, and the rope helpers assume 1D | squeeze the vacuous batch dim at MBS=1; **refuse MBS>1 loudly** (merging rows changes semantics) |
| `NotImplementedError: only for SFTTokenizer` | `--tokenizer-type HuggingFaceTokenizer` | use `SFTTokenizer` |
| `NVTE_FLASH_ATTN set to 0, but expected 1` | image sets `NVTE_FUSED_ATTN=1`; `--attention-backend auto` disagrees | `--attention-backend fused` (now overridable — it is a config choice, not an adapter fact) |
| `int // NoneType` in `get_train_valid_test_num_samples` | `--eval-interval` is used unconditionally even with `--eval-iters 0` | set it |
| Prompt format | the three non-`default` formats hardcode Nemotron templates and pad tokens absent from Qwen's vocabulary | `default`; it skips prompt masking, which does not touch the compute graph — padding is still masked, so packing stays measurable |

## 8. Process lessons (they cost real runs)

* **Never edit a script while it is executing.** bash reads lazily by byte offset; an in-place edit shifts the remaining bytes and the running process parses garbage. It destroyed the summary stage of a completed 8-GPU run (the probe sidecars survived, so it was recoverable by re-summarising).
* **Verify that an edit applied.** A string-replace that silently does not match leaves every later edit keyed on its intended output no-op'ing too, while the tooling reports success — it only proves the script ran. **Four GPU runs** were lost to this before edits started asserting the pattern matched.
* **A negative test is worth more than a passing one.** The `×1.005` candidate passing every check is what produced `scale_bias`; the 2×2 control is what exposed three non-swaps as non-swaps.
