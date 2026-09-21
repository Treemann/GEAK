# Measuring SFT training throughput (read this before you report any number)

Every number in this workflow comes from `scripts/bench_train.sh`, which reports
`bench_summary.json`. The facts below are **measured on this box** (8x MI308X gfx942,
`rocm/primus:v26.4`, bare Megatron `pretrain_gpt.py --sft`), not assumed.

## 1. The headline is `useful_tok_s_per_gpu`, never the stack's own number
Upstream Megatron reports **no tokens/s at all**. Stacks that do report it (Primus) compute
`seq_length x batch_size / elapsed / world_size` — **nominal tokens, padding included**.

Measured on the reference workload: **71.6% of tokens are padding.**
`nominal 16748` vs `useful 4760` tok/s/GPU — the conventional number overstates useful work by
**3.5x**. Optimizing against it means optimizing a number that is 71% air.

`useful_tokens` = `loss_mask.sum()` where the stack provides one (Megatron masks padding by
pad-token-id **and** prompts by `-100`, so `labels != -100` alone misses padding, the larger term),
falling back to `(labels != -100).sum()` for HF Trainer / TRL / ms-swift.

## 2. Skip the warmup, and MEASURE where it ends
| window | n | median (ms) | spread |
|---|---|---|---|
| skip 2 (the common default) | 28 | 11648.9 | **12.75%** |
| skip 10 (measured warmup end) | 20 | 11647.5 | **0.18%** |

The median is **unchanged**; the spread differs **70x**. A 12.7% band makes every kernel win below
12% invisible. Warmup length is also not a constant — the 7B/8-GPU run settles after ~10 steps, a
0.5B run with warm caches after 1 — so `detect_warmup_end()` derives it from step-time stability
instead of trusting a skip count.

## 3. Report spread AND robust spread; never collapse them
The same 25-step run reports **38.6% raw spread** and **0.072% IQR spread**: 22 of 24 steady steps
sit inside 0.07%, two isolated stalls carry the rest. Raw spread as a noise band gates nothing;
hiding the stalls discards a real host-side signal (they appear in one job and not the next).
`spread_pct`, `iqr_spread_pct`, `outlier_steps`, `outlier_max_pct` always travel together.

## 4. The noise bands, measured
- `within_job` (IQR): **0.07-0.13%**
- `across_job`: **0.128%** (two independent jobs, identical config, on the headline)

An order of magnitude tighter than the inference layer's 2-3%. Sub-1% deltas are gateable — which is
precisely why the band must be derived from each pair's own dispersion (floor 0.3%) and never set to
a convenient constant.

## 5. A leg is not a measurement unless it is clean
`status == "ok"`, no hang, and `occupancy_verdict == "clean"`. This node carries ~20 unrelated
containers; a neighbour taking a GPU mid-leg drags one side down and the delta becomes fiction.
`unknown` occupancy counts as unusable, not as clean — unprovable isolation is not isolation.

**Contamination policy (locked): RE-MEASURE, bounded.** A contaminated leg is re-run, up to
`--retry-contaminated` times (default 2, so 3 attempts). Rationale: re-measuring costs one leg;
killing a neighbour's job could cost days of someone else's work, and the workflow cannot tell which
it is. Nothing is ever killed automatically.

Every attempt is recorded (`attempts`, `contamination_retries`) because a leg that needed three tries
is itself a finding about the node.

**Exhausting the retries is NOT a "no win".** It means the node was busy every time, which no amount
of further gating can fix, so it is fatal: `contamination_exhausted` -> `status: unusable` ->
`gateVerdict` returns `klass: blocked, fatal: true` -> the run STOPS and says so. Treating it as an
ordinary non-win would quietly spend the rest of the gate budget on measurements that cannot mean
anything. (Order matters in `legUsable`: a blocked leg also has `status != "ok"`, so the
contamination check must run FIRST or the generic one shadows it.)

## 5b. Compare two legs on a COMMON window, or the delta is mostly artifact
`useful_tok_s_per_gpu` is `median(useful_tokens) / median(step_ms)`, and on SFT data `useful_tokens`
swings **+-25% step to step** (7389..11945 on the reference workload). Each leg detects its own
warmup end — legitimately 8, 10 or 11 — so two legs summarised on their own windows average
**different samples**:

| three runs of the SAME config | useful tok/s/GPU |
|---|---|
| each on its own window | 863.2, 862.6, 872.2 — spread **1.1%** |
| on a common window | 872.4, 871.9, 872.2 — spread **0.06%** |

1.1% is 3.7x the 0.3% band floor. Measured on one such pair: unaligned **+0.3294%** (a "win") versus
aligned **+0.0235%** (nothing) — 93% of it was window placement. Data order is deterministic, so
aligning the window makes the term cancel exactly; there is nothing to correct statistically.
Always compare with `scripts/compare_legs.py`.

## 5c. A leg too short to estimate dispersion is not a measurement
The README smoke settings (`--iters 8`) leave 7 steady steps, and the quartiles then straddle the
near-warmup value: measured IQR **1.243%** against **0.078%** on a 30-step run of the same workload.
Small samples err in both directions, and an artificially TIGHT band is the dangerous one — it
accepts changes that are not there. Below 12 steady steps `dispersion_reliable` is false and the gate
refuses the leg. Use `--iters 8` to check that the plumbing works, never to measure.

## 6. Two legs must have the same `config_digest`
The digest is computed from what the **probe observed inside the ranks**, not from what the caller
intended. `ranks_agree_on_argv` additionally catches two configurations mixed inside ONE leg. A ratio
across two identities is fiction no matter how tight each leg is.

## 7. A profiled run is not a measurement
The profiler window perturbs the steps it traces. `bench_summary.json.profiled_run == true` forces
`status: unusable`, so a profiled run can never be used as a gate leg.

## Operational hazards found the hard way
- **Never edit a shell script while it is executing.** bash reads scripts lazily by byte offset; an
  in-place edit shifts the remaining bytes and the running process starts parsing garbage. This
  destroyed the summary stage of a completed 8-GPU run (the probe sidecars survived, so the run was
  recoverable by re-summarising after the fact).
- Give the jsonl dataset a fixed, frozen subset on local disk: data order must be identical between
  legs or L1 compares different batches.
