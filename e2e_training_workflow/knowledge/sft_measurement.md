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

### 4b. The band was RECALIBRATED 2026-09-23, and the old one was backwards

Measured over **47 same-identity baseline pairs** on disk (true delta zero by construction, compared
with `compare_legs.py`):

| class | pairs | median \|aligned Δ\| | worst | old band assigned | old verdict |
|---|---|---|---|---|---|
| 7B ws=8 | 10 | 0.0236% | **0.0823%** | 0.914% | ~11× too **loose** |
| 0.5B ws=2 | 37 | 0.1441% | **0.4966%** | 0.15% floor | **6 FALSE WINS** |

The old formula `max(floor, max_iqr × 2)` gave the **widest** band to the **most reproducible**
workload. `worst_iqr_spread_pct` is a *within-leg* dispersion and carries almost no information about
across-job reproducibility: pearson(max_iqr, |aligned Δ|) = **+0.07** at 0.5B and **−0.20** at 7B.
Where it is predictive at all the measured OLS slope is **0.147**, not the 2.0 assumed.

The term was also **dimensionally wrong** — it priced the dispersion of ONE step as the uncertainty
of a MEDIAN OF n STEPS, dropping the 1/√n. That single error produces both failure directions at
once. Corrected: `se(leg) = 0.929 × iqr / √n` (0.929 = 1.253/1.349), combined in quadrature.

```
band = max( FLOOR_WORKLOAD , 4 × se_pair , 2 × run_measured_across_job )
```

**The floor does the real work.** Measured: the dimensional fix *alone* is worse (15/47 false wins);
with the per-workload floor it is **0/47**, against 6/47 for the old formula. The `se` term rarely
binds — it is kept for the noisy-candidate case the IQR term was reaching for (the 7B candidate leg's
own IQR was 0.533% against its reference's 0.049%).

Floors live in `workloads/*.env` as `ACROSS_JOB_FLOOR_PCT`: **0.15%** at 7B, **0.5%** at 0.5B
(the shipped default). Do not use the 7B value at 0.5B — 15 of those 37 known-null pairs then read
as wins.

**The cheapest lever is not the band, it is window length.** At 0.5B, pairs where *both* legs had
≥29 steady steps have worst |Δ| **0.2544%**; pairs where either leg had 19 have **0.4966%**. About
ten extra iterations nearly halves the noise floor, and unlike a band it costs nothing in
sensitivity.

Caveats, recorded rather than buried: pairs are non-independent (legs are reused), 7B rests on 5
clean legs and 0.5B on 9, the 0.5B legs span several days while the 7B legs span two, and the
noisy-candidate evidence is n=1.

### 4c. Both controls, for the throughput gate

The L2 soak failure taught the general rule the hard way: **a gate whose sensitivity has not been
demonstrated on a known input is not a gate.** L0 has had a negative control since `scale_bias` was
introduced. L1 now has a null control inside `calibrate()`. The throughput gate's two controls,
measured 2026-09-23 on real legs with the recalibrated band:

| control | input | result |
|---|---|---|
| **null** — must not fire | 47 same-identity baseline pairs, true delta 0 | **0 false wins** (old formula: 6) |
| **positive** — must fire | `td_lmhead_legA_ref` vs `td_lmhead_legB`, aligned **+4.6597%** | **fires**, `klass: win`, band 0.415% |

The band on that real pair is **0.415%** where the old formula assigned **0.914%** — roughly double
the sensitivity *and* fewer false positives, which is the combination that tells you the old
constant was mis-specified rather than merely cautious. Note the same pair's raw ratio carries a
**3.19%** window artifact, so it is also a live demonstration of why an unaligned delta may never
accept.

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

**The artifact is NOT bounded by the 0.5% unaligned floor (measured 2026-09-22, 7B/8GPU).** The
worst case above was 0.3867%, which is where the `unaligned: 0.5` floor came from. A later pair
blew straight through it: two **identical-config** baseline legs, true delta zero by construction,
gave `unaligned_delta_pct` **+4.5925%** against `aligned` **-0.0588%** — window placement alone
manufacturing a win **5x the run's 0.914% band**. No floor can bound this term, because it scales
with the +-25% `useful_tokens` swing rather than with a constant.

Consequence, now enforced in `gateVerdict()`: **a raw ratio may REJECT but may never ACCEPT**
(`klass: no_aligned_delta`). The asymmetry is deliberate — a false reject costs one candidate, which
can be re-gated once an aligned comparison is reported; a false accept installs the candidate as the
reference every later delta compounds on, and puts the artifact in the headline permanently.

How this went unnoticed for two runs: the Integrator's role card *did* require `comparison`, and the
unit tests *did* exercise the aligned path — but they passed `comparison` straight into
`gateVerdict()`, while the single production call site never forwarded it. `alignedDelta()` therefore
returned null on **every real gate**, and the honest `delta_source` label was the only symptom. Test
the call site, not just the function.

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

**Correction made 2026-09-22: the guard was far weaker than this section implied — schema 3.**
Until then `workload_identity` contained **five** realized fields (`micro-batch-size`,
`global-batch-size`, `seq-length`, `sft-tokenizer-prompt-format`, `world_size`). The probe scraped
nine `--flag value` pairs out of `sys.argv` and nothing else, so:

* **`attention_dropout` 0.0 and 0.1 digested identically** (`288f95e335ac66a4` on both). Diffing the
  two 686-line realized Megatron arg dumps gave exactly **one** differing line, while step-1 loss
  moved **3.1689 -> 4.3728** on identical data (`useful_tokens[step1] = 7273` in both) and throughput
  differed by **+7.3%**. A headline "+7.3% win" was in substantial part "we did not do the dropout".
* **The model was never covered.** `ARCH_ARGS` never reached the snapshot, so a 0.5B and a 7B leg at
  the same mbs/gbs/seq/world_size would have digested the same.
* **TP/PP/CP were in the allowlist but never appeared**, because the adapter does not pass them
  explicitly — so the "parallel mapping" this guard was documented to cover was covered on no leg
  ever measured.
* store_true flags (`--bf16`, `--sft`, `--finetune`, ...) could never be captured by a pair-only scan.

Schema 3 captures the regularization/optimizer/architecture set plus presence-only flags: **37 config
fields, 33 of them identity**, verified on a real leg. The version lives in the **snapshot** (header
`snapshot_keys: v3`), not in the summariser, because a v2 sidecar physically does not contain
`attention_dropout` — stamping v3 on it would produce a digest that *claims* to cover dropout and
does not. **A v2-vs-v3 pair must therefore be RE-MEASURED, not re-summarised**, which is the one
previous bump this remedy does not apply to.

Residual limitation, stated rather than papered over: the digest still reads **argv**, not the
realized arg table. A leg that relies on a framework default (the baseline does not pass
`--attention-dropout` at all) records nothing for that key, so a default that changed between runs
would still be invisible. The container image is pinned, which is what currently makes that safe.

`impl_digest` had the mirror-image defect: it hashed the overlay manifest's **basename**, a label
that moves on a repackage and does **not** move when the kernel bytes under a kept filename are
swapped — yet it was cited as proof that a leg "IS the TRUE baseline stack re-measured". It now
hashes the manifest's semantic fields plus the **contents** of every file it points at, excluding
paths and `engagement_out` (those are per-leg output locations; hashing them would make two legs
running the identical kernel look like different implementations).

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
