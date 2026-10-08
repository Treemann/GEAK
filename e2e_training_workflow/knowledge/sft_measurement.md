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

### 4d. Per-candidate deltas do NOT add, and the combined number has to be measured

Each candidate is gated against **the stack carried at that moment**, and a fresh run carries the
bare stack. So two kernels accepted in two runs were both measured against bare:

| kernel | measured | against |
|---|---|---|
| `lmhead_ce_unmasked_gather` | +4.66% | bare stack |
| `mlp_fc2_down` | +7.34% | bare stack |

**Neither +12.0% (adding) nor +12.4% (compounding) is a measurement.** Both kernels shorten the same
step, and whether they overlap depends on where their time sits on the critical path — the lmhead
head was 6.41% of it, fc2's ancestor 36.17% with little headroom. The answer is an empirical
question about one step, not an arithmetic one.

Until `GEAK_OVERLAY` took a list this was not merely unmeasured but **unmeasurable**: the two
overlays rebind different symbols (`GPTModel._postprocess`, `transformer_engine…linear:_Linear`) and
only one manifest could be installed. A workflow that can accept N candidates and deliver none of
them as a stack has a hole where its deliverable should be.

`scripts/measure_stack.sh` closes it: it verifies each bundle's bytes against its own
`SHA256SUMS.overlay`, installs every overlay with its own engagement file, and takes **interleaved**
ref/cand pairs so node drift cannot align with condition. It deliberately does **not** re-gate —
L0 and L1 judged each kernel when it was accepted — but it **does** re-prove engagement per overlay,
because a stack that silently drops one reports a smaller number and looks like an honest
disappointment.

**MEASURED 2026-09-29: the combined stack is +12.36%, and the savings COMPOUND.**

| pairing | aligned | unaligned | window artifact |
|---|---|---|---|
| ref1 vs cand1 | **+12.3746%** | +12.9547% | 0.5801% |
| ref2 vs cand2 | **+12.2809%** | +7.3874% | **4.8935%** |
| ref1 vs cand2 | **+12.3664%** | +9.6050% | 2.7614% |
| ref2 vs cand1 | **+12.3985%** | +10.6693% | 1.7292% |

Four pairings, spread **0.12 points**. Null control `ref1 vs ref2` = **+0.0706%**. Window-immune
cross-check on `step_median_ms`: −10.972% ⇒ **+12.324%**, agreeing with the aligned figure to 0.05
points. Engagement proved per overlay on all 8 ranks (`fwd == bwd`, zero fallbacks).

Against the two hypotheses:

```
adding      4.66 + 7.34                 = +12.00%   off by 0.36 points
compounding 1.0466 x 1.0734 - 1         = +12.34%   off by 0.013 points
measured                                  +12.36%
```

**They compound, which means they do not overlap on the critical path** — each shortens a distinct
part of the step, consistent with them being different ops (the LM head and the FC2 down-projection).
That is an empirical fact about this step, not a general rule: two kernels contending for the same
stretch of the critical path would sub-add, and nothing here predicts which case a future pair falls
into. Measure it.

Note also what the unaligned column does: on identical data it ranges **+7.39% to +12.95%**, a
**5.5-point** spread, with the artifact reaching 4.89%. The same four legs would support almost any
story if the windows were not aligned.

### 4e. RETRACTED, and the retraction is the lesson (2026-10-08)

**What was claimed here hours earlier, and is false:** that `attn_dense_wgrad_rowslice` had been
recorded `no_win` at +1.147% because a noisy reference leg "corrupted the delta by seven points",
and that it was really worth +8.46%.

**What actually happened.** The two measurements used **different references**, so they measure
different quantities and both are correct:

| | reference leg carried | candidate leg carried | measures |
|---|---|---|---|
| original gate | bare **+ fc2** | + attn | attn's increment **on top of fc2** |
| re-measurement | **bare** | attn overlay, which **chains fc2 internally** | **fc2 + attn combined** |

`overlays/attn_outproj_0928/impl.py` says so in its own header — it *delegates to* the accepted fc2
overlay rather than replacing it, importing that module by path, because both rebind the same symbol
(`transformer_engine…linear:_Linear`) and `_Linear` serves two call sites per layer. The
re-measurement's reference had **zero** `geak-overlay` lines in its log; the original gate's
reference had them.

**They agree.** fc2 alone is +7.34% against bare; fc2 + attn is +8.46% against bare; so attn's
increment is `1.0846 / 1.0734 − 1 = +1.043%`, against the **+1.147%** the original gate measured
directly — **0.10 points apart**. The original figure was right, the noisy legs did not corrupt it,
and the candidate is worth about **+1.0%** on top of fc2, not +8.46%.

**The lesson is the one this file already states and I violated anyway** (§4d): *per-candidate
deltas are measured against different references and do not compare.* The guard that would have
caught it exists in the orchestrator — an accepted candidate becomes the carried reference, and the
Integrator is told to use that, not the bare baseline — but a hand-run re-measurement bypasses the
orchestrator entirely, and nothing checked what the reference leg was carrying. **Before comparing
two deltas, verify both references, by reading the overlay lines in each leg's log.**

`inconclusive_noise` (committed the same day) survives this retraction intact and for its original
reason: a 2.467% band genuinely cannot resolve a +1.147% signal, so "we could not measure" was the
honest verdict. What does not survive is the justification added with it — the delta was never
corrupted.

### 4f. The deliverable had never been cold-started (2026-10-08)

Five runs produced bundles and four `bundle_recheck_*` legs re-ran them — but always **from the
Integrator that had just built them**, never from a cold start through the documented entry points.
Doing that once, in twenty minutes, found two defects in series.

**Defect 1: the candidate entry point worked exactly once.** `launch.sh` cleared a fixed install
directory with `rm -rf` before copying the overlay in. The training container runs as **root** and
leaves a root-owned `__pycache__/`, so the second invocation died with `Permission denied` and the
candidate leg never started. `--ref` kept working, because it installs nothing — so the bundle looked
*half* healthy and the failure read as a candidate problem rather than a script problem.

**Defect 2, introduced by the fix, and this is the instructive one.** Moving the install to a
per-invocation directory left the manifest's `replacement` pointing at the **hardcoded** old
container path. The resulting leg was flawless on every surface check — `status: ok`, 48 steady
steps, IQR 0.075% — and measured **−0.0242%**. It ran the *original* implementation. A leg like that
does not look like a bug; it looks like an honest disappointment, and would have been read as "this
kernel does not actually help".

What caught it was `launch.sh`'s own engagement check: `0 ranks wrote a proof, expected 8 — PARTIAL
ENGAGEMENT`, `rc=1`. That is DECISIONS 4.7 earning its place — an overlay bound on only one half
produces a plausible number and a plausible loss curve; **zero** binding produces the most plausible
number of all.

After pairing the container path to the install directory, the third cold start agrees with every
earlier measurement of this kernel:

| source | delta |
|---|---|
| gate, during the run | +7.52% |
| Director's independent re-measurement | +7.34% |
| **cold start through `./launch.sh`** | **+7.4348%** |

The rule that prevents the first defect is now in `roles/integrator.md`: install into a
per-invocation directory and never `rm -rf` a path the container may own. `l2_soak.sh` and
`measure_stack.sh` had both hit the identical trap and solved it the same way; the bundle generator
simply never learned it.

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

### `config_digest` records INTENT, not what the framework realized (designed 2026-10-08)

`train_probe._config_snapshot()` reads `sys.argv`. So two legs share a `config_digest` when the same
flags were *requested*, which is not the same as the same config being *run*. Megatron's
`validate_args` clamps, defaults and silently drops flags -- `--recompute-granularity` without a
`--recompute-method` is the standard example -- and this layer has already been bitten by exactly
that shape once, when `--kind` was silently ignored and applied a 40x tighter floor for a day.

The realized config is available: inside the training process `megatron.training.get_args()` returns
the namespace *after* validation, which is what actually ran.

**The fix is additive, and deliberately so.** The obvious move -- have `config_digest` hash the
realized namespace -- bumps the identity schema to v4, and v3-to-v4 requires RE-MEASURE, not
re-summarise. That would invalidate every leg now on disk, including the three 0923 reference legs
the L2 v3 design depends on and the baselines a run is using today. The cost is real and the benefit
does not require paying it. So:

* keep `config_digest` as it is, at schema 3, argv-derived. Old and new legs stay comparable.
* add `realized_config` and `realized_digest` as NEW fields, captured from `get_args()` when the
  framework exposes it and absent otherwise.
* have the comparison report an argv-vs-realized MISMATCH as a finding. That is where the safety
  actually lives: the question worth answering is not "what is the realized hash" but "was the flag
  I passed honoured?", and a mismatch answers it directly.
* move to schema 4 later, as a deliberate re-measure, once enough legs carry both.

**Not implemented yet, and the reason is scheduling, not doubt:** every new leg re-reads
`scripts/train_probe.py` at launch, so editing it while a run is measuring would make legs from
before and after the edit incomparable inside one run.


## Operational hazards found the hard way
- **Never edit a shell script while it is executing.** bash reads scripts lazily by byte offset; an
  in-place edit shifts the remaining bytes and the running process starts parsing garbage. This
  destroyed the summary stage of a completed 8-GPU run (the probe sidecars survived, so the run was
  recoverable by re-summarising after the fact).
- Give the jsonl dataset a fixed, frozen subset on local disk: data order must be identical between
  legs or L1 compares different batches.
