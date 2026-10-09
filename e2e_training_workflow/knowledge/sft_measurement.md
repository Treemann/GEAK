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

### The full Tier-0 lever pack, measured (2026-10-09)

All three Class-A levers have now been measured on the 7B reference workload. **None of them pays**,
and two of the three cost memory rather than banking it. Recorded so the next run does not re-spend
GPU hours rediscovering it.

Reference leg 837.5 useful tok/s/GPU; both candidates judged against the same leg and the same band,
which is why they were measured in one run rather than two.

| lever | aligned delta | window artifact | memory | verdict |
|---|---|---|---|---|
| comm overlap | +0.1178% (band 0.15%) | 0.3298% | **−780 MB** | `no_win` |
| recompute `selective` | **−3.403%** | 0.5102% | **−838 MB** | regression |
| attention backend `flash` | −0.7284% (band 1.978%) | 0.5243% | — | `no_win`, inconclusive |

Three things worth keeping beyond the verdicts.

**The memory converter has nothing to convert.** `recompute selective` is the knob `fast` banks
memory *for*, and on this configuration it **freed nothing and cost 838 MB**. The reason is
specific: with NVTE fused attention the core-attention path never materialises the seq x seq score
matrix, so the activations selective recompute exists to drop are already not stored, and what
remains is its workspace. The whole bank-and-spend design path therefore has no work to do on this
workload, and that is a property of the stack, not a measurement artifact.

**The band was dominated by the REFERENCE leg's noise**, not the candidate's: 4 x se_pair = 1.978%,
with the reference's forced-window IQR at 2.607% against the candidate's 0.07%. The reference's own
window IQR was 0.096% — six outlier stall steps land inside the *common* window. This is exactly the
case the se term exists for, and it arrived with the noise on the side nobody expects.

**A window-immune cross-check caught what the band could not resolve.** For the attention backend,
`step_median_ms` moved +0.734%, implying −0.729% throughput, agreeing with the aligned −0.7284% to
0.001 points. So the direction is real even though |delta| sits under the band: flash is at best
neutral and more likely a small regression here. Step-median is immune to window placement, and
quoting it alongside a sub-band delta distinguishes "too small to resolve" from "not there".

**And the v4 identity fix is confirmed on real legs.** On both levers `config_digest` stayed
identical while `impl_digest` moved — the behaviour whose absence on the comm-overlap pair exposed
the blindness in the first place.

### The band was estimated on a different window than the delta (fixed 2026-10-09)

The band estimates the uncertainty OF the delta, so it has to be computed from the same sample. It
was not. `compare_legs.py` forces both legs onto a common window and reports the delta there, while
`noiseBandPct` read `worst_iqr_spread_pct` out of each leg's `bench_summary.json` — each leg's OWN
window. Usually the two barely differ, which is why this survived every run so far.

**Measured, on the `cfgrecompute_ref` leg, identically on all 8 ranks:**

| window | steps | IQR |
|---|---|---|
| own, `[5:30]` | 25 | **0.096%** |
| common, `[6:30]` | 24 | **2.607%** |

One step of difference, 27x the dispersion. The sorted durations show why: nineteen values inside
11642–11660 ms and then a cliff to 11952, 12006, 12023, 12041, 12113, 13143. The third quartile
lands exactly on the edge of it, and `v[3n/4]` picks the last tight value at n=25 and the first
outlier at n=24.

**The verdict turned on it.** The attention-backend lever at −0.728% was published as a `regression`
against a band of 0.475%; on the window-matched band of 1.978% it is `inconclusive_noise` — the
band cannot resolve it. Re-derived for all three levers:

| lever | delta | published band | window-matched band | verdict |
|---|---|---|---|---|
| comm overlap | +0.118% | 0.150% | **0.150%** (unchanged — quiet in both windows) | `no_win` |
| recompute selective | −3.403% | 0.537% | **1.979%** | `regression` (unchanged) |
| attention flash | −0.728% | 0.475% | **1.978%** | **`inconclusive_noise`** (was `regression`) |

**The direction that matters is the other one.** A band too tight produces false REJECTS here, which
cost one candidate each. The same mechanism would have called a **+0.728% non-win a WIN** — exactly
what `legUsable`'s own comment warns about: *"an artificially tight band is the dangerous one: it
accepts changes that are not there."* Nothing has been accepted on a narrow band, but nothing was
stopping it.

Each verdict now carries `band_window`, naming the sample its band came from, because two bands 27x
apart are not comparable across runs and a reader cannot tell them apart from the number alone.

**The underlying fragility is NOT fixed**, and is recorded rather than quietly patched: the quartile
estimator is nearest-rank (`v[n//4]`, `v[3n//4]`) with no interpolation, so it jumps discontinuously
when the quartile sits on a cliff. Linear interpolation would have given 0.70% instead of 2.607% —
smoother, still a 7x jump, because the cliff is real. Changing it would move every band in the
system and invalidate the 0/47 false-win calibration, so it needs its own re-validation and is not
being done as a side effect of this fix.

### The null control is a script now, and it says the 7B floor was too tight (2026-10-09)

The band's licence was one line in this file — *"47 same-identity baseline pairs, true delta 0, 0
false wins"* — and every proposal to touch anything the band depends on has been declined on the
grounds that re-validating it would be expensive. It is not expensive. The legs are on disk and the
pairs rebuild from their identity digests. `scripts/calibrate_band.py` does it in minutes, no GPU.

A NULL PAIR is two legs with the same `config_digest` AND `impl_digest`: the true delta is zero by
construction, so any "win" is false. Reconstructing that set took three corrections, and each one
moved the answer by more than the thing being measured:

| reconstruction | false wins | what was wrong |
|---|---|---|
| all legs with ≥25 steps | 73 / 386 | included **capture runs**, which `bench_train.sh` itself prints are not valid throughput measurements |
| + the gate's usability rules | 49 / 400 | still re-derived identity instead of reading it, and omitted `world_size`, so legs at different world sizes grouped together |
| + identity read from `bench_summary.json` | 32 / 400 | still paired **v2 legs**, whose 9-field snapshot cannot separate them |
| + identity schema ≥ v3 | **16 / 400** | — |

The v2 exclusion is the one worth remembering. `td_ref_legA_p0` runs **7.3% faster** than
`td_baselineA` with a byte-identical config snapshot, which looks exactly like a catastrophic digest
collision. Diffing Megatron's own 647-line realized dump gives **one** differing line —
`attention_dropout 0.1 vs 0.0` — the precise gap v3 was built to close. The legs were never the same
measurement; v2 could not say so. Grading the band on them would have indicted it for a limitation
fixed weeks earlier.

**The result, on 400 clean v3 same-identity pairs:**

| | |
|---|---|
| false wins at the shipped floor (0.15%) | **16 / 400 (4%)** |
| |delta| median / max | 0.051% / **0.225%** |
| every false win | at the FLOOR, none driven by the `se` term |

All sixteen are 7B pairs, all marginal (0.150–0.225%), all spanning different jobs on different
days. So `ACROSS_JOB_FLOOR_PCT=0.15` in `workloads/qwen2_5_math_7b.env` **understates the measured
across-job reproducibility of this workload**, and is raised to **0.25%**.

That is the existing method applied to better data, not a retune. The 0.15 came from "~1.8× the
worst of 10 clean pairs (0.0823%)"; the multiplier was carrying the thinness of a 10-sample maximum.
Over 400 pairs the worst is 0.225% and 0.25% covers it with 1.11×. Cost: nothing accepted to date
comes near — the three accepted wins are +1.147%, +7.4348% and +12.36%.

**And the quartile question is settled.** The nearest-rank estimator was left unpatched because
changing it "would invalidate the 0/47 calibration". Re-run both ways: **nearest 16/400, interpolated
17/400**, identical median band, max band 2.026% vs 1.965%. The estimator is not what the null
control is sensitive to, so the fragility can be fixed on its own merits whenever it is convenient,
and the blocker was imaginary.

**This is only the null half.** An infinitely wide band scores 0/400 and gates nothing. The positive
half is not reconstructible from same-identity legs and lives in the accepted candidates. Read them
together or neither means anything.

### OPEN: the reference workload trains with dropout the model does not use (found 2026-10-09)

Found while matching the HF stack to Megatron. The Megatron adapter passes neither
`--attention-dropout` nor `--hidden-dropout`, so every Megatron leg — the 7B reference included —
runs at Megatron's defaults, **attention 0.1 and hidden 0.1**. Qwen2.5's own `config.json` says
`attention_dropout: 0.0` and has no hidden dropout. HF loads that config, so an HF leg trains with no
dropout at all.

This was known as an IDENTITY problem (v2 could not tell 0.0 from 0.1) but was never raised as a
RECIPE question, and the cost is not small: on the 0.5B workload attention dropout alone is worth
**+7.3% throughput** and moves step-1 loss 4.3728 → 3.1689 on identical data. Hidden dropout is still
on in both of those legs, so the full gap to Qwen's own recipe is not yet measured.

**Not changed, deliberately.** Switching the reference to dropout 0 would move every baseline, every
band and the three accepted wins (+1.147%, +7.4348%, +12.36%), all of which were measured against a
dropout-0.1 stack, and it would change the profile itself — dropout kernels leave the critical path
and the hot-kernel ranking shifts. Continuity of the existing evidence is worth more than fidelity
of a recipe nobody has asked to change.

**What IS done:** cross-stack comparisons run Megatron with `--attention-dropout 0.0
--hidden-dropout 0.0` explicitly, so the two stacks train the same model the same way. Comparing
HF-at-0 against Megatron-at-0.1 would attribute a recipe difference to the stack.

**The question for whoever owns the workload:** should the reference recipe follow the model's own
config? If yes, it is a new workload descriptor with its own baselines and bands, not an edit to
this one.

### The Profiler phase on the second stack (2026-10-09)

A profiled HF TRL leg (0.5B, 2 GPUs, steps 10–12) goes through `parse_train_profile.py` unchanged:
trace written by the probe-driven profiler, four buckets, Top-N with op and shapes. So the
Profiler phase is stack-neutral in practice, not just in intent.

| bucket | HF TRL 0.5B | Megatron 7B reference |
|---|---|---|
| compute | 70.1% | 92.2% |
| idle (no kernel resident) | **24.4%** | 2.5% |
| exposed comm | 5.5% | 3.4–5.4% |
| comm overlapped | **61%** (DDP buckets) | **0%** |

The diagnosis comes back **`config_first`**: a quarter of the step has no kernel resident — eager
PyTorch issuing many small elementwise kernels, launch-bound at this size — so no kernel win can
show end to end until that is addressed (compile/fusion, larger micro-batches). That is the right
call, made by logic written for Megatron and applied unmodified.

One classification leak fixed: ROCm AOTriton's flash-attention backward kernels (`bwd_kernel_dk_dv`,
`bwd_kernel_fuse`, `bwd_kernel_dq` — 13.8% of the step) were filed as `other` because their NAMES
match no attention pattern. The classifier now falls back to the parent op
(`aten::_flash_attention_backward`), which is stack-neutral vocabulary. Megatron's classification is
unchanged.

### HF trained pure bf16 at first, and identity could not see it (fixed 2026-10-09)

The HF launcher first loaded the model with `torch_dtype=bfloat16`. With that, HF's Trainer trains in
**pure bf16** — weights and AdamW states in bf16, no fp32 master copy — while Megatron's `--bf16`
with the distributed optimizer keeps **fp32 main params and fp32 Adam state**. A different training
recipe. Every parity check passed anyway, because step-1 loss is a forward on identical weights and
cannot see the optimizer: 1.68850 pure-bf16 vs 1.68960 fp32-master.

Fixed: weights load in fp32 and `bf16=True` autocasts the compute. Cost: useful throughput
10,396 → **9,367** tok/s/GPU (−9.9%) at 0.5B.

**The second defect is the one worth remembering.** The switch was made in launcher CODE, and argv
did not change — so the two legs carried identical `config_digest` (`325fbe030fd6f6b7`) and identical
`impl_digest`. The "digest reads argv, not the realized config" limitation recorded long ago,
actually biting, on the very first stack where the recipe lives in code rather than flags. The
launcher now states the choice as `--master-weights fp32|bf16`, captured as workload identity
(legs since: `fbb87a9ee13c16db`).

Rule for any future launcher: **every recipe choice must appear in argv**, or identity cannot see
it. HF legs digested `325fbe030fd6f6b7` are ambiguous between the two recipes and must not be paired
across that boundary; `hf_smoke_1009` and `hf_null_b` are both pure bf16 and remain a valid null pair.

### Cross-stack at 7B, matched recipe (2026-10-09)

Both at dropout 0, fp32 master weights, same day, 8 GPUs, 30 iters:

| | useful tok/s/GPU | nominal | pad | step median | peak mem |
|---|---|---|---|---|---|
| HF TRL, FSDP full_shard | **1,098.6** | 1,098.6 | 0% | 8.80 s | 73.6 GB |
| Megatron | 893.7 | **2,948.9** | 69.7% | 11.11 s | 79.9 GB |

At 7B HF leads by **+22.9%** — not the 2.0x of 0.5B. Megatron processes a token **2.68x faster**
than HF; at this size the compute efficiency nearly pays for the padding, but not quite. Two things
follow.

**The packing ceiling.** If padding were removed at Megatron's present per-token efficiency, useful
throughput would approach its nominal figure: ~2,949 against ~894, **up to ~3.3x**. Every remaining
kernel head is worth ~1%. Nothing else on the board is within an order of magnitude of this.

**Dropout at 7B, roughly:** 893.7 (dropout 0) against 832.9 for a dropout-0.1 reference leg from a
different day — about +7%. Cross-day, so an estimate, not a measurement; the 0.5B same-era figure
was +7.3% for attention dropout alone.

HF needs FSDP at 7B: with fp32 master weights DDP replicates weights, grads and AdamW state and
OOMs at step 3 (183.5 GB allocated of 192 GB).

### PACKING: 3.66x useful throughput at 7B, with a correctness case built from controls (2026-10-09)

Production recipe (dropout 0.1, fp32 master, fused attention), same day, 8 GPUs, 30 iters:

| | useful tok/s/GPU | nominal | pad | step median | peak mem |
|---|---|---|---|---|---|
| unpacked (the reference) | 833.1 | 2,812.1 | 70.4% | 11.65 s | 80.6 GB |
| **packed** (`scripts/pack_sft.py`) | **3,048.2** | 3,103.7 | **1.8%** | **10.56 s** | 80.6 GB |

**3.66x** — above the ~3.3x ceiling estimated from nominal throughput, because the step got FASTER:
variable-length attention costs Σ Lᵢ² rather than a full 4096² causal block. Memory unchanged.

The correctness case, every part measured with its own control:

| check | result | control |
|---|---|---|
| token identity (explicit default system prompt) | 500/500 on both models | the default differs by model and is read from the tokenizer |
| forward equivalence, real 5-conversation pack | +0.057% vs isolated | non-isolated control 7.86% away |
| kernel-level backward (`flash_attn_varlen_func` vs per-segment) | out/dq/dk/dv bit-identical | both within 0.3% of fp32 |
| objective after boundary masking | −0.053% vs unpacked, 4090 = 4094 − 4 loss tokens | unmasked: −0.424% |

What packing changes that is NOT a defect, and must be chosen deliberately: `--global-batch-size`
counts packs, so GBS 64 is now ~212 conversations per optimizer step (3.3x the unpacked 64). To keep
the recipe's samples-per-update, use GBS 16 or 24 packs (multiples of DP 8 × MBS 1); throughput per
token does not depend on it.

Still manual: the orchestrator's ConfigSweep keeps refusing Class B (`pack` is on its refusal list)
until the checks above run automatically inside it.
