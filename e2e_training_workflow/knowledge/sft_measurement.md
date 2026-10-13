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

**Re-measured as a ConfigSweep lever (config_tuner, 2026-10-09 12:13–12:28): it reproduces.** Pack
rebuilt with `pack_sft.py` into a fresh path: **sha256 identical** to the 10:32 artifact (`e4e7be83…`),
so packing is deterministic. `pack_gate.py --pack-index 3` (a pack the gate was not designed on):
**PASS**, 765 conversations token-identical (0 mismatches), 4052 = 4052 loss tokens, forward +0.0046%
vs isolated with the non-isolated control 5.29% away. Then a fresh back-to-back pair, both `ok`/clean
on the first attempt, GBS 64 on both legs (the workload descriptor's value, kept per the measurement
invariant), only `train-data-path` differing in the realized config (digest `d1548a7a764bdada` →
`6716c26a104f449d`; impl_digest `49d9009f46b82686` on both):

| leg | useful tok/s/GPU | pad | step median | peak reserved |
|---|---|---|---|---|
| ref `cfgpack1009_ref_1009_121334` | 834.4 | 70.38% | 11,635.7 ms | 80,578 MB |
| cand `cfgpack1009_cand_1009_122110` | **3,049.7** | 1.71% | 10,560.2 ms | 80,602 MB |

`compare_legs_tokens.py` (equal useful tokens, the valid basis for Class B): **+258.1%**.
`compare_legs.py` aligned: +265.1% (flagged WORKLOAD IDENTITY DIFFERS, as it must). L1
(`trajectory_gate.py compare`) refuses with "legs trained a different WORKLOAD" — correct for Class B;
packing's correctness evidence is `pack_gate`, not L1. Diagnosis ceiling from 71.1% padding:
1/(1−0.711) = 3.46x (+246%); measured exceeds it because the varlen step is also 9.2% faster.

**Finalize (2026-10-09 12:32–12:52): the assembled bundle reproduces the packing stack.** Bundle:
`runs/finalize_bundle_1009_pack/`. It contains 0 kernel overlays. `final_patch.diff` is one key of
the realized config, `train-data-path`, with no source change against `86c9857c`. `launch.sh` checks
the source and pack by sha256 and builds the pack only if it is missing. After each leg it checks
that the lever engaged: digest, data path, GBS/SEQ/MBS, pad fraction, and the boundary-mask marker
on every rank. Both invocations ran from `/tmp` and exited 0:

| leg (bundle `launch.sh`) | useful tok/s/GPU | pad | step median | aligned vs config-phase leg |
|---|---|---|---|---|
| `--pack-gate` → `finalize_repro_pack1009_cand` | 3,045.4 | 1.855% | 10,560.2 ms | +0.0427% (window artifact 0.18%) |
| `--ref` → `finalize_repro_pack1009_ref` | 837.5 | 70.22% | 11,653.1 ms | −0.1431% |

The `pack_gate` run that `launch.sh --pack-gate` triggered on pack 3 passed:
- token identity: 765/765 conversations
- boundary targets masked: 4052 = 4052 loss tokens
- forward: 0.0046% from isolated, against a 5.29% control

The same-session paired equal-token gain is **+253.1%**, against +258.1% in the config phase. The
two configurations reproduced to within 0.15%, so the 5-point difference is the comparator window.
At ITERS=30 the 19 steady reference steps hold only 190,708 useful tokens, so the equal-token basis
covers just 5.94 packed steps (31% of the leg), and a few points of window placement is its
resolution. Quote the gain as **~3.6x (+253 to +258%)**, not to a decimal place. A tighter
equal-token figure needs longer reference legs for both Setup and the gate.

**Validate (training_director, 2026-10-09 12:55–13:16): CONFIRMED.** The director took a fresh
back-to-back pair with the bundle's `launch.sh`, which runs `bench_train.sh` with the same workload,
megatron, nproc 8 and 30 iters. Both legs were `ok`/clean on the first attempt, and every engagement
check exited 0. The packed leg had the boundary mask on 8/8 ranks.

| leg | config_digest | useful (own window) | pad | step median | peak |
|---|---|---|---|---|---|
| `validate_td1009_ref_1009_125547` | d1548a7a764bdada | 811.6 (aligned to base leg1: **837.0**) | 71.13% | 11,659.2 ms | 80,578 MB |
| `validate_td1009_cand_1009_125547` | 6716c26a104f449d | **3,042.3** | 1.79% | 10,577.7 ms | 80,602 MB |

- **A headline-only trap.** The ref leg's own useful figure is 811.6, which is −3.3% from baseline leg1
  (839.2). It detected a 2-step warmup where the baseline detected 6, so its window took in a
  different batch mix (pad 71.1% against 70.2%). `compare_legs.py` aligned against Setup legs 1/2/3
  gives −0.262 / −0.180 / −0.107%, with window artifacts of 2.5–3.0%. The configuration reproduced;
  the summary number did not. Never compare two `useful_tok_s_per_gpu` headlines whose `own_warmup`
  differs; use the aligned comparison. −0.262% is just over the 0.25% floor and inside this run's
  ~0.31% gate band. The L1 per-step envelope (advisory) passes the ref leg against leg1.
- **The packed leg reproduces.** Aligned against the config-phase cand leg it is −0.020% (paired
  −0.056%). Against the finalize cand leg it is −0.200% (paired +0.023%).
- **Gain, from `compare_legs_tokens.py` over 239,144 equal useful tokens.** Against the TRUE
  baseline leg1 it is **+256.75%**. Against legs 2 and 3 it is +260.20% and +260.65%. Against the
  same-session ref it is +258.87%. All of these fall inside the +253 to +258% range from the earlier
  phases, give or take the comparator's few-point window resolution. Only one lever was accepted,
  so there is no sum of per-candidate deltas to dispute.
- **L1.** `trajectory_gate compare` of baseline leg1 against the packed leg refuses with "different
  WORKLOAD", which is correct for Class B. The stand-in is `pack_gate.py` on a pack that no earlier
  phase used, **pack 11** (4 conversations). It PASSED: token identity 765/765, boundary-masked
  3979 = 3979 loss tokens, forward 0.0065% from isolated, against a non-isolated control 6.37% away.
  Evidence: `runs/validate_td1009_packgate/pack_gate_p11.json`.

### Baseline setup 2026-10-09 (`baseline_td1009_leg{1,2,3}`): offered schema-3 legs refused, three fresh legs

**The offered baseline legs could not be reused, and they were not defective.** `stackB0929_ref1`,
`stackB0929_ref2` and `coldstart_ref` passed every per-leg check: `status: ok`, occupancy clean,
`dispersion_reliable`, not profiled or captured, zero overlay lines in the log, and the same
`config_digest` `d1548a7a764bdada`. All three are **`digest_schema` 3**, and every leg measured today
is schema 4. `training_workflow.js:comparable()` refuses any pair whose schema differs, so each
candidate leg this run takes (schema 4) would have been refused against them. Re-summarising cannot
fix this, because `digest_schema` comes from the sidecar's `snapshot_keys` header. The legs also carry
the schema-3 `impl_digest` (`2da763eeb36d03fb`, against `49d9009f46b82686` today). Rule: when an
offered baseline is reused, check its schema against **what bench_train produces today**, not only
against the other offered legs. The three offered legs agree with each other, so that check alone
would have passed them.

Fresh legs, 30 iters, 8 GPUs. All `ok`, clean on the first attempt, `identity_vs_realized` 26 keys
with 0 mismatches, and argv_hash `ed00d2f999a555d9` on all three:

| leg | useful tok/s/GPU | pad | warmup | steady | worst IQR | step median |
|---|---|---|---|---|---|---|
| 1 | 839.2 | 70.22% | 6 | 24 | 0.051% | 11629.8 ms |
| 2 | 834.0 | 70.375% | 5 | 25 | 0.137% | 11640.3 ms |
| 3 | 837.9 | 70.22% | 6 | 24 | **3.085%** | 11648.3 ms |

Throughput null pairs from `compare_legs.py`, aligned / paired: 1v2 −0.0953 / −0.1011%, 1v3
**−0.1549** / **−0.1776%**, 2v3 −0.0596 / −0.054%. The worst aligned value is 0.1549%, under the
0.25% floor, so the floor holds. The gate band `max(floor, 4·se, 2·across_job)` is about 0.31% on
the aligned figure and 0.36% on the paired one, so the run-measured across-job term binds here, not
the floor. Leg 3's 3.085% IQR is the same tail-stall cliff described in §4.11: it is a clean leg, but
the se term will be wide whenever leg 3 is the reference.

Trajectory band (`baseline_td1009_band.json`, 3 legs, holdout `tdval1009_base`: same digest,
schema, impl and argv_hash): the in-sample null control **passed** (0/3 pairs), and the
**out-of-sample control FAILED**. Against the holdout, grad_norm exceeded the envelope at steps 13, 20,
22, 23, 26 and 30 (up to 5.9x) and loss reached 1.28x. So `envelope_discriminating: false` and the
envelope is ADVISORY. Step-1 loss was bit-identical across all legs (0.0%), and step-1 grad_norm
varied 0.314%. An out-of-sample holdout falsifies a three-leg band here, as it did twice on 2026-09-24
(`trajectory_gate._out_of_sample_control`). The status was not hand-recalibrated. To get a voting envelope, calibrate
on at least 5 legs.

### The workflow accepted packing on its own (2026-10-09, run wf_01fab37a-6da)

First end-to-end deep run with `config_levers: "packing"`. 7 agents, 0 errors, 111 minutes. The
Config Tuner ran the whole procedure itself -- packed the data, gated it on a fresh pack (index 3:
765 conversations token-identical, 0.0046% from the isolated expectation, control 5.29% wide),
measured both legs, compared on equal useful tokens -- and the orchestrator accepted it:
**`win`, +258.11% > 5%, packing gate PASS.**

Validate re-measured through the bundle's own `launch.sh`: **3,042.3 vs 839.2 useful tok/s/GPU,
+256.75%** (+256.8 / +260.2 / +260.7% against the three baseline legs, +258.9% against a same-session
reference), and re-ran the gate on yet another pack (index 11). It noted, correctly, that the
trajectory gate refuses this pair as "a different workload" -- step i is a different batch -- and
that the packing gate is what stands in its place. Boundary masking confirmed on 8/8 ranks.

Independent of the manual measurement, the number reproduced: manual 3.66x, workflow 3.63–3.66x.

The bundle records the one realized-config difference (`--train-data-path`), the packed file's
sha256, and refuses to overwrite a pack whose hash does not match. The Training Director dropped the
three v3 baselines and re-measured, for reasons that turned out to be a mistake of mine (DECISIONS
4.10b). The result now also carries `accepted_config` at the top level, because `accepted: []`
read as "nothing was accepted".

### The accepted kernel stack does NOT survive packing (2026-10-10)

`measure_stack.sh --data <packed>`: the +12.36% stack (lmhead_ce_unmasked_gather + mlp_fc2_down),
both arms on the packed dataset, 2 reps each, engagement re-proved on all 8 ranks
(1,792 / 107,520 fwd = bwd per overlay):

| pair | paired delta | band |
|---|---|---|
| ref1 vs cand1 | +0.170% | 0.250% |
| ref2 vs cand2 | +0.076% | 0.478% |
| ref1 vs cand2 | +0.277% | 0.460% |
| ref2 vs cand1 | +0.111% | 0.258% |
| ref1 vs ref2 (null) | +0.067% | 0.250% |

**Every pair is `no_win`.** On padded data the same stack measured +12.81% / +12.25%. The wins were
padding wins: the unmasked-gather LM head skips loss-masked rows, fc2-down slices its backward GEMMs
to the live-row prefix. Packing removes the rows they skipped, and the +258% from packing absorbs
them entirely. **Every kernel result on this workload to date was harvesting padding.**

**The packed profile** (7B, ranks 0+4, steps 10–12): compute 93.5%, exposed comm 3.7% (0%
overlapped), idle 2.8%; GEMMs ~78% of the span, attention backward down to 6.3% (from 14.5%
unpacked — varlen attention costs Σ Lᵢ²), SwiGLU elementwise 3.0%. The previous queue
(`mlp_fc1_rowslice` +14.7%, `attn_fmha_true_seqlen` +8.9%) was ranked on dead rows that no longer
exist. What is left is dense GEMM efficiency, where re-selecting hipBLASLt solutions has measured
~1.07x isolated — so each head is now worth ~1–2% end to end. Kernel work must be ranked on THIS
profile from here on.

A false positive found in the new realized-identity check on these legs: eight ranks share one
stdout, and a dump line was spliced by another rank's tag (`4[geak-probe] installed (rank 4/8)`), so
a clean leg read as `captured 4, realized 4[geak-probe]...` and would have been refused. Tags are
now stripped before comparing; over all 237 legs, 2 had a spliced line and only the one genuine
mismatch is still flagged. Legs summarised before the fix carry the false entry in their
`bench_summary.json` until re-summarised.

### Packing and the recipe: what preserving samples-per-update costs (2026-10-10)

After packing `--global-batch-size` counts packs (3.31 conversations each). The unpacked recipe runs
64 conversations per optimizer step. Packed legs at three GBS, 7B, 30 iters, all ok and clean:

| GBS (packs) | conversations / step | useful tok/s/GPU | vs GBS 64 | vs unpacked (839.2) |
|---|---|---|---|---|
| 64 | ~212 | 3,052 | — | 3.64x |
| 24 | ~79 | 2,941 | −3.6% | **3.50x** |
| 16 | ~53 | 2,875 | −5.8% | **3.43x** |

The original 64 conversations per step sits between GBS 16 and 24. Keeping the recipe costs 4–6% of
the packing gain -- fewer micro-batches amortise each optimizer step less -- and leaves 3.4–3.5x.

**Not decided here; a recipe decision for the workload owner.** Recommendation, stated as one: GBS 16
(53 conversations, the closer of the two to the original 64 without exceeding it) or GBS 24 (79,
slightly above) if a larger effective batch is acceptable; GBS 64 only if a 3.3x larger batch is a
deliberate change to the recipe. Learning rate was tuned for 64 conversations; a larger batch usually
wants it revisited.

### Kernel work on the packed stack: the first win is real GEMM efficiency (2026-10-10)

Fast-mode run `wf_3593b41a-adb` on `qwen2_5_math_7b_packed.env` (36 agents, 0 errors, 2.9 h after a
directory-scope fix). One head through the lane: `mlp_fc1_gemm` -- exact-shape hipBLASLt solution
selection for fc1 forward/dgrad/wgrad plus an overlapped wgrad. Throughput **+1.601% paired** against
a 0.25% band; L0 exactly as accurate as the incumbent on all four outputs. The run rejected it on the
step-1 gradient probe (+64%); a control showed any accuracy-neutral 1-ulp fc1 forward solution moves
that number 44–64% on this workload, so the probe was retired from voting (DECISIONS 3.4h) and the
candidate re-gated to PASS. On padded data the kernel wins were padding wins; this one is not.

The same run's profile also re-ranks the config levers for the packed recipe: at GBS 16 (2
micro-batches per step on DP 8) the DP grad all-reduce is **11.6% exposed, 0% overlapped** -- one
~300 ms collective per step that no kernel can hide. At GBS 64 unpacked it was 2–3% and comm overlap
measured no win; at GBS 16 packed it is the largest remaining lever. Next to measure.

### Comm overlap on the packed recipe: still no win, and an L1 defect it exposed (2026-10-10)

`--overlap-grad-reduce --overlap-param-gather` on `qwen2_5_math_7b_packed.env` (GBS 16), two
candidate legs against the two null legs: paired **−0.76 to −0.96%** against bands of 0.88–0.96%,
+778 MB. The profile's 11.6% exposed all-reduce is not recoverable this way here: with 2
micro-batches per step the bucketed reduce can only overlap the last micro-batch's backward, and on
this stack the overlapped collective appears to slow the compute it overlaps. Not a lever on this
recipe either; recorded so it is not re-run.

**The defect it exposed is worth more than the measurement.** L1 refused both legs as "a different
WORKLOAD". `trajectory_gate.py` kept a hand-maintained SECOND COPY of the implementation-key set,
and v4 had updated `probe_summarize` and not it -- so every overlap, recompute and hf-fsdp lever read
as a workload change to L1. Combined with the rule that config levers must pass L1, every such lever
was unacceptable by construction. The copy is gone: `IMPL_KEYS` is a module-level definition in
`probe_summarize.py` that `trajectory_gate.py` imports, and `test_identity_keys.py` fails if a
literal implementation set reappears. With it, both overlap legs pass L1 (step-1 loss identical) and
the implementation delta is reported correctly (it iterated only the reference's keys and showed
`{}` for flags only the candidate passes). Digests on disk unchanged.

### Two kernels accepted on the packed stack: +7.19% validated (2026-10-10, run wf_cc06b402-312)

Fast mode on `qwen2_5_math_7b_packed.env`, fc1 skipped (measured the run before). 71 agents, 0
errors, 5.8 h. Two heads through the lane, both accepted:

| head | gate delta (paired) | L0 | L1 |
|---|---|---|---|
| `rope_fused_qk` | **+3.89%** vs band 0.38% | PASS; forward ~3x MORE accurate than the original (fp32 throughout, one rounding) | PASS; step-1 loss +0.044% (kernel_swap limit 2%), grad_norm +13.9% reported only |
| `swiglu_bwd_fwd_bw` | **+3.15%** vs band 0.25% | PASS; errors equal to the original's | PASS |

**Validate: 2,874.2 -> 3,081.8 useful tok/s/GPU, +7.19%, confirmed**, and the assembled bundle
(`runs/finalize_bundle_1010_rope_swiglu`) engages both overlays forward and backward on 8/8 ranks
for 30/30 steps with no fallback. These are compute-efficiency wins, not padding wins -- the first
of their kind on this workload apart from fc1 (+1.60%, re-gated, not yet in a bundle).

Cumulative against the original unpacked reference (839.2), at the recipe-preserving GBS 16:
**3,081.8 / 839.2 = 3.67x**.

The retired step-1 gradient probe reported +13.9% on rope and did not vote; at the old 10% limit it
would have rejected a kernel that is three times more accurate than what it replaces.

Remaining queue on this profile: `mlp_fc1_wgrad` (14.2% share, +1.52% expected), `mlp_fc2_gemm`
(20.4%), and fc1 to be stacked with the two accepted overlays.

### fc1 stacks on top of rope + swiglu: three kernels, +9.1% (2026-10-13)

`scripts/measure_stack3.sh`, packed 7B descriptor, three conditions interleaved, 2 reps (legs
`runs/stack3_1013_031326_*`): ref (no overlay), s2 (rope + swiglu, the accepted bundle's bytes,
sha256-verified), s3 (rope + swiglu + fc1, fc1's kernel bytes the ones L0 judged). All six legs ok,
clean, `identity_vs_realized` empty. Engagement exact on every candidate leg, 8/8 ranks, 0 fallback:
rope 26,880 = 8 x 30 x 112, swiglu and fc1 13,440 = 8 x 30 x 56 each.

| leg | ref | s2 (rope+swiglu) | s3 (+fc1) |
|---|---|---|---|
| rep 1 useful tok/s/GPU | 2,874.8 | 3,080.4 | 3,131.6 |
| rep 2 | 2,868.9 | 3,082.9 | 3,132.8 |

* s2 vs ref: +7.15% / +7.46% -- reproduces the +7.19% Validate measured on 10-10.
* **fc1's increment, s3 vs s2: +1.66% / +1.62% (cross pairs +1.70% / +1.58%)** -- it keeps its whole
  stand-alone +1.60% when stacked; nothing overlaps.
* **s3 vs ref: +8.93% / +9.20%, ~+9.1%.** Null pairs within condition: −0.21%, +0.08%, +0.04%.
* Cumulative against the original unpacked reference (839.2) at GBS 16: **3,132 / 839.2 = 3.73x**.

(Ratios here are of per-leg headlines. On the packed stack every step has the same shape, so they
track the paired estimator closely -- for fc1 alone 1.514% per-leg vs 1.601% paired.)

Stacking needed one change outside the gated kernel bytes: rope's and fc1's `impl.py` both wrote
their per-call proof next to the FIRST manifest in `GEAK_OVERLAY`, so stacked they would have
overwritten each other's `engagement_fused.json`. `measure_stack3.sh` patches only fc1's impl.py to
locate its own manifest (as swiglu already does); fc1's proof landed in its own directory on every
leg, and `kernel_src/` was copied verbatim (sha256 printed in the log). A bundle that ships fc1 with
rope must carry that patch. Per-leg `impl_digest` values differ between reps because the manifest
basenames differ per leg (they are part of impl_digest), not because the implementation does.

### Five kernels stacked: +13.8% on the packed reference, 3.89x cumulative (2026-10-13)

**The run.** Fast-mode run `wf_6d27fb7e-e69`, packed 7B, skipping the three accepted heads, 2 gates
in 5.0 h, both heads measured against the BARE packed reference:

* `lmhead_ce_dense_gemm` (dense lm-head GEMM + fused CE, fp32 `addmm` wgrad into `main_grad`):
  **accepted**. Gate +2.25% paired, band 0.40%. Director-validated +2.27%. Peak reserved memory
  −4.65%. L0 is equal to the original on every max_abs. L1 passes as both kernel_swap and overlay.
  The audit's apparent late-loss drift is within the spread of 15 clean same-identity refs: the
  candidate's late-window mean has z = +0.52, rank 10/16. The drift came from that session's ref
  leg being low, not from the candidate.
* `mlp_fc2_oproj_gemm` (hipBLASLt solution reselection for fc2 and o_proj, fwd/dgrad/wgrad):
  **inconclusive_noise**, not no_win. +1.78% paired sat inside a 2.01% band, which was 8x the floor
  because both legs were noisy (IQR 1.7% / 2.6%). L0 and L1 passed. The verdict asked for a quiet
  re-measure.
* Not reached (task budget): `lmhead_wgrad_reselect` (+1.51% expected), `attn_fmha_ck_varlen`
  (+0.98%), `qkv_lnlinear_gemm` (+0.39%).

**The stacked re-measure.**
* Script: `scripts/measure_packed_stack.sh --conds ref,s3,s4,s5 --reps 3`. It generalizes
  `measure_stack3.sh`, and conditions are cumulative.
* Conditions: ref; s3 = rope + swiglu + fc1; s4 = s3 + lmhead; s5 = s4 + fc2_oproj.
* 12 legs interleaved, legs `runs/pstack_1013_111120_*`.
* All legs ok and clean.
* Engagement is exact on every s5 leg, 8/8 ranks, with 0 fallback and 0 delegation. Per rank:
  rope 3,360, swiglu and fc1 1,680, lmhead 60, fc2 and o_proj 1,680 each.

| useful tok/s/GPU | ref | s3 | s4 (+lmhead) | s5 (+fc2_oproj) |
|---|---|---|---|---|
| rep 1 | 2,866.3 | 3,131.4 | 3,207.2 | 3,261.6 |
| rep 2 | 2,870.2 | 3,124.8 | 3,206.5 | 3,266.6 |
| rep 3 | 2,872.7 | 3,129.4 | 3,213.6 | 3,265.1 |
| peak reserved MB | 80,598 | 80,892 | 77,178 | 77,472 |

Paired per-step ratios (`compare_legs.py`, same batches, per rep; IQR is the per-step ratio's):

| increment | rep 1 | rep 2 | rep 3 | ratio IQR |
|---|---|---|---|---|
| s3 vs ref (rope+swiglu+fc1) | +9.10% | +9.05% | +9.06% | 1.1–1.5% |
| **s4 vs s3 (lmhead)** | **+2.51%** | **+2.49%** | **+2.55%** | 0.18–0.61% |
| **s5 vs s4 (fc2_oproj)** | **+1.74%** | **+1.76%** | **+1.72%** | 0.13–0.18% |
| **s5 vs ref (all five)** | +13.80% | +13.90% | +13.79% | 0.7–1.4% |

**Reading.**
* s3 reproduces the stack3 measurement (+9.1%).
* lmhead adds slightly MORE on the stack (+2.5%) than against the bare reference (+2.27%). The rest
  of the step got shorter, so its fixed saving is a larger fraction.
* **fc2_oproj's quiet re-measure resolves it as a win.** On the stack it gives +1.74% with three
  reps within 0.04% of each other and a per-step ratio IQR of 0.13–0.18%. That is 7x the 0.25% floor
  and nowhere near its noise band. The workflow's `inconclusive_noise` (rather than no_win) was the
  right call.
* L1 (`trajectory_gate.py --kind kernel_swap`, packed band `baseline_packed7b_bands_1013`) passes
  s4 vs s3 and s5 vs s4 on all three reps. That is in addition to each head's own L0/L1 against the
  bare reference.
* lmhead's memory saving carries over: −3.7 GB reserved per GPU at s4. fc2_oproj costs +294 MB.
* **Cumulative: 2,869.7 → 3,264.4 useful tok/s/GPU = +13.75% per-leg (+13.8% paired) over the packed
  reference, and 3,264.4 / 839.2 = 3.89x the original unpacked reference** at GBS 16.

**Coexistence.**
* Targets are disjoint:
  * rope: `apply_rotary_pos_emb`
  * swiglu: `bias_swiglu_impl`
  * fc1: TE `_LayerNormLinear`
  * lmhead: `GPTModel._postprocess`
  * fc2/o_proj: TE `_Linear`
* Each overlay writes its proof next to its own manifest. lmhead and fc2_oproj find their manifest
  by their directory name, so they need no patch. fc1 still needs the patch from stack3.
* fc1 and fc2_oproj both ship modules named `gemm_select` / `lt_gemm.cpp`:
  * fc2_oproj pins its own `gemm_select` into `sys.modules` only while its `kernel.py` loads, then
    restores the previous entry.
  * The C++ extensions have different names (`fc1_lt_gemm` / `fc2oproj_lt_gemm`).
  * A bundle shipping both must keep both properties.

### Seven kernels stacked: +20.7% on the packed reference, 4.12x cumulative (2026-10-13)

**The run.** Fast-mode run `wf_def62c9e-e34`, packed 7B, skipping the five accepted heads and
`lmhead_wgrad_reselect` (same `_postprocess` site as the accepted lmhead overlay, which already does
its own fp32 wgrad). Queue exhausted in 3.9 h:

* `attn_fmha_ck_varlen` (CK varlen FMHA fwd+bwd with the incumbent's dropout mask reproduced):
  **accepted**.
  * Gate +3.55% paired, band 2.38%. Director-validated **+4.27%** (supersedes the gate figure).
  * L0: the backward is numerically the incumbent's (dq/dk max_abs identical to 8 digits); the
    forward is not bit-identical.
  * L1 passes as kernel_swap.
  * Bundle `runs/finalize_bundle_1013_attn_ck_varlen`.
* `qkv_lnlinear_gemm` (QKV LayerNormLinear GEMMs + triton RMSNorm/dbias): **inconclusive_noise**.
  +0.88% sat in a 1.22% band set by leg dispersion. L0 and L1 pass.

**The stacked re-measure.**
* `measure_packed_stack.sh --conds ref,s5,s6,s7 --reps 3`: s6 = s5 + attn, s7 = s6 + qkv.
* 12 legs `runs/pstack_1013_154204_*`, all ok and clean.
* Engagement is exact on every s7 leg, 8/8 ranks, 0 fallback:
  * attn 1,680 fwd/bwd.
  * qkv 1,680, delegating the 1,680 MLP calls down the chain to fc1.
  * fc1 1,680, now with 0 delegated: qkv takes the attention call site.

| useful tok/s/GPU | ref | s5 | s6 (+attn) | s7 (+qkv) |
|---|---|---|---|---|
| rep 1 | 2,859.8 | 3,262.0 | 3,434.6 | 3,470.2 |
| rep 2 | 2,871.6 | 3,264.1 | 3,429.8 | 3,439.6 |
| rep 3 | 2,865.2 | 3,263.5 | 3,439.3 | 3,469.2 |
| peak reserved MB | 80,598 | 77,472 | 77,584 | 77,818 |

| increment (paired, per rep) | rep 1 | rep 2 | rep 3 | ratio IQR |
|---|---|---|---|---|
| s5 vs ref | +13.73% | +13.73% | +13.88% | 0.7–1.4% (reproduces the earlier +13.8%) |
| **s6 vs s5 (attn)** | **+4.56%** | **+5.49%** | **+5.53%** | 2.1–2.8% |
| **s7 vs s6 (qkv)** | **+1.11%** | **+0.99%** | **+1.05%** | 0.18% / 1.72% / 0.18% |
| s7 vs ref (all seven) | +21.24% | +20.27% | +21.55% | 2.9–3.9% |

**Reading.**
* **attn adds +5.25% per leg on the stack, more than its bare +4.27%.** The step is shorter, so the
  saving is a larger fraction.
  * Its per-step ratio IQR is wide (2–3%) because attention cost follows each step's pack
    composition: a varlen kernel's saving is not a constant per step.
  * The three per-leg ratios still agree within 0.3%.
* **qkv resolves as a win: ~+1.05%.**
  * On the two quiet reps the per-step IQR is 0.18%, 4x clear of the 0.25% floor.
  * Rep 2's s7 leg was noisy (IQR 1.95%; per-leg ratio only +0.28%), but its paired estimate still
    reads +0.99%. The paired estimator is what the gate uses.
* **L1.**
  * s6 vs s5 passes on all three same-rep pairs.
  * s7 vs s6 passes on reps 1 and 2. Rep 3 FAILS on the strict early-loss sign rule (8/8 one-sided).
  * The 9 cross pairs show it is the s63 leg: both failing cross pairs involve s63. No systematic
    loss shift exists between s6 and s7 in the early window, late window, or final loss.
  * Read as a false reject; the analysis is in `gate_convergence.md`, "Stacked correct kernels".
* **Cumulative:** 2,865.5 -> 3,459.7 useful tok/s/GPU.
  * **+20.7%** per-leg, +21.0% paired, over the packed reference.
  * **3,459.7 / 839.2 = 4.12x** the original unpacked reference at GBS 16.
  * Peak reserved memory is 2.7 GB/GPU below the reference.

**Coexistence.**
* qkv and fc1 both rebind TE `_LayerNormLinear.apply`. qkv was written to chain: at import it
  captures the current binding as `_PREV` and delegates non-QKV calls to it.
* So the order in `GEAK_OVERLAY` matters: fc1 must come before qkv, and `measure_packed_stack.sh`
  fixes that order.
* qkv pins its four helper modules (`gemm_select`, `bwd_overlap`, `rmsnorm_triton`, `dbias_triton`)
  only while it loads. Its extension is `qkv_lt_gemm`.
* attn compiles its CK instances under its own `build/ckfa`.

No combined seven-kernel bundle exists; `measure_packed_stack.sh` is the reproducible recipe.
