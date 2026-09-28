# The correctness gate: L0 + L1 (L2 is opt-in)

Inference gates on byte-identical output. Training has no output, and its failure mode is slower and
nastier: a wrong backward, a systematically biased kernel, or an overlay that engaged on only SOME
ranks looks healthy for tens of steps and destroys convergence thousands of steps later.

## L0 — op level (`scripts/l0_gate.py`)
The oracle covers `out`, **`dgrad` and `wgrad`** against an INDEPENDENT fp32 reference (not a replay
of the fused op — the captured call carries runtime handles that cannot be rebuilt offline).

**No hand-picked tolerance.** "Is rtol=1e-2 ok?" has no defensible answer. Instead the capture
records the error the ORIGINAL implementation already carries against the fp32 reference, and a
candidate is acceptable iff it is **no further from the reference than the original is** (x1.25).
Nothing to argue about, and no penalty for rounding the baseline also does.

**Check the sign distribution, not only the magnitude.** A small systematic bias passes every
magnitude check and still destroys convergence. Measured baseline profile on TE `_LayerNormLinear`:
`out` rel 0.196%, sign_balance 0.489 — unbiased, as a healthy bf16 op should be.

**Backward is enforceable.** Measured baseline profile on TE `_LayerNormLinear`: `dgrad` 2.69%,
`dln_weight` 3.85%, `wgrad` 3.85%, all with sign balance ~0.5.

Getting a valid backward comparison requires one specific thing: **re-run the original op in
isolation, in the live process, on the frozen inputs**. Every cheaper route fails, and fails
plausibly:
* `register_hook` on an input tensor returns the SUM over all consumers (the hidden state also feeds
  the residual add), not this op's contribution;
* intercepting the op's own `backward` cannot be paired to the captured forward — backward runs in
  reverse and one `Function` class serves several call sites (TE's `_LayerNormLinear` is both the
  QKV projection and the MLP's first layer). Mis-pairing shows up as `rel_mean_abs` near exactly
  1.0, the signature of comparing uncorrelated tensors;
* pairing those call sites by shape does not discriminate: they differ only in the weight, whose
  gradient is absent under `gradient_accumulation_fusion`.

Under grad-accum fusion, attach a zeroed `main_grad` buffer to the cloned weight: TE dereferences it
unconditionally, and afterwards it holds exactly this op's weight-gradient contribution.

## L1 — trajectory level (`scripts/trajectory_gate.py`)
**Training is chaotic. This is measured, not theoretical.** Two runs at identical config on identical
data (useful-token sequences match step for step) still diverge:

| step | loss delta | grad_norm delta |
|---|---|---|
| 1 | **0.0 — bit-identical** | **0.87%** |
| 5 | 0.76% | 22.7% |
| 12 | 4.40% | 40.2% |
| 21 | **2.04 absolute** (loss is ~3) | — |

Two consequences:

1. **A scalar aggregate over a long window is the wrong instrument.** Calibrating `max |dloss|` over
   25 steps yields a limit of **6.1 on a loss of 3** — a gate that passes any bug. The band must be a
   **function of step index**: the candidate's drift at step i against how far a baseline PAIR had
   drifted by that same step.
2. **Step 1 is the sharpest instrument available** — a pure forward on identical weights, so its loss
   is bit-reproducible: a zero-noise probe of the forward path. `grad_norm` at step 1 is NOT
   reproducible (0.87%, backward atomics / reduction order), so backward nondeterminism is calibrated
   rather than assumed away.

Three independent detectors:
- **step-1 forward probe** — zero natural noise.
- **per-step envelope** — fails on a *pattern* of exceedances, not one chaotic step.
- **early-window sign balance** — a one-sided delta is a systematic bias even when every magnitude
  check passes (same doctrine as the L0 sign rule).

Validated against a synthetic +0.3% systematic loss bias: all three fire, while the real baseline
pair passes.

**The band is measured, never assumed.** `trajectory_gate.py calibrate` needs >= 2 independent
baseline runs at identical config. Re-calibrate per workload: divergence rates differ with model
size and parallel mapping.

**The band must match what the candidate IS.** Measured: same kernels give a step-1 delta of exactly
0 (bit-identical at 0.5B and 7B alike); two attention backends give **0.2917%**. So a band calibrated
from a same-kernel pair rejects any kernel swap out of hand. Pass `candidate_kind`:

| kind | meaning | step-1 floor |
|---|---|---|
| `overlay` | should not change numerics at all | 0.05% |
| `kernel_swap` | different implementation, same mathematics | 2.0% (`STEP1_FLOOR_PCT` in trajectory_gate.py; an UPPER bound — the fp32 reference makes it loose for a bf16→bf16 swap) |

**RESOLVED, and the earlier diagnosis here was wrong.** This section used to say the envelope
"over-rejects a legitimate kernel swap" and needed "a band from a KNOWN-GOOD kernel-swap pair". That
prescription would not have worked, because the envelope does not over-reject *swaps* — it rejects
**everything**, including pairs where nothing changed at all. The null control proves it:

| run | null control (same kernel, true delta = 0) | the candidate it was meant to discriminate |
|---|---|---|
| 0.5B | failed every baseline-vs-baseline pair, including the band's **own calibration pair** | — |
| 7B | failed 8 loss / 10 grad_norm steps, worst 36.27× | 8 / 12 steps, worst 37.36× — indistinguishable |

Two causes, both now fixed in `trajectory_gate.py`:
1. **One sample.** `envelope[i]` was `max−min` over the calibration legs, and with two legs that is a
   single `|A−B|` observation that collapses to ~0 wherever those two legs happened to cross. Fixed
   by requiring `MIN_BAND_LEGS = 3` (5 recommended); below it the envelope may not vote.
2. **A borrowed floor.** The zero-width-step floor was `STEP1_REL_FLOOR_PCT` (0.05%) for *every*
   metric — a constant calibrated for step-1 loss, which is bit-reproducible in a healthy stack.
   grad_norm is not: three identical 0.5B runs gave **152.10 / 218.77 / 83.44** at step 2. Fixed by
   `rel_floor_pct`, measured per metric from the calibration legs themselves.

`calibrate()` now runs the null control on itself and stores `envelope_discriminating` /
`envelope_status` in the band file. A band that fails its own control is **advisory**: its numbers are
still reported, but they do not vote, and the blocking detectors remain the step-1 forward probe and
the early-window sign balance. Covered by `scripts/tests/test_trajectory_gate.py`.

### What L1 can and cannot actually detect (measured 2026-09-23, 7B/8GPU, 3 fresh legs)

Injected perturbations into a real leg and re-ran the gate against a band calibrated from three
identical legs (`envelope_discriminating: true`, null control passed on all 3 pairs):

| perturbation | detected? | by what |
|---|---|---|
| step-1 loss +0.2%, kind `overlay` | **yes** | step-1 forward probe (limit 0.05%) |
| step-1 loss +0.2%, kind `kernel_swap` | no | 2.0% limit is a deliberate upper bound |
| step-1 loss +5%, either kind | **yes** | step-1 forward probe |
| loss +25% across the whole STEADY window, step 1 clean | **NO** | nothing |
| grad_norm ×1.5 across the steady window | no | nothing |
| grad_norm ×3 across the steady window | **yes** | per-step envelope |

Read that fourth row carefully. **The per-step envelope cannot see a 25% loss shift**, because by
step 12–30 three *identical* legs already disagree by ~8.9% (median relative) and the envelope allows
`safety × that`. This is chaotic divergence, not a defect: it is the same physics that made the
scalar band useless (a 6.1 limit on a loss of 3). The envelope is a **coarse net for gross (≈3×)
deviations only.**

The discriminating detector is the **step-1 forward probe**, which has zero natural noise (these three
legs agree to `step1_loss_rel_pct = 0.0`). A real numerics change moves step 1, which is why the
fourth row is pessimistic about practice — it holds step 1 artificially clean.

Two consequences:
1. **Prefer `candidate_kind: overlay`** whenever the candidate is *supposed* to be numerics-neutral.
   Its 0.05% limit is 40× tighter than `kernel_swap`'s 2.0%, and 2.0% is an acknowledged upper bound
   inherited from an fp32 reference, not a measurement of what a swap costs.
2. **L1 is not convergence evidence.** Nothing above can see slow convergence damage, so for a
   `risk_class: backward` candidate the L2 soak is not optional decoration — it is the only
   instrument that looks at the thing L1 is named after.

A note on the floor, since it is easy to get backwards in both directions: `rel_floor_pct` must be a
LOW quantile (p10: loss 0.31%, grad_norm 30.9%) of the per-step spread. Set to the MEDIAN (loss 8.93%,
grad_norm 152.3%) it swamps the envelope and even the ×3 grad_norm case passes; set to the old
borrowed 0.05% constant it rejects everything including null controls. Both failures were observed.

### L2 soak — specification (PRE-REGISTERED 2026-09-23, before any soak leg was run)

Until now "L2 soak" was a name in this repo with no spec, no threshold and no tooling. Tooling is
`scripts/l2_soak.sh`. The criteria below were written down **before** the first leg started, so that
a soak cannot be turned into a story about whatever the data happened to show.

**Design.** 5 legs at 300 steps (~1 epoch: 20000 samples / GBS 64 = 312.5 steps), run **interleaved**
`ref1, cand, ref2, pos, ref3` so that node drift over ~5 h cannot masquerade as a candidate effect.
A 30-step gate leg invokes the swapped kernel ~224 times; 300 steps invokes it ~19200 times.

**The two controls decide whether the soak may speak at all.**
* **Null control (leave-one-out):** treat each `ref` leg in turn as if it were the candidate, judged
  against the other two. All four criteria must PASS. A criterion that rejects a *clean* leg is
  measuring chaos, exactly as the L1 envelope did — if any fail, the soak is **advisory**.
* **Positive control:** the `pos` leg runs the same overlay with a known-bad kernel (the
  `× 1.004` multiplicative bias from `negtest/lmhead_ce_neg`). It **must FAIL** at least one
  criterion. If a known-bad kernel passes, the soak has demonstrated no sensitivity and is
  **advisory** — a passing candidate then means nothing.

**Criteria.** `S` = the max pairwise value of the same statistic among the three `ref` legs, i.e. the
spread of the thing when nothing changed. `K = 3`, the safety factor used elsewhere in this layer.

| # | criterion | fails if |
|---|---|---|
| 1 | **Liveness** | leg does not reach 300 steps, or `status != ok`, or occupancy not clean, or a hang, or (cand/pos) engagement not proven forward AND backward on all 8 ranks with `fallback_fwd == 0` |
| 2 | **Terminal loss level** — mean loss over the last 10% of steps | `\|mean_cand − mean_ref\| > K × S` |
| 3 | **Δloss drift slope** — OLS slope of `loss_cand[i] − loss_ref[i]` over the steady window | `\|slope\| > K × S` — catches slow divergence a level test cannot see |
| 4 | **Full-window sign balance** — fraction of steps with `loss_cand > loss_ref` | outside `0.5 ± K × (max ref-vs-ref deviation from 0.5)` — a one-directional bias that every magnitude test passes |

**What a PASS does and does not mean.** It means: over one epoch, at this configuration, on this
box, the accepted stack's loss trajectory is not distinguishable from the reference's by four
pre-registered statistics whose sensitivity was demonstrated on a known-bad kernel. It does **not**
mean the kernel is numerically correct (that is L0), nor that a longer run, a different dataset, a
different parallel mapping, or multi-node would agree. One epoch is one epoch.

### L2 soak RESULT, 2026-09-23 — the soak is UNDERPOWERED and its verdict is ADVISORY

First execution of the spec above: 5 legs × 300 steps on 7B/8GPU (`l2soak_0923_*`), all
`status ok` / `occupancy clean`, candidate and positive control both engaged on all 8 ranks
(`fwd = bwd = 2400`, `fallback_fwd = 0`).

| control / subject | verdict |
|---|---|
| null control (leave-one-out over 3 refs) | **PASSED** — the criteria do not reject a clean leg |
| positive control (×1.004 gradient bias) | **NOT REJECTED** ← the soak has no demonstrated sensitivity |
| candidate | PASS on all criteria — **which therefore means nothing** |

**Read the second row first.** A kernel with a deliberate 0.4% gradient bias, invoked 2400×/rank
over 300 steps, passed every criterion. The candidate passing the same criteria is not evidence that
it is safe; it is evidence that this instrument cannot tell the two apart.

**The measured power.** Terminal loss level, which was the most sensitive of the three:

```
reference-vs-reference spread  S       = 0.02901
effect of the ×1.004 bias              = 0.03742  = 1.29 × S
effect of the candidate                = 0.02791  = 0.96 × S   (indistinguishable from noise)
threshold demanded by K = 3            = 0.08704
```

Scaling linearly, this design resolves a gradient bias of roughly **0.9%**, not 0.4%.

**Two things that do NOT buy power, both measured rather than assumed:**
1. **More steps within a leg.** Averaging the offset over all 288 steady steps gives 1.12 × S —
   *worse* than the 28-step tail's 1.29 × S. Chaotic divergence is serially correlated: once two legs
   separate they stay separated, so there is no √n to collect. The noise floor of a soak is set by
   run-to-run chaos, not by sampling within a run.
2. **A cleverer statistic on the same legs.** The drift-slope criterion carried no signal at all —
   the positive control's slope (−3.31e-4) is *smaller* in magnitude than the candidate's (−3.73e-4).

Power has to come from **more reference legs**, which is the one place a √n exists (averaging the
chaos itself). Rough order: resolving a 0.4% bias at K=3 needs on the order of 5–9 reference legs,
i.e. 5–9 h of additional 8-GPU time.

**Criterion 4 (`sign_balance`) was VACUOUS as pre-registered** and is withdrawn. Measured
ref-vs-ref spread S = 0.2049, so K=3 sets a limit of 0.6146 — while the statistic's maximum possible
deviation from 0.5 is 0.5. It could not fail under any data. Cause: chaotic trajectories that
separate early stay on one side, so "fraction of steps above the reference" is not a
noise-like quantity and a threshold expressed as a multiple of its own spread is meaningless.
K was **not** retuned after seeing the data — doing so is exactly what pre-registration exists to
prevent, and the vacuity is recorded instead.

**What this leaves.** The lmhead overlay's convergence evidence remains L0 (clean, and the candidate
was *more* accurate than the original on all four outputs) plus the step-1 forward probe (8.2e-05%
against a 2.0% limit). The soak added no evidence in either direction. Do not describe this candidate
as "soak-validated".

### L2 soak spec v2 — PRE-REGISTERED 2026-09-23, before any v2 leg ran

v1 failed its positive control. The redesign below is registered *before* the legs that will judge
it, and it corrects a wrong recommendation made when v1 was reported ("add 5–9 reference legs").

**Why more reference legs cannot work.** The binding constraint is the effect measured in units of
the between-leg spread: σ (between-leg SD of terminal loss, from 3 refs) = **0.01495**, and the
×1.004 bias moves terminal loss by **0.03742 = 2.50 σ**. v1 demanded 3 σ. More reference legs does
not shrink σ — it is a population spread, not an estimation error, and v1's `S = max pairwise` gets
*larger* with more legs. Even with infinitely many references, a **single** candidate leg carries one
full σ of its own, so the threshold tends to K·σ = 0.0448 > 0.0374. It is unreachable by leg count
alone.

**The fix is replicate CANDIDATE legs**, i.e. compare *means* rather than a leg against a mean:

| n_ref | m_cand | SE factor √(1/n+1/m) | t = δ/(σ·SE) | detected at α=0.05 |
|---|---|---|---|---|
| 3 | 1 | 1.155 | 2.17 | no |
| 3 | 2 | 0.913 | 2.74 | no |
| 3 | 3 | 0.816 | 3.07 | yes (thin) |
| **3** | **4** | **0.764** | **3.28** | **yes** |

**v2 design.** n = 3 reference legs (reuse `l2soak_0923_ref{1,2,3}`), m = 4 candidate legs and
m = 4 positive-control legs, all 300 steps, legs interleaved `cand, pos, cand, pos, …` so drift
cannot align with condition.

**v2 statistic and threshold, fixed in advance.** Primary: terminal loss level (mean loss over the
last 10% of steps), two-sample **Welch t-test, two-sided, α = 0.05**. Secondary, reported but not
gating: the same test on mean offset over the full steady window.

α = 0.05 rather than v1's ≈3 σ is a deliberate, stated trade: for a safety gate a false REJECT costs
one candidate and a re-run, while a false ACCEPT ships a kernel that quietly damages training. The
threshold is chosen *before* the data and its adequacy is not asserted — it is **demonstrated by the
positive control**, which must be rejected or v2 is advisory exactly as v1 was.

**Criterion 4 (`sign_balance`) is not carried into v2** — it was vacuous by construction (see above).
Drift slope is demoted to reported-only: on v1 data it carried no signal (the positive control's
slope was *smaller* in magnitude than the candidate's).

### L2 soak v2 RESULT — and the finding that reframes the whole soak

v2 (n=3 ref reused, m=4 cand, m=4 pos, 300 steps; 6 NEW legs, ~6 h, ~11 h across v1+v2) **also
failed its positive control**,
for a different and more informative reason than v1.

**The bias acts on DISPERSION, not location.** Terminal loss of the four positive-control legs:
1.41330 / 1.42205 / 1.48753 / 1.48966 — bimodal, range 0.076 ≈ 5σ, while their *mean* (1.45314)
lands almost exactly on the reference mean (1.45072), giving t = 0.109.

| group | n | mean | sd | sd vs ref |
|---|---|---|---|---|
| ref | 3 | 1.45072 | 0.01495 | 1.00× |
| cand | 4 | 1.42807 | 0.00916 | 0.61× |
| pos | 4 | 1.45314 | **0.04111** | **2.75×** (variance ratio 7.6) |

A mean-comparison test is structurally the wrong instrument for this. (It also corrects a reading
made from the single v1 leg: "a 1.004 gradient scale acts like a higher learning rate and descends
faster". With one leg it looked systematic; with four it is clearly dispersion.)

**`step-1 grad_norm` is not a zero-noise probe.** Reference legs: 68.096 / 68.195 / 67.549 — they
differ by up to **0.96%**, against a candidate offset of +0.38% and a positive-control offset of
+0.15%. Unlike step-1 *loss* (bit-reproducible, 0.0%), step-1 grad_norm cannot resolve this.

**THE FINDING. The same ×1.004 gradient bias is caught by L0 in 11.5 seconds end to end:**

```
[L0] dgrad: scale_bias=+3.80e-03  -> FAIL SCALE-BIAS        (tolerance 2e-3)
[L0] wgrad: scale_bias=+4.41e-03  -> FAIL magnitude SCALE-BIAS
correct: false
```

while the accepted candidate passes the same gate cleanly (`scale_bias` −3.42e-04 / +3.70e-04).

| instrument | cost | detects a 0.4% gradient bias? |
|---|---|---|
| **L0 op-level oracle** | **11.5 s** (~40 ms for the comparison) | **YES, decisively** |
| L1 per-step envelope | free | no |
| L1 step-1 probes (loss, grad_norm) | free | no |
| L2 soak v1 (5 legs × 300 steps) | ~5 h | no |
| L2 soak v2 (10 legs × 300 steps) | ~10 h | no |

**So the soak's positive control was the wrong class of defect.** Op-local numerics is precisely what
L0 is for, and L0 is orders of magnitude more sensitive and five orders of magnitude cheaper. A soak
will keep "failing" a control of that class while the system is in fact well defended against it.

**What the soak should actually target**, and what a v3 control must therefore be: defects L0
*cannot* see, because L0 evaluates one frozen microbatch in isolation — cross-step state corruption,
accumulation error that only compounds over thousands of invocations, allocator/memory effects,
non-determinism that averages out in one call but not over an epoch. Until a positive control of
*that* class exists, **do not run another soak**: it costs ~10 h and cannot demonstrate its own
sensitivity.

**Status of the lmhead overlay.** Convergence evidence is L0 (clean, and the candidate is *more*
accurate than the original on all four outputs) plus the bit-reproducible step-1 loss probe
(8.2e-05% against a 2.0% limit). The soak neither adds nor subtracts. Both soak runs are reported as
ADVISORY with their candidate PASS carrying no weight — but note this is a stronger safety position
than it sounds, because the class of defect the soak failed to catch is the class L0 catches best.

### The in-sample null control was FALSIFIED (2026-09-24) — a holdout leg is now required

The control added above runs calibration legs against each other. The band has already been fitted
to exactly those legs, so it is **self-fitting**, and the run of 2026-09-24 falsified it twice in
one session:

| control | result |
|---|---|
| in-sample (calibration legs vs each other) | **0 exceedances** → band self-declared `BLOCKING` |
| out-of-sample: a **fresh** clean reference leg vs each of the 3 band legs | **all 3 FAIL** — 5/5/6 loss exceedances, worst ratios 2.332 / 1.999 / 2.200 |
| out-of-sample: two legs differing only by an **already-accepted** overlay | **FAIL** — 6 loss / 4 grad_norm steps, worst 9.534× |

Those out-of-sample pairs have a true delta of exactly zero (same kernels, same seed, same
`config_digest`, step-1 loss delta 0.0). So `envelope_discriminating: true` was **wrong out of
sample**, and it is the flag that licenses the envelope to reject candidates.

**Fix:** `calibrate()` now takes `--holdout <clean leg not in the calibration set>` and only an
out-of-sample control may license `BLOCKING`. Without a holdout the envelope is **advisory by
default**. The in-sample control is still computed and reported — it is cheap and catches gross
defects — but on its own it may only ever DEMOTE, never license. Passing in-sample is necessary and
not sufficient; this is the second time that distinction has cost a run's worth of confusion.

### `--kind` was silently ignored, applying a 40× tighter floor

`trajectory_gate.py compare` parsed argv by index lookup, so any unrecognised `--flag` was **dropped
silently**. An auditor typed `--candidate-kind kernel_swap`; the real flag is `--kind`, so the call
fell through to the default `overlay` and applied a **0.05%** step-1 floor where **2.0%** was
intended. Both `compare` and `calibrate` now refuse unknown flags, and `--kind` is validated against
`STEP1_FLOOR_PCT`. A gate that silently applies the wrong threshold is worse than one that refuses.

### A band may only judge the config it was calibrated on (2026-09-24)

Found by an auditor running an out-of-sample null pair by hand, after the gate had already returned
a confident `pass=false`. The band was calibrated from 5 legs at `argv_hash ed00d2f999a555d9`, 30
steps; the legs it judged were `argv_hash 977d6aac1123c555`, 60 steps. Its sole firing detector
(`grad_norm`, 6 steps, worst ratio 7.107) then rejected a **true-zero-delta pair 2.5× harder** —
4 steps, worst ratio **17.928** — than it rejected the candidate.

A per-step envelope is fitted to legs of a particular argv **and a particular length**; applied to
anything else it is describing a different process. `calibrate()` now records
`calibrated_argv_hashes`, and `verdict()` demotes the envelope to advisory when either leg is out of
scope — with the mismatch printed, so it cannot be discovered by hand a second time.

Note the interaction with the holdout rule: a band that legitimately earned `BLOCKING` on its own
config still may not vote outside it. Scope and discrimination are separate questions.

### DECISION 2026-09-28 — the per-step envelope is RETIRED from voting

Roadmap W1 asked for a decision rather than a fifth patch: *"either an envelope that passes an
out-of-sample control on ≥3 workloads, or the envelope is formally retired. Do not leave it
half-alive."* The evidence decides it.

**Every time it fired in a real run, a control falsified it. Four for four:**

| run | what the envelope said | what the control said |
|---|---|---|
| 0.5B | fired → rejected the attention candidate | band came from a same-kernel pair, which over-rejects a legitimate swap; the auditor refused to treat it as load-bearing and the **sign test** carried the rejection |
| 7B #2 | fired, 8 loss / 10 grad_norm steps | null control failed **8/10**, worst **36.27×** against the candidate's 37.36× — indistinguishable |
| 7B #3 | fired | a **fresh** clean leg failed **3/3** (5/5/6 steps); separately, two legs differing only by an **already-accepted** overlay failed at worst **9.534×** |
| 7B #4 | fired, 6 steps, worst **7.107** | a **true-zero-delta pair** was rejected **2.5× harder** — worst **17.928** |

**Its ceiling is structural, not a tuning problem.** Measured by injection into a real leg: a **+25%
loss shift across the entire steady window** is invisible to it, and so is `grad_norm ×1.5`. Only a
gross ~3× deviation registers — because by step 12 three *identical* legs already disagree by ~8.9%
and the envelope allows `safety ×` that. The noise floor is chaos, and chaos is not something more
legs or a better statistic can buy down *within* a run.

**It was patched four times** — `MIN_BAND_LEGS ≥ 3`, a p10 floor instead of a borrowed 0.05%
constant, a required out-of-sample holdout, an argv/length scope check. **Every patch fixed a real
defect.** None produced a verdict anyone could rely on. A detector that has never been right does not
get a fifth attempt at voting.

**What changes:** `blocking` is hard-wired `False` in `verdict()` and is not reachable by
configuration. **What does not:** the envelope is still computed and reported — the numbers are real
measurements — and the null / out-of-sample controls still run, now answering *"is this band worth
reading at all?"* rather than *"may it vote?"*.

**What still votes**, and what each has actually caught:

| detector | natural noise | caught, in a real run |
|---|---|---|
| step-1 forward probe | **0.0%** (bit-reproducible) | the attention backward defect (grad_norm −88.50%); the 0.5B attention swap (step-1 loss 0.4274%) |
| early-window sign balance | ~0.25–0.75 spread | carried the 0.5B attention rejection when the envelope was discounted |

**Still open:** the 0.2917% attention-backend pair has still not been separated into "pure numerics"
versus "the flash path does not preserve packing isolation" (the separating test is
packed-vs-unpacked per backend). That is a question about that pair, not about the envelope.

**Workload identity vs implementation.** L1 compares only the workload-identity keys (batch,
sequence, mapping, data). The implementation keys (`attention-backend`, `NVTE_*`) are what a candidate
changes, and requiring them to match refuses every config-level candidate without looking at the
trajectory. `config_digest` covers identity; `impl_digest` covers implementation; the verdict reports
`implementation_delta`.

## L2 — soak
Opt-in, plus auto-triggered for high-risk classes: anything touching backward, the optimizer, a
reduction order, or precision/scaling. Not in the default per-candidate path.

## What can accept, and what can only reject
L0 and L1 can **only reject**. Acceptance requires a throughput delta above the measured band. A
throughput-neutral, memory-saving change is **banked, not accepted** — memory is a currency that only
ConfigSweep (deep mode) can spend.
