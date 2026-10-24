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
| `kernel_swap` | different implementation, same mathematics | **0.25%** since 2026-10-22 (DECISIONS 4.48: 5.7x the largest step-1 delta of 20 correct bf16→bf16 gate pairs, 0.044%); 2.0% only with `--l0-strictly-better` (the fp32-reference, more-accurate case) |

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

### The sign test was falsified the same way (2026-09-28) — now held to a MEASURED range

Retiring the envelope left L1 with two voting detectors. Within one run, the second one failed the
same test the first one did.

`attn_fmha_bwd_retry` was rejected on **early-window loss sign balance = 0.000** — explicitly "the
only voting detector that fired"; the step-1 probe passed at 0.19988% against a 2.0% limit. In the
*same* run, on a different head, the auditor measured what clean pairs score:

| pair | true delta | loss sign balance |
|---|---|---|
| `tdop0928_cand` vs `tdop0928_cand2` | **exactly zero** | **0.000** ← would be rejected |
| `tdop0928_ref` vs `tdop0928_ref2` | exactly zero | 0.750 |

So `sb ∈ {0.0, 1.0} ⇒ systematic bias` was an **assumption about what noise looks like**, and it is
false here. With `EARLY_STEPS = 8` the samples are serially correlated — trajectories separate early
and stay separated — so they are nothing like 8 independent coin flips, and a clean pair reaching
0.0 is not rare.

**Fix, and it is the same discipline the envelope was held to:** `calibrate()` now records the sign
balance its **own clean pairs** produce (`sign_balance_null_range`), and `verdict()` fires only when
the candidate falls **outside** that observed range. Inside it, the one-sided move is reported as an
advisory. With no calibrated range the old rule still applies, and says of itself that it is assumed
rather than measured.

**An unresolved disagreement between layers, recorded rather than smoothed over.** For this
candidate L0 says it is *strictly more accurate than the incumbent on all four outputs*, and that it
satisfies an analytic first-row `dq` invariant **the incumbent fails by 1355.79×**. The auditor was
explicit that the detector "cannot separate *a real correctness fix that makes the model descend
faster* from *a systematic bias that points downhill*, because a one-directional delta toward lower
loss is the signature of both". That is a genuine limit of a sign test, not a tuning problem: the
statistic is blind to whether "downhill" is deserved. **A candidate that fixes a bug currently looks
to L1 exactly like one that introduces one.** Until that is resolved, an L1 sign-test rejection on a
candidate L0 rates as *more* accurate should be read as investigate-and-report.

### THE L0/L1 RULE (written 2026-10-08, before being applied to any candidate)

`attn_fmha_bwd_retry` was rejected on loss sign balance **0.000** — the only voting detector that
fired — while **L0 rated it strictly more accurate than the incumbent on every output**, the
incumbent failing an analytic `dq` invariant by **1355.79×**. Both statements were true, the
framework had no rule, and the verdict therefore depended on which layer was read last.

**The rule.** When L0 rates the candidate strictly more accurate than the incumbent on all outputs,
a one-sided move **toward lower loss** is corroboration rather than evidence of bias, and the sign
test may not reject on it alone.

**The asymmetry is the entire safety argument**, so it is enumerated rather than implied:

| loss direction | L0 verdict | sign test |
|---|---|---|
| **lower** | strictly more accurate | **WAIVED** — the expected shape of a correctness fix |
| **higher** | strictly more accurate | **BLOCKS** — accuracy on one frozen microbatch does not license a trajectory moving the wrong way; unexplained is not safe |
| lower | not strictly better | **BLOCKS** — *this is the dangerous case*: a less accurate op descending faster is exactly what a downhill-pointing bias looks like |
| higher | not strictly better | **BLOCKS** |

A waiver is reported as an advisory with its reason; it is never silent. All four rows are pinned by
tests.

**What this rule did NOT turn out to decide, and the honest version matters.** Implementing it
revealed that the *earlier* fix already covers this candidate: the sign test only blocks outside the
range clean pairs themselves produce, and clean pairs reach **0.0** on both the synthetic data (6
pairs: 0.125 0.25 0.25 0.125 0.125 **0.0**) and the real 7B legs (**0.0** and 0.75 both observed).
So `sb = 0.000` was already unblockable before this rule existed. The L0 rule is a **second,
independent** reason that matters only where sign balance falls *outside* the clean range — narrower
than it first appeared. It is kept because the case it covers is real and will recur; it is not
credited with resolving the candidate that motivated it.

### ConfigSweep needs a different comparison basis, and loses two gates (2026-10-08)

The roadmap called ConfigSweep "wiring a designed seam". Checking first showed it is not. Every
trajectory instrument here rests on one premise, stated in `trajectory_gate.py:38` — *both legs must
run the same seed and the same data order, or step i is not the same batch*. Config levers split on
exactly that line:

| class | levers | step i same batch? | step-1 probe | step-aligned delta |
|---|---|---|---|---|
| **A — changes HOW** | attention-backend, `NVTE_*`, recompute, grad-accum-fusion | yes | valid | valid |
| **B — changes WHAT per step** | **packing**, GBS, MBS, seq-length | **no** | **invalid** | **invalid** |

Class A is already gateable today — those keys live in `IMPL_KEYS`, i.e. the digest treats them as
implementation, not identity. **The valuable part of ConfigSweep is exactly the part the existing
gates cannot judge**: the one lever worth the most here, 70.1% padding, is Class B.

`scripts/compare_legs_tokens.py` supplies the only basis that survives a Class-B change: **time to
consume the same number of useful tokens.** `useful_tokens` counts non-pad, loss-unmasked tokens, so
packing changes how fast they are consumed, not how many exist.

**Measured, including the mistake that nearly shipped.** The first version let each leg use its own
detected warmup — so the two windows started at different places in the data stream and re-acquired
the sample-composition artifact the whole comparator exists to avoid:

| | null control (true delta 0) |
|---|---|
| each leg's own warmup | **1.6 – 2.9%** |
| common warmup (`max` of both), as `compare_legs.py` does | **0.007 – 0.30%** |
| step-aligned comparator, same pairs | 0.035 – 0.071% |

So it is usable for signals above roughly 1%, and it is **not** equivalent to the step-aligned
comparator — about 4–8× noisier at best. Against four known same-data-order pairs it agrees to
+0.004 / −0.330 points on two and diverges by −0.590 / −1.827 on the other two. That divergence is
**semantic, not error**: when a kernel's step time correlates with how many useful tokens the step
holds — the row-slice candidate is exactly that — "per step" and "per useful token" are different
questions with different right answers.

**What no basis restores:** the step-1 forward probe and the per-step envelope compare step i to
step i, so a Class-B config change has **no trajectory gate at all**. That is a safety gap to state
wherever such a result is reported, not to discover afterwards. L0 does not apply either — no kernel
changed. A packing result would rest on throughput alone.

### ConfigSweep ships Class A only, and the refusal is in code (2026-10-08)

`deep` mode now runs. What it may propose is bounded by the class split above, and the bound is
enforced in `training_workflow.js` (`CLASS_B_LEVERS`, `classOfLever`), not only in
`roles/config_tuner.md` — **a rule only a prompt knows is a rule the next prompt can forget.**

A Class-B proposal is **refused with its reason**, never silently downgraded, because a Class-B
result looks exactly like a Class-A one once it is a number in a table.

**The Tier-0 pack, each lever with a ceiling read from the diagnosis rather than guessed:**

| lever | flags | ceiling | source |
|---|---|---|---|
| comm overlap | `--overlap-grad-reduce`, `--overlap-param-gather` | **2.27%** | `exposed_comm_pct`, with `comm_overlap_pct` measured at **0.0** — none of it is currently recovered |
| recompute granularity | `--recompute-granularity` | memory ↔ compute | also **the memory converter**: the knob `fast` banks memory for and cannot spend |
| attention backend | `--attention-backend` | — | a proven-comparable pair (step-1 delta 0.2917% between backends) |

**One deliberate inversion of the kernel rule.** A kernel candidate must keep `config_digest`
identical, and `comparable()` refuses it otherwise. A config candidate is *expected* to move the
digest — that is what makes it a config change — so for this phase the digest difference is
**reported rather than enforced**, and the Tuner must name exactly which key moved. If anything other
than the proposed lever moved, the two legs are not a controlled comparison and the delta is not
attributable.

Everything else is unchanged: same `bench_train.sh`, same window-aligned comparison, same
`gateVerdict`, same `no_aligned_delta` refusal. A config win is gated exactly as a kernel win is.

**Still open:** the 0.2917% attention-backend pair has still not been separated into "pure numerics"
versus "the flash path does not preserve packing isolation" (the separating test is
packed-vs-unpacked per backend). That is a question about that pair, not about the envelope.

**Workload identity vs implementation.** L1 compares only the workload-identity keys (batch,
sequence, mapping, data). The implementation keys (`attention-backend`, `NVTE_*`) are what a candidate
changes, and requiring them to match refuses every config-level candidate without looking at the
trajectory. `config_digest` covers identity; `impl_digest` covers implementation; the verdict reports
`implementation_delta`.

### L2 soak spec v3 — PRE-REGISTERED 2026-10-08, before any v3 leg ran

v1 and v2 both failed their positive control. The diagnosis in DECISIONS 3.6c is that the control
was the **wrong class**: a x1.004 gradient bias is op-local numerics, which L0 rejects in 11.5 s, so
it never reaches the soak. v3 changes the control, and the power arithmetic that should have been
done before v1.

**The control.** `scripts/controls/stale_state_control.py` — cross-step staleness. The overlay keeps
the previous invocation's `gscale` and blends it into the current one:

    gscale_eff[n] = gscale[n] + alpha * (gscale[n-1] - gscale[n])

This is invisible to L0 **by construction, not by luck**. L0 evaluates one frozen microbatch in one
call (`l0_gate.py:8,273`): there is no previous invocation, and on any repeat of the same input the
difference is exactly 0.0, so the blend returns its operand bit-identically — for every alpha,
including 1.0. (The lerp form is load-bearing; `(1-a)*cur + a*prev` leaves ~1 ulp and the claim
would have been false as written. The self-test caught that.) In training, consecutive steps hold
different batches, so from step 2 the optimizer is fed a partly stale gradient, ~19 200 times a leg.
It is also a real bug class, not a straw man: uncleared workspace reuse, and fp8 amax-history
scaling does it deliberately.

**Why v1 could not have worked, arithmetically.** With sigma = 0.01495 (the measured between-leg sd
of terminal loss) the minimum detectable effect is `t_crit * sigma * sqrt(1/n + 1/m)`:

| cand n | ref m | df | t_crit | MDE | vs. the bias's apparent single-leg 0.0374 |
|---|---|---|---|---|---|
| **1** | **3** | 2 | 4.303 | **0.0742** | **no — this was v1, and see below** |
| 2 | 4 | 4 | 2.776 | 0.0359 | barely |
| 3 | 3 | 4 | 2.776 | 0.0339 | yes — but its null control is vacuous, see below |
| **3** | **4** | **5** | **2.571** | **0.0294** | yes — **the v3 design** |
| 5 | 5 | 8 | 2.306 | 0.0218 | yes |

v1 was not mis-thresholded. It was underpowered with a single candidate leg, and no threshold
choice could have rescued it. This table is the thing to consult before spending GPU hours on any
future soak.

**And the effect it was chasing is itself probably zero (measured 2026-10-08).** Re-running the
analyzer over the legs on disk:

| group | legs | terminal loss vs reference mean |
|---|---|---|
| v1 positive (`l2soak_0923_pos2`) | n=1 | **−0.03742** (2.50 sigma) — the number quoted in DECISIONS 3.6b |
| v2 positive (`l2soak_v2_pos{2,3,4}`) | n=3 | **+0.01570** (1.05 sigma), Welch t = 0.659 vs crit 3.180 |

**The sign flips between generations**, and the v1 figure is a single draw from a population with
sigma = 0.01495 — exactly what a 2.5-sigma noise draw looks like at n=1. So the honest statement is
that the x1.004 bias has **no demonstrated location effect on terminal loss at all**, and 0.03742
was never an effect size; it was one leg. Note the direction too: the "known-bad" leg trained to a
*lower* loss than the references.

This strengthens DECISIONS 3.6c rather than weakening it. The soak did not merely lack the power to
see this defect — there was most likely nothing in terminal-loss location to see, because the
defect is op-local and L0 is where it is visible (`dgrad scale_bias +3.80e-03`, 11.5 s). It also
means **the v3 control's effect must itself be measured, not assumed**: if the stale control shows
no location effect either, that is a finding about this defect class, not a licence to loosen a
threshold.

The analyzer now refuses to pretend otherwise: with one positive leg it prints the delta, says
`NO TEST POSSIBLE -- no within-group variance`, and tells the reader to quote it as a single draw.
It also refuses to run at all when any leg is short or empty — pointed at a truncated leg it used
to report a terminal loss of 0.00000 for every leg and an MDE of 0.00000 before dying in a later
division, every one of those numbers looking like a measurement.

**The v3 design, and its cost.** `l2soak_0923_ref{1,2,3}` are reusable: already measured, overlay
off, 300 steps, `snapshot_keys: v3`, all at `config_digest d1548a7a764bdada` — checked, not assumed,
because reusing out-of-scope legs is an error this layer has already made once.

**AMENDED the same day, before any v3 leg ran.** The design was first written as n=3 vs m=3, using
only the free reference legs. That design has a **vacuous null control** and would have shipped one:
with m=3 the held-out split is 1-vs-2, df=1, t_crit=12.71, and nothing the data could do would make
it fire. That is the same defect as the withdrawn `sign_balance` criterion, whose K=3 limit of
0.6146 could not be exceeded by a statistic bounded at 0.5. A control that cannot fire is not a
control, and it is worse than none because it reads as one.

So v3 is **n=3 positive vs m=4 reference**: the three free legs plus ONE new reference leg, which
buys 2-vs-2 holdout splits (df=2, t_crit=4.303 — weak, but able to fire) and tightens the MDE to
0.0294. Cost is 4 new legs, ~4 GPU-hours instead of ~3. The new reference leg must be verified at
`config_digest d1548a7a764bdada` with the overlay off before it joins the group; a 0923 group and a
1008 leg are only the same reference if the identity says so.

Amending a design before the data exists is the process working. Amending a THRESHOLD after seeing
the data is the thing that is forbidden, and nothing below has moved.

**Pre-registered criteria.** Fixed now, before any v3 leg exists:

1. **Location.** Welch t on terminal loss, n=3 vs m=4. Reject at |t| > t_crit(df, 0.05);
   for the pooled df=5 case that is 2.571, i.e. |delta| >= 0.0294. This is the primary criterion.
2. **Dispersion.** F = s2_pos / s2_ref on terminal loss, df (2,2), reject at F > 19.0.
   **Declared weaker on purpose:** this criterion was motivated by watching the v2 bias control,
   which shifted dispersion (sd ratio 2.75, variance ratio 7.6) while the means coincided. It is
   therefore NOT independent evidence if it is the only thing that fires — say so in the result.
3. **Null control.** All 2-vs-2 holdout splits of the four reference legs; criteria 1-2 must reject
   none of them. An in-sample null control was falsified on 2026-09-24 and a holdout has been
   mandatory since. This is the criterion the m=3 design could not have delivered.

**Verdict rule, also fixed now.** The soak is licensed as a gate only if the stale control at some
alpha is REJECTED and the held-out clean leg is NOT. A pass on the candidate alone licenses nothing;
that was the v1 mistake. If alpha = 1.0 is not rejected, the honest report is "the soak has no
demonstrated sensitivity to cross-step staleness at 300 steps / 3 legs", and the soak stays
unlicensed — the thresholds above are **not** to be retuned to make it fire.

**The control must reach BOTH instruments, or "L0 passed" means nothing.** This nearly shipped
wrong. The control was first written to patch only `impl.py`, which is what the soak legs run. L0
runs the extracted task's candidate, `kernel_src/kernel.py`. Had it stayed that way, L0 would have
passed for the uninteresting reason that **it never executed the patched bytes** — coverage, not
structure — and the entire argument would have been circular.

Checking rather than assuming showed the claim is in fact sound, for a reason only reading both
files reveals: the L0 task's `target_callable` in `meta.json` is `GPTModel._postprocess`, the very
hook the overlay replaces, and `kernel.py` carries the same line under a different name —

```
impl.py    gscale    = (g_f.index_select(0, idx) * mg).contiguous()
kernel.py  coef[:n]  =  g_flat.index_select(0, idx) * mg
```

So L0 really does execute this computation, with `num_cases: 1`. "L0 cannot see it" is therefore a
statement about L0's one-call **structure**, which is what makes it interesting. The control now
recognises both sites and refuses a file containing neither. This is the same error — comparing
against a path the change does not lie on — that produced the retracted +8.46% attention figure.

**MEASURED 2026-10-08 — the invisibility claim, demonstrated rather than argued.** Three arms, the
same L0 task dir (`l0_gather0923_gate_0924`), the same harness, and crucially **the same line of
`kernel_src/kernel.py`** as the injection point, so nothing differs between arms but the defect:

| arm | injected at `coef[:n] = g_flat.index_select(0, idx) * mg` | L0 verdict |
|---|---|---|
| 1 | nothing (the accepted candidate) | `correct: true` |
| 2 | **cross-step staleness, alpha = 1.0** (maximal) | **`correct: true`** |
| 3 | x1.004 bias (op-local numerics) | **`correct: false`** — dgrad `scale_bias +3.78e-03`, wgrad `+4.72e-03` |

Arm 2 did not merely pass. Its numbers are **bit-identical to arm 1** — `out max_abs`
0.09031867980957031, `scale_bias` −1.27e-04, `dgrad` 4.172325134277344e-07, every digit the same.
That is the algebra showing up in the instrument: with one frozen input there is no previous
invocation, so the blend returns its operand untouched and L0 computes exactly what it computes
without the defect. At full staleness. There is no magnitude at which L0 begins to see it.

Arm 3 is the contrast that makes arm 2 mean something, and it reproduces DECISIONS 3.6c's figures
from the other injection point (dgrad +3.78e-03 here against +3.80e-03 recorded; wgrad +4.72e-03
against +4.41e-03 — the small difference is the different site, and is reported rather than smoothed
over). Same file, same line, same gate: one defect class is caught decisively and the other is
invisible. **The soak now has a control that is its own to catch.**

**MEASURED 2026-10-08 — the control engages, and L1 catches it at full staleness.** A 30-step leg
at alpha = 1.0: engagement `fwd=224 bwd=224 fallback_fwd=0` on all 8 ranks, training survives, and

| check | alpha = 1.0 | alpha = 0.25 |
|---|---|---|
| L1 verdict | **FAIL** | **pass** |
| `grad_norm` early sign balance | **0.750**, outside the measured null range 0.5–0.625 | 0.625, at the boundary |
| `step1_loss_rel_pct` | 8.2e-05 | 8.2e-05 |

Two consequences, and they point in opposite directions.

**Good for L1.** The sign test has been carried as "cannot tell a fix from a bug" since it was held
to a measured range. It now has a demonstrated positive control **of a class L0 cannot see**, which
is the first one it has ever had. Read the margin honestly though: with n=8 the statistic moves in
steps of 0.125, so 0.750 clears the range by exactly one unit.

`step1_loss_rel_pct` stays at 8.2e-05 — effectively zero — at BOTH alphas, and that is correct
rather than a miss: the staleness is injected in the backward, and the step-1 probe is a forward on
identical weights. The probe is blind to this class by construction, as it should be.

**Awkward for the soak.** At alpha = 1.0 the defect is NOT soak-only, so a soak that rejected it
would demonstrate nothing L1 had not already done for free. The pre-registration said this case had
to be declared, and it is. The soak's niche, if it has one, is the regime where L1 is silent —
**alpha = 0.25 is the largest staleness L1 does not catch**, so that is where the soak must prove
itself, and the four legs are run there and nowhere else.

A possibility worth stating before the data rather than after: that niche may be **empty**. L1 goes
silent at 0.25, and the soak's MDE on terminal loss is 0.0294, which the far cruder x1.004 bias
never came close to reaching. If the soak cannot separate alpha = 0.25 either, the finding is not
"tune it" — it is that a ~10 h instrument has no region of its own at 300 steps, and should be
retired rather than repaired. That outcome is as useful as the other and the thresholds stay fixed
either way.

**AMENDED AGAIN 2026-10-08, still before any v3 positive leg existed.** The design compared stale
legs (overlay **ON** plus the patch) against reference legs (overlay **OFF**), so the overlay itself
sat inside the comparison. The overlay was separately shown not to shift terminal loss detectably
(v2 candidate vs reference, t = −1.99 against crit 2.78), so this was probably harmless — but
"probably harmless" is not a reason to leave a known confound in a 4-hour experiment when removing
it is free.

Four clean **overlay-ON** 300-step legs are already on disk at the same overlay sha `308fc0247b`:
`l2soak_0923_cand1` and `l2soak_v2_cand{2,3,4}`. They are the correct reference group — identical to
the stale legs in every respect except the patch — and m=4 still gives the 2-vs-2 holdout null
control. So no new reference leg is needed at all, and the hour it would have cost buys a **fourth
positive leg** instead: n=4 vs m=4, df=6, t_crit 2.447, **MDE 0.0259**.

Final v3 design: **4 stale legs at alpha=0.25 vs 4 clean overlay-ON legs, MDE 0.0259, ~4 GPU-h.**
The three overlay-OFF legs stay available as a secondary, cross-checking report. Criteria below are
unchanged; only the groups and n moved, and they moved before any v3 leg finished.

**Cheap checks that run BEFORE the 3 hours**, because a control that fails to install reads as a
pass and that has happened here before (`measure_stack.sh` wrote 3528 silent engagement failures):

* `stale_state_control.py --self-test` — the invisibility claim, as arithmetic, no GPU.
* L0 on the patched **`kernel_src/kernel.py`** — must PASS. On the overlay it would prove
  nothing; see above.
* one 30-step leg at alpha = 1.0 — confirms the control engages, and tells us whether L1 already
  catches it. If L1 does, the control is still valid but it is no longer *soak-only*, and the
  result must say which.

### A Class-B trajectory gate is possible: align loss on TOKENS, not steps (designed 2026-10-08)

`roles/config_tuner.md` ships Class A only, and the stated reason is that a Class-B change -- one
that alters WHAT a step contains -- has **no trajectory gate at all**: step i is not the same batch
on both legs, so the step-1 probe and the per-step envelope compare unlike things, and L0 does not
apply either because no kernel changed. That remains true of the instruments as built. It is not an
argument that no instrument exists.

**The idea.** Loss has a second natural abscissa. Both legs read the same dataset in the same order,
so after consuming the same number of **useful (non-pad, loss-unmasked) tokens** they have seen the
same samples, whatever the step boundaries did in between. `loss(cumulative_useful_tokens)` is
therefore defined on both legs and comparable by interpolation onto a common token grid -- exactly
the move `compare_legs_tokens.py` already makes for throughput, applied to the trajectory instead.

**Where it is valid, and where it is not.** The two legs must be doing the same *optimization*, not
merely reading the same data. That splits Class B again:

| change | same samples per optimizer update? | token-aligned loss comparable? |
|---|---|---|
| **packing / padding removal** | yes -- the same samples, regrouped into fewer sequences | **yes**, up to attention cross-contamination between packed segments and a partial sample at the window edge |
| **sequence length** | yes, if samples are not truncated differently | **yes**, with the same caveats |
| **GBS / MBS** | **no** -- a different number of updates over the same tokens | **no.** A different update count is a different optimizer trajectory and it is *supposed* to differ |

So the instrument covers the lever actually worth the most here -- 70.1% of tokens on this workload
are padding -- and does not cover batch-size changes, which is honest rather than unfortunate: for
those the trajectory difference is the intended effect, not an artifact.

**Why Megatron's packing really is Class B**, so this is needed: `--global-batch-size` counts
SEQUENCES (`scripts/adapters/megatron.sh:54`). Pack four short samples into one buffer and a step of
64 sequences now holds ~256 original samples. Step i is genuinely a different batch.

**What it would have to prove before it counts as a gate**, by the rule that has held here all
month -- both controls, and no exceptions for an instrument its author likes:

* a **null control** it must pass: two clean legs at the same config, token-aligned, must not be
  separated. The band for that is a new measurement, not a reuse of the step-aligned band -- the
  interpolation adds its own error, and a band used out of its calibration scope is an error already
  made here once.
* a **positive control** it must fail: a known-bad run token-aligned must be rejected.

Until both exist this is a design, not a gate, and `config_tuner` keeps refusing Class B. Recording
it because the current refusal reads as "Class B is ungateable", which is a stronger claim than the
evidence supports and would discourage the work that closes it.

## L2 — soak
Opt-in, plus auto-triggered for high-risk classes: anything touching backward, the optimizer, a
reduction order, or precision/scaling. Not in the default per-candidate path.

## What can accept, and what can only reject
L0 and L1 can **only reject**. Acceptance requires a throughput delta above the measured band. A
throughput-neutral, memory-saving change is **banked, not accepted** — memory is a currency that only
ConfigSweep (deep mode) can spend.

## A token-aligned trajectory comparator exists, and needs 300 steps to say anything

`scripts/trajectory_tokens.py` implements the Class-B instrument designed above: interpolate
`loss(cumulative_useful_tokens)` onto a common grid, so two legs are compared after consuming the
same useful work rather than at the same step index.

**Null control, measured on three clean 300-step reference pairs:**

| pair | median | p90 abs | max abs | sign balance |
|---|---|---|---|---|
| ref1 vs ref2 | +0.45% | 7.61% | 40.7% | 0.60 |
| ref1 vs ref3 | −0.54% | 8.90% | 103.5% | 0.36 |
| ref2 vs ref3 | −0.90% | 9.29% | 91.2% | 0.30 |

Median under 1%, sign balance in 0.30–0.60, and a tail that reaches **103%** on two clean legs. The
median is the usable statistic; the max is chaos and must never be quoted as a bound.

**At 30 steps the instrument has no sensitivity, and the check that shows it is worth keeping.** A
clean 30-step pair gives median **+9.49%** with sign balance 0.96, while the known-bad alpha = 1.0
leg gives **−7.58%** with sign balance 0.06. The clean pair deviates *more* than the defective one.
Both are one-sided, so the sign statistic does not rescue it either. Any 30-step verdict from this
comparator would be noise wearing a number, and the tool is therefore **300 steps or nothing**.

It still owes a positive control at 300 steps. The four v3 stale legs supply one for free, which is
why the same legs serve P4 and this instrument. Until that is measured it is not a gate, and
`roles/config_tuner.md` keeps refusing Class B.

## L2 soak v3 RESULT — the soak is RETIRED (2026-10-08)

Four positive legs at alpha = 0.25 against four clean overlay-ON legs, criteria exactly as
pre-registered above. Engagement verified on every leg first: **19 200 forward and 19 200 backward
invocations per leg**, 8 ranks, 0 fallbacks, `status: ok`, occupancy clean, the alpha=0.25 patch
present in each overlay. The control ran, at the scale the soak exists to cover.

| criterion | result |
|---|---|
| **null control** (2-vs-2 holdout splits of the 4 references) | **silent** — the criteria do not reject clean legs |
| **positive control** (location) | **NOT REJECTED.** Welch t = 0.375 against crit 2.780; delta +0.00576 (0.63 sigma) |
| dispersion (declared weaker) | F = 10.25 against crit 15.44 — silent |

Terminal loss per positive leg: 1.42260, 1.41718, **1.47767**, 1.41786. Three clustered low and one
far out — **dispersion, not location, for the third positive control running**. The same signature
as v2's bias, and the same reason the location test cannot see it: the outlier inflates the
positive group's own variance, which is what the t statistic divides by.

**The token-aligned trajectory comparator does not separate these legs either** (see the 2026-10-09
note at the end of this file: calling that a failed positive control was wrong, because alpha = 0.25
is the defect size chosen for being unresolvable, and the claim is withdrawn pending an alpha = 1.0
leg at 300 steps). Positive
medians −0.438, −1.346, +0.088, −0.493%; null medians (clean pairs) −0.021, −1.358, +0.231%. Every
positive sits inside the null range, and one CLEAN pair separates further than any defective leg.
So `trajectory_tokens.py` does not get promoted to a gate either, and `config_tuner` keeps refusing
Class B.

### The decision, and the arithmetic behind it

The pre-registration committed to this disposition before the data: *"if the soak cannot separate
alpha = 0.25 either, the conclusion is not to tune it — it is that a ~10 h instrument has no region
of its own at 300 steps and should be retired."* The data says exactly that, so:

**The L2 soak is retired as a gate.** Not deleted — `l2_soak.sh` and the analyzer stay, and the
pre-registered spec stays — but the Convergence Auditor no longer recommends it, and no run should
spend GPU hours on it.

The arithmetic that makes this a decision rather than a preference:

* at **alpha = 1.0**, L1's sign test rejects the defect **for free**, in a 30-step gate leg that was
  going to run anyway;
* at **alpha = 0.25**, where L1 is silent, the observed effect is 0.00576 against an MDE of 0.01587.
  Closing that gap needs `(0.01587/0.00576)^2` ≈ **7.6x more legs — n = m ≈ 30, about 30 GPU-hours**
  — to detect a defect whose class L1 already catches one notch up in severity;
* and the op-local class the soak was first built against is **L0's**, at 11.5 s.

Three generations of positive control and roughly 17 GPU-hours produced no rejection. The honest
reading is not that the controls were unlucky three times. It is that between L0 (op-local, 11.5 s)
and L1 (cross-step, free, demonstrated) there is no band of defects left that 300 steps and a
handful of legs can resolve. **Chaos is the floor, and the soak buys no power against it** — the one
thing it could buy, replicate legs, costs an order of magnitude more than the gap it would close.

### What this does NOT say

* Not that long-run convergence damage is impossible. It says this *instrument*, at this length and
  this leg count, cannot demonstrate sensitivity to any defect we have been able to build.
* Not that the stale control is wasted. It is now **L1's** positive control, the first one L1 has had
  of a class L0 cannot see, and it stays in the tree for that.
* Not that the criteria were wrong. The null control passed every time; the thresholds were never
  retuned; and the v3 run is the first to state its own MDE before its results — which is what made
  the retirement decision arithmetic rather than a judgement call.

## The token comparator's positive control was tested against a defect nothing can see (2026-10-09)

Correcting my own test design, because it blocks the largest remaining lever.

`trajectory_tokens.py` was declared "fails its positive control" on the strength of the four
alpha = 0.25 stale legs: positive medians −0.438, −1.346, +0.088, −0.493% all sitting inside a null
range of −1.358 … +0.231%. That reading is accurate and the conclusion drawn from it is not.

**alpha = 0.25 is the defect size chosen specifically because NOTHING resolves it.** It is where L1
goes silent; it is where the soak's own effect (0.00576) sits at a third of its MDE (0.01587); and
the retirement argument for the soak is precisely that no instrument can see a defect that small at
300 steps. Handing that same defect to a second instrument and recording that it also failed is not
a test of the second instrument. It is re-measuring the defect.

**The control that was never run is alpha = 1.0 at 300 steps.** alpha = 1.0 was only ever measured
on a 30-step leg — a window this comparator has since been *shown* blind on, and now refuses outright
(`MIN_STEPS = 200`). So the one combination that could demonstrate sensitivity, a visible defect on a
usable window, has no data behind it either way.

Why it is worth one GPU-hour to close: Class B is refused because it has no trajectory gate, Class B
contains **packing**, and packing is the largest unexploited lever on this workload at 70.1% padding.
The refusal is currently resting on a test that was never properly run.

What the result will mean, fixed before the leg runs so neither outcome can be read generously:

* **rejected** → the comparator has both controls (null measured on three clean 300-step pairs,
  positive on a visible defect) and may be promoted to a Class-B gate, with its sensitivity floor
  stated as "somewhere between alpha 0.25 and 1.0", not as a bare pass.
* **not rejected** → the comparator genuinely cannot see a maximal cross-step defect over 300 steps,
  the earlier conclusion stands on evidence that actually supports it, and Class B stays refused.

Either way the claim in the retirement note — that the comparator fails its positive control —
is **withdrawn as unsupported** until this leg exists.

### RESULT: the alpha = 1.0, 300-step positive control (2026-10-09)

The leg the correction above called for. `l2soak_a10_300`: `status ok`, occupancy clean, 290 steady
steps, engagement `fwd=2400 bwd=2400 fallback_fwd=0` on all 8 ranks. Judged against the four clean
overlay-ON legs, under the criterion fixed before it ran.

**Pre-registered verdict: NOT REJECTED. Class B stays refused.**

| statistic | null (6 clean pairs) | alpha = 1.0 (vs each clean leg) |
|---|---|---|
| token-aligned median, whole trajectory | −1.365 … +0.422% | −1.087 … +0.129% — **inside** |
| sign balance | 0.24 … 0.74 | 0.28 … 0.58 — **inside** |

That stands, and the comparator is not promoted on this data.

**But the defect is plainly visible in the same leg**, which changes what the failure means:

| | clean (4 legs) | alpha = 1.0 | |
|---|---|---|---|
| terminal loss (mean, last 10% of steps) | 1.4281 ± 0.0092 | **1.4853** | **+6.25 sd** |
| terminal grad_norm | 1.92 ± 0.50 | **11.11** | +18.4 sd (spike-driven — read with care) |

Reading the per-step tail explains the disagreement. From about step 262 the alpha = 1.0 loss sits
above the clean leg on almost every step, by 0.03–0.11 (step 272: 1.57 vs 1.46; step 268: 1.59 vs
1.53), and grad_norm spikes to 48, 43 and **85.7** where the clean leg peaks at 15.1. It is a
**late-onset, accumulating** divergence that appears only after ~19 200 invocations — precisely the
cross-step class the soak was meant for. The comparator's whole-trajectory median weights the
~250 steps where the two legs agree as heavily as the ~40 where they do not, and dilutes it away.
Splitting the token grid into quarters still leaves the last quarter overlapping the null range,
because a median over 75 steps is still mostly pre-onset.

So the failure is the STATISTIC, not the token-aligned basis — and that is a hypothesis, not a
finding, for one reason: it was formed by looking at this leg. Choosing a terminal-window statistic
now and declaring the comparator passed would be fitting the test to the one positive example it has.
What it would take, stated before any of it exists:

* pre-register a terminal-window statistic (for instance: mean token-aligned deviation over the last
  10% of the shared token range, with its null range taken from the six clean pairs), and
* confirm it on a **fresh** alpha = 1.0 leg, not this one, and
* re-check that the null pairs stay inside it — a terminal window is noisier than a whole-trajectory
  median, so the null range will widen and may swallow the effect.

**What this does to the soak retirement: nothing, and it sharpens it.** One leg is one draw and no
formal test exists at n = 1, but +6.25 sd strongly suggests a terminal-loss test at n >= 2 would
reject alpha = 1.0. The soak is therefore not blind to everything: its floor sits somewhere between
alpha 0.25 (not rejected, n = 4) and 1.0 (very likely rejected). That does not revive it, because at
alpha = 1.0 L1's sign test already rejects the defect for free inside a 30-step gate leg. The soak's
only exclusive region would be the band between those two, and nothing measured so far lands in it.

### PRE-REGISTRATION: a terminal-window token comparator (written 2026-10-09, before the test leg)

Fixed here, committed, and only then measured. Nothing below may move after the fresh leg exists.

**Statistic.** `trajectory_tokens.py --points 300`, field `terminal_mean_rel_pct`: the mean relative
loss deviation over the last 10% of the shared useful-token range. 300 points ≈ one per step, so the
window covers ~29 steps — the same scale as the soak spec's existing terminal-loss definition, which
is where both "10%" and "mean" come from. Neither was chosen by looking at `l2soak_a10_300`.

**Null.** The six pairs of the four clean overlay-ON legs (`l2soak_0923_cand1`,
`l2soak_v2_cand{2,3,4}`), ref = the earlier-listed leg. Null bound **B = max |statistic|** over those
six. Computed AFTER this text is committed, by a rule that does not depend on the values.

**Positive.** A FRESH alpha = 1.0, 300-step stale leg — **not** `l2soak_a10_300`, which suggested the
statistic and is therefore excluded from judging it. Compared as cand against each of the four clean
legs as ref: four values.

**Verdict rule.** REJECTED iff **at least 3 of the 4** positive |values| exceed B. Two-sided, so the
rule does not assume the defect raises loss. Otherwise NOT REJECTED.

**What each outcome licenses.**
* REJECTED → the token comparator has a null control (6 pairs) and a positive control (a fresh
  visible defect), and may become the Class-B trajectory gate **with the terminal statistic**, its
  floor stated as "between alpha 0.25 and 1.0". Class B is then gated by: token-aligned throughput
  (`compare_legs_tokens.py`) + this statistic. The step-1 probe stays unavailable for Class B.
* NOT REJECTED → the terminal window is too noisy at 300 steps even for a defect at +6 sd on one leg;
  Class B stays refused, and this instrument line is closed rather than retried with a third
  statistic.

**Known weakness, stated up front.** B is the maximum of only six values, so it is a coarse bound
and its own sampling error is large. A REJECT therefore needs a clear margin to mean anything; a
marginal one (positive values within ~10% of B) will be reported as marginal, not as a pass.

**Null bound, computed after the pre-registration was committed and before the fresh leg finished:**

| null pair | terminal_mean_rel_pct |
|---|---|
| 0923_cand1 vs v2_cand2 | +0.1797% |
| 0923_cand1 vs v2_cand3 | −0.0407% |
| 0923_cand1 vs v2_cand4 | +1.3894% |
| v2_cand2 vs v2_cand3 | −0.2192% |
| v2_cand2 vs v2_cand4 | +1.2000% |
| v2_cand3 vs v2_cand4 | +1.4399% |

**B = 1.4399%.** The three largest all involve `v2_cand4`, the one clean leg whose terminal loss
(1.4417) sits away from the other three (~1.423). That is the chaos floor showing through, and it is
why B is a max over all six pairs rather than over the quiet ones: excluding the inconvenient clean
leg would be choosing the null to make the test easier.

### RESULT: the terminal-window comparator, and what a second alpha = 1.0 leg did to everything (2026-10-09)

**Pre-registered verdict: NOT REJECTED.** The fresh leg `l2soak_a10_300_b` (status ok, clean, 294
steady steps, full engagement) against the four clean overlay-ON legs, `terminal_mean_rel_pct` at
300 points: +0.2153, +0.0368, +0.2555, −1.1438% — **0 of 4** exceed B = 1.4399%, largest 0.79×B. Per
the rule written before it ran, the token comparator line is **closed**; Class B stays refused.

**The first alpha = 1.0 leg's divergence did not reproduce.** Same config, same seed:

| | `l2soak_a10_300` | `l2soak_a10_300_b` |
|---|---|---|
| terminal loss vs clean | **+6.25 sd** | **−0.20 sd** |
| terminal grad_norm vs clean | +18.4 sd | +0.11 sd |

The late divergence on the first leg was one chaotic instability episode, not an effect of the
defect. This withdraws my statement that "+6.25 sd strongly suggests a terminal-loss test at n ≥ 2
would reject alpha = 1.0" and the soak floor "between alpha 0.25 and 1.0" built on it: at 300 steps
the soak has no demonstrated sensitivity to cross-step staleness at ANY alpha tried. It is the same
lesson as the x1.004 bias's 0.03742 — one leg is one draw — and this time the draw was mine to
over-read. The retirement stands, more firmly.

### The L1 sign test: its null range was falsified, and a strict rule replaces it (2026-10-09)

Following up the alpha = 1.0 legs exposed two things about the early-window sign test.

**Its claimed positive control was confounded and does not reproduce.** "L1 catches alpha = 1.0"
rested on ONE 30-step leg at 0.750, one unit outside 0.5–0.625, compared against an overlay-OFF
reference while the stale leg had the overlay ON. Against overlay-matched references the three
alpha = 1.0 legs score 0.375–0.75, indistinguishable from clean pairs. Withdrawn.

**Its null range was the wrong instrument.** The range was the min..max of the band's three clean
pairs. Over **693 clean same-identity v3 pairs** a value outside such a range is what noise does
most of the time:

| rule | loss false-reject | grad_norm false-reject |
|---|---|---|
| shipped: outside the band's 3-pair range | **63.5%** | **55.6%** |
| strict: all ≥ 5 non-identical early diffs one sign | **5.1%** | **0.6%** |

At n = 8 the statistic moves in steps of 0.125 and a fair coin lands in {0.5, 0.625} 49% of the
time; three draws cannot bound that. The shipped rule was a reject-only gate rejecting a coin-flip of
good candidates — it "carried the 0.5B attention rejection" and rejected `attn_fmha_bwd_retry` on a
detector with these rates, so neither verdict has a sound basis on this evidence.

**Replaced, not retired.** Retiring it was tried first and reverted: a synthetic loss shift of +0.9
from step 3 onward, step 1 untouched, would then pass L1 entirely — the case the step-1 probe
structurally cannot see. The strict rule has both controls:

* **null** — the 693 pairs above;
* **positive** — the real 7B attention backward defect on its dropout-0 pair (step-1 grad_norm
  74 → 8.5): loss 0.0 and grad_norm 0.0, both firing; and the synthetic gross drift.

Its blind spots, stated: cross-step staleness and the x1.004 bias on every leg tried, and **at
dropout 0.1 — how the 7B reference runs — dropout noise hides even the real defect** (0.375 / 0.5);
the step-1 probe caught that one. It is a backstop for gross one-sided moves, not a detector of
subtle ones. Identical steps are now excluded rather than counted as "not up", which is what made a
leg identical to its reference read as one-sided.

## Packing: a structural equivalence probe instead of a trajectory gate (2026-10-09)

Class B was refused for having no trajectory gate. But packing's risks are not trajectory risks:
they are whether attention stays isolated between conversations, whether positions restart, and how
the loss is normalised -- all checkable at step 1 on identical weights, the way L0 checks an op.
The probe: put the same samples through the model packed and unpacked, and compare.

Megatron packs natively -- a jsonl line may hold several conversations, each starting with a system
message -- so `scripts/pack_sft.py` is a pure data transform. Token identity is its contract: the
Qwen template inserts a default system prompt when none is present, the default differs by model
("You are a helpful assistant." / "Please reason step by step, ..."), so it is read from the
tokenizer, and 500/500 conversations were verified token-identical on both models. The 7B dataset
packs 20,000 conversations into 6,042 packs, 3.31 per pack, 97.5% fill against 29.5% unpacked.

**Forward: PASS.** Pack 0 (5 conversations, 4094 loss tokens) through Megatron 7B at lr 0:

| | loss |
|---|---|
| expected, isolated (HF, each conversation alone + boundary targets) | 2.11936 |
| positive control: NOT isolated (one sequence, positions continuing) | 1.95279 (−7.86%) |
| **Megatron packed** | **2.12057 — +0.057% vs isolated, +8.59% vs non-isolated** |

The control discriminates (7.86% apart), and broken isolation LOWERS the loss -- a packing bug would
read as a quality improvement. Megatron's THD path on ROCm isolates attention in the forward.

**A semantic finding: the boundary targets.** Megatron shifts labels across the whole pack, so the
last token of each conversation is trained to predict the NEXT conversation's first token,
`<|im_start|>`, where unpacked it is masked. The model expects `<|endoftext|>` there: those four
tokens cost **10.6, 10.1, 10.7 and 13.4 nats**. 0.1% of tokens, 0.37% of the loss -- and a training
signal to start a new conversation instead of stopping.

**Backward: an unexplained 4.3% gradient discrepancy.** Within Megatron, with
`--calculate-per-token-loss` so that packed and unpacked layouts of the same samples have the same
objective by construction:

| | step-1 grad_norm |
|---|---|
| HF, no boundary terms | 47.89 |
| Megatron UNPACKED, same samples | 47.93 (+0.08% vs HF) |
| HF, with boundary terms = expected for packed | 48.01 |
| **Megatron PACKED** | **50.06 (+4.27% vs expected)** |

The boundary terms explain the loss difference completely (ratio 1.0042 HF, 1.0037 Megatron) but
only 0.25% of the gradient. Megatron unpacked matches HF to 0.08%, so Megatron's backward is sound in
the single-segment case. **The multi-segment THD backward is where 4.3% appears**, with a forward
that is correct -- the signature of an attention backward that does not respect segment boundaries
the way its forward does. (Cross-stack grad_norm WITHOUT per-token loss differed by 16% on one
conversation: normalisation, not numerics -- so grad norms are only compared within a stack and
under the same objective.)

Packing is NOT licensed by this. Next: localise -- attention backend, and the rocm_compat cu_seqlens
patch.

### Localising: the flash backend's backward is broken; L1 gains a gradient probe (2026-10-09)

Fused vs flash, packed vs unpacked, Megatron 7B, lr 0, dropout 0, per-token loss, same samples:

| backend | packed | unpacked | expected (HF) |
|---|---|---|---|
| fused (the reference) | 50.06 | **47.93** | 48.01 / 47.89 |
| **flash** | **19.90** | **17.45** | 48.01 / 47.89 |

Flash's FORWARD is right (loss 2.1103 vs fused 2.1127) and its GRADIENT is 2.7x too small, packed
or not. In the production recipe (dropout 0.1) the same backend reads step-1 grad_norm 26.7 against
68.0. The reference workload runs fused, so no baseline is affected -- but today's attention-backend
lever measured flash, and ConfigSweep called the gate with **no L1 at all**: had flash been faster,
it would have been accepted. Both fixed (DECISIONS 3.4f): L1 now checks step-1 grad_norm against a
10% limit (null max 1.40% over 696 pairs; known defects 61-88.5%), and a config lever cannot be
accepted without an L1 result.

The fused PACKED discrepancy (+4.27%, not explained by boundary targets) is still open. The
rocm_compat THD patch only squeezes a vacuous batch dimension and is not a plausible cause.

### CORRECTION (same day): step-1 grad_norm cannot compare implementations finely — the 4.3% is withdrawn

Two measurements overturned the "packed backward is 4.3% off" reading.

**At kernel level, packed attention is exact.** `scripts/tests/check_varlen_attention_l0.py`:
random tensors at Qwen2.5-7B geometry (GQA 28/4, d128) and pack 0's real segment lengths through
`flash_attn_varlen_func` versus per-segment `flash_attn_func`: out, dq, dk, dv **bit-identical**
(difference 0.0), both within 0.21–0.33% of an independent fp32 reference.

**Step-1 grad_norm is dominated by the embedding gradient, which is hypersensitive to numerics.**
`scripts/tests/check_grad_impl_sensitivity.py`, the SAME isolated computation (no packing), only the
attention implementation changed:

| implementation | total grad_norm | non-embedding | loss |
|---|---|---|---|
| sdpa | 48.01 | 6.17 | 2.11727 |
| flash_attention_2 | **54.06** | 6.46 | 2.11817 |
| eager | 49.13 | 6.67 | 2.13870 |

`embed_tokens` carries ~99% of the norm. Two implementations that are both correct at op level
read **12.6% apart**; even without the embedding they are 4–8% apart. So:

* The Megatron fused packed vs unpacked +4.27% sits inside the spread between correct
  implementations. **Withdrawn as evidence of a packing defect.** HF FA2 packed vs FA2 isolated
  differs by only −1.2%. Packing's backward is correct at op level, and nothing at model level
  contradicts that once the right comparison is made.
* The step-1 gradient detector's first null (696 SAME-implementation pairs, max 1.40%) was the
  wrong null for a detector that judges implementation CHANGES; its 10% limit would have rejected a
  correct attention kernel. Recalibrated to **30%**: correct changes reach 12.6% (sdpa↔FA2), accepted
  kernel stacks 3.9–4.2%; defects start at 61%. A coarse net for gross backward defects only —
  implementation correctness proper is L0's job, at op level, where it is sharp.
* The flash backend's **2.7x** is far outside the 12.6% band and remains a defect by that measure.
  Its root cause is not localised; the `flash_attn` varlen kernel itself is exact, which points at
  TE's or Megatron's integration of it rather than the kernel. Treat flash as unsafe on this stack.

**Packing status.** Forward equivalence PASSES (0.057% against a 7.86%-wide positive control).
Kernel-level backward equivalence PASSES (bit-identical). The boundary-target semantics are a real,
separate issue (each conversation's last token is trained toward the next `<|im_start|>`, 10–13 nats
each) and must be masked before packing is used for training.

### The flash backend: the attention OP is correct; the model-level gap is unexplained (2026-10-10)

`scripts/tests/check_te_attention_l0.py` runs TE's `DotProductAttention` directly -- THD, GQA 28/4,
d128, pack 0's real segment lengths and a single 4096 segment -- under each backend, against an
independent fp32 reference:

| backend | dq / dk / dv rel. error vs fp32 | grad-norm ratio vs fp32 |
|---|---|---|
| fused | 0.63–0.65% | 0.9945 |
| flash | **0.25–0.30%** | **1.0000** |

TE's flash attention is correct at op level, and more accurate than fused. So the 2.7x lower TOTAL
step-1 grad_norm of Megatron's flash path is not the attention kernel. "Flash's backward is broken"
overstated it; the accurate statement is **an unexplained model-level gradient difference, with the
attention op ruled out.** Total step-1 grad_norm is ~99% the embedding gradient, which moves 12.6%
between two correct implementations, so a large but benign shift is not excluded either -- though
2.7x is far outside anything a correct change has shown.

Time-boxed here. Settling it needs per-parameter step-1 gradients from Megatron, which the
distributed optimizer shards. Flash stays quarantined: the reference runs fused, the attention-
backend lever measured no win anyway (−0.73%, inconclusive), and L1's step-1 gradient probe rejects
flash at 60.7%. Revisit if anyone needs flash on this stack.

### Stacked correct kernels: two L1 facts the stand-alone gates never saw (2026-10-13)

The seven-kernel interleaved stack measurement (`measure_packed_stack.sh --conds ref,s5,s6,s7 --reps
3`, legs `runs/pstack_1013_154204_*`; see `sft_measurement.md`) produced 12 legs of four
implementations whose every kernel passed L0 on its own. Read as an L1 sample:

**1. Step-1 grad_norm is a numerics hash, not a correctness signal.**
* Values per condition (rank 0, reproducible within a condition to ~1%):
  * ref 140.7
  * s3 118.9
  * s5 219.6 / 221.5
  * s6 172.9 / 176.1
  * s7 132.1 / 132.7
* Step-1 loss agrees to 0.02% across all of them.
* Correct changes therefore move it by up to **85%** (s3 -> s5), against the "12.6% max for a correct
  change" recorded above and the 30% limit derived from it.
* The probe is already advisory (4.15 / 2026-10-10). This retires the 30% figure as a threshold of
  any kind: the field is reported only.

**2. The strict early-loss sign rule fires more often on implementation-change pairs than on its
calibrated null.**
* Its 5.1% false-reject rate was measured on 693 SAME-implementation pairs.
* Here, within-condition nulls fired **0/12** (sign balance 0.125–0.75).
* Cross-condition pairs fired **3/18**:
  * s6->s7: 2/9, both against leg s63.
  * s5->s6: 1/9, s53->s62 at 0.0, the opposite direction.
* Every firing traces to one leg whose early-window mean loss sits at the edge of its condition's
  spread. Per-condition mean loss over steps 2-9: ref 2.99–3.06, s5 2.90–2.95, s6 2.89–3.02,
  s7 2.93–3.02.
* The window means show no systematic shift between conditions (s7−s6 = +0.03, inside the
  within-condition spread). Late-window means and final losses show none either.
* So the rep-3 failure of s7 vs s6 is a false reject, and the documented remedy (re-gate) clears it:
  7/9 cross pairs pass.

**The open problem this exposes.** A different-but-correct implementation decorrelates the
bf16/dropout trajectory more than a re-run of the same implementation does. Two consequences:
* The rule's real false-reject rate on correct kernels is higher than 5.1%. It is about 17% on these
  correlated pairs, which is an upper-ish estimate because rows share legs.
* A single-pair gate on a correct kernel has a fair chance of a false reject.

The right null is "correct implementation change" pairs. These 18 cross pairs, and the
`step1_grad_norm` table above, are its first sample. Until it is calibrated, a single strict-sign
failure on a kernel with a clean L0 should be re-gated on a second pair before it is read as a
defect. That is what the workflow's verdict text already says ("a re-gate is the remedy").

### CORRECTION (2026-10-15): the sign rule does NOT over-reject implementation changes

The section above read 3 firings in 18 cross-condition pairs as "the rule's real false-reject rate on
correct kernels is ~17%". That was a small, correlated sample: two of the three firings shared one
leg (s63). The re-runnable control `scripts/controls/sign_rule_impl_null.py` measures it properly.

**The sample.** Every clean 30-step leg of the packed 7B workload whose overlays are all ACCEPTED
heads: 96 legs, 13 distinct implementations, from the bare reference to the 8-overlay s8 stack.
Implementation identity is `GEAK_OVERLAY_SHA`.

| pairs | loss fired | grad_norm fired | n |
|---|---|---|---|
| same implementation (re-runs) | 6.1% | 1.1% | 643 |
| different correct implementations, all | **0.8%** | **0.6%** | 3,917 |
| ... differing by exactly ONE overlay (the gate's own shape) | **3.4%** | **1.0%** | 589 |

**Reading.**
* The hypothesis was backwards. Different correct numerics decorrelate the early trajectory, so the
  signs of the 8 early diffs become close to independent coin flips (2/2^8 = 0.78%, matching the
  0.8%).
* Same-code re-runs fire MORE, because their small divergence persists in one direction across
  steps.
* A one-overlay change sits in between: it perturbs the trajectory less.
* For a gate the relevant rate is ~3.4% (loss) and ~1% (grad_norm). That is the same order as the
  5.1% quoted since 10-09, so **the rule stands unchanged**.
* `trajectory_gate.py` now also reports `measured_impl_change_rate_pct` and quotes it in the reason.
* The operational advice is unchanged: a single strict-sign failure on a kernel with a clean L0 is
  re-gated on a second pair before it is read as a defect. The 10-13 s7-vs-s6 rep-3 failure remains
  a false reject.

**What this does not change.** The rule's blind spots are stated above:
* it misses subtle biases;
* at dropout 0.1 it misses a real backward defect.

It is a backstop for gross one-sided moves. Its false-reject rate is now measured for the case it is
actually used on.

## P0-4 PRE-REGISTRATION: training quality of the accepted stack, judged by OUTCOME (written 2026-10-15, before any leg)

**Question.** Does one epoch of SFT on the accepted s8 stack produce a model as good as the
reference recipe? "As good" means held-out loss, not trajectory similarity. The trajectory
instruments (L1 envelope, soak) were retired because bf16/dropout chaos makes 30-step and 300-step
trajectories non-comparable. An outcome metric does not need trajectories to agree; it needs the
trained weights to be equally good.

**Data.** `sft_ultrachat20k.jsonl` split by a fixed rule: line index % 20 == 19 is held out.
* `..._train19k.jsonl` (19,000 conversations, sha 89f8aa06) packs to 5,729 packs (sha e3bb6e7b).
* `..._heldout1k.jsonl` (1,000, sha 6409b130) packs to 309 packs (sha 54fad019).
* No line is shared. Both were packed by `pack_sft.py` (`--seq 4096`), with 300/300 token identity
  verified.

**Recipe.** `workloads/qwen2_5_math_7b_packed.env` as confirmed in DECISIONS 4.21: GBS 16 packs, MBS 1,
seq 4096, dropout 0.1, LR 1e-5 constant, AdamW, clip 1.0, bf16, distributed optimizer, TP1 x DP8.
* `--iters 358` = one epoch (5,729 / 16).
* `--data` = the train19k pack.
* `--valid-data-path` = the held-out pack, `--eval-iters 19 --eval-interval 1000000`. That gives one
  evaluation, at the end of training, over 19 × 16 = 304 of the 309 held-out packs. Which 304
  depends on the seed, so the arms of one seed see exactly the same packs.
  (`--full-validation` crashes with SFTDataset: smoke `runs/evalsmoke_1015_a`.)

**Arms**, three seeds each (1234, 2345, 3456; the seed sets data order AND dropout masks), run
interleaved per seed in the order REF, S8, CTRL:
* **REF**: no overlay.
* **S8**: `stack_overlay_env.sh --cond s8`, the eight accepted kernels.
* **CTRL**, the positive control: REF with `--lr 7.0e-6` (x0.7). It is a known, modest,
  non-equivalent change to the recipe. Its role is to prove the metric can see a difference of a
  known kind and size. Without it a PASS would be uninterpretable: this is exactly how L2 died.

**Metric.** The end-of-training `validation loss ... lm loss value` printed by Megatron. Eval-mode
note, measured in smoke `runs/evalsmoke_1015_s8`:
* Under no_grad, the rope / swiglu / attn / lmhead overlays fall back to the reference forward.
  fc1 / fc2_oproj / qkv / rmsnorm stay engaged.
* So the metric mostly judges the WEIGHTS the stack trained, through reference kernels. That is the
  intended object.

**Statistics.** For arm X, paired by seed: d_k = X_k − REF_k, k = 1..3. Let m = mean(d) and
t = m / (sd(d) / sqrt 3). The critical value is t_crit = 4.30 (two-sided 95%, df 2).

**Pre-registered verdicts.**
1. Control validity: CTRL is DETECTED iff all three d_k have the same sign AND |t| ≥ 4.30. If it is
   not detected, the experiment is **INCONCLUSIVE (underpowered)**. Report the observed d's and stop;
   no S8 verdict is drawn.
2. With the control detected:
   * **S8 FAIL** if m_S8 > 0 and t_S8 ≥ 4.30: significantly worse held-out loss.
   * **S8 PASS** if not FAIL and |m_S8| ≤ 0.25 × |m_CTRL|. The stack's effect on held-out loss is at
     most a quarter of a 30% LR cut.
   * **S8 DIFFERS-BETTER** if m_S8 < 0 and |t| ≥ 4.30. Reported and investigated; it is not a failure.
   * Otherwise **INCONCLUSIVE at the observed effect**, reported with its number.
3. Reported, not voting:
   * final training loss (mean of the last 50 steps);
   * grad-norm summary;
   * throughput and per-overlay engagement (fallback 0 in training required, or the S8 leg is void).

These criteria are fixed now and are not to be changed after any leg is read. A deviation from this
plan (a crashed leg re-run, a seed swapped) is reported as such.

### P0-4 RESULT (2026-10-15): INCONCLUSIVE by the pre-registered rule, and a late-drift signal on S8

Run `scripts/quality_ab.sh`, 9 legs `runs/qab_1015_151502_*`. All legs are ok, 358 steps. The S8
legs show 64 proofs and no training-path fallback.

Held-out validation loss (`quality_ab_analyze.py`):

| seed | REF | S8 | CTRL (LR x0.7) |
|---|---|---|---|
| 1234 | 1.2980 | 1.3870 | 1.2975 |
| 2345 | 1.2895 | 1.4320 | 1.3320 |
| 3456 | 1.2953 | 1.2874 | 1.3010 |

Paired against REF:
* CTRL: d = −0.0005 / +0.0425 / +0.0057, t = 1.18. **The control is NOT detected.**
* S8: d = +0.0889 / +0.1425 / −0.0080, t = 1.69.

**Pre-registered verdict: INCONCLUSIVE (underpowered). No S8 verdict is drawn**, as the plan requires.

**Exploratory, beyond the pre-registration.** It is labelled as such and decides nothing by itself.
Arms of one seed see identical training batches (`useful_tokens` equal on every step), so the
per-step training loss is directly paired.
* **The control's effect is real but gone by the end.** CTRL − REF training loss runs +0.15 to +0.32
  over steps 1–50 and +0.08 to +0.16 at steps 51–100, then shrinks to +0.01 to +0.04 by steps
  341–358. A constant LR ×0.7 over one epoch converges to nearly the same place. So the end-of-epoch
  held-out loss with 3 seeds was the wrong-powered instrument for this control. The design needs
  either more seeds or a control whose end-point effect is larger.
* **S8 shows no systematic offset over the run.** Windowed S8 − REF over steps 1–340 alternates in
  sign: seed 1234 is +0.06 early and −0.07 at 301–340; seed 2345 is −0.15 at 101–150 and ≈0
  elsewhere.
* **But two of three S8 seeds drift late.** From about step 348, S8 − REF on identical batches rises
  monotonically from +0.00 to +0.10 (seed 1234) and +0.12 (seed 2345), and S8's grad norm climbs from
  2–3 to 6–12 while REF stays at 1–2. Seed 3456 shows nothing. The held-out excess on those two seeds
  (+0.09, +0.14) is this drift caught by the end-of-training evaluation.
* Two seeds with different data orders starting a smooth drift at about the same step is unlikely to
  be coincidence. Something that accumulates with steps or calls is the leading hypothesis. Until it
  is explained, **the stack is NOT cleared for long training**.

Diagnostics running (exploratory):
* reproduce S8 and REF at seed 1234;
* s7 (s8 without rmsnorm) at seed 1234, to bisect the newest overlay;
* S8 at seed 2345 again.

### What the P0-4 diagnostics showed: the end-point metric was dominated by recipe-level instability (exploratory, 2026-10-15)

* **Same code, same seed, rerun.** REF seed 1234 ended at held-out **1.401** against the first run's
  1.298. Its training loss differed by +0.21 over steps 1–30 and ±0.08 later.
  * Run-to-run nondeterminism (atomics, chaos) plus late-training instability is enough to move
    the end-of-epoch held-out loss by about 0.1 for the REFERENCE itself.
  * The first three REF seeds landing within 0.009 of each other was luck.
  * The same-seed pairing the design leaned on is much weaker than assumed.
* **Late instability is a property of the recipe**, not of S8. Constant LR 1e-5, no warmup, no decay.
  Count of steps after step 150 with grad norm > 10:
  * REF: 10 / 14 / 11 / 18;
  * CTRL (LR x0.7): 43 / 7 / 84;
  * S8: 19 / 3 / 12 / 36.
* **S8 rerun at seed 1234.** A severe episode from about step 321: training loss on identical
  batches is +0.18 at steps 321–340 and +0.32 at 341–350, grad norm median 38. Held-out 1.426.
* **Progressive validation is the better instrument.** In a single epoch every pack is seen once,
  so each step's training loss, taken before that step's update, is a loss on unseen data.
  * Its mean over steps 201–358 covers about 2,500 packs, against 304 for the end eval, and it
    averages over transients.
  * Arms of one seed share the exact batches.
  * On the existing legs it DETECTS the control. CTRL − REF = +0.015 / +0.017 / +0.048 (mean +0.027,
    all positive). S8 − REF = −0.012 / −0.005 / +0.004 (mean −0.004).
  * The S8 rerun is the outlier at 1.511, against REF's 1.440–1.461.
* So the open question is **whether S8 has episodes more often than REF**. 1 severe episode in 4 S8
  runs, against 0 in 4 REF runs, cannot answer it. The question is settled by P0-4b below, not by
  these legs. These legs were looked at to choose the metric, so they cannot also confirm it.

## P0-4b PRE-REGISTRATION: progressive validation, 5 fresh seeds (written 2026-10-15, before any P0-4b leg)

Same data, recipe, arms and script as P0-4 (`quality_ab.sh`, REF / S8 / CTRL = LR x0.7). Seeds 4567,
5678, 6789, 7890 and 8901, none used before. 15 legs.

**Primary metric:** PV = mean training loss over steps 201–358 (`probe_rank0.jsonl`). It is paired
by seed: d_k = PV(X)_k − PV(REF)_k, n = 5, t = mean / (sd / sqrt 5), t_crit = 2.776 (two-sided 95%,
df 4).

**Episode** (per leg against its own seed's REF leg): any 10-step window after step 150 in which the
mean paired loss difference exceeds +0.10. REF legs are scored for episodes against the S8 leg of
the same seed with the sign reversed, so both arms are judged by the same symmetric rule.

**Verdicts.**
1. **Control detected** iff t_CTRL ≥ 2.776. If not, the run is INCONCLUSIVE (underpowered) and no S8
   verdict is drawn.
2. **S8 FAIL (quality)** if m_S8 > 0 and t_S8 ≥ 2.776.
3. **S8 FAIL (instability)** if the count of S8 legs with an episode exceeds REF's count by ≥ 4 (of
   5). Fisher exact p ≈ 0.048 at 4:0.
4. **S8 PASS** if neither FAIL fires and |m_S8| ≤ 0.25 × |m_CTRL|.
5. Otherwise INCONCLUSIVE at the observed effect.

Secondary, reported only: end-of-training held-out loss, late grad-norm counts, throughput,
engagement (training-path fallback 0 required, or the S8 leg is void). The P0-4 and diagnostic legs
are NOT pooled into this test.

### P0-4b RESULT (2026-10-16): the accepted stack has a LATE-TRAINING INSTABILITY defect

`runs/qabpv_1015_191546_*`, 15 legs, all ok. Progressive validation (PV, mean training loss over
steps 201–358):

| seed | REF | S8 | CTRL | S8 episode (first step) |
|---|---|---|---|---|
| 4567 | 1.4336 | 1.4786 | 1.4608 | 279 |
| 5678 | 1.4510 | **2.5041** | 1.4767 | 254 |
| 6789 | 1.4815 | **1.9982** | 1.4754 | 196 |
| 7890 | 1.4524 | 1.5279 | 1.4564 | 327 |
| 8901 | 1.4332 | 1.4334 | 1.4743 | none |

* CTRL − REF: mean +0.0184, t = 2.16.
* S8 − REF: mean +0.338, t = 1.68 (one huge outlier inflates the sd).
* Episodes: S8 4/5, REF 0/5.

**Verdict by the pre-registered rule order: INCONCLUSIVE.** Rule 1 gates every S8 verdict on the
control (t 2.16 < 2.776).

**Verdict on the substance: S8 FAILS on instability, and the stack is not cleared for training
beyond a few hundred steps.** The pre-registration had a design flaw:
* The control calibrates the size of a QUALITY effect. The instability rule compares S8 with REF
  directly and needs no calibration.
* Gating it on the control was a mistake. It is recorded here rather than silently re-ordered.
* The instability rule on its own fires: 4 vs 0 against the pre-registered ≥ 4.

Across all S8-family legs to date:
* severe episodes in 6 of 11 (P0-4b 4/5, the P0-4 rerun at seed 1234, the s7 diagnostic);
* REF: 0 of 9 by the same rule.

The episodes are catastrophic, not drift:
* Seed 5678: training loss 1.47 → 2.26 (steps 286–300) → **8.50** (301–315), grad norm up to 2,241,
  then recovery to 1.53 by the end.
* Seed 6789: loss 5.42 at steps 211–225, grad-norm median 213.
* No iteration was skipped.
* Shape: a few corrupted gradients, clipped to norm 1.0 but pointing the wrong way, then recovery.

**Localisation so far.**
* **s7 (no rmsnorm) diverges too:** seed 1234 PV 2.914, 70 steps after 150 with grad norm > 10.
  The newest overlay is not the cause.
* Leading hypothesis: a race or missing stream synchronisation in a SIDE-STREAM wgrad path (fc1's
  and qkv's overlapped wgrad, lm-head's two-stream backward). A rare unsynchronised read of a
  partially written buffer corrupts one step's gradient now and then. No 30-step gate (L1, the
  throughput gate) can see an event with a per-step probability this low.
* Bisection running on the cumulative stack, three seeds per condition, starting with s4 (rope,
  swiglu, fc1, lmhead).

**Consequence for delivery.** Every stacked bundle (`finalize_bundle_1015_s7_stack`, `_s7_rmsn`,
`_s8`, and the per-head bundles until cleared individually) is **NOT cleared for production
training** until the defect is found and the stack passes a P0-4b-style test. The throughput
numbers stand; the training-safety claim does not.

**Bisection log** (cumulative stack, scored against the P0-4b REF leg of the same seed with the
P0-4b episode rule):

| condition | overlays | seeds → episode? |
|---|---|---|
| s8 | all 8 | 4567 ✗279, 5678 ✗254, 6789 ✗196, 7890 ✗327, 8901 ✓ |
| s7 | no rmsnorm | 1234 ✗ (PV 2.91) |
| s4 | rope, swiglu, fc1, lmhead | 4567 ✓, 5678 ✓, 6789 ✗292 (max loss 8.30) |

So the defect is in rope / swiglu / fc1 / lmhead. Running next: s2 (rope, swiglu) and s3 (+fc1) on
seeds 6789, 5678, 4567 and 7890.

**Bisection, round 2 (2026-10-16), with a corrected severity scale.**

The pre-registered episode threshold (a 10-step mean paired difference > 0.10) sits AT the noise
level:
* a same-code, same-seed REF rerun reaches 0.109 against its original;
* CTRL legs reach up to 0.115.

So "episode" is re-scored by severity. The P0-4b conclusion survives this:
* S8 max10 values 0.22 / 7.39 / 5.60 / 0.80 / 0.01;
* REF 0/5 at any threshold;
* 4/5 at a threshold of 0.2, 3/5 at 0.3.

The threshold was a second pre-registration flaw and is recorded as such.

Reference ranges for the scale:
* max training loss after step 150 on the 8 REF legs: 1.61–1.81;
* same-code rerun: 1.62.

| condition | catastrophic (max loss ≥ 5) | severe (1.9–2.5, or max10 > 0.25) | within REF range |
|---|---|---|---|
| s2 rope+swiglu (4 seeds) | 0 | 0 | 4 (max loss ≤ 1.74; max10 ≤ 0.18) |
| s3 +fc1 (4) | 0 | 1 (4567: 2.27, max10 0.38, gn 1,949) | 3 (7890 gn spike 6,159) |
| s4 +lmhead (3) | 1 (6789: 8.30) | 0 | 2 |
| s7 / s8 (11) | 4 | 3 | 4 |

* rope and swiglu are cleared at this resolution.
* fc1 is the first overlay whose addition produces events outside the REF range.
* lmhead is not yet separable from fc1.
* Running next: each in isolation on the bare workload (`stack_overlay_env.sh --cond only:<name>`),
  same 4 seeds.
* The severity criteria are fixed BEFORE that data: severe = max loss after step 150 > 1.90 or max10
  > 0.25; catastrophic = max loss ≥ 5.

**Isolation (2026-10-16)**, `runs/qabiso_1016_030317_*`, scored on the pre-fixed severity scale:
* fc1 alone: 1/4 severe (seed 7890: max loss 2.02, max10 0.32).
* lmhead alone: 1/4 severe (seed 6789: max loss 2.69, max10 0.47, gn 1,158).
* No catastrophic event with either alone.
* Progressive validation was above REF on 4/4 seeds for each: fc1 +0.024 / +0.018 / +0.003 /
  +0.049; lmhead +0.014 / +0.016 / +0.076 / +0.011. That is CTRL-sized, but n = 4.

**The wgrad-precision hypothesis is REFUTED in situ** (`runs/wgprobe_1016_*`, the overlays' own
probes, `GEAK_FC1_WGRAD_PROBE` / `GEAK_LMHEAD_WGRAD_PROBE`):
* fc1's increment to `main_grad` is within 1.7e-7 of the fp32 reference, against bf16 rounding of
  2.7e-4.
* lmhead's is within 0.7–1.0e-6, against 0.5–1.4e-3.
* Both are genuinely fp32 and `+=`. The lm-head integrator's unreproduced "1.39e-3" was a
  mis-measurement.

**What correlates instead: side streams.**
* No events: REF (0/9), CTRL (0/8) and rope+swiglu (0/4), all single-stream.
* Events with each overlay that runs a second stream: fc1 (wgrad on a side stream) and lmhead
  (dgrad on a side stream), each 1/4.
* Events most often with S8: fc1, qkv and lmhead all two-stream, 7/11.

That is the signature of a rare race: correct when probed, wrong now and then.

**Test, criteria fixed before data:**
* s8 with every side stream off: `FC1_BWD_MODE=serial QKV_BWD_MODE=serial LMHEAD_CE_TWO_STREAM=0`.
  The kernel bytes are unchanged; arm `s8serial` in `quality_ab.sh`.
* It runs on the five P0-4b seeds, where s8 had 4/5 severe-or-worse.
* ≤ 1/5 severe-or-worse supports the race hypothesis; ≥ 3/5 refutes it; 2/5 is inconclusive.

**Side-stream test result (2026-10-16), `runs/qabser_1016_053856_*`.** `lmhead two_stream=False`
was confirmed on every leg, and throughput fell to 3,408 (from 3,488) as expected without overlap.

| seed | s8 (P0-4b) | s8serial |
|---|---|---|
| 4567 | severe | within |
| 5678 | catastrophic 9.36 | within |
| 6789 | catastrophic 8.39 | within |
| 7890 | severe | severe (2.04) |
| 8901 | within | **catastrophic 8.72** |

* By the fixed criteria this is 2/5, INCONCLUSIVE. On substance, **the side-stream race hypothesis
  is rejected as the main cause**: a catastrophic episode occurs with every side stream off.
* PV is still above REF on 5/5 seeds (+0.037 / +0.021 / +0.003 / +0.030 / +0.70).

**Where the evidence stands**, events at "severe or worse" on the fixed scale:

| arm | legs | severe+ | catastrophic | note |
|---|---|---|---|---|
| CTRL (correct, LR x0.7) | 8 | 0 | 0 | out-of-sample null (the REF legs defined the scale, so they cannot be their own null) |
| REF rerun | 1 | 0 | 0 | same-code null |
| s2 rope+swiglu | 4 | 0 | 0 | |
| s3 (+fc1) | 4 | 1 | 0 | |
| fc1 alone | 4 | 1 | 0 | |
| lmhead alone | 4 | 1 | 0 | |
| s4 | 3 | 1 | 1 | |
| s7 | 1 | 1 | 1 | |
| s8 (incl. reruns) | 11 | 7 | 2 | |
| s8serial | 5 | 2 | 1 | |

* s8-family 9/16 against the nulls' 0/9: Fisher p ≈ 0.004.
* Catastrophic events appear only with ≥ 4 overlays (5 of 24 legs), never in 17 null legs. On its
  own that is not yet significant (p ≈ 0.06).

**Reading.**
* The stack makes late-training excursions much more likely. Side streams are not the cause, and
  wgrad precision is not the cause.
* Single overlays give a milder signal; the effect grows with how many are stacked.
* Two candidate mechanisms remain, and they are not exclusive:
  1. An ill-conditioned recipe tipped over by accumulated, individually tolerable numerical
     differences. Constant LR 1e-5, no warmup or decay, and grad norms of 100–7,000 early show
     the recipe is marginal on its own; every arm has spikes.
  2. A specific numerical defect that per-op L0 cannot see and that combines across overlays. For
     example, the step-1 grad norm moves 20–85% between "correct" stacks, while a same-code rerun
     moves it 0%.
* The step-1 grad-norm sensitivity was dismissed as a "numerics hash" (4.20). In light of this it
  deserves a per-parameter look: which parameters carry that norm, and how they differ between REF
  and s8 on identical weights and batch.

**Production status.** Not cleared. **rope + swiglu (s2)** is the only part of the stack with no
event in 4 legs: +7.2% over packed REF. That is too few legs to clear it, but it is the natural
candidate for a reduced stack that a P0-4b-style test could clear.

## P0-4c PRE-REGISTRATION: can the reduced stack rope + swiglu (s2) be cleared? (written 2026-10-16, before any leg)

Same protocol as P0-4b: `quality_ab.sh`, one epoch of 358 iterations, progressive validation over
steps 201–358. Arms REF and s2. Five seeds never used before: 9012, 1123, 2234, 3345 and 4456. 10
legs. The four bisection legs of s2 are NOT pooled, because they were looked at.

**s2 is CLEARED iff all three hold:**
1. No s2 leg is severe-or-worse on the fixed scale (max loss after step 150 > 1.90, or max10 > 0.25
   against the same seed's REF).
2. |mean PV(s2) − PV(REF)| ≤ 0.0046, which is 0.25 × the P0-4b control effect of +0.0184.
3. It is not significantly worse: t < 2.776 (df 4).

Otherwise s2 is NOT cleared, and the reason is reported. No positive-control arm is run: the P0-4b
CTRL effect is used as the fixed yardstick, and the null for criterion 1 is the 0/9 above.

## P0-4d PRE-REGISTRATION: the stack under a STANDARD LR schedule (written 2026-10-16, before any leg)

**Why.** The user clarified on 2026-10-16 that "no tuning" means follow general default training
practice (warmup allowed) and keep it fixed during optimisation. It never meant constant LR. The
P0-4/P0-4b recipe (constant 1e-5, no warmup, no decay) was marginally stable on its own: every arm
had grad-norm excursions. P0-4c (s2 under that recipe) was stopped during its first leg and never
analysed.

**Recipe.** `workloads/qwen2_5_math_7b_packed_cosine.env`: the packed descriptor plus a common SFT
default schedule, not tuned.
* peak LR 1e-5;
* linear warmup over 11 iterations (3% of 358);
* cosine decay to 1e-6 (10%) at iteration 358.
The adapter now reads LR / MIN_LR / LR_DECAY_STYLE / LR_WARMUP_ITERS / LR_DECAY_ITERS from the
descriptor. Its argv for the older descriptors is byte-identical (80 tokens, diffed). The smoke
`runs/lrsmoke_1016_cos` shows LR 9.1e-7 at step 1, 1e-5 at step 11, and 9.93e-6 at step 30.

**Protocol.** Same data and evaluation as P0-4b. Arms: REF, S8 (`stack_overlay_env.sh --cond s8`),
and CTRL70 (the whole schedule ×0.7: `--lr 7.0e-6 --min-lr 7.0e-7`). Seeds 9012, 1123, 2234, 3345
and 4456 (P0-4c's, never analysed). 15 legs. `scripts/quality_ab_verdict.py` applies:

* **Stability, decided on its own**, with severity per leg against its seed's REF (REF scored
  symmetrically against S8): max10 = the largest 10-step mean of the paired loss difference after
  step 150.
  * severe: max10 > 0.25 (2.3× the same-code null of 0.109);
  * catastrophic: max10 > 1.0.
  * PASS: no severe S8 leg.
  * FAIL: S8 severe ≥ 2 and exceeding REF by ≥ 2; OR any catastrophic S8 leg while REF and CTRL70
    have none.
  * Otherwise INCONCLUSIVE.
* **Quality, decided on its own**, on PV (steps 201–358) paired by seed.
  * The control is detected iff t ≥ 2.776; if not, quality is INCONCLUSIVE.
  * FAIL: S8 significantly higher.
  * PASS: |m_S8| ≤ 0.25 × |m_CTRL70|.
  * Otherwise INCONCLUSIVE.
* **CLEARED iff PASS-stability AND PASS-quality.** All rules are fixed now.

### P0-4d RESULT (2026-10-16): NOT CLEARED by the pre-registered rule; under the standard schedule the catastrophic episodes are gone and the final model matches REF

`runs/qabcos_1016_*`, 15 legs, all ok; the cosine LR schedule is verified in the logs.

| seed | PV REF / S8 / CTRL70 | max10 S8 / REF(rev) / CTRL70 | held-out REF / S8 / CTRL70 |
|---|---|---|---|
| 1123 | 1.4550 / 1.5645 / 1.5248 | **0.379** / 0.006 / 0.108 | 1.2959 / 1.2978 / 1.3205 |
| 2234 | 1.4586 / 1.4594 / 1.5232 | 0.010 / 0.003 / 0.112 | 1.2951 / 1.2967 / 1.3236 |
| 3345 | 1.4873 / 1.4617 / 1.5326 | −0.006 / 0.088 / 0.095 | 1.2997 / 1.2990 / 1.3150 |
| 4456 | 1.4671 / 1.4950 / 1.5114 | 0.156 / 0.018 / 0.065 | 1.3003 / 1.3025 / 1.3145 |
| 9012 | 1.4610 / 1.4563 / 1.6249 | 0.002 / 0.011 / **0.527** | 1.2985 / 1.2962 / 1.3188 |

**Pre-registered verdicts (`quality_ab_verdict.py`).**
* The control is DETECTED: CTRL70 − REF PV = +0.0776, t = 3.50.
* Stability: INCONCLUSIVE. S8 severe 1/5 against REF 0/5; CTRL70 also 1/5. No catastrophic leg in
  any arm.
* Quality: INCONCLUSIVE. S8 − REF PV = +0.0216, which is 0.28 × the control, just above 0.25. It is
  carried almost entirely by seed 1123's episode.
* **Overall NOT CLEARED.**

**Reading (exploratory where marked).**
* **The standard schedule removed the catastrophic class.** S8 had 2/5 catastrophic under constant
  LR and 0/5 under cosine.
* Severe transients still occur at about 1 in 5 for S8, but **also for a correct implementation**
  (CTRL70 9012: max loss 2.68, max10 0.53). On this evidence a severe transient is a recipe-level
  event, not an S8 signature. 1 S8 against 1 CTRL70 against 0 REF is not a difference.
* **Exploratory: with LR decay, the end-of-training held-out loss becomes a clean instrument.**
  * REF spans 1.2951–1.3003.
  * S8 − REF paired: +0.0019 / +0.0015 / −0.0007 / +0.0023 / −0.0023, mean **+0.0005**.
  * CTRL70 − REF: +0.0246 / +0.0284 / +0.0153 / +0.0142 / +0.0203, mean **+0.0206**, all positive.
  * S8's effect is about 1/40 of the control's. The decayed final model shows no trace of the
    mid-run transient.
  * The P0-4 end-eval failure was a constant-LR artefact: there was no decay to settle the weights.
  * This metric was chosen after seeing these data, so these legs cannot confirm it. P0-4e below
    does that.

## P0-4e PRE-REGISTRATION: confirm the stack under the standard schedule (written 2026-10-16, before any leg)

* Descriptor `qwen2_5_math_7b_packed_cosine.env`, same data and protocol as P0-4d.
* Arms REF and S8 only. Seeds 5567, 6678, 7789, 8890 and 9901, never used before. 10 legs.
* `scripts/quality_ab_final.py` applies the rules below.

**Primary quality metric:** the end-of-training held-out loss, paired by seed. The yardstick is
FIXED from P0-4d: the ×0.7 schedule's effect of +0.0206.
* PASS-quality iff |mean d| ≤ 0.00515 (0.25 × yardstick), and not (mean > 0 with t ≥ 2.776).
* FAIL-quality iff mean > 0.00515 with t ≥ 2.776.
* Otherwise INCONCLUSIVE.

**Stability:**
* FAIL iff any S8 leg has max10 > 1.0.
* PASS iff there is none and at most 1 S8 leg has max10 > 0.25. The null rate is about 1 in 5:
  CTRL70 under this schedule.
* Otherwise INCONCLUSIVE.

**CLEARED iff both PASS.** The P0-4d legs chose this metric and are not pooled. Applied to them for
reference only, the script returns CLEARED: mean +0.00054, t 0.60, 1 severe, 0 catastrophic. That
is exactly why they cannot be the confirmation.

### P0-4e RESULT (2026-10-16): **CLEARED** — the 8-kernel stack is safe to train with under the standard schedule

`runs/qabfin_1016_140340_*`, 10 legs, all ok. Every S8 leg has 64 overlay proofs with training-path
fallback 0. `quality_ab_final.py`:

| seed | held-out REF | S8 | d | max10 |
|---|---|---|---|---|
| 5567 | 1.29119 | 1.29368 | +0.00249 | 0.010 |
| 6678 | 1.29928 | 1.29331 | −0.00597 | −0.012 |
| 7789 | 1.30361 | 1.30162 | −0.00199 | −0.003 |
| 8890 | 1.29723 | 1.29619 | −0.00104 | 0.005 |
| 9901 | 1.29984 | 1.29462 | −0.00522 | 0.002 |

* **Quality: PASS.** Mean d = −0.0023, t = −1.54: within the ±0.00515 equivalence margin, not
  significantly different, nominally better.
* **Stability: PASS.** 0 severe, 0 catastrophic; max loss after step 150 is 1.63–1.68 on every S8
  leg.
* **Overall: CLEARED.**

**The full arc, for the record:**
* P0-4 (constant LR, end-eval): INCONCLUSIVE.
* Diagnostics: the end metric was dominated by recipe-level instability.
* P0-4b (constant LR, PV): the stack fails on late catastrophic episodes. The bisection and
  isolation then ruled out rmsnorm, wgrad precision and side streams.
* The user clarified that "no tuning" means default practice, warmup included.
* P0-4d (standard schedule): the catastrophic class is gone; the decayed held-out loss matches.
* P0-4e (confirmatory, fresh seeds): CLEARED.

Conclusion: the constant-LR, no-warmup recipe was marginally stable, and the stack's accumulated,
individually tolerable numerical differences tipped it over. Under the standard schedule they do
not. **The clearance is recipe-specific: the stack is cleared for the scheduled recipe and remains
NOT cleared for constant-LR, no-warmup training.**

Process note: the P0-4e driver exited with code 2 AFTER its tenth leg. `quality_ab.sh` was edited
(the `cand` arm) while it was running; bash reads a script incrementally, so the tail failed to
parse. No leg was affected. Never edit a shell script that a running job is executing.

## P0-4g PRE-REGISTRATION: outcome test of the HF stack (written 2026-10-17, before any leg)

**Question.** Is the HF TRL stack (`hf2`: attention CK/aiter flash + fused Qwen2RMSNorm, DECISIONS
4.27) safe to train with? Its attention backward recasts to fp16 without an overflow guard, and no
30-step gate can see a late failure.

**Protocol.** `quality_ab.sh --stack hf_trl --workload workloads/qwen2_5_math_7b_hf_cosine_zero2_q.env
--iters 296`.
* Data: the unpacked 19k / 1k split (`sft_ultrachat20k_train19k.jsonl` sha 89f8aa06, held-out
  `..._heldout1k.jsonl` sha 6409b130).
* Recipe: ZeRO-2, GBS 64, warmup 9, cosine to 10% over 296 iterations, i.e. one epoch.
* Held-out evaluation after training, over all 1,000 conversations (`hf_sft.py --valid-data-path`).
* Arms, interleaved per seed:
  * `ref`: bare;
  * `cand`: `--cand-cmd "stack_overlay_env.sh --cond hf2"`;
  * `ctrl70`: the schedule ×0.7, as a positive control measured IN THIS RUN. The Megatron yardstick
    is a different stack, data and GBS, so it is not reused.
* Seeds 1111, 2222, 3333, 4444 and 5555. 15 legs.

**Verdict** (`quality_ab_final.py <prefix> --base ref --cand cand --ctrl ctrl70`, rules fixed now):
* **Control:** ctrl70 − ref held-out must have mean > 0 and t ≥ 2.776. Otherwise quality is
  INCONCLUSIVE and the stack is not cleared.
* **Quality PASS:** |mean(cand − ref)| ≤ 0.25 × the measured control effect, and not significantly
  worse.
* **Quality FAIL:** mean > margin with t ≥ 2.776.
* **Stability:** FAIL if any cand leg has max10 > 1.0. PASS if none, with at most 1 at > 0.25.
* **CLEARED iff both PASS.** The bare HF stack is bit-deterministic, so ref legs are exact repeats of
  a seed. The arms decorrelate only through the kernels and the LR.

### P0-4g RESULT (2026-10-17): **CLEARED** — the HF stack is safe to train with

`runs/qabhf_1017_051602_*`, 15 legs, all ok, 296 steps each. Every cand leg has 16 overlay proofs
with fallback 0.

| seed | held-out ref | cand (hf2) | d | ctrl70 − ref |
|---|---|---|---|---|
| 1111 | 1.265669 | 1.265579 | −0.00009 | +0.00633 |
| 2222 | 1.265321 | 1.265395 | +0.00007 | +0.00658 |
| 3333 | 1.265618 | 1.265624 | +0.00001 | +0.00640 |
| 4444 | 1.265519 | 1.265456 | −0.00006 | +0.00656 |
| 5555 | 1.265823 | 1.265705 | −0.00012 | +0.00642 |

* **Control: detected.** +0.00646, t = 138. The deterministic HF stack has no same-seed noise, so
  the instrument is extremely sharp here.
* **Quality: PASS.** Mean d = −0.00004, about 0.6% of the control effect (margin 0.00161), t −1.10.
* **Stability: PASS.** max10 ≈ 0 on every leg; max training loss after step 150 is 1.39–1.41.
* **Overall: CLEARED.**

The attention backward's unguarded fp16 recast did no measurable harm over one epoch. The two HF
models are numerically almost identical.

Analysis fix: `quality_ab_final.py` had the Megatron protocol's 358 steps hardcoded and reported the
296-step HF legs as INCOMPLETE. It now requires equal step counts covering the episode window. The
P0-4e verdict is unchanged (re-checked: CLEARED).

## P0-4h PRE-REGISTRATION: outcome test of the FINAL HF configuration (written 2026-10-17, before any leg)

**Candidate.** hf3 (attn + rmsnorm + mlp gate/up+SwiGLU) with TRL step metrics off, on
`workloads/qwen2_5_math_7b_hf_cosine_zero2_nm_q.env`. Same protocol as P0-4g: 296 iterations, 19k / 1k
unpacked split, held-out evaluation over 1,000 conversations. Seeds 1111..5555, the SAME as P0-4g.

**Reference reused from P0-4g, justified and checked.**
* The bare HF stack is bitwise deterministic (4.28).
* Metrics off is bitwise identical to metrics on (4.31).
* So P0-4g's `ref_seed*` legs ARE the REF of this test.
* **Check, run first:** one bare leg at seed 1111 on the `_nm_q` descriptor. Its per-step loss and
  its held-out loss must equal P0-4g's `ref_seed1111` exactly. If they do not, reuse is void and all
  five REF legs are re-run.
* The yardstick is P0-4g's in-run control: ctrl70 legs, +0.00646, t 138. Those legs are linked into
  the same prefix.

**Verdict.** `quality_ab_final.py <prefix> --base ref --cand cand --ctrl ctrl70`. P0-4g's rules,
unchanged:
* PASS-quality iff |mean d| ≤ 0.25 × 0.00646 and not significantly worse;
* stability FAIL on any max10 > 1.0.
* CLEARED iff both pass.

**P0-4h REF-reuse check: PASSED.** The bare seed-1111 leg on the `_nm_q` descriptor
(`runs/qabhfchk_1017_203738_ref_seed1111`, TRL metrics off) matches P0-4g's `ref_seed1111` (metrics
on) bitwise: loss and grad_norm at all 296 steps, and held-out loss 1.265669 on both. P0-4g's REF
legs are therefore this test's REF. Running: the five cand legs `runs/qabhf3_1017_210912_*`.

### P0-4h RESULT (2026-10-18): **CLEARED** — the final HF configuration (2.07x) is safe to train with

`runs/qabhf3_1017_210912_*`: 5 cand legs plus P0-4g's REF and ctrl70 legs, linked into the prefix
(reuse validated bitwise). Every cand leg has 24 overlay proofs with fallback 0.

| seed | held-out ref | cand (hf3 + metrics off) | d |
|---|---|---|---|
| 1111 | 1.265669 | 1.265674 | +0.00001 |
| 2222 | 1.265321 | 1.265369 | +0.00005 |
| 3333 | 1.265618 | 1.265718 | +0.00010 |
| 4444 | 1.265519 | 1.265550 | +0.00003 |
| 5555 | 1.265823 | 1.265824 | +0.00000 |

* **Quality: PASS.** Mean +0.00004, which is 0.6% of the control's +0.00646 (margin 0.00161).
  t = 2.06 is below 2.776, so the difference is not significant.
* **Stability: PASS.** max10 = 0 on every leg.
* **Overall: CLEARED.**
* Stated, not hidden: all five d are ≥ 0. Even if the shift is real, it is about 40× inside the
  equivalence margin.

## `hf_mlp_down_proj` re-adjudicated after the warmup fix, and P0-4i PRE-REGISTRATION (2026-10-18, before any P0-4i leg)

**Re-adjudication.** `wf_499a73c3-cd2` returned `hf_mlp_down_proj` as `unusable_ref`: 4 steady steps,
caused by the warmup defect fixed in DECISIONS 4.33.
* L0 passed: 18/18 checks, error equal to the original's.
* L1 passed on re-gate. The first pair's strict sign rule fired with the candidate LOWER; two
  further pairs passed with the direction flipping.
* The SAME gate legs (`runs/gate_hf_mlp_down_proj_1018_023628`) re-summarised with the corrected
  detector give 25 steady steps: aligned +6.54%, **paired +7.18%**, ratio IQR 2.45%.
* The workflow's own `gateVerdict` (extracted and applied unchanged) returns **win: +7.18% > band
  1.82%**.
* Engagement on the cand leg: attn / rmsnorm / mlp / down all on 8 ranks. mlp and down are both
  6,720 forward and backward, fallback 0.
* down_proj chains on `Qwen2MLP.forward` after the gate/up wrapper (`_PREV`).
* **Accepted** (DECISIONS 4.34). Stack condition `hf4` = hf3 + down_proj (gate-dir install, bytes
  verified against its SHA256SUMS).

**P0-4i PRE-REGISTRATION.** It is P0-4h with cand = hf4.
* Descriptor `_nm_q`, 296 iterations, seeds 1111..5555.
* REF and ctrl70 are P0-4g's legs: the bitwise reuse was already validated in P0-4h.
* `quality_ab_final.py <prefix> --base ref --cand cand --ctrl ctrl70`, rules unchanged.
* CLEARED iff PASS-quality and PASS-stability.
* Run before it: an interleaved throughput confirmation, hf3 vs hf4, 2 reps on `_nm`.

### P0-4i RESULT (2026-10-18): **CLEARED** — hf4 (attn + rmsnorm + mlp gate/up + down_proj, metrics off) is safe to train with

* Throughput confirmation (`runs/hf34_1018_034217_*`, corrected detector): hf3 2,181 / 2,176, hf4 2,290 / 2,290.
  Paired **+7.09 / +7.29%**, fallback 0.
* Outcome (`runs/qabhf4_1018_040121_*`, REF and ctrl70 reused from P0-4g): held-out d −0.00011 / +0.00002 /
  +0.00004 / +0.00016 / −0.00019, mean −0.00002 (t −0.28), against a control of +0.00646.
  Stability: max10 0 everywhere. 32 proofs per cand leg, fallback 0. **CLEARED.**


## P0-4j PRE-REGISTRATION (2026-10-19, before any P0-4j leg): the outcome test of hf6

hf5 (`hf_attn_qkvo_proj`, 4.36) and hf6 (`hf_lmhead_fused_ce`, 4.37) were accepted on 30-step gates only.
On this stack the grad_norm envelope does not discriminate: the hf4-vs-hf4 null already reaches 23.6x.
**One** outcome test covers both increments, P0-4i's protocol with cand = hf6:
* Descriptor `_nm_q`, 296 iterations, seeds 1111..5555, arm `cand` = `stack_overlay_env.sh --cond hf6`.
* REF and ctrl70 reused from P0-4g (`qabhf_1017_051602_*`). Bitwise reuse was validated in P0-4h.
* `quality_ab_final.py <prefix> --base ref --cand cand --ctrl ctrl70`, rules unchanged.
* CLEARED iff PASS-quality and PASS-stability.
* If NOT CLEARED: bisect by running cand = hf5. hf4 stays the recommended stack until then.
* Run before it: interleaved throughput confirmation hf4 vs hf6, 2 reps on `_nm`.

### P0-4j, first attempt (2026-10-19): throughput confirmed; outcome legs lost their evaluation (harness bug, fixed)

* **Throughput** (`runs/hf46_1019_154350_*`, corrected detector): hf4 2,286 / 2,289, hf6 2,406 / 2,407.
  Paired **+5.23 / +5.24%** (ratio IQR 0.9 / 1.1%), same_batches, occupancy clean.
* **Outcome legs `runs/qabhf6_1019_160153_cand_*` are INCOMPLETE.** Every leg trained all 296 steps,
  then died in `trainer.evaluate()` with `KeyError: 'eval_loss'`.
  * Cause: the overlay wrapper's bare `(*a, **kw)` signature. HF Trainer's `find_labels(model.__class__)`
    found no `labels` on the rebound `Qwen2ForCausalLM.forward`, so evaluation computed no loss.
  * hf6 is the first model-level rebind; every earlier overlay rebinds a submodule.
  * Fix 410207cb: `wrapped.__signature__ = inspect.signature(orig)`. Verified in the image:
    find_labels `[]` -> `['labels']`. VAR_KEYWORD, which Trainer's `model_accepts_loss_kwargs` reads,
    is unchanged, so the training path, the gates and the throughput above are unaffected.
* Protocol unchanged. The 5 cand legs are re-run as `qabhf6b_*`.

### P0-4j RESULT (2026-10-19): **CLEARED** — hf6 (hf4 + qkvo + lm_head/CE) is safe to train with

* Outcome legs: `runs/qabhf6b_1019_175827_*`. REF and ctrl70 are symlinks to P0-4g's `qabhf_1017_051602_*`.
* Held-out d: −0.00015 / +0.00013 / −0.00003 / −0.00006 / +0.00006, mean **−0.00001** (t −0.19).
  Control ctrl70 − ref: +0.00646 (t 138). PASS-quality: |mean| ≤ 0.00161.
* Stability: max10 0 on every leg, PASS.
* Engagement per cand leg, 8/8 ranks: attn, rmsn, mlp, down, qkvo and lmce all fallback 0 in training.
  lmce: 2,368 candidate fwd = bwd per rank. Its 125 `grad_disabled` fallbacks per rank are the eval
  pass, which runs HF's own lm_head + loss.
* **CLEARED.** hf6 replaces hf4 as the recommended HF stack (DECISIONS 4.38).

## P0-4k PRE-REGISTRATION (2026-10-20, before any P0-4k leg): the outcome test of hf8

hf8 = hf6 + flat-grad accumulation + bf16-exact RoPE (4.39). Both are bitwise to the original by
oracle, so the expected outcome is "indistinguishable". The test still runs (4.23): the gates saw 30
steps, and the flat-grad overlay touches the gradient path of every parameter.
* P0-4j's protocol with cand = `stack_overlay_env.sh --cond hf8`: `_nm_q`, 296 iterations, seeds 1111..5555.
* REF and ctrl70 reused from P0-4g.
* `quality_ab_final.py <prefix> --base ref --cand cand --ctrl ctrl70`. CLEARED iff PASS-quality and
  PASS-stability.
* If NOT CLEARED: bisect with cand = hf7.
* Run before it: interleaved throughput confirmation hf6 vs hf8, 2 reps on `_nm`.

### P0-4k RESULT (2026-10-20): **CLEARED** — hf8 (hf6 + flat-grad accumulation + bf16-exact RoPE) is safe to train with

* Throughput (`runs/hf68_1020_020814_*`): hf6 2,399 / 2,404; hf8 2,621 / 2,616. Paired **+9.02 / +8.89%**.
* Outcome (`runs/qabhf8_1020_022718_*`): held-out d −0.00004 / +0.00003 / +0.00005 / +0.00008 / −0.00022,
  mean **−0.00002** (t −0.38). Control +0.00646 (t 138). Stability max10 0. **CLEARED.**
* Engagement in training: 8/8 ranks, fallback 0 for all 8 overlays.
  * rope: candidate fwd = bwd = 66,304 per rank (296 × 8 × 28).
  * Eval-only fallbacks: lmce 125 and rope 3,500 per rank, all no_grad, running the original path.

## P0-4l PRE-REGISTRATION (2026-10-23, before any P0-4l leg): the model-matched RMSNorm epsilon recipe

Every Megatron leg so far fine-tuned with RMSNorm eps 1e-5 (Megatron's default; the torch_dist checkpoint
does not carry it), while Qwen2.5-Math-7B was trained with 1e-6 (DECISIONS 4.55). The fix is the
descriptor `_cosine_avg_eps6` and, for the stack, condition `s8eps` (s8 with the rmsnorm overlay's eps
envelope widened; L0 at eps 1e-6 PASS, L1 overlay PASS, throughput +0.34/+0.31% over s7).
* Workload `qwen2_5_math_7b_packed_cosine_avg_eps6.env`, 358 iterations (one epoch), the pre-registered
  train19k / heldout1k packs, `quality_ab.sh` protocol. Seeds **1123, 2234, 3345, 4456, 5567** (fresh).
* Arms, interleaved per seed: `ref` (bare, eps 1e-6), `s8eps`, `cand` = bare with
  `--norm-epsilon 1e-5` (the old recipe as a per-leg arm; `--cand-cmd true`).
* Q1, the GATE (clears the deliverable stack on the new recipe): `quality_ab_final.py <prefix>
  --base ref --cand s8eps` (rule noninf, margin 0.00515). CLEARED iff PASS-quality and PASS-stability.
* Q2, REPORT (the effect of the epsilon itself): `--base cand --cand ref`, i.e. d = held-out(eps 1e-6)
  - held-out(eps 1e-5). eps 1e-6 becomes the RECOMMENDED recipe iff it is non-inferior to 1e-5 on the
  same rule (upper 95% bound of d <= 0.00515). It is a correctness fix approved by the owner, so a
  non-significant d is the expected outcome; a significantly POSITIVE d would be reported and the
  recommendation revisited, not silently kept.
* No positive control arm: the Megatron yardstick (+0.0206 for the x0.7 schedule, P0-4d) is fixed.
* If Q1 is INCONCLUSIVE: one pre-registered extension, seeds 6678, 7789, 8890, 9901, 1012.

### P0-4l RESULT (2026-10-23): s8eps CLEARED on eps 1e-6, but the eps 1e-6 recipe itself is NOT recommended

Legs: `runs/qabeps6_1023_073707_*` (15 legs, 07:37-12:02, all rc 0). Per-arm held-out loss by seed (1123, 2234, 3345, 4456, 5567):
* ref (bare, eps 1e-6): 1.300942 / 1.297982 / 1.303304 / 1.307243 / 1.302431
* s8eps (eps 1e-6):     1.298323 / 1.300178 / 1.304090 / 1.297119 / 1.296110
* cand (bare, eps 1e-5): 1.294472 / 1.295910 / 1.299808 / 1.295370 / 1.292632

**Q1 (gate): s8eps vs ref — CLEARED.** d −0.00262 / +0.00220 / +0.00079 / −0.01012 / −0.00632, mean −0.00322,
upper 95% +0.00162 ≤ 0.00515. Stability 0 severe, 0 catastrophic (worst max10 0.242 at seed 3345, just
under the 0.25 severe line). All 8 overlays engaged in training on 8/8 ranks; the only fallbacks are the
end-of-training eval pass (`no_grad`, `dropout_p0`, `not_training`); rmsnorm fallback 0.

**Q2 (report): eps 1e-6 vs eps 1e-5 — FAILS non-inferiority.** d = held-out(1e-6) − held-out(1e-5) =
+0.00647 / +0.00207 / +0.00350 / +0.01187 / +0.00980, mean **+0.00674**, t 3.65 (5/5 positive), upper 95%
+0.01068 > 0.00515. Stability: the bare eps-1e-6 leg of seed 4456 had a CATASTROPHIC late episode (max10
1.046 around step 328; progressive-validation loss 1.764 against 1.453 for eps 1e-5). Without that seed
the mean is still +0.0055. s8eps against eps 1e-5 is worse on 5/5 seeds as well (+0.0017..+0.0043).
Per the pre-registered rule, eps 1e-6 does NOT become the recommended recipe: production stays
`_cosine_avg` (eps 1e-5) and its s8 bundle stays the deliverable. The step-1 loss IS lower with 1e-6
(2.66771 vs 2.66823 on the throughput legs), so matching pretraining helps the first forward and still
trains to a worse and less stable model over the epoch at this LR schedule; one plausible mechanism is
that a smaller eps raises rstd -- and with it dx -- for low-norm rows. Not investigated further:
the measurement answers the recipe question.

## P0-4m PRE-REGISTRATION (2026-10-23, before any P0-4m leg): outcome test of token-balanced rank assignment on hf8

The lever (`--hf-balance-tokens on`, DECISIONS 4.61) keeps every optimizer step's sample SET and only moves
samples between ranks / micro-batch slots; under transformers' global loss normalization
(`average_tokens_across_devices` true, now explicit and in workload identity) the step gradient is the same
sum, changed only by floating-point summation order. 30-step gates: +14.7..+16.9% paired; step-1 loss
identical; L1 PASS on bare (deterministic stack) and on hf8 rep 2, early-sign FAIL on hf8 rep 1, inside
hf8's own run-to-run nondeterminism (its off-vs-off null pair shows the same one-sided pattern). Per 4.23 a
change to a cleared stack still needs the outcome test.
* cand = `stack_overlay_env.sh --cond hf8 --stack hf_trl` + `--hf-balance-tokens on` (the balance
  algorithm as committed when the legs launch), workload `qwen2_5_math_7b_hf_cosine_zero2_nm_q.env`,
  296 iterations, seeds 1111, 2222, 3333, 4444, 5555 (P0-4k's).
* BASE = P0-4k's hf8 legs (`runs/qabhf8_1020_022718_cand_seed*`, same overlay bytes, same descriptor and
  data, hf_sft training path unchanged since -- the only later edit before today was capture-only),
  symlinked into the new prefix as arm `hf8`; control = P0-4k's `ctrl70` legs, symlinked likewise. This
  reuse is the P3 item "reuse HF QualityGate ref/ctrl70 legs"; it puts cross-session nondeterminism into
  d, which can only make the test harder to pass.
* `quality_ab_final.py <prefix> --base hf8 --cand cand --ctrl ctrl70` (rule noninf). CLEARED iff
  PASS-quality and PASS-stability. If INCONCLUSIVE: one extension, seeds 6666, 7777, 8888, 9999, 1212,
  with fresh hf8 and cand arms.

### P0-4m RESULT (2026-10-23): **CLEARED** — token-balanced rank assignment on hf8 is safe to train with

* Legs: cand `runs/qabhfbal_1023_143846_cand_seed*` (balance v2, capacity-constrained LPT); `hf8` and `ctrl70`
  are symlinks to P0-4k's legs. `quality_reuse.py check`: valid on all 5 seeds -- the realized argument
  tables differ ONLY in hf_balance_tokens (off -> on).
* Held-out d (cand - hf8): -0.00013 / +0.00002 / +0.00005 / -0.00008 / +0.00028, mean **+0.00003** (t 0.38),
  upper 95% +0.00018 <= 0.00162. Control ctrl70 - hf8 +0.00648 (t 122.6). Stability: max10 0 on every leg.
* Engagement on the cand legs: all 8 hf8 overlays, 8/8 ranks, training fallback 0.
* **CLEARED.** The recommended HF configuration becomes hf8 + `--hf-balance-tokens on` (DECISIONS 4.61).

## P0-4n PRE-REGISTRATION (2026-10-23, before any P0-4n leg): outcome test of s9 (s8 with the 1022 attention + NaN check from the grad norm) under synchronized GC

* s9 = `stack_overlay_env.sh --cond s8a22` (attention overlay replaced by finalize_bundle_1022_attn_ck1022, cleared on
  the bare recipe by 4.45) + `--no-check-for-nan-in-loss-and-grad` with `GEAK_NAN_FROM_GRAD_NORM=1` (DECISIONS 4.62),
  run with `--manual-gc --manual-gc-interval 0` (the deliverable's intended default). 30-step gates on s8+gc:
  +1.15 / +1.16% paired (wall +2.1 / +1.6%; s8 nulls +-0.02%), L1 kernel_swap PASS both reps (step-1 loss 0.0).
* cand legs: workload `qwen2_5_math_7b_packed_cosine_avg.env`, 358 iterations, seeds 1357, 2468, 3579, 4680, 5791.
* BASE = P0-4f's s8 legs (`runs/qabavg_1016_175311_cand_seed*` = s8 + --ddp-average-in-collective, the _cosine_avg
  recipe), linked as arm `s8` with `quality_reuse.py`; its check must show only check_for_nan_in_loss_and_grad,
  manual_gc and manual_gc_interval differing (DECISIONS 4.63).
* `quality_ab_final.py <prefix> --base s8 --cand cand` (rule noninf, fixed P0-4d yardstick, margin 0.00515).
  CLEARED iff PASS-quality and PASS-stability. If INCONCLUSIVE: one extension, seeds 7013, 8124, 9235, 1346, 2457,
  with fresh s8 and cand arms.

### P0-4n RESULT (2026-10-23): **NOT CLEARED** — s9 (the 1022 attention on s8) has a late divergence; the extension was not run

* Legs `runs/qabs9_1023_175600_*` (cand) vs P0-4f's s8 legs (linked as `s8`; `quality_reuse.py check` valid: only
  check_for_nan_in_loss_and_grad and manual_gc differ). Held-out d: +0.00321 / -0.00573 / +0.00084 / **+0.27336** /
  -0.00303 (seed 4680), mean +0.0537, upper 95% +0.171 -> INCONCLUSIVE; stability PASS by the max10 rule (0.145).
* Seed 4680 is a genuine late divergence, not noise: grad norm sustained at 5-20 from step ~300 (s8: 1-2), train
  loss 0.1-0.2 above s8 at the end, held-out 1.569 vs 1.296.
* DEVIATION FROM THE PRE-REGISTRATION, stated: the one extension was NOT run. With d = +0.273 in the sample, the
  combined n = 10 upper bound stays >= ~+0.077 even if all five new seeds were perfect -- far above the 0.00515
  margin -- so the only reachable verdicts were INCONCLUSIVE or FAIL; 2.7 GPU-hours would have bought no decision.
* ATTRIBUTION (one leg, `runs/qabs8n4680_*_seed4680`): s8 + NaN check from the grad norm + --gc on seed 4680 is
  healthy (held-out 1.2929 vs s8 1.2958; late grad norm median 1.41). The divergence comes with the 1022 attention
  on s8. (On bare, the 1022 kernel cleared at n = 10 after one outlier seed, 4.45.)
* Consequence: the 1022 attention is NOT folded into s8 (DECISIONS 4.64).

## P0-4o PRE-REGISTRATION (2026-10-24, before any new P0-4o leg): outcome test of the deliverable's `fast` default

`fast` = s8 + `--manual-gc --manual-gc-interval 0` + `--no-check-for-nan-in-loss-and-grad` with
`GEAK_NAN_FROM_GRAD_NORM=1` (DECISIONS 4.62 / 4.64), shipped on the "numerically inert" argument plus one seed.
This closes that gap.
* cand: `fast`, workload `_cosine_avg`, 358 iterations, seeds 1357, 2468, 3579, 4680, 5791 (P0-4f's). Seed 4680
  REUSES the leg already run with exactly this configuration on 2026-10-23 (`runs/qabs8n4680_1023_191833_cand_seed4680`,
  run before this pre-registration, recorded here as such); the other four are new.
* BASE = P0-4f's s8 legs (`runs/qabavg_1016_175311_cand_seed*`), linked as arm `s8`; `quality_reuse.py check`
  must show only check_for_nan_in_loss_and_grad and manual_gc(+_interval) differing.
* `quality_ab_final.py <prefix> --base s8 --cand cand` (noninf, fixed yardstick, stability incl. the held-out
  arm of DECISIONS 4.66). CLEARED iff PASS-quality and PASS-stability. If INCONCLUSIVE (and not FAIL): one
  extension, seeds 7013, 8124, 9235, 1346, 2457, with fresh s8 and cand arms. If NOT CLEARED, the bundle default
  reverts to `--plain`.

### P0-4n CORRECTION (2026-10-24): the attribution to the 1022 kernel is withdrawn (DECISIONS 4.69)
Seed 4680 re-run twice with the 1022 kernel (`runs/qaba22r_1024_040138_cand_seed4680` = s8a22 alone,
`runs/qabs9r_1024_041744_cand_seed4680` = s9 exactly): held-out 1.2959 and 1.2958 (s8: 1.2958), no drift. The
P0-4n divergence was one chaotic realisation (the fp32-atomic summation order differs per run); the code
diff shows the two kernels compute the same thing. P0-4n stays NOT CLEARED as recorded; folding the kernel
needs a fresh pre-registered test.

### P0-4o RESULT, first sample (2026-10-24): INCONCLUSIVE -> the pre-registered extension is running
`runs/qabfast_1024_061114_*` (cand 4680 linked from qabs8n4680; base = P0-4f s8 legs; reuse check valid --
only check_for_nan_in_loss_and_grad and manual_gc differ). d +0.01070 / -0.00546 / -0.00074 / -0.00293 /
+0.00385, mean +0.00108, upper 95% +0.00716 > 0.00515; stability PASS (0 severe, 0 catastrophic).
Extension: `runs/qabfastx_1024_074442_*`, fresh s8 and cand arms, seeds 7013, 8124, 9235, 1346, 2457; the
combined n = 10 verdict is final. Until then the bundle keeps `fast` as default; NOT CLEARED reverts it.

### P0-4o RESULT (2026-10-24): **CLEARED** at the pre-registered n = 10 — the `fast` default stays
Combined `runs/qabfast_1024_061114_*` + `runs/qabfastx_1024_074442_*` (`quality_ab_final.py` with both prefixes):
d per seed -0.00066 / +0.01070 / -0.00992 / -0.00546 / -0.00074 / -0.00293 / +0.00385 / -0.00145 / -0.02849 /
+0.00286, mean -0.00323, upper 95% +0.00284 <= 0.00515 -> PASS-quality; stability PASS (1 severe: seed 1346 max10
0.494, a training-loss excursion with normal held-out d -0.0007; 0 catastrophic; held-out arm of 4.66: none).
Stated: seed 8124's d -0.0285 comes from the s8 BASE leg running poorly (held-out 1.3212); without that seed the
mean is -0.00042 and the upper bound +0.0033, still inside the margin. `fast` (s8 + --gc + the NaN check from the
grad norm) is cleared by its own outcome test, no longer only by the "numerically inert" argument.
