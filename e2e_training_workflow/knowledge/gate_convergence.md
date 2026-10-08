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

**Cheap checks that run BEFORE the 3 hours**, because a control that fails to install reads as a
pass and that has happened here before (`measure_stack.sh` wrote 3528 silent engagement failures):

* `stale_state_control.py --self-test` — the invisibility claim, as arithmetic, no GPU.
* L0 on the patched overlay — must PASS, confirming invisibility end to end and not just on paper.
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
