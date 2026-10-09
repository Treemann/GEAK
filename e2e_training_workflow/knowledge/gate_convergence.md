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
