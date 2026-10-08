# e2e_training_workflow — 2-month roadmap (DRAFT)

> Chinese version: `ROADMAP.zh-CN.md`. Both are maintained together; the numbers here are
> authoritative if the two ever drift.

Status, revised **2026-10-08** (first written 2026-09-24). `fast` mode runs end to end. **Five real
runs** (0.5B/2GPU ×1, 7B/8GPU ×4) plus 12 soak legs and a 4-leg stack measurement.

**Two kernels accepted, and the combined stack is MEASURED at +12.36%** — four ref/cand pairings
spanning 0.12 points, null control +0.0706%, and a window-immune `step_median_ms` cross-check at
+12.324%. The savings **compound** (adding predicts +12.00%, compounding +12.34%, measured +12.36%),
so the two kernels shorten disjoint parts of the step.

Everything below is ordered by two constraints that were **measured**, not assumed:

**Constraint A — a gate that cannot discriminate makes every later number fiction.** Three gates were
found non-discriminating *after* they had already issued verdicts: the throughput gate accepted on a
raw ratio where window placement alone manufactures +4.59%; the L1 envelope rejected provably-clean
legs (out-of-sample null control fails 3/3); the L2 soak passed a kernel with a deliberate ×1.004
gradient bias. **Gate work therefore precedes optimization work in this plan** — not out of caution,
but because optimization output is unfalsifiable until the gates are.

**Constraint B — a run costs 8–10 h and decides ~2 heads, and that did NOT change.** Overhead removal
took a run from 10.6 h to 7.9 h, but the head count stayed at 2 both times. The timeline says why:
`kernel_lane` is **46%** of a run and it is the actual optimisation work. Per-head cost is ~2.5–3 h
and is not overhead. **The original exit criterion (≥3 heads/run) was the wrong target** — see "What
this plan got wrong" below.

Assumptions (flag if wrong — they change the plan materially): one 8×MI308X node, shared;
`rocm/primus:v26.4/5`; bare Megatron + the `--sft` path; no multi-node access yet; effort is
primarily this workflow rather than the kernel layer underneath it.

---

## Where this is now — the design as built

### The layered picture

```
  training_workflow.js          ← THIS layer: a system wrapper. Owns identity, bands, gates, budgets.
    ├─ 6 agent roles            training_director · system_architect · profiler
    │                           kernel_extractor · convergence_auditor · integrator
    ├─ scripts/                 the deterministic half: probe, summarise, compare, gates, overlay
    ├─ workloads/*.env          Seam 1 — the pinned workload descriptor (arch args, GBS/MBS/SEQ,
    │                           and now the per-workload band floor)
    ├─ scripts/adapters/<stack>.sh   Seam 2 — framework plug (megatron only, today)
    └─ ../kernel_workflow/kernel_lane.js   the UNCHANGED single-kernel layer, called directly
                                (the dispatcher would spend the one nesting level the runtime allows)
```

The division of labour is deliberate: **judgement is the agent's, control flow is the orchestrator's.**
Roles return structured JSON; `training_workflow.js` applies every threshold. No agent can accept its
own candidate.

### How one run flows

```
Setup      Director measures the TRUE baseline ×3 → across-job band (R_cal) + L1 trajectory band
Profile    Profiler decomposes the step across ranks into compute / exposed-comm / bubble /
           straggler, emits Top-N, plus a READ-ONLY config diagnosis (fast mode's stand-in for
           ConfigSweep, and simultaneously ConfigSweep's requirements list)
Strategize Architect ranks heads by  cp_share × (1 − 1/attainable)  — critical-path share, NOT
           %GPU time. Validated: predicted 3.94% on the lmhead head, measured +4.66%.
KernelTrack  per head, until budget / gates / wall clock run out:
             Extract   fwd+bwd oracle frozen from a live capture (never a replay)
             ↓ kernel_lane (recursive, budget-limited)
             L0        op-level numerics vs a frozen fp32 oracle — can only REJECT
             Overlay   reversible rebind; engagement proven per rank, forward AND backward
             L1        trajectory — can only REJECT
             GATE      exclusive whole-box A/B; the ONLY thing that can ACCEPT
             on accept: the candidate becomes the carried reference, and the run RE-PROFILES
             (the bottleneck moves — confirmed: after accepting lmhead a new head appeared)
Finalize   reversible bundle + SHA256SUMS + a launch.sh that verifies itself
Report     architect report (out-of-repo)
Validate   Director re-measures INDEPENDENTLY and arbitrates. This is the official number.
```

Budgets are three-fold and all three have fired in practice: `budget` (kernel tasks), `max_gates`
(one A/B = two job launches and it blocks every lane), and a wall clock sourced from
`args.t0_epoch_ms` plus an agent-read clock (the sandbox has none).

### The gates, and what each can actually detect

This table is the most important thing in the document, and every row is measured:

| gate | can it ACCEPT? | null control | positive control | measured sensitivity |
|---|---|---|---|---|
| **Throughput** | **yes — the only one** | 0 false wins / 47 null pairs | fires on a known +4.66% | band 0.415% on that real pair (old formula: 0.914%) |
| **L0** op numerics | no (reject only) | scores exactly 0 ref-vs-ref | **rejects ×1.004 bias** | `scale_bias` 2e-3 tol; 11.5 s end to end, ~40 ms for the comparison |
| **L1** step-1 probe | no | 0.0% (bit-reproducible) | rejects the attention backward | caught −88.5% grad_norm collapse — **the only detector L1 has left** |
| **L1** sign balance | no | **a true-zero pair scored 0.000** | — | held to a measured clean-pair range; cannot tell a fix from a bug |
| **L1** per-step envelope | no | **FAILS out-of-sample 3/3** | — | **RETIRED from voting 2026-09-28** |
| **L2** soak | no | passes | **does NOT reject ×1.004** | underpowered; ~0.9% bias, not 0.4% |

The through-line: **op-local defects belong to L0** (11.5 s end to end), **forward-path changes to the step-1
probe** (free), and the soak is left with only what neither can see. That is why P4 exists.

### Identity, the anti-fiction spine

`config_digest` (schema 3: 33 identity fields) answers "did these two legs measure the same thing?";
`impl_digest` (content-hashed) answers "is this the implementation that was gated?"; `digest_schema`
versions the definition, and v2↔v3 is deliberately **not** reconcilable — it demands a re-measure,
because a v2 sidecar physically never recorded the fields v3 digests.

---

## Progress — what is built, and what is *verified*

Built and **verified on real runs** (~260 agents across five training runs, `agents_error: 0`):

| area | state |
|---|---|
| Measurement layer | **solid.** The instrument that disproved a +7.85% "win" and that answers the headline question |
| Six role prompts | all execute; accept, reject, and *integration-blocked* paths all exercised end to end |
| Throughput gate | recalibrated from 47 null pairs; **both controls demonstrated** |
| L0 | both controls demonstrated; four op oracles; rejects a ×1.004 gradient bias in 11.5 s |
| Extraction → kernel lane | works; 18.86× geomean on an isolated kernel → +4.66% end to end |
| Overlay | **now a STACK** — `GEAK_OVERLAY` takes a list, all-or-nothing, per-overlay counters |
| Combined delivery | `measure_stack.sh`; **+12.36% measured, engagement proved per overlay on 8 ranks** |
| Independent validation | an independent role overturned an upstream conclusion in **all five** runs |

**Decided heads** (7B):

| head | result |
|---|---|
| `lmhead_ce_unmasked_gather` | **accepted, +4.66%** |
| `mlp_fc2_down` | **accepted, +7.34%** — wgrad 13× closer to fp32 than the original |
| `attn_fmha_fwdbwd_gqa_d128_causal` | rejected — backward defect, step-1 grad_norm −88.50% |
| `mlp_fc1_swiglu` | rejected — inf gradient at iteration #2, reproduced twice |
| `attn_dense_wgrad_rowslice` | **+1.147% on top of fc2** — confirmed by an independent re-measurement (fc2+attn is +8.46% vs bare ⇒ +1.043% increment, 0.10 points apart). `inconclusive_noise`: real but below a band its own 2.2–2.9% IQR produces |
| `attn_fmha_bwd_retry` | **decided: +0.31%**, not the 2.52% predicted. Delta is trustworthy (4 pairings, spread 0.014 pts; both null controls 0.0000%; step-median agrees to 0.003 pts). Worth keeping for **correctness**, not speed |

**Built but not trustworthy, and the picture got worse rather than better:**

- **L1 is now effectively the step-1 probe alone.** The per-step envelope was retired from voting
  (4 runs, 4 firings, 0 times load-bearing). Within a week the *other* voting detector, early-window
  sign balance, was falsified the same way: a pair with a **true delta of exactly zero** scored
  0.000. It is now held to a measured clean-pair range, but the deeper problem is recorded and open —
  **a sign test cannot separate a correctness fix that descends faster from a bias pointing
  downhill**, and it rejected a candidate L0 rates *strictly more accurate* than the incumbent.
- **L2 soak** — tooling and a pre-registered spec exist; still has never issued a verdict with weight.

**Not built:** ConfigSweep / `deep` mode · `mode=bakeoff` from this layer (structurally unreachable)
· multi-node · TP > 1 (refused, not supported) · a second stack adapter.

### What is actually open on these two

Both were "lost to instrument defects" in the first draft of this plan. Checking each before acting
changed both stories, and neither in the direction expected:

- **`attn_dense_wgrad_rowslice` is real but small.** Its +1.147% was *correct*, not corrupted — an
  independent re-measurement against a bare reference gives fc2+attn = +8.46%, implying a +1.043%
  increment, 0.10 points from the original. What blocked it is that its own step time is
  **data-dependent by construction** (runtime row-slice, `n_live/M` spanning 0–100%), so its legs
  carry 2.2–2.9% IQR and produce a band wider than the signal. Accepting it needs either many more
  legs or a statistic that is not the median of a data-dependent step — not a re-run.
- **`attn_fmha_bwd_retry` is unblocked but undecided.** Sign balance 0.000 sits *inside* the range
  clean pairs produce, so the detector that rejected it cannot legitimately do so. What it has never
  had is a **throughput** number: it was rejected before the gate measured one. It needs an A/B, not
  an argument.

## What this plan got wrong

Worth recording, because the errors are more useful than the parts that went as written.

**1. "≥3 heads per run" was the wrong exit criterion.** Overhead removal worked — 10.6 h → 7.9 h —
and the head count did not move, because `kernel_lane` is 46% of a run and it *is* the work. Chasing
heads-per-run would have meant cutting optimisation depth to make a number go up. The useful metric
is **cost per decided head including rejections**, and on that the real lever turned out to be not
re-deriving decided answers (`skip_heads`, `baseline_dirs`) rather than running faster.

**2. The plan audited the decision path and never looked at the delivery path.** Nothing in it
noticed that two accepted kernels **could not be run together** — `GEAK_OVERLAY` took one manifest,
and the two overlays rebind different symbols. The workflow could accept N candidates and deliver
none of them as a stack. A roadmap for a system that *decides* things should ask, at every stage,
what happens to the things it decided.

**3. "Decide the envelope's fate" was too small a question.** Deciding it exposed that the problem
was not one detector: the sign test fell to the same control within a week. The right question was
**"what can L1 actually detect, and what is left when the undemonstrable parts are removed?"** — and
the answer is: the step-1 probe, plus a sign test that cannot tell a fix from a bug.

## Month 1 — status against the original plan

| week | item | status |
|---|---|---|
| W1 | decide the L1 envelope's fate | ✅ **retired from voting**, on 4-run evidence |
| W1 | feed the attention defect back to the kernel layer | ✅ both channels (in-run constraint, cross-run KB card) |
| W1 | rejection diagnostics become KB inputs | ⚠️ **gap**: fires on L0/L1 failure only, so the *inf-gradient* candidate — the most catastrophic failure available — produced no card |
| W2 | ConfigSweep / `deep` mode | ❌ not started |
| W3 | L2 soak with a correct positive-control class | ❌ not started |
| W4 | iteration speed | ⚠️ 10.6 h → 7.9 h, head count unchanged; criterion was wrong (see above) |
| W4 | TP > 1, packed coverage | ❌ still refused / untouched |
| — | **multi-overlay delivery + combined measurement** | ✅ **unplanned, and the most valuable thing done** |

## Revised remainder, in priority order

### P1 — DONE (2026-10-08): the original +1.147% was RIGHT; my "recovery" was a reference mismatch

**Correction to this plan's own first draft.** P1 originally said the +1.147% was lost to a *reused
noisy baseline leg* and that the quietest-leg fix would recover it. Checking before acting showed
that was wrong: the 2.467% band came from the **gate legs' own** dispersion — reference IQR
**3.324%**, candidate 1.316% — where this workload normally gives 0.05–0.11%. The baseline fix does
not touch it.

What the check did find is worse and more useful: the run reported **`no_win`** when the truth was
**"we could not measure"**. The band was **16.4× the workload floor**, i.e. set by the legs rather
than by the workload, and on quiet legs the same delta clears the 0.15% floor outright. Now fixed —
`gateVerdict` returns `inconclusive_noise` with `remeasure: true` in that regime, and it does not
count toward the stop rule, because a leg too noisy to measure says nothing about whether candidates
are paying off.

So P1 is a re-measurement, not a recomputation: one A/B pair on quiet legs. The overlay is on disk
and already passed L0.

### P2 — DONE (2026-10-08): the L0/L1 rule is written, and narrower than it looked

This is a **framework** gap, not a candidate problem, and it will recur. L0 says a candidate is
strictly more accurate than the incumbent on every output; L1's sign test rejects it for descending
one-directionally. Both statements are true. The framework has no rule, so the verdict today depends
on which layer is read last.

A defensible rule, to be written down *before* it is applied to this candidate: **when L0 rates a
candidate strictly more accurate than the incumbent on all outputs, a one-sided loss improvement is
corroboration rather than evidence of bias, and the sign test may not reject on it alone.** The
dangerous case it must not cover is a candidate that is *less* accurate and still descends faster.

### P3 — ConfigSweep / `deep` mode — **DONE 2026-10-08**

A `deep` run completed end to end: 7 agents, 0 errors, 64 minutes, exit criterion met. ConfigSweep
proposed comm overlap, measured a matched pair, and was gated through the kernel path's own rule —
`no_win`, aligned delta **+0.1178%** inside a 0.15% band. Two things are worth more than the verdict:

* **the unaligned number had the opposite sign** (−0.212%) with a window artifact of 0.3298%, larger
  than the signal itself. Reporting the raw ratio would have been mostly artifact, with a sign.
* **the lever COSTS 780 MB** (+0.97%), it does not bank any — overlapped param-gather needs live
  all-gather buffers the serialized path reuses. So the memory-conversion path is exercised in the
  direction nobody designed for, and stacking this with `--recompute-granularity selective` would be
  spending memory this lever already consumed.

**The run found a defect in the layer underneath it.** Both legs reported identical `config_digest`
AND `impl_digest` with only `argv_hash` moving: no `overlap-*` key existed in any of the three key
lists, so the snapshot was blind to the only thing that differed. The Config Tuner established the
controlled comparison by diffing Megatron's 686-line arg dump instead — i.e. by routing around the
instrument whose job that was. Inspection then found the mirror image, `attention-backend`
classified in `IMPL_KEYS` and captured by nothing. Fixed at schema v4, which needs **no re-measure**
(verified on four legs), and pinned by `scripts/tests/test_identity_keys.py`.

**The sequencing note below was wrong, and harmlessly so.** It predicted a config change would move
`config_digest` by design. It does not and should not: the workload is the same, only the
implementation differs. What actually happened is that it moved *neither* digest, which is the
defect above.

### P3 (original entry) — the largest remaining lever by a wide margin

The kernel track at 7B is visibly converging: queue expectations fell from 5.82% to **1.37% / 1.24%**.
Meanwhile every run has reported the same unexploited opportunity: **70.1% of tokens are padding**.
Packing raises `useful_tokens` *directly*, where a kernel win only shortens `step_time` — and the
remaining heads are worth ~1% each.

Seam hooks 1–6 already exist in `fast` with trivial behaviour, so this is wiring a designed seam.
**Exit criterion:** a `deep` run that gates a config change through the same gates as a kernel
change, with the memory-conversion path exercised.

**Sequencing note:** do P3 *after* P2. A config change moves `config_digest` by design, so every
sweep leg is a new identity; starting that while a gate rule is unsettled compounds two unknowns.

### P4 — L2 soak — **DONE 2026-10-08: the answer was to RETIRE it**

The control was built and is demonstrated invisible to L0 **by construction** — cross-step
staleness, L0's output bit-identical at alpha = 1.0, at the same line where a x1.004 bias is
rejected decisively. Then it was run exactly as pre-registered: 4 positive legs at alpha = 0.25
against 4 clean overlay-matched legs, with engagement verified first at 19 200 fwd + 19 200 bwd per
leg.

**Null control silent. Positive control NOT rejected** — Welch t = 0.375 against crit 2.780, delta
+0.00576 (0.63 sigma). Positive terminal losses 1.4226 / 1.4172 / **1.4777** / 1.4179: dispersion,
not location, for the third positive control running.

The disposition was committed to in writing before the data, so it can be acted on without
arguing. **The soak is retired as a gate** — the Convergence Auditor now always reports
`l2_recommended: false`. The arithmetic:

* at alpha = 1.0, L1's sign test rejects this defect **for free**, in a leg that was running anyway;
* at alpha = 0.25 where L1 is silent, closing the gap from 0.00576 to an MDE of 0.01587 needs
  ~7.6x more legs — **n = m ≈ 30, about 30 GPU-hours** — for a class L1 catches one notch up;
* the op-local class the soak was first built for is **L0's**, at 11.5 s.

Three control generations, ~17 GPU-hours, zero rejections. Between L0 and L1 there is no band of
defects that 300 steps and a handful of legs can resolve.

**The control is not wasted**: it becomes L1's positive control — the first one L1 has ever had of a
class L0 cannot see, which closes the "L1 is thin" item as well.

**The token-aligned Class-B comparator fails the same control on the same legs** (positive medians
all inside the null range; one *clean* pair separates further than any defective leg), so it is not
promoted to a gate and `config_tuner` keeps refusing Class B. `scripts/trajectory_tokens.py` stays
as a characterised instrument — 300 steps or nothing, median is the usable statistic.

### P5 — Close the KB gap found in W1 — **DONE**

`roles/convergence_auditor.md` now fires on "whose fault is it" rather than on which gate voted: a
card is written whenever the CANDIDATE is at fault — `l0.pass` false, `l1.pass` false, a measured
regression, **or `CAND_USABLE: false` with a diagnosis in `INTEGRATOR_NOTES`**, which is the case
that was missing (a candidate that diverged, hung or crashed never reached a gate, so no gate could
vote). It deliberately does NOT fire when the fault is the measurement — a band too wide, a missing
aligned comparison — because a card claiming the kernel was bad on that evidence is worse than none.

## Month 2 — adaptation: other stacks, other cases, tuning

Gated, as before: if the gates are not trustworthy, adapting the instrument multiplies the fiction
across stacks. That gate is **closer to met than it was** — the throughput gate and L0 both have two
demonstrated controls — but L1 is now one probe plus an unresolved rule, so read "trustworthy" as
"trustworthy for throughput and op numerics; weak for trajectory".

### A second training stack

`scripts/adapters/<stack>.sh` still has exactly one implementation, so the seam is **unvalidated**.
What it actually tests: whether the probe is genuinely stack-neutral (it exists because upstream
Megatron reports no tokens/s; a stack that *does* will expose whether ours is comparable), whether
the config IR survives or is Megatron vocabulary in an IR's clothes, and whether the capability
manifest routes correctly.

**Exit criterion:** one accepted kernel on the second stack through the same gates, with the adapter
diff confined to `scripts/adapters/`. Anything forced outside that directory is a leak — record it.

### Other cases, by information per GPU-hour

| case | why | what it probably breaks |
|---|---|---|
| **Packed sequences** | the largest known lever; also the stated limit of the current +12.36% | attention isolation across segments; the varlen/rope path already needed squeezing at MBS=1 |
| **TP > 1 / PP > 1** | `config_digest` claims to cover the parallel mapping but has covered it on *no leg ever measured* | vocab-parallel references (the LM-head capture already refuses TP>1); per-rank capture; the straggler bucket, empty at TP1×DP8 and untested above |
| **A different model family** (MoE) | the head op's *class* already changes with scale | routing/expert ops have no oracle; the critical-path model assumes dense |
| **Longer sequence / fp8** | changes the arithmetic intensity the whole ranking rests on | fp8 needs new L0 tolerances — the current ones anchor on the original's own bf16 error |

### Campaign mode and per-workload bands

KB identity is already `config_digest`-hashed, so carrying knowledge across workloads is close. Each
new workload needs its own `ACROSS_JOB_FLOOR_PCT` measured into `workloads/*.env`; and the cheapest
precision lever remains **window length**, not the band — ≥29 steady steps halved the 0.5B floor.

## Cross-cutting, every week

- **No gate ships without both controls.** A null control it must pass and a positive control it must
  fail. This caught three defects this month and cost nothing compared to what it saved.
- **Pre-register thresholds before the data that judges them.** It worked: two L2 criteria turned out
  defective and were withdrawn *without* retuning K.
- **Run artifacts do not live in the repo.** One report reached the repo root because a role card
  named no path.
- **Every claim in DECISIONS.md names a number and where it came from.** Keep it that way; it is the
  reason this month's wrong turns were findable.
- **Audit the delivery path, not just the decision path.** Two accepted kernels could not be run
  together for a fortnight and nothing noticed, because every check asked "is this verdict sound?"
  and none asked "can we ship what we accepted?".
- **Writing the warning is not the same as heeding it.** The combined-stack measurement failed its
  first attempt on exactly the per-run-manifest trap that the bundle's `launch.sh` warns about and
  that `l2_soak.sh` guards against — both of which this project had already written down.

## Known risks

1. **Shared box.** Occupancy contamination is already handled (bounded re-measure, then fatal), but
   it caps throughput unpredictably and Month 2's schedule assumes a lot of GPU.
2. **One node.** Multi-node is unverified and the overlay mechanism explicitly does not survive it
   (`torchrun` inherits the parent environment on one node; multi-node needs per-node materialisation).
3. **Chaos is the noise floor.** Measured: no statistic averaged *within* a run buys power, because
   divergence is serially correlated. Any future convergence work must buy power with **legs**, not
   steps — this is a permanent property of the problem, not a current limitation.
4. **The plan is gate-heavy and may read as slow.** The counter-evidence is this fortnight: the
   headline number survived every audit — and is now a *measured* +12.36% rather than an arithmetic
   guess — while **five** gates or detectors did not survive. The gates were the risk.
5. **L1 is thin and the project should say so.** After two retirements it is one zero-noise probe
   plus a sign test that cannot distinguish a correctness fix from a bias. Trajectory-level
   confidence currently rests on L0 being strong, not on L1 being strong.
