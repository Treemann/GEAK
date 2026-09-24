# e2e_training_workflow — 2-month roadmap (DRAFT)

Status at the time of writing: `fast` mode runs end to end. Four real runs exist (0.5B/2GPU ×1,
7B/8GPU ×3) plus 15 soak legs. One kernel is accepted and independently re-validated at **+4.66%**.
Everything below is ordered by two constraints that were **measured this month**, not assumed:

**Constraint A — a gate that cannot discriminate makes every later number fiction.** Three gates were
found non-discriminating *after* they had already issued verdicts: the throughput gate accepted on a
raw ratio where window placement alone manufactures +4.59%; the L1 envelope rejected provably-clean
legs (out-of-sample null control fails 3/3); the L2 soak passed a kernel with a deliberate ×1.004
gradient bias. **Gate work therefore precedes optimization work in this plan** — not out of caution,
but because optimization output is unfalsifiable until the gates are.

**Constraint B — one run costs 8–10 h and yields ~2 heads.** The last 7B run: 69 agents, 1331 tool
calls, 8 h 15 m, 2 of 4 queued heads, stopped on the wall clock. At that rate a month holds roughly
60–80 runs of wall time *if the box were exclusive*, which it is not. **Iteration speed, not agent
capability, is the binding constraint on everything in Month 2.**

Assumptions (flag if wrong — they change the plan materially): one 8×MI308X node, shared;
`rocm/primus:v26.4/5`; bare Megatron + the `--sft` path; no multi-node access yet; effort is
primarily this workflow rather than the kernel layer underneath it.

---

## Month 1 — make the thing correct, complete, and fast enough to iterate

### Week 1 — close the open gate defects and the feedback loop

The gates are the product. Three concrete items, all already diagnosed:

| item | evidence | exit criterion |
|---|---|---|
| **Decide the fate of the L1 per-step envelope** | Falsified out-of-sample twice in one session (5/5/6 exceedances on true-zero-delta pairs); structurally can't see a +25% steady-window loss shift because identical legs already disagree ~8.9% by step 12 | Either an envelope that passes an out-of-sample control on ≥3 workloads, or the envelope is **formally retired** and L1 is documented as = step-1 probe + sign balance. Do not leave it half-alive. |
| **Feed the attention backward defect back to the kernel layer** | step-1 grad_norm 74.109 → 8.524 (−88.50%, 100–200× the noise floor) with the forward clean (p=0 pair −0.1999%) | A rejection emits a machine-readable diagnostic the kernel lane consumes as a starting constraint; re-running that head does not re-discover it from zero |
| **Rejection diagnostics become KB inputs** | KB learned cards exist but record *wins*; the attention rejection's isolating experiment (p=0 matched pair) was re-invented by hand | A rejected candidate writes a card; the next lane on the same op reads it |

**Why first:** these are the only items where the analysis is already done and only the plumbing is
missing. They are also what stops Week 2 from producing numbers nobody can trust.

### Week 2 — ConfigSweep / `deep` mode

The single largest missing feature. `deep_mode: true` currently **refuses to run**, and `fast` mode's
read-only diagnosis already emits its requirements list:

- **71.3% of tokens are padding.** Packing raises `useful_tokens` *directly*; a kernel win only
  shortens `step_time`. This is arithmetically the larger lever and it has been sitting in the
  diagnosis output of every run.
- `comm_overlap_pct = 0.0` — latent, scales with model size and DP degree.

Seam hooks 1–6 exist in `fast` with trivial behaviour, deliberately (DECISIONS 5.4), so this is
wiring a designed seam rather than inventing one.

**Exit criterion:** a `deep` run that proposes, measures and gates at least one config change through
the same gates as a kernel change, with the **memory-conversion** path exercised (today memory is
banked because there is no knob to spend it with).

**Risk:** config changes alter `config_digest` by design, so every sweep leg is a new identity.
Budget for the comparison discipline, not just the sweep.

### Week 3 — L2 that can speak, and the soak's real target

L2 is named in the docs, auto-triggers on `risk_class: backward`, and **has never issued a verdict
that carries weight**. Two runs (~15 h) both ended ADVISORY.

The reframing from Week-0 evidence: **op-local numerics is L0's job** — L0 rejects the ×1.004 bias in
~27 s while two 300-step soaks miss it entirely. So the soak's positive control must be a defect of a
class L0 *structurally cannot see*, because L0 evaluates one frozen microbatch:

- cross-step state corruption (the accepted lmhead kernel accumulates into `main_grad` — precisely
  this class)
- accumulation error that only compounds over ~19 200 invocations
- allocator / fragmentation effects
- non-determinism that averages out in one call but not over an epoch

**Exit criterion:** a positive control of that class, injected deliberately, that the soak **rejects**
— and that L0 **passes**. Until both halves hold, the soak stays unrun; ~10 h that cannot demonstrate
its own sensitivity buys nothing.

### Week 4 — iteration speed, then coverage

**Speed (Constraint B).** Today: Setup takes 3 baseline legs per run (~25 min at 7B) and every run
re-profiles from the bare stack. Candidates:
- persist and reuse baselines keyed by `config_digest` + `digest_schema` (the digest now actually
  covers identity, since schema 3)
- resume a stopped run from `unexplored_heads` instead of re-deriving the queue
- overlap lane work with gate legs where the barrier allows

**Exit criterion:** ≥3 heads per run at the same wall clock, measured.

**Coverage.** Two known holes, both currently *refused* rather than wrong — which is the right state,
but they bound applicability:
- **TP > 1**: the LM-head capture now refuses it (the oracle would freeze a shard-local logsumexp).
  Unblocking needs a vocab-parallel reference.
- **Packed data**: the accepted +4.66% is specific to ~70% padding and **shrinks toward zero** when
  packed. Any claim outside SFT-with-padding is currently unsupported.

---

## Month 2 — adaptation: other stacks, other cases, tuning

Month 2 assumes Month 1 delivered trustworthy gates and ≥3 heads/run. **If it did not, do not start
Month 2** — adapting an instrument that cannot discriminate multiplies the fiction across stacks.

### Week 5–6 — a second training stack

`scripts/adapters/<stack>.sh` is a designed seam with exactly one implementation (megatron), so the
seam is **unvalidated**: one implementation cannot show which assumptions leaked. Candidates in
rough order of value: Primus (same box, least distance), TorchTitan / DeepSpeed (different
abstractions, harder and therefore more informative).

What the second adapter actually tests, and why this is not busywork:
- Is the **probe** genuinely stack-neutral? It exists because upstream Megatron reports no tokens/s
  at all; a stack that *does* report its own number will expose whether ours is comparable.
- Does the **config IR** survive, or is it Megatron vocabulary wearing an IR's clothes (DECISIONS 6.1)?
- Does the **capability manifest** route correctly — it has already caught one over-claim
  (`step_ranged_profiler`).

**Exit criterion:** one accepted kernel on the second stack, gated by the same gates, with the
adapter diff confined to `scripts/adapters/`. Any change forced *outside* that directory is a leak
and should be recorded as such.

### Week 7 — other cases

In rough order of expected information per GPU-hour:

| case | why | what it probably breaks |
|---|---|---|
| **Packed sequences** | the largest known lever, and the current win's stated limitation | attention isolation across segments; the rope/varlen path already needed squeezing at MBS=1 |
| **TP > 1 / PP > 1** | `config_digest` claims to cover the parallel mapping but was covered on *no leg ever measured* | vocab-parallel references; per-rank capture; straggler bucket (empty at TP1×DP8, untested above) |
| **A different model family** (e.g. MoE) | head-op *class* already changes with scale (0.5B → attention bwd; 7B → GEMM) | routing/expert ops have no oracle; critical-path model assumes dense |
| **Longer sequence / different precision** | changes the arithmetic intensity the whole ranking rests on | fp8 would need new L0 tolerances — the current ones are anchored on the original's own bf16 error |

### Week 8 — tuning and campaign mode

- **Campaign mode**: several workloads, one budget, KB carried across. The KB identity is already
  `config_digest`-hashed (seam hook 6), so this is closer than it looks.
- **Revisit the band per workload class.** `ACROSS_JOB_FLOOR_PCT` now lives in `workloads/*.env`
  (0.15% at 7B, 0.5% at 0.5B) from 47 measured null pairs. Each new workload needs its own, and the
  cheapest lever remains **window length**, not the band: ≥29 steady steps halved the 0.5B noise
  floor.
- **Close the loop on the three unexplored heads** (swiglu 1.13%, mlp_gate_up 1.05%, rope 0.78%) once
  runs are cheap enough that ~1% items are worth a gate.

---

## Cross-cutting, every week

- **No gate ships without both controls.** A null control it must pass and a positive control it must
  fail. This caught three defects this month and cost nothing compared to what it saved.
- **Pre-register thresholds before the data that judges them.** It worked: two L2 criteria turned out
  defective and were withdrawn *without* retuning K.
- **Run artifacts do not live in the repo.** One report reached the repo root because a role card
  named no path.
- **Every claim in DECISIONS.md names a number and where it came from.** Keep it that way; it is the
  reason this month's wrong turns were findable.

## Known risks

1. **Shared box.** Occupancy contamination is already handled (bounded re-measure, then fatal), but
   it caps throughput unpredictably and Month 2's schedule assumes a lot of GPU.
2. **One node.** Multi-node is unverified and the overlay mechanism explicitly does not survive it
   (`torchrun` inherits the parent environment on one node; multi-node needs per-node materialisation).
3. **Chaos is the noise floor.** Measured: no statistic averaged *within* a run buys power, because
   divergence is serially correlated. Any future convergence work must buy power with **legs**, not
   steps — this is a permanent property of the problem, not a current limitation.
4. **The plan is gate-heavy and may read as slow.** The counter-evidence is this month: the headline
   number survived every audit, while three gates did not. The gates were the risk.
