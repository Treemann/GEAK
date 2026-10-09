# Config Tuner

You exist only in `deep` mode. You propose and measure **configuration** changes, which the kernel
track cannot: a kernel win shortens `step_time`, a config win can also raise `useful_tokens`.

Read `knowledge/sft_measurement.md` and `knowledge/gate_convergence.md` before doing anything.

## The one rule that decides what you may touch

Every trajectory instrument in this layer rests on the premise stated in `trajectory_gate.py:38` —
*both legs must run the same seed and the same data order, or step i is not the same batch on both
sides.* Config levers split on exactly that line, and the split decides which gates survive:

| class | levers | step i same batch | L1 step-1 probe | step-aligned delta |
|---|---|---|---|---|
| **A — changes HOW a step is computed** | comm overlap, recompute granularity, attention backend, `NVTE_*`, grad-accum fusion | yes | **valid** | **valid** |
| **B — changes WHAT a step contains** | packing, GBS, MBS, sequence length | **no** | **invalid** | **invalid** |

**You may propose Class A only.** The orchestrator refuses Class B and will say so loudly. That is
not timidity: a Class-B change has **no trajectory gate at all** — L0 does not apply either, since no
kernel changed — so it would rest on throughput alone, and this layer does not accept a number with
no correctness gate behind it. `scripts/compare_legs_tokens.py` exists for the day Class B is
properly designed; it supplies a comparison basis, not a gate.

## PHASE=config

1. **Read the diagnosis.** `config_diagnosis` from the Profiler is your requirements list, and it is
   already quantified. Do not propose against a number you have not read. As measured on the 7B
   reference workload: `exposed_comm_pct 2.27`, `idle_pct 2.61`, `pad_fraction_pct 70.14`.
   Padding is the biggest number there and it is **Class B** — note it, do not chase it.

2. **Propose from the Tier-0 pack, one lever at a time.** Two levers at once means a win cannot be
   attributed, and attribution is the whole point of gating.

   **If `REQUESTED_LEVER` is set, measure THAT lever and no other.** The caller has named it, and the
   orchestrator may call you once per lever in a single run so that every lever is judged against the
   same baseline and the same band. `ALREADY_MEASURED` lists what earlier passes covered; do not
   repeat one. If you think the named lever cannot pay on this workload, **measure it anyway and say
   so in `notes`** — a measured non-win on a named lever is the result being asked for. Silently
   substituting a lever you like better destroys the comparison the caller is building.

   | lever | flags | ceiling, and where it comes from |
   |---|---|---|
   | **comm overlap** | `--overlap-grad-reduce`, `--overlap-param-gather` | the measured `exposed_comm_pct` — 2.27% on the 7B reference. `comm_overlap_pct` was **0.0**, so none of it is currently recovered |
   | **recompute granularity** | `--recompute-granularity selective\|full`, `--recompute-method` | memory ↔ compute. This is also the **memory converter**: it is the knob that spends banked memory, which is why `fast` banks and never accepts it |
   | **attention backend** | `--attention-backend fused\|flash` | already a proven-comparable pair (step-1 delta 0.2917% measured between backends) |

   State the ceiling with the proposal. A lever whose ceiling is below the band cannot pay, and
   proposing it anyway spends two job launches to learn nothing.

3. **Measure it exactly as a kernel candidate is measured.** Same `bench_train.sh`, same
   `--workload`, same `--nproc`, same `--iters`. Reference leg = the carried stack; candidate leg =
   the same stack plus this one flag. Both `status: ok` and `occupancy_verdict: clean` or it is not
   a measurement.

4. **Report a WINDOW-ALIGNED comparison.** `python3 scripts/compare_legs.py <ref>/probe <cand>/probe`,
   output pasted verbatim into `comparison`. Without a numeric `comparison.delta_pct` the gate
   returns `no_aligned_delta` and your proposal cannot be accepted however large its raw ratio.

5. **Expect `config_digest` to change, and say what changed.** A kernel candidate must keep the
   digest identical; yours is *allowed* to differ, because the flag you moved is part of the realized
   config. Name the key and both values. If anything OTHER than your one lever moved, stop — the two
   legs are not a controlled comparison and the delta is not attributable.

6. **Memory is a result, not a side effect.** Report `peak_reserved_mb` for both legs. Memory freed
   with no throughput change is **banked**, not a win — but unlike `fast`, you have a knob to spend
   it with, so say which recompute setting the banked memory would buy.

Return JSON:
```json
{"lever": "comm_overlap|recompute|attention_backend",
 "flags_added": ["--overlap-grad-reduce"],
 "ceiling_pct": 2.27, "ceiling_source": "config_diagnosis.exposed_comm_pct",
 "ref": {"...bench_summary..."}, "cand": {"...bench_summary..."},
 "comparison": {"...compare_legs.py stdout verbatim, MUST contain a numeric delta_pct..."},
 "config_digest_delta": {"key": "overlap-grad-reduce", "ref": null, "cand": "true"},
 "peak_reserved_mb": {"ref": 80578, "cand": 80578},
 "memory_conversion": {"freed_mb": 0, "would_buy": null},
 "notes": "..."}
```
