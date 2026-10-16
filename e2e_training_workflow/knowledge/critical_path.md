# The bottleneck model: critical path, not %GPU time

## Why the inference formula misleads here
`e2e_workflow` ranks candidates by `pct_gpu_time x achievable_speedup`. In training that is wrong in
a specific, silent way: **a collective fully overlapped with compute can own 30% of GPU time and be
worth exactly zero**, because removing it removes nothing from the step's critical path.

So `scripts/parse_train_profile.py` decomposes each step into four buckets and ranks on the
recoverable part only:

| bucket | meaning | recoverable by |
|---|---|---|
| `compute` | GPU busy with non-communication kernels | the kernel track |
| `exposed_comm` | communication NOT hidden behind compute | comm config (overlap, buckets), Tier-0 |
| `idle` | no kernel resident: host/dataloader stalls, launch-bound, PP bubble | data pipeline, launch, schedule |
| straggler | cross-rank skew (from the probe's per-rank medians) | placement / mapping |

`critical_path_share_pct` for a compute kernel is its share of the step span; for a communication
kernel it is scaled by the **exposed fraction** (`exposed_comm / comm_total`). That single scaling is
what stops the router from spending the budget on a collective that is already hidden.

Ranking input for the Architect:
`expected_gain_pct = critical_path_share_pct x (1 - 1/attainable_speedup)`

## Multi-rank sampling is not optional
The step is as slow as its slowest rank, so the run-level median takes the **max** across ranks and
reports `rank_skew_pct` separately rather than averaging it away — skew is the straggler bucket's
only input. Profile at least one rank per TP group plus DP rank 0. (Measured at TP1xDP8 on this box,
skew is 0.002-0.013%, i.e. the straggler bucket is empty at this scale; re-check with TP>1.)

## The config diagnosis (fast mode's substitute for ConfigSweep)
`fast` has no ConfigSweep, so it goes straight to the expensive lever. The hazard: if the step is
dominated by exposed comm, idle, or padding, **no kernel win can show up end to end**, and the budget
burns producing isolated speedups the gate then correctly rejects.

`config_diagnosis()` therefore emits a read-only verdict — `kernel_work_can_pay` or `config_first` —
naming the offending knob and why it blocks. It **changes nothing**. It also doubles as ConfigSweep's
requirements list when deep mode lands.

**Two categories, and they demand different answers.**
* `blocking` — the measurement cannot show a win however good the kernel:
  exposed comm > 15% (a shorter kernel exposes more comm), idle > 15% (a host-bound step converts a
  shorter kernel into more idle, not a shorter step), rank skew > 5% (the step is as slow as its
  slowest rank).
* `opportunity` — a larger lever exists, but kernel wins still pay in full: padding > 40%.

**Padding is NOT a blocker, and the first version of this file said it was.** The headline is
`useful_tokens / step_time`. A kernel win shortens `step_time` and leaves `useful_tokens` unchanged,
so the padding fraction cancels: a 20% win on a head that is 19% of the step is **+3.95% at 70%
padding and +3.95% at 0% padding**. What padding means is that packing would raise `useful_tokens`
directly and is a much larger lever — an opportunity-cost argument, not an impossibility argument.
Misclassifying it would have told the first real run to give up on work that would have paid.

## Read the Top-N `op` column as a hint, not an attribution (reprofile 2026-10-09)
`parse_train_profile.py` aggregates per **kernel name** and labels each entry with the FIRST cpu_op
it links to. hipBLASLt reuses one GEMM solution across several ops, so a shared kernel inherits
whichever op ran first. Measured at 7B (`/work/runs/reprofile_baseline_d1548a7a_1009b`, ranks 0+4,
3 steps, span 35.5 s): the #2 entry is labelled `aten::mm [4096,151680]x[151680,3584]` (lm-head
wgrad), but of its 6556 ms only 835 ms is lm-head — 3750 ms is MLP fc1 backward and 1971 ms fc2
backward. Likewise #1 (`_Linear [3584,3584]`, o_proj) is 4047 ms fc1 fwd + 2232 ms fc2 fwd + 415 ms
o_proj. Before a kernel is extracted from a Top-N entry, split it by External id
(`*_opsplit_rank0.json` next to the analysis does this). Per logical op, % of span:
MLP fc1 bwd 23.1, attention bwd (CK FMHA, group mode) 14.5, MLP fc1 fwd 11.4, fc2 fwd 6.3,
fc2 bwd ~11.8, lm-head fwd+bwd 5.8, QKV fwd+bwd 5.0, o_proj fwd+bwd 3.6, attention fwd 3.3,
SwiGLU (triton) fwd+bwd 3.5, RoPE elementwise ~2.4, exposed grad all-reduce 2.1 (0% overlapped),
idle 2.7. Verdict `kernel_work_can_pay`; GEMMs are ~70% of the step.

## Strategize 2026-10-09 (config d1548a7a, bare stack): route on dead rows, not on GEMM tuning
> **Historical (UNPACKED workload, 70% padding).** Packing (2026-10-09) removed the padding this
> section routes on, and every row-slice lever with it. The current routing and its results are in
> `sft_measurement.md` ("packed" sections) and the RUN.md queue.
The largest lever measured to date inside a kernel lane is FLOP deletion on padded rows, not vendor
GEMM re-selection: `mlp_fc2_down` got 1.88x isolated (+7.34% e2e) mostly by slicing both backward
GEMMs to the live-row prefix of `grad_output`; the fc1 lane that only re-selected GEMM solutions and
replaced the pointwise tail got 1.073x (and was rejected for an fp16 wgrad overflow). With 70% padding,
every backward GEMM whose `grad_output` has a zero row suffix carries the same lever. Forward cannot
use it (pad rows are not detectable from activations), so attainable speedup is bounded by the
backward share: `1 / (fwd_share + bwd_share * ~0.38)` (0.38 = 30% live rounded up to 256-row tiles
plus the detector).

Estimates used for routing (expected = share x (1 - 1/speedup)), heads NOT already accepted:
| head | share % | attainable | expected % |
|---|---|---|---|
| MLP fc1 (LayerNormLinear) fwd+bwd, bwd row-sliced | 34.47 | ~1.74x | ~14.7 |
| attention CK FMHA fwd+bwd, true per-sample seqlens | 17.80 | ~2.0x (conservative) | ~8.9 |
| QKV (LayerNormLinear) fwd+bwd, bwd row-sliced | 5.03 | ~1.70x | ~2.1 |
| SwiGLU triton bwd row-sliced (fold into fc1 if possible) | 3.49 | ~1.9x | ~1.7 |
| RoPE mul/cat/add fused | 2.43 | ~3x | ~1.6 |
Already accepted in earlier runs and NOT to be re-derived (carry the bundles via `measure_stack.sh`):
fc2 (+7.34%), lm-head CE gather (+4.66%), o_proj dense row-slice (+1.0% on top of fc2).
Constraints carried from rejections: wgrad must not accumulate in fp16 (fc1, Inf at iter 2); two
Triton attention backwards were rejected on trajectory/grad_norm, so the attention lane should keep
the CK kernel and change only the lengths it is given, and must first verify the reference really
runs 4096-long causal blocks (fwd:bwd ratio 1:4.5 at 30% live suggests it does; if not, the head
drops to ~1.15x and ~2%). `_LayerNormLinear` serves fc1 and QKV, `_Linear` serves fc2 and o_proj:
overlays on the same symbol must chain, as `attn_outproj_0928` chains fc2.
