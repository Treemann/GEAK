# Levers above the kernel, for SFT training throughput

Headline metric: **useful (non-pad, loss-unmasked) tokens/s/GPU** on a steady-state step window, at a
fixed (dataset, global batch size, sequence length). Secondary: `peak_memory`. Diagnostic only:
TFLOP/s/GPU — never a gate, because recompute policy changes its denominator convention.

## The invariance envelope
| Fixed (changing it = cheating) | Free (that is the search space) |
|---|---|
| dataset + sample order, global batch size, sequence length | micro-batch size, grad-accum steps |
| optimizer + hyperparameters, LR schedule | TP / DP / CP / EP / SP mapping |
| **attention isolation semantics** | recompute granularity + method |
| numerics within the L0/L1 tolerance | kernel backends, fusions, comm knobs, dataloader |

**Packing is the boundary case.** It is inside the envelope only when attention stays isolated across
packed sub-samples (thd + `cu_seqlens`). Without that isolation, packed sub-samples attend each other
— that changes the mathematics, not the schedule, and must never be accepted as a throughput win.
This is the training analogue of "speculative decoding is not an optimization".

## The memory rule
A throughput-neutral, memory-saving change is **banked headroom, not a win**. It is accepted only
once the Architect converts it (raise MBS, drop a recompute level) and the *converted* configuration
clears the throughput gate. In `fast` mode there is no ConfigSweep to convert it with, so such a
result is recorded as `banked_memory_pct` and carried — never accepted. Keeps the headline honest
without discarding the lever that unlocks the next configuration.

Measured headroom on the reference workload: peak 78 GB of 192 GB, i.e. a large MBS / recompute
margin waiting for deep mode.

## Tier 0 — config only (deep mode)
Not available in `fast`; the config diagnosis reports these instead of fixing them.
1. Kernel-backend knobs (Primus-Turbo `use_turbo_*`; TE/FA backend selection on bare Megatron)
2. Recompute: granularity (selective|full), method (uniform|block), num layers
3. Parallel mapping: TP x DP (x CP for long seq, EP for MoE), sequence parallel
4. Comm: `overlap_grad_reduce`, `overlap_param_gather`, `use_distributed_optimizer`, bucket sizes, RCCL env
5. SFT data path: packing (with attention isolation), dataloader workers
6. Fusions: grad-accum fusion, rope / masked-softmax / swiglu, MoE permute
7. Precision: bf16 vs fp8 (TE) — a numerics change, therefore an L1 customer
8. MBS x grad-accum at fixed GBS — the memory<->throughput conversion knob
9. Offline GEMM tuning (hipBLASLt) for the exact SFT shapes

## Tier 1 — editable hot kernels (the `fast` track)
Extract with real shapes **and `grad_output`**; the task objective is **fwd+bwd combined** latency.
Optimizing forward alone is how you ship a kernel 2x faster forward and 3x slower backward.
Integration rebinds a `torch.autograd.Function`; engagement must be proven **separately for forward
and backward**. This track is framework-agnostic by construction: it targets libraries the trainers
share (TE, aiter, Primus-Turbo, FA), not the trainer.

## Tier 2 — overlap / scheduling / host
Kill exposed comm, fix launch-bound steps, native layouts. The host/launch half is reachable in
`fast`; the comm-overlap half needs config and therefore waits for deep.

## Stop rule
After each candidate, estimate remaining headroom as
`max over untouched heads of critical_path_share_pct x (1 - 1/attainable_speedup)`.
If the best remaining candidate cannot plausibly move the headline past the **measured** band, STOP.
With a 0.3% band, that threshold is low — which is the point of measuring the band instead of
assuming 2-3%.
