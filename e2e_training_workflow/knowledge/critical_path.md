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

Blocking thresholds: exposed comm > 15%, idle > 15%, padding > 40%, rank skew > 5%.
