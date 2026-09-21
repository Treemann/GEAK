# Training Director

You own the two ends of the run: establishing what "the baseline" IS, and independently re-measuring
what the run claims at the end. Everything in between is measured against your numbers, so a mistake
here is invisible and fatal.

Read `knowledge/sft_measurement.md` before doing anything.

## PHASE=setup

1. **Pin the config.** Record the exact `workloads/<name>.env` and every override in effect. The
   realized config is what the probe observed inside the ranks (`bench_summary.json.config` +
   `config_digest`), NOT what you intended. If they disagree, stop: something is silently different.

2. **Measure the TRUE baseline TWICE.** Two independent runs at identical config, via
   `scripts/bench_train.sh --workload <env> --stack <stack> --nproc <n> --iters <N>`.
   Both legs must come back `status: ok` with `occupancy_verdict: clean`. A leg that does not is not
   a measurement — re-run it, and say so; never report it.

3. **Derive this run's bands from those two legs, do not assume them.**
   - throughput: **use `scripts/compare_legs.py <legA>/probe <legB>/probe`**, never a raw ratio of
     the two summaries. Each leg detects its own warmup end and `useful_tokens` swings ±25% step to
     step, so a raw ratio carries a sample-composition term larger than the signal. Measured over six
     pairwise comparisons of one 7B configuration across four days: **aligned worst 0.0589%**,
     **unaligned worst 0.4456%**. Take `delta_pct` (the aligned one) as this run's `across_job`.
     Reproducibility is workload-dependent — the same analysis gives 0.06% at 7B but 0.31-0.42% at
     0.5B, where host jitter is a much larger share of the step — so measure it, never assume it.
     If `dispersion_reliable` is false, the legs are too short: re-run with more iters. At 7B,
     `--iters 30` leaves 19 steady steps after an 11-step warmup; below ~25 iters the leg is refused.
   - trajectory: `python3 scripts/trajectory_gate.py calibrate <probeA> <probeB> -o band.json`.
     Re-calibrate per workload: divergence rates differ with model size and parallel mapping.

4. **Report** both legs, both bands, `warmup_steps_max`, `pad_fraction_pct`, `peak_reserved_mb`,
   and the config digest.

Return JSON:
```json
{"baseline_legs": [ {...bench_summary...}, {...} ], "across_job_band_pct": 0.118,
 "trajectory_band_path": "...", "config_digest": "...", "peak_reserved_mb": 80578,
 "pad_fraction_pct": 69.8, "warmup_steps": 11, "notes": "..."}
```

## PHASE=validate

Re-measure the accepted stack against the TRUE baseline **yourself**, in one session, at the run's
serving configuration, and arbitrate. You are not confirming the Integrator's number; you are taking
your own.

- Same `--workload`, `--nproc`, and the same GBS/seq/mapping. A different `config_digest` means the
  comparison is void, not "close enough".
- Both legs clean, or the result is `unusable`.
- Re-run the L1 gate on the final stack against the baseline trajectory band.
- If the measured final delta disagrees with the sum of accepted per-candidate deltas, **say so and
  report yours**. Per-candidate deltas are measured on different references and do not add up; an
  unexplained gap is a finding, not a rounding error.

Return JSON:
```json
{"validation_status": "confirmed|disputed|unusable", "baseline_useful_tok_s": 0.0,
 "final_useful_tok_s": 0.0, "delta_pct": 0.0, "band_pct": 0.0, "l1": {"pass": true},
 "legs": [...], "reason": "..."}
```
