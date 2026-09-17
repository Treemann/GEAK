# Profiler

You turn one profiled window into the ONE standardized artifact every later phase routes on. Read
`knowledge/critical_path.md` first — the ranking model here is not the inference layer's.

## PHASE=baseline | reprofile

1. **Take a profiled run, and know it is not a measurement.**
   `scripts/bench_train.sh ... --profile-steps <a>:<b> --profile-ranks 0,<one more>`
   Choose `<a>` **after** the warmup ends (`warmup_steps_max` from the baseline; on this box that is
   ~10 steps at 7B). Profile 2 steps, not more — traces grow fast and buy nothing.
   The dispatcher marks the run `profiled_run: true` and `status: unusable` on purpose: the window
   perturbs the steps it traces, so this run can never be a gate leg.

2. **Decompose and rank.**
   `python3 scripts/parse_train_profile.py <probe_dir> <bench_summary.json>`
   Report the four buckets (`compute / exposed_comm / idle` + rank skew) and the Top-N ordered by
   `critical_path_share_pct` — NOT by share of GPU time. A collective that is fully overlapped is
   worth zero no matter how large it is.

3. **Profile more than one rank.** The step is as slow as its slowest rank. Report per-rank buckets
   and `rank_skew_pct`; do not average skew away, it is the straggler bucket's only input.

4. **Emit the read-only config diagnosis.** Verdict `kernel_work_can_pay` or `config_first`, with the
   offending knob named. You do NOT fix anything — `fast` mode has no ConfigSweep. Saying "62% of the
   step is exposed grad-reduce" is what makes a later rejected kernel win attributable to the
   configuration instead of blamed on the kernel.

5. **Attach shapes.** The Top-N entries carry `op` and `input_dims` from the trace's cpu_op linkage.
   The Kernel Extractor consumes them; an entry without shapes is much less useful, so say which ones
   lack them rather than silently dropping the column.

Return JSON:
```json
{"buckets": {...}, "top_n": [ {"kernel": "...", "family": "...", "critical_path_share_pct": 0.0,
  "op": "...", "input_dims": [...]} ], "per_rank_buckets": {...}, "rank_skew_pct": 0.0,
 "config_diagnosis": {"verdict": "...", "blocking": [...]}, "profile_path": "..."}
```
