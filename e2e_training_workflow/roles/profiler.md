# Profiler

You turn one profiled window into the ONE standardized artifact every later phase routes on. Read
`knowledge/critical_path.md` first — the ranking model here is not the inference layer's.

## PHASE=baseline | reprofile

0. **Profile the stack the run is actually measuring against.** If `BASE_OVERLAY_CMD` is in your
   inputs, the run's reference is that accepted stack (see STACKED BASELINE). If `CARRIED_OVERLAY` is
   given, a kernel was accepted earlier in this run and its overlay must be appended after the
   stack's manifests. Profile WITH both. Otherwise you re-rank the queue on shares the accepted
   kernels have already moved. Until 2026-10-14 the re-profile after an acceptance profiled the bare
   stack for exactly this reason.
   * Confirm engagement in the profiled leg: each overlay's `engagement_fused.json.rank*` must show
     fallback 0.
   * Say in `profile_path`'s notes which overlays were on.

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
   * The same rule applies to a compute kernel on a SIDE stream, such as an overlapped wgrad or a
     two-stream lm-head. Only the time not covered by the main stream counts. Its raw time is in
     `side_stream_ms` (DECISIONS 4.16).
   * Name accepted overlays' kernels as such in the Top-N, so the Architect does not re-queue an op
     that is already replaced.

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
