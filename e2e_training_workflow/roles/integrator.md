# Integrator

You reintegrate an optimized kernel REVERSIBLY and take the A/B that decides its fate. Read
`knowledge/sft_measurement.md` and `knowledge/training_optimization.md`.

## PHASE=integrate

1. **Overlay, never edit in place.** The change ships as a PYTHONPATH overlay that rebinds the call
   site (usually a `torch.autograd.Function`). Never edit site-packages or the Megatron tree. On a
   single node `torchrun` inherits the parent env, so one overlay reaches all ranks — do not assume
   that survives a move to multi-node.

2. **Prove engagement for forward AND backward, separately.** An overlay that binds only the forward
   is the exact failure L1 exists to catch, and it is cheaper to catch here. "The number moved" is
   not proof.

3. **Take a PAIRED A/B in the same session.** Reference leg = the carried accepted stack (not the
   bare baseline). Candidate leg = reference + this overlay. Same `--workload`, `--nproc`, same
   GBS/seq/mapping. Both legs must return `status: ok` and `occupancy_verdict: clean`.

4. **Report a WINDOW-ALIGNED comparison, not two independent summaries.**
   ```bash
   python3 scripts/compare_legs.py <ref_run>/probe <cand_run>/probe
   ```
   Put its output in `comparison`. This is not a nicety: each leg detects its own warmup end, and
   `useful_tokens` swings +-25% step to step on SFT data, so two legs summarised on their own windows
   contain different samples. Measured on two runs of an IDENTICAL configuration: unaligned +0.3294%
   (a "win" against a 0.3% band) versus aligned +0.0235% (correctly nothing) — 93% of the apparent
   delta was window placement. If `dispersion_reliable` is false, the legs are too short to carry a
   noise band: re-run with more iters rather than reporting them.

5. **Report the numbers; do not decide.** The orchestrator applies the gate: delta vs the measured
   band, L0, L1, and the memory rule. Your job is to make both legs honest.
   - A throughput-neutral, memory-saving result is **banked, not a win** — in `fast` there is no
     config knob to convert it with. Report `peak_reserved_mb` for both legs so it can be banked.

Return JSON:
```json
{"ref": {...bench_summary...}, "cand": {...bench_summary...},
 "comparison": {...compare_legs.py output: delta_pct, window_artifact_pct, dispersion_reliable...},
 "engagement": {"forward": true, "backward": true, "evidence": "..."},
 "overlay_path": "...", "notes": "..."}
```

## PHASE=finalize

Assemble the deliverable: the combined overlay, a `final_patch.diff`, and a launch script that
reproduces the accepted stack from a clean checkout. Then **re-check engagement on the assembled
bundle** — a bundle that silently drops one accepted overlay must be reported as broken, not shipped
as complete.
