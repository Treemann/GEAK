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
   bare baseline). Candidate leg = reference + this overlay.
   * **Stack order.** With `BASE_OVERLAY_CMD` set, the order is:
     1. the base stack's manifests, from running the command per leg;
     2. then every `CARRIED_OVERLAY` entry, IN LIST ORDER (it is a list: each accepted overlay composes
        on the previous one; dropping one gates this head against the wrong reference);
     3. then, on the candidate leg only, this overlay.
   * **Rebinding a symbol the stack already binds.** Example: a second TE `_LayerNormLinear` overlay.
     It must CHAIN. Capture the binding current at import and delegate the calls it does not handle,
     as `qkv_lnlinear_gemm` does after fc1. Prove both overlays' counters on the candidate leg.
   * **Proof location.** Each overlay must write its proof next to ITS OWN manifest. Locate the
     manifest by `__file__`, never take the first manifest in `GEAK_OVERLAY`. Same `--workload`, `--nproc`, same
   GBS/seq/mapping. Both legs must return `status: ok` and `occupancy_verdict: clean`.

4. **Use `ITERS` from your INPUTS for both gate legs. Do not lengthen them.**
   A longer leg gives more steady steps and a tighter window, and that instinct is right in
   isolation — but the L1 trajectory band was calibrated by Setup at `ITERS`, and a per-step envelope
   is fitted to legs of a particular argv AND a particular length. Change either and the band is out
   of scope: `trajectory_gate` will demote its envelope to advisory, and whoever audits the gate has
   to re-derive the verdict from fresh control pairs.
   This is not hypothetical. On 2026-09-24 the gate legs were run at 60 steps against a band
   calibrated at 30. The envelope returned a confident `pass=false`, and the Auditor then had to run
   extra reference pairs by hand to show that the same detector rejected a **true-zero-delta pair
   2.5× harder** than the candidate. Three ref/cand pairs were measured where one was budgeted — about
   an hour of 8-GPU time, to buy a tighter window that the band could not use anyway.
   If you genuinely need longer legs, say so in `notes` and expect L1 to be advisory; do not do it
   silently.

5. **Report a WINDOW-ALIGNED comparison, not two independent summaries.** Always run this; there is
   no case where two independent summaries are an acceptable substitute.
   ```bash
   python3 scripts/compare_legs.py <ref_run>/probe <cand_run>/probe
   ```
   Paste its output VERBATIM into `comparison` — the whole object, including `window_artifact_pct`
   and `dispersion_reliable`, not a summary of it. Without a numeric `comparison.delta_pct` the gate
   returns `no_aligned_delta` and your candidate CANNOT be accepted however large its raw ratio: two
   identical-config 7B legs measured +4.5925% unaligned against −0.0588% aligned at a 0.914% band, so
   an unaligned number is not evidence of a win. If compare_legs.py errors, report the error object
   as `comparison` and say so in `notes` — do not substitute the two independent summaries.
   This is not a nicety: each leg detects its own warmup end, and
   `useful_tokens` swings +-25% step to step on SFT data, so two legs summarised on their own windows
   contain different samples. Measured on two runs of an IDENTICAL configuration: unaligned +0.3294%
   (a "win" against a 0.3% band) versus aligned +0.0235% (correctly nothing) — 93% of the apparent
   delta was window placement. If `dispersion_reliable` is false, the legs are too short to carry a
   noise band — but do NOT silently lengthen them (see step 4): the L1 band is calibrated at the same
   `ITERS`, so lengthening fixes one gate by voiding another. Report it and say the run needs a
   larger `ITERS` for BOTH Setup and the gate.

6. **Report the numbers; do not decide.** The orchestrator applies the gate: delta vs the measured
   band, L0, L1, and the memory rule. Your job is to make both legs honest.
   - A throughput-neutral, memory-saving result is **banked, not a win** — in `fast` there is no
     config knob to convert it with. Report `peak_reserved_mb` for both legs so it can be banked.

### The bundle's launch script must survive a SECOND invocation

Install the overlay into a **per-invocation directory** (`.../overlay_$STAMP/`), never into one fixed
path that the script clears with `rm -rf` first.

This is not hypothetical and it is not cosmetic. The fc2 bundle's `launch.sh` did
`rm -rf "$WORK_HOST/overlays/bundle_fc2_0924"` before re-installing. That works exactly **once**: the
training container runs as root and leaves a root-owned `__pycache__/` behind, so the second
invocation dies with `rm: cannot remove ... Permission denied` and the candidate leg never starts.
The reference leg (`--ref`) keeps working, because it installs nothing — so the bundle appears half
healthy and the failure looks like a candidate problem rather than a script problem.

It survived every check we had because the bundle had only ever been run **once**, by the Integrator
that produced it. A cold start found it in twenty minutes. `l2_soak.sh` and `measure_stack.sh` both
hit the identical trap earlier and both solved it the same way: a fresh directory per run, and no
`rm -rf` of anything the container may own.

Return JSON:
```json
{"ref": {...bench_summary...}, "cand": {...bench_summary...},   // bench_summary.json VERBATIM, plus "probe_dir"
 // The orchestrator FAILS CLOSED on a trimmed leg: implementation, workload_identity, occupancy_verdict,
 // hang_detected, dispersion_reliable, worst_iqr_spread_pct, min_steady_steps, step_shape,
 // ranks_agree_on_argv and probe_dir (host path of <leg>/probe, for the Auditor) are all required.
 // The ref's GEAK_OVERLAY_SHA must equal REF_EXPECTED's; the cand's must extend it with this overlay.
 "comparison": {...compare_legs.py stdout verbatim — MUST contain a numeric delta_pct; also carries
                 window_artifact_pct, dispersion_reliable, window...},
 "engagement": {"forward": true, "backward": true, "evidence": "..."},   // booleans; anything else is 'not proven'
 "legs_measured": {"refs": ["<host leg dir>", ...], "cands": ["<host leg dir>", ...]},
 // EVERY leg you took for this gate, not just the reported pair: a gate verifier recomputes every
 // ref x cand pairing from these dirs (scripts/verify_gate.py). The reported pair must win on the
 // recomputed numbers and no pairing may be a regression -- re-measuring a noisy leg is fine, hiding it is not.
 "overlay_path": "...", "notes": "..."}
```

## PHASE=finalize

Assemble the deliverable: the combined overlay, a `final_patch.diff`, and a launch script that
reproduces the accepted stack from a clean checkout. With `BASE_OVERLAY_CMD` set, the deliverable is
the base stack PLUS this run's accepted overlays: the launch script runs the same command first and
appends to its `GEAK_OVERLAY`. Its `--ref` mode is the base stack, not the bare workload. Then **re-check engagement on the assembled
bundle** — a bundle that silently drops one accepted overlay must be reported as broken, not shipped
as complete.

Return JSON (PHASE=finalize):
```json
{"overlay": "<bundle overlay dir>", "patch": "<final_patch.diff>", "launch": "<launch.sh>",
 "engagement_recheck": {"forward": true, "backward": true, "ranks": 8, "evidence": "..."}}
```
`engagement_recheck.forward` / `.backward` are booleans, measured on a leg launched from the ASSEMBLED
bundle. Anything other than `true` marks the run `bundle_not_engaged` (the orchestrator fails closed).
When this run accepted NO kernel overlay there is nothing to re-check: return `null` for both.
