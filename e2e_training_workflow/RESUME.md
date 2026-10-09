> **RESUMED AND COMPLETED 2026-10-09.** The interrupted run's validate phase finished from cache; the
> alpha = 1.0 300-step leg ran and is recorded in `knowledge/gate_convergence.md`. Nothing below is
> pending any more — this file is kept as the record of the pause.

# Paused 2026-10-09 — GPUs released for another task

Stopped at the user's request. **The GPUs are free**: `rocm-smi --showpids` reports no KFD PIDs, and
no container of this project is running. Nothing is queued and nothing will restart on its own.

## What was running when it stopped

A `deep` ConfigSweep run measuring the two levers that had never been measured. It got through
Setup, Profile, Strategize, **both ConfigSweep legs**, Finalize and Report, and was interrupted in
the last phase, `training_director:validate`.

**The measurements survived.** They are in the run journal, not only in memory:

```
/home/treemann/.claude/projects/-home-treemann-treemann-github-AMD-AGI-GEAK-claude-e2e-training-workflow/
  cb433b82-735d-4533-ac61-d16e8af2836c/subagents/workflows/wf_ad87f338-2c1/journal.jsonl
```

Run ID `wf_ad87f338-2c1`. Both lever records are complete; only the final validation leg is missing.

## Results, already obtained (see the summary in chat for the full reading)

Reference leg `useful_tok_s_per_gpu` 837.5, both candidates measured against it, same band.

| lever | flags | aligned delta | memory | verdict |
|---|---|---|---|---|
| `recompute_granularity` | `--recompute-granularity selective` | **−3.403%** | **COSTS 838 MB** (80578 → 81416) | regression |
| `attention_backend` | `--attention-backend flash`, `NVTE_FLASH_ATTN=1` | **−0.7284%** (band 1.978%) | — | `no_win` / inconclusive_noise |

Both carry a window-aligned delta; window placement alone accounted for 0.51% and 0.52%, so the
unaligned numbers would have been mostly artifact in both cases.

**The v4 identity fix is confirmed working.** On both levers `config_digest` stayed identical
(`d1548a7a764bdada`, same workload) while `impl_digest` MOVED (`49d9009f…` → `545a0456…` and
`c1b6712f…`). That is exactly the behaviour the comm-overlap run showed was missing, where neither
digest moved. The Tuner also diffed the captured snapshots directly and found exactly one differing
key for recompute and three for attention-backend, nothing else.

## To resume

Nothing needs repairing first. Three options, cheapest first.

1. **Just finish the validation leg** of the interrupted run — completed agents replay from cache,
   only `validate` runs live:

   ```
   Workflow({scriptPath: ".../training_workflow.js", resumeFromRunId: "wf_ad87f338-2c1"})
   ```

   Caveat: the orchestrator reads the clock through `args.t0_epoch_ms`, which was
   `1791513468747` for that run. A resume re-uses it, so the time budget will look mostly spent.
   Pass a fresh `t0_epoch_ms` if the budget matters.

2. **Accept the two measurements as they are** and move on. The lever verdicts do not depend on the
   validate phase, which checks the final bundle rather than the sweep; and with zero accepted
   changes the bundle is empty anyway.

3. **Continue the roadmap.** Month 1 (P0–P5) is complete. Next is Month 2, starting with the second
   training-stack adapter — see `ROADMAP.md`.

## Loose ends, unchanged by the pause

* Five scripts have been modified in the working tree since **09-24** and are still uncommitted:
  `bench_train.sh`, `build_task_dir.py`, `capture_train_shapes.py`, `l0_gate.py`, `rocm_compat.py`.
  They are the agent-authored changes set aside for separate review. Nothing in this session touched
  them. They are coherent improvements (an MLP capture target added to the measurement-validity
  guard; an explicit engagement contract for composite ops), but they are not mine to commit.
* `runs/l2soak_v3_ref4` is a partial leg whose files are root-owned and cannot be deleted from the
  host. Harmless: every analyzer refuses a short window and names it.
* `scripts/trajectory_tokens.py` is a characterised instrument, not a gate — it still owes a
  positive control it can actually fail.
