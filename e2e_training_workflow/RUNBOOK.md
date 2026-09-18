# Runbook — the commands actually used

Verbatim commands from bringing this up on the reference box, in the order they were run. Paths are
that box's; substitute your own.

**No named containers.** Everything ran in ephemeral `docker run --rm` containers — nothing was left
running, and none of the pre-existing containers on the node (`fada_*`, `openrlhf_author`,
`ms-swift-*`, …) were touched. They belong to other people.

```
HOST PATH                                         CONTAINER MOUNT
/home/treemann/models                          -> /models  (ro)    checkpoints + tokenizers
/home/treemann/geak_train                      -> /work            data, runs, tasks, overlays
<repo>/e2e_training_workflow                   -> /repo (or /geak) this workflow
image: rocm/primus:v26.4
```

## 0. Environment
```bash
rocm-smi --showproductname | head -3         # MI308X / gfx942
rocm-smi --showpids                          # "No KFD PIDs currently running" = quiet box
docker images | grep rocm/primus
```

## 1. Freeze the dataset (once; needs network)
```bash
mkdir -p /home/treemann/geak_train/{data,runs,scripts,tasks,overlays}
docker run --rm --network host -v /home/treemann/geak_train:/work -e HF_HOME=/work/.hf \
  --entrypoint bash rocm/primus:v26.4 \
  -c 'python /work/scripts/freeze_sft_data.py 20000 /work/data/sft_ultrachat20k.jsonl'
```
20 000 UltraChat conversations, ~115 MB, native `messages` field — no conversion needed.

## 2. Read the model's real hyperparameters out of the checkpoint (never guess them)
```bash
docker run --rm -v /home/treemann/models:/models:ro -v /home/treemann/geak_train:/work \
  --entrypoint bash rocm/primus:v26.4 \
  -c 'PYTHONPATH=/workspace/Primus/third_party/Megatron-LM \
      python /work/scripts/dump_ckpt_args.py /models/qwen2_5_math_7b_torch_dist/release/common.pt'
```
`PYTHONPATH` is required — the pickle references `megatron.*` classes. Copy `num_layers /
hidden_size / ffn_hidden_size / num_attention_heads / num_query_groups / kv_channels / rotary_base /
untie_embeddings_and_output_weights` into `workloads/<name>.env`.

## 3. One measurement leg
Works from the host or inside the container — the script bridges it.
```bash
bash <repo>/e2e_training_workflow/scripts/bench_train.sh \
  --workload <repo>/e2e_training_workflow/workloads/qwen2_5_math_7b.env \
  --nproc 8 --iters 30
```
Check `status: ok` and `occupancy_verdict: clean`. Anything else is not a measurement.

## 4. Baseline pair + bands (what Setup does)
```bash
for rep in A B; do
  bash .../scripts/bench_train.sh --workload .../workloads/qwen2_5_math_7b.env \
    --nproc 8 --iters 30 --tag calib$rep
done
python3 .../scripts/trajectory_gate.py calibrate <runA>/probe <runB>/probe -o band_7b.json
```

## 5. Profile (never a gate leg — `status` is forced to `unusable`)
```bash
bash .../scripts/bench_train.sh --workload .../workloads/qwen2_5_math_7b.env \
  --nproc 8 --iters 20 --tag prof --profile-steps 14:15 --profile-ranks 0,4
python3 .../scripts/parse_train_profile.py <run>/probe <run>/bench_summary.json
```
Put the window AFTER the measured warmup (`warmup_steps_max`, ~10 at 7B). Two steps is enough — the
trace is ~195 MB per rank at 7B.

## 6. Capture a head op -> task dir -> run its oracle
```bash
# capture (also not a measurement: status forced unusable, ~15% slower from writing the record)
bash .../scripts/bench_train.sh --workload .../workloads/qwen2_5_math_7b.env \
  --nproc 8 --iters 14 --tag head \
  --extra-env "GEAK_CAPTURE_TARGET=transformer_engine.pytorch.module.linear:_Linear \
               GEAK_CAPTURE_STEP=12 GEAK_CAPTURE_DIR=/work/tasks/te_linear_7b"

# build the task dir the kernel layer consumes, then run its immutable oracle
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
  -v /home/treemann/geak_train:/work -v <repo>:/repo -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint bash rocm/primus:v26.4 -c '
    python /repo/e2e_training_workflow/scripts/build_task_dir.py \
      /work/tasks/te_linear_7b/reference_io_rank0.pt /work/tasks/head7b --reference linear
    cp /repo/e2e_workflow/scripts/harness_lib.py /work/tasks/head7b/
    cd /work/tasks/head7b && python unittest.py'
```
`--reference` selects BOTH the fp32 reference and the argument layout: `linear` (weight first) for
TE's `_Linear`, `rmsnorm_linear` (input first) for `_LayerNormLinear`. Getting it wrong unpacks the
arguments silently wrong.

## 7. Overlay + engagement proof
```bash
cat > /home/treemann/geak_train/overlays/identity_manifest.json <<'J'
{"target": "transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear",
 "engagement_out": "/work/runs/identity_engagement.json"}
J
bash .../scripts/bench_train.sh --workload .../workloads/qwen2_5_0_5b.env --nproc 2 --iters 10 \
  --tag overlay --extra-env "GEAK_OVERLAY=/work/overlays/identity_manifest.json"
python3 .../scripts/overlay_train.py /home/treemann/geak_train/runs/identity_engagement.json
```
Omitting `replacement` installs an identity rebind — counters only. That separates "did the rebind
engage?" from "is the new kernel correct?", so a failure has one meaning instead of two.

## 8. Calibrate a kernel-swap band (2x2 — and check the pair is usable)
```bash
for rep in A B; do bash .../bench_train.sh ... --tag base$rep; done
for rep in A B; do bash .../bench_train.sh ... --tag swap$rep \
   --extra-env "GEAK_OVERLAY=/work/overlays/ref_lnlinear_manifest.json"; done
python3 .../scripts/trajectory_gate.py calibrate_swap <baseA>/probe <baseB>/probe \
   -- <swapA>/probe <swapB>/probe -o band_kernel_swap.json
```
Read `pair_usable` FIRST. `NVTE_USE_HIPBLASLT` on/off and `gradient_accumulation_fusion` on/off both
come back **unusable** — they do not change the kernels at all, and a band built from either would
have read as "kernel swaps are numerically free".

## 9. Judge a candidate
```bash
python3 .../scripts/trajectory_gate.py compare <ref>/probe <cand>/probe \
  --band band_kernel_swap.json --kind kernel_swap      # or --kind overlay
```

## 10. Tests (no GPU)
```bash
NODE=~/.cursor-server/bin/linux-x64/*/node            # this box has no system node
$NODE .../scripts/tests/test_training_gate.js
$NODE .../scripts/tests/test_pipeline_integration.js <runA> <runB>
python3 .../scripts/tests/test_parse_train_profile.py
```

## Two rules learned the hard way
* **Never edit a script while it is executing.** bash reads lazily by byte offset; an in-place edit
  shifts the remaining bytes and the running process starts parsing garbage. It destroyed the summary
  stage of a completed 8-GPU run (the probe sidecars survived, so it was recoverable).
* **Verify that an edit actually applied.** A string-replace that silently does not match leaves every
  later edit keyed on its intended output no-op'ing too, while the tooling reports success — it only
  proves the script ran, not that it changed anything. Four GPU runs were lost to this.
