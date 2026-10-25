#!/usr/bin/env python3
"""The packing gate: the structural checks that license a packed dataset, run end to end.

WHY PACKING GETS A GATE AND OTHER CLASS-B LEVERS DO NOT. GBS, MBS and sequence length change the
optimizer's trajectory -- that is what they are FOR -- so no equivalence check can call them correct.
Packing is different: it is a layout change whose risks are structural (attention isolation between
conversations, position restarts, which tokens are targets) and every one of them is checkable at
step 1 on identical weights. Measured on 2026-10-09 with each check's control (knowledge/
sft_measurement.md, "PACKING"); this script makes those checks a verdict rather than a notebook.

Checks, each with the threshold fixed before this script existed:
  1. TOKEN IDENTITY  -- every conversation in the first N packs renders to exactly the tokens its
     ORIGINAL (looked up in --source) renders to unpacked. Required: 0 mismatches. The first version
     compared against "the default system prompt" instead of the original, skipped every conversation
     whose prompt was not the default, and passed a pack built with the wrong prompt -- caught by the
     positive control before this script was ever relied on.
  2. FORWARD EQUIVALENCE -- pack k through the real stack at lr 0 must match the isolated
     expectation (each conversation alone) within 0.5%, AND the non-isolated control (one sequence)
     must sit >= 1% from the isolated value (else the control cannot discriminate -> INCONCLUSIVE),
     AND the measured value must be >= 3x closer to isolated than to non-isolated.
  3. OBJECTIVE EQUIVALENCE -- with boundary targets masked, the packed per-token loss on pack k's
     conversations must equal the unpacked expectation (isolated WITHOUT boundary targets) within
     0.5%. This is what catches an unmasked boundary: unmasked measured -0.424% off this expectation's
     neighbour, i.e. the check sits on the right side of the defect it exists for.

Kernel-level backward equivalence (scripts/tests/check_varlen_attention_l0.py) is a property of the
attention kernel, not of a dataset; it is reported here if its result file exists, not re-run.

Usage (host):
  pack_gate.py --workload workloads/qwen2_5_math_7b.env --packed /work/data/<packed>.jsonl
               --model /models/qwen2_5_math_7b [--pack-index 0] [--verify 200] [--nproc 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(HERE)
REPO = os.path.dirname(WF)
WORK = os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train")
MODELS = os.environ.get("GEAK_MODELS_DIR", "/home/treemann/models")
IMAGE = os.environ.get("GEAK_IMAGE", "rocm/primus:v26.4")

FWD_TOL_PCT = 0.5
CONTROL_MIN_PCT = 1.0
SEPARATION = 3.0
OBJ_TOL_PCT = 0.5

# Runs INSIDE the image. Split exactly as Megatron's SFTDataset does: a new conversation at every
# system message.
_INNER = r'''
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir, packed, k, verify, source = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tok = AutoTokenizer.from_pretrained(model_dir)
def split(msgs):
    out, cur = [], []
    for m in msgs:
        if m["role"] == "system" and cur: out.append(cur); cur = []
        cur.append(m)
    if cur: out.append(cur)
    return out
lines = [json.loads(l)["messages"] for _, l in zip(range(max(verify, k + 1)), open(packed))]
default = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
default = default.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]
# Compare every packed conversation with its ORIGINAL, unpacked, as the source file has it. The first
# version compared only conversations whose system prompt already equalled the default -- so a pack
# built with the WRONG system prompt skipped every conversation and passed, 196 checked, 0 flagged.
# The positive control caught that; this is the check the contract actually states.
key = lambda msgs: json.dumps([m for m in msgs if m["role"] != "system"], sort_keys=True)
orig = {}
for l in open(source):
    m = json.loads(l)["messages"]; orig.setdefault(key(m), m)
checked = mism = missing = 0
for msgs in lines[:verify]:
    for c in split(msgs):
        checked += 1
        o = orig.get(key(c))
        if o is None:
            missing += 1; mism += 1
        elif tok.apply_chat_template(c, tokenize=True) != tok.apply_chat_template(o, tokenize=True):
            mism += 1
ids = [tok.apply_chat_template(c, tokenize=True) for c in split(lines[k])]
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ce = torch.nn.functional.cross_entropy
with torch.no_grad():
    within = bnd = 0.0; n_within = 0
    for i, x in enumerate(ids):
        t = torch.tensor([x]).cuda(); lg = model(t).logits[0].float()
        within += ce(lg[:-1], t[0, 1:], reduction="sum").item(); n_within += len(x) - 1
        if i + 1 < len(ids):
            bnd += ce(lg[-1:], torch.tensor([ids[i + 1][0]]).cuda(), reduction="sum").item()
    nb = len(ids) - 1
    flat = torch.tensor([sum(ids, [])]).cuda(); lg = model(flat).logits[0].float()
    non = ce(lg[:-1], flat[0, 1:]).item()
print("PACKGATE " + json.dumps({
    "token_identity": {"conversations_checked": checked, "mismatches": mism, "not_found_in_source": missing,
                       "default_system_prompt": default},
    "conversations_in_pack": len(ids), "lens": [len(x) for x in ids],
    "isolated_with_boundary": (within + bnd) / (n_within + nb),
    "isolated_without_boundary": within / n_within,
    "non_isolated": non, "loss_tokens_with_boundary": n_within + nb, "loss_tokens_without": n_within}))
'''


def _docker(args, gpus=""):
    cmd = ["docker", "run", "--rm", "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
           "--ipc=host", "-v", f"{WORK}:/work", "-v", f"{MODELS}:/models:ro",
           "-e", f"HIP_VISIBLE_DEVICES={gpus}", "--entrypoint", "python", IMAGE] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def _bench(workload, data_ctr, tag, nproc, extra):
    cmd = ["bash", os.path.join(HERE, "bench_train.sh"), "--stack", "megatron", "--workload", workload,
           "--nproc", str(nproc), "--iters", "2", "--tag", tag, "--data", data_ctr,
           "--extra-train-args", extra]
    subprocess.run(cmd, capture_output=True, text=True)
    runs = sorted(glob.glob(os.path.join(WORK, "runs", f"bench_*_{tag}_*")))
    if not runs:
        return None
    steps = [json.loads(l) for l in open(os.path.join(runs[-1], "probe", "probe_rank0.jsonl"))]
    steps = [s for s in steps if s.get("record") == "step"]
    return {"dir": runs[-1], "loss": steps[0].get("loss"),
            "loss_tokens_per_microbatch": steps[0]["useful_tokens"] / max(1, steps[0]["micro_batches"])} if steps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--packed", required=True, help="container path, e.g. /work/data/x_packed.jsonl")
    ap.add_argument("--source", required=True, help="container path of the ORIGINAL unpacked jsonl")
    ap.add_argument("--model", required=True, help="container path of the HF model directory")
    ap.add_argument("--pack-index", type=int, default=0)
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--nproc", type=int, default=8)
    # The verdict as a FILE (audit 2026-10-25 ORCH-A4): the gate verifier reads it from disk instead of
    # trusting an agent's transcription of stdout.
    ap.add_argument("--out", default=None, help="host path to write the verdict JSON to")
    a = ap.parse_args()

    os.makedirs(os.path.join(WORK, "runs", "pack_gate"), exist_ok=True)
    inner = os.path.join(WORK, "runs", "pack_gate", "inner.py")
    open(inner, "w").write(_INNER)
    r = _docker(["/work/runs/pack_gate/inner.py", a.model, a.packed, str(a.pack_index), str(a.verify),
                 a.source], gpus="0")
    line = next((l for l in r.stdout.splitlines() if l.startswith("PACKGATE ")), None)
    if not line:
        sys.exit("pack_gate: reference computation failed:\n" + (r.stderr or r.stdout)[-2000:])
    ref = json.loads(line[len("PACKGATE "):])

    # The single pack, repeated, so every micro-batch of the lr-0 leg is exactly this pack.
    host_packed = a.packed.replace("/work", WORK, 1)
    pack_line = open(host_packed).readlines()[a.pack_index]
    rep = f"/work/runs/pack_gate/pack{a.pack_index}_x64.jsonl"
    open(rep.replace("/work", WORK, 1), "w").write(pack_line * 64)
    base = "--lr 0.0 --attention-dropout 0.0 --hidden-dropout 0.0 --calculate-per-token-loss"
    meg = _bench(a.workload, rep, f"packgate_p{a.pack_index}", a.nproc, base)
    if not meg or meg["loss"] is None:
        sys.exit("pack_gate: the Megatron lr-0 leg produced no loss")

    rel = lambda x, y: 100.0 * (x - y) / y                          # noqa: E731
    iso_w, iso_wo, non = ref["isolated_with_boundary"], ref["isolated_without_boundary"], ref["non_isolated"]
    masked = abs(meg["loss_tokens_per_microbatch"] - ref["loss_tokens_without"]) < 0.5
    # The leg runs WITH the boundary mask (rocm_compat Fix 3), so its expectation is the isolated loss
    # WITHOUT boundary targets; the non-isolated control is a forward property and keeps its value.
    expect = iso_wo if masked else iso_w
    d_iso, d_non = abs(rel(meg["loss"], expect)), abs(rel(meg["loss"], non))
    control = abs(rel(non, expect))
    checks = {
        "token_identity": {"pass": ref["token_identity"]["mismatches"] == 0, **ref["token_identity"]},
        "boundary_targets_masked": {"pass": masked, "loss_tokens_per_pack": meg["loss_tokens_per_microbatch"],
                                    "expected_without_boundaries": ref["loss_tokens_without"]},
        "control_discriminates": {"pass": control >= CONTROL_MIN_PCT, "control_rel_pct": round(control, 4),
                                  "min_pct": CONTROL_MIN_PCT},
        "forward_and_objective_equivalence": {"pass": d_iso <= min(FWD_TOL_PCT, OBJ_TOL_PCT) and d_non >= SEPARATION * d_iso,
                                              "rel_to_isolated_pct": round(d_iso, 4),
                                              "rel_to_non_isolated_pct": round(d_non, 4),
                                              "tolerance_pct": min(FWD_TOL_PCT, OBJ_TOL_PCT)},
    }
    k = os.path.join(WORK, "runs", "pack_gate", "varlen_l0.json")
    if os.path.exists(k):
        checks["kernel_backward_reported"] = json.load(open(k))
    if not checks["control_discriminates"]["pass"]:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if all(c["pass"] for n, c in checks.items() if isinstance(c, dict) and "pass" in c) else "FAIL"
    res = {"verdict": verdict, "pass": verdict == "PASS", "pack_index": a.pack_index, "packed": a.packed,
           "source": a.source, "conversations_in_pack": ref["conversations_in_pack"], "megatron_loss": meg["loss"],
           "expected_loss": expect, "non_isolated_loss": non, "checks": checks, "leg": meg["dir"]}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
