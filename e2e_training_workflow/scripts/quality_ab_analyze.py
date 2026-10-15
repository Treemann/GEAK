#!/usr/bin/env python3
"""Apply the P0-4 pre-registered verdicts (knowledge/gate_convergence.md) to quality_ab.sh legs.

Usage: quality_ab_analyze.py <runs/qab_STAMP prefix>
Reads <prefix>_<arm>_seed<k>/train.log (final "validation loss ... lm loss value") and the probe.
"""
import glob
import json
import math
import os
import re
import statistics
import sys

T_CRIT = 4.30          # two-sided 95%, df = 2 (three paired seeds)
PASS_FRACTION = 0.25   # |m_S8| <= 0.25 x |m_CTRL|
VAL_RE = re.compile(r"validation loss at .*? lm loss value: ([0-9.Ee+-]+)")


def leg(prefix, arm, seed):
    d = f"{prefix}_{arm}_seed{seed}"
    out = {"dir": d}
    try:
        vals = VAL_RE.findall(open(os.path.join(d, "train.log"), errors="replace").read())
        out["val_loss"] = float(vals[-1]) if vals else None
    except FileNotFoundError:
        out["val_loss"] = None
    try:
        b = json.load(open(os.path.join(d, "bench_summary.json")))
        out.update(status=b.get("status"), useful=b.get("useful_tok_s_per_gpu"))
    except FileNotFoundError:
        out["status"] = None
    steps = []
    p = os.path.join(d, "probe", "probe_rank0.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            if r.get("record") == "step":
                steps.append(r)
    out["steps"] = len(steps)
    if steps:
        out["train_loss_last50"] = statistics.fmean(s["loss"] for s in steps[-50:])
        out["grad_norm_median"] = statistics.median(s["grad_norm"] for s in steps)
    # Engagement IN TRAINING: per_step buckets 1..steps must show fwd > 0 and fallback 0 for every
    # overlay on every rank. The bucket after the last training step is the end-of-training
    # evaluation, where rope/swiglu/attn/lmhead fall back to the reference forward by design (no_grad
    # path, smoke runs/evalsmoke_1015_s8); those fallbacks are reported, not counted against the leg.
    proofs = glob.glob(os.path.join(d, "*", "engagement_fused*.json.rank*"))
    bad, eval_fb = [], 0
    for f in proofs:
        ps = json.load(open(f)).get("per_step") or {}
        for k, v in ps.items():
            if not isinstance(v, dict):
                continue
            fbk = sum(x for kk, x in v.items() if kk.startswith("fallback") and isinstance(x, int))
            if int(k) <= len(steps):
                fw = sum(x for kk, x in v.items() if kk.startswith("fwd") and isinstance(x, int))
                if fbk or not fw:
                    bad.append(f"{os.path.relpath(f, d)} step {k}")
            else:
                eval_fb += fbk
    out["overlay_proofs"] = len(proofs)
    out["train_engagement_bad"] = bad[:5] + ([f"... {len(bad) - 5} more"] if len(bad) > 5 else [])
    out["eval_fallback_calls"] = eval_fb
    return out


def paired(xs, refs):
    d = [x - r for x, r in zip(xs, refs)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else float("nan")
    t = m / (sd / math.sqrt(len(d))) if sd and sd > 0 else float("inf") * (1 if m >= 0 else -1)
    same_sign = all(v > 0 for v in d) or all(v < 0 for v in d)
    return {"d": [round(v, 5) for v in d], "mean": round(m, 5), "sd": round(sd, 5), "t": round(t, 3),
            "same_sign": same_sign, "significant": same_sign and abs(t) >= T_CRIT}


def main():
    prefix = sys.argv[1].rstrip("/")
    seeds = sorted({int(m.group(1)) for f in glob.glob(prefix + "_*_seed*")
                    for m in [re.search(r"_seed(\d+)$", f)] if m})
    arms = {a: [leg(prefix, a, s) for s in seeds] for a in ("ref", "s8", "ctrl")}
    print(f"seeds {seeds}")
    for a, ls in arms.items():
        for s, l in zip(seeds, ls):
            print(f"  {a:4s} seed {s}: val {l['val_loss']}  status {l.get('status')}  steps {l['steps']}  "
                  f"train_last50 {round(l.get('train_loss_last50', float('nan')), 4)}  "
                  f"gn_med {round(l.get('grad_norm_median', float('nan')), 2)}  useful {l.get('useful')}")
    for l in arms["s8"]:
        print(f"  s8 {os.path.basename(l['dir'])}: {l['overlay_proofs']} proofs, training-path "
              f"violations {l['train_engagement_bad'] or 'none'}, eval-mode fallback calls {l['eval_fallback_calls']}")
        if l["overlay_proofs"] == 0 or l["train_engagement_bad"]:
            print("VERDICT: VOID -- an S8 leg did not run the stack in training (pre-registered rule)")
            return
    ok = all(l["val_loss"] is not None for ls in arms.values() for l in ls)
    if not ok:
        print("VERDICT: INCOMPLETE -- a leg has no validation loss")
        return
    ref = [l["val_loss"] for l in arms["ref"]]
    ctrl = paired([l["val_loss"] for l in arms["ctrl"]], ref)
    s8 = paired([l["val_loss"] for l in arms["s8"]], ref)
    print("CTRL - REF:", ctrl)
    print("S8   - REF:", s8)
    if not ctrl["significant"]:
        verdict = "INCONCLUSIVE (underpowered): the positive control was not detected"
    elif s8["mean"] > 0 and s8["significant"]:
        verdict = "S8 FAIL: significantly higher held-out loss"
    elif s8["mean"] < 0 and s8["significant"]:
        verdict = "S8 DIFFERS-BETTER: significantly lower held-out loss (investigate)"
    elif abs(s8["mean"]) <= PASS_FRACTION * abs(ctrl["mean"]):
        verdict = (f"S8 PASS: |{s8['mean']}| <= {PASS_FRACTION} x |{ctrl['mean']}| "
                   f"= {round(PASS_FRACTION * abs(ctrl['mean']), 5)}")
    else:
        verdict = f"INCONCLUSIVE at the observed effect: S8 mean {s8['mean']} vs control {ctrl['mean']}"
    print("VERDICT:", verdict)
    print(json.dumps({"seeds": seeds, "ref": ref, "ctrl": ctrl, "s8": s8, "verdict": verdict}))


if __name__ == "__main__":
    main()
