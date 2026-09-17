"""L1 — the trajectory gate: does the candidate still train the same way?

WHY L1 EXISTS. Inference can gate on byte-identical output. Training has no output, and its failure
mode is slower and nastier: a wrong backward, a systematically biased kernel, or an overlay that
engaged on only SOME ranks looks completely healthy for tens of steps and destroys convergence
thousands of steps later. L0 (op-level fwd+bwd numerics) cannot see the last one at all -- every op
is individually within tolerance because every op is individually correct.

THE BAND IS MEASURED, NEVER ASSUMED. Training is natively nondeterministic (reduction order, backward
atomics, RCCL algorithm selection), so `calibrate` runs the statistics between TWO INDEPENDENT
BASELINE runs at identical config and takes that as the noise floor. A threshold picked by intuition
is worthless here: observed grad norms on this workload swing 60 -> 1023 step to step, so almost any
fixed threshold is either blind or permanently red.

TRAINING IS CHAOTIC, SO THE ENVELOPE IS STEP-INDEXED. Measured on this box, two runs at identical
config and identical data (the useful-token sequences match step for step) still diverge:

    step  1   loss identical to the last bit   grad_norm already differs by 0.87%
    step  5   loss  0.76%                      grad_norm 22.7%
    step 12   loss  4.40%                      grad_norm 40.2%
    step 21   loss delta 2.04 ABSOLUTE (on a loss of ~3)

Two consequences that shape this file:
  * A SCALAR aggregate over a long window is the wrong instrument. Calibrating `max |dloss|` over 25
    steps yields a limit of 6.1 on a loss of 3 -- a gate that passes anything. The band must be a
    FUNCTION OF STEP INDEX, compared against how far the baseline pair had drifted BY THAT STEP.
  * Step 1 is special and precious: it is a pure forward on identical weights, so its loss is
    bit-reproducible. That gives one zero-noise probe of the forward path. grad_norm at step 1 is
    NOT bit-reproducible (backward atomics / reduction order), so its floor is calibrated too.

WHAT IS COMPARED:
  * step-1 loss        -- zero-noise forward probe; the sharpest single check available.
  * per-step envelope  -- |cand - ref| at step i against the baseline pair's own drift at step i.
  * drift + sign balance over the EARLY window only, where natural divergence has not yet swamped
    the signal. Unbiased noise sits near 0.5; a one-sided kernel sits near 0 or 1 while every
    magnitude check still passes -- same doctrine as the L0 sign-distribution rule.

Both legs must be run with the SAME seed and the SAME data order; otherwise step i is not the same
batch on both sides and none of this means anything. bench_train.sh pins both.
"""
from __future__ import annotations

import json
import os
import statistics
import sys


def _series(probe_dir, rank=0):
    """loss/grad_norm are global (already reduced across DP), so one rank carries the trajectory."""
    path = os.path.join(probe_dir, f"probe_rank{rank}.jsonl")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(probe_dir) if f.startswith("probe_rank")]
        if not cands:
            raise FileNotFoundError(f"no probe sidecars in {probe_dir}")
        path = os.path.join(probe_dir, sorted(cands)[0])
    loss, gn, cfg = [], [], None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("record") == "header":
                cfg = r.get("config")
            elif r.get("record") == "step":
                loss.append(r.get("loss"))
                gn.append(r.get("grad_norm"))
    return {"loss": loss, "grad_norm": gn, "config": cfg}


def _slope(ys):
    """Least-squares slope of ys against step index — the systematic-drift detector."""
    n = len(ys)
    if n < 3:
        return 0.0
    mx = (n - 1) / 2.0
    my = statistics.fmean(ys)
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den if den else 0.0


def compare_series(ref, cand):
    """Raw statistics between two trajectories. No thresholds here — see `verdict`."""
    out = {}
    n = min(len(ref["loss"]), len(cand["loss"]))
    out["steps_compared"] = n
    out["steps_ref"], out["steps_cand"] = len(ref["loss"]), len(cand["loss"])
    if n < 3:
        out["error"] = "fewer than 3 comparable steps"
        return out

    for key in ("loss", "grad_norm"):
        a = [x for x in ref[key][:n]]
        b = [x for x in cand[key][:n]]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) < 3:
            out[key] = {"error": "not enough non-null samples"}
            continue
        d = [y - x for x, y in pairs]
        rel = [abs(y - x) / abs(x) for x, y in pairs if x]
        out[key] = {
            "max_abs_delta": round(max(abs(v) for v in d), 6),
            "mean_abs_delta": round(statistics.fmean(abs(v) for v in d), 6),
            "median_rel_delta_pct": round(100 * statistics.median(rel), 4) if rel else None,
            "max_rel_delta_pct": round(100 * max(rel), 4) if rel else None,
            # Drift is expressed per 100 steps so short calibration runs stay readable.
            "drift_per_100_steps": round(_slope(d) * 100, 6),
            "sign_balance": round(sum(1 for v in d if v > 0) / len(d), 4),
        }
    return out


# The early window is where natural divergence is still small enough for a bug to stand out.
EARLY_STEPS = 8
# Floor for the step-1 forward probe. A kernel change legitimately perturbs rounding, so this is not
# zero; it is small because at step 1 there is no accumulated drift to hide behind.
STEP1_REL_FLOOR_PCT = 0.05


def _early(stats_series, n=EARLY_STEPS):
    return stats_series[:n]


def calibrate(dirs, safety=3.0):
    """Noise floor from >=2 INDEPENDENT baseline runs at identical config.

    Produces a STEP-INDEXED envelope, not a scalar: chaotic divergence means the tolerable delta at
    step 20 is orders of magnitude larger than at step 2, and collapsing that into one number yields
    a gate that passes everything (measured: a 6.1 limit on a loss of 3).
    """
    series = [_series(d) for d in dirs]
    if len(series) < 2:
        raise SystemExit("calibration needs at least two baseline runs")
    n = min(len(s["loss"]) for s in series)
    env = {"loss": [], "grad_norm": []}
    for key in ("loss", "grad_norm"):
        for i in range(n):
            vals = [s[key][i] for s in series if s[key][i] is not None]
            if len(vals) < 2:
                env[key].append(None)
                continue
            # Widest disagreement seen among baseline runs AT THIS STEP.
            env[key].append(round(max(vals) - min(vals), 8))
    step1_rel = None
    l1 = [s["loss"][0] for s in series if s["loss"] and s["loss"][0] is not None]
    if len(l1) >= 2 and l1[0]:
        step1_rel = round(100 * (max(l1) - min(l1)) / abs(l1[0]), 6)
    g1 = [s["grad_norm"][0] for s in series if s["grad_norm"] and s["grad_norm"][0] is not None]
    gn1_rel = round(100 * (max(g1) - min(g1)) / abs(g1[0]), 6) if len(g1) >= 2 and g1[0] else None
    return {
        "safety": safety, "runs": list(dirs), "steps": n,
        "envelope": env,
        "step1_loss_rel_pct": step1_rel,
        "step1_grad_norm_rel_pct": gn1_rel,
        "early_steps": EARLY_STEPS,
        "note": ("envelope[i] is the widest baseline-vs-baseline disagreement AT STEP i; the gate "
                 "allows safety x that. step1 loss is bit-reproducible in a healthy stack, so its "
                 "floor is deliberately tight."),
    }


def verdict(ref_dir, cand_dir, band):
    """The L1 verdict, in the shape training_workflow.js's gateVerdict() consumes: {pass, reason}."""
    rs, cs = _series(ref_dir), _series(cand_dir)
    res = {"pass": None, "reason": "", "checks": {}}

    if rs.get("config") != cs.get("config"):
        return {"pass": False, "reason": "legs ran different configs; step-aligned trajectories are "
                                         "not comparable", "checks": {}}
    n = min(len(rs["loss"]), len(cs["loss"]))
    if n < 3:
        return {"pass": False, "reason": "fewer than 3 comparable steps", "checks": {}}

    safety = float(band.get("safety", 3.0))
    env = band.get("envelope", {})
    failures = []

    # ---- check 1: the step-1 forward probe (zero natural noise) -------------------------------
    if rs["loss"] and cs["loss"] and rs["loss"][0]:
        rel = 100 * abs(cs["loss"][0] - rs["loss"][0]) / abs(rs["loss"][0])
        lim = max(STEP1_REL_FLOOR_PCT, (band.get("step1_loss_rel_pct") or 0) * safety)
        res["checks"]["step1_loss_rel_pct"] = {"value": round(rel, 6), "limit": round(lim, 6)}
        if rel > lim:
            failures.append(f"step-1 loss differs by {rel:.4f}% (limit {lim:.4f}%): the forward path "
                            f"changed before any optimizer update could explain it")

    # ---- check 2: per-step envelope ------------------------------------------------------------
    for key in ("loss", "grad_norm"):
        e = env.get(key) or []
        exceed, worst = [], 0.0
        for i in range(min(n, len(e))):
            if e[i] is None or rs[key][i] is None or cs[key][i] is None:
                continue
            d = abs(cs[key][i] - rs[key][i])
            lim = e[i] * safety
            # A zero-width envelope step (perfect baseline agreement) still needs a floor, or any
            # rounding difference trips it.
            lim = max(lim, abs(rs[key][i]) * (STEP1_REL_FLOOR_PCT / 100.0))
            if d > lim:
                exceed.append(i + 1)
                worst = max(worst, d / lim if lim else float("inf"))
        res["checks"][f"{key}_envelope"] = {
            "steps_checked": min(n, len(e)), "exceeded_steps": exceed,
            "worst_ratio_to_limit": round(worst, 3)}
        # One isolated exceedance is chaos; a pattern is a bug.
        if len(exceed) > max(1, int(0.1 * min(n, len(e)))):
            failures.append(f"{key} left the calibrated envelope on {len(exceed)} steps "
                            f"({exceed[:6]}{'...' if len(exceed) > 6 else ''})")

    # ---- check 3: one-sidedness over the early window ------------------------------------------
    for key in ("loss", "grad_norm"):
        d = [cs[key][i] - rs[key][i] for i in range(min(n, EARLY_STEPS))
             if rs[key][i] is not None and cs[key][i] is not None]
        if len(d) >= 5:
            sb = sum(1 for v in d if v > 0) / len(d)
            res["checks"][f"{key}_sign_balance_early"] = {"value": round(sb, 4), "n": len(d)}
            if sb in (0.0, 1.0):
                failures.append(f"{key} moved the SAME direction on all {len(d)} early steps: a "
                                f"systematic bias, not noise (every magnitude check can still pass)")

    res["pass"] = not failures
    res["reason"] = "; ".join(failures) if failures else "within the calibrated per-step envelope"
    return res


def _usage():
    print(__doc__.strip().splitlines()[0])
    print("usage:\n  trajectory_gate.py calibrate <baseline_probe> <baseline_probe> [more...] [-o band.json]"
          "\n  trajectory_gate.py compare <ref_probe> <cand_probe> --band <band.json>")
    return 2


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(_usage())
    cmd = sys.argv[1]
    if cmd == "calibrate":
        dirs = [a for a in sys.argv[2:] if not a.startswith("-")]
        if "-o" in sys.argv:
            dirs = [d for d in dirs if d != sys.argv[sys.argv.index("-o") + 1]]
        b = calibrate(dirs)
        out = None
        if "-o" in sys.argv:
            out = sys.argv[sys.argv.index("-o") + 1]
            json.dump(b, open(out, "w"), indent=2)
        print(json.dumps(b, indent=2))
    elif cmd == "compare":
        bandpath = sys.argv[sys.argv.index("--band") + 1] if "--band" in sys.argv else None
        band = json.load(open(bandpath)) if bandpath else {"limits": {}}
        print(json.dumps(verdict(sys.argv[2], sys.argv[3], band), indent=2))
    else:
        sys.exit(_usage())
