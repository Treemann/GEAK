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
    loss, gn, cfg, argv_hash = [], [], None, None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("record") == "header":
                cfg = r.get("config")
                argv_hash = r.get("argv_hash")
            elif r.get("record") == "step":
                loss.append(r.get("loss"))
                gn.append(r.get("grad_norm"))
    return {"loss": loss, "grad_norm": gn, "config": cfg, "argv_hash": argv_hash}


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
STEP1_GN_LIMIT_PCT = 10.0     # see check 1b: null max 1.40% over 696 pairs, mildest defect 61%

# THE BAND MUST MATCH WHAT THE CANDIDATE IS. Measured, and it is not a small effect:
#
#   same kernels, two runs      step-1 loss delta = 0.0      (bit-identical, 0.5B and 7B alike)
#   two attention backends      step-1 loss delta = 0.2917%  (TE fused/CK vs flash)
#
# A band calibrated from a SAME-KERNEL baseline pair therefore rejects ANY kernel swap out of hand --
# which would make L1 useless for the one job it exists to do. Two presets, and the caller must say
# which situation it is in:
#
#   "overlay"     the candidate should not change numerics at all (a rebind, a schedule change, a
#                 fusion that is exactly equivalent). Floor stays tight.
#   "kernel_swap" the candidate is a different implementation of the same mathematics, so rounding
#                 legitimately differs. Floor comes from a measured equivalent-intent pair.
#
# CAVEAT, not yet resolved: the 0.2917% figure is ONE pair, and it is not yet separated into "pure
# numerics" versus "packing isolation is not preserved by the flash path". The test that separates
# them is a packed-vs-unpacked run of each backend; until then treat it as an upper bound on
# legitimate kernel-swap noise, and read a step-1 failure just above it as "investigate", not "bug".
# MEASURED, from an L0-VERIFIED equivalent pair (TE's fused _LayerNormLinear vs a pure-PyTorch
# reimplementation of the same mathematics, swapped in through the overlay):
#     same kernels, two runs  -> step-1 loss delta 0.000000%   (bit-identical)
#     verified kernel swap    -> step-1 loss delta 1.748837%
# That is 35x the original 0.05% guess. Three cheaper candidates were tried first and all three were
# rejected BY MEASUREMENT: NVTE_USE_HIPBLASLT on/off and gradient_accumulation_fusion on/off are
# numerically identical (the knobs do not change the kernels at all), and --transformer-impl local
# cannot run packed sequences. The lesson is in calibrate_swap's `pair_usable` verdict: a pair whose
# `between` spread is not larger than its `within` spread calibrates nothing.
# NOTE the reference implementation computes internally in fp32, so 1.75% is an UPPER bound for a
# same-precision (bf16->bf16) swap. Tighten it when such a pair is available.
STEP1_FLOOR_PCT = {"overlay": 0.05, "kernel_swap": 2.0}
STEP1_REL_FLOOR_PCT = STEP1_FLOOR_PCT["overlay"]


def _early(stats_series, n=EARLY_STEPS):
    return stats_series[:n]


# How many INDEPENDENT baseline legs a band needs. Measured, not conventional: a 2-leg band makes
# envelope[i] a single abs(A-B) sample, which collapses to ~0 wherever those two legs happened to
# cross, and 3x~0 is still ~0. Both real runs then rejected their own null control --
#   0.5B: the band failed every baseline-vs-baseline pair, including its own calibration pair;
#   7B:   a same-kernel pair failed on 8 loss / 10 grad_norm steps, WORSE than the candidate's.
# A band built from fewer than MIN legs is not refused outright (it still carries the step-1 probe,
# which is the detector that actually discriminates) but it cannot cast a blocking envelope vote.
MIN_BAND_LEGS = 3
RECOMMENDED_BAND_LEGS = 5


def _rel_floor_pct(series, key, n):
    """A per-metric relative floor MEASURED from the calibration legs themselves.

    The envelope needs a floor or a step where the legs happened to agree exactly trips on any
    rounding difference. That floor used to be STEP1_REL_FLOOR_PCT (0.05%) for EVERY metric at every
    step -- a constant borrowed from step-1 loss, which is bit-reproducible in a healthy stack.
    grad_norm is not remotely that stable: three identical 0.5B runs gave 152.10 / 218.77 / 83.44 at
    step 2, so a 0.05% floor there is ~4 orders of magnitude too tight and every real run trips it.
    Take the metric's own relative disagreement at a LOW QUANTILE (p10). The quantile matters as much
    as the fix: the floor's only job is to cover the QUIETEST steps -- measured on three identical 7B
    legs, loss has a step where all three agree to 0.0000% -- while the per-step envelope
    (env[i] x safety) already scales everything else. Using the MEDIAN instead was tried and is a
    different way of being wrong: it lifts the floor to the TYPICAL disagreement (loss 8.93%,
    grad_norm 152.3%) and swamps the envelope, so the gate passed a +25% loss shift over the whole
    steady window. p10 (loss 0.31%, grad_norm 30.9%) covers the degenerate steps and leaves the
    envelope free to speak everywhere else.
    """
    rels = []
    for i in range(n):
        vals = [s[key][i] for s in series if s[key][i] is not None]
        if len(vals) < 2:
            continue
        mid = statistics.median([abs(v) for v in vals])
        if mid:
            rels.append(100.0 * (max(vals) - min(vals)) / mid)
    if not rels:
        return None
    rels.sort()
    return round(rels[int(0.10 * (len(rels) - 1))], 6)


def _envelope_exceedances(rs, cs, band, key, n):
    """THE envelope check, factored out so the gate and its null control run the SAME instrument.

    A control that used a reimplementation of this logic would prove nothing about the gate.
    Returns (exceeded_step_numbers, worst_ratio_to_limit).
    """
    env = (band.get("envelope") or {}).get(key) or []
    safety = band.get("safety", 3.0)
    rel_floor = ((band.get("rel_floor_pct") or {}).get(key)
                 or STEP1_REL_FLOOR_PCT)
    exceed, worst = [], 0.0
    for i in range(min(n, len(env))):
        if env[i] is None or rs[key][i] is None or cs[key][i] is None:
            continue
        d = abs(cs[key][i] - rs[key][i])
        lim = max(env[i] * safety, abs(rs[key][i]) * (rel_floor / 100.0))
        if d > lim:
            exceed.append(i + 1)
            worst = max(worst, d / lim if lim else float("inf"))
    return exceed, round(worst, 3)


def _exceedance_is_pattern(exceed, checked):
    """One isolated exceedance is chaos; a pattern is a bug. Shared by the gate and its control."""
    return len(exceed) > max(1, int(0.1 * checked))


def _null_control(series, dirs, band, n):
    """Does this band reject a pair that changed NOTHING?

    Every calibration leg is a baseline, so any two of them have a true numerics delta of zero by
    construction. If the band flags such a pair, it is not measuring a kernel change -- it is
    measuring chaos, and an L1 FAIL carries no information. Running it here, inside calibrate(),
    means the band cannot come into existence without its own control verdict attached: there is no
    separate step for a Director or an Auditor to skip, which is how both real runs ended up
    hand-recalibrating mid-flight.

    HONEST LIMITATION, recorded in the band: these pairs are the same legs the band was fitted to, so
    the test is self-fitting and therefore WEAK -- passing it is necessary, not sufficient. It still
    catches the observed failures: the 0.5B band rejected its own calibration pair.
    """
    worst = {"loss": 0.0, "grad_norm": 0.0}
    failures = []
    for a in range(len(series)):
        for b in range(a + 1, len(series)):
            for key in ("loss", "grad_norm"):
                exceed, w = _envelope_exceedances(series[a], series[b], band, key, n)
                worst[key] = max(worst[key], w)
                checked = min(n, len((band.get("envelope") or {}).get(key) or []))
                if _exceedance_is_pattern(exceed, checked):
                    failures.append({
                        "pair": [os.path.basename(str(dirs[a]).rstrip("/")),
                                 os.path.basename(str(dirs[b]).rstrip("/"))],
                        "metric": key, "exceeded_steps": exceed[:8],
                        "worst_ratio_to_limit": w})
    return {
        "passed": not failures,
        "pairs_tested": len(series) * (len(series) - 1) // 2,
        "worst_ratio_to_limit": {k: round(v, 3) for k, v in worst.items()},
        "failures": failures[:6],
        "note": ("Baseline-vs-baseline pairs have a true delta of zero. A band that flags them "
                 "cannot discriminate a kernel swap from nothing and must not cast a blocking vote. "
                 "Self-fitting (same legs the band was built from), so passing is necessary, not "
                 "sufficient."),
    }


def _out_of_sample_control(holdout_dir, series, dirs, band, n):
    """The control the band CANNOT run on itself: a clean leg that is NOT in the calibration set.

    Added 2026-09-24 after the in-sample control was falsified twice on one run. `_null_control`
    tests calibration legs against each other, so the band has already been fitted to exactly those
    legs and the test is self-fitting -- it scored 0 exceedances in-sample while a FRESH reference
    leg (same kernels, same seed, same config_digest, step-1 loss delta exactly 0.0) failed against
    all three band legs with 5/5/6 loss exceedances, worst ratios 2.332/1.999/2.200. A second,
    independent instance the same day: two legs differing only by an already-ACCEPTED overlay failed
    with worst 9.534x. So `envelope_discriminating: true` from the in-sample control was simply
    wrong out of sample, and it is the flag that licenses the envelope to reject candidates.

    Passing in-sample is necessary and NOT sufficient; only this function's verdict may license
    BLOCKING.
    """
    hs = _series(holdout_dir)
    failures, worst = [], {"loss": 0.0, "grad_norm": 0.0}
    for i, s in enumerate(series):
        for key in ("loss", "grad_norm"):
            exceed, w = _envelope_exceedances(s, hs, band, key, n)
            worst[key] = max(worst[key], w)
            checked = min(n, len((band.get("envelope") or {}).get(key) or []))
            if _exceedance_is_pattern(exceed, checked):
                failures.append({"band_leg": os.path.basename(str(dirs[i]).rstrip("/")),
                                 "metric": key, "exceeded_steps": exceed[:8],
                                 "worst_ratio_to_limit": w})
    return {
        "holdout": os.path.basename(str(holdout_dir).rstrip("/")),
        "passed": not failures,
        "pairs_tested": len(series),
        "worst_ratio_to_limit": {k: round(v, 3) for k, v in worst.items()},
        "failures": failures[:6],
        "note": ("A clean leg OUTSIDE the calibration set. Unlike the in-sample control this one is "
                 "not fitted to the band, so only this verdict may license a blocking envelope."),
    }


def calibrate(dirs, safety=3.0, holdout=None):
    """Noise floor from >=2 INDEPENDENT baseline runs at identical config.

    Produces a STEP-INDEXED envelope, not a scalar: chaotic divergence means the tolerable delta at
    step 20 is orders of magnitude larger than at step 2, and collapsing that into one number yields
    a gate that passes everything (measured: a 6.1 limit on a loss of 3).

    The band it returns carries its own PROVENANCE and its own NULL-CONTROL verdict, because a 2-leg
    band and a 5-leg band used to be byte-indistinguishable to verdict() -- which is exactly how a
    non-discriminating band got used as evidence in two consecutive runs.
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
            # Widest disagreement seen among baseline runs AT THIS STEP. With 2 legs this is ONE
            # sample of the noise; the leg count is what makes it an estimate rather than a coin flip.
            env[key].append(round(max(vals) - min(vals), 8))
    step1_rel = None
    l1 = [s["loss"][0] for s in series if s["loss"] and s["loss"][0] is not None]
    if len(l1) >= 2 and l1[0]:
        step1_rel = round(100 * (max(l1) - min(l1)) / abs(l1[0]), 6)
    g1 = [s["grad_norm"][0] for s in series if s["grad_norm"] and s["grad_norm"][0] is not None]
    gn1_rel = round(100 * (max(g1) - min(g1)) / abs(g1[0]), 6) if len(g1) >= 2 and g1[0] else None
    band = {
        "safety": safety, "runs": list(dirs), "steps": n,
        "envelope": env,
        "rel_floor_pct": {k: _rel_floor_pct(series, k, n) for k in ("loss", "grad_norm")},
        "step1_loss_rel_pct": step1_rel,
        "step1_grad_norm_rel_pct": gn1_rel,
        "early_steps": EARLY_STEPS,
        "n_legs": len(series),
        # WHAT THIS BAND APPLIES TO. A band is a per-step envelope fitted to legs of a particular
        # argv and a particular length; applied to anything else it is measuring a different process.
        # This was found by hand once: a band calibrated on argv_hash ed00d2f9 / 30 steps was used to
        # judge legs of argv_hash 977d6aac / 60 steps, and its sole firing detector then rejected a
        # true-zero-delta pair 2.5x harder than it rejected the candidate.
        "calibrated_argv_hashes": sorted({s.get("argv_hash") for s in series if s.get("argv_hash")}),
        "note": ("envelope[i] is the widest baseline-vs-baseline disagreement AT STEP i; the gate "
                 "allows safety x that, floored by the metric's OWN measured relative dispersion "
                 "(rel_floor_pct). step1 loss is bit-reproducible in a healthy stack, so its "
                 "floor is deliberately tight."),
    }
    # The band must survive its own control before it may block anything. Computed here, stored here:
    # verdict() reads the flag rather than re-deriving it, so the two can never disagree.
    # What sign balance do CLEAN pairs produce on this workload? Without this the gate treats 0.0 as
    # proof of a one-sided bias -- and on 2026-09-28 a true-zero-delta pair scored exactly 0.0 while
    # another scored 0.75. With n = EARLY_STEPS (8), a clean pair reaching 0.0 by chance is not rare:
    # trajectories separate early and stay separated, so the 8 samples are serially correlated and
    # nothing like 8 independent coin flips.
    sb_null = {}
    for key in ("loss", "grad_norm"):
        vals = []
        for a in range(len(series)):
            for b in range(a + 1, len(series)):
                d = [series[b][key][i] - series[a][key][i] for i in range(min(n, EARLY_STEPS))
                     if series[a][key][i] is not None and series[b][key][i] is not None]
                if len(d) >= 5:
                    vals.append(sum(1 for v in d if v > 0) / len(d))
        if vals:
            sb_null[key] = {"min": round(min(vals), 4), "max": round(max(vals), 4),
                            "pairs": len(vals), "values": [round(v, 4) for v in vals]}
    band["sign_balance_null_range"] = sb_null
    band["null_control"] = _null_control(series, dirs, band, n)
    # An OUT-OF-SAMPLE control is now REQUIRED to license a blocking envelope. The in-sample one is
    # still computed and reported -- it is cheap and it catches gross defects -- but it was
    # falsified twice in one run (see _out_of_sample_control), so on its own it may only ever
    # DEMOTE, never license.
    band["out_of_sample_control"] = (
        _out_of_sample_control(holdout, series, dirs, band, n) if holdout else None)
    band["envelope_discriminating"] = bool(
        band["null_control"]["passed"]
        and band["out_of_sample_control"] and band["out_of_sample_control"]["passed"]
        and len(series) >= MIN_BAND_LEGS)
    if not band["envelope_discriminating"]:
        why = ("only %d leg(s); %d is the measured minimum" % (len(series), MIN_BAND_LEGS)
               if len(series) < MIN_BAND_LEGS
               else "no out-of-sample holdout leg was supplied, and an in-sample control is "
                    "self-fitting (it scored 0 while a fresh clean leg failed 3/3)"
               if not holdout
               else "a clean leg OUTSIDE the calibration set fails it"
               if not band["out_of_sample_control"]["passed"]
               else "it rejects a baseline-vs-baseline null control")
        band["envelope_status"] = (
            "ADVISORY -- the per-step envelope is REPORTED BUT DOES NOT VOTE, because %s. The "
            "blocking detectors remain the step-1 forward probe and the early-window sign balance, "
            "which are unaffected. Recalibrate from >=%d independent clean legs of the SAME identity "
            "to restore a blocking envelope." % (why, RECOMMENDED_BAND_LEGS))
        print("[trajectory_gate] WARNING: " + band["envelope_status"], flush=True)
    else:
        band["envelope_status"] = "BLOCKING -- passed its null control on %d baseline pair(s)" % (
            band["null_control"]["pairs_tested"])
    return band


def verdict(ref_dir, cand_dir, band, candidate_kind="overlay", l0_strictly_better=False):
    """The L1 verdict, in the shape training_workflow.js's gateVerdict() consumes: {pass, reason}."""
    rs, cs = _series(ref_dir), _series(cand_dir)
    res = {"pass": None, "reason": "", "checks": {}}

    # Compare WORKLOAD IDENTITY only. A candidate is supposed to change the implementation; requiring
    # the whole config to match refuses every backend swap as "incomparable" without ever looking at
    # the trajectory -- which is the evidence that actually decides whether the swap is safe.
    # SECOND COPY of probe_summarize.py's filter -- keep the two in step. identity is the COMPLEMENT
    # of IMPL here too, so an unlisted new key becomes workload identity and this returns "legs
    # trained a different WORKLOAD" for every overlay candidate. snapshot_keys is record metadata and
    # belongs to neither half (a v2-vs-v3 pair is caught by digest_schema, with a remedy that says
    # RE-MEASURE rather than the impossible re-summarise).
    IMPL = {"attention-backend", "GEAK_OVERLAY", "GEAK_OVERLAY_SHA"}
    META = {"snapshot_keys"}
    ident = lambda c: {k: v for k, v in (c or {}).items()
                       if k not in IMPL and k not in META and not k.startswith("NVTE_")}
    if ident(rs.get("config")) != ident(cs.get("config")):
        return {"pass": False, "reason": "legs trained a different WORKLOAD (batch/seq/mapping/data "
                                         "differ); step-aligned trajectories are not comparable",
                "checks": {}}
    impl_delta = {k: (v, (cs.get("config") or {}).get(k))
                  for k, v in (rs.get("config") or {}).items()
                  if (k in IMPL or k.startswith("NVTE_")) and (cs.get("config") or {}).get(k) != v}
    n = min(len(rs["loss"]), len(cs["loss"]))
    if n < 3:
        return {"pass": False, "reason": "fewer than 3 comparable steps", "checks": {}}

    safety = float(band.get("safety", 3.0))
    env = band.get("envelope", {})
    failures = []
    # Things the envelope WOULD have failed on if it had proven it can discriminate. Kept separate
    # from `failures` so they never silently become a verdict, and never silently vanish either.
    advisories = []

    # ---- check 1: the step-1 forward probe (zero natural noise) -------------------------------
    if rs["loss"] and cs["loss"] and rs["loss"][0]:
        rel = 100 * abs(cs["loss"][0] - rs["loss"][0]) / abs(rs["loss"][0])
        floor = STEP1_FLOOR_PCT.get(candidate_kind, STEP1_REL_FLOOR_PCT)
        lim = max(floor, (band.get("step1_loss_rel_pct") or 0) * safety)
        res["checks"]["step1_loss_rel_pct"] = {"value": round(rel, 6), "limit": round(lim, 6),
                                               "candidate_kind": candidate_kind}
        if rel > lim:
            failures.append(f"step-1 loss differs by {rel:.4f}% (limit {lim:.4f}%): the forward path "
                            f"changed before any optimizer update could explain it")

    # ---- check 1b: step-1 GRADIENT probe (added 2026-10-09) ------------------------------------
    # The step-1 loss probe watches the FORWARD. This watches the BACKWARD, at the one step where
    # both legs hold identical weights and see the identical batch.
    #
    # Found because a forward-only probe passed a broken backward: Megatron's flash attention
    # backend on this ROCm stack produces step-1 loss within 0.11% of fused at dropout 0, while its
    # gradient norm is 2.7x too small (17.45 vs 47.93, fused matching HF's 47.89 to 0.08%). In the
    # production recipe the same backend reads grad_norm 26.7 against 68.0.
    #
    # Both controls measured before the threshold was set:
    #   null      696 clean same-identity pairs: |rel diff| p50 0.26%, p99 1.05%, MAX 1.40%
    #             (step-1 grad_norm is not bit-reproducible -- backward atomics -- but it is tight)
    #   positive  flash backend -61% (dropout 0.1) and -64% (dropout 0); the rejected 7B attention
    #             candidate -88.5%
    # 10% sits 7x above the worst null pair and 6x below the mildest known defect.
    if rs["grad_norm"] and cs["grad_norm"] and rs["grad_norm"][0]:
        relg = 100 * abs(cs["grad_norm"][0] - rs["grad_norm"][0]) / abs(rs["grad_norm"][0])
        res["checks"]["step1_grad_norm_rel_pct"] = {"value": round(relg, 4), "limit": STEP1_GN_LIMIT_PCT,
                                                    "null_max_pct": 1.401, "pairs": 696}
        if relg > STEP1_GN_LIMIT_PCT:
            failures.append(f"step-1 grad_norm differs by {relg:.2f}% (limit {STEP1_GN_LIMIT_PCT}%; "
                            f"696 clean pairs never exceeded 1.40%): the backward computed a different "
                            f"gradient on identical weights and an identical batch")

    # ---- check 2: per-step envelope ------------------------------------------------------------
    # Blocking ONLY if the band proved it can tell a kernel swap from nothing (calibrate() runs that
    # control and stores the verdict). An advisory envelope still reports every number it would have
    # blocked on -- the numbers are real and worth reading -- it just does not convert them into a
    # FAIL. Reporting a failure we have PROVEN is an instrument artifact is worse than silence: it
    # trains the reader to override L1, which is the habit that lets a real rejection through.
    # Legacy bands (written before this field existed) have no control verdict; treat a MISSING flag
    # as advisory rather than assuming the band is sound.
    # RETIRED FROM VOTING, 2026-09-28. The per-step envelope NEVER blocks. This is a decision taken
    # on evidence, not a temporary demotion, and it is deliberately not reachable by configuration.
    #
    # Across four real runs it fired every time and was a load-bearing detector zero times:
    #   0.5B run   fired -> the auditor discounted it (band from a same-kernel pair over-rejects a
    #              legitimate swap); the sign test carried that rejection.
    #   7B run 2   fired 8 loss / 10 grad_norm; the NULL control failed 8/10 at worst 36.27x against
    #              the candidate's 37.36x -- statistically indistinguishable.
    #   7B run 3   fired; an out-of-sample clean leg failed 3/3 (5/5/6 steps), and a second pair
    #              differing only by an ALREADY-ACCEPTED overlay failed at worst 9.534x.
    #   7B run 4   fired 6 steps at worst 7.107; a TRUE-ZERO-DELTA pair was rejected 2.5x HARDER
    #              (worst 17.928).
    # And its ceiling is structural, measured by injection: a +25% loss shift across the whole steady
    # window is invisible to it, as is grad_norm x1.5. Only a gross ~3x deviation registers -- because
    # by step 12 three IDENTICAL legs already disagree by ~8.9% and the envelope allows safety x that.
    #
    # It was patched four times (>=3 legs, a p10 floor, an out-of-sample holdout, an argv/length scope
    # check). Every patch fixed a real defect. None of them produced a verdict anyone could rely on.
    # A detector that has never been right, and whose noise floor is set by chaos rather than by
    # anything we can buy, does not get a fifth attempt at voting.
    #
    # What still votes: the step-1 forward probe (zero natural noise; it caught the attention backward
    # defect at -88.50% grad_norm and the 0.5B attention swap at 0.4274% step-1 loss) and the
    # early-window sign balance. The envelope's numbers are still COMPUTED and REPORTED -- they are
    # real measurements and occasionally interesting -- they simply carry no verdict.
    blocking = False
    # APPLICABILITY. Even a band that earned BLOCKING may only vote on legs it was calibrated for.
    # Checked here rather than left to the reader: the one time this was caught, it took an auditor
    # running an out-of-sample null pair by hand to notice, and the detector had already returned a
    # confident pass=false on a candidate it could not judge.
    cal = band.get("calibrated_argv_hashes") or []
    leg_hashes = [h for h in (rs.get("argv_hash"), cs.get("argv_hash")) if h]
    out_of_scope = bool(cal and leg_hashes and any(h not in cal for h in leg_hashes))
    res["checks"]["band_applicability"] = {
        "calibrated_argv_hashes": cal, "leg_argv_hashes": leg_hashes,
        "band_steps": band.get("steps"), "in_scope": not out_of_scope}
    if out_of_scope:
        blocking = False
        res["checks"]["envelope_status"] = (
            "ADVISORY -- this band was calibrated on argv_hash %s and these legs are %s. A per-step "
            "envelope is fitted to a particular argv and leg length; applied to anything else it "
            "describes a different process. Re-calibrate on >=3 clean legs of THIS config (with a "
            "holdout) before the envelope may vote." % (cal, leg_hashes))
        print("[trajectory_gate] WARNING: " + res["checks"]["envelope_status"], flush=True)
    res["checks"]["envelope_blocking"] = blocking
    res["checks"]["envelope_status"] = band.get(
        "envelope_status", "ADVISORY -- band predates the null-control self-test")
    for key in ("loss", "grad_norm"):
        e = env.get(key) or []
        exceed, worst = _envelope_exceedances(rs, cs, band, key, n)
        res["checks"][f"{key}_envelope"] = {
            "steps_checked": min(n, len(e)), "exceeded_steps": exceed,
            "worst_ratio_to_limit": worst, "blocking": blocking}
        # One isolated exceedance is chaos; a pattern is a bug.
        if _exceedance_is_pattern(exceed, min(n, len(e))):
            msg = (f"{key} left the calibrated envelope on {len(exceed)} steps "
                   f"({exceed[:6]}{'...' if len(exceed) > 6 else ''})")
            if blocking:
                failures.append(msg)
            else:
                advisories.append(msg)

    # ---- check 3: one-sidedness over the early window ------------------------------------------
    for key in ("loss", "grad_norm"):
        d = [cs[key][i] - rs[key][i] for i in range(min(n, EARLY_STEPS))
             if rs[key][i] is not None and cs[key][i] is not None]
        if len(d) >= 5:
            sb = sum(1 for v in d if v > 0) / len(d)
            res["checks"][f"{key}_sign_balance_early"] = {"value": round(sb, 4), "n": len(d)}
            # A detector may only reject a candidate for doing something CLEAN PAIRS DO NOT DO.
            # `sb in (0.0, 1.0)` was an assumption about what noise looks like, and the assumption is
            # false on this workload: measured 2026-09-28, a pair with a true delta of exactly zero
            # scored 0.0 on loss while another scored 0.75. Firing on 0.0 there would have rejected a
            # leg for being identical to itself. So the bar is the observed clean range, when the band
            # carries one -- the same discipline the per-step envelope was held to, and failed.
            # THE L0/L1 RULE (written 2026-10-08, BEFORE being applied to any candidate).
            #
            # A sign test cannot tell a correctness fix from a bias: both make the loss move one way.
            # Measured case: `attn_fmha_bwd_retry` was rejected on loss sign balance 0.000 -- the only
            # voting detector that fired -- while L0 rated it strictly MORE accurate than the
            # incumbent on every output, the incumbent failing an analytic dq invariant by 1355.79x.
            # Both statements were true and the framework had no rule, so the verdict depended on
            # which layer was read last.
            #
            # THE RULE: when L0 rates the candidate strictly more accurate than the incumbent on all
            # outputs, a one-sided move TOWARD LOWER LOSS is corroboration, not evidence of bias, and
            # the sign test may not reject on it alone.
            #
            # THE ASYMMETRY IS THE WHOLE SAFETY ARGUMENT, so it is spelled out rather than implied:
            #   * lower loss  + L0 strictly better -> waived. A more accurate op that descends faster
            #                                         is the expected shape of a real fix.
            #   * HIGHER loss + L0 strictly better -> STILL BLOCKS. Being more accurate on one frozen
            #                                         microbatch does not license a trajectory moving
            #                                         the wrong way; that combination is unexplained
            #                                         and unexplained is not safe.
            #   * either direction, L0 not strictly better -> STILL BLOCKS. This is the dangerous case
            #                                         the rule must never cover: a candidate that is
            #                                         LESS accurate and still descends faster is
            #                                         exactly what a downhill-pointing bias looks like.
            # Waived is not silent: it is reported as an advisory with the reason.
            if l0_strictly_better and key == "loss" and sb == 0.0:
                advisories.append(
                    "loss moved one way on all early steps (sign balance 0.000) -- WAIVED, not "
                    "blocking: L0 rates this candidate strictly more accurate than the incumbent on "
                    "every output, so a faster descent is the expected shape of a correctness fix "
                    "rather than evidence of bias. A one-sided move toward HIGHER loss, or either "
                    "direction without that L0 verdict, still blocks.")
                res["checks"][f"{key}_sign_balance_early"]["waived_by_l0"] = True
                continue
            # THE RULE WAS REPLACED 2026-10-09; THE DETECTOR WAS NOT RETIRED.
            #
            # What it used to be held to -- the min..max of the band's 3 clean pairs -- is what noise
            # leaves most of the time. Re-measured on 693 clean same-identity pairs: OUTSIDE that
            # range on loss 63.5%, grad_norm 55.6%. A reject-only gate rejecting a coin-flip of good
            # candidates. The arithmetic agrees: at n = 8 the statistic moves in 0.125 steps and a
            # fair coin lands in {0.5, 0.625} only 49% of the time. Three draws cannot bound that.
            #
            # STRICT RULE: block only when every one of >= 5 non-identical early diffs has the same
            # sign. Identical steps are excluded rather than counted as "not up", which is what made
            # a leg identical to its reference read as one-sided. Both controls are measured:
            #   null      693 clean pairs: loss 5.1%, grad_norm 0.6% false reject
            #   positive  fires on a real gross backward defect (7B attention, step-1 grad_norm
            #             74 -> 8.5, dropout-0 pair: loss 0.0, grad_norm 0.0), and on a gross one-sided
            #             drift that the step-1 probe structurally cannot see (synthetic, below)
            # Its blind spots are stated, not implied: it misses cross-step staleness and a x1.004
            # gradient bias on every leg tried, and at dropout 0.1 the dropout noise hides even the
            # real defect above (0.375 / 0.5) -- which the step-1 probe catches. So this is a
            # backstop for GROSS one-sided moves, not a detector of subtle ones.
            #
            # Retiring it outright was considered and rejected: a loss shift of +0.9 from step 3 on,
            # step 1 untouched, would then pass L1 entirely.
            nz = [v for v in d if v != 0]
            strict = len(nz) >= 5 and (all(v > 0 for v in nz) or all(v < 0 for v in nz))
            chk = res["checks"][f"{key}_sign_balance_early"]
            chk["strict_one_sided"] = strict
            chk["non_identical_steps"] = len(nz)
            chk["measured_null_rate_pct"] = {"loss": 5.1, "grad_norm": 0.6}[key]
            null_rng = (band.get("sign_balance_null_range") or {}).get(key)
            if null_rng:
                chk["band_range_reported_only"] = [null_rng["min"], null_rng["max"]]
            chk["blocking"] = strict
            if strict:
                failures.append(
                    f"{key} early sign balance {sb:.3f}: moved the SAME direction on all {len(nz)} non-identical early steps, a "
                    f"gross one-sided move. Clean pairs do this {chk['measured_null_rate_pct']}% of the "
                    f"time (693 measured pairs), so a re-gate is the remedy if this is a false reject")

    res["implementation_delta"] = impl_delta
    res["pass"] = not failures
    res["advisories"] = advisories
    if failures:
        res["reason"] = "; ".join(failures)
    elif advisories:
        # PASS, but say exactly what was set aside and why -- an unqualified "within the envelope"
        # here would be a lie, since the envelope was never allowed to vote.
        res["reason"] = ("no blocking detector fired. NOT CLEARED BY THE ENVELOPE: "
                         + "; ".join(advisories)
                         + f" -- reported only, because {res['checks'].get('envelope_status', '')}")
    else:
        res["reason"] = "within the calibrated per-step envelope"
    return res


def calibrate_swap(a_runs, b_runs, safety=3.0):
    """Calibrate a KERNEL-SWAP band from a 2x2: two configs, two repeats each.

    A 1x1 comparison cannot tell the swap's contribution from the machine's noise. With repeats, the
    two are separable:

        within  = disagreement between two runs of the SAME config   -> the noise floor
        between = disagreement across configs                        -> noise + the swap

    The band that a kernel-swap candidate should be judged against is `between`; `within` is the
    control that says whether `between` means anything. If within ~= between, the "swap" changed
    nothing measurable and the pair is useless for calibration. If within is ~0 (which is what a
    healthy stack gives at step 1 -- bit-identical) then `between` IS the swap.
    """
    A = [_series(d) for d in a_runs]
    B = [_series(d) for d in b_runs]
    n = min(len(s["loss"]) for s in A + B)

    def spread(group, i, key):
        vals = [g[key][i] for g in group if g[key][i] is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else None

    def cross(i, key):
        av = [g[key][i] for g in A if g[key][i] is not None]
        bv = [g[key][i] for g in B if g[key][i] is not None]
        if not av or not bv:
            return None
        return max(abs(x - y) for x in av for y in bv)

    env_within, env_between = {"loss": [], "grad_norm": []}, {"loss": [], "grad_norm": []}
    for key in ("loss", "grad_norm"):
        for i in range(n):
            w = max([x for x in (spread(A, i, key), spread(B, i, key)) if x is not None] or [0.0])
            env_within[key].append(round(w, 8))
            c = cross(i, key)
            env_between[key].append(None if c is None else round(c, 8))

    def rel1(group_a, group_b, key):
        av = [g[key][0] for g in group_a if g[key][0] is not None]
        bv = [g[key][0] for g in group_b if g[key][0] is not None]
        if not av or not bv or not av[0]:
            return None
        return round(100 * max(abs(x - y) for x in av for y in bv) / abs(av[0]), 6)

    # VERDICT ON THE PAIR ITSELF. Without this the reader has to notice that `between` and `within`
    # are the same size and conclude the swap did nothing -- and the first pair tried
    # (NVTE_USE_HIPBLASLT on/off) was exactly that case: step-1 loss BIT-IDENTICAL across configs,
    # envelopes indistinguishable, throughput within noise. A band built from it would have looked
    # like "kernel swaps are free", which is the opposite of the truth.
    def _usable(between, within):
        pairs = [(b_, w_) for b_, w_ in zip(between, within)
                 if b_ is not None and w_ is not None and (b_ > 0 or w_ > 0)]
        if not pairs:
            return 0.0
        return sum(1 for b_, w_ in pairs if b_ > w_ * 1.5) / len(pairs)

    frac = _usable(env_between["loss"], env_within["loss"])
    step1_between = rel1(A, B, "loss") or 0.0
    step1_within = max(x for x in [rel1(A, A, "loss"), rel1(B, B, "loss"), 0.0] if x is not None)
    usable = bool(step1_between > max(step1_within, 1e-9) or frac >= 0.5)

    return {
        "safety": safety, "steps": n, "kind": "kernel_swap",
        "runs": {"a": list(a_runs), "b": list(b_runs)},
        "pair_usable": usable,
        "pair_verdict": (
            "usable: the two configs differ beyond their own run-to-run spread"
            if usable else
            "UNUSABLE for calibration: `between` is not larger than `within`, i.e. the two configs "
            "produced the same numerics. Whatever knob was toggled did not change the kernels, so a "
            "band built from this pair would understate real kernel-swap noise."),
        "steps_where_between_exceeds_within_frac": round(frac, 3),
        "step1_loss_rel_pct": rel1(A, B, "loss"),
        "step1_loss_rel_pct_within": max(
            x for x in [rel1(A, A, "loss"), rel1(B, B, "loss"), 0.0] if x is not None),
        "step1_grad_norm_rel_pct": rel1(A, B, "grad_norm"),
        "step1_grad_norm_rel_pct_within": max(
            x for x in [rel1(A, A, "grad_norm"), rel1(B, B, "grad_norm"), 0.0] if x is not None),
        "envelope": env_between,
        "envelope_within": env_within,
        "note": ("`envelope` is the band a kernel-swap candidate is judged against; "
                 "`envelope_within` is the control. If they are the same size, the swap changed "
                 "nothing measurable and this pair cannot calibrate anything."),
    }


def _usage():
    print(__doc__.strip().splitlines()[0])
    print("usage:\n  trajectory_gate.py calibrate <baseline_probe> <baseline_probe> [more...] [-o band.json]"
          "\n  trajectory_gate.py compare <ref> <cand> --band <band.json> [--kind overlay|kernel_swap]"
          "\n  trajectory_gate.py calibrate_swap <A1> <A2> -- <B1> <B2> [-o band.json]")
    return 2


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(_usage())
    cmd = sys.argv[1]
    if cmd == "calibrate_swap":
        # calibrate_swap A1 A2 -- B1 B2 [-o band.json]
        rest = [x for x in sys.argv[2:] if not x.startswith("-")]
        if "-o" in sys.argv:
            rest = [d for d in rest if d != sys.argv[sys.argv.index("-o") + 1]]
        split = rest.index("--") if "--" in rest else len(rest) // 2
        band = calibrate_swap(rest[:split], rest[split + 1:] if "--" in rest else rest[split:])
        if "-o" in sys.argv:
            json.dump(band, open(sys.argv[sys.argv.index("-o") + 1], "w"), indent=2)
        print(json.dumps(band, indent=2))
    elif cmd == "calibrate":
        KNOWN = {"-o", "--holdout"}
        for tok in sys.argv[2:]:
            if tok.startswith("-") and tok not in KNOWN:
                sys.exit(f"unknown flag {tok!r}. Known for `calibrate`: {sorted(KNOWN)}.")
        holdout = sys.argv[sys.argv.index("--holdout") + 1] if "--holdout" in sys.argv else None
        dirs = [a for a in sys.argv[2:] if not a.startswith("-")]
        for drop in ([sys.argv[sys.argv.index("-o") + 1]] if "-o" in sys.argv else []) + \
                    ([holdout] if holdout else []):
            dirs = [d for d in dirs if d != drop]
        b = calibrate(dirs, holdout=holdout)
        out = None
        if "-o" in sys.argv:
            out = sys.argv[sys.argv.index("-o") + 1]
            json.dump(b, open(out, "w"), indent=2)
        print(json.dumps(b, indent=2))
    elif cmd == "compare":
        # REFUSE unknown flags. The parsing below is positional + index lookups, so an unrecognised
        # flag is silently DROPPED and its value silently becomes a directory argument. Measured
        # consequence: `--candidate-kind kernel_swap` (the flag is `--kind`) fell through to the
        # default "overlay", applying a 0.05% step-1 floor where 2.0% was intended -- 40x tighter,
        # silently, on the gate that decides whether a candidate is numerically acceptable.
        KNOWN = {"--band", "--kind", "--l0-strictly-better"}
        for i, tok in enumerate(sys.argv[2:], start=2):
            if tok.startswith("--") and tok not in KNOWN:
                sys.exit(f"unknown flag {tok!r}. Known flags for `compare`: {sorted(KNOWN)}. "
                         f"Refusing rather than ignoring it: an ignored --kind silently applies the "
                         f"40x tighter `overlay` floor instead of `kernel_swap`.")
        bandpath = sys.argv[sys.argv.index("--band") + 1] if "--band" in sys.argv else None
        band = json.load(open(bandpath)) if bandpath else {"limits": {}}
        kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "overlay"
        l0_better = "--l0-strictly-better" in sys.argv
        if kind not in STEP1_FLOOR_PCT:
            sys.exit(f"unknown --kind {kind!r}; known: {sorted(STEP1_FLOOR_PCT)}")
        print(json.dumps(verdict(sys.argv[2], sys.argv[3], band, kind, l0_better), indent=2))
    else:
        sys.exit(_usage())
