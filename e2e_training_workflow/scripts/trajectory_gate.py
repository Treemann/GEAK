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


def calibrate(dirs, safety=3.0):
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
        "note": ("envelope[i] is the widest baseline-vs-baseline disagreement AT STEP i; the gate "
                 "allows safety x that, floored by the metric's OWN measured relative dispersion "
                 "(rel_floor_pct). step1 loss is bit-reproducible in a healthy stack, so its "
                 "floor is deliberately tight."),
    }
    # The band must survive its own control before it may block anything. Computed here, stored here:
    # verdict() reads the flag rather than re-deriving it, so the two can never disagree.
    band["null_control"] = _null_control(series, dirs, band, n)
    band["envelope_discriminating"] = bool(
        band["null_control"]["passed"] and len(series) >= MIN_BAND_LEGS)
    if not band["envelope_discriminating"]:
        why = ("only %d leg(s); %d is the measured minimum" % (len(series), MIN_BAND_LEGS)
               if len(series) < MIN_BAND_LEGS else "it rejects a baseline-vs-baseline null control")
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


def verdict(ref_dir, cand_dir, band, candidate_kind="overlay"):
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

    # ---- check 2: per-step envelope ------------------------------------------------------------
    # Blocking ONLY if the band proved it can tell a kernel swap from nothing (calibrate() runs that
    # control and stores the verdict). An advisory envelope still reports every number it would have
    # blocked on -- the numbers are real and worth reading -- it just does not convert them into a
    # FAIL. Reporting a failure we have PROVEN is an instrument artifact is worse than silence: it
    # trains the reader to override L1, which is the habit that lets a real rejection through.
    # Legacy bands (written before this field existed) have no control verdict; treat a MISSING flag
    # as advisory rather than assuming the band is sound.
    blocking = bool(band.get("envelope_discriminating"))
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
            if sb in (0.0, 1.0):
                failures.append(f"{key} moved the SAME direction on all {len(d)} early steps: a "
                                f"systematic bias, not noise (every magnitude check can still pass)")

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
        kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "overlay"
        print(json.dumps(verdict(sys.argv[2], sys.argv[3], band, kind), indent=2))
    else:
        sys.exit(_usage())
