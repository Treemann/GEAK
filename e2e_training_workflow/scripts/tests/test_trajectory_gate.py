#!/usr/bin/env python3
"""Tests for scripts/trajectory_gate.py's band calibration and the L1 envelope.

WHY THIS FILE EXISTS. trajectory_gate.py had NO test coverage, and that is how a one-sample
envelope reached two real training runs. In both, the L1 gate returned FAIL on a candidate AND on a
null control -- a same-kernel pair whose true numerics delta is zero by construction -- so the FAIL
carried no information and the Director had to hand-recalibrate mid-flight to get a usable verdict.

The numbers below are the MEASURED ones, not invented:
  - grad_norm at step 2 across three identical 0.5B runs: 152.10 / 218.77 / 83.44. That is the
    dispersion a 0.05% floor was being applied to.
  - the 7B null control failed on 8 loss / 10 grad_norm steps, worse than the candidate it was
    supposed to discriminate from.

No GPU and no torch: probe sidecars are plain JSONL, so the legs are synthesised directly.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import trajectory_gate as tg  # noqa: E402

FAILED = []


def ok(cond, msg):
    print(("  ok: " if cond else "  FAIL: ") + msg)
    if not cond:
        FAILED.append(msg)


def make_leg(root, name, losses, grad_norms, config=None):
    """Write a probe sidecar in the format _series() reads."""
    d = os.path.join(root, name, "probe")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "probe_rank0.jsonl"), "w") as f:
        f.write(json.dumps({"record": "header",
                            "config": config or {"micro-batch-size": "1", "seq-length": "4096"}}) + "\n")
        for i, (l, g) in enumerate(zip(losses, grad_norms)):
            f.write(json.dumps({"record": "step", "step": i + 1, "loss": l, "grad_norm": g}) + "\n")
    return d


def synth(seed, n=30, cross_at=None):
    """A deterministic baseline leg: loss decays smoothly, grad_norm is wildly unstable.

    `cross_at` forces two legs to agree EXACTLY at a step -- the degenerate case that collapses a
    two-leg envelope to zero width and makes every later comparison trip it.
    """
    losses, gns = [], []
    for i in range(n):
        jitter = ((seed * 7919 + i * 104729) % 1000) / 1000.0 - 0.5     # deterministic, no RNG
        losses.append(round(3.2 - 0.02 * i + 0.05 * jitter, 6))
        # grad_norm swings by ~2x run to run, as measured (152/219/83 at step 2)
        gns.append(round(150.0 + 70.0 * jitter * 2 + 3.0 * i, 6))
        if cross_at is not None and i == cross_at:
            losses[-1] = 3.0            # identical across legs -> zero-width envelope step
            gns[-1] = 150.0
    return losses, gns


def main():
    root = tempfile.mkdtemp(prefix="tgtest_")
    try:
        legs = []
        for s in (1, 2, 3, 4, 5):
            lo, gn = synth(s, cross_at=11)
            legs.append(make_leg(root, f"base{s}", lo, gn))

        # ---------------------------------------------------------------- 1. leg count gates voting
        b2 = tg.calibrate(legs[:2])
        ok(b2["n_legs"] == 2 and b2["envelope_discriminating"] is False,
           "a 2-leg band is NOT allowed to cast a blocking envelope vote (one sample is not a noise estimate)")
        ok("ADVISORY" in b2["envelope_status"],
           "the 2-leg band SAYS it is advisory rather than looking identical to a good band")

        b5 = tg.calibrate(legs)
        ok(b5["n_legs"] == 5, "a 5-leg band records its own provenance (n_legs)")

        # ---------------------------------------------------------------- 2. the measured floor
        fl = b5["rel_floor_pct"]
        ok(fl["grad_norm"] > fl["loss"],
           f"the floor is PER-METRIC and measured: grad_norm {fl['grad_norm']}% >> loss {fl['loss']}% "
           "(one 0.05% constant for both was ~4 orders too tight for grad_norm)")
        ok(fl["grad_norm"] > tg.STEP1_REL_FLOOR_PCT,
           "grad_norm's measured floor is wider than the borrowed step-1 loss constant it replaced")

        # ---------------------------------------------------------------- 3. THE regression test
        # A band must not reject a pair that changed nothing. This is the exact defect both runs hit.
        ok(b5["null_control"]["pairs_tested"] == 10,
           "the null control tests every baseline pair (C(5,2) = 10)")
        v_null = tg.verdict(legs[0], legs[1], b5)
        ok(v_null["pass"] is True,
           "a NULL CONTROL (two baseline legs, true delta zero) does NOT fail L1 -- "
           "a gate that rejects the baseline rejects everything")

        # ---------------------------------------------------------------- 4. advisory != silent
        # A candidate that really does leave the envelope, judged by an advisory band and by a
        # blocking one. The exceedance must be REPORTED either way and COUNTED only by the band that
        # earned the right to count it.
        lo_d, gn_d = synth(1, cross_at=11)
        for i in range(3, 30):
            lo_d[i] += 0.9                        # large mid-run divergence, step 1 untouched
        drift = make_leg(root, "cand_drift", lo_d, gn_d)

        v_adv = tg.verdict(legs[0], drift, b2)
        ok(any("envelope" in a for a in v_adv.get("advisories", [])),
           "an advisory band still REPORTS the exceedance it would have blocked on")
        ok(not any("envelope" in f for f in v_adv["reason"].split(";")),
           "...but the exceedance does NOT enter the advisory band's verdict")

        v_blk = tg.verdict(legs[0], drift, b5)
        ok("envelope" in v_blk["reason"] and v_blk["pass"] is False,
           "a band that PASSED its null control does block on the same exceedance")
        ok(v_adv["checks"]["loss_envelope"]["worst_ratio_to_limit"] > 0,
           "the advisory numbers are real measurements, not zeroed out")

        # ---------------------------------------------------------------- 5. legacy bands
        legacy = dict(b5)
        legacy.pop("envelope_discriminating", None)
        legacy.pop("envelope_status", None)
        v_leg = tg.verdict(legs[0], legs[1], legacy)
        ok(v_leg["checks"]["envelope_blocking"] is False,
           "a band written BEFORE the self-test existed is treated as advisory, not assumed sound")

        # ---------------------------------------------------------------- 6. the step-1 probe lives
        # The detector that actually discriminates must still reject a changed forward path.
        lo, gn = synth(1, cross_at=11)
        lo[0] = lo[0] * 1.05                      # 5% step-1 loss change: a real forward-path change
        bad = make_leg(root, "cand_fwd", lo, gn)
        v_bad = tg.verdict(legs[0], bad, b5, candidate_kind="overlay")
        ok(v_bad["pass"] is False and "step-1 loss" in v_bad["reason"],
           "demoting the envelope did NOT disarm L1: a changed forward path is still rejected")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
