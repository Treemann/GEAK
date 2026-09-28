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


def make_leg(root, name, losses, grad_norms, config=None, argv_hash="aaaa1111bbbb2222"):
    """Write a probe sidecar in the format _series() reads."""
    d = os.path.join(root, name, "probe")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "probe_rank0.jsonl"), "w") as f:
        f.write(json.dumps({"record": "header", "argv_hash": argv_hash,
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

        # A blocking envelope now requires an OUT-OF-SAMPLE holdout: the in-sample control is
        # self-fitting and was falsified twice on real 7B legs (it scored 0 while a fresh clean leg
        # failed against all three band legs). Calibrate on 4 legs, hold the 5th out.
        b5 = tg.calibrate(legs[:4], holdout=legs[4])
        ok(b5["n_legs"] == 4 and b5.get("out_of_sample_control") is not None,
           "a band records its provenance AND its out-of-sample control")
        ok(tg.calibrate(legs[:4])["envelope_discriminating"] is False,
           "without a holdout the envelope may NOT vote -- an in-sample control is self-fitting and was falsified on real legs")

        # ---------------------------------------------------------------- 2. the measured floor
        fl = b5["rel_floor_pct"]
        ok(fl["grad_norm"] > fl["loss"],
           f"the floor is PER-METRIC and measured: grad_norm {fl['grad_norm']}% >> loss {fl['loss']}% "
           "(one 0.05% constant for both was ~4 orders too tight for grad_norm)")
        ok(fl["grad_norm"] > tg.STEP1_REL_FLOOR_PCT,
           "grad_norm's measured floor is wider than the borrowed step-1 loss constant it replaced")

        # The floor must be a LOW quantile of the per-step spread, not a typical one. Setting it to
        # the MEDIAN was tried on real 7B legs and is a different way of being wrong: it lifts the
        # floor to loss 8.93% / grad_norm 152.3%, swamping the envelope so completely that a +25%
        # loss shift across the whole steady window passed. The floor covers the degenerate steps
        # (three identical 7B legs agree to 0.0000% on at least one of them); the envelope speaks
        # everywhere else.
        import statistics as _st
        for key in ("loss", "grad_norm"):
            rels = []
            for i in range(min(len(s[key]) for s in [tg._series(d) for d in legs])):
                vals = [tg._series(d)[key][i] for d in legs]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    mid = _st.median([abs(v) for v in vals])
                    if mid:
                        rels.append(100.0 * (max(vals) - min(vals)) / mid)
            if rels:
                ok(fl[key] <= _st.median(rels),
                   f"{key} floor ({fl[key]}%) is at or below the MEDIAN per-step spread "
                   f"({_st.median(rels):.3f}%) — a floor at the typical disagreement disarms the envelope")

        # ---------------------------------------------------------------- 3. THE regression test
        # A band must not reject a pair that changed nothing. This is the exact defect both runs hit.
        ok(b5["null_control"]["pairs_tested"] == 6,
           "the in-sample control still tests every calibration pair (C(4,2) = 6)")
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

        # RETIRED 2026-09-28: the envelope never blocks, whatever its controls say. Four runs, four
        # firings, zero times load-bearing -- and every firing falsified by a control, most sharply
        # when a true-zero-delta pair was rejected 2.5x harder than the candidate. This assertion is
        # inverted from what it used to be, deliberately: the old one pinned the licensing path that
        # no longer exists.
        v_blk = tg.verdict(legs[0], drift, b5)
        ok(v_blk["checks"]["envelope_blocking"] is False,
           "even a band that passed BOTH controls may not block on an envelope exceedance")
        ok(any("envelope" in a for a in v_blk.get("advisories", [])),
           "...and the exceedance is still reported as an advisory, not silently dropped")
        ok(v_adv["checks"]["loss_envelope"]["worst_ratio_to_limit"] > 0,
           "the advisory numbers are real measurements, not zeroed out")

        # ------------------------------------------------- 4b. the band may only judge its own config
        # A band is fitted to legs of a particular argv and length. Applied to anything else it
        # describes a different process -- found the hard way when a 30-step/ed00d2f9 band returned a
        # confident pass=false on 60-step/977d6aac legs, and its sole firing detector then rejected a
        # true-zero-delta pair 2.5x harder than the candidate.
        oos_root = os.path.join(root, "otherconfig")
        lo_o, gn_o = synth(2, cross_at=11)
        other = make_leg(oos_root, "leg", lo_o, gn_o, argv_hash="deadbeefdeadbeef")
        v_oos = tg.verdict(other, drift, b5)
        ok(v_oos["checks"]["band_applicability"]["in_scope"] is False,
           "a band refuses to claim scope over legs of a different argv_hash")
        ok(v_oos["checks"]["envelope_blocking"] is False,
           "and it may NOT cast a blocking envelope vote on them, even though it earned BLOCKING "
           "on its own config")

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
