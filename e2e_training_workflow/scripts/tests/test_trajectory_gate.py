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

        # ------------------------------------------- 4c. sign balance is held to a MEASURED range
        # A detector may only reject a candidate for doing something clean pairs do not do. Measured
        # 2026-09-28: a true-zero-delta pair scored sign balance 0.0 on loss while another scored
        # 0.75, and the old `sb in (0.0, 1.0)` rule would have rejected a leg for being identical to
        # itself. It DID reject a real candidate that L0 called strictly more accurate than the
        # incumbent.
        ok("sign_balance_null_range" in b5,
           "the band records what sign balance its own clean pairs produced")
        rng = b5["sign_balance_null_range"].get("loss") or {}
        ok(rng.get("pairs", 0) >= 2,
           f"...over {rng.get('pairs')} pairs, so it is a range and not a single sample")
        # a candidate sitting inside the clean range must not be blocked on this detector
        v_in = tg.verdict(legs[0], legs[1], b5)
        lsb = v_in["checks"].get("loss_sign_balance_early") or {}
        if "blocking" in lsb:
            ok(lsb["blocking"] is False,
               "a clean pair's own sign balance does not fire the detector calibrated from it")

        # ----------------------------------------------- 4d. the L0/L1 rule, all four combinations
        # A sign test cannot tell a correctness fix from a bias. The rule waives a one-sided move
        # TOWARD LOWER LOSS when L0 rates the candidate strictly more accurate. The asymmetry is the
        # safety argument, so every combination is pinned -- not just the one that passes.
        # grad_norm is given an ALTERNATING perturbation in both candidates so its own sign test
        # lands mid-range. Without it an untouched grad_norm has delta exactly 0 everywhere, which
        # scores sign balance 0.0 and blocks for a reason that has nothing to do with the loss rule
        # under test -- a test artifact, not a property of the gate.
        def alt(gn):
            return [g + (0.5 if i % 2 else -0.5) for i, g in enumerate(gn)]
        lo_lo, gn_lo = synth(1, cross_at=11)
        for i in range(0, 8):
            lo_lo[i] -= 0.05                      # one-sided TOWARD LOWER loss over the early window
        lower = make_leg(root, "cand_lower", lo_lo, alt(gn_lo))
        lo_hi, gn_hi = synth(1, cross_at=11)
        for i in range(0, 8):
            lo_hi[i] += 0.05                      # one-sided toward HIGHER loss
        higher = make_leg(root, "cand_higher", lo_hi, alt(gn_hi))

        # The null-range rule already makes sb=0.0 unblockable when CLEAN PAIRS THEMSELVES reach
        # 0.0 -- which they do here (6 pairs: 0.125 0.25 0.25 0.125 0.125 0.0) and did on the real
        # 7B legs too (0.0 and 0.75 both observed). So on that data the simpler fix alone already
        # resolves `attn_fmha_bwd_retry`; the L0 rule is a SECOND, independent reason, and it matters
        # only where sb falls OUTSIDE the clean range. Test it against a band whose range excludes 0.
        import copy
        b_tight = copy.deepcopy(b5)
        b_tight["sign_balance_null_range"] = {
            "loss": {"min": 0.3, "max": 0.7, "pairs": 6, "values": [0.3, 0.5, 0.7]},
            "grad_norm": {"min": 0.3, "max": 0.7, "pairs": 6, "values": [0.3, 0.5, 0.7]}}

        def sb_blocks(cand, l0_better):
            v = tg.verdict(legs[0], cand, b_tight, l0_strictly_better=l0_better)
            return v["pass"] is False and any("sign balance" in f for f in v["reason"].split(";"))

        ok(sb_blocks(lower, False),
           "lower loss, L0 NOT strictly better -> BLOCKS (a less accurate op descending faster is "
           "exactly what a downhill bias looks like)")
        ok(not sb_blocks(lower, True),
           "lower loss, L0 strictly better -> waived (the expected shape of a correctness fix)")
        ok(sb_blocks(higher, True),
           "HIGHER loss, L0 strictly better -> STILL BLOCKS; accuracy on one frozen microbatch does "
           "not license a trajectory moving the wrong way")
        ok(sb_blocks(higher, False),
           "higher loss, L0 not strictly better -> blocks")
        # ------------------------------------- 4e. the STRICT rule replaces the 3-pair range
        # The old bar was min..max of the band's 3 clean pairs; 693 measured clean pairs fall
        # outside such a range 55-64% of the time. A 6-up/2-down pair (0.75) is ordinary noise and
        # must not block, even against a band whose recorded range excludes it.
        lo_m, gn_m = synth(1, cross_at=11)
        for i in range(0, 8):
            lo_m[i] += (0.05 if i not in (2, 5) else -0.05)
        mixed = make_leg(root, "cand_mixed", lo_m, alt(gn_m))
        v_m = tg.verdict(legs[0], mixed, b_tight)
        sbm = v_m["checks"].get("loss_sign_balance_early") or {}
        ok(sbm.get("blocking") is False and not sbm.get("strict_one_sided"),
           f"a 6-up/2-down early window (sign balance {sbm.get('value')}) does NOT block, though it lies "
           "outside the band's recorded range -- the falsified 3-pair rule would have rejected it")
        ok("band_range_reported_only" in sbm,
           "the band's range is still REPORTED, so a reader can see what the old rule would have said")
        # Identical steps carry no direction. Counting them as "not up" is what made a leg identical
        # to its reference read as one-sided.
        lo_i, gn_i = synth(1, cross_at=11)
        for i in (1, 3, 5):
            lo_i[i] += 0.05                       # 3 differing steps, all up; 5 identical
        ident = make_leg(root, "cand_mostly_identical", lo_i, alt(gn_i))
        v_i = tg.verdict(legs[0], ident, b_tight)
        sbi = v_i["checks"].get("loss_sign_balance_early") or {}
        ok(sbi.get("blocking") is False and sbi.get("non_identical_steps") == 3,
           f"3 differing steps out of 8 are too few to call a direction ({sbi.get('non_identical_steps')} "
           "non-identical) -- identical steps are excluded, not counted as down")

        v_w = tg.verdict(legs[0], lower, b_tight, l0_strictly_better=True)
        ok(any("WAIVED" in a for a in v_w.get("advisories", [])),
           "a waiver is REPORTED, never silent")

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

        # ---------------------------------------------------------------- 7. snapshot v5 boundary
        # v5 captures --lr-decay-iters, which a v4 sidecar never recorded: the pair must be refused
        # as RE-MEASURE, not mislabelled "a different WORKLOAD". v3 vs v4 stays comparable.
        lo, gn = synth(1, cross_at=11)
        cfg4 = {"micro-batch-size": "1", "seq-length": "4096", "snapshot_keys": "v4"}
        cfg5 = dict(cfg4, **{"snapshot_keys": "v5", "lr-decay-iters": "358",
                             "ddp-average-in-collective": "true"})
        r4, c5 = make_leg(root, "ref_v4", lo, gn, cfg4), make_leg(root, "cand_v5", lo, gn, cfg5)
        v45 = tg.verdict(r4, c5, b5, candidate_kind="kernel_swap")
        ok(v45["pass"] is False and "RE-MEASURE" in v45["reason"] and "WORKLOAD" not in v45["reason"],
           f"a v4-vs-v5 pair is refused with RE-MEASURE, not 'different workload' ({v45['reason'][:60]})")
        r3 = make_leg(root, "ref_v3", lo, gn, dict(cfg4, snapshot_keys="v3"))
        v34 = tg.verdict(r3, make_leg(root, "cand_v4", lo, gn, cfg4), b5, candidate_kind="kernel_swap")
        ok(v34["pass"] is True, "a v3-vs-v4 pair is still comparable in L1 (every key v4 added is "
                                "implementation, which the identity check excludes)")
        # ddp-average-in-collective is IMPLEMENTATION: two v5 legs differing only in it are comparable
        c5b = make_leg(root, "cand_v5_noavg", lo, gn, {k: v for k, v in cfg5.items()
                                                       if k != "ddp-average-in-collective"})
        v55 = tg.verdict(c5b, c5, b5, candidate_kind="overlay")
        ok(v55["pass"] is True and "ddp-average-in-collective" in (v55.get("implementation_delta") or {}),
           "--ddp-average-in-collective is an implementation delta L1 can gate, not a workload change")

        # ---------------------------------------------------------------- 8. non-finite candidate
        lo, gn = synth(1, cross_at=11)
        nan_lo = [lo[0]] + [float("nan")] * (len(lo) - 1)
        nan_gn = [float("nan")] * len(gn)
        bad_nan = make_leg(root, "cand_nan", nan_lo, nan_gn)
        for kind in ("overlay", "kernel_swap"):
            vn = tg.verdict(legs[0], bad_nan, b5, candidate_kind=kind)
            ok(vn["pass"] is False and "non-finite" in vn["reason"],
               f"a NaN loss/grad_norm trajectory FAILS L1 ({kind}); every NaN comparison is False, so it used to pass")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
