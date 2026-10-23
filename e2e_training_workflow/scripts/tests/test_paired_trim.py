#!/usr/bin/env python3
"""The paired band must not be set by stalls, and the paired DELTA must not be trimmed (DECISIONS 4.54).

One or two 300-1000 ms stalls per leg land on a quartile of the per-step ratio and inflate its IQR
3-10x; on 525 Megatron bare true nulls that left 16% `inconclusive_noise`. compare_legs now drops a
step from the DISPERSION when either leg ran it at > 1.04x its own median -- fixed step shape only --
and keeps the delta as the median over every step (a trimmed-median delta read 14% on a null pair).
Also pins calibrate_band's injection: scaling the cand by (1+p) moves only the deltas.
"""
import json, os, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_legs import compare, _quantile  # noqa: E402
import calibrate_band  # noqa: E402

fails = 0


def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m)
    fails += 0 if c else 1


def leg(root, name, durs, tokens=None):
    d = os.path.join(root, name, "probe"); os.makedirs(d)
    for rk in range(2):
        with open(os.path.join(d, f"probe_rank{rk}.jsonl"), "w") as f:
            f.write(json.dumps({"record": "header", "rank": rk, "world_size": 2, "argv_hash": "x",
                                "config": {"snapshot_keys": "v5"}}) + "\n")
            for i, t in enumerate(durs):
                tt = tokens[i] if tokens else 8192
                f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": t, "useful_tokens": tt - 100,
                                    "total_tokens": tt, "peak_reserved_mb": 1}) + "\n")
    return d


root = tempfile.mkdtemp(prefix="trim_")
try:
    ok(_quantile([1, 2, 3, 4], 0.25) == 1.75 and _quantile([1, 2, 3, 4, 5], 0.5) == 3,
       "quartiles are linearly interpolated, not an index pick")
    base = [40000] + [2300.0 + (i % 5) for i in range(39)]
    a_d = list(base); a_d[12] += 900; a_d[30] += 700
    b_d = [x * 1.001 for x in base]
    for i in (3, 6, 9, 15, 18, 20, 25, 33, 36, 38, 21):     # 11 stalls: > a quarter of the window
        b_d[i] += 500 + 30 * i
    c = compare(leg(root, "a", a_d), leg(root, "b", b_d))
    p = c["paired"]
    ok(p["steps_dropped"] >= 10 and p["steps"] == p["steps_total"] - p["steps_dropped"],
       f"the stalled steps in the window leave the dispersion ({p['steps_dropped']} dropped of {p['steps_total']})")
    ok(p["ratio_iqr_pct"] < p["ratio_iqr_raw_pct"],
       f"trimmed IQR {p['ratio_iqr_pct']}% < raw {p['ratio_iqr_raw_pct']}%")
    ok(abs(p["delta_pct"] - (100 / 1.001 - 100)) < 0.02,
       f"the delta is still the untrimmed paired median ({p['delta_pct']}%)")

    # Too many slow steps: they are the leg's behaviour, not stalls -> fall back to every step.
    many = list(base); many[5:25] = [x * 1.08 for x in many[5:25]]
    p2 = compare(leg(root, "c", base), leg(root, "d", many))["paired"]
    ok(p2["steps_dropped"] == 0 and p2["steps"] == p2["steps_total"],
       "a trim that would remove more than half the window falls back to every step")

    # Variable shape (dynamic padding): duration follows content, so no duration trim at all.
    tok = [4096 + (i * 997) % 4000 for i in range(40)]
    vd = [40000] + [0.3 * t + 50 for t in tok[1:]]
    p3 = compare(leg(root, "e", vd, tok), leg(root, "f", vd, tok))["paired"]
    ok(p3["steps_dropped"] == 0, "variable step shape is never duration-trimmed")

    # calibrate_band's power injection: +1% faster cand moves only the deltas and the cand rate.
    spec = {"comparison": {"delta_pct": 0.1, "unaligned_delta_pct": -0.2,
                           "paired": {"delta_pct": 0.0, "ratio_iqr_pct": 0.3, "steps": 30}},
            "cand": {"useful_tok_s_per_gpu": 1000.0}, "ref": {"useful_tok_s_per_gpu": 999.0}}
    s = calibrate_band._inject(spec, 1.0)
    ok(abs(s["comparison"]["paired"]["delta_pct"] - 1.0) < 1e-9 and
       abs(s["comparison"]["delta_pct"] - 1.101) < 1e-9 and s["cand"]["useful_tok_s_per_gpu"] == 1010.0,
       "injection compounds the effect onto every delta")
    ok(s["comparison"]["paired"]["ratio_iqr_pct"] == 0.3 and spec["comparison"]["paired"]["delta_pct"] == 0.0,
       "...leaves the dispersion alone and does not mutate the original spec")
finally:
    shutil.rmtree(root)
print("\nall checks passed" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
