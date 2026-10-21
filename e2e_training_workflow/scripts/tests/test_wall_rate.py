#!/usr/bin/env python3
"""The wall-time rate must see the host stalls the median headline hides (DECISIONS 4.41).

Synchronized ~300-1000 ms stalls on 1 step in 10-20 cost a real run 1-2% of wall time while the
median step does not move at all, so a lever that removes them (synchronized GC) measured +0.07% on
the headline and +0.8% on wall time. This pins: identical median, lower wall rate, positive stall_pct.
"""
import json, os, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from probe_summarize import summarize  # noqa: E402

fails = 0


def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m)
    fails += 0 if c else 1


def leg(root, name, durs):
    d = os.path.join(root, name, "probe"); os.makedirs(d)
    for rk in range(2):
        with open(os.path.join(d, f"probe_rank{rk}.jsonl"), "w") as f:
            f.write(json.dumps({"record": "header", "rank": rk, "world_size": 2, "argv_hash": "x",
                                "config": {"snapshot_keys": "v5"}}) + "\n")
            for i, t in enumerate(durs):
                f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": t, "useful_tokens": 8000,
                                    "total_tokens": 8192, "peak_reserved_mb": 1}) + "\n")
    return d


root = tempfile.mkdtemp(prefix="wall_")
try:
    base = [40000] + [2300.0 + (i % 3) for i in range(29)]
    stalled = list(base); stalled[12] += 900; stalled[24] += 600
    a, b = summarize(leg(root, "clean", base)), summarize(leg(root, "stalled", stalled))
    ok(a["useful_tok_s_per_gpu"] == b["useful_tok_s_per_gpu"], "two stalls leave the median headline unchanged")
    ok(b["wall_useful_tok_s_per_gpu"] < a["wall_useful_tok_s_per_gpu"], "the wall-time rate sees them")
    ok(b["stall_pct"] > a["stall_pct"] and b["stall_pct"] > 1.5,
       f"stall_pct reports the gap ({a['stall_pct']} -> {b['stall_pct']})")
    ok(a["config_digest"] == b["config_digest"], "report-only fields do not enter any digest")
    # Variable-shape legs: mean - median is data skew there, so stall_pct must not be reported.
    d = os.path.join(root, "var2", "probe"); os.makedirs(d)
    with open(os.path.join(d, "probe_rank0.jsonl"), "w") as f:
        f.write(json.dumps({"record": "header", "rank": 0, "world_size": 1, "argv_hash": "x", "config": {}}) + "\n")
        for i, t in enumerate(base):
            tok = 4000 + (i % 5) * 1500
            f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": t * tok / 7000, "useful_tokens": tok,
                                "total_tokens": tok, "peak_reserved_mb": 1}) + "\n")
    v = summarize(os.path.join(root, "var2", "probe"))
    ok(v["step_shape"] == "variable" and v["stall_pct"] is None,
       f"stall_pct is None on a variable-shape leg (shape={v['step_shape']}, stall_pct={v['stall_pct']})")
finally:
    shutil.rmtree(root, ignore_errors=True)
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
