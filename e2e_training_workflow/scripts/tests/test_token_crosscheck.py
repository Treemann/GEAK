#!/usr/bin/env python3
"""HF token cross-check (audit HFC-7): the probe's useful tokens + first-position labels must equal the trainer's
own num_items_in_batch per step (summed over ranks under global normalization). Synthetic ranks; no torch."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from probe_summarize import _token_crosscheck  # noqa: E402
fails = 0


def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1


def ranks(per_rank, stack, first=8):
    return {r: ({}, [{"useful_tokens": u, "first_labels": first, "stack_items": stack} for u in us])
            for r, us in enumerate(per_rank)}


good = ranks([[100, 110], [90, 95]], None)
for r in good.values():
    for i, s in enumerate(r[1]):
        s["stack_items"] = [100 + 90 + 16, 110 + 95 + 16][i]
v = _token_crosscheck(good, {"hf-average-tokens-across-devices": "true"})
ok(v["steps_checked"] == 2 and v["mismatched"] == 0, f"global: probe + first labels == trainer count ({v})")
bad = ranks([[100, 110], [90, 95]], 999)
ok(_token_crosscheck(bad, {"hf-average-tokens-across-devices": "true"})["mismatched"] == 2, "a disagreeing step is reported")
per = {0: ({}, [{"useful_tokens": 100, "first_labels": 8, "stack_items": 108}]),
       1: ({}, [{"useful_tokens": 90, "first_labels": 8, "stack_items": 98}])}
ok(_token_crosscheck(per, {"hf-average-tokens-across-devices": "false"})["mismatched"] == 0, "per-rank normalization: per-rank check")
ok(_token_crosscheck({0: ({}, [{"useful_tokens": 5}])}, {}) is None, "no trainer count (Megatron) -> None")
print("all checks passed" if not fails else f"{fails} check(s) FAILED"); sys.exit(1 if fails else 0)
