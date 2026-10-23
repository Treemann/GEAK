"""Guards the warmup detector (no GPU).

Fixed-shape legs (Megatron, padded or packed: total_tokens constant) keep the original rule. Variable-
shape legs (HF dynamic padding) use time per token with a robust tolerance (DECISIONS 4.33): the
fixed rule pinned warmup at step 26 of 30 on the 7B HF ZeRO-2 stack, which had settled by step 3.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from probe_summarize import detect_warmup_end, detect_warmup_end_variable  # noqa: E402

fails = []


def ok(c, m):
    print(("  ok: " if c else "  FAIL: ") + m)
    if not c:
        fails.append(m)


# runs/baseline_hfnm_1017_231802_leg1, rank 0: step ms and total tokens (verbatim).
HF_MS = [31509, 10080, 6116, 4678, 4834, 4711, 5051, 4435, 4587, 4529, 4518, 5165, 4733, 4408, 4858,
         4299, 4355, 5038, 4372, 4065, 4524, 4382, 4608, 4513, 4974, 5047, 4314, 4015, 4216, 3853]
HF_TOK = [10558, 7607, 16491, 9493, 8933, 10588, 9961, 9788, 7247, 12122, 7980, 13289, 10821, 11415,
          13587, 8700, 10859, 11111, 11297, 8815, 9318, 10543, 11373, 12135, 8428, 7469, 9207, 9562,
          10283, 7714]
old = detect_warmup_end(HF_MS)
new = detect_warmup_end_variable(HF_MS, HF_TOK)
ok(old >= 20, f"the fixed rule mis-detects the HF leg (warmup {old}): the defect this guards against")
ok(2 <= new <= 5, f"the variable rule finds the real settling point (warmup {new}, run settled by step 3)")
ok(len(HF_MS) - new >= 12, f"leaving >= 12 steady steps for dispersion ({len(HF_MS) - new})")

# A fixed-shape Megatron-like series: the variable rule is not used there, and the fixed rule is unchanged.
MEG = [40759, 2796, 2819, 2822, 2817, 2821, 2780, 2809, 2807, 2808, 2820] + [2810] * 19
ok(detect_warmup_end(MEG) == 1, "fixed rule unchanged on a Megatron-like series (warmup 1)")
# A genuinely slow settling tail must still be detected as warmup by the variable rule.
SLOW = [9000, 8000, 7000, 6500, 6000] + [4500] * 25
ok(detect_warmup_end_variable(SLOW, [10000] * 30) == 5, "a real monotone settling phase is still warmup")
# Zero token counts fall back to the fixed rule rather than dividing by zero.
ok(detect_warmup_end_variable(MEG, [0] * 30) == detect_warmup_end(MEG), "zero tokens fall back to the fixed rule")

# The headline window floor (DECISIONS 4.57): summarize() never counts the first MIN_WARMUP_SKIP steps
# of a long-enough leg, and keeps the detector's boundary on a short one.
import json, tempfile, shutil  # noqa: E402
from probe_summarize import summarize, MIN_WARMUP_SKIP, MIN_STEADY_FOR_DISPERSION  # noqa: E402
root = tempfile.mkdtemp(prefix="wfloor_")
try:
    def leg(name, durs):
        d = os.path.join(root, name); os.makedirs(d)
        with open(os.path.join(d, "probe_rank0.jsonl"), "w") as f:
            f.write(json.dumps({"record": "header", "rank": 0, "world_size": 1, "argv_hash": "x", "config": {}}) + "\n")
            for i, t in enumerate(durs):
                f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": t, "useful_tokens": 8000,
                                    "total_tokens": 8192, "peak_reserved_mb": 1}) + "\n")
        return summarize(d)
    s30 = leg("m30", MEG)
    ok(MIN_WARMUP_SKIP == 8 and s30["warmup_steps_max"] == MIN_WARMUP_SKIP and "floored" in s30["per_rank"][0]["warmup_rule"],
       f"30-step leg detected at 1 is floored to {MIN_WARMUP_SKIP} ({s30['warmup_steps_max']})")
    ok(s30["per_rank"][0]["steady_steps"] == 30 - MIN_WARMUP_SKIP, "...leaving 22 steady steps")
    short = MEG[:MIN_WARMUP_SKIP + MIN_STEADY_FOR_DISPERSION - 1]
    ok(leg("short", short)["warmup_steps_max"] == 1, "a leg too short to keep 12 steady steps keeps the detector's boundary")
    late = [40000] + [5000] * 11 + [2810] * 18
    ok(leg("late", late)["warmup_steps_max"] == detect_warmup_end(late) > MIN_WARMUP_SKIP,
       "a detector boundary later than the floor is kept")
finally:
    shutil.rmtree(root)

print(f"\n{len(fails)} FAILURE(S)" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
