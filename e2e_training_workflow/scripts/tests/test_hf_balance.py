#!/usr/bin/env python3
"""Token-balanced rank assignment (DECISIONS 4.61): the step's sample SET is unchanged, every rank gets
one micro-batch per slot, and the per-rank cost spread shrinks. accelerate shards the global
micro-batch stream as batch b -> rank b % world, so position (slot j, rank r, m) is (j*world + r)*mbs + m.
"""
import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hf_sft", os.path.join(HERE, "..", "adapters", "hf_sft.py"))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)

fails = 0


def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m)
    fails += 0 if c else 1


random.seed(7)
L = [int(random.lognormvariate(6.6, 0.7)) % 4096 + 50 for _ in range(5000)]
cost = lambda i: L[i] * (1 + L[i] / 3800.0)  # noqa: E731
for world, mbs, ga in ((8, 1, 8), (8, 2, 4), (4, 1, 3)):
    worst_def = worst_bal = 0.0
    same_set = one_per_slot = True
    for _ in range(100):
        idx = random.sample(range(len(L)), world * mbs * ga)
        out = hf.balanced_window(idx, L, world, mbs, ga)
        same_set &= sorted(out) == sorted(idx) and None not in out
        tot = lambda order: [sum(cost(order[(j * world + r) * mbs + m]) for j in range(ga) for m in range(mbs))  # noqa: E731
                             for r in range(world)]
        td, tb = tot(idx), tot(out)
        # No assignment can beat the larger of the mean load and the single costliest sample
        # (lengths are heavy-tailed and attention is quadratic: one sample can outweigh a rank's share).
        lb = max(sum(td) / world, max(cost(i) for i in idx))
        worst_def = max(worst_def, max(td) / lb)
        worst_bal = max(worst_bal, max(tb) / lb)
    ok(same_set, f"world {world} mbs {mbs} ga {ga}: the window's sample SET is unchanged (a permutation)")
    ok(worst_bal <= worst_def and worst_bal < 1.25,
       f"world {world} mbs {mbs} ga {ga}: worst rank total / achievable lower bound {worst_def:.2f} -> {worst_bal:.2f}")

# A trailing partial window keeps the default order (the sampler yields it untouched).
class _S(hf.TokenBalancedSampler):
    def __init__(self, order, lengths, world, mbs, ga):
        self.base, self.lengths, self.world, self.mbs, self.ga = order, lengths, world, mbs, ga
order = list(range(0, 8 * 8 + 5))
got = list(_S(order, L, 8, 1, 8))
ok(sorted(got[:64]) == list(range(64)) and got[64:] == list(range(64, 69)),
   "the full window is permuted within itself; the trailing partial window keeps its order")
print(f"\n{fails} FAILURE(S)" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
