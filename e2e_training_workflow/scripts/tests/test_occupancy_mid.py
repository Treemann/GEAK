#!/usr/bin/env python3
"""In-leg occupancy samples (gpu_occupancy.mid_foreign): our ranks persist, a neighbour does not."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gpu_occupancy import compare, mid_foreign  # noqa: E402
fails = 0
def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1
own = list(range(100, 108))
clean = [{"pids": own}] * 5
ok(mid_foreign(clean, 8) == [], "8 persistent ranks of an 8-rank job are ours: clean")
ok(mid_foreign([{"pids": own[:3]}] + clean[1:], 8) == [], "ranks still starting in the first sample are not foreign")
ok(mid_foreign(clean[:2] + [{"pids": own + [999]}] + clean[3:], 8) == [999], "a PID that comes and goes is foreign")
ok(mid_foreign([{"pids": own + [99]}] * 5, 8) == [], "8 ranks + the launcher's KFD handle (9 persistent) is our own job")
ok(len(mid_foreign([{"pids": own + [99, 999]}] * 5, 8)) == 10, "a neighbour present the whole leg (10 > 8 + 1 persistent) is caught")
# Exact mode (own host PIDs recorded per sample): our JIT build children are not foreign; a neighbour is.
ex = [{"pids": own + [500, 501], "own_host_pids": own + [500, 501, 7]}, {"pids": own, "own_host_pids": own}]
ok(mid_foreign(ex, 8) == [], "exact: our own short-lived children (17 KFD processes at start-up) are not foreign")
ok(mid_foreign(ex + [{"pids": own + [999], "own_host_pids": own}], 8) == [999], "exact: a process not in our namespace is foreign")
B = {"available": True, "pids": []}
ok(compare(B, B, (), clean, 8)["verdict"] == "clean" and
   compare(B, B, (), clean[:2] + [{"pids": own + [999]}], 8)["verdict"] == "contaminated",
   "compare(): verdict uses the in-leg samples")
ok(compare(B, B, (), [], 8)["verdict"] == "clean", "no in-leg samples (short leg) -> before/after decide, as before")
av = {"available": True, "pids": []}
ok(compare(av, av, mid=[{"available": True, "pids": []}] * 3)["verdict"] == "clean", "available samples, nobody else: clean")
ok(compare(av, av, mid=[{"available": True, "pids": []}, {"available": False, "pids": []}])["verdict"] == "unknown",
   "a mid-leg sample taken while rocm-smi was unavailable is not evidence of clean (OPS-8)")
print(f"\n{fails} FAILURE(S)" if fails else "\nall checks passed"); sys.exit(1 if fails else 0)
