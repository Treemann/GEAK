"""Guards for the step decomposition — the arithmetic the whole routing model rests on.

The central claim being protected: a collective that is FULLY OVERLAPPED with compute contributes
ZERO to the critical path, so ranking by %GPU time (the inference layer's rule) systematically sends
the budget to the wrong kernel in training. If _subtract_ms ever stops meaning "exposed", the router
silently reverts to the inference heuristic and nothing else in the system would notice.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parse_train_profile import _subtract_ms, _union_ms, config_diagnosis  # noqa: E402

fails = []
def ok(cond, msg):
    print(("  ok: " if cond else "  FAIL: ") + msg)
    if not cond:
        fails.append(msg)

# ---- union: overlapping intervals are counted once (two streams busy at once is still one ms) ----
ok(_union_ms([(0, 1000)]) == 1.0, "a 1000us interval is 1.0 ms")
ok(_union_ms([(0, 1000), (500, 1500)]) == 1.5, "overlapping intervals are unioned, not summed")
ok(_union_ms([(0, 1000), (2000, 3000)]) == 2.0, "disjoint intervals add")
ok(_union_ms([]) == 0.0, "empty is zero")

# ---- subtract: THE doctrine. exposed = comm not hidden behind compute ----
comm = [(0, 1000)]
ok(_subtract_ms(comm, [(0, 1000)]) == 0.0,
   "a collective fully overlapped with compute is FULLY HIDDEN -> 0 exposed (worth optimizing: no)")
ok(_subtract_ms(comm, []) == 1.0, "a collective with no overlapping compute is fully exposed")
ok(_subtract_ms(comm, [(0, 500)]) == 0.5, "half-overlapped -> half exposed")
ok(_subtract_ms(comm, [(200, 400), (600, 800)]) == 0.6, "multiple compute windows punch multiple holes")
ok(_subtract_ms([(0, 1000), (2000, 3000)], [(0, 3000)]) == 0.0, "several collectives, all hidden")

# ---- the diagnosis: names the knob, and only BLOCKS on things that actually block ----
d = config_diagnosis({"exposed_comm_pct": 2.0, "idle_pct": 1.0}, {"pad_fraction_pct": 5}, 0.1)
ok(d["verdict"] == "kernel_work_can_pay", "a clean configuration lets the kernel track proceed")

d = config_diagnosis({"exposed_comm_pct": 62.0, "idle_pct": 1.0}, {"pad_fraction_pct": 5}, 0.1)
ok(d["verdict"] == "config_first" and "overlap_grad_reduce" in d["blocking"][0]["knob"],
   "62% exposed comm blocks the kernel track AND names overlap_grad_reduce")
ok("exposes MORE comm" in d["blocking"][0]["why"],
   "the reason explains the trap: a faster kernel exposes more comm, so the step barely moves")

d = config_diagnosis({"exposed_comm_pct": 1.0, "idle_pct": 30.0}, {"pad_fraction_pct": 5}, 0.1)
ok(d["verdict"] == "config_first" and "num_workers" in d["blocking"][0]["knob"],
   "30% idle DOES block: a host-bound step converts a shorter kernel into more idle, not into a "
   "shorter step")
ok("MORE idle" in d["blocking"][0]["why"], "and the reason names that mechanism")

d = config_diagnosis({"exposed_comm_pct": 1.0, "idle_pct": 1.0}, {"pad_fraction_pct": 71.6}, 0.1)
ok(d["verdict"] == "kernel_work_can_pay",
   "71.6% padding does NOT block: the headline is useful_tokens/step_time, and a kernel win shortens "
   "step_time while leaving useful_tokens alone, so the relative gain is independent of padding "
   "(a 20% win on a 19% head is +3.95% at 70% padding and at 0% padding alike)")
ok(d["opportunity"] and "packing" in d["opportunity"][0]["knob"],
   "it is reported as an OPPORTUNITY instead: packing raises useful_tokens directly, so the kernel "
   "budget carries a high opportunity cost — a priority argument, not an impossibility argument")
ok("not a blocker" in d["opportunity"][0]["why"],
   "and the reason says so explicitly, because the earlier text claimed the opposite")

d = config_diagnosis({"exposed_comm_pct": 1.0, "idle_pct": 1.0}, {"pad_fraction_pct": 5}, 12.0)
ok(d["verdict"] == "config_first" and "skew" in d["blocking"][0]["finding"],
   "a 12% straggler caps any kernel win and blocks")

ok(all("READ-ONLY" in config_diagnosis({"exposed_comm_pct": x, "idle_pct": 0}, {}, 0)["note"]
       for x in (1, 50)), "the diagnosis always declares itself read-only (it never treats)")

print(f"\n{len(fails)} FAILURE(S)" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
