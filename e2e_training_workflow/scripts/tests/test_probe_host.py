#!/usr/bin/env python3
"""The probe's per-step host diagnostics (report-only, DECISIONS 4.41) must be per-step DELTAS.

Step time is synchronized across ranks, so only per-rank host facts can say which rank stalled and
why: Python GC time per generation, caching-allocator retries / segment (de)allocations, host RSS.
Read the two methods out of train_probe.py without importing it (it imports torch at module scope)
and drive them with a stub torch.
"""
import ast, gc, os, sys, time, types

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(os.path.dirname(HERE), "train_probe.py")).read()
cls = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "_Probe"][0]
stats = {"num_alloc_retries": 3, "num_device_alloc": 10, "num_device_free": 4}
torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True,
                                                         memory_stats=lambda: dict(stats)))
ns = {"time": time, "os": os, "torch": torch}
P = type("P", (object,), {})
for f in cls.body:
    if isinstance(f, ast.FunctionDef) and f.name in ("_host_init", "_host_record"):
        exec(compile(ast.Module(body=[f], type_ignores=[]), "train_probe.py", "exec"), ns)
        setattr(P, f.name, ns[f.name])

fails = 0
def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1

p = P(); p._host_init()
h1 = p._host_record()
ok(all(h1[k] == 0 for k in ("num_alloc_retries", "num_device_alloc", "num_device_free")),
   f"the first step reports zero allocator deltas, not cumulative counts ({h1})")
stats.update(num_alloc_retries=5, num_device_alloc=12)
x = [[i] for i in range(10000)]
[a.append(a) for a in x]
del x
gc.collect()
h2 = p._host_record()
ok(h2["num_alloc_retries"] == 2 and h2["num_device_alloc"] == 2 and h2["num_device_free"] == 0,
   f"later steps report per-step deltas ({h2})")
ok(h2["gc_n"][2] >= 1 and h2["gc_ms"] > 0, "a full collection is timed and counted as generation 2")
ok("rss_mb" in h2, "host RSS is recorded (the --manual-gc-interval 0 memory check reads it)")
h3 = p._host_record()
ok(h3["gc_ms"] == 0 and h3["gc_n"] == [0, 0, 0], "GC counters reset every step")
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
