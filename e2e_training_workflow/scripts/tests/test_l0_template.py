#!/usr/bin/env python3
"""The generated L0 unittest must score EVERY output (audit 2026-10-22).

make_call drops None gradients, so a candidate that omitted a trailing gradient returned a shorter
tuple; zip() truncated it and that label was never compared -- L0 reported correct=true. 28 of 34
template-built task oracles had the hole. Running the template needs torch + a capture inside the
training image, so this test pins the guard statically and checks the rendered file compiles.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(os.path.dirname(HERE), "build_task_dir.py")).read()
i = src.index("UNITTEST = r'''") + len("UNITTEST = r'''")
tmpl = src[i:src.index("'''", i)].replace("__REFERENCE__", "rmsnorm_linear")
fails = 0
def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1

compile(tmpl, "unittest.py", "exec")
ok(True, "the rendered template compiles")
l0 = tmpl[tmpl.index("    def _l0("):]
first_zip = l0.index("zip(_labels, got)")
ok(re.search(r"if len\(got\) != len\(_labels\) or len\(ref_outs\) != len\(_labels\):", l0) is not None
   and l0.index("len(got) != len(_labels)") < first_zip,
   "output COUNT is checked against the labels before any per-label scoring")
ok("_unscored = [l for l in base if l not in _labels" in l0, "every label with a base error must be scored")
ok("actual is None or (r is not None and tuple(actual.shape) != tuple(r.shape))" in l0,
   "a missing or mis-shaped candidate output fails")
ok("zip(META[\"signature\"][\"outputs\"], got)" not in l0, "no remaining zip over the raw outputs list")
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
