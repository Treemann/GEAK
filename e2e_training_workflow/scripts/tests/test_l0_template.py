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
# ulp1_elementwise (DECISIONS 4.52): static pins + a functional run of the helper in the training image
# (torch is not on the host; skipped when docker or the image is unavailable).
ok("def ulp1_forgive(" in tmpl and "def bf16_ulp(" in tmpl, "the per-element 1-ulp helpers are in the template")
bt = open(os.path.join(os.path.dirname(HERE), "build_task_dir.py")).read()
ok('"ulp1_elementwise": {"enabled": True, "max_frac": 0.001}' in bt, "new task dirs enable it by default")
ok('if _u1.get("enabled") and not mag_ok and label not in acc.get("ulp_labels", [])' in l0,
   "it only rescues the MAGNITUDE check (the bias checks are untouched)")
import shutil, subprocess, tempfile
if shutil.which("docker"):
    code = r"""
import ast, math, torch
t = open('/t.py').read()
ns = {"torch": torch, "math": math}
for n in ast.parse(t).body:
    if isinstance(n, ast.FunctionDef) and n.name in ("bf16_ulp", "ulp1_forgive"):
        exec(compile(ast.Module(body=[n], type_ignores=[]), "t", "exec"), ns)
f, ulp = ns["ulp1_forgive"], ns["bf16_ulp"]
torch.manual_seed(0); ref = torch.randn(100000); k = 777
one = torch.zeros_like(ref); one[k] = ulp(ref)[k]
two = torch.zeros_like(ref); two[k] = 2 * ulp(ref)[k]
allr = ref.to(torch.bfloat16).float() - ref
print(int(f(one, ref, 0.0)[0]), int(f(two, ref, 0.0)[0]), int(f(allr, ref, 0.0)[0]))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(tmpl); tpath = tf.name
    r = subprocess.run(["docker", "run", "--rm", "--name", f"geak_l0tmpl_{os.getpid()}", "--label", "geak=1",
                        "-v", f"{tpath}:/t.py:ro", "--entrypoint", "python3", "rocm/primus:v26.4", "-B", "-c", code],
                       capture_output=True, text=True, timeout=600)
    os.unlink(tpath)
    if r.returncode == 0 and r.stdout.strip():
        a, b, c = r.stdout.split()[-3:]
        ok((a, b, c) == ("1", "0", "0"),
           f"functional: 1 elem @1ulp forgiven, 1 elem @2ulp not, 100% of elems over the limit not ({a}{b}{c})")
    else:
        print("  SKIP functional check (container unavailable):", (r.stderr or "")[-200:])
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
