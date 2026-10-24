#!/usr/bin/env python3
"""The HF lm-head overlay must read `--hf-train-metrics` as argparse does: LAST occurrence wins (review HF-7,
DECISIONS 4.65). Reads _metrics_on() from the bundle stack_overlay_env.sh installs; skips if absent."""
import ast, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
env = open(os.path.join(HERE, "..", "stack_overlay_env.sh")).read()
m = re.search(r'^HF_LMCE_BUNDLE="\$WORK_HOST/(.+)"$', env, re.M)
work = os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train")
path = os.path.join(work, m.group(1), "overlays", "hf_lmhead_fused_ce", "impl.py") if m else ""
if not os.path.exists(path):
    print(f"SKIP: {path or 'HF_LMCE_BUNDLE'} not found"); sys.exit(0)
fn = [n for n in ast.parse(open(path).read()).body if isinstance(n, ast.FunctionDef) and n.name == "_metrics_on"][0]
ns = {"sys": sys}
exec(compile(ast.Module([fn], []), path, "exec"), ns)
fails = 0
for argv, want in (([], True), (["--hf-train-metrics", "off"], False),
                   (["--hf-train-metrics", "off", "--hf-train-metrics", "on"], True),
                   (["--hf-train-metrics", "on", "--hf-train-metrics=off"], False)):
    sys.argv = ["hf_sft.py"] + argv
    ok = ns["_metrics_on"]() == want
    fails += 0 if ok else 1
    print(("  ok: " if ok else "  FAIL: ") + f"{argv} -> metrics on = {want}")
print(f"\n{fails} FAILURE(S)" if fails else "\nall checks passed"); sys.exit(1 if fails else 0)
