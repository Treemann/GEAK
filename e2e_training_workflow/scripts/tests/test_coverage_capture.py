#!/usr/bin/env python3
"""Full argv/env capture and its classification (audit 2026-10-25 ID-1).

Every setting the audit reproduced as moving NEITHER digest must come out UNMAPPED (so the gate compares
it), every captured key MAPPED (the digests judge it), and the per-leg noise -- run length, logging, output
and cache paths, which GPUs -- IGNORED. Also checks train_probe._coverage's argparse semantics. No torch.
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
import probe_summarize as ps  # noqa: E402

fails = 0


def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m)
    fails += 0 if c else 1


# train_probe._coverage, executed from its source (the module imports torch).
tree = ast.parse(open(os.path.join(SCRIPTS, "train_probe.py")).read())
ns = {"os": os}
for n in tree.body:
    if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_COVERAGE_ENV_PREFIXES" for t in n.targets)) or \
            (isinstance(n, ast.FunctionDef) and n.name == "_coverage"):
        exec(compile(ast.Module([n], []), "train_probe.py", "exec"), ns)
saved = sys.argv, dict(os.environ)
try:
    sys.argv = ["pretrain_gpt.py", "--lr", "1e-5", "--bf16", "--lr=2e-5", "--data-path", "a", "b", "--tp-comm-overlap"]
    os.environ.update({"NCCL_ALGO": "Tree", "PYTORCH_TUNABLEOP_ENABLED": "1", "QKV_NEW_KNOB": "3", "UNRELATED": "x"})
    cov = ns["_coverage"]()
finally:
    sys.argv = saved[0]
    os.environ.clear()
    os.environ.update(saved[1])
ok(cov["argv"] == {"lr": "2e-5", "bf16": "true", "data-path": "a b", "tp-comm-overlap": "true"},
   f"argparse semantics: last wins, --k=v, nargs values joined, valueless = true ({cov['argv']})")
ok(cov["env"].get("NCCL_ALGO") == "Tree" and cov["env"].get("QKV_NEW_KNOB") == "3" and "UNRELATED" not in cov["env"],
   "env: kernel / collective families captured (incl. a NEW knob of an overlay family), others not")

AUDIT_ARGV = {"fp8-format": "hybrid", "tp-comm-overlap": "true", "use-precision-aware-optimizer": "true",
              "exp-avg-dtype": "bf16", "optimizer": "sgd", "no-bias-swiglu-fusion": "true"}
AUDIT_ENV = {"NVTE_CK_USES_BWD_V3": "0", "NVTE_CK_IS_V3_ATOMIC_FP32": "0", "PYTORCH_TUNABLEOP_ENABLED": "1",
             "NCCL_ALGO": "Tree"}
MAPPED_ARGV = {"lr": "1e-5", "attention-backend": "fused", "ddp-bucket-size": "2e8", "no-rope-fusion": "true",
               "manual-gc": "true", "seed": "1234", "hf-balance-tokens": "on"}
MAPPED_ENV = {"FC1_BWD_MODE": "serial", "NVTE_FUSED_ATTN": "1", "GEAK_OVERLAY": "/x/m.json",
              "GEAK_NAN_FROM_GRAD_NORM": "1"}
IGN_ARGV = {"train-iters": "30", "save": "/out", "eval-interval": "1000", "log-interval": "1", "output-dir": "/o",
            "tensorboard-dir": "/t", "profile": "true", "wandb-exp-name": "x"}
IGN_ENV = {"TRITON_CACHE_DIR": "/c", "TORCH_EXTENSIONS_DIR": "/e", "HIP_VISIBLE_DEVICES": "0,1", "GEAK_PROBE_DIR": "/p",
           "GEAK_IMAGE_ID": "sha256:x", "HF_HOME": "/h"}
u = ps.unmapped_of({"v": 1, "argv": {**AUDIT_ARGV, **MAPPED_ARGV, **IGN_ARGV}, "env": {**AUDIT_ENV, **MAPPED_ENV, **IGN_ENV}})
ok(set(u["argv"]) == set(AUDIT_ARGV), f"every audit argv example is UNMAPPED, nothing else ({sorted(u['argv'])})")
ok(set(u["env"]) == set(AUDIT_ENV), f"every audit env example is UNMAPPED, nothing else ({sorted(u['env'])})")
ok(ps.unmapped_of(None) is None and ps.unmapped_of({}) is None, "a leg without the capture -> None (predates it)")
ok(ps.unmapped_of({"v": 1, "argv": {"lr": "1.0e-5"}, "env": {}}) == {"argv": {}, "env": {}},
   "a captured key never appears as unmapped")
ok(ps.unmapped_of({"v": 1, "argv": {"adam-eps-x": "1e-5"}, "env": {}})["argv"] ==
   ps.unmapped_of({"v": 1, "argv": {"adam-eps-x": "1.0e-05"}, "env": {}})["argv"],
   "unmapped values are normalised like the realized table (1e-5 == 1.0e-05)")
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
