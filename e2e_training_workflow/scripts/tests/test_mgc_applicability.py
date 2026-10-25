#!/usr/bin/env python3
"""main_grad_contract applicability must fail CLOSED on composite sites (audit 2026-10-25 G-1).

The real counterexample: mlp_fc1_swiglu's target `megatron.core.transformer.mlp:MLP.forward
[_LayerNormLinear (RMSNorm+fc1) -> bias_swiglu_impl]` matched no head substring, so the contract said
"not applicable" -- read as a pass -- while its impl.py runs fc1 with fuse_wgrad_accumulation into main_grad.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mgc_applicability import applicability, code_hits  # noqa: E402

FC1_SWIGLU = "megatron.core.transformer.mlp:MLP.forward [_LayerNormLinear (RMSNorm+fc1) -> bias_swiglu_impl]"
GEMM_CODE = "def f(w, g):\n    if w.fuse_wgrad_accumulation:\n        w.main_grad.add_(g)\n"
DOC_ONLY = '"""Replaces the two passes over the fp32 main_grad bucket."""\n# main_grad is scaled\nx = grad_data * s\n'
CASES = [
    ("mapped _LayerNormLinear -> its head", {"target_callable": "transformer_engine.pytorch.module.layernorm_linear:_LayerNormLinear"}, "",
     ("LNLinearHead", True)),
    ("mapped _Linear -> its head", {"target_callable": "transformer_engine.pytorch.module.linear:_Linear"}, "", ("LinearHead", True)),
    ("mapped lm head -> its head", {"target_callable": "megatron.core.models.gpt.gpt_model:GPTModel._postprocess [output_layer]"}, "",
     ("LMHead", True)),
    ("composite fc1+swiglu (target names _LayerNormLinear) -> applicable, NO head", {"target_callable": FC1_SWIGLU}, GEMM_CODE,
     (None, True)),
    ("composite whose target hides it but whose CODE accumulates into main_grad -> applicable, no head",
     {"target_callable": "megatron.core.transformer.mlp:MLP.forward"}, GEMM_CODE, (None, True)),
    ("main_grad only in a docstring / comment -> not applicable (grad tail)",
     {"target_callable": "megatron.core.distributed.param_and_grad_buffer:_ParamAndGradBucketGroup.start_grad_sync"},
     DOC_ONLY, (None, False)),
    ("attention -> not applicable", {"target_callable": "transformer_engine.pytorch.attention.dot_product_attention.backends:FusedAttnFunc"},
     "def f(q, k, v):\n    return q\n", (None, False)),
    ("unparsable impl falls toward applicable", {"target_callable": "x:y"}, "def f(:\n  main_grad\n", (None, True)),
]
bad = 0
for label, meta, src, (head, appl) in CASES:
    h, a, ev = applicability(meta, src)
    good = (h, a) == (head, appl)
    bad += not good
    print(f"  {'ok' if good else 'FAIL'}: {label}" + ("" if good else f" -> {(h, a, ev)}"))
good = code_hits("s = 'main_grad'\n") == ["main_grad"]
bad += not good
print(f"  {'ok' if good else 'FAIL'}: a string literal in code (getattr(p, 'main_grad')) counts")
if bad:
    print(f"{bad} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
