"""mgc_applicability.py -- does main_grad_contract apply to a task + overlay, and which head models it?

Torch-free so it is unit-testable on the host (main_grad_contract.py imports it).

WHY (audit 2026-10-25 G-1). Applicability was a substring match on the task's rebind target: exactly
`layernorm_linear:_LayerNormLinear`, `module.linear:_Linear` or `GPTModel._postprocess`. A COMPOSITE
overlay -- mlp_fc1_swiglu's target is `megatron.core.transformer.mlp:MLP.forward [_LayerNormLinear
(RMSNorm+fc1) -> bias_swiglu_impl]`, and its impl.py runs the fc1 GEMM with fuse_wgrad_accumulation into
main_grad -- matched none of them and came back "not applicable", which the workflow reads as a pass.
Now: a mapped target gets its head; an UNMAPPED target whose target text names a fused-wgrad GEMM module,
or whose overlay CODE (comments and docstrings excluded) touches main_grad / the accumulation-fusion
flags, is APPLICABLE WITH NO HEAD -- reported as not passed until a head models that site. Only a site
with no such evidence is not applicable.
"""
from __future__ import annotations

import io
import re
import tokenize

HEADS = (("layernorm_linear:_LayerNormLinear", "LNLinearHead"),
         ("module.linear:_Linear", "LinearHead"),
         ("GPTModel._postprocess", "LMHead"))
TARGET_HINT = re.compile(r"_LayerNormLinear|\b_Linear\b|LayerNormLinear|output_layer|main_grad|"
                         r"gradient_accumulation_fusion|fuse_wgrad")
CODE_HINT = re.compile(r"\b(main_grad|fuse_wgrad_accumulation|gradient_accumulation_fusion)\b")


def code_hits(src):
    """Hint words in the CODE of an impl.py: names and string literals, not comments or docstrings
    (a docstring that says 'the fp32 main_grad bucket' is not code that accumulates into it)."""
    hits, prev = set(), tokenize.NEWLINE
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL):
                continue
            stmt_start = prev in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING)
            if tok.type in (tokenize.NAME, tokenize.STRING) and not (tok.type == tokenize.STRING and stmt_start):
                hits.update(CODE_HINT.findall(tok.string))
            prev = tok.type
    except (tokenize.TokenError, IndentationError, SyntaxError):
        hits.update(CODE_HINT.findall(src))   # unparsable: fail toward applicable
    return sorted(hits)


def applicability(meta, impl_src):
    """-> (head name or None, applicable: bool, evidence: list[str])"""
    t = str(meta.get("target_callable") or meta.get("target") or "")
    for sub, head in HEADS:
        if sub in t:
            return head, True, [f"target: {sub}"]
    ev = [f"target: {m}" for m in sorted(set(TARGET_HINT.findall(t)))] + \
         [f"impl code: {m}" for m in code_hits(impl_src or "")]
    return None, bool(ev), ev


if __name__ == "__main__":
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser(description="main_grad_contract applicability (torch-free, host)")
    ap.add_argument("--task", required=True, help="L0 task dir (meta.json)")
    ap.add_argument("--impl", required=True, help="overlay impl.py, or the overlay dir holding it")
    a = ap.parse_args()
    impl = os.path.join(a.impl, "impl.py") if os.path.isdir(a.impl) else a.impl
    meta = json.load(open(os.path.join(a.task, "meta.json")))
    head, appl, ev = applicability(meta, open(impl, errors="replace").read())
    print(json.dumps({"task": a.task, "impl": impl, "head": head, "applicable": appl, "evidence": ev}))
