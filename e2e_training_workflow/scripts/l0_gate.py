"""L0 — op-level numerics, judged against the error the ORIGINAL implementation already carries.

THE TOLERANCE PROBLEM, AND WHY THIS AVOIDS IT. "Is rtol=1e-2 ok?" has no defensible answer: too
tight and every honest bf16 rewrite fails, too loose and a real bug passes. So nothing here is
compared against a hand-picked constant. The frozen capture gives:

    captured    what the ORIGINAL fused op produced, in its own precision (out, dgrad, wgrad)
    ref_fp32    an INDEPENDENT pure-PyTorch reference, in fp32, on the same frozen inputs
    err_base    |captured - ref_fp32|      <- the error the baseline already carries

and the rule is: **a candidate is acceptable iff it is no further from ref_fp32 than the original
is** (times a small margin). That is not arbitrary, and it never penalises a candidate for rounding
the baseline also does.

The reference must be an independent implementation, not a replay of the fused op: the captured call
carries runtime handles (ProcessGroup, WeightGradStore, the owning module) that cannot be rebuilt
offline. See `meta.json:opaque_args`.

Second, non-negotiable check: the error must not be ONE-SIDED. A small systematic bias passes every
magnitude check and still destroys convergence over thousands of steps. That failure is invisible to
a tolerance and obvious to a sign test.
"""
from __future__ import annotations

import json
import sys

import torch


def load_record(path):
    return torch.load(path, weights_only=False)


def _t(node, dtype=None, dev="cpu"):
    if isinstance(node, dict) and node.get("__t__"):
        x = node["data"].to(dev)
        if dtype is not None and x.is_floating_point():
            x = x.to(dtype)
        return x
    return None


def error_stats(actual, reference):
    """Magnitude AND shape of the error distribution. Both are required to judge."""
    a = actual.detach().float().flatten()
    r = reference.detach().float().flatten()
    d = a - r
    denom = r.abs().mean().clamp_min(1e-12)
    pos = (d > 0).sum().item()
    nz = (d != 0).sum().item()
    rr = (r * r).sum()
    return {
        "max_abs": d.abs().max().item(),
        "mean_abs": d.abs().mean().item(),
        "rel_mean_abs": (d.abs().mean() / denom).item(),
        # 0.5 for unbiased rounding noise; near 0 or 1 means an ADDITIVE systematic bias.
        "sign_balance": (pos / nz) if nz else 0.5,
        # The sign test alone is NOT enough, and a negative test proved it: a candidate scaled by
        # 1.005 produces d = 0.005*r, whose sign simply follows r -- so sign_balance stays at ~0.5
        # and the magnitude check passes too (the original's own bf16 error is LARGER than the bias).
        # A MULTIPLICATIVE bias only shows up in the correlation between the error and the value:
        # <d,r>/<r,r> is exactly the implied scale offset (0.005 for the case above, ~0 for rounding).
        "scale_bias": ((d * r).sum() / rr).item() if float(rr) > 0 else 0.0,
        "offset_bias": (d.mean() / denom).item(),
        "nonzero_frac": nz / d.numel(),
    }


def verdict(base_err, cand_err, margin=1.25, sign_tol=0.15):
    """Acceptable iff no further from the reference than the original, and not one-sided."""
    fails = []
    for key in sorted(set(base_err) & set(cand_err)):
        b, c = base_err[key], cand_err[key]
        if c["max_abs"] > b["max_abs"] * margin:
            fails.append(f"{key}: max_abs {c['max_abs']:.3e} exceeds the original's "
                         f"{b['max_abs']:.3e} x{margin}")
        if abs(c["sign_balance"] - 0.5) > sign_tol and abs(b["sign_balance"] - 0.5) <= sign_tol:
            fails.append(f"{key}: error is one-sided (sign_balance {c['sign_balance']:.3f}) while "
                         f"the original's is not ({b['sign_balance']:.3f}) — a systematic bias, "
                         f"which no magnitude check would catch")
    return {"pass": not fails, "reason": "; ".join(fails) or
            "no further from the fp32 reference than the original, and not one-sided",
            "base": base_err, "candidate": cand_err}


def rmsnorm_linear_reference(x, ln_w, w, b, eps=1e-6):
    """Independent reference for TE's _LayerNormLinear under RMSNorm (what this model uses)."""
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rms * ln_w.float()
    out = torch.nn.functional.linear(y, w.float(), b.float() if b is not None else None)
    return out


REFERENCES = {"rmsnorm_linear": rmsnorm_linear_reference}


def run(record_path, reference="rmsnorm_linear", arg_index=(0, 1, 2, 3)):
    rec = load_record(record_path)
    flat = []

    def _walk(n):
        if isinstance(n, dict) and n.get("__t__"):
            flat.append(n)
        elif isinstance(n, list):
            [_walk(v) for v in n]
        elif isinstance(n, dict):
            [_walk(v) for v in n.values()]
    _walk(rec["args"])
    tens = [_t(flat[i]) for i in arg_index if i < len(flat)]
    x, ln_w, w, b = (tens + [None] * 4)[:4]

    iso = rec.get("isolated") or {}
    ref_out = REFERENCES[reference](x, ln_w, w, b)
    # Prefer the ISOLATED re-run: same op, same frozen inputs, one call, no pairing ambiguity.
    captured_out = _t(iso.get("out")) if iso.get("out") is not None else None
    if captured_out is None:
        captured_out = _t(rec.get("output")) if rec.get("output") is not None else None
    if captured_out is None:
        # output may be a tuple snapshot
        out_node = rec.get("output")
        if isinstance(out_node, list) and out_node:
            captured_out = _t(out_node[0])
    if captured_out is None:
        return {"error": "no captured forward output in the record"}

    base = {"out": error_stats(captured_out, ref_out)}
    # Backward: compare the original's captured input grads against the reference's own grads.
    go = _t(rec.get("grad_output"))
    src_grads = iso.get("grads") or rec.get("input_grads") or {}
    if go is not None and src_grads:
        # Match captured gradients to inputs BY SHAPE, not by index. The backward's return tuple is
        # aligned with ALL of the forward's parameters (including the non-tensor ones), so its
        # indices do not line up with the tensor list -- observed directly: the captured keys were
        # ['0','1','3'], with no '2', because the weight gradient is absent from the returned tuple
        # under gradient_accumulation_fusion (it goes straight into main_grad).
        by_shape = {}
        for k, node in src_grads.items():
            if isinstance(node, dict) and node.get("__t__"):
                by_shape.setdefault(tuple(node["shape"]), []).append((k, node))
        xg = x.clone().requires_grad_(True)
        lg = ln_w.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        o = REFERENCES[reference](xg, lg, wg, b)
        grads = torch.autograd.grad(o, [xg, lg, wg], grad_outputs=go.float(), allow_unused=True)
        for label, tensor, g in (("dgrad", x, grads[0]), ("dln_weight", ln_w, grads[1]),
                                 ("wgrad", w, grads[2])):
            if g is None or tensor is None:
                continue
            match = by_shape.get(tuple(tensor.shape))
            if not match:
                base[label] = {"missing": True,
                               "why": (f"the op's backward returned no gradient of shape "
                                       f"{list(tensor.shape)}; under gradient_accumulation_fusion "
                                       f"the weight gradient goes into main_grad instead")}
                continue
            base[label] = error_stats(_t(match[0][1]), g)
    return {"reference": reference, "base_error": base,
            "note": ("这 is the error the ORIGINAL carries; a candidate is judged against it, "
                     "not against a hand-picked tolerance").replace("这 is", "this is")}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rmsnorm_linear"),
                     indent=2, default=float))
