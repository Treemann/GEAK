"""Turn a training capture into an op task dir the UNCHANGED kernel_workflow can consume.

THE SEAM. `kernel_workflow` expects an op task dir holding an IMMUTABLE oracle: `meta.json` +
`unittest.py` + a frozen baseline (`baseline_src/` or a resolvable `baseline_callable`) + optionally
`reference_io.pt`. Our capture produces none of that shape, so without this converter the recursive
call hands the kernel layer something it cannot read.

THE ONE IDEA THAT MAKES IT FIT. A training op's task is `fwd+bwd`, but the kernel layer's harness is
built around `call(args) -> out` with component-wise comparison of multi-tensor returns. So the
generated `unittest.py` wraps forward AND backward into a SINGLE callable returning
`(out, dgrad, dln_weight, wgrad)`. Everything downstream then works unchanged:
  * `h.check_correct_multi` compares all four component-wise -- L0 on the whole op, for free;
  * `h.time_op` on that same closure measures fwd+bwd COMBINED, which is the objective;
  * the output-independence check still applies.
No change to the kernel layer, which is the point.

WHAT THE TASK IS, PRECISELY. The captured call carries runtime handles (ProcessGroup, main_grad
buffers, the owning module) that cannot be serialized. So the task is NOT "replay this call": it is
the PURE-TENSOR mathematical operation the op implements, with the golden taken from an isolated
re-run of the original in the live process. The handles are recorded in `meta.json` as context, not
as inputs -- a candidate kernel is not expected to reproduce them.

TOLERANCE. `meta.json` carries the error the ORIGINAL already carries against an fp32 reference, so
the unittest can enforce "no further from fp32 than the original" instead of a hand-picked rtol, and
the sign test that no magnitude check can replace.

Usage:
    python3 build_task_dir.py <capture.pt> <task_dir> [--reference rmsnorm_linear] [--name NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _is_t(n):
    return isinstance(n, dict) and n.get("__t__")


def _tensors(node, out=None):
    out = [] if out is None else out
    if _is_t(node):
        out.append(node)
    elif isinstance(node, list):
        for v in node:
            _tensors(v, out)
    elif isinstance(node, dict):
        for v in node.values():
            _tensors(v, out)
    return out


def _harness_tensor(node):
    """Our snapshot marker -> the harness's (`capture_shapes._snapshot`) marker."""
    return {"__tensor__": True, "data": node["data"]}



def _base_error(rec, golden, labels, reference):
    """Error the ORIGINAL carries against an fp32 reference, per output. Computed once, offline."""
    import sys
    sys.path.insert(0, HERE)
    from l0_gate import REFERENCES, error_stats

    args_t = _tensors(rec["args"])
    x, ln_w, w, b = (args_t + [None] * 4)[:4]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _ld(n, grad=False):
        t = n["data"].to(dev)
        return t.requires_grad_(True) if grad else t

    xg, lg, wg = _ld(x, True), _ld(ln_w, True), _ld(w, True)
    bt = _ld(b) if b is not None else None
    out = REFERENCES[reference](xg, lg, wg, bt)
    go = _ld(rec["grad_output"])
    grads = torch.autograd.grad(out, [xg, lg, wg], grad_outputs=go.float(), allow_unused=True)
    ref_by_label = {"out": out, "dgrad": grads[0], "dln_weight": grads[1], "wgrad": grads[2]}
    stats = {}
    for label, g in zip(labels, golden):
        r = ref_by_label.get(label)
        if r is None:
            continue
        stats[label] = error_stats(g["data"].to(dev).float(), r)
    return stats

def build(capture_path, task_dir, reference="rmsnorm_linear", name=None):
    rec = torch.load(capture_path, map_location="cpu", weights_only=False)
    iso = rec.get("isolated")
    if not iso:
        raise SystemExit(
            "capture has no `isolated` block. The golden MUST come from an isolated re-run of the "
            "original op (see capture_train_shapes): intercepting the live backward cannot be paired "
            "to the captured forward, and a mis-paired oracle is worse than none.")

    os.makedirs(os.path.join(task_dir, "baseline_src"), exist_ok=True)
    args_t = _tensors(rec["args"])
    go = rec.get("grad_output")
    if not _is_t(go):
        raise SystemExit("capture has no grad_output; a fwd+bwd task cannot be built without it")

    # Order the golden the same way the unittest returns it.
    grads = iso.get("grads") or {}
    by_shape = {tuple(v["shape"]): v for v in grads.values() if _is_t(v)}
    x, ln_w, w, b = (args_t + [None] * 4)[:4]
    golden = [iso["out"]]
    labels = ["out"]
    for label, node in (("dgrad", x), ("dln_weight", ln_w), ("wgrad", w)):
        g = by_shape.get(tuple(node["shape"])) if node else None
        if g is not None:
            golden.append(g)
            labels.append(label)

    blob = {
        "records": [{
            "args": [_harness_tensor(t) for t in args_t] + [_harness_tensor(go)],
            "output": [_harness_tensor(g) for g in golden],
            "sig": f"{'x'.join(str(d) for d in x['shape'])}|fwd+bwd",
            "regime": "training",
        }],
    }
    ref_path = os.path.join(task_dir, "reference_io.pt")
    torch.save(blob, ref_path)
    sha = hashlib.sha256(open(ref_path, "rb").read()).hexdigest()

    shutil.copy2(os.path.join(HERE, "l0_gate.py"),
                 os.path.join(task_dir, "baseline_src", "reference_impl.py"))

    # The acceptance thresholds ARE the error the original already carries. Computing them here, once,
    # makes the task self-contained: the unittest never has to pick an rtol, and a candidate is never
    # penalised for rounding the baseline also does.
    base_error = _base_error(rec, golden, labels, reference)

    meta = {
        "name": name or (rec["target"].split(":")[-1] + "_fwd_bwd"),
        "kind": "training_op",
        "objective": "fwd+bwd COMBINED latency — a forward-only task can call a slower kernel a win",
        "target_callable": rec["target"],
        "baseline_callable": rec["target"],
        "baseline_frozen": True,
        "candidate_bind": {"kind": "rebind", "module": rec["target"].split(":")[0],
                           "symbol": rec["target"].split(":")[-1],
                           "note": "training ops are autograd.Function subclasses: rebind .apply, and "
                                   "prove engagement for forward AND backward separately"},
        "signature": {"inputs": [{"name": n, "shape": t["shape"], "dtype": t["dtype"],
                                  "requires_grad": t["requires_grad"]}
                                 for n, t in zip(["x", "ln_weight", "weight", "bias"], args_t)],
                      "grad_output": {"shape": go["shape"], "dtype": go["dtype"]},
                      "outputs": labels},
        "reference_implementation": f"baseline_src/reference_impl.py:{reference}",
        "opaque_context": rec.get("opaque_args") or [],
        "opaque_note": ("runtime handles from the live call. They are CONTEXT, not inputs: the task is "
                        "the pure-tensor operation, and a candidate is not expected to reproduce them."),
        "base_error": base_error,
        "acceptance": {
            "rule": ("a candidate is correct iff, for every output, it is no further from the fp32 "
                     "reference than the ORIGINAL is (x margin), and its error is not one-sided"),
            "margin": 1.25,
            "sign_tol": 0.15,
            "scale_bias_tol": 0.002,
            "offset_bias_tol": 0.01,
            "why_bias_tests": ("sign_balance catches an ADDITIVE bias; a MULTIPLICATIVE one leaves it "
                               "at 0.5, because the error simply follows the value's sign. Proven by "
                               "a negative test: a x1.005 candidate passed BOTH the magnitude and the "
                               "sign check, and its magnitude passed because the original's own bf16 "
                               "error is larger than the bias. scale_bias = <err,ref>/<ref,ref> is the "
                               "implied scale offset and catches it directly."),
            "why_not_rtol": ("a flat per-element rtol is meaningless here: dgrad is ~77% zeros "
                             "(padding carries no gradient), so max-rel-err explodes on tiny values. "
                             "Measured on the untouched original: max_rel_err 11.7 against its OWN "
                             "fp32 reference, at a mean relative error of 2.7%."),
        },
        "num_cases": 1,
        "regime": "training",
        "build": False,
        "reference_io_sha256": sha,
        "captured_from": {"step": rec.get("step"), "rank": rec.get("rank")},
    }
    json.dump(meta, open(os.path.join(task_dir, "meta.json"), "w"), indent=2)

    with open(os.path.join(task_dir, "unittest.py"), "w") as f:
        f.write(UNITTEST.replace("__REFERENCE__", reference))
    os.chmod(os.path.join(task_dir, "unittest.py"), 0o444)
    os.chmod(os.path.join(task_dir, "meta.json"), 0o444)
    return {"task_dir": task_dir, "outputs": labels, "sha256": sha,
            "inputs": [t["shape"] for t in args_t]}


UNITTEST = r'''"""IMMUTABLE fwd+bwd oracle. Do not edit: the validator sha-checks it with reference_io.pt.

A training op is judged on forward AND backward together, so the measured callable runs both and
returns (out, dgrad, dln_weight, wgrad). The harness compares multi-tensor returns component-wise,
so one closure gives correctness on every quantity and a combined-latency measurement.
"""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("harness_lib", os.path.join(HERE, "harness_lib.py"))
h = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(h)
if HERE in sys.path:
    sys.path.remove(HERE)          # this file is named unittest.py; unshadow the stdlib module
import torch

META = json.load(open(os.path.join(HERE, "meta.json")))


def _reference(x, ln_w, w, b, eps=1e-6):
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return torch.nn.functional.linear(xf * rms * ln_w.float(), w.float(),
                                      b.float() if b is not None else None)


def make_call(entry):
    """Wrap fwd+bwd into ONE callable, so latency covers both and correctness covers all outputs."""
    def call(args):
        x, ln_w, w, b, go = args
        x = x.detach().clone().requires_grad_(True)
        ln_w = ln_w.detach().clone().requires_grad_(True)
        w = w.detach().clone().requires_grad_(True)
        out = entry(x, ln_w, w, b)
        tgt = out[0] if isinstance(out, (list, tuple)) else out
        grads = torch.autograd.grad(tgt, [x, ln_w, w], grad_outputs=go, allow_unused=True)
        return (tgt,) + tuple(g for g in grads if g is not None)
    return call


def main():
    cases = list(h.iter_eager_cases_from_oracle(os.path.join(HERE, "reference_io.pt"), device="cuda"))
    entry = _reference
    kernel_src = os.path.join(HERE, "kernel_src")
    if os.path.isdir(kernel_src):                    # a candidate has been committed
        sys.path.insert(0, kernel_src)
        import kernel  # noqa: F401  (the candidate module)
        entry = kernel.entry
    call = make_call(entry)

    # Structural checks from the harness (shapes, NaN, output independence). Its per-element
    # max_rel_err is REPORTED, not gated on: dgrad is ~77% zeros here (padding carries no gradient),
    # so per-element relative error explodes on tiny values -- the untouched original scores 11.7
    # against its own fp32 reference. The real gate is the base-error rule below.
    _, per_case = h.check_correct_multi(call, cases, tol=2e-2)
    for c in per_case:
        print(f"  [structural] {c.get('case','')}: max_rel_err={c.get('max_rel_err')}")

    got = call(cases[0]["args"])
    x, ln_w, w, b, go = cases[0]["args"]
    xg = x.detach().clone().requires_grad_(True)
    lg = ln_w.detach().clone().requires_grad_(True)
    wg = w.detach().clone().requires_grad_(True)
    ref_out = _reference(xg, lg, wg, b)
    ref_grads = torch.autograd.grad(ref_out, [xg, lg, wg], grad_outputs=go.float(), allow_unused=True)
    ref_by_label = {"out": ref_out, "dgrad": ref_grads[0], "dln_weight": ref_grads[1],
                    "wgrad": ref_grads[2]}

    base, acc = META.get("base_error", {}), META.get("acceptance", {})
    margin, sign_tol = acc.get("margin", 1.25), acc.get("sign_tol", 0.15)
    scale_tol = acc.get("scale_bias_tol", 0.002)
    offset_tol = acc.get("offset_bias_tol", 0.01)
    ok, report = True, {}
    for label, actual in zip(META["signature"]["outputs"], got):
        r = ref_by_label.get(label)
        if r is None or label not in base:
            continue
        a = actual.detach().float().flatten()
        rr = r.detach().float().flatten()
        d = a - rr
        nz = int((d != 0).sum())
        denom = rr.abs().mean().clamp_min(1e-12)
        rr2 = float((rr * rr).sum())
        st = {"max_abs": float(d.abs().max()),
              "rel_mean_abs": float(d.abs().mean() / denom),
              "sign_balance": (float((d > 0).sum()) / nz) if nz else 0.5,
              "scale_bias": float((d * rr).sum() / rr2) if rr2 > 0 else 0.0,
              "offset_bias": float(d.mean() / denom)}
        lim = base[label]["max_abs"] * margin
        mag_ok = st["max_abs"] <= lim
        sign_ok = abs(st["sign_balance"] - 0.5) <= sign_tol or \
            abs(base[label]["sign_balance"] - 0.5) > sign_tol
        # A MULTIPLICATIVE bias survives both checks above -- proven by a negative test -- so the
        # error-vs-value correlation is checked against the original's own.
        scale_ok = abs(st["scale_bias"]) <= max(scale_tol,
                                                abs(base[label].get("scale_bias", 0.0)) * margin)
        offset_ok = abs(st["offset_bias"]) <= max(offset_tol,
                                                  abs(base[label].get("offset_bias", 0.0)) * margin)
        good = mag_ok and sign_ok and scale_ok and offset_ok
        ok = ok and good
        report[label] = {**st, "limit": lim, "magnitude_ok": mag_ok, "sign_ok": sign_ok,
                         "scale_ok": scale_ok, "offset_ok": offset_ok,
                         "base_max_abs": base[label]["max_abs"]}
        flags = ''.join(x for x, bad in ((' magnitude', not mag_ok), (' sign', not sign_ok),
                                         (' SCALE-BIAS', not scale_ok),
                                         (' offset-bias', not offset_ok)) if bad)
        print(f"  [L0] {label}: max_abs={st['max_abs']:.4g} (limit {lim:.4g}, original "
              f"{base[label]['max_abs']:.4g}) sign={st['sign_balance']:.3f} "
              f"scale_bias={st['scale_bias']:+.2e} -> {'PASS' if good else 'FAIL' + flags}")

    t = h.time_op(lambda: call(cases[0]["args"]))
    ms = t["ms"] if isinstance(t, dict) else t
    print(json.dumps({"correct": bool(ok), "latency_ms": ms, "l0": report,
                      "objective": META["objective"], "outputs": META["signature"]["outputs"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("task_dir")
    ap.add_argument("--reference", default="rmsnorm_linear")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    print(json.dumps(build(a.capture, a.task_dir, a.reference, a.name), indent=2))
