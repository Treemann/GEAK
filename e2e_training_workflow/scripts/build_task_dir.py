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



def _base_error(rec, golden, labels, reference, layout):
    """Error the ORIGINAL carries against an fp32 reference, per output. Computed once, offline."""
    import sys
    sys.path.insert(0, HERE)
    from l0_gate import REFERENCES, error_stats

    args_t = _tensors(rec["args"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    named = {}
    for name, node in zip(layout["args"], args_t):
        t = node["data"].to(dev)
        # Differentiate exactly the roles the layout says are differentiable. The old
        # `name != "bias"` rule was specific to the two Linear-shaped ops: attention's cu_seqlens
        # are int32 index metadata and `requires_grad_` on them RAISES.
        named[name] = t.requires_grad_(True) if name in layout["grad_of"] else t
    call_args = [named.get(n) for n in layout["args"]]
    out = REFERENCES[reference](*call_args, **layout.get("const", {}))
    out_labels = layout.get("out_labels", ["out"])
    outs = tuple(out) if len(out_labels) > 1 else (out,)
    gos = rec["grad_output"] if isinstance(rec["grad_output"], list) else [rec["grad_output"]]
    gos = [g["data"].to(dev).float() for g in gos]
    wrt = [named[n] for n in layout["grad_of"] if named.get(n) is not None]
    grads = torch.autograd.grad(outs, wrt, grad_outputs=gos, allow_unused=True)
    ref_by_label = dict(zip(out_labels, outs))
    ref_by_label.update(dict(zip(layout["grad_labels"], grads)))
    # Score the original AS THE HARNESS CAN OBSERVE IT. The unittest takes every gradient through
    # autograd onto the captured leaf, so a candidate's gradient always arrives in the LEAF's dtype
    # (bf16) -- but under gradient_accumulation_fusion the original's wgrad was captured from the fp32
    # main_grad. Measuring the limit on the fp32 golden made it tighter than ONE bf16 ulp: on the
    # mlp_fc1_gemm oracle (2026-10-10) an exact replica of TE's math, fp32 wgrad included, failed
    # wgrad at 0.003906 (= 1 bf16 ulp) against a limit of 0.003244, i.e. the oracle rejected the
    # original itself. Rounding the golden to the leaf dtype compares like with like; the fp32
    # main_grad contract is outside what this harness can see and is recorded in task_notes instead.
    leaf_dtype = {lab: named[r].dtype for lab, r in zip(layout["grad_labels"], layout["grad_of"])
                  if named.get(r) is not None}
    stats = {}
    for label, g in zip(labels, golden):
        r = ref_by_label.get(label)
        if r is None:
            continue
        gd = g["data"].to(dev)
        if label in leaf_dtype and gd.dtype != leaf_dtype[label]:
            gd = gd.to(leaf_dtype[label])
        stats[label] = error_stats(gd.float(), r)
    return stats

def build(capture_path, task_dir, reference="rmsnorm_linear", name=None, const=None,
          notes=None, engagement=None, ulp_labels=None):
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
    # A multi-output op (layout `out_labels`, e.g. RoPE on q AND k) carries one grad_output per
    # forward output, in out_labels order.
    gos = go if isinstance(go, list) else [go]
    if not gos or not all(_is_t(g) for g in gos):
        raise SystemExit("capture has no grad_output; a fwd+bwd task cannot be built without it")

    # Order the golden the same way the unittest returns it.
    from l0_gate import LAYOUTS

    layout = dict(LAYOUTS[reference])
    if const:
        layout["const"] = const
    grads = iso.get("grads") or {}
    named_nodes = dict(zip(layout["args"], args_t))
    out_labels = layout.get("out_labels", ["out"])
    iso_outs = iso["out"] if isinstance(iso["out"], list) else [iso["out"]]
    if len(iso_outs) != len(out_labels) or len(gos) != len(out_labels):
        raise SystemExit(f"layout names {len(out_labels)} forward outputs; the capture has "
                         f"{len(iso_outs)} outputs and {len(gos)} grad_outputs")
    golden = list(iso_outs)
    labels = list(out_labels)
    # ORDER, not shape, whenever the isolated re-run gives a complete positional set. Shape matching
    # is AMBIGUOUS for attention -- k and v are both [total_kv, h_kv, d], so a shape map silently
    # hands dk's tensor to dv and the oracle is then wrong in a way that passes every check it runs
    # against itself. The isolated re-run collects leaves in argument order, so its "0","1","2" keys
    # line up with grad_of; the shapes are re-verified below before that is trusted.
    idx_keys = sorted((k for k in grads if k.isdigit() and _is_t(grads[k])), key=int)
    want = [tuple(named_nodes[r]["shape"]) for r in layout["grad_of"] if r in named_nodes]
    positional = (len(idx_keys) == len(want)
                  and all(tuple(grads[k]["shape"]) == w for k, w in zip(idx_keys, want)))
    by_shape = {}
    for v in grads.values():
        if _is_t(v):
            by_shape.setdefault(tuple(v["shape"]), []).append(v)
    for i, (label, role) in enumerate(zip(layout["grad_labels"], layout["grad_of"])):
        node = named_nodes.get(role)
        if node is None:
            continue
        if positional:
            g = grads[idx_keys[i]]
        else:
            cands = by_shape.get(tuple(node["shape"]), [])
            if len(cands) > 1:
                raise SystemExit(
                    f"ambiguous gradient for {role}: {len(cands)} captured gradients share shape "
                    f"{node['shape']} and the positional set is incomplete. Guessing here produces "
                    f"an oracle that is wrong and self-consistent.")
            g = cands[0] if cands else None
        if g is not None:
            golden.append(g)
            labels.append(label)

    blob = {
        "records": [{
            "args": [_harness_tensor(t) for t in args_t] + [_harness_tensor(g) for g in gos],
            "output": [_harness_tensor(g) for g in golden],
            "sig": f"{'x'.join(str(d) for d in args_t[0]['shape'])}|fwd+bwd",
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
    base_error = _base_error(rec, golden, labels, reference, layout)

    meta = {
        "name": name or (rec["target"].split(":")[-1] + "_fwd_bwd"),
        "kind": "training_op",
        "objective": "fwd+bwd COMBINED latency — a forward-only task can call a slower kernel a win",
        "target_callable": rec["target"],
        "baseline_callable": rec["target"],
        "baseline_frozen": True,
        # For a single-Function op the bind point IS the captured target, so it is derived. For a
        # COMPOSITE (the LM head, the MLP fc1+SwiGLU) the target string names a SPAN of call sites
        # and splitting it on ":" yields a symbol that does not resolve -- an engagement contract
        # that cannot be executed is worse than none, so the extractor passes it explicitly.
        "candidate_bind": engagement or {
            "kind": "rebind", "module": rec["target"].split(":")[0],
            "symbol": rec["target"].split(":")[-1],
            "note": "training ops are autograd.Function subclasses: rebind .apply, and "
                    "prove engagement for forward AND backward separately"},
        "signature": {"inputs": [{"name": n, "shape": t["shape"], "dtype": t["dtype"],
                                  "requires_grad": t["requires_grad"]}
                                 for n, t in zip(layout["args"], args_t)],
                      "grad_output": ({"shape": go["shape"], "dtype": go["dtype"]}
                                      if not isinstance(go, list) else
                                      [{"shape": g["shape"], "dtype": g["dtype"]} for g in go]),
                      "outputs": labels},
        "reference_implementation": f"baseline_src/reference_impl.py:{reference}",
        # The generated unittest reads this instead of hardcoding one op's argument order. TE's
        # _Linear takes the WEIGHT first and _LayerNormLinear takes the input first; a template that
        # assumes either one silently unpacks the other wrong.
        "layout": layout,
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
            # Outputs whose MAGNITUDE check is measured in bf16 ulps of the reference (limit = the
            # original's max_ulp_bf16 x margin) instead of in absolute units. Only for outputs whose
            # reference is itself bf16-rounded and whose original rounds once -- there the absolute
            # max_abs is set by where the 1-ulp disagreements happened to land (see
            # l0_gate.max_ulp_bf16). The bias checks below are unchanged for every output.
            "ulp_labels": list(ulp_labels or []),
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
        # Everything the extractor knows that the tensors do not say: what the oracle deliberately
        # does NOT cover, and what the integration therefore still has to prove. A caveat that lives
        # only in a chat message is a caveat that will be discovered by a failed gate.
        "task_notes": notes or {},
    }
    json.dump(meta, open(os.path.join(task_dir, "meta.json"), "w"), indent=2)

    with open(os.path.join(task_dir, "unittest.py"), "w") as f:
        f.write(UNITTEST.replace("__REFERENCE__", reference))
    os.chmod(os.path.join(task_dir, "unittest.py"), 0o444)
    os.chmod(os.path.join(task_dir, "meta.json"), 0o444)
    # The frozen inputs are the whole point: the kernel layer must be scored against data it cannot
    # influence, so the oracle is made read-only here rather than by convention.
    os.chmod(ref_path, 0o444)
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


LAYOUT = META["layout"]


def _rmsnorm_linear(x, ln_w, w, b=None, eps=1e-6):
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return torch.nn.functional.linear(xf * rms * ln_w.float(), w.float(),
                                      b.float() if b is not None else None)


def _linear(w, x, b=None):
    return torch.nn.functional.linear(x.float(), w.float(), b.float() if b is not None else None)


_REFS = {"rmsnorm_linear": _rmsnorm_linear, "linear": _linear}
# Anything more than two lines lives in baseline_src/reference_impl.py (the frozen copy of
# l0_gate.py) and is loaded from there rather than duplicated into this template: two copies of an
# attention reference are two chances to disagree, and the one that disagrees silently is the oracle.
try:
    _rspec = importlib.util.spec_from_file_location(
        "reference_impl", os.path.join(HERE, "baseline_src", "reference_impl.py"))
    _ri = importlib.util.module_from_spec(_rspec); _rspec.loader.exec_module(_ri)
    _REFS.update(getattr(_ri, "REFERENCES", {}))
except Exception as _e:
    print(f"  [warn] could not load baseline_src/reference_impl.py: {_e!r}")
_reference = _REFS["__REFERENCE__"]
CONST = LAYOUT.get("const", {})


def make_call(entry):
    """Wrap fwd+bwd into ONE callable: latency covers both, correctness covers every output.

    Arguments are bound BY NAME from meta's layout. Positional unpacking would be wrong for one of
    the two ops shipped here -- TE's _Linear takes the weight first, _LayerNormLinear the input.
    """
    names = LAYOUT["args"]
    out_labels = LAYOUT.get("out_labels", ["out"])
    n_go = len(out_labels)          # one grad_output per forward output, appended after the inputs

    def call(args):
        tensors, gos = list(args[:-n_go]), list(args[-n_go:])
        named = {}
        for name, t in zip(names, tensors):
            c = t.detach().clone()
            # Exactly the roles the layout calls differentiable. `!= "bias"` was a Linear-ism:
            # attention's cu_seqlens are int32 index metadata and requires_grad_ on them RAISES.
            if name in LAYOUT["grad_of"]:
                c.requires_grad_(True)
            named[name] = c
        out = entry(*[named.get(n) for n in names], **CONST)
        if n_go > 1:
            outs = tuple(out)
            if len(outs) != n_go:
                raise RuntimeError(f"entry returned {len(outs)} outputs, the op has {n_go}")
        else:
            outs = (out[0] if isinstance(out, (list, tuple)) else out,)
        wrt = [named[n] for n in LAYOUT["grad_of"] if named.get(n) is not None]
        grads = torch.autograd.grad(outs, wrt, grad_outputs=gos, allow_unused=True)
        return outs + tuple(g for g in grads if g is not None)
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
    ref_call = make_call(_reference)
    ref_outs = ref_call(cases[0]["args"])
    _ol = LAYOUT.get("out_labels", ["out"])
    ref_by_label = dict(zip(_ol, ref_outs[:len(_ol)]))
    ref_by_label.update(dict(zip(LAYOUT["grad_labels"], ref_outs[len(_ol):])))

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
        if label in acc.get("ulp_labels", []):
            # Magnitude in bf16 ulps of the (bf16-rounded) reference; see acceptance.ulp_labels.
            ulp = torch.where(rr != 0,
                              torch.exp2(torch.floor(torch.log2(rr.abs().clamp_min(2.0 ** -126))) - 7),
                              torch.full_like(rr, 2.0 ** -133))
            st["max_ulp_bf16"] = float((d.abs() / ulp).max())
            lim_ulp = base[label]["max_ulp_bf16"] * margin
            st["limit_ulp"] = lim_ulp
            mag_ok = st["max_ulp_bf16"] <= lim_ulp
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
        mag_txt = (f"max_ulp={st['max_ulp_bf16']:.3g} (limit {st['limit_ulp']:.3g} ulp, original "
                   f"{base[label]['max_ulp_bf16']:.3g}) max_abs={st['max_abs']:.4g}"
                   if "limit_ulp" in st else
                   f"max_abs={st['max_abs']:.4g} (limit {lim:.4g}, original "
                   f"{base[label]['max_abs']:.4g})")
        print(f"  [L0] {label}: {mag_txt} sign={st['sign_balance']:.3f} "
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
    # Non-tensor arguments the reference needs (attn_scale, causal, ...). They are CONSTANTS of the
    # captured call, recorded in meta so the oracle is self-describing and the candidate sees the
    # same ones. Defaulting them silently is how an oracle ends up scoring a different operation.
    ap.add_argument("--const", default=None, help="JSON dict of non-tensor reference arguments")
    ap.add_argument("--notes", default=None, help="JSON dict (or @file.json) recorded as meta.task_notes")
    # A composite op spans several call sites, so its bind point cannot be derived from the capture
    # target string. Passed explicitly rather than guessed.
    ap.add_argument("--engagement", default=None,
                    help="JSON dict (or @file.json) recorded as meta.candidate_bind")
    ap.add_argument("--ulp-labels", default="",
                    help="comma list of outputs whose magnitude check is in bf16 ulps "
                         "(see acceptance.ulp_labels)")
    a = ap.parse_args()
    print(json.dumps(build(a.capture, a.task_dir, a.reference, a.name,
                       json.loads(a.const) if a.const else None,
                       json.loads(open(a.notes[1:]).read() if a.notes.startswith("@") else a.notes)
                       if a.notes else None,
                       json.loads(open(a.engagement[1:]).read() if a.engagement.startswith("@")
                                  else a.engagement) if a.engagement else None,
                       [s for s in a.ulp_labels.split(",") if s.strip()]), indent=2))
