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

def _record(rec, layout):
    """One capture record -> (harness blob, golden, labels, args_t, grad_output node(s))."""
    iso = rec.get("isolated")
    if not iso:
        raise SystemExit(
            "capture has no `isolated` block. The golden MUST come from an isolated re-run of the "
            "original op (see capture_train_shapes): intercepting the live backward cannot be paired "
            "to the captured forward, and a mis-paired oracle is worse than none.")

    args_t = _tensors(rec["args"])
    go = rec.get("grad_output")
    # A multi-output op (layout `out_labels`, e.g. RoPE on q AND k) carries one grad_output per
    # forward output, in out_labels order.
    gos = go if isinstance(go, list) else [go]
    if not gos or not all(_is_t(g) for g in gos):
        raise SystemExit("capture has no grad_output; a fwd+bwd task cannot be built without it")

    # Order the golden the same way the unittest returns it.
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
    return blob, golden, labels, args_t, go


def build(capture_path, task_dir, reference="rmsnorm_linear", name=None, const=None,
          notes=None, engagement=None, ulp_labels=None, holdouts=None, ulp_floor=None,
          abs_floor_max_ulp=None, exact_labels=None):
    """`exact_labels`: outputs that must be BIT-IDENTICAL (value-equal, NaN where NaN) to the frozen
    golden, i.e. to the ORIGINAL's own output (the golden is the isolated re-run of the original), on
    the primary case and on every holdout -- in ADDITION to the fp32-reference rule. For a remedy whose
    whole point is reproducing the original's rounding (hf_rope_bf16_exact, 2026-10-19: a candidate
    strictly CLOSER to fp32 was rejected by L1 for a deterministic +0.030% step-1 loss shift), "no
    further from fp32 than the original" is the wrong contract: it admits exactly the variant that
    failed.

    `holdouts`: optional list of (tag, capture_path) for the SAME op at other call sites / ranks /
    shapes. Each is frozen as holdout_<tag>.pt with its OWN base error (the original's error differs
    per call: on hf_rmsnorm_fused_fwdbwd out max_abs is 0.029 at layer 0 and 1.12 at model.norm), and
    the generated unittest gates on ALL of them. Why gate rather than advise: on that oracle the
    primary call (layer 14, mean(x^2) >= 3) cannot see a wrong eps or a bf16-rounded rstd, and the
    layer-0 call (mean(x^2) ~ 4e-10..1e-3, eps-dominated) cannot see a missing dx projection term --
    no single call covers the op's contract."""
    rec = torch.load(capture_path, map_location="cpu", weights_only=False)
    from l0_gate import LAYOUTS

    layout = dict(LAYOUTS[reference])
    if const:
        layout["const"] = const
    os.makedirs(os.path.join(task_dir, "baseline_src"), exist_ok=True)
    blob, golden, labels, args_t, go = _record(rec, layout)
    ref_path = os.path.join(task_dir, "reference_io.pt")
    torch.save(blob, ref_path)
    sha = hashlib.sha256(open(ref_path, "rb").read()).hexdigest()

    shutil.copy2(os.path.join(HERE, "l0_gate.py"),
                 os.path.join(task_dir, "baseline_src", "reference_impl.py"))

    # The acceptance thresholds ARE the error the original already carries. Computing them here, once,
    # makes the task self-contained: the unittest never has to pick an rtol, and a candidate is never
    # penalised for rounding the baseline also does.
    base_error = _base_error(rec, golden, labels, reference, layout)

    holdout_meta = []
    for tag, hpath in (holdouts or []):
        hrec = torch.load(hpath, map_location="cpu", weights_only=False)
        hc = (hrec.get("const") or {})
        if layout.get("const") and hc and any(hc.get(k) != v for k, v in layout["const"].items()
                                              if k in hc):
            raise SystemExit(f"holdout {tag} carries const {hc}, the oracle {layout['const']}: "
                             f"a different operation, not a holdout")
        hblob, hgold, hlabels, hargs, _ = _record(hrec, layout)
        if hlabels != labels:
            raise SystemExit(f"holdout {tag} yields outputs {hlabels}, the oracle {labels}")
        fn = f"holdout_{tag}.pt"
        torch.save(hblob, os.path.join(task_dir, fn))
        os.chmod(os.path.join(task_dir, fn), 0o444)
        holdout_meta.append({
            "tag": tag, "file": fn,
            "sha256": hashlib.sha256(open(os.path.join(task_dir, fn), "rb").read()).hexdigest(),
            "inputs": [t["shape"] for t in hargs],
            "captured_from": {"step": hrec.get("step"), "rank": hrec.get("rank"),
                              "call": hrec.get("call"), "module": hrec.get("module")},
            "base_error": _base_error(hrec, hgold, hlabels, reference, layout)})

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
            # The sign test only votes when the one-sidedness is STATISTICALLY SIGNIFICANT: |#pos -
            # nz/2| > sign_min_z * sqrt(nz)/2 (binomial, z sigma). Without it a near-EXACT candidate
            # fails: on the hf_rmsnorm_fused_fwdbwd oracle (2026-10-17) fp64-accumulated dgamma differed
            # from round_bf16(fp32 ref) on a handful of the 3584 elements, all one way -> sign 0.000
            # at max_abs 3.7e-9 (the original's is 1.9e-6), and aiter's triton rms_norm scored sign
            # 1.000 at max_abs 2.9e-11 -- both FAILED. An additive bias that matters moves thousands
            # of elements and stays far above this bar (the x+0.01*mean|dx| negative still fails).
            "sign_min_z": 4.0,
            # Outputs whose MAGNITUDE check is measured in bf16 ulps of the reference (limit = the
            # original's max_ulp_bf16 x margin) instead of in absolute units. Only for outputs whose
            # reference is itself bf16-rounded and whose original rounds once -- there the absolute
            # max_abs is set by where the 1-ulp disagreements happened to land (see
            # l0_gate.max_ulp_bf16). The bias checks below are unchanged for every output.
            "ulp_labels": list(ulp_labels or []),
            # Per-label FLOOR (in bf16 ulps) under the ulp limit of an `ulp_labels` output, for outputs
            # where the original is EXACTLY round_bf16(fp32 reference) (base max_ulp_bf16 == 0, so the
            # x margin limit is 0 and only a bitwise-identical rounding passes). First needed on
            # hf_attn_qkvo_proj (2026-10-19): the original's dbias is an fp32-accumulated column sum
            # rounded once, and matches the fp32 reference bitwise on all 4608 elements of every
            # case; an equally correct fp32 sum in a different ORDER flips the final rounding by 1 ulp
            # wherever the fp32 value sits within ~1e-7 relative of a rounding midpoint. A floor of
            # 1 ulp admits faithful rounding and nothing else: a bf16-accumulated sum is off by many
            # ulps and still fails (negative control dbias_bf16acc). The bias checks are unchanged.
            "ulp_floor": dict(ulp_floor or {}),
            # Per-label FLOOR under the ABSOLUTE magnitude limit, in bf16 ulps of the case's LARGEST
            # reference element: limit = max(base max_abs x margin, k * ulp_bf16(max|ref|)). WHY: on a
            # bf16-rounded GEMM output the original's max_abs is ONE rounding flip, and which binade
            # it landed in is luck. Measured on hf_attn_qkvo_proj (2026-10-19): the original's own
            # numerics with the three bf16 q/k/v dgrads summed in a different ORDER failed dgrad at
            # 9.5e-7 vs a limit of 6.0e-7 (original 4.8e-7), and an fp32 split-K wgrad (MORE
            # accurate) failed wgrad_v at 3.8e-6 vs 1.2e-6 -- in both, one flip at a larger element.
            # One ulp at the tensor's largest element is the error a correctly rounded result may
            # carry there; anything a real defect produces (bf16/fp16 accumulation, a dropped M tail,
            # a swapped role, x1.004) is far above it or is caught by the bias checks, which are
            # unchanged (negative controls in the task's extract_controls_summary.json).
            "abs_floor_max_ulp": dict(abs_floor_max_ulp or {}),
            # DEFAULT for every output of a NEW task (2026-10-22, DECISIONS 4.52): an element over the
            # magnitude limit is forgiven iff its error is <= 1 bf16 ulp of ITS OWN reference value
            # (faithful rounding: a different but correct summation order flips a value sitting on a
            # rounding midpoint), and at most max_frac of the elements (>= 1) may be forgiven. Found by
            # the s8 holdouts (4.47): qkv dbias and rmsnorm dgrad each failed ONE holdout on ONE
            # element 1 ulp off, because the original happened to be exact there (limit 0). The bias
            # checks (sign, scale, offset) and exact_labels are NOT relaxed -- a systematic 1-ulp
            # error (wrong rounding mode) moves thousands of elements one way and is caught there.
            "ulp1_elementwise": {"enabled": True, "max_frac": 0.001},
            # Outputs that must equal the frozen golden (= the original's output) BITWISE; see
            # build(exact_labels=...). Signed zeros compare equal (-0 == +0); NaN must sit where the
            # golden has NaN. Absent -> [] (every older task unchanged).
            "exact_labels": list(exact_labels or []),
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
        # Gated cases beyond reference_io.pt (see build(holdouts=...)). Each carries its own limits.
        "holdouts": holdout_meta,
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
            "holdouts": [{k: h[k] for k in ("tag", "file", "sha256", "inputs")} for h in holdout_meta],
            "inputs": [t["shape"] for t in args_t]}


UNITTEST = r'''"""IMMUTABLE fwd+bwd oracle. Do not edit: the validator sha-checks it with reference_io.pt.

A training op is judged on forward AND backward together, so the measured callable runs both and
returns (out, dgrad, dln_weight, wgrad). The harness compares multi-tensor returns component-wise,
so one closure gives correctness on every quantity and a combined-latency measurement.
"""
import importlib.util, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("harness_lib", os.path.join(HERE, "harness_lib.py"))
h = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(h)
if HERE in sys.path:
    sys.path.remove(HERE)          # this file is named unittest.py; unshadow the stdlib module
import torch


def bf16_ulp(ref):
    """One bf16 ulp of each reference element (2^(exponent-7)); a tiny value for exact zeros."""
    return torch.where(ref != 0, torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0 ** -126))) - 7),
                       torch.full_like(ref, 2.0 ** -133))


def ulp1_forgive(d, ref, limit_abs, max_frac=0.001):
    """acceptance.ulp1_elementwise (DECISIONS 4.52): elements whose |error| exceeds the magnitude limit
    are forgiven iff EACH is within 1 bf16 ulp of its own reference value AND there are at most
    max(1, ceil(max_frac * N)) of them. Returns (ok, info). The bias checks are not touched here."""
    over = d.abs() > limit_abs
    n_over = int(over.sum())
    if n_over == 0:
        return True, {"ulp1_over": 0, "ulp1_forgiven": 0}
    within = bool((d.abs()[over] <= bf16_ulp(ref)[over] * (1 + 1e-6)).all())
    cap = max(1, int(math.ceil(max_frac * d.numel())))
    ok = within and n_over <= cap
    return ok, {"ulp1_over": n_over, "ulp1_cap": cap, "ulp1_all_within_1ulp": within,
                "ulp1_forgiven": n_over if ok else 0}

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

    # Optional LAYOUT GLUE (layout["glue"]; absent on every older task -> unchanged behaviour). It makes
    # the timed callable span the live layout copies AROUND the op, so a candidate that removes them is
    # credited for it and one that adds them is charged (first used by hf_rope_layout_fused):
    #   view_from_base: [names] -- the live arg is a [B,H,S,D] TRANSPOSED VIEW of a [B,S,H*D] projection
    #       output. The leaf is that contiguous base; the op receives base.view(B,S,H,D).transpose(1,2)
    #       (exactly the live view), and the gradient is taken on the base -- so whatever layout the
    #       candidate's dgrad has, the ViewBackward copy it costs in training is timed here too. It is
    #       reported back as [B,H,S,D] (a view, values unchanged) for scoring.
    #   contiguous_out: true -- each output goes through .contiguous() before the backward, as the
    #       consumer does live (sdpa_attention_forward); a no-op for an output already contiguous.
    glue = LAYOUT.get("glue") or {}
    from_base = set(glue.get("view_from_base") or [])

    def call(args):
        tensors, gos = list(args[:-n_go]), list(args[-n_go:])
        named, leaves = {}, {}
        for name, t in zip(names, tensors):
            if name in from_base:
                B, H, S, D = t.shape
                base = t.detach().transpose(1, 2).reshape(B, S, H * D).clone()
                base.requires_grad_(name in LAYOUT["grad_of"])
                leaves[name] = base
                named[name] = base.view(B, S, H, D).transpose(1, 2)
                continue
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
        if glue.get("contiguous_out"):
            outs = tuple(o.contiguous() for o in outs)
        wrt_names = [n for n in LAYOUT["grad_of"] if named.get(n) is not None]
        wrt = [leaves.get(n, named[n]) for n in wrt_names]
        grads = torch.autograd.grad(outs, wrt, grad_outputs=gos, allow_unused=True)
        grads = tuple(g.view(named[n].shape[0], named[n].shape[2], named[n].shape[1],
                             named[n].shape[3]).transpose(1, 2)
                      if (g is not None and n in leaves) else g for n, g in zip(wrt_names, grads))
        return outs + tuple(g for g in grads if g is not None)
    return call


def main():
    cases = list(h.iter_eager_cases_from_oracle(os.path.join(HERE, "reference_io.pt"), device="cuda"))
    entry = _reference
    if META.get("acceptance", {}).get("exact_labels"):
        # A BITWISE task's baseline is the ORIGINAL (its golden), not the fp32 reference -- which by
        # construction is not bit-identical to it and would fail every exact check. The fp32 reference
        # stays the scorer of the fp32 rule below.
        import importlib
        _m, _s = META["baseline_callable"].split(":")
        entry = getattr(importlib.import_module(_m), _s)
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

    def _l0(case_args, base, tag="", golden=None):
        """L0 on one frozen case against THAT case's original error (its own limits)."""
        got = call(case_args)
        ref_call = make_call(_reference)
        ref_outs = ref_call(case_args)
        _ol = LAYOUT.get("out_labels", ["out"])
        ref_by_label = dict(zip(_ol, ref_outs[:len(_ol)]))
        ref_by_label.update(dict(zip(LAYOUT["grad_labels"], ref_outs[len(_ol):])))

        acc = META.get("acceptance", {})
        margin, sign_tol = acc.get("margin", 1.25), acc.get("sign_tol", 0.15)
        scale_tol = acc.get("scale_bias_tol", 0.002)
        offset_tol = acc.get("offset_bias_tol", 0.01)
        ok, report = True, {}
        # EVERY output must be scored (audit 2026-10-22). make_call drops None gradients, so a
        # candidate that omits a trailing gradient (e.g. "accumulated into main_grad") returned a
        # SHORTER tuple, zip() truncated it, that label was never compared and L0 said correct=true.
        # A None in the middle shifted every later label. Count, coverage and shape are now gated.
        _labels = list(META["signature"]["outputs"])
        if len(got) != len(_labels) or len(ref_outs) != len(_labels):
            print(f"  [L0{tag}] output count: candidate {len(got)}, reference {len(ref_outs)}, "
                  f"labels {len(_labels)} {_labels} -> FAIL (a missing output is never 'not scored')")
            return False, {"count": {"candidate": len(got), "reference": len(ref_outs), "labels": _labels}}
        _unscored = [l for l in base if l not in _labels or ref_by_label.get(l) is None]
        if _unscored:
            print(f"  [L0{tag}] labels with a base error but no reference output: {_unscored} -> FAIL")
            return False, {"unscored": _unscored}
        for label, actual in zip(_labels, got):
            r = ref_by_label.get(label)
            if actual is None or (r is not None and tuple(actual.shape) != tuple(r.shape)):
                print(f"  [L0{tag}] {label}: candidate output "
                      f"{'missing' if actual is None else tuple(actual.shape)} vs reference "
                      f"{None if r is None else tuple(r.shape)} -> FAIL")
                return False, {"shape": {label: [None if actual is None else list(actual.shape),
                                                 None if r is None else list(r.shape)]}}
        for label, actual in zip(_labels, got):
            r = ref_by_label.get(label)
            if r is None or label not in base:
                continue
            a = actual.detach().float().flatten()
            rr = r.detach().float().flatten()
            d = a - rr
            nz = int((d != 0).sum())
            npos = int((d > 0).sum())
            denom = rr.abs().mean().clamp_min(1e-12)
            rr2 = float((rr * rr).sum())
            st = {"max_abs": float(d.abs().max()),
                  "rel_mean_abs": float(d.abs().mean() / denom),
                  "sign_balance": (float((d > 0).sum()) / nz) if nz else 0.5,
                  "scale_bias": float((d * rr).sum() / rr2) if rr2 > 0 else 0.0,
                  "offset_bias": float(d.mean() / denom)}
            lim = base[label]["max_abs"] * margin
            _fl = float(acc.get("abs_floor_max_ulp", {}).get(label, 0.0))
            if _fl > 0:
                _m = float(rr.abs().max())
                if _m > 0:
                    st["floor_max_ulp"] = _fl * 2.0 ** (math.floor(math.log2(_m)) - 7)
                    lim = max(lim, st["floor_max_ulp"])
            mag_ok = st["max_abs"] <= lim
            # Per-element faithful-rounding allowance (acceptance.ulp1_elementwise, DECISIONS 4.52).
            # Absent from older metas -> disabled, so every frozen task judges exactly as before.
            _u1 = acc.get("ulp1_elementwise") or {}
            def _forgive(limit_abs):
                fok, info = ulp1_forgive(d, rr, limit_abs, float(_u1.get("max_frac", 0.001)))
                st.update(info)
                return fok, info["ulp1_over"]
            if _u1.get("enabled") and not mag_ok and label not in acc.get("ulp_labels", []):
                mag_ok, _ = _forgive(lim)
            if label in acc.get("ulp_labels", []):
                # Magnitude in bf16 ulps of the (bf16-rounded) reference; see acceptance.ulp_labels.
                ulp = torch.where(rr != 0,
                                  torch.exp2(torch.floor(torch.log2(rr.abs().clamp_min(2.0 ** -126))) - 7),
                                  torch.full_like(rr, 2.0 ** -133))
                st["max_ulp_bf16"] = float((d.abs() / ulp).max())
                lim_ulp = max(base[label]["max_ulp_bf16"] * margin,
                              float(acc.get("ulp_floor", {}).get(label, 0.0)))
                st["limit_ulp"] = lim_ulp
                mag_ok = st["max_ulp_bf16"] <= lim_ulp
                if _u1.get("enabled") and not mag_ok:
                    # the same allowance in ulp units: elements over the ulp limit but <= 1 ulp, capped
                    mag_ok, _ = _forgive(lim_ulp * bf16_ulp(rr))
            # Significance guard (acceptance.sign_min_z): a few one-sided disagreements on a near-exact
            # output are not a bias. Absent from older metas -> 0.0, i.e. the old behaviour.
            sign_sig = abs(npos - nz / 2.0) > acc.get("sign_min_z", 0.0) * (nz ** 0.5) / 2.0
            st["sign_nonzero"] = nz
            sign_ok = abs(st["sign_balance"] - 0.5) <= sign_tol or not sign_sig or \
                abs(base[label]["sign_balance"] - 0.5) > sign_tol
            # A MULTIPLICATIVE bias survives both checks above -- proven by a negative test -- so the
            # error-vs-value correlation is checked against the original's own.
            scale_ok = abs(st["scale_bias"]) <= max(scale_tol,
                                                    abs(base[label].get("scale_bias", 0.0)) * margin)
            offset_ok = abs(st["offset_bias"]) <= max(offset_tol,
                                                      abs(base[label].get("offset_bias", 0.0)) * margin)
            # BITWISE vs the frozen golden (= the original's own output), acceptance.exact_labels.
            exact_ok = True
            if label in acc.get("exact_labels", []):
                gi = META["signature"]["outputs"].index(label)
                gt = golden[gi] if golden is not None and gi < len(golden) else None
                if gt is None:
                    exact_ok = False
                    st["exact_mismatch"] = -1
                else:
                    av = actual.detach()
                    gv = gt.to(av.device)
                    if av.dtype != gv.dtype or tuple(av.shape) != tuple(gv.shape):
                        exact_ok = False
                        st["exact_mismatch"] = -2
                        st["exact_note"] = f"dtype/shape {av.dtype}{tuple(av.shape)} vs golden {gv.dtype}{tuple(gv.shape)}"
                    else:
                        an, gn = torch.isnan(av), torch.isnan(gv)
                        mism = ((av != gv) & ~(an & gn)) | (an ^ gn)
                        st["exact_mismatch"] = int(mism.sum())
                        st["exact_max_abs"] = float((av.float() - gv.float()).abs().nan_to_num(0).max())
                        exact_ok = st["exact_mismatch"] == 0
            good = mag_ok and sign_ok and scale_ok and offset_ok and exact_ok
            ok = ok and good
            report[label] = {**st, "limit": lim, "magnitude_ok": mag_ok, "sign_ok": sign_ok,
                             "scale_ok": scale_ok, "offset_ok": offset_ok, "exact_ok": exact_ok,
                             "base_max_abs": base[label]["max_abs"]}
            flags = ''.join(x for x, bad in ((' magnitude', not mag_ok), (' sign', not sign_ok),
                                             (' SCALE-BIAS', not scale_ok),
                                             (' offset-bias', not offset_ok),
                                             (f" NOT-BITWISE({st.get('exact_mismatch')} elems)",
                                              not exact_ok)) if bad)
            if st.get("ulp1_forgiven"):
                print(f"  [L0]{tag} {label}: {st['ulp1_forgiven']} element(s) over the limit, each <= 1 ulp "
                      f"of its reference (cap {st['ulp1_cap']}): faithful rounding, forgiven (4.52)")
            mag_txt = (f"max_ulp={st['max_ulp_bf16']:.3g} (limit {st['limit_ulp']:.3g} ulp, original "
                       f"{base[label]['max_ulp_bf16']:.3g}) max_abs={st['max_abs']:.4g}"
                       if "limit_ulp" in st else
                       f"max_abs={st['max_abs']:.4g} (limit {lim:.4g}, original "
                       f"{base[label]['max_abs']:.4g})")
            ex_txt = (f" bitwise_mismatch={st['exact_mismatch']}" if "exact_mismatch" in st else "")
            print(f"  [L0]{tag} {label}: {mag_txt} sign={st['sign_balance']:.3f} "
                  f"scale_bias={st['scale_bias']:+.2e}{ex_txt} -> {'PASS' if good else 'FAIL' + flags}")

        return ok, report

    ok, report = _l0(cases[0]["args"], META.get("base_error", {}), golden=cases[0].get("ref"))
    # Holdouts GATE (meta.holdouts): the same op at other call sites / ranks / shapes, each against its
    # own original error. One call site cannot see every defect (see build_task_dir.build).
    for hd in META.get("holdouts", []):
        hcases = list(h.iter_eager_cases_from_oracle(os.path.join(HERE, hd["file"]), device="cuda"))
        hok, hrep = _l0(hcases[0]["args"], hd["base_error"], tag=f"[holdout {hd['tag']}]",
                        golden=hcases[0].get("ref"))
        ok = ok and hok
        report[f"holdout:{hd['tag']}"] = hrep

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
    ap.add_argument("--ulp-floor", default=None,
                    help="JSON dict label -> floor in bf16 ulps (see acceptance.ulp_floor)")
    ap.add_argument("--abs-floor-max-ulp", default=None,
                    help="JSON dict label -> k (see acceptance.abs_floor_max_ulp)")
    ap.add_argument("--exact-labels", default="",
                    help="comma list of outputs that must be BITWISE equal to the original's "
                         "(see acceptance.exact_labels)")
    ap.add_argument("--holdout", action="append", default=[],
                    help="TAG=capture.pt; repeatable. Gated holdout cases (see build(holdouts=...))")
    a = ap.parse_args()
    print(json.dumps(build(a.capture, a.task_dir, a.reference, a.name,
                       json.loads(a.const) if a.const else None,
                       json.loads(open(a.notes[1:]).read() if a.notes.startswith("@") else a.notes)
                       if a.notes else None,
                       json.loads(open(a.engagement[1:]).read() if a.engagement.startswith("@")
                                  else a.engagement) if a.engagement else None,
                       [s for s in a.ulp_labels.split(",") if s.strip()],
                       [tuple(x.split("=", 1)) for x in a.holdout],
                       json.loads(a.ulp_floor) if a.ulp_floor else None,
                       json.loads(a.abs_floor_max_ulp) if a.abs_floor_max_ulp else None,
                       [s for s in a.exact_labels.split(",") if s.strip()]), indent=2))
