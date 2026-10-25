"""main_grad_contract.py -- L0 companion for GEMM overlays: is the fp32 main_grad accumulation right?

WHY (DECISIONS 4.50, audit G5). Under Megatron's gradient_accumulation_fusion / TE's
fuse_wgrad_accumulation the weight gradient never reaches autograd: the GEMM ADDS it into a persistent
fp32 `main_grad` across micro-batches (and OVERWRITES it on a run's first micro-batch). The L0 oracle
sees one call's wgrad and cannot see that path at all; and the oracle-style magnitude limit, applied to
the final main_grad, is blind to a bf16-ROUNDED wgrad added in fp32 (it passed on fc1, qkv and lmhead).
This runs the op twice per step through the real TE / Megatron module with DDP's post-hook emulated,
for the incumbent, the candidate and three deliberately wrong accumulators, and judges:
  * oracle-style: final main_grad vs an fp32 reference, the L0 rule (x1.25 the incumbent's error,
    sign / scale / offset bias), steady (S1) and run-start overwrite (S2) scenarios;
  * CONTRACT, independent of GEMM numerics: linearity in the initial main_grad (a correct fp32 `+=`
    gives final(init) - init == final(0) to a few fp32 ulps) and the bf16-exact fraction of the
    per-micro-batch increment (an fp32-output GEMM leaves ~none exactly bf16-representable).
PASS iff the candidate passes both AND every control fails both-combined (else the run could not
discriminate, and it is reported as not passed). Promoted from scratch/oracle_gaps_1022 (g5_harness).

  main_grad_contract.py --task <L0 task dir> --impl <overlay impl.py> [--attr name] --out result.json
Runs inside rocm/primus (TE + Megatron on PYTHONPATH), one GPU. exit 0 = passed or not applicable;
1 = not passed (incl. an unmapped site that touches main_grad: applicable, no head -- mgc_applicability.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True

import torch  # noqa: E402  (torch BEFORE transformer_engine: LLVM option clash otherwise)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgc_applicability import applicability  # noqa: E402

MARGIN, SIGN_TOL, SCALE_TOL, OFFSET_TOL = 1.25, 0.15, 0.002, 0.01
DEV = "cuda"


def log(*a):
    print(f"[main_grad_contract {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_case(task_dir):
    h = load_file_module("_mgc_harness_lib", os.path.join(task_dir, "harness_lib.py"))
    meta = json.load(open(os.path.join(task_dir, "meta.json")))
    case = next(iter(h.iter_eager_cases_from_oracle(os.path.join(task_dir, "reference_io.pt"),
                                                     device="cpu")))
    return meta, case["args"]


def roll_tok(t, k):
    return torch.roll(t, shifts=k, dims=0).contiguous()


# ------------------------------------------------------------------------------------------------
# statistics: exactly the L0 oracle's (unittest.py) error stats and acceptance rule
# ------------------------------------------------------------------------------------------------
def err_stats(actual, ref):
    a = actual.detach().double().flatten()
    rr = ref.detach().double().flatten()
    d = a - rr
    nz = int((d != 0).sum())
    denom = rr.abs().mean().clamp_min(1e-30)
    rr2 = float((rr * rr).sum())
    return {"max_abs": float(d.abs().max()),
            "rel_mean_abs": float(d.abs().mean() / denom),
            "sign_balance": (float((d > 0).sum()) / nz) if nz else 0.5,
            "scale_bias": float((d * rr).sum() / rr2) if rr2 > 0 else 0.0,
            "offset_bias": float(d.mean() / denom),
            "nonzero_frac": nz / d.numel(),
            "ref_abs_max": float(rr.abs().max())}


def verdict(st, base):
    lim = base["max_abs"] * MARGIN
    mag_ok = st["max_abs"] <= lim
    sign_ok = abs(st["sign_balance"] - 0.5) <= SIGN_TOL or abs(base["sign_balance"] - 0.5) > SIGN_TOL
    scale_ok = abs(st["scale_bias"]) <= max(SCALE_TOL, abs(base["scale_bias"]) * MARGIN)
    offset_ok = abs(st["offset_bias"]) <= max(OFFSET_TOL, abs(base["offset_bias"]) * MARGIN)
    return {"limit_max_abs": lim, "magnitude_ok": mag_ok, "sign_ok": sign_ok, "scale_ok": scale_ok,
            "offset_ok": offset_ok, "pass": bool(mag_ok and sign_ok and scale_ok and offset_ok)}


# ------------------------------------------------------------------------------------------------
# DDP emulation
# ------------------------------------------------------------------------------------------------
def attach_main_grad(params, inits):
    for n, p in params.items():
        p.main_grad = inits[n].clone()
        p.grad_added_to_main_grad = False
        p.grad = None


def ddp_hook(params, rec):
    for n, p in params.items():
        g = p.grad
        entry = {"grad_returned": None if g is None else [list(g.shape), str(g.dtype)],
                 "grad_added_to_main_grad": bool(getattr(p, "grad_added_to_main_grad", False))}
        added = False
        if g is not None and (not p.grad_added_to_main_grad or getattr(p, "zero_out_wgrad", False)):
            p.main_grad.add_(g.data)
            added = True
        entry["hook_added_grad"] = added
        p.grad = None
        rec.setdefault(n, []).append(entry)


# ------------------------------------------------------------------------------------------------
# controls: deliberately wrong accumulation wrapped around the overlay's microbatch
# ------------------------------------------------------------------------------------------------
class Control:
    """mode: 'overwrite' (main_grad = this microbatch's wgrad), 'bf16_accum' (accumulator held in
    bf16), 'bf16_wgrad' (per-microbatch wgrad rounded to bf16, then fp32 add)."""

    def __init__(self, mode, names):
        self.mode, self.names = mode, names

    def run(self, params, mb_fn):
        saved = {}
        for n in self.names:
            p = params[n]
            saved[n] = p.main_grad
            p.main_grad = torch.zeros_like(saved[n])
        mb_fn()
        for n in self.names:
            p = params[n]
            w = p.main_grad
            mg = saved[n]
            if self.mode == "overwrite":
                mg.copy_(w)
            elif self.mode == "bf16_accum":
                mg.copy_((mg.to(torch.bfloat16) + w.to(torch.bfloat16)).float())
            elif self.mode == "bf16_wgrad":
                mg.add_(w.to(torch.bfloat16).float())
            else:
                raise ValueError(self.mode)
            p.main_grad = mg


# ------------------------------------------------------------------------------------------------
# heads
# ------------------------------------------------------------------------------------------------
class TEHeadBase:
    gate_params = ()

    def run_step(self, ifms, ctl=None, inits=None, snap=None):
        """Two microbatches through the CURRENT binding; returns per-param final main_grad + records.
        snap: optional list; receives {param: main_grad clone} after each microbatch (gate params)."""
        attach_main_grad(self.params, self.inits if inits is None else inits)
        rec = {}
        for i, ifm in enumerate(ifms):
            def mb(i=i, ifm=ifm):
                self.fwd_bwd(i, ifm)
            if ctl is None:
                mb()
            else:
                ctl.run(self.params, mb)
            ddp_hook(self.params, rec)
            if snap is not None:
                torch.cuda.synchronize()
                snap.append({n: self.params[n].main_grad.clone() for n in self.gate_params})
        torch.cuda.synchronize()
        return {n: p.main_grad.clone() for n, p in self.params.items()}, rec


def _te():
    import transformer_engine.pytorch as te
    return te


class LNLinearHead(TEHeadBase):
    """TE LayerNormLinear (RMSNorm + GEMM, optional bias) with fuse_wgrad_accumulation, as Megatron runs
    fc1 and qkv. The overlay rebinds `_LayerNormLinear.apply`."""
    gate_params = ("weight",)

    def __init__(self, task_dir):
        meta, args = load_case(task_dir)
        names = meta["layout"]["args"]
        named = dict(zip(names, args[:-1]))
        self.eps = meta["layout"]["const"]["eps"]
        x, lnw, w, b, go = named["x"], named["ln_weight"], named["weight"], named.get("bias"), args[-1]
        T = x.shape[0]
        self.mbs = [(x.to(DEV), go.to(DEV)), (roll_tok(x, T // 2).to(DEV), roll_tok(go, T // 3).to(DEV))]
        te = _te()
        self.mod = te.LayerNormLinear(w.shape[1], w.shape[0], eps=self.eps, bias=b is not None,
                                      normalization="RMSNorm", fuse_wgrad_accumulation=True,
                                      params_dtype=torch.bfloat16, parallel_mode="column", tp_size=1,
                                      sequence_parallel=False, device=DEV)
        with torch.no_grad():
            self.mod.weight.copy_(w.to(DEV))
            self.mod.layer_norm_weight.copy_(lnw.to(DEV))
            if b is not None:
                self.mod.bias.copy_(b.to(DEV))
        self.params = {"weight": self.mod.weight, "ln_weight": self.mod.layer_norm_weight}
        if b is not None:
            self.params["bias"] = self.mod.bias

    def fwd_bwd(self, i, ifm):
        x, go = self.mbs[i]
        xi = x.detach().clone().requires_grad_(True)
        out = self.mod(xi, is_first_microbatch=ifm)
        out.backward(go)

    def ref_grads(self, ri):
        out = {n: [] for n in self.params}
        for x, go in self.mbs:
            xf = x.float().requires_grad_(True)
            lnw = self.mod.layer_norm_weight.detach().float().requires_grad_(True)
            w = self.mod.weight.detach().float().requires_grad_(True)
            b = self.mod.bias.detach().float().requires_grad_(True) if "bias" in self.params else None
            y = ri.rmsnorm_linear_reference(xf, lnw, w, b, eps=self.eps)
            wrt = [w, lnw] + ([b] if b is not None else [])
            gs = torch.autograd.grad(y, wrt, go.float())
            out["weight"].append(gs[0].detach()); out["ln_weight"].append(gs[1].detach())
            if b is not None:
                out["bias"].append(gs[2].detach())
            del y, xf
        return out

    def install_overlay(self, impl_path, attr):
        from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
        impl = load_file_module("_mgc_impl", impl_path)
        _LayerNormLinear.apply = staticmethod(getattr(impl, attr or "lnlinear_apply"))
        return impl


class LinearHead(TEHeadBase):
    """One or more TE Linear GEMMs (row-parallel, no bias) with fuse_wgrad_accumulation. Layout: pairs
    weight_<s> / x_<s> (or a single weight / x); one grad_output per GEMM, in layout order. The overlay
    rebinds `_Linear.apply`."""

    def __init__(self, task_dir):
        meta, args = load_case(task_dir)
        names = meta["layout"]["args"]
        sfx = [n[len("weight"):] for n in names if n.startswith("weight")]
        nin = len(names)
        named = dict(zip(names, args[:nin]))
        gos = args[nin:]
        if len(gos) != len(sfx):
            raise SystemExit(f"{len(sfx)} GEMMs in the layout but {len(gos)} grad_outputs")
        te = _te()
        self.lin, self.mbs_in = {}, {}
        for k, s in enumerate(sfx):
            w, x = named["weight" + s], named["x" + s]
            T = x.shape[0]
            mod = te.Linear(w.shape[1], w.shape[0], bias=False, fuse_wgrad_accumulation=True,
                            params_dtype=torch.bfloat16, parallel_mode="row", tp_size=1,
                            sequence_parallel=False, device=DEV)
            with torch.no_grad():
                mod.weight.copy_(w.to(DEV))
            key = (s.lstrip("_") or "linear") + ".weight"
            self.lin[key] = mod
            self.mbs_in[key] = [(x.to(DEV), gos[k].to(DEV)),
                                (roll_tok(x, T // 2).to(DEV), roll_tok(gos[k], T // 3).to(DEV))]
        self.params = {k: m.weight for k, m in self.lin.items()}
        self.gate_params = tuple(self.params)

    def fwd_bwd(self, i, ifm):
        outs, gs = [], []
        for key, mod in self.lin.items():
            x, g = self.mbs_in[key][i]
            outs.append(mod(x.detach().clone().requires_grad_(True), is_first_microbatch=ifm)); gs.append(g)
        torch.autograd.backward(outs, gs)

    def ref_grads(self, ri):
        out = {k: [] for k in self.lin}
        for key, mod in self.lin.items():
            for x, g in self.mbs_in[key]:
                w = mod.weight.detach().float().requires_grad_(True)
                y = ri.linear_reference(w, x.float(), None)
                (gw,) = torch.autograd.grad(y, [w], g.float())
                out[key].append(gw.detach())
                del y
        return out

    def install_overlay(self, impl_path, attr):
        from transformer_engine.pytorch.module.linear import _Linear
        impl = load_file_module("_mgc_impl", impl_path)
        _Linear.apply = staticmethod(getattr(impl, attr or "linear_apply"))
        return impl


class LMHead(TEHeadBase):
    """Megatron output layer (ColumnParallelLinear, gradient_accumulation_fusion) + loss, through
    GPTModel._postprocess. The overlay replaces `_postprocess`."""
    gate_params = ("weight",)

    def __init__(self, task_dir):
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29571")
            dist.init_process_group("nccl", rank=0, world_size=1)
        from megatron.core import parallel_state
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(1, 1)
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import tensor_parallel
        from megatron.core.tensor_parallel import layers as _layers
        meta, args = load_case(task_dir)
        names = meta["layout"]["args"]
        named = dict(zip(names, args[:-1]))
        h, w, labels, mask, go = (named["hidden_states"], named["weight"], named["labels"],
                                  named["loss_mask"], args[-1])
        T = h.shape[0]
        self.mbs = [(h.to(DEV), labels.to(DEV), mask.to(DEV), go.to(DEV)),
                    (roll_tok(h, T // 2).to(DEV), labels.to(DEV), mask.to(DEV), go.to(DEV))]
        self.cfg = TransformerConfig(num_layers=1, hidden_size=w.shape[1], num_attention_heads=28,
                                     num_query_groups=4, ffn_hidden_size=18944, bf16=True,
                                     params_dtype=torch.bfloat16, gradient_accumulation_fusion=True,
                                     perform_initialization=False, cross_entropy_loss_fusion=False)
        assert _layers._grad_accum_fusion_available, "apex fused_weight_gradient_mlp_cuda unavailable"
        self.ol = tensor_parallel.ColumnParallelLinear(
            w.shape[1], w.shape[0], config=self.cfg, init_method=lambda t: None, bias=False,
            skip_bias_add=False, gather_output=False).to(DEV)
        with torch.no_grad():
            self.ol.weight.copy_(w.to(DEV))
        assert self.ol.gradient_accumulation_fusion is True
        self.params = {"weight": self.ol.weight}
        from megatron.core.models.gpt.gpt_model import GPTModel
        fake = types.SimpleNamespace(
            post_process=True, pre_process=False, training=True, config=self.cfg,
            share_embeddings_and_output_weights=False, output_layer=self.ol, mtp=None,
            parallel_output=True)
        fake._scale_logits = types.MethodType(GPTModel._scale_logits, fake)
        fake.compute_language_model_loss = types.MethodType(GPTModel.compute_language_model_loss, fake)
        self.fake = fake
        self.GPTModel = GPTModel
        self.postprocess = GPTModel._postprocess  # incumbent (original) until install_overlay
        self.overlay = False

    def fwd_bwd(self, i, ifm):
        h, labels, mask, go = self.mbs[i]
        hi = h.detach().clone().requires_grad_(True)
        kw = dict(hidden_states=hi, input_ids=None, position_ids=None, labels=labels,
                  rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None, loss_mask=mask)
        out = self.postprocess(self.fake, **kw)
        if not self.overlay:
            out = out * mask      # the captured op is `_postprocess(...) * loss_mask` (task capture_meta)
        out.backward(go)

    def ref_grads(self, ri):
        out = {"weight": []}
        for h, labels, mask, go in self.mbs:
            w = self.ol.weight.detach().float().requires_grad_(True)
            y = ri.lmhead_ce_reference(h.float(), w, labels, mask)
            (gw,) = torch.autograd.grad(y, [w], go.float())
            out["weight"].append(gw.detach())
            del y
            torch.cuda.empty_cache()
        return out

    def install_overlay(self, impl_path, attr):
        impl = load_file_module("_mgc_impl", impl_path)
        self.postprocess = getattr(impl, attr or "postprocess")
        self.overlay = True
        return impl


def make_inits(params, ref):
    """Known non-zero fp32 init (full mantissa), scaled to the std of microbatch-1's fp32 wgrad."""
    g = torch.Generator(device=DEV)
    g.manual_seed(20261022)
    inits = {}
    for n, p in params.items():
        s = float(ref[n][0].float().std())
        inits[n] = (torch.randn(p.shape, generator=g, device=DEV, dtype=torch.float32) * s).contiguous()
    return inits


def _ulp32(v):
    a = v.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(a)) - 23)


def _bf16_exact_frac_nz(t):
    nz = t != 0
    if int(nz.sum()) == 0:
        return 0.0
    return float((t[nz] == t[nz].to(torch.bfloat16).float()).float().mean())


def contract(H, ifms, ctl=None):
    """GEMM-numerics-independent checks of the accumulation contract on the gate params.
    Z = all main_grads start at 0; I = start at the known non-zero init. A correct fp32 `+=`
    accumulator gives (final_I - init) == final_Z up to fp32 rounding of the two additions
    (a few fp32 ulps), provided the kernel is run-to-run deterministic (checked: Z twice).
    fp32 OUTPUT: in the Z run, main_grad after microbatch 1 IS the per-microbatch wgrad; an fp32-output
    GEMM leaves almost no non-zero element exactly bf16-representable, a bf16-output one leaves all."""
    zeros = {n: torch.zeros_like(v) for n, v in H.inits.items()}
    snapZ, snapZ2 = [], []
    finZ, _ = H.run_step(ifms, ctl, inits=zeros, snap=snapZ)
    finZ2, _ = H.run_step(ifms, ctl, inits=zeros, snap=snapZ2)
    finI, _ = H.run_step(ifms, ctl)
    out = {}
    for n in H.gate_params:
        lin = (finI[n].double() - H.inits[n].double()) - finZ[n].double()
        scale = (H.inits[n].abs() + finZ[n].abs()).float()
        out[n] = {
            "determinism_Z_runs_max_abs": float((finZ[n] - finZ2[n]).abs().max()),
            "linearity_max_abs": float(lin.abs().max()),
            "linearity_max_ulp32": float((lin.abs() / _ulp32(scale).double()).max()),
            "linearity_rel_to_final_absmax": float(lin.abs().max() / finZ[n].abs().max().clamp_min(1e-30)),
            "mb1_bf16_exact_frac_nonzero": _bf16_exact_frac_nz(snapZ[0][n]),
            "mb2_bf16_exact_frac_nonzero": _bf16_exact_frac_nz(snapZ[1][n]),
            "mb2_increment_bf16_exact_frac_nonzero": _bf16_exact_frac_nz(snapZ[1][n] - snapZ[0][n]),
        }
    del finZ, finZ2, finI, snapZ, snapZ2
    torch.cuda.empty_cache()
    return out


def contract_verdict(c, base):
    lin_lim = max(base["linearity_max_ulp32"] * MARGIN, 2.0)
    lin_ok = c["linearity_max_ulp32"] <= lin_lim
    v = {"linearity_limit_ulp32": lin_lim, "linearity_ok": lin_ok}
    ok = lin_ok
    for k in ("mb1_bf16_exact_frac_nonzero", "mb2_bf16_exact_frac_nonzero"):
        lim = max(base[k] * MARGIN, base[k] + 0.01)
        v[f"{k}_limit"] = lim
        v[f"{k}_ok"] = c[k] <= lim
        ok = ok and v[f"{k}_ok"]
    v["pass"] = bool(ok)
    return v


def head_for(meta):
    """Which accumulation path the task's op runs, from its rebind target (mgc_applicability.HEADS).
    None = no head models this site; whether the contract then APPLIES is mgc_applicability's call."""
    name = applicability(meta, "")[0]
    return {"LNLinearHead": LNLinearHead, "LinearHead": LinearHead, "LMHead": LMHead}.get(name)


CONTROLS = ("overwrite", "bf16_accum", "bf16_wgrad")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, help="the L0 task dir (reference_io.pt, meta.json, harness_lib.py, baseline_src/)")
    ap.add_argument("--impl", required=True, help="the candidate overlay's impl.py")
    ap.add_argument("--attr", default=None, help="replacement attribute in impl.py (default per head)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.task, "meta.json")))
    Head = head_for(meta)
    res = {"task": os.path.abspath(a.task), "impl": os.path.abspath(a.impl), "torch": torch.__version__}
    if Head is None:
        _, appl, ev = applicability(meta, open(a.impl, errors="replace").read())
        if appl:
            # Audit 2026-10-25 G-1: a composite / unmapped site that touches the fused-wgrad path is NOT
            # "not applicable" -- that read as a pass. Fail closed until a head models it.
            res.update(applicable=True, passed=False, head=None, evidence=ev,
                       why=f"target {meta.get('target_callable')!r} touches the fused-wgrad / main_grad path "
                           f"({', '.join(ev)}) but no head models this site: write a head or decompose the overlay")
            json.dump(res, open(a.out, "w"), indent=1)
            log("applicable, no head:", res["why"])
            return 1
        res.update(applicable=False, passed=None, evidence=[],
                   why=f"target {meta.get('target_callable')!r} is not a fused-wgrad GEMM site and the overlay "
                       f"code does not touch main_grad")
        json.dump(res, open(a.out, "w"), indent=1)
        log("not applicable:", res["why"])
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    H = Head(a.task)
    ri = load_file_module("_mgc_ref_impl", os.path.join(a.task, "baseline_src", "reference_impl.py"))
    rg = H.ref_grads(ri)
    H.inits = make_inits(H.params, rg)
    refs = {"S1": {n: H.inits[n] + rg[n][0] + rg[n][1] for n in H.inits},
            # S2 (is_first_microbatch True then False): TE OVERWRITES the fused GEMM weights' main_grad on
            # the first microbatch; other params are returned to autograd and ADDED by the DDP hook.
            "S2": {n: (rg[n][0] + rg[n][1]) if n in H.gate_params else (H.inits[n] + rg[n][0] + rg[n][1])
                   for n in H.inits}}
    del rg
    torch.cuda.empty_cache()
    scen = {"S1": (False, False), "S2": (True, False)} if Head is not LMHead else {"S1": (None, None)}
    finals, contracts = {}, {}
    for s, ifms in scen.items():
        finals[("incumbent", s)], _ = H.run_step(ifms)
    contracts["incumbent"] = contract(H, scen["S1"])
    H.install_overlay(a.impl, a.attr)
    for s, ifms in scen.items():
        finals[("candidate", s)], _ = H.run_step(ifms)
    contracts["candidate"] = contract(H, scen["S1"])
    for mode in CONTROLS:
        finals[(f"ctl_{mode}", "S1")], _ = H.run_step(scen["S1"], ctl=Control(mode, list(H.gate_params)))
        contracts[f"ctl_{mode}"] = contract(H, scen["S1"], ctl=Control(mode, list(H.gate_params)))
    oracle = {}
    for (path, s), fin in finals.items():
        if path == "incumbent":
            continue
        base = finals[("incumbent", s)]
        oracle[f"{path}/{s}"] = {n: verdict(err_stats(fin[n], refs[s][n]), err_stats(base[n], refs[s][n]))
                                 for n in H.gate_params}
    contract_v = {p: {n: contract_verdict(contracts[p][n], contracts["incumbent"][n]) for n in H.gate_params}
                  for p in contracts if p != "incumbent"}
    ok = lambda path: (all(all(v["pass"] for v in oracle[k].values()) for k in oracle if k.startswith(path + "/"))  # noqa: E731
                       and all(v["pass"] for v in contract_v[path].values()))
    combined = {p: bool(ok(p)) for p in contract_v}
    # SELF-CHECK. Every deliberately wrong accumulator must FAIL, or this run could not have told a
    # broken candidate from a correct one (bf16-rounded wgrad passes the oracle-style limit on fc1,
    # qkv and lmhead -- only the contract catches it, DECISIONS 4.50).
    discriminating = not any(combined[f"ctl_{m}"] for m in CONTROLS)
    res.update(applicable=True, head=Head.__name__, gate_params=list(H.gate_params),
               oracle_style=oracle, contract={p: {n: {**contracts[p][n], **contract_v[p][n]}
                                                  for n in H.gate_params} for p in contract_v},
               contract_incumbent=contracts["incumbent"], combined=combined,
               controls_all_fail=discriminating,
               passed=bool(combined["candidate"] and discriminating),
               elapsed_s=round(time.time() - t0, 1))
    if not discriminating:
        res["why"] = "a deliberately wrong accumulator PASSED: the check cannot discriminate on this task"
    elif not combined["candidate"]:
        res["why"] = "the candidate's fp32 main_grad accumulation differs from the incumbent's contract"
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    log(f"{Head.__name__}: candidate {'PASS' if combined['candidate'] else 'FAIL'}; controls "
        + ", ".join(f"{m} {'PASS(!)' if combined['ctl_' + m] else 'FAIL'}" for m in CONTROLS)
        + f" -> {'PASSED' if res['passed'] else 'NOT PASSED'}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
