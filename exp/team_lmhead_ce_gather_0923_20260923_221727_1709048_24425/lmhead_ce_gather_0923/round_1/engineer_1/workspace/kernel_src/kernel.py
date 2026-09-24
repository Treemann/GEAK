"""Gathered-bf16 LM-head + CE, fwd+bwd, with the GEMM selection of lane r1_d1.

SCOPE. This lane owns the backward GEMMs (wgrad, and dgrad's split-over-V). The gather, the CE
epilogue and the allocation strategy are D0's; what is here is the MINIMAL scaffolding needed to
measure those GEMMs in situ, written to be the same eager gather-bf16 shape as
`controls_bf16.py:gather_bf16` so D0's patch can replace it wholesale. The lane's real contribution
is `wgrad_gemm.py` -- the GEMM selection and the measured table that justifies it.

Contract facts encoded here (COMMANDMENT sec.2):
* differentiable leaves are bf16 -> `backward` RETURNS (dgrad bf16, wgrad bf16, None, None);
* labels/loss_mask are [b, s] while the logits are [s, b, V] -- the transposes are written even
  though they are invisible at MBS=1;
* cost is linear in nnz(loss_mask); 1276 / 31.15% appear nowhere. nnz is only rounded UP to the
  next multiple of `_PAD` so the tuned-GEMM key set stays bounded (see wgrad_gemm.py).
"""
import torch
import torch.nn.functional as F

from wgrad_gemm import GEMM_M_BUCKET, dgrad_gemm, enable_tuned_gemm, fwd_gemm, wgrad_gemm

enable_tuned_gemm()


class _LMHeadCEGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_states, weight, labels, loss_mask):
        s, b, hsz = hidden_states.shape
        n = s * b
        h2 = hidden_states.reshape(n, hsz)
        if h2.dtype != torch.bfloat16:
            h2 = h2.to(torch.bfloat16)
        w = weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)

        # [b, s] -> [s, b] -> [s*b]. A no-op at b=1 and required at b>1.
        m = loss_mask.float().transpose(0, 1).reshape(n)
        idx = torch.nonzero(m, as_tuple=True)[0]              # [nnz], data-dependent
        nnz = idx.numel()
        pad = (-nnz) % GEMM_M_BUCKET                          # <= 127 inert rows, never 1276

        mg = m.index_select(0, idx)                           # [nnz]
        hg = h2.index_select(0, idx)                          # [nnz, H] bf16
        tgt = labels.transpose(0, 1).reshape(n).index_select(0, idx)
        if pad:
            # Zero rows: their grad_output scale is 0, so their dlogits rows are exactly 0 and
            # contribute nothing to wgrad; their dgrad rows are simply not scattered back.
            hg = F.pad(hg, (0, 0, 0, pad))
            mg = F.pad(mg, (0, pad))
            tgt = F.pad(tgt, (0, pad), value=-1)              # -1 => treated as out-of-range
        hg = hg.contiguous()

        logits = fwd_gemm(hg, w)                              # [nnz+pad, V] bf16
        vocab = logits.shape[-1]
        bad = (tgt < 0) | (tgt >= vocab)
        safe = torch.where(bad, torch.zeros_like(tgt), tgt)

        lf = logits.float()
        lse = torch.logsumexp(lf, dim=-1)
        pred = lf.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        pred = torch.where(bad, torch.zeros_like(pred), pred)
        loss_g = (lse - pred) * mg

        full = torch.zeros(n, dtype=loss_g.dtype, device=loss_g.device)
        full.index_copy_(0, idx, loss_g[:nnz])
        out = full.reshape(s, b).transpose(0, 1).contiguous().reshape(loss_mask.shape)

        ctx.save_for_backward(logits, lse, hg, w, idx, safe, bad, mg)
        ctx.shapes = (s, b, hsz, n, nnz)
        ctx.h_dtype = hidden_states.dtype
        ctx.w_dtype = weight.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        logits, lse, hg, w, idx, safe, bad, mg = ctx.saved_tensors
        s, b, hsz, n, nnz = ctx.shapes
        pad = mg.numel() - nnz

        # grad_out is [b, s] -> [s, b] -> [s*b], gathered to the live rows.
        g = grad_out.float().transpose(0, 1).reshape(n).index_select(0, idx)
        if pad:
            g = F.pad(g, (0, pad))                            # padded rows carry zero gradient
        g = g * mg

        # d(lse - pred)/dlogits = softmax(logits) - onehot(target), scaled by g.
        dlogits = torch.exp(logits.float() - lse.unsqueeze(-1))
        dlogits.mul_(g.unsqueeze(-1))
        # out-of-range targets contribute no one-hot term (their predicted logit was zeroed).
        sub = torch.where(bad, torch.zeros_like(g), g)
        dlogits.scatter_add_(-1, safe.unsqueeze(-1), (-sub).unsqueeze(-1))
        dlogits = dlogits.to(torch.bfloat16)                  # [nnz+pad, V] bf16

        dhg = dgrad_gemm(dlogits, w)                          # [nnz+pad, H] bf16
        dgrad = torch.zeros(n, hsz, dtype=torch.bfloat16, device=dhg.device)
        dgrad.index_copy_(0, idx, dhg[:nnz])
        dgrad = dgrad.reshape(s, b, hsz).to(ctx.h_dtype)

        wgrad = wgrad_gemm(dlogits, hg).to(ctx.w_dtype)       # [V, H] bf16
        return dgrad, wgrad, None, None


def entry(hidden_states, weight, labels, loss_mask):
    return _LMHeadCEGather.apply(hidden_states, weight, labels, loss_mask)
