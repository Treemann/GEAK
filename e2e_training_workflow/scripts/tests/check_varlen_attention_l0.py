import json, torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
torch.manual_seed(0)
H, KV, D = 28, 4, 128                     # Qwen2.5-7B attention geometry (GQA 28/4, d128)
lens = [688, 1499, 422, 1269, 217]        # pack 0's real segment lengths
T = sum(lens)
q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(T, KV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(T, KV, D, device="cuda", dtype=torch.bfloat16)
go = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
def grads(fn):
    qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = fn(qq, kk, vv); o.backward(go)
    return o.detach(), qq.grad, kk.grad, vv.grad
cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), device="cuda", dtype=torch.int32)
def varlen(qq, kk, vv):
    return flash_attn_varlen_func(qq, kk, vv, cu, cu, max(lens), max(lens), causal=True)
def per_segment(qq, kk, vv):
    outs, s = [], 0
    for n in lens:
        outs.append(flash_attn_func(qq[None, s:s+n], kk[None, s:s+n], vv[None, s:s+n], causal=True)[0]); s += n
    return torch.cat(outs)
def ref_fp32(qq, kk, vv):                 # independent fp32 reference, per segment
    outs, s = [], 0
    for n in lens:
        Q = qq[s:s+n].float().transpose(0, 1); K = kk[s:s+n].float().repeat_interleave(H // KV, 1).transpose(0, 1)
        V = vv[s:s+n].float().repeat_interleave(H // KV, 1).transpose(0, 1)
        outs.append(torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True).transpose(0, 1).to(qq.dtype)); s += n
    return torch.cat(outs)
A, B, R = grads(varlen), grads(per_segment), grads(ref_fp32)
rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm())
names = ("out", "dq", "dk", "dv")
print(json.dumps({
  "varlen_vs_fp32_ref": {n: round(rel(a, r), 5) for n, a, r in zip(names, A, R)},
  "per_segment_vs_fp32_ref": {n: round(rel(b, r), 5) for n, b, r in zip(names, B, R)},
  "varlen_vs_per_segment": {n: round(rel(a, b), 5) for n, a, b in zip(names, A, B)},
  "grad_norms": {"varlen": [round(float(t.float().norm()), 2) for t in A[1:]],
                 "per_segment": [round(float(t.float().norm()), 2) for t in B[1:]],
                 "fp32_ref": [round(float(t.float().norm()), 2) for t in R[1:]]}}, indent=1))
