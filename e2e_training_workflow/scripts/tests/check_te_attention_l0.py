import json, os, torch
import transformer_engine.pytorch as te
torch.manual_seed(0)
H, KV, D = 28, 4, 128
lens = [int(x) for x in os.environ.get("LENS", "688,1499,422,1269,217").split(",")]
T = sum(lens)
cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), device="cuda", dtype=torch.int32)
q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(T, KV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(T, KV, D, device="cuda", dtype=torch.bfloat16)
go = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
attn = te.DotProductAttention(H, D, num_gqa_groups=KV, attention_dropout=0.0, qkv_format="thd",
                              attn_mask_type="padding_causal").cuda()
qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
o = attn(qq, kk, vv, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
o = o.view(T, H, D); o.backward(go)
def ref():
    Q, K, V = (t.detach().clone().float().requires_grad_(True) for t in (q, k, v))
    outs, s = [], 0
    for n in lens:
        a = Q[s:s+n].transpose(0, 1); b = K[s:s+n].repeat_interleave(H // KV, 1).transpose(0, 1)
        c = V[s:s+n].repeat_interleave(H // KV, 1).transpose(0, 1)
        outs.append(torch.nn.functional.scaled_dot_product_attention(a, b, c, is_causal=True).transpose(0, 1)); s += n
    O = torch.cat(outs); O.backward(go.float()); return O.detach(), Q.grad, K.grad, V.grad
R = ref()
rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm())
print("TEL0 " + json.dumps({"backend": os.environ.get("BACKEND"), "lens": lens,
  "rel_err_vs_fp32": {n: round(rel(a, r), 5) for n, a, r in zip(("out", "dq", "dk", "dv"), (o.detach(), qq.grad, kk.grad, vv.grad), R)},
  "grad_norm_ratio_vs_fp32": {n: round(float(a.float().norm() / r.float().norm()), 4) for n, a, r in zip(("dq", "dk", "dv"), (qq.grad, kk.grad, vv.grad), R[1:])}}))
