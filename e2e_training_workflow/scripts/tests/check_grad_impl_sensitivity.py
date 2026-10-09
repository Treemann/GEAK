import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "/models/qwen2_5_math_7b"; tok = AutoTokenizer.from_pretrained(M)
pack = json.loads(open("/work/data/sft_ultrachat20k_packed_7b.jsonl").readline())["messages"]
convs, cur = [], []
for m in pack:
    if m["role"] == "system" and cur: convs.append(cur); cur = []
    cur.append(m)
convs.append(cur)
ids = [tok.apply_chat_template(c, tokenize=True) for c in convs]
n = sum(len(x) for x in ids) - 1
ce = torch.nn.functional.cross_entropy
out = {}
for impl in ("flash_attention_2", "sdpa", "eager"):
    model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32, attn_implementation=impl).cuda()
    model.zero_grad(set_to_none=True); tot = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, x in enumerate(ids):                       # ISOLATED, one conversation at a time
            t = torch.tensor([x]).cuda(); lg = model(t).logits[0].float()
            l = ce(lg[:-1], t[0, 1:], reduction="sum")
            if i + 1 < len(ids): l = l + ce(lg[-1:], torch.tensor([ids[i + 1][0]]).cuda(), reduction="sum")
            tot += l.item(); (l / n).backward()
    g = torch.sqrt(sum((q.grad.float() ** 2).sum() for q in model.parameters() if q.grad is not None)).item()
    gne = torch.sqrt(sum((q.grad.float() ** 2).sum() for nm, q in model.named_parameters() if q.grad is not None and "embed_tokens" not in nm)).item()
    # where does the norm live? top parameter groups
    per = sorted(((float(q.grad.float().norm()), nm) for nm, q in model.named_parameters() if q.grad is not None), reverse=True)[:3]
    out[impl] = {"isolated_loss": round(tot / n, 5), "isolated_grad_norm": round(g, 4), "non_embedding_grad_norm": round(gne, 4),
                 "top_params": [(nm.replace("model.", ""), round(v, 2)) for v, nm in per]}
    del model; torch.cuda.empty_cache()
print(json.dumps(out, indent=1))
