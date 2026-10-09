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
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
ce = torch.nn.functional.cross_entropy
def run(with_boundary):
    n = sum(len(x) - 1 for x in ids) + (len(ids) - 1 if with_boundary else 0)
    model.zero_grad(set_to_none=True); tot = 0.0; bce = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, x in enumerate(ids):
            t = torch.tensor([x]).cuda(); lg = model(t).logits[0].float()
            l = ce(lg[:-1], t[0, 1:], reduction="sum")
            if with_boundary and i + 1 < len(ids):
                b = ce(lg[-1:], torch.tensor([ids[i + 1][0]]).cuda(), reduction="sum"); bce.append(round(b.item(), 3)); l = l + b
            tot += l.item(); (l / n).backward()
    g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    return tot / n, g, bce
lw, gw, bce = run(True); lo, go, _ = run(False)
print(json.dumps({"with_boundary": {"loss": round(lw, 5), "grad_norm": round(gw, 4)},
                  "without_boundary": {"loss": round(lo, 5), "grad_norm": round(go, 4)},
                  "boundary_token_CE_nats": bce, "next_token_after_conversation_end": tok.decode([ids[1][0]]),
                  "grad_ratio_with/without": round(gw / go, 4), "loss_ratio": round(lw / lo, 5)}, indent=1))
