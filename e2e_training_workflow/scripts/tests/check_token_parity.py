import sys, json, torch
sys.path.insert(0, "/workspace/Primus/third_party/Megatron-LM")
from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
SEQ, N = 4096, 300
mt = SFTTokenizer("/models/qwen2_5_0_5b", "default")
pad = mt.pad_id
rows = [json.loads(l) for _, l in zip(range(N), open("/work/data/sft_ultrachat20k.jsonl"))]
meg = []
for r in rows:
    toks, tg = mt.tokenize_conversation(r["messages"], return_target=True, add_generation_prompt=False)
    tg = list(tg)[:SEQ] + [pad]               # truncate body to SEQ, then the alignment pad
    labels = tg[1:]
    meg.append(sum(1 for x in labels if x != pad and x != -100))
tok = AutoTokenizer.from_pretrained("/models/qwen2_5_0_5b")
ds = load_dataset("json", data_files="/work/data/sft_ultrachat20k.jsonl", split=f"train[:{N}]")
cfg = SFTConfig(output_dir="/tmp/o", max_length=SEQ, assistant_only_loss=False,
                completion_only_loss=False, packing=False, report_to=[], per_device_train_batch_size=1, bf16=False, use_cpu=True)
from transformers import AutoConfig, AutoModelForCausalLM
m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("/models/qwen2_5_0_5b"))
tr = SFTTrainer(model=m, args=cfg, train_dataset=ds, processing_class=tok)
hf = []
for i in range(N):
    b = tr.data_collator([tr.train_dataset[i]])
    hf.append(int((b["labels"][:, 1:] != -100).sum()))
d = [h - m_ for h, m_ in zip(hf, meg)]
print(json.dumps({"megatron_pad_id": pad, "eos": tok.eos_token_id, "n": N,
  "megatron_total": sum(meg), "hf_total": sum(hf),
  "rel_diff_pct": round(100*(sum(hf)-sum(meg))/sum(meg), 4),
  "per_sample_diff_hist": {str(k): d.count(k) for k in sorted(set(d))},
  "truncated_samples": sum(1 for x in meg if x >= SEQ-1)}, indent=1))
