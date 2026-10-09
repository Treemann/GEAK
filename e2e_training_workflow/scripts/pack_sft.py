#!/usr/bin/env python3
"""Pack SFT conversations into lines Megatron's SFTDataset unpacks with attention isolation.

WHY THIS IS A DATA TRANSFORM AND NOT A FRAMEWORK CHANGE. Megatron's SFT path already packs: one jsonl
line's "messages" may hold several conversations, each starting with a `system` message, and
SFTDataset splits on the system role, tokenises each conversation separately, concatenates them with
one cu_seqlens entry per conversation (THD attention isolation) and restarts position_ids per
conversation. So packing is "write the lines differently". On the 7B reference workload 69.7% of every
step is padding, and the measured ceiling for removing it is ~3.3x useful throughput.

TOKEN IDENTITY IS THE WHOLE CONTRACT. The source data has no system messages, and the Qwen chat
template inserts a DEFAULT system prompt when none is present. Packing needs an explicit system message
at every conversation boundary, so it must carry exactly that default or every token changes. The
default differs by model -- "You are a helpful assistant." for Qwen2.5-0.5B, "Please reason step by
step, and put your final answer within \\boxed{}." for Qwen2.5-Math-7B -- so it is READ from the
tokenizer, never written here. Verified 500/500 conversations token-identical with the explicit
default, on both models, before this script existed. --verify re-checks it on every run.

PACKING. First-fit in data order: deterministic, preserves order as far as fitting allows, and every
pack's total length is <= SEQ so Megatron never truncates a packed conversation (a conversation
longer than SEQ on its own goes alone and is truncated exactly as it is unpacked).

Usage:
  pack_sft.py <in.jsonl> <out.jsonl> --tokenizer <dir> --seq 4096 [--limit N] [--verify 500]
"""
from __future__ import annotations

import argparse
import json
import sys


def default_system_prompt(tok):
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
    marker = "<|im_start|>system\n"
    if marker not in s:
        sys.exit("the chat template inserts no default system prompt; packing by system-role "
                 "boundaries would change every token -- refusing")
    return s.split(marker, 1)[1].split("<|im_end|>", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=500)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sysmsg = {"role": "system", "content": default_system_prompt(tok)}

    rows = []
    with open(a.src) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            rows.append(json.loads(line)["messages"])

    # The contract, re-checked: explicit default system == implicit default, token for token.
    bad = 0
    for msgs in rows[:a.verify]:
        if msgs and msgs[0]["role"] == "system":
            continue
        if tok.apply_chat_template(msgs, tokenize=True) != tok.apply_chat_template([sysmsg] + msgs, tokenize=True):
            bad += 1
    if bad:
        sys.exit(f"{bad}/{min(a.verify, len(rows))} conversations change tokens when the default system "
                 "prompt is made explicit -- packing would not be a pure layout change; refusing")

    lens = []
    convs = []
    for msgs in rows:
        c = msgs if (msgs and msgs[0]["role"] == "system") else [sysmsg] + msgs
        convs.append(c)
        lens.append(len(tok.apply_chat_template(c, tokenize=True)))

    bins, room = [], []                       # first-fit, in data order
    for idx, n in enumerate(lens):
        if n >= a.seq:
            bins.append([idx]); room.append(0)
            continue
        for b, r in enumerate(room):
            if n <= r:
                bins[b].append(idx); room[b] -= n
                break
        else:
            bins.append([idx]); room.append(a.seq - n)

    with open(a.dst, "w") as f:
        for b in bins:
            f.write(json.dumps({"messages": [m for i in b for m in convs[i]]}) + "\n")

    tot = sum(min(n, a.seq) for n in lens)
    print(json.dumps({
        "conversations": len(rows), "packs": len(bins),
        "conversations_per_pack": round(len(rows) / len(bins), 3),
        "fill_pct": round(100.0 * tot / (len(bins) * a.seq), 2),
        "unpacked_fill_pct": round(100.0 * tot / (len(rows) * a.seq), 2),
        "default_system_prompt": sysmsg["content"],
        "token_identity_verified": min(a.verify, len(rows)),
    }, indent=1))


if __name__ == "__main__":
    main()
