#!/usr/bin/env python3
"""The two identity key lists must agree, or a key classifies nothing.

Identity is split across two files. `train_probe.py` decides which flags are CAPTURED from argv;
`probe_summarize.py` decides which captured keys count as IMPLEMENTATION rather than workload. A key
present in one and absent from the other is silently inert, and the failure is invisible: every leg
still produces a digest, the digest just does not cover the thing you believe it covers.

Both halves of that failure were live simultaneously until 2026-10-08:

  * `attention-backend` sat in `IMPL_KEYS` and was captured by nothing, so the classification could
    never fire. It had been that way since IMPL_KEYS was written, and the role card meanwhile called
    the attention-backend pair "already a proven-comparable pair".
  * no `overlap-*` key existed in EITHER list, so a ConfigSweep run measured two comm-overlap legs
    whose `config_digest` AND `impl_digest` were both identical. Two legs that computed differently
    carried the same identity, and anything keyed on that pair would have treated them as one
    measurement. The run's own Config Tuner caught it by diffing Megatron's 686-line arg dump --
    i.e. by going around the instrument that was supposed to answer the question.

Neither needed a GPU to find. Both needed someone to compare two lists, which is what this does.

This test does NOT check that the lists are COMPLETE -- no test can know which flags matter. It
checks the one property that is objectively wrong whenever it fails: a key that is classified must
be capturable, and a key that is captured must be classified into exactly one half.
"""
from __future__ import annotations

import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)

fails = 0


def ok(cond, msg):
    global fails
    print(("  ok: " if cond else "  FAIL: ") + msg)
    if not cond:
        fails += 1


def tuples_from(path, names):
    """Read tuple literals out of a module WITHOUT importing it.

    train_probe.py imports torch at module scope, which is absent outside the container. Parsing is
    not a workaround for that -- it is also the only way to read the lists without running any of
    the probe's install side effects.
    """
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def impl_keys():
    """IMPL_KEYS lives inside a function, so walk the whole tree rather than the top level."""
    tree = ast.parse(open(os.path.join(SCRIPTS, "probe_summarize.py")).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "IMPL_KEYS":
            return ast.literal_eval(node.value)
    return None


def main():
    probe = os.path.join(SCRIPTS, "train_probe.py")
    lists = tuples_from(probe, {"_IDENTITY_PAIR_KEYS", "_IDENTITY_FLAG_KEYS"})
    pair = lists.get("_IDENTITY_PAIR_KEYS")
    flag = lists.get("_IDENTITY_FLAG_KEYS")
    impl = impl_keys()
    ok(pair and flag, "train_probe.py exposes both capture lists")
    ok(impl is not None, "probe_summarize.py exposes IMPL_KEYS")
    if not (pair and flag and impl is not None):
        return 1

    # Capture strips the leading dashes, so compare in the stripped namespace the digests use.
    captured = {k.lstrip("-") for k in tuple(pair) + tuple(flag)}

    # Environment variables are captured by a different path (the probe reads os.environ), so they
    # are legitimately in IMPL_KEYS without appearing in the argv lists.
    ENV_SOURCED = {"GEAK_OVERLAY", "GEAK_OVERLAY_SHA"}
    # DERIVED keys: written from the realized state, not copied from one argv token (v6:
    # `rope-fusion` is "false" iff --no-rope-fusion is present). Each must really be written.
    DERIVED = {"rope-fusion"}
    src_probe = open(probe).read()
    for k in DERIVED:
        ok(f'out["{k}"]' in src_probe, f"derived implementation key `{k}` is written by train_probe")
    classified = set(impl) - ENV_SOURCED - DERIVED

    orphans = sorted(classified - captured)
    ok(not orphans,
       "every argv-sourced IMPL_KEY is actually captured"
       + (f" -- ORPHANS: {orphans}. Each is classified as implementation and arrives from nowhere,"
          " so it classifies nothing and the digest does not cover it." if orphans else ""))

    ok(len(set(pair)) == len(pair) and len(set(flag)) == len(flag),
       "no duplicate keys within a capture list")
    ok(not (set(pair) & set(flag)),
       "no key is both a value-pair flag and a presence-only flag")

    # The levers ConfigSweep is allowed to propose must be recorded, or an accepted config result
    # cannot be told apart from the reference it beat. This is the defect the run found, pinned so
    # it cannot come back by someone trimming a list.
    for lever in ("overlap-grad-reduce", "overlap-param-gather", "attention-backend",
                  "recompute-granularity", "recompute-method", "hf-fsdp",
                  # v5: the accepted DECISIONS 4.25 lever that no list captured, and its neighbours
                  "ddp-average-in-collective", "ddp-bucket-size", "use-distributed-optimizer",
                  "manual-gc", "manual-gc-interval", "hf-balance-tokens"):
        ok(lever in captured and lever in set(impl),
           f"Class-A lever `{lever}` is captured AND classed as implementation")

    # Schema version must advance with the lists, or a v3 and a v4 leg are indistinguishable while
    # covering different keys -- the exact confusion `snapshot_keys` exists to prevent.
    src = open(probe).read()
    m = re.search(r'out\["snapshot_keys"\]\s*=\s*"(v\d+)"', src)
    ok(m is not None and m.group(1) == "v6",
       f"snapshot_keys is stamped v6 at the capture site (found {m.group(1) if m else 'nothing'})")
    # v6 (DECISIONS 4.55): the RMSNorm epsilon is WORKLOAD identity; RoPE / CE fusion are implementation.
    ok("--norm-epsilon" in pair and "norm-epsilon" not in set(impl), "`norm-epsilon` is captured as workload identity")
    ok("--no-rope-fusion" not in flag and "rope-fusion" in set(impl),
       "RoPE fusion left the workload half: recorded as implementation key `rope-fusion`")
    ok("hf-average-tokens-across-devices" in captured and "hf-average-tokens-across-devices" not in set(impl),
       "HF loss normalization is captured as workload identity (DECISIONS 4.61)")
    for k in ("cross-entropy-loss-fusion", "cross-entropy-fusion-impl"):
        ok(k in captured and k in set(impl), f"`{k}` is captured AND classed as implementation")
    summ = open(os.path.join(SCRIPTS, "probe_summarize.py")).read()
    sys.path.insert(0, SCRIPTS)
    from probe_summarize import snapshot_schema
    ok([snapshot_schema(c) for c in ({"snapshot_keys": "v5"}, {"snapshot_keys": "v4"}, {},
                                     {"snapshot_keys": "v6"})] == [5, 4, 2, 6],
       "snapshot_schema parses vN generically (v5->5, absent->2, a future v6->6, not 2)")
    # The schedule length is WORKLOAD identity (what is trained), never implementation.
    # The GC interval must keep its VALUE (interval 0 vs 10 are different runs), so it is a pair key.
    ok("--manual-gc-interval" in pair and "--manual-gc" in flag,
       "--manual-gc-interval is a value (pair) key and --manual-gc a flag key")
    # So are a precision-recipe change and the loss normalization.
    for k in ("lr-decay-iters", "lr-decay-samples", "lr-warmup-fraction",
              "grad-reduce-in-bf16", "calculate-per-token-loss"):
        ok(k in captured and k not in set(impl), f"`{k}` is captured as workload identity")

    # ARGPARSE SEMANTICS. Identity must record the value that RAN: the last occurrence of a flag,
    # and `--flag=value` as well as `--flag value`. It took the first occurrence, so every
    # --extra-train-args override of an adapter default was recorded as the default.
    import hashlib
    g = {"os": os, "sys": sys, "hashlib": hashlib, "json": __import__("json")}
    tree = ast.parse(open(probe).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in (
                "_IDENTITY_PAIR_KEYS", "_IDENTITY_FLAG_KEYS", "_KNOBS"):
            g[n.targets[0].id] = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name in ("_config_snapshot", "_overlay_content_hash"):
            exec(compile(ast.Module(body=[n], type_ignores=[]), probe, "exec"), g)
    saved = sys.argv
    try:
        sys.argv = ["x", "--lr", "1.0e-5", "--hf-fsdp", "none", "--lr", "0.0", "--hf-fsdp=full_shard"]
        snap = g["_config_snapshot"]()
    finally:
        sys.argv = saved
    ok(snap.get("lr") == "0.0", f"a repeated flag records the LAST value, as argparse uses (got {snap.get('lr')})")
    ok(snap.get("hf-fsdp") == "full_shard", f"`--flag=value` is captured too (got {snap.get('hf-fsdp')})")
    # Kernel-selecting env knobs: captured as implementation when SET, absent otherwise (audit OVL-2).
    saved_env = dict(os.environ)
    try:
        for k in [k for k in os.environ if k.startswith(("KCK_", "FC1_", "QKV_", "ROPE_"))]:
            del os.environ[k]
        sys.argv = ["x"]
        # ROPE_BASE / ARCH_ARGS are workload variables bench_train.sh exports to EVERY rank: they must
        # never become knobs (the prefix version captured ROPE_BASE and moved every impl_digest).
        os.environ.update({"ROPE_BASE": "10000", "ARCH_ARGS": "--x", "GEAK_RMSN_CAPTURE_DIR": "/c"})
        base_snap = g["_config_snapshot"]()
        os.environ.update({"FC1_BWD_MODE": "serial", "KCK_BUILD_DIR": "/tmp/b", "GEAK_QKV_ENGAGEMENT": "/x",
                           "QKV_LT_TABLE": "/some/install/qkv_table.csv"})
        knob_snap = g["_config_snapshot"]()
    finally:
        os.environ.clear(); os.environ.update(saved_env); sys.argv = saved
    ok(not any(k.startswith("env:") for k in base_snap),
       f"no knob set -> no env key, even with ROPE_BASE/ARCH_ARGS exported ({[k for k in base_snap if k.startswith('env:')]})")
    ok(knob_snap.get("env:QKV_LT_TABLE") == "qkv_table.csv", "a table knob records the table NAME, not the install path")
    ok(knob_snap.get("env:FC1_BWD_MODE") == "serial", "a set kernel knob (FC1_BWD_MODE=serial) is captured")
    ok("env:KCK_BUILD_DIR" not in knob_snap and "env:GEAK_QKV_ENGAGEMENT" not in knob_snap,
       "paths / engagement outputs are not captured (they would make the digest install-dependent)")
    sys.path.insert(0, SCRIPTS)
    from probe_summarize import is_impl_key
    ok(is_impl_key("env:FC1_BWD_MODE") and is_impl_key("hf-train-metrics"),
       "knobs and --hf-train-metrics classify as implementation")

    # NO SECOND COPY. trajectory_gate.py kept its own literal implementation set and missed v4's keys,
    # so L1 refused every overlap/recompute lever as "a different workload". It must import the one
    # definition; a literal set of implementation keys there is exactly the defect returning.
    tg = open(os.path.join(SCRIPTS, "trajectory_gate.py")).read()
    ok("from probe_summarize import IMPL_KEYS" in tg and 'IMPL = {"attention-backend"' not in tg,
       "trajectory_gate imports IMPL_KEYS instead of keeping its own copy")
    ok(any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "IMPL_KEYS"
           for n in ast.parse(open(os.path.join(SCRIPTS, "probe_summarize.py")).read()).body),
       "IMPL_KEYS is defined at MODULE level in probe_summarize, where it can be imported")

    # F8 (v6): the overlay manifest NAME is a label -- renaming it must not move impl_digest when the
    # content hash is equal; a v5 summary keeps its historical digest (schema-aware).
    import json, tempfile, shutil
    from probe_summarize import summarize
    root = tempfile.mkdtemp(prefix="f8_")
    try:
        def leg(name, schema, label):
            d = os.path.join(root, name); os.makedirs(d)
            cfg = {"seq-length": "4096", "snapshot_keys": schema, "GEAK_OVERLAY": label, "GEAK_OVERLAY_SHA": "abc"}
            with open(os.path.join(d, "probe_rank0.jsonl"), "w") as f:
                f.write(json.dumps({"record": "header", "rank": 0, "world_size": 1, "argv_hash": "x", "config": cfg}) + "\n")
                for i in range(30):
                    f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": 1000.0, "useful_tokens": 8000,
                                        "total_tokens": 8192, "peak_reserved_mb": 1}) + "\n")
            return summarize(d)["impl_digest"]
        ok(leg("a6", "v6", "m_base.json") == leg("b6", "v6", "m_other.json"),
           "v6: renaming the overlay manifest leaves impl_digest unchanged (content hash decides)")
        ok(leg("a5", "v5", "m_base.json") != leg("b5", "v5", "m_other.json"),
           "v5 summaries keep their historical digest (the label still counted there)")
    finally:
        shutil.rmtree(root)

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
