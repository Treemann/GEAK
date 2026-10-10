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
    classified = set(impl) - ENV_SOURCED

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
                  "recompute-granularity", "recompute-method", "hf-fsdp"):
        ok(lever in captured and lever in set(impl),
           f"Class-A lever `{lever}` is captured AND classed as implementation")

    # Schema version must advance with the lists, or a v3 and a v4 leg are indistinguishable while
    # covering different keys -- the exact confusion `snapshot_keys` exists to prevent.
    src = open(probe).read()
    m = re.search(r'out\["snapshot_keys"\]\s*=\s*"(v\d+)"', src)
    ok(m is not None and m.group(1) == "v4",
       f"snapshot_keys is stamped v4 at the capture site (found {m.group(1) if m else 'nothing'})")
    summ = open(os.path.join(SCRIPTS, "probe_summarize.py")).read()
    ok('"v4": 4' in summ, "probe_summarize maps v4 to digest_schema 4")

    # ARGPARSE SEMANTICS. Identity must record the value that RAN: the last occurrence of a flag,
    # and `--flag=value` as well as `--flag value`. It took the first occurrence, so every
    # --extra-train-args override of an adapter default was recorded as the default.
    import hashlib
    g = {"os": os, "sys": sys, "hashlib": hashlib, "json": __import__("json")}
    tree = ast.parse(open(probe).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in (
                "_IDENTITY_PAIR_KEYS", "_IDENTITY_FLAG_KEYS"):
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

    # NO SECOND COPY. trajectory_gate.py kept its own literal implementation set and missed v4's keys,
    # so L1 refused every overlap/recompute lever as "a different workload". It must import the one
    # definition; a literal set of implementation keys there is exactly the defect returning.
    tg = open(os.path.join(SCRIPTS, "trajectory_gate.py")).read()
    ok("from probe_summarize import IMPL_KEYS" in tg and 'IMPL = {"attention-backend"' not in tg,
       "trajectory_gate imports IMPL_KEYS instead of keeping its own copy")
    ok(any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "IMPL_KEYS"
           for n in ast.parse(open(os.path.join(SCRIPTS, "probe_summarize.py")).read()).body),
       "IMPL_KEYS is defined at MODULE level in probe_summarize, where it can be imported")

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
