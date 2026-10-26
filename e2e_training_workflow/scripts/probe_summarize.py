"""Summarize per-rank probe sidecars: warmup detection, steady window, useful tok/s, rank skew.

POLICY LIVES HERE, not in the probe. Measured on the 7B baseline, the SAME run reports a 12.75% or a
0.18% spread depending only on where the warmup is declared to end -- so the rule is explicit and
re-runnable against the stored series rather than baked into the collector.

Warmup rule: walk forward and take the first index from which every subsequent step is within
`tol` of the median of the remaining tail. That detects the real settling point (autotune, allocator,
comm bring-up) instead of trusting a hard-coded skip count.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import statistics
import sys


def load_rank(path):
    """One sidecar == ONE run. More than one header means two runs were appended to it.

    Found on disk: td_baseline_legA/probe/probe_rank{0,1}.jsonl each held TWO headers -- a
    ws=2/gbs=8 0.5B run (argv_hash a414f348..., ~980 ms/step) followed by a ws=8/gbs=64 7B run
    (ed00d2f9..., ~11650 ms/step) -- while ranks 2-7 held only the 7B run. The old loop took the
    LAST header but appended EVERY step, so a 0.5B run and a 7B run were summarised as one 7B leg:
    worst_iqr_spread_pct 91.69%, useful_tok_s_per_gpu 777.1. Paired with anything that leg assigns a
    band of 183%, i.e. it silently disables the throughput gate.

    `ranks_agree_on_argv` could not catch it precisely BECAUSE last-header-wins made rank 0's
    reported argv_hash match the uncorrupted ranks'. Refuse here, at the only place that can see it.
    """
    headers, steps = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("record") == "header":
                headers.append(r)
            elif r.get("record") == "step":
                steps.append(r)
    if len(headers) > 1:
        seen = [h.get("argv_hash") for h in headers]
        raise SystemExit(
            f"CORRUPT PROBE SIDECAR: {path} contains {len(headers)} header records "
            f"(argv_hash {seen}). Two or more runs were appended to one sidecar, which merges their "
            f"steps into a single fictitious leg. Do not re-summarise this directory -- the steps of "
            f"the two runs are interleaved in one file and cannot be separated by rank. Re-measure "
            f"into a FRESH output directory.")
    return (headers[0] if headers else None), steps


# Below this many steady steps the IQR is dominated by where the quartiles happen to land, not by
# the machine. Measured: 7 steady steps gave 1.243%, 24 gave 0.072% on the same workload.
MIN_STEADY_FOR_DISPERSION = 12
# The headline window never starts before this step index (0-based), even when the detector settles
# earlier -- owner decision 2026-10-23: keep 30-step legs, but do not count the first, unstable steps.
# MEASURED (DECISIONS 4.57) on 227 fixed-shape 30-step legs / 1,080 same-identity pairs: pairwise
# per-step disagreement is ~1% (median) on steps 2-11 and 0.1-0.8% after; recomputing every headline
# with the window starting at max(detected, K), K=8 gave the lowest pairwise p90 (0.731% vs 0.760% at
# the detector alone; 0.308% vs 0.347% on post-Triton-cache legs) and cut the worst pair from 3.29% to
# 1.89%. A leg too short to keep MIN_STEADY_FOR_DISPERSION steady steps after it keeps the detector's.
MIN_WARMUP_SKIP = 8


def detect_warmup_end(durs, tol=0.02, patience=3):
    """First index from which the run has SETTLED to its steady level.

    Warmup is a monotone settling phenomenon (autotune, allocator, comm bring-up); later isolated
    spikes (a dataloader stall, another tenant) are a DIFFERENT phenomenon and must not push the
    warmup boundary forward. So the reference level is the median of the back half, and the boundary
    is the first index that stays at or below ref*(1+tol) for `patience` consecutive steps. Outliers
    after that point are counted separately rather than absorbed into the warmup.
    """
    n = len(durs)
    if n < patience + 1:
        return 0
    ref = statistics.median(durs[max(0, n // 2):])
    limit = ref * (1 + tol)
    for i in range(n - patience + 1):
        if all(d <= limit for d in durs[i:i + patience]):
            return i
    return 0


def detect_warmup_end_variable(durs, toks, tol=0.02, patience=3, k=3.0):
    """Warmup end for a VARIABLE-shape step (dynamic padding: total_tokens varies step to step).

    The fixed-shape rule asks step TIME to settle within 2% of the back-half median. A dynamically
    padded step's time follows its token count (cv ~20% on HF), so that never holds and the boundary
    drifted to the end: on the 7B HF ZeRO-2 stack (2026-10-18) warmup was declared at step 26 of 30
    while the run had settled by step 3 (31.5 s, 10.1 s, then ~4.5 s), leaving 4 steady steps and an
    unusable reference -- a +6.3% candidate could not be judged. Here the quantity is time PER TOKEN
    and the tolerance is the back half's own robust spread (k x 1.4826 x MAD), never below `tol`.
    """
    rates = [d / t for d, t in zip(durs, toks) if t]
    if len(rates) != len(durs) or len(rates) < patience + 1:
        return detect_warmup_end(durs, tol=tol, patience=patience)
    back = rates[len(rates) // 2:]
    ref = statistics.median(back)
    mad = statistics.median(abs(r - ref) for r in back)
    limit = ref + max(tol * ref, k * 1.4826 * mad)
    for i in range(len(rates) - patience + 1):
        if all(r <= limit for r in rates[i:i + patience]):
            return i
    return 0


def dispersion(vals):
    """Report spread AND a robust spread, so 'wide because of warmup' is distinguishable from
    'wide because of periodic stalls'. Never collapse them into one number."""
    med = statistics.median(vals)
    out = {
        "median_ms": round(med, 2),
        "spread_pct": round(100 * (max(vals) - min(vals)) / med, 3),
        "stdev_ms": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
    }
    s = sorted(vals)
    if len(s) >= 4:
        # INTERPOLATED quartiles, as compare_legs._quantile (audit MS-7): the index pick s[n // 4] jumped a
        # whole order statistic when n moved by one -- up to 27x on a leg straddling a quartile cliff.
        def _q(q):
            x = q * (len(s) - 1); i = int(x)
            return s[i] if i + 1 >= len(s) else s[i] + (x - i) * (s[i + 1] - s[i])
        q1, q3 = _q(0.25), _q(0.75)
        out["iqr_spread_pct"] = round(100 * (q3 - q1) / med, 3)
    outliers = [v for v in vals if v > med * 1.02]
    out["outlier_steps"] = len(outliers)
    out["outlier_max_pct"] = round(100 * (max(outliers) - med) / med, 2) if outliers else 0.0
    return out


_DUMP = re.compile(r"^\s+([a-z_0-9]+) \.+ (.*)$")
_ENUM = re.compile(r"^[A-Za-z_]+\.([A-Za-z_0-9]+)$")


# Another process's log tag spliced onto a dump line. Eight ranks share one stdout, and a line of the
# realized table can be interrupted mid-write: measured on stackPK1010_ref1, a clean leg,
#   num_query_groups ........ 4[geak-probe] installed (rank 4/8) -> /work/runs/...
# which read as "captured 4, realized 4[geak-probe]..." -- a FALSE mismatch, and legUsable would have
# refused a clean leg. Tags this repo prints all look like `[geak-...]` / `[rocm-compat]` / `[rank..]`.
_SPLICE = re.compile(r"\[(geak-[a-z]+|rocm-compat|rank\d*)\].*$")


def _norm(x):
    x = _SPLICE.sub("", str(x)).strip().strip("'\"")
    m = _ENUM.match(x)
    if m:
        x = m.group(1)                                   # AttnBackend.fused -> fused
    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1].strip().strip("'\"")                # ['/data.jsonl'] -> /data.jsonl
    try:
        return float(x)
    except ValueError:
        return x


def identity_vs_realized(probe_dir, cfg):
    """Compare the identity captured from argv with what the framework says it RAN.

    The digest reads argv, which records what was ASKED for. Megatron prints its realized argument
    table at startup, which records what was DONE. Where both name a key they must agree, or the leg's
    identity is a statement about a config that did not run. This is the realized-config check
    designed earlier and deferred; it lands as an additive field, so no digest changes.

    Run over all 205 Megatron legs on disk when it was written: 1 mismatch -- a diagnostic leg that
    passed `--lr 0.0` over the adapter's `--lr 1.0e-5` while capture still took the FIRST
    occurrence. Returns None for a stack that prints no realized table (HF), rather than claiming a
    check that did not happen.
    """
    log = os.path.join(os.path.dirname(os.path.abspath(probe_dir)), "train.log")
    if not os.path.exists(log):
        return None
    real = {}
    with open(log, errors="replace") as f:
        for i, line in enumerate(f):
            if i > 4000:
                break
            m = _DUMP.match(line)
            if m:
                real[m.group(1)] = m.group(2)
    # The dotted "  key ....... value" format is what identifies a realized table; its size does not.
    # Megatron prints ~650 entries, the HF launcher ~25. A size threshold of 100 (the first version)
    # silently skipped every HF leg and reported None -- "not checked" -- for a stack that does print
    # one. Ten entries rules out a stray matching line while admitting a short, deliberate table.
    if len(real) < 10:
        return None
    checked, bad = 0, []
    for k, v in (cfg or {}).items():
        rk = k.replace("-", "_")
        if rk in real and k != "snapshot_keys":
            checked += 1
            # store_true flags were skipped (audit HFC-6): a captured "true" must realize as True
            if v == "true":
                if str(real[rk]).strip().lower() != "true":
                    bad.append({"key": k, "captured": v, "realized": real[rk].strip()})
            elif _norm(v) != _norm(real[rk]):
                bad.append({"key": k, "captured": v, "realized": real[rk].strip()})
    return {"checked": checked, "mismatches": bad}


def _cv_pct(series_by_rank):
    vals = [v for s in series_by_rank for v in s if v]
    if len(vals) < 2:
        return 0.0
    m = statistics.mean(vals)
    return round(100 * statistics.pstdev(vals) / m, 3) if m else 0.0


def _step_shape(series_by_rank):
    """'fixed' when every step carries the same number of tokens (padded to SEQ), else 'variable'.

    1% is not a tuned threshold: a padded stack's total_tokens is EXACTLY constant (cv 0.000), and
    the one dynamically padded stack measured varies by tens of percent. Anything in between would
    be a new kind of stack and deserves to be looked at, which 'variable' -- the stricter path --
    forces.
    """
    return "variable" if _cv_pct(series_by_rank) > 1.0 else "fixed"


# THE implementation half of a config snapshot -- the ONE definition. trajectory_gate.py imports it.
# It used to keep a hand-maintained SECOND COPY, which v4 updated here and not there: L1 then
# treated overlap-grad-reduce as WORKLOAD identity and refused every comm-overlap and recompute
# candidate as 'a different workload' -- unacceptable by construction once config levers had to
# pass L1. A key is implementation if it is listed here or starts with NVTE_.
IMPL_KEYS = ("attention-backend", "GEAK_OVERLAY", "GEAK_OVERLAY_SHA",
             # v4. A ConfigSweep run compared two comm-overlap legs and BOTH digests were
             # identical, because no overlap-* key was captured or classified anywhere. Listing
             # them here without capturing them in train_probe would fix nothing -- that is
             # exactly the state `attention-backend` was in, named here and captured nowhere.
             "overlap-grad-reduce", "overlap-param-gather",
             # recompute changes HOW a step is computed, not what is trained, so it belongs in
             # this half. It sat in the workload half until v4; the move is free because the
             # flag is absent from every leg on disk.
             "recompute-granularity", "recompute-method",
             # HF's parallel strategy: HOW the step is computed, like the distributed optimizer.
             "hf-fsdp",
             # v5: the gradient-reduction path. HOW gradients are reduced, not what is trained;
             # --ddp-average-in-collective was accepted as a Class-A lever (DECISIONS 4.25) and
             # was captured nowhere until v5 (train_probe.py).
             # (--grad-reduce-in-bf16 is NOT here: it also turns off fp32 main-grad accumulation,
             # a precision-recipe change like --master-weights, so it is workload identity.)
             "ddp-average-in-collective", "ddp-bucket-size", "use-distributed-optimizer",
             # host-side GC schedule: when Python collects, not what is computed
             "manual-gc", "manual-gc-interval",
             # HF: TRL's per-micro-batch metric gathers (logging work, not training math)
             "hf-train-metrics",
             # v6: which kernel applies RoPE / computes cross-entropy -- same math, other kernel.
             # New key NAMES: no pre-v6 sidecar contains them, so listing them moves no old digest.
             "rope-fusion", "cross-entropy-loss-fusion", "cross-entropy-fusion-impl",
             # HF: WHICH rank / micro-batch slot a sample of the step lands on (DECISIONS 4.61)
             "hf-balance-tokens",
             # where the NaN/Inf check runs (DECISIONS 4.62)
             "no-check-for-nan-in-loss-and-grad",
             # audit ID-10: HOW attention / activations are partitioned and computed, not what is trained.
             # Absent on every leg on disk (TP1, fused attention), so no existing digest moves.
             "sequence-parallel", "use-flash-attn")
# Implementation keys that are LABELS, not implementation: recorded, never digested from v6 on.
# GEAK_OVERLAY is the manifest file NAMES; GEAK_OVERLAY_SHA is the content. Renaming a manifest moved
# impl_digest with identical bytes running (audit F8). Schema-aware so a v5 summary keeps its digest.
IMPL_LABEL_KEYS_FROM_V6 = ("GEAK_OVERLAY",)


def _wall_span_ms(steps, i0, steady):
    """Wall time covering exactly the steady steps: t_wall(last) - t_wall(step before the window).
    None when it cannot be formed (window starts at step 0, or t_wall missing); callers fall back to
    sum(dur_ms), which excludes between-step host work."""
    if not steady or i0 < 1:
        return None
    t_prev, t_last = steps[i0 - 1].get("t_wall"), steady[-1].get("t_wall")
    if t_prev is None or t_last is None or t_last <= t_prev:
        return None
    return (t_last - t_prev) * 1000.0


def _wall_rate(steps, i0, steady):
    span = _wall_span_ms(steps, i0, steady) or sum(s["dur_ms"] for s in steady)
    return round(sum(s.get("useful_tokens", 0) for s in steady) / (span / 1000.0), 1) if span else None


def is_impl_key(k):
    """The ONE classifier: listed, an NVTE_* backend env, or a captured overlay knob `env:<NAME>`."""
    return k in IMPL_KEYS or k.startswith("NVTE_") or k.startswith("env:")


# ---- full-capture classification (audit 2026-10-25 ID-1) ------------------------------------------
# The header's `coverage` holds EVERY trainer flag and every env var of the kernel / collective /
# allocator / compiler families (train_probe._coverage). A key is MAPPED when the snapshot already
# captures it (train_probe's two identity lists, the env knobs, the 3 NVTE backends, the overlay) --
# its difference is then judged by the digests; IGNORED when it legitimately differs between two legs
# of one comparison (run length, logging, output / cache paths, which GPUs); otherwise UNMAPPED. The gate
# refuses a pair whose UNMAPPED keys differ: classify the key first (workload -> _IDENTITY_*; how-it-runs
# -> also IMPL_KEYS), never let it vary silently.
COVERAGE_IGNORE_ARGV = {
    "train-iters", "train-samples", "eval-interval", "eval-iters", "save", "save-interval", "ckpt-format",
    "async-save", "no-save-optim", "no-save-rng", "exit-interval", "exit-duration-in-mins", "exit-signal-handler",
    "distributed-timeout-minutes", "data-cache-path", "tensorboard-dir", "profile", "use-pytorch-profiler",
    "profile-step-start", "profile-step-end", "profile-ranks", "output-dir", "logging-dir", "run-name",
    "logging-steps", "save-steps", "valid-data-path", "log-throughput", "timing-log-level", "timing-log-option"}
COVERAGE_IGNORE_ARGV_PREFIXES = ("log-", "wandb-", "tensorboard-")
COVERAGE_IGNORE_ENV = {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                       "GEAK_IMAGE", "GEAK_IMAGE_ID",   # compared via the harness record
                       "GEAK_IN_CONTAINER", "GEAK_HOST_WORK", "GEAK_SCRIPTS", "GEAK_KERNEL_WF", "NCCL_DEBUG",
                       "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "TORCH_CPP_LOG_LEVEL", "TORCH_SHOW_CPP_STACKTRACES",
                       "TORCH_DISTRIBUTED_DEBUG", "TRANSFORMERS_VERBOSITY", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM"}
COVERAGE_IGNORE_ENV_SUFFIXES = ("_DIR", "_PATH", "_HOME", "_ROOT", "_FILE", "_CACHE", "_WORK")
_MAPPED_ENV = {"NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_USE_HIPBLASLT", "GEAK_OVERLAY", "GEAK_OVERLAY_SHA"}
_CAPTURED = None


def _captured_names():
    """(argv names, env knob names) the snapshot captures, read from train_probe.py's own lists (AST: the
    probe imports torch, the host may not have it). One source, like test_identity_keys."""
    global _CAPTURED
    if _CAPTURED is None:
        import ast
        tree = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_probe.py")).read())
        lists = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and \
                    n.targets[0].id in ("_IDENTITY_PAIR_KEYS", "_IDENTITY_FLAG_KEYS", "_KNOBS"):
                lists[n.targets[0].id] = [e.value for e in n.value.elts]
        argv = {k.lstrip("-") for k in lists["_IDENTITY_PAIR_KEYS"] + lists["_IDENTITY_FLAG_KEYS"]}
        argv |= {"rope-fusion", "no-rope-fusion"}      # captured as the derived `rope-fusion`
        _CAPTURED = (argv, set(lists["_KNOBS"]))
    return _CAPTURED


def unmapped_of(coverage):
    """{'argv': {...}, 'env': {...}} of the keys nobody classified, or None for a leg without the capture."""
    if not coverage or coverage.get("v") is None:
        return None
    argv_names, knobs = _captured_names()
    ua = {k: str(_norm(v)) for k, v in (coverage.get("argv") or {}).items()
          if k not in argv_names and k not in COVERAGE_IGNORE_ARGV and not k.startswith(COVERAGE_IGNORE_ARGV_PREFIXES)}
    ue = {k: v for k, v in (coverage.get("env") or {}).items()
          if k not in knobs and k not in _MAPPED_ENV and k not in COVERAGE_IGNORE_ENV
          and not k.endswith(COVERAGE_IGNORE_ENV_SUFFIXES)}
    return {"argv": ua, "env": ue}


def snapshot_schema(cfg):
    """digest_schema of a captured config: 'vN' -> N, absent -> 2. The ONE parser; trajectory_gate
    imports it. A fixed {'v5': 5, ...} lookup silently read any newer value as 2."""
    v = (cfg or {}).get("snapshot_keys")
    if v is None:
        return 2
    try:
        return int(str(v).lstrip("vV"))
    except ValueError:
        return 2


def summarize(dirpath, tol=0.02, force_warmup=None, max_steps=None):
    ranks = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "probe_rank*.jsonl"))):
        h, s = load_rank(p)
        if h is None or not s:
            continue
        ranks[h["rank"]] = (h, s)
    if not ranks:
        return {"error": f"no probe sidecars in {dirpath}"}

    per_rank, world = {}, max(len(ranks), int(next(iter(ranks.values()))[0].get("world_size", 1)))
    per_rank_tokens = []
    # Shape decides the warmup RULE, so it is read from every step up front (the steady-window value
    # reported below as step_shape is computed the same way, after the window is known).
    shape_all = _step_shape([[s.get("total_tokens", 0) for s in st] for _, st in ranks.values()])
    for rank, (h, steps) in sorted(ranks.items()):
        durs = [s["dur_ms"] for s in steps]
        # force_warmup/max_steps exist so two legs can be summarised on a COMMON window. The headline
        # divides median(useful_tokens) by median(step_ms), and useful_tokens swings +-25% step to
        # step on SFT data (7389..11945 on the reference workload). So a window that starts one or
        # two steps later contains DIFFERENT SAMPLES and the headline moves -- measured at 1.1%
        # across three runs of the same config, i.e. 9x the machine's own 0.118% reproducibility.
        # Data order is deterministic, so aligning the window makes that term cancel exactly.
        if force_warmup is not None:
            i0, rule = int(force_warmup), "forced"
        elif shape_all == "variable":
            i0 = detect_warmup_end_variable(durs, [s.get("total_tokens", 0) for s in steps], tol=tol)
            rule = "variable: time per token, tol max(2%, 3 x 1.4826 x MAD of the back half)"
        else:
            i0, rule = detect_warmup_end(durs, tol=tol), "fixed: step time within 2% of the back half"
        if force_warmup is None and i0 < MIN_WARMUP_SKIP and len(steps) - MIN_WARMUP_SKIP >= MIN_STEADY_FOR_DISPERSION:
            i0, rule = MIN_WARMUP_SKIP, rule + f"; floored at step index {MIN_WARMUP_SKIP} (the first steps are unstable)"
        steady = steps[i0:]
        if max_steps:
            steady = steady[:int(max_steps)]
        sd = [s["dur_ms"] for s in steady]
        per_rank_tokens.append([s.get("total_tokens", 0) for s in steady])
        disp = dispersion(sd)
        med = disp["median_ms"]
        useful = statistics.median([s.get("useful_tokens", 0) for s in steady])
        total = statistics.median([s.get("total_tokens", 0) for s in steady])
        # A dispersion estimate needs samples. With the README smoke settings (--iters 8) the steady
        # window is 7 steps, and the quartiles then straddle the near-warmup value: measured IQR
        # 1.243% against 0.078% on a 30-step run of the same workload. A band built on that is ~8x
        # too loose, so the number is flagged rather than quietly used.
        reliable = len(sd) >= MIN_STEADY_FOR_DISPERSION
        per_rank[rank] = {
            "steps_total": len(steps), "warmup_steps": i0, "warmup_rule": rule, "steady_steps": len(sd),
            "dispersion_reliable": reliable,
            **disp,
            # Per-GPU rate: this rank's own tokens over its own step time.
            "useful_tokens_per_step": useful, "total_tokens_per_step": total,
            "pad_fraction_pct": round(100 * (1 - useful / total), 2) if total else None,
            "useful_tok_s_per_gpu": round(useful / (med / 1000.0), 1) if med else None,
            "nominal_tok_s_per_gpu": round(total / (med / 1000.0), 1) if med else None,
            # WALL-TIME rate over the same steady window: total useful tokens / total time. The median
            # headline excludes the isolated host stalls that a real run pays (synchronized across
            # ranks, ~300-1000 ms, 1-2% of wall time on the 7B legs; DECISIONS 4.41). Report-only.
            # The span is END of the step before the window -> END of its last step, from t_wall, so
            # host work BETWEEN train_step calls (gc.collect at --manual-gc-interval N, logging) is
            # paid too; sum(dur_ms) excluded it and credited a GC-interval lever ~1 pp (audit TP-3).
            "wall_useful_tok_s_per_gpu": _wall_rate(steps, i0, steady),
            # Fixed-shape legs only: on a dynamically padded stack step time follows content, so
            # mean - median measures data skew, not stalls.
            "stall_pct": (round(100 * ((_wall_span_ms(steps, i0, steady) or sum(sd)) / len(sd) - med) / med, 3)
                          if sd and med and shape_all == "fixed" else None),
            "peak_reserved_mb": max((s.get("peak_reserved_mb", 0) for s in steady), default=0),
            "peak_alloc_mb": max((s.get("peak_alloc_mb", 0) for s in steady), default=0),
        }

    # Realized config, as OBSERVED by the probe in each rank's process (not as intended by the
    # caller). This is what makes "did these two legs measure the same thing?" answerable, and it is
    # the pinned-config verification input (Seam 1).
    headers = [h for h, _ in ranks.values()]
    cfg = dict(headers[0].get("config", {}))
    cfg["world_size"] = str(len(ranks))
    argv_hashes = sorted({h.get("argv_hash") for h in headers})

    # Two different things live in this snapshot and conflating them breaks the gate in both
    # directions. WORKLOAD IDENTITY (what is being trained) must match or a ratio is fiction.
    # IMPLEMENTATION (how it is computed) is exactly what a candidate changes -- requiring it to match
    # would refuse every config-level candidate as "incomparable", which is what happened the first
    # time an attention-backend pair was compared.
    # GEAK_OVERLAY_SHA is IMPLEMENTATION, and listing it here is mandatory, not cosmetic: `identity`
    # is the COMPLEMENT of `impl`, so an unlisted new key silently becomes WORKLOAD IDENTITY and
    # every overlay candidate would be refused as "a different workload" before it could be gated.
    # Snapshot-format metadata is NEITHER half: it describes the record, not the run. It is excluded
    # from both digests and surfaces as digest_schema below.
    META_KEYS = ("snapshot_keys",)
    impl = {k: v for k, v in cfg.items() if is_impl_key(k)}
    identity = {k: v for k, v in cfg.items() if k not in impl and k not in META_KEYS}
    cfg_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    impl_digested = ({k: v for k, v in impl.items() if k not in IMPL_LABEL_KEYS_FROM_V6}
                     if snapshot_schema(cfg) >= 6 else impl)
    impl_digest = hashlib.sha256(json.dumps(impl_digested, sort_keys=True).encode()).hexdigest()[:16]

    # POOLED rate over the common steady window (audit MS-3): every rank's useful tokens over the step time
    # (slowest rank per step) and the world size. On a variable-shape (HF) leg the median-of-rank-medians
    # headline is biased, and the bias changes sign with token balancing (+0.8% / -0.6%); pooled is the
    # quantity a run actually delivers. Fixed-shape legs keep the median headline (pooled reported).
    _i0 = max(v["warmup_steps"] for v in per_rank.values())
    _st = [st for _, (_, st) in sorted(ranks.items())]
    _n = min(len(x) for x in _st)
    if max_steps:
        _n = min(_n, _i0 + int(max_steps))
    _tok = sum(x[i].get("useful_tokens", 0) for x in _st for i in range(_i0, _n))
    _dur = sum(max(x[i]["dur_ms"] for x in _st) for i in range(_i0, _n))
    pooled = round(_tok / (_dur / 1000.0) / world, 1) if _dur > 0 and _n > _i0 else None
    meds = [v["median_ms"] for v in per_rank.values()]
    slowest = max(per_rank.items(), key=lambda kv: kv[1]["median_ms"])[0]
    out = {
        "dir": dirpath, "ranks": len(per_rank), "world_size": world,
        "per_rank": per_rank,
        # The step is as slow as its slowest rank, so the run-level number takes the max, and the
        # skew is reported instead of being averaged away -- it is the straggler bucket's input.
        "step_median_ms": round(max(meds), 2),
        # FIXED vs VARIABLE step shape, decided from the data rather than from which stack ran.
        # Megatron pads every sample to SEQ, so total_tokens is constant and step time is close to
        # constant: per-leg dispersion is measurement noise. A dynamically padded stack (HF TRL)
        # has step time that follows content -- and under DDP, the content on the SLOWEST rank, so
        # even a dur = a + b*tokens fit on one rank leaves 16% residual IQR. For such legs per-leg
        # IQR (~25%) measures the dataset, not the measurement, and only a PAIRED comparison
        # (compare_legs' per-step ratio, 0.26% on an identical pair) separates noise from signal.
        "step_shape": _step_shape(per_rank_tokens),
        "identity_vs_realized": identity_vs_realized(dirpath, cfg),
        # Every flag / env var the snapshot does not classify (audit ID-1); None = the leg predates the
        # full capture. Ranks must agree on it (argv_hash covers argv; env is per process).
        "unmapped": unmapped_of(headers[0].get("coverage")),
        # content identity (audit #9 / #10): overlay directory trees and input fingerprints; None = predates it
        "content": headers[0].get("content"),
        "unmapped_ranks_agree": len({json.dumps(unmapped_of(h.get("coverage")), sort_keys=True)
                                     for h in headers}) == 1,
        # Report-only: which harness code ran inside the leg (rocm_compat/overlay_train/train_probe).
        "harness": headers[0].get("harness"),
        "total_tokens_cv_pct": _cv_pct(per_rank_tokens),
        "rank_skew_pct": round(100 * (max(meds) - min(meds)) / statistics.median(meds), 3),
        "slowest_rank": slowest,
        "useful_tok_s_per_gpu": (pooled if (_step_shape(per_rank_tokens) == "variable" and pooled) else
                                 round(statistics.median([v["useful_tok_s_per_gpu"] or 0 for v in per_rank.values()]), 1)),
        "useful_tok_s_per_gpu_pooled": pooled,
        "useful_tok_s_per_gpu_rank_median": round(statistics.median(
            [v["useful_tok_s_per_gpu"] or 0 for v in per_rank.values()]), 1),
        # Report-only wall-time view (see per_rank): what a production run of this config pays,
        # stalls included. The gate still decides on the median; this makes the gap visible.
        # Median over ranks, exactly like the headline: the step boundary is shared, so the span is
        # the same on every rank and the rates differ only by each rank's tokens. The worst rank
        # (2026-10-21) measured DP data imbalance, not stalls (audit TP-4: 781.7 vs 830.4 unpacked).
        "wall_useful_tok_s_per_gpu": round(statistics.median(
            [v["wall_useful_tok_s_per_gpu"] or 0 for v in per_rank.values()]), 1),
        "stall_pct": (round(statistics.median(v["stall_pct"] for v in per_rank.values()), 3)
                      if all(v["stall_pct"] is not None for v in per_rank.values()) else None),
        "nominal_tok_s_per_gpu": round(statistics.median(
            [v["nominal_tok_s_per_gpu"] or 0 for v in per_rank.values()]), 1),
        "peak_reserved_mb": max(v["peak_reserved_mb"] for v in per_rank.values()),
        "warmup_steps_max": max(v["warmup_steps"] for v in per_rank.values()),
        "config": cfg,
        "workload_identity": identity,
        "implementation": impl,
        # Bumped whenever the digest DEFINITION changes (e.g. when implementation keys were split
        # out of identity). Without it, a summary written under an older definition looks like a
        # different workload, and the gate refuses a comparison that is actually fine after a
        # re-summarise -- which is free, since the probe sidecars are on disk.
        #
        # READ FROM THE HEADER, not hardcoded, and that distinction is load-bearing. Schema 3 widened
        # workload identity from 5 realized fields to the regularization/optimizer/architecture set,
        # after attention_dropout 0.0 vs 0.1 digested identically. A v2 sidecar PHYSICALLY DOES NOT
        # CONTAIN attention_dropout, so unlike every previous bump this one cannot be repaired by
        # re-summarising: hardcoding 3 here would stamp v3 on a v2 header and produce a digest that
        # CLAIMS to cover dropout and does not -- precisely the fiction the guard exists to prevent.
        # A v2-vs-v3 pair must be RE-MEASURED, not re-summarised.
        "digest_schema": snapshot_schema(cfg),
        "config_digest": cfg_digest,      # identity only: what is being trained
        "impl_digest": impl_digest,       # how it is computed: a candidate is EXPECTED to differ here
        # Ranks running different argv would silently mix two configurations inside ONE leg.
        "ranks_agree_on_argv": len(argv_hashes) == 1,
        "worst_spread_pct": max(v["spread_pct"] for v in per_rank.values()),
        "worst_iqr_spread_pct": max(v.get("iqr_spread_pct", 0) for v in per_rank.values()),
        "dispersion_reliable": all(v.get("dispersion_reliable") for v in per_rank.values()),
        "min_steady_steps": min(v["steady_steps"] for v in per_rank.values()),
        "outlier_steps_max": max(v["outlier_steps"] for v in per_rank.values()),
        "pad_fraction_pct": statistics.median([v["pad_fraction_pct"] or 0 for v in per_rank.values()]),
    }
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    fw = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else None
    print(json.dumps(summarize(d, tol, fw, ms), indent=2))
