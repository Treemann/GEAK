"""GEAK training probe — stack-neutral measurement, per-rank JSONL sidecars (Seam 2).

WHY THIS EXISTS. Upstream Megatron logs no tokens/s at all; other stacks log it with a nominal
(padding-inclusive) definition that is wrong for SFT, and each logs loss/grad-norm/memory in its own
format or not at all. Parsing logs therefore costs one parser per framework AND yields numbers that
are not comparable across frameworks. So measurement is taken at the torch level instead, and the
framework's own log is demoted to a cross-check.

WHAT IS NEUTRAL AND WHAT IS NOT. Everything here is framework-neutral except ONE binding: what marks
a step boundary and where the batch appears. That binding is declared per stack in STEP_BINDINGS
(Seam 3, capability manifest). Adding a framework = adding one entry, not a new probe.

RAW IN, POLICY OUT. The probe records RAW per-step facts only. It does NOT decide where the warmup
ends, what the steady window is, or what the noise band means — probe_summarize.py does, so the
policy is explicit, reviewable, and re-runnable against a stored series. Measured on the 7B baseline:
the same run yields a 12.75% or a 0.18% spread depending purely on that policy, so it must not be
buried inside the collector.

USEFUL TOKENS. `loss_mask.sum()` when the stack provides one (Megatron masks padding by pad-token-id
AND prompts by -100, so `labels != -100` alone would miss padding — the larger term). Falls back to
`(labels != -100).sum()`, the convention shared by HF Trainer / TRL / ms-swift.

PROFILING LIVES HERE TOO, for the same reason. Upstream Megatron ships NO profiler arguments at all
(the `--profile` family in Primus recipes is Primus's own addition), and other stacks each expose a
different one or none. A probe-driven window works on every stack the probe can bind to, and costs
the adapter nothing.

Enable with GEAK_PROBE_DIR=<dir>. Absent -> import is a no-op and the run is unaffected.
Optional: GEAK_PROFILE_STEPS="start:end" (1-based, inclusive) and GEAK_PROFILE_RANKS="0,4".
"""
from __future__ import annotations

import json
import os
import time

import torch

_DIR = os.environ.get("GEAK_PROBE_DIR", "")
_ENABLED = bool(_DIR)


class _Probe:
    def __init__(self, out_dir: str):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"probe_rank{self.rank}.jsonl")
        self.hb_path = os.path.join(out_dir, f"heartbeat_rank{self.rank}.json")
        # ONE probe dir == ONE leg. The handle is opened "a", so re-using a --tag appends a second
        # run's steps to the first's sidecar; probe_summarize then merges them into one fictitious
        # leg. That happened: td_baseline_legA holds a 0.5B run followed by a 7B run and summarised
        # as a single 7B leg with 91.69% IQR, which would assign a 183% band and disable the gate.
        # Refuse rather than truncate: an existing sidecar is somebody's measurement, and silently
        # overwriting it is the same class of harm as silently appending to it.
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            raise SystemExit(
                f"[geak-probe] REFUSING to append to an existing sidecar: {self.path} "
                f"({os.path.getsize(self.path)} bytes). One probe dir is one leg. Appending a second "
                f"run here produces a merged, fictitious leg that passes every downstream check. "
                f"Use a fresh --tag / GEAK_PROBE_DIR, or delete this directory if it is scrap.")
        self.fh = open(self.path, "a", buffering=1)
        self.step = 0
        self.prof = None
        self.prof_range = _profile_range()
        self.prof_ranks = _profile_ranks()
        # Batch facts land here from the batch hook, are consumed by the step hook. A step spans
        # several micro-batches (grad accumulation), so they accumulate and reset per step.
        self.pending = {"useful_tokens": 0, "total_tokens": 0, "micro_batches": 0}
        self._t0 = None
        self._host_init()
        self._emit({"record": "header", "rank": self.rank, "world_size": self.world,
                    "local_rank": self.local_rank, "t_wall": time.time(),
                    "device": torch.cuda.get_device_name(self.local_rank)
                    if torch.cuda.is_available() else "cpu",
                    "argv_hash": _argv_hash(), "config": _config_snapshot(), "harness": _harness_shas()})

    def _emit(self, obj):
        self.fh.write(json.dumps(obj) + "\n")

    # ---- host diagnostics (report-only, 2026-10-22) -----------------------------------
    # Isolated 300-1000 ms steps cost 1-3% of wall time (DECISIONS 4.41). Step time is synchronized
    # across ranks, so the step record alone cannot say WHICH rank stalled or WHY. These per-rank,
    # per-step facts can: Python GC time per generation (gc.callbacks), host time inside the batch
    # fetch, caching-allocator retries / segment (de)allocations, and host RSS (which also answers
    # whether --manual-gc-interval 0 grows host memory over an epoch). Never enters a digest.
    def _host_init(self):
        self._gc = {"ms": 0.0, "n": [0, 0, 0]}
        self._gc_t = None
        self._batch_ms = 0.0
        self._mem_prev = {}
        try:
            import gc as _gc

            def _cb(phase, info):
                if phase == "start":
                    self._gc_t = time.perf_counter()
                elif self._gc_t is not None:
                    self._gc["ms"] += (time.perf_counter() - self._gc_t) * 1000.0
                    g = info.get("generation", 0)
                    if 0 <= g < 3:
                        self._gc["n"][g] += 1
                    self._gc_t = None
            _gc.callbacks.append(_cb)
        except Exception:
            pass

    def _host_record(self):
        h = {"gc_ms": round(self._gc["ms"], 2), "gc_n": list(self._gc["n"]),
             "batch_ms": round(self._batch_ms, 2)}
        try:
            with open("/proc/self/statm") as f:
                h["rss_mb"] = round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                st = torch.cuda.memory_stats()
                for k in ("num_alloc_retries", "num_device_alloc", "num_device_free"):
                    if k in st:
                        prev = self._mem_prev.get(k)
                        h[k] = int(st[k]) - prev if prev is not None else 0   # delta this step
                        self._mem_prev[k] = int(st[k])
        except Exception:
            pass
        self._gc = {"ms": 0.0, "n": [0, 0, 0]}
        self._batch_ms = 0.0
        return h

    # ---- batch hook -------------------------------------------------------------------
    def note_batch(self, batch):
        """Count USEFUL tokens for one micro-batch. Never raises into the training loop."""
        try:
            lm = batch.get("loss_mask") if isinstance(batch, dict) else None
            labels = batch.get("labels") if isinstance(batch, dict) else None
            if lm is not None:
                useful = int(lm.sum().item())
                total = int(lm.numel())
            elif labels is not None:
                useful = int((labels != -100).sum().item())
                total = int(labels.numel())
            else:
                return
            self.pending["useful_tokens"] += useful
            self.pending["total_tokens"] += total
            self.pending["micro_batches"] += 1
        except Exception as e:  # measurement must never break the run
            self.pending.setdefault("errors", []).append(repr(e))

    # ---- profiler window ---------------------------------------------------------------
    # A step-ranged window, driven from the step boundary the probe already owns. record_shapes is ON
    # because shape attribution is what the Kernel Extractor consumes; with_stack is OFF because
    # python-function events dominate the trace size and buy nothing for kernel attribution.
    def _maybe_profile_start(self):
        if self.prof is not None or not self.prof_range:
            return
        if self.rank not in self.prof_ranks or (self.step + 1) != self.prof_range[0]:
            return
        try:
            from torch.profiler import ProfilerActivity, profile
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                record_shapes=True, with_stack=False, with_modules=False)
            self.prof.__enter__()
            print(f"[geak-probe] profiler ON rank{self.rank} steps {self.prof_range[0]}-{self.prof_range[1]}", flush=True)
        except Exception as e:
            self.prof = None
            self._emit({"record": "profile_error", "error": repr(e)})

    def _maybe_profile_stop(self):
        if self.prof is None or self.step < self.prof_range[1]:
            return
        try:
            self.prof.__exit__(None, None, None)
            out = os.path.join(os.path.dirname(self.path), f"trace_rank{self.rank}.json")
            self.prof.export_chrome_trace(out)
            self._emit({"record": "profile", "trace": out,
                        "steps": list(self.prof_range), "rank": self.rank})
            print(f"[geak-probe] trace written: {out}", flush=True)
        except Exception as e:
            self._emit({"record": "profile_error", "error": repr(e)})
        finally:
            self.prof = None

    # ---- step hooks -------------------------------------------------------------------
    def step_begin(self):
        self._maybe_profile_start()
        self._t0 = time.perf_counter()

    def step_end(self, loss=None, grad_norm=None, skipped=None, t_end=None):
        """`t_end` lets a binding close the timing window BEFORE it has the step's loss.

        HF fires on_step_end before on_log, so at the moment the step's compute ends the loss and
        grad_norm do not exist yet. The HF binding synchronises and records t_end there, then calls
        this from on_log. Without it the logging all-reduce would be timed as part of the step on HF
        and not on Megatron, and the two stacks' step times would differ for a reason that is not
        the stack.
        """
        if self._t0 is None:
            return
        if t_end is None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_end = time.perf_counter()
        dur_ms = (t_end - self._t0) * 1000.0
        self.step += 1
        rec = {
            "record": "step", "step": self.step, "dur_ms": round(dur_ms, 3),
            "useful_tokens": self.pending["useful_tokens"],
            "total_tokens": self.pending["total_tokens"],
            "micro_batches": self.pending["micro_batches"],
            "t_wall": time.time(),
        }
        if torch.cuda.is_available():
            rec["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            rec["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        if loss is not None:
            rec["loss"] = loss
        if grad_norm is not None:
            rec["grad_norm"] = grad_norm
        if skipped is not None:
            rec["skipped"] = skipped
        try:
            rec["host"] = self._host_record()
        except Exception:
            pass
        self._emit(rec)
        # Heartbeat: step progress, so a hung collective (every rank alive, nobody advancing) is
        # detectable from outside. A liveness check on the processes would report "healthy".
        try:
            with open(self.hb_path, "w") as f:
                json.dump({"rank": self.rank, "step": self.step, "t_wall": time.time()}, f)
        except Exception:
            pass
        self.pending = {"useful_tokens": 0, "total_tokens": 0, "micro_batches": 0}
        self._t0 = None
        self._maybe_profile_stop()


def _profile_range():
    spec = os.environ.get("GEAK_PROFILE_STEPS", "")
    if ":" not in spec:
        return None
    try:
        a, b = spec.split(":", 1)
        return (int(a), int(b))
    except Exception:
        return None


def _profile_ranks():
    spec = os.environ.get("GEAK_PROFILE_RANKS", "0")
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out or {0}


def _harness_shas():
    """sha256[:16] of the harness code that runs INSIDE every leg (audit OVL-6): rocm_compat patches
    Megatron at runtime (it changes the loss on packed data) and overlay_train rebinds symbols, yet
    neither was part of any leg's record. REPORT-ONLY provenance, outside `config`, so no digest moves;
    two legs whose harness differs can be told apart after the fact."""
    import hashlib
    here, out = os.path.dirname(os.path.abspath(__file__)), {}
    for f in ("rocm_compat.py", "overlay_train.py", "train_probe.py"):
        try:
            with open(os.path.join(here, f), "rb") as fh:
                out[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError:
            out[f] = None
    out["image"] = os.environ.get("GEAK_IMAGE")
    out["image_id"] = os.environ.get("GEAK_IMAGE_ID")
    return out


def _argv_hash():
    import hashlib
    import sys
    return hashlib.sha256(" ".join(sys.argv).encode()).hexdigest()[:16]


def _overlay_content_hash(manifest_path):
    """Hash what the overlay WILL RUN, not what it is CALLED.

    The basename alone moved for a repackage that kept the kernel bytes identical, and would NOT
    move for a swapped kernel under a kept filename -- so it cannot support the "same implementation
    re-measured" claim that was made of impl_digest in the 7B run.

    What is hashed: the manifest's SEMANTIC fields (the rebind target, and the attribute half of
    `replacement`) plus the BYTES of every file it points at, length-prefixed so two files cannot be
    concatenated into a third's hash. What is NOT hashed: any path, and `engagement_out`. Those are
    per-leg output locations -- hashing the raw manifest would give two legs running the identical
    kernel different impl_digests purely because they wrote engagement JSON to different files,
    which is the same class of false alarm this hash exists to remove.

    A file that cannot be read yields a value that is visibly NOT a digest, carried in-band into the
    config and therefore into impl_digest and the report. Skipping it and hashing a shorter list
    would return a confident-looking 16 hex chars for an unverified overlay -- a guard that silently
    degrades is worse than no guard.
    """
    import hashlib
    try:
        with open(manifest_path, "rb") as f:
            cfg = json.loads(f.read())
    except Exception as e:
        print(f"[geak-overlay] MANIFEST UNREADABLE ({e!r}) -- impl_digest cannot identify this "
              f"overlay; do not cite it as provenance.", flush=True)
        return f"UNREADABLE_MANIFEST:{type(e).__name__}"
    rep_path, _, rep_attr = (cfg.get("replacement") or "").partition(":")
    h = hashlib.sha256()
    # Semantic head: what gets rebound, and to which attribute. Both change meaning; neither is a path.
    h.update(("target=" + str(cfg.get("target") or "") + "\nattr=" + rep_attr + "\n").encode())
    files = ([rep_path] if rep_path else []) + list(cfg.get("sources") or [])
    for p in files:
        try:
            with open(p, "rb") as f:
                b = f.read()
        except Exception as e:
            print(f"[geak-overlay] OVERLAY FILE UNREADABLE {p!r} ({e!r}) -- impl_digest cannot "
                  f"identify this overlay; do not cite it as provenance.", flush=True)
            return f"UNREADABLE_FILE:{os.path.basename(p)}:{type(e).__name__}"
        h.update(b"%d:" % len(b) + b)
    return h.hexdigest()[:16]


# WORKLOAD IDENTITY KEYS (Seam 1). Every flag here changes WHAT is trained, so two legs that differ
# in any of them are not comparable and a ratio between them is fiction.
#
# This list was NINE flags and covered five realized fields (mbs/gbs/seq/tokenizer-format/world_size).
# That was not enough by a wide margin, proven twice on real runs:
#   - attention_dropout 0.0 vs 0.1 digested IDENTICALLY (288f95e335ac66a4). Diffing the two 686-line
#     Megatron arg dumps gave exactly ONE differing line, and step-1 loss moved 3.1689 -> 4.3728 on
#     the same data. A +7.3% "win" was partly just "we did not do the dropout".
#   - the model itself was never covered: a 0.5B and a 7B leg at the same mbs/gbs/seq/world_size
#     would have digested identically, because ARCH_ARGS never reached the snapshot.
#   - tensor/pipeline/context-parallel-size were in the old list but NEVER appeared in any recorded
#     config, because the adapter does not pass them explicitly -- so the "parallel mapping" the
#     gate's docstring claims is covered was in fact covered on no leg ever measured.
# train_iters is deliberately EXCLUDED: it is a measurement length, not a workload, and legs are
# window-aligned by compare_legs.py anyway.
_IDENTITY_PAIR_KEYS = (
    # mapping / batching
    "--micro-batch-size", "--global-batch-size", "--seq-length", "--tensor-model-parallel-size",
    "--pipeline-model-parallel-size", "--context-parallel-size",
    "--sft-tokenizer-prompt-format",
    # HOW a step is computed (classified as implementation in probe_summarize.IMPL_KEYS, not as
    # workload identity). Added after a ConfigSweep run measured two comm-overlap legs whose
    # config_digest AND impl_digest were both identical: no overlap-* key existed anywhere in this
    # file, so the snapshot was blind to the only thing that differed between them. `--attention-
    # backend` is the same bug found by inspection rather than by a run -- probe_summarize has
    # listed it in IMPL_KEYS all along, while nothing here ever captured it, so the key it
    # classified could never arrive. A key named in one list and absent from the other classifies
    # nothing.
    "--attention-backend", "--recompute-granularity", "--recompute-method",
    # WHICH STACK. Workload identity, not implementation: a Megatron leg and an HF leg with otherwise
    # identical arguments must NOT share a config_digest, or the gate would compare a Megatron
    # reference against an HF candidate as "the same workload". Only the HF launcher passes it, so
    # every Megatron leg on disk digests exactly as before -- the same additive property that made
    # v3->v4 free.
    "--geak-stack",
    # Optimizer/master precision is part of the recipe, like dropout: it changes the training run,
    # not just how fast it goes. Only the HF launcher passes it; Megatron's --bf16 implies fp32 main
    # params and every Megatron leg digests as before.
    "--master-weights",
    # HF parallel strategy (implementation half, see probe_summarize.IMPL_KEYS).
    "--hf-fsdp",
    # v5, implementation half: DDP bucketing (a ConfigSweep lever, measured -0.42% at 2e8), and the
    # GC schedule (--manual-gc's interval; host-stall lever, 2026-10-21).
    "--ddp-bucket-size", "--manual-gc-interval",
    # HF only: TRL's per-micro-batch metric gathers (on/off moved throughput 21-24% with both digests
    # equal; audit 2026-10-22 TP-2). Implementation: logging work, not training math.
    "--hf-train-metrics",
    # regularization + optimizer: these move the loss/grad-norm trajectory L1 gates on
    "--attention-dropout", "--hidden-dropout", "--weight-decay", "--clip-grad",
    "--lr", "--min-lr", "--lr-decay-style", "--lr-warmup-iters",
    # v5. The schedule LENGTH is recipe too: _cosine and _cosine_avg decay over 358 iterations, and
    # a leg decaying over a different length trains a different run with every other key equal.
    "--lr-decay-iters", "--lr-decay-samples", "--lr-warmup-fraction",
    "--adam-beta1", "--adam-beta2", "--adam-eps", "--seed",
    # what is trained, on what
    "--load", "--tokenizer-model", "--train-data-path",
    "--num-layers", "--hidden-size", "--ffn-hidden-size", "--num-attention-heads",
    "--num-query-groups", "--kv-channels", "--max-position-embeddings", "--rotary-base",
)
# Presence-only flags (store_true). The old scan read `--flag value` pairs ONLY, so none of these
# could ever be captured -- a leg run with --bf16 and one without were the same workload to the gate.
_IDENTITY_FLAG_KEYS = (
    "--bf16", "--fp16", "--sft", "--finetune", "--swiglu", "--disable-bias-linear",
    "--add-qkv-bias", "--no-rope-fusion", "--untie-embeddings-and-output-weights",
    "--group-query-attention", "--sequence-parallel", "--use-flash-attn",
    # comm overlap: implementation, see the note above
    "--overlap-grad-reduce", "--overlap-param-gather",
    # v5, implementation half. --ddp-average-in-collective is an ACCEPTED lever (DECISIONS 4.25)
    # that no list captured: a _cosine leg (off) and a _cosine_avg leg (on) carried the SAME
    # config_digest 2e296ee3c3e6aba2 AND impl_digest 49d9009f46b82686 (pcos_1016_170258_ref1 vs
    # baseline_td1020_1425_leg1), so a reused reference could carry its ~0.9% into a gate silently.
    # The same defect class v4 fixed for overlap-*. --use-distributed-optimizer is its neighbour in
    # the reduction path.
    "--ddp-average-in-collective", "--use-distributed-optimizer",
    # Synchronized Python GC (Megatron: gc.disable() + gc.collect() every --manual-gc-interval steps
    # on every rank). Host-side only; captured so a GC-lever leg cannot share a digest with its ref.
    "--manual-gc",
    # v5, WORKLOAD identity. --grad-reduce-in-bf16 also turns off fp32 main-grad accumulation
    # (Megatron arguments.py: accumulate_allreduce_grads_in_fp32 = False): a precision-recipe
    # change, like --master-weights. --calculate-per-token-loss changes the loss normalization
    # (pack_gate.py passes it).
    "--grad-reduce-in-bf16", "--calculate-per-token-loss",
)


# An explicit ALLOWLIST (from grepping every shipped overlay for os.environ reads), not prefixes: the
# first version used "ROPE_", which also matched ROPE_BASE -- a workload variable bench_train.sh
# exports to every rank -- and would have moved every leg's impl_digest. Paths, build parallelism,
# capture tooling and debug output are left out: they select no kernel, and a path would make the
# digest install-dependent. The three *_LT_TABLE knobs select a GEMM table by PATH: their basename is
# recorded (the table name), not the path.
_KNOBS = (
    "KCK_AITER_BWD", "KCK_ASM_VARIANT", "KCK_BF16_RND", "KCK_BWD", "KCK_CACHE_WS", "KCK_CK_AUX", "KCK_DQACC",
    "KCK_FAST_EXT", "KCK_FWD", "KCK_MODE_DKV", "KCK_MODE_DQ", "KCK_PRELAUNCH",
    "FC1_BWD_MODE", "FC1_WGRAD_CHUNKS", "FC1_WGRAD_IMPL", "FC1_LT_TABLE", "FC2OP_LT_FORCE", "FC2OP_LT_TABLE",
    "LMHEAD_CE_DGRAD", "LMHEAD_CE_FWD", "LMHEAD_CE_TWO_STREAM",
    "QKV_BWD_MODE", "QKV_BWD_ORDER", "QKV_DBIAS_IMPL", "QKV_LT_FORCE", "QKV_LT_TABLE", "QKV_RN", "QKV_RN_BWD",
    "QKV_RN_FWD", "QKV_WGRAD_MODE",
    "RMSN_BWD_GRID", "RMSN_DEFS", "GEAK_RMSN_BWD_GROUP", "GEAK_RMSN_BWD_MODE", "GEAK_RMSN_BWD_NWG",
    "GEAK_RMSN_BWD_ROWS", "GEAK_RMSN_BWD_RPWG", "GEAK_RMSN_FWD_CFG", "GEAK_RMSN_XFLAGS", "GEAK_RMSN_XNAME",
    "ROPE_BT", "ROPE_CPP", "ROPE_FAST", "ROPE_FAST_LAUNCH", "ROPE_HC", "ROPE_NW",
    "SWIGLU_EXTRA_FLAGS", "GEAK_ATTN_AMAX",
)


def _config_snapshot():
    """Pinned-config verification input (Seam 1): what the process ACTUALLY runs with."""
    import sys
    argv, out = sys.argv, {}
    # ARGPARSE SEMANTICS: the LAST occurrence wins, and `--flag=value` is the same as `--flag value`.
    # This used argv.index(k) -- the FIRST occurrence -- which is exactly backwards whenever an
    # adapter writes a default and --extra-train-args appends an override. Measured: an HF 7B leg run
    # with `--extra-train-args "--hf-fsdp full_shard"` captured `hf-fsdp none` while FSDP was plainly
    # active (73.6 GB peak where the DDP leg OOMs at 183.5 GB). The same shape hit Megatron: a leg
    # passing `--lr 0.0` over the adapter's `--lr 1.0e-5` recorded 1.0e-5. Identity recorded a config
    # that did not run, which is worse than recording nothing.
    for k in _IDENTITY_PAIR_KEYS:
        val = None
        for i, tok in enumerate(argv):
            if tok == k and i + 1 < len(argv):
                val = argv[i + 1]
            elif tok.startswith(k + "="):
                val = tok.split("=", 1)[1]
        if val is not None:
            out[k.lstrip("-")] = val
    for k in _IDENTITY_FLAG_KEYS:
        if k in argv:
            out[k.lstrip("-")] = "true"
    # Version the SNAPSHOT at the capture site, not the summary. A v2 sidecar physically does not
    # contain attention_dropout, so re-summarising it cannot recover the value: stamping a newer
    # schema onto it in probe_summarize would produce a digest that CLAIMS to cover dropout and does
    # not. Old dirs stay v2 forever, new dirs are v3, and any v2-vs-v3 pair must be RE-MEASURED.
    # v4 adds the comm-overlap / recompute / attention-backend keys above and moves
    # recompute-granularity from the workload half to the implementation half.
    #
    # UNLIKE v2->v3, THIS DOES NOT REQUIRE A RE-MEASURE, and that is a checked claim rather than a
    # convenience: every key added here is captured only when the flag is PRESENT, and all five are
    # absent from every leg on disk (verified against stackB0929_ref1). An absent flag contributes
    # no key, so a v3 leg and a v4 leg of the same run produce byte-identical identity and impl
    # dicts, hence identical digests. Only legs that actually USE one of these levers become
    # distinguishable -- which is the entire point of adding them.
    #
    # v5 (2026-10-21) is NOT free (nor was v4: it moved impl_digest on every Megatron leg, DECISIONS
    # 4.10b): --lr-decay-iters and --use-distributed-optimizer are
    # PRESENT on current legs (every _cosine* leg, every Megatron leg), so a v4 and a v5 summary of
    # the same run digest differently. The header keeps argv_hash, not argv, so a v4 sidecar cannot
    # be re-captured: a v4-vs-v5 pair must be RE-MEASURED, exactly like v2-vs-v3. The gate and L1
    # both refuse such a pair and say so (training_workflow.js comparable, trajectory_gate verdict).
    out["snapshot_keys"] = "v5"
    # Kernel-selecting env knobs the overlays read (FC1_BWD_MODE, QKV_*, KCK_*, ...): legs running
    # serial vs overlap schedules shared BOTH digests with a 0.64% throughput gap (audit OVL-2).
    # Captured only when SET, as `env:<NAME>` (implementation half), so default legs -- every gated
    # and bundled leg, whose launchers refuse these overrides -- digest exactly as before.
    for k in _KNOBS:
        if k in os.environ:
            v = os.environ[k]
            out["env:" + k] = os.path.basename(v) if k.endswith("_LT_TABLE") else v
    out["world_size"] = os.environ.get("WORLD_SIZE", "1")
    for e in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_USE_HIPBLASLT"):
        if e in os.environ:
            out[e] = os.environ[e]
    # An overlay changes which kernel runs, so it belongs in the IMPLEMENTATION half of the snapshot.
    # Without it two legs differing only by an overlay look byte-identical in the record, and the
    # report cannot say what was actually swapped.
    if os.environ.get("GEAK_OVERLAY"):
        # GEAK_OVERLAY may name ONE manifest or a comma-separated STACK. Hash each and join, so a
        # stacked run has an identity too: before this, a stack made every manifest path unreadable
        # as a single filename and impl_digest silently recorded UNREADABLE_MANIFEST -- i.e. the run
        # that most needs provenance was the one that had none.
        manifests = [m.strip() for m in os.environ["GEAK_OVERLAY"].split(",") if m.strip()]
        out["GEAK_OVERLAY"] = ",".join(os.path.basename(m) for m in manifests)
        # The basename is a LABEL. This is the identity: it moves iff the bytes that will run move.
        out["GEAK_OVERLAY_SHA"] = "+".join(_overlay_content_hash(m) for m in manifests)
    return out


PROBE = None


def install_megatron():
    """The ONE stack-specific binding: Megatron's step boundary + batch site."""
    global PROBE
    if not _ENABLED or PROBE is not None:
        return
    PROBE = _Probe(_DIR)

    import megatron.training.training as _tr
    import megatron.training.utils as _tu

    _orig_train_step = _tr.train_step
    _orig_get_batch = _tu.get_batch_on_this_tp_rank

    def _probed_get_batch(*a, **kw):
        t = time.perf_counter()
        batch = _orig_get_batch(*a, **kw)
        PROBE._batch_ms += (time.perf_counter() - t) * 1000.0
        PROBE.note_batch(batch)
        return batch

    def _probed_train_step(*a, **kw):
        PROBE.step_begin()
        out = _orig_train_step(*a, **kw)
        loss = grad_norm = skipped = None
        try:
            # Megatron returns (loss_dict, skipped_iter, should_checkpoint, should_exit,
            #                   exit_code, grad_norm, num_zeros_in_grad)
            if isinstance(out, tuple):
                if isinstance(out[0], dict):
                    ld = out[0]
                    for k in ("lm loss", "loss"):
                        if k in ld:
                            loss = float(ld[k].item() if hasattr(ld[k], "item") else ld[k])
                            break
                if len(out) > 1:
                    skipped = int(out[1]) if out[1] is not None else None
                if len(out) > 5 and out[5] is not None:
                    grad_norm = float(out[5])
        except Exception:
            pass
        PROBE.step_end(loss=loss, grad_norm=grad_norm, skipped=skipped)
        return out

    _tu.get_batch_on_this_tp_rank = _probed_get_batch
    _tr.get_batch_on_this_tp_rank = getattr(_tr, "get_batch_on_this_tp_rank", _probed_get_batch)
    _tr.train_step = _probed_train_step
    print(f"[geak-probe] installed (rank {PROBE.rank}/{PROBE.world}) -> {PROBE.path}", flush=True)


def install_hf(trainer):
    """The HF transformers / TRL binding: Trainer.training_step for batches, a callback for steps.

    Called explicitly by adapters/hf_sft.py after the trainer exists, rather than by a launcher
    shim, because HF's step boundary lives on an instance (callbacks), not on a module function
    that can be rebound before import.

    USEFUL TOKENS ARE COUNTED AS `labels[:, 1:] != -100`, NOT `labels != -100`. The model shifts
    labels internally, so the first position of every sequence is never a target -- exactly as
    Megatron's loss_mask counts the shifted targets. The unshifted count over-states useful tokens
    by one per sample, which is small and is precisely the kind of small that makes two stacks'
    useful tok/s quietly incomparable.

    TIMING DIFFERENCE FROM MEGATRON, recorded rather than hidden: HF's loop prefetches all of an
    update step's micro-batches (get_batch_samples) BEFORE on_step_begin, so data loading is outside
    the timed window here, while Megatron's train_step pulls its batches inside. With dataloader
    workers this is small, but it is a real asymmetry for any cross-stack step-time comparison.
    """
    global PROBE
    if not _ENABLED or PROBE is not None:
        return
    PROBE = _Probe(_DIR)
    from transformers import TrainerCallback

    _orig_training_step = trainer.training_step

    def _probed_training_step(model, inputs, *a, **kw):
        try:
            labels = inputs.get("labels") if isinstance(inputs, dict) else None
            if labels is not None:
                PROBE.note_batch({"loss_mask": (labels[:, 1:] != -100)})
        except Exception as e:  # measurement must never break the run
            PROBE.pending.setdefault("errors", []).append(repr(e))
        return _orig_training_step(model, inputs, *a, **kw)

    trainer.training_step = _probed_training_step

    class _StepCallback(TrainerCallback):
        def __init__(self):
            self.t_end = None

        def _flush(self, loss=None, grad_norm=None):
            if self.t_end is not None:
                PROBE.step_end(loss=loss, grad_norm=grad_norm, t_end=self.t_end)
                self.t_end = None

        def on_step_begin(self, args, state, control, **kw):
            self._flush()            # a step whose log never came still gets its timing recorded
            PROBE.step_begin()

        def on_step_end(self, args, state, control, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t_end = time.perf_counter()

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            loss = logs.get("loss")
            gn = logs.get("grad_norm")
            self._flush(loss=float(loss) if loss is not None else None,
                        grad_norm=float(gn) if gn is not None else None)

        def on_train_end(self, args, state, control, **kw):
            self._flush()

    trainer.add_callback(_StepCallback())
    print(f"[geak-probe] installed HF binding (rank {PROBE.rank}/{PROBE.world}) -> {PROBE.path}",
          flush=True)


STEP_BINDINGS = {"megatron": install_megatron, "hf_trl": install_hf}
