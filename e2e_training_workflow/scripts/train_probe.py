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
        self.fh = open(self.path, "a", buffering=1)
        self.step = 0
        self.prof = None
        self.prof_range = _profile_range()
        self.prof_ranks = _profile_ranks()
        # Batch facts land here from the batch hook, are consumed by the step hook. A step spans
        # several micro-batches (grad accumulation), so they accumulate and reset per step.
        self.pending = {"useful_tokens": 0, "total_tokens": 0, "micro_batches": 0}
        self._t0 = None
        self._emit({"record": "header", "rank": self.rank, "world_size": self.world,
                    "local_rank": self.local_rank, "t_wall": time.time(),
                    "device": torch.cuda.get_device_name(self.local_rank)
                    if torch.cuda.is_available() else "cpu",
                    "argv_hash": _argv_hash(), "config": _config_snapshot()})

    def _emit(self, obj):
        self.fh.write(json.dumps(obj) + "\n")

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

    def step_end(self, loss=None, grad_norm=None, skipped=None):
        if self._t0 is None:
            return
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dur_ms = (time.perf_counter() - self._t0) * 1000.0
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
    "--pipeline-model-parallel-size", "--context-parallel-size", "--recompute-granularity",
    "--sft-tokenizer-prompt-format",
    # regularization + optimizer: these move the loss/grad-norm trajectory L1 gates on
    "--attention-dropout", "--hidden-dropout", "--weight-decay", "--clip-grad",
    "--lr", "--min-lr", "--lr-decay-style", "--lr-warmup-iters",
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
)


def _config_snapshot():
    """Pinned-config verification input (Seam 1): what the process ACTUALLY runs with."""
    import sys
    argv, out = sys.argv, {}
    for k in _IDENTITY_PAIR_KEYS:
        if k in argv:
            i = argv.index(k)
            if i + 1 < len(argv):
                out[k.lstrip("-")] = argv[i + 1]
    for k in _IDENTITY_FLAG_KEYS:
        if k in argv:
            out[k.lstrip("-")] = "true"
    # Version the SNAPSHOT at the capture site, not the summary. A v2 sidecar physically does not
    # contain attention_dropout, so re-summarising it cannot recover the value: stamping a newer
    # schema onto it in probe_summarize would produce a digest that CLAIMS to cover dropout and does
    # not. Old dirs stay v2 forever, new dirs are v3, and any v2-vs-v3 pair must be RE-MEASURED.
    out["snapshot_keys"] = "v3"
    out["world_size"] = os.environ.get("WORLD_SIZE", "1")
    for e in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_USE_HIPBLASLT"):
        if e in os.environ:
            out[e] = os.environ[e]
    # An overlay changes which kernel runs, so it belongs in the IMPLEMENTATION half of the snapshot.
    # Without it two legs differing only by an overlay look byte-identical in the record, and the
    # report cannot say what was actually swapped.
    if os.environ.get("GEAK_OVERLAY"):
        out["GEAK_OVERLAY"] = os.path.basename(os.environ["GEAK_OVERLAY"])
        # The basename is a LABEL. This is the identity: it moves iff the bytes that will run move.
        out["GEAK_OVERLAY_SHA"] = _overlay_content_hash(os.environ["GEAK_OVERLAY"])
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
        batch = _orig_get_batch(*a, **kw)
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


STEP_BINDINGS = {"megatron": install_megatron}
