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

Enable with GEAK_PROBE_DIR=<dir>. Absent -> import is a no-op and the run is unaffected.
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

    # ---- step hooks -------------------------------------------------------------------
    def step_begin(self):
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


def _argv_hash():
    import hashlib
    import sys
    return hashlib.sha256(" ".join(sys.argv).encode()).hexdigest()[:16]


def _config_snapshot():
    """Pinned-config verification input (Seam 1): what the process ACTUALLY runs with."""
    import sys
    keys = ("--micro-batch-size", "--global-batch-size", "--seq-length", "--tensor-model-parallel-size",
            "--pipeline-model-parallel-size", "--context-parallel-size", "--recompute-granularity",
            "--attention-backend", "--sft-tokenizer-prompt-format")
    argv, out = sys.argv, {}
    for k in keys:
        if k in argv:
            i = argv.index(k)
            if i + 1 < len(argv):
                out[k.lstrip("-")] = argv[i + 1]
    out["world_size"] = os.environ.get("WORLD_SIZE", "1")
    for e in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_USE_HIPBLASLT"):
        if e in os.environ:
            out[e] = os.environ[e]
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
