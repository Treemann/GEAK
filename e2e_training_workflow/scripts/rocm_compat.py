"""ROCm compatibility shim + launcher for bare upstream Megatron-LM.

Upstream Megatron calls megatron.training.initialize._compile_dependencies() unconditionally, which
reaches megatron.legacy.fused_kernels.load() -> _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME).
On ROCm CUDA_HOME is None -> TypeError: NoneType + str. The path is CUDA-only (it builds .cu fused
softmax kernels), so it is skipped, exactly as Primus does
(primus/backends/megatron/patches/runtime_hooks_patches.py, "ROCm compatibility").

This is a manual stand-in for what the workflow will later install as a reversible PYTHONPATH overlay.
Usage:  python rocm_compat_launch.py <script.py> [args...]
"""
import runpy
import sys

import megatron.training.initialize as _mi


def _skip_compile_dependencies():
    print("[rocm-compat] skipped _compile_dependencies() (CUDA-only fused-kernel build)", flush=True)


_skip_compile_dependencies._rocm_compat = True
_mi._compile_dependencies = _skip_compile_dependencies

# ---------------------------------------------------------------------------
# Fix 2: upstream --sft + THD packing is self-inconsistent on this commit.
# SFTDataset returns a PER-SAMPLE 1D cu_seqlens; the default collate stacks it into
# [MBS, n+1], and get_thd_batch_on_this_cp_rank passes that 2D tensor straight into
# PackedSeqParams. Megatron's own rope helpers then do cu_seqlens[-1] (1D assumption):
#   fused     -> RuntimeError: expected 1D tensor
#   non-fused -> RuntimeError: Boolean value of Tensor with more than one value is ambiguous
# For micro_batch_size == 1 the batch dim is vacuous, so squeeze it and THD is coherent.
# For MBS > 1 a correct fix must re-base each row by its sample offset AND flatten tokens to
# one sequence; that is a real change in semantics, so refuse loudly instead of guessing.
# ---------------------------------------------------------------------------
import megatron.core.utils as _mcu

_orig_thd = _mcu.get_thd_batch_on_this_cp_rank


def _thd_squeeze_batch_dim(batch, cu_seqlens, cu_seqlens_padded, max_seqlen, *a, **kw):
    if cu_seqlens is not None and cu_seqlens.dim() == 2:
        if cu_seqlens.size(0) != 1:
            raise RuntimeError(
                "[rocm-compat] packed SFT (THD) with micro_batch_size=%d is not supported: "
                "cu_seqlens arrives as %s and merging rows would change sequence semantics. "
                "Use micro_batch_size=1 and raise global_batch_size via grad accumulation."
                % (cu_seqlens.size(0), tuple(cu_seqlens.shape))
            )
        cu_seqlens = cu_seqlens[0]
        if cu_seqlens_padded is not None and cu_seqlens_padded.dim() == 2:
            cu_seqlens_padded = cu_seqlens_padded[0]
    return _orig_thd(batch, cu_seqlens, cu_seqlens_padded, max_seqlen, *a, **kw)


_thd_squeeze_batch_dim._rocm_compat = True
_mcu.get_thd_batch_on_this_cp_rank = _thd_squeeze_batch_dim
print("[rocm-compat] patched get_thd_batch_on_this_cp_rank (squeeze vacuous batch dim)", flush=True)

# ---------------------------------------------------------------------------
# Fix 3: packed SFT trains each conversation's last token toward the NEXT conversation.
# SFTDataset shifts labels across the whole pack (labels = pack_targets[1:]), so the last position of
# conversation i is given conversation i+1's first token as its target. Unpacked, that position's
# target is the padding token and is masked. Measured on a real 5-conversation pack of the 7B
# dataset: those targets cost 10.6, 10.1, 10.7 and 13.4 nats (the model expects <|endoftext|>, the
# label says <|im_start|>) -- 0.1% of the tokens, 0.37% of the loss, and a training signal to keep
# going into a new conversation instead of stopping.
# Masking loss_mask at every INTERNAL boundary (cu_seqlens[1:-1] - 1) makes a packed sample's
# objective exactly the unpacked one. An unpacked sample has no internal boundary, so this is a
# no-op on every existing workload.
# ---------------------------------------------------------------------------
try:
    import megatron.training.datasets.sft_dataset as _sftds

    _orig_sft_getitem = _sftds.SFTDataset.__getitem__

    def _sft_getitem_mask_boundaries(self, idx):
        item = _orig_sft_getitem(self, idx)
        cu = item.get("cu_seqlens")
        lm = item.get("loss_mask")
        if cu is not None and lm is not None and cu.numel() > 2:
            for b in cu[1:-1].tolist():
                if 0 < b <= lm.numel():
                    lm[b - 1] = 0.0
        return item

    _sft_getitem_mask_boundaries._rocm_compat = True
    _sftds.SFTDataset.__getitem__ = _sft_getitem_mask_boundaries
    print("[rocm-compat] patched SFTDataset: packed-conversation boundary targets are masked", flush=True)
except Exception as _e:  # an SFT-less Megatron build has nothing to patch
    print(f"[rocm-compat] SFT boundary mask not installed: {_e!r}", flush=True)

# ---------------------------------------------------------------------------
# GEAK measurement probe (Seam 2). No-op unless GEAK_PROBE_DIR is set, so a run without it is
# unaffected. Installed here, before the trainer is imported, exactly as the reversible overlay
# will do it later.
# ---------------------------------------------------------------------------
import os as _os

if _os.environ.get("GEAK_PROBE_DIR"):
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import train_probe as _probe

    _probe.install_megatron()

    # Shape/oracle capture rides the same install point; no-op unless GEAK_CAPTURE_* is set.
    try:
        import capture_train_shapes as _cap

        if _cap.ENABLED:
            _cap.install()
            _orig_step_end = _probe._Probe.step_end

            def _step_end(self, *a, **kw):
                _cap.note_step(self.step)
                return _orig_step_end(self, *a, **kw)

            _probe._Probe.step_end = _step_end
    except Exception as _e:  # capture is optional; never break a measurement run
        print(f"[geak-capture] not installed: {_e!r}", flush=True)

    # The LM-head + CE COMPOSITE capture. Separate module because the op spans two call sites and
    # no single autograd.Function covers it; no-op unless GEAK_LMHEAD_CAPTURE_DIR is set.
    try:
        import capture_lmhead_ce as _lmcap

        if _lmcap.ENABLED:
            _lmcap.install()
            _orig_step_end_lm = _probe._Probe.step_end

            def _step_end_lm(self, *a, **kw):
                _lmcap.note_step(self.step)
                return _orig_step_end_lm(self, *a, **kw)

            _probe._Probe.step_end = _step_end_lm
    except Exception as _e:
        print(f"[geak-lmhead] not installed: {_e!r}", flush=True)

    # The MLP fc1 + SwiGLU COMPOSITE capture. Separate module for the same reason as the LM head:
    # the op spans a TE autograd.Function and a Megatron fusion, and no single Function covers it.
    # No-op unless GEAK_MLP_CAPTURE_DIR is set.
    try:
        import capture_mlp_fc1_swiglu as _mlpcap

        if _mlpcap.ENABLED:
            _mlpcap.install()
            _orig_step_end_mlp = _probe._Probe.step_end

            def _step_end_mlp(self, *a, **kw):
                _mlpcap.note_step(self.step)
                return _orig_step_end_mlp(self, *a, **kw)

            _probe._Probe.step_end = _step_end_mlp
    except Exception as _e:
        print(f"[geak-mlp] not installed: {_e!r}", flush=True)

    # The unfused RoPE (q AND k of one layer) capture. The op is a PAIR of calls of one function,
    # so no single-call capture covers it. No-op unless GEAK_ROPE_CAPTURE_DIR is set.
    try:
        import capture_rope_qk as _ropecap

        if _ropecap.ENABLED:
            _ropecap.install()
            _orig_step_end_rope = _probe._Probe.step_end

            def _step_end_rope(self, *a, **kw):
                _ropecap.note_step(self.step)
                return _orig_step_end_rope(self, *a, **kw)

            _probe._Probe.step_end = _step_end_rope
    except Exception as _e:
        print(f"[geak-rope] not installed: {_e!r}", flush=True)

    # The fc1 RMSNorm (fwd+bwd) capture, on the accepted stack: hooks transformer_engine_torch's
    # rmsnorm_fwd/bwd and keeps only calls made from the mlp_fc1_gemm overlay's kernel. Step noted
    # AFTER the probe's own step_end (so it is the 1-based number of the step just finished), which
    # lets the capture finalise at the end of the capture step. No-op unless GEAK_RMSN_CAPTURE_DIR.
    try:
        import capture_fc1_rmsnorm as _rmsncap

        if _rmsncap.ENABLED:
            _rmsncap.install()
            _orig_step_end_rmsn = _probe._Probe.step_end

            def _step_end_rmsn(self, *a, **kw):
                r = _orig_step_end_rmsn(self, *a, **kw)
                _rmsncap.note_step(self.step)
                return r

            _probe._Probe.step_end = _step_end_rmsn
    except Exception as _e:
        print(f"[geak-rmsn] not installed: {_e!r}", flush=True)

    # The dist-opt grad tail (pre-RS norm check + grad scale) capture: hooks
    # _ParamAndGradBucketGroup.start_grad_sync. Step noted AFTER the probe's own step_end, like the
    # rmsnorm capture. No-op unless GEAK_GRADTAIL_CAPTURE_DIR is set.
    try:
        import capture_grad_tail as _gtcap

        if _gtcap.ENABLED:
            _gtcap.install()
            _orig_step_end_gt = _probe._Probe.step_end

            def _step_end_gt(self, *a, **kw):
                r = _orig_step_end_gt(self, *a, **kw)
                _gtcap.note_step(self.step)
                return r

            _probe._Probe.step_end = _step_end_gt
    except Exception as _e:
        print(f"[geak-gradtail] not installed: {_e!r}", flush=True)

    # Reversible rebind + engagement proof. No-op unless GEAK_OVERLAY points at a manifest.
    try:
        import overlay_train as _ovl

        if _ovl.ENABLED:
            _ovl.install()
    except Exception as _e:
        print(f"[geak-overlay] not installed: {_e!r}", flush=True)

if len(sys.argv) < 2:
    raise SystemExit("usage: rocm_compat_launch.py <script.py> [args...]")
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
