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
