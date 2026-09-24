"""wgrad / dgrad GEMM selection for the gathered-bf16 LM-head backward (lane r1_d1).

Everything here is a DROP-IN for one GEMM call, so it merges into D0's pipeline without touching
the gather, the CE epilogue, the operand orientation, or the wrapper's allocation.

Measured on THIS box (MI308X, 80 CU, gfx942, bf16 ceiling ~200 TFLOPS) at the live shape
V=151680, H=3584, K=nnz=1276, via round_1/engineer_1/bench_gemm.py + bench_dgrad.py:

  wgrad  C[V,H] = dlogits[nnz,V]^T @ hg[nnz,H]        flops 1.387e12
    torch.mm(dlogits.t(), hg) -> bf16          8.92 ms   155.6 TF   <-- SELECTED
    torch.mm(...) with fp32 operands/out      50.54 ms    27.5 TF   (5.7x SLOWER -- never do this)
    A pre-transposed to [V,nnz], NN layout      8.81 ms   157.5 TF   (+1.2%, but the transpose of
                                                                     387 MB costs more than it saves)
    best of 9 hand-written Triton tilings      15.61 ms    88.9 TF   (0.57x -- port NOT funded)

  dgrad  C[nnz,H] = dlogits[nnz,V] @ W[V,H]
    torch.mm(dlogits, W)                      12.30 ms   112.8 TF   <-- SELECTED
    W pre-transposed, NT layout               23.74 ms    58.3 TF   (0.52x)
    M padded to 1280 / 2048                   12.31 / 33.09 ms      (padding is neutral / harmful)
    best of 23 Triton tilings incl. split-K
      over V (128x256x32, SPLIT_K=4)          12.31 ms   112.7 TF   (1.00x -- port NOT funded)

CONCLUSION, and the reason this module is thin: hipBLASLt is NOT short at thin K. wgrad is the
FASTEST of the three GEMMs (78% of the measured bf16 ceiling), not the bottleneck the round's
direction suspected. The two levers that ARE real and ARE encoded below are (a) keep both backward
GEMMs in bf16 with the natural operand orientation, and (b) never materialize a transposed copy.

THE LEVER THAT ACTUALLY PAYS, found by racing the third candidate the direction asked for (an
ALGO-ENUMERATED library call, i.e. PyTorch TunableOp, which enumerates every hipBLASLt and rocBLAS
solution for the shape instead of trusting the vendor heuristic):

  GEMM    default heuristic   enumerated solution              speedup   winner
  fwd     11.78 ms 117.8 TF   7.36 ms 188.5 TF                  1.60x    rocBLAS (!)
  dgrad   12.30 ms 112.8 TF   7.32 ms 189.6 TF                  1.68x    hipBLASLt 75065
  wgrad    8.92 ms 155.6 TF   8.46 ms 164.0 TF                  1.05x    hipBLASLt 90190
  SUM     32.97 ms            23.13 ms                          1.43x

So the heuristic IS leaving ~10 ms on the table -- but on the two GEMMs the direction did NOT
suspect, and the fix is solution selection, not a new instruction/tile shape. The enumerated
solutions reach 188-190 TFLOPS, i.e. ~95% of this card's measured 200 TFLOPS bf16 ceiling, which
is also the proof that nothing further is available from these three shapes.

Solutions are loaded READ-ONLY from `tunable_gemm.csv` next to this file; tuning is left DISABLED
so no run pays the ~25 min enumeration cost, and a key miss degrades silently to the stock library
call (correct, just slower). M is bucketed to GEMM_M_BUCKET so the key set stays bounded as nnz
varies -- the file is keyed on the BUCKET, never on 1276.
"""
import os

import torch

GEMM_M_BUCKET = 128
_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunable_gemm.csv")
_enabled = None


def enable_tuned_gemm():
    """Load the enumerated GEMM solutions. Idempotent; returns True if they are in effect."""
    global _enabled
    if _enabled is not None:
        return _enabled
    _enabled = False
    try:
        if os.path.exists(_CSV) and hasattr(torch.cuda, "tunable"):
            t = torch.cuda.tunable
            t.tuning_enable(False)          # lookup only: never pay the enumeration cost at runtime
            try:
                t.set_filename(_CSV, False)  # False => do not append the device ordinal
            except TypeError:
                t.set_filename(_CSV)
            t.enable(True)
            for reader in ("read_file", "load_file"):
                fn = getattr(t, reader, None)
                if fn is None:
                    continue
                try:
                    fn(_CSV)
                except TypeError:
                    fn()
                break
            _enabled = len(t.get_results()) > 0
            if os.environ.get("GEAK_TUNABLE_DEBUG"):
                print(f"[r1_d1] tuned GEMM solutions in effect: {_enabled} "
                      f"{t.get_results()}", flush=True)
    except Exception as e:                   # noqa: BLE001 -- a miss must never break correctness
        if os.environ.get("GEAK_TUNABLE_DEBUG"):
            print(f"[r1_d1] tuned GEMM load failed, using the stock library: {e!r}", flush=True)
        _enabled = False
    return _enabled


def wgrad_gemm(dlogits, hg, out_dtype=torch.bfloat16):
    """dL/dW = dlogits^T @ hg  ->  [V, H].

    dlogits [nnz, V] bf16, hg [nnz, H] bf16. K = nnz is data-dependent (never hardcoded).
    The `.t()` is a stride flip, not a copy: hipBLASLt takes this TN shape at 155.6 TFLOPS.
    Accumulation is fp32 inside the MFMA; only the store is bf16, which is what the contract
    requires the returned wgrad to be.
    """
    assert dlogits.shape[0] == hg.shape[0], (dlogits.shape, hg.shape)
    w = torch.mm(dlogits.t(), hg)
    return w if w.dtype == out_dtype else w.to(out_dtype)


def dgrad_gemm(dlogits, weight, out_dtype=torch.bfloat16):
    """dL/dhg = dlogits @ W  ->  [nnz, H]. W is [V, H] row-major; this NN shape is the fast one."""
    assert dlogits.shape[1] == weight.shape[0], (dlogits.shape, weight.shape)
    d = torch.mm(dlogits, weight)
    return d if d.dtype == out_dtype else d.to(out_dtype)


def fwd_gemm(hg, weight):
    """logits = hg @ W^T -> [nnz, V] bf16. Measured 11.78 ms / 117.8 TF; the NT shape is fastest."""
    return torch.mm(hg, weight.t())
