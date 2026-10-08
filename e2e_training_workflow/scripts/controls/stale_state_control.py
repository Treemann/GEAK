#!/usr/bin/env python3
"""POSITIVE CONTROL for the L2 soak: a defect L0 **structurally cannot see**.

WHY A SECOND CONTROL EXISTS. The soak's first positive control was a x1.004 multiplicative bias on
the gradient, applied on every invocation. Two soaks (5 legs x 300 steps, then 6 more legs) failed to
reject it, and the diagnosis in DECISIONS.md 3.6c is that the control was the WRONG CLASS: L0
rejects that same defect in 11.5 s (`dgrad scale_bias +3.80e-03` against a 2e-3 tolerance). A soak
whose control is op-local numerics will keep failing its control while the system is in fact well
defended, because the defect never reaches the soak -- L0 eats it first.

So the control for a ~10 h instrument must be a defect that L0 cannot catch **by construction**, not
one that L0 merely happens to miss. That is a much stronger requirement than "we tried L0 and it
passed", and it is the requirement this file is built to meet.

THE MECHANISM. L0 evaluates a frozen microbatch in isolation -- `l0_gate.py:8`, "an INDEPENDENT pure
PyTorch reference, in fp32, on the same frozen inputs", and `l0_gate.py:273`, "same op, same frozen
inputs, ONE call". Every quantity it compares is a function of that single call's inputs alone.

A defect whose output depends on the PREVIOUS call's inputs is therefore invisible to it, and the
invisibility is exact rather than probabilistic:

    gscale_eff[n] = gscale[n] + alpha * (gscale[n - 1] - gscale[n])

  * on the first call there is no n-1, so the result is `gscale[0]`, bit-identical to correct;
  * on any repeat of the SAME input, gscale[n-1] == gscale[n], so the difference is exactly 0.0,
    alpha * 0.0 is exactly 0.0, and adding it returns the operand unchanged -- for EVERY value of
    alpha, including alpha = 1.0.

The lerp form is load-bearing and was chosen after the self-test rejected the obvious one. Written
as `(1-a)*cur + a*prev` the identity holds in algebra but NOT in floating point: at alpha=0.05,
0.95*3.5 + 0.05*3.5 = 3.4999999999999996. That residue is ~1 ulp and some 13 orders of magnitude
under L0's 2e-3 tolerance, so the control would have worked either way -- but the CLAIM would have
been false as written, and a structural-invisibility argument that is only approximately true is the
kind of overclaim this layer has been burned by before. `cur + a*(prev - cur)` makes it literal.

L0 can only ever present one of those two situations. Its measured error against the fp32 oracle is
exactly 0.0 at full staleness. There is no magnitude at which it starts to see this, which is what
"structurally" means and is why this control is the right size of hammer for a 10 h gate.

In real training consecutive steps hold different batches, so from step 2 onward the optimizer is
fed a gradient that is partly the previous step's. Accumulated over ~19 200 invocations per leg,
that is precisely the class DECISIONS.md 3.6c reserves for the soak: cross-step state.

WHY THIS IS A REAL DEFECT AND NOT A STRAW MAN. Reusing a workspace buffer across invocations without
clearing it is one of the most common ways a hand-written persistent kernel goes wrong, and fp8
training does this ON PURPOSE -- amax-history scaling scales the current step by a factor derived
from previous steps. A kernel that reuses a scale one step too long is a plausible bug, not an
invented one, and `alpha` is exactly "what fraction of the stale value survives".

WHAT alpha MEANS AND WHY IT IS A KNOB. The soak's sensitivity is a number to be MEASURED, not
assumed. alpha sweeps the defect from "none" (0.0) to "fully one step stale" (1.0), so the honest
deliverable is a detection floor -- "the soak rejects cross-step staleness at alpha >= X" -- rather
than a bare pass. Tuning alpha to find that floor is legitimate; retuning the soak's THRESHOLD after
seeing the result is not, and the thresholds stay pre-registered in knowledge/gate_convergence.md.

Usage:
  stale_state_control.py <impl.py> <alpha>     # patch in place, refuse if the site is gone
  stale_state_control.py --self-test           # prove the invisibility claim, no GPU needed
"""
from __future__ import annotations

import sys

# The exact line the control hooks, shared with scripts/l2_soak.sh's x1.004 control. `gscale` feeds
# dlogits -> dh -> dgrad AND wgrad, so it is what the optimizer actually sees. Biasing the forward
# loss instead would shift the REPORTED number while training proceeded identically -- a cosmetic
# control that looks like it works. See the long comment in l2_soak.sh for the two wrong choices.
SITE = "gscale = (g_f.index_select(0, idx) * mg).contiguous()"

PATCH = '''gscale = (g_f.index_select(0, idx) * mg).contiguous()
        # ---- POSITIVE CONTROL: cross-step staleness, alpha={alpha} -------------------------
        # Blend in the PREVIOUS invocation's gscale, emulating a workspace buffer that is reused
        # without being cleared. Invisible to L0 by construction: with one frozen input there is no
        # previous call, and with a repeated identical input the blend is an algebraic identity.
        _prev = globals().get("_GEAK_STALE_PREV")
        _a = {alpha}
        if _prev is not None and _prev.dtype == gscale.dtype:
            # Shapes differ step to step (the unmasked-token count varies), so blend over the common
            # prefix and leave the tail current -- which is what reusing a fixed buffer really does.
            _k = min(_prev.numel(), gscale.numel())
            if _k > 0:
                gscale = gscale.clone()
                # lerp form, NOT (1-a)*cur + a*prev: when prev == cur the difference is exactly 0.0,
                # so this is bit-identical for every alpha. The other form leaves ~1 ulp behind.
                gscale[:_k] = gscale[:_k] + _a * (_prev[:_k].to(gscale.device) - gscale[:_k])
        globals()["_GEAK_STALE_PREV"] = gscale.detach().clone()
        # ---- end positive control ----------------------------------------------------------
'''


def patch(path: str, alpha: float) -> None:
    src = open(path).read()
    if SITE not in src:
        # Same refusal as the x1.004 control: a control that silently fails to install is a second
        # candidate leg wearing a control's name, and it would read as "the soak passed".
        sys.exit("POSITIVE CONTROL: gradient injection point not found in impl.py -- refusing to "
                 "run a control that would silently be a second candidate leg")
    if "_GEAK_STALE_PREV" in src:
        sys.exit("POSITIVE CONTROL: already patched -- refusing to stack two controls")
    open(path, "w").write(src.replace(SITE, PATCH.format(alpha=alpha).rstrip(), 1))
    print(f">>> POSITIVE CONTROL: cross-step staleness alpha={alpha} injected at gscale")
    print(">>> class: cross-step state. L0 cannot see this one -- see --self-test for the proof.")


def self_test() -> int:
    """Demonstrate the invisibility claim rather than asserting it.

    Pure arithmetic on lists: the claim is algebraic, so it needs no GPU, no torch and no model, and
    it therefore runs in preflight where a GPU claim could not.
    """
    fails = 0

    def ok(cond, msg):
        nonlocal fails
        print(("  ok: " if cond else "  FAIL: ") + msg)
        if not cond:
            fails += 1

    def blend(cur, prev, a):
        if prev is None:
            return list(cur)
        k = min(len(prev), len(cur))
        # lerp form -- see the module docstring on why this, and not (1-a)*cur + a*prev.
        return [cur[i] + a * (prev[i] - cur[i]) if i < k else cur[i] for i in range(len(cur))]

    g = [1.0, -2.0, 3.5, 0.25]
    for a in (0.05, 0.5, 1.0):
        # (1) What L0 does: one call, no previous state.
        ok(blend(g, None, a) == g, f"alpha={a}: first call is bit-identical (L0 makes exactly one)")
        # (2) What L0 would do even if it called twice on its frozen input.
        ok(blend(g, g, a) == g, f"alpha={a}: repeated identical input is an identity, so error = 0")
        # (3) What real training does: the next step is a different batch.
        h = [2.0, -1.0, 0.5, 4.0]
        ok(blend(h, g, a) != h or a == 0, f"alpha={a}: a DIFFERENT batch is corrupted (training)")

    # The defect must survive to the optimizer, not just exist. gscale feeds dgrad and wgrad both.
    ok(SITE.startswith("gscale ="), "the hook is on gscale, which reaches dgrad and wgrad")
    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        return 1
    print("all checks passed: L0-invisibility is algebraic, not empirical")
    return 0


def main() -> None:
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    patch(sys.argv[1], float(sys.argv[2]))


if __name__ == "__main__":
    main()
