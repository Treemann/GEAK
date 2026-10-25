#!/usr/bin/env python3
"""verify_gate.py recomputes a gate from the leg DIRECTORIES (audit 2026-10-22 ORCH-1).

Synthetic legs: a ref and a cand whose steps are 3% faster on the same batches, with proof files for
the overlay the cand adds. Checks: the real gate (gate_eval.js) calls it a win; engagement is proven
only from the ADDED overlay's proofs on every rank; a fallback counter or a missing rank breaks it.
Skips (exit 0) when no node binary is available, since the gate is JavaScript.
"""
import glob, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VG = os.path.join(os.path.dirname(HERE), "verify_gate.py")
node = os.environ.get("NODE") or shutil.which("node") or next(iter(sorted(
    glob.glob(os.path.expanduser("~/.cursor-server/bin/*/node")) +
    glob.glob(os.path.expanduser("~/.cursor-server/bin/*/*/node")))), None)
if not node:
    print("SKIP: no node binary"); sys.exit(0)
fails = 0
def ok(c, m):
    global fails
    print(("  ok: " if c else "  FAIL: ") + m); fails += 0 if c else 1

IMPL = {"attention-backend": "fused", "NVTE_FUSED_ATTN": "1"}
def leg(root, name, speed, overlay=None, nranks=8, fallback=0, drop_rank=None, loss_rel=0.0, coverage=None):
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "probe"))
    impl = dict(IMPL, **({"GEAK_OVERLAY": "m.json", "GEAK_OVERLAY_SHA": "k1"} if overlay else {}))
    cfg = {"seq-length": "4096", "snapshot_keys": "v5", **impl}
    for rk in range(nranks):
        with open(os.path.join(d, "probe", f"probe_rank{rk}.jsonl"), "w") as f:
            f.write(json.dumps({"record": "header", "rank": rk, "world_size": nranks, "argv_hash": "a", "config": cfg,
                                **({"coverage": coverage} if coverage else {})}) + "\n")
            t = 0.0
            for i in range(30):
                dur = (40000 if i == 0 else 2300 + 7 * ((i * 37) % 5)) * speed
                t += dur / 1000 + 0.001
                f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": dur, "useful_tokens": 8000,
                                    "total_tokens": 8192, "peak_reserved_mb": 80000, "t_wall": t,
                                    "loss": (3.0 - 0.01 * i) * (1 + (loss_rel if i == 0 else 0.0)), "grad_norm": 5.0}) + "\n")
    json.dump({"status": "ok", "hang_detected": False, "occupancy_verdict": {"verdict": "clean"}},
              open(os.path.join(d, "bench_summary.json"), "w"))
    if overlay:
        os.makedirs(os.path.join(d, overlay))
        for rk in range(nranks):
            if rk == drop_rank:
                continue
            json.dump({"rank": rk, "forward": 56, "backward": 56}, open(os.path.join(d, overlay, f"engagement.json.rank{rk}"), "w"))
            json.dump({"rank": rk, "candidate_fwd": 56, "candidate_bwd": 56, "fallback_fwd": fallback},
                      open(os.path.join(d, overlay, f"engagement_fused.json.rank{rk}"), "w"))
    return d

def run(ref, cand, floor=("--floor", "0.25")):
    r = subprocess.run([sys.executable, VG, ref, cand, "--nproc", "8", *floor],
                       capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE=node))
    return json.loads(r.stdout)

root = tempfile.mkdtemp(prefix="vgtest_")
try:
    ref = leg(root, "ref", 1.0)
    good = run(ref, leg(root, "cand", 0.97, overlay="attn"))
    ok(good["verdict"]["klass"] == "win", f"a 3% faster cand on the same batches is a win (got {good['verdict'].get('klass')})")
    ok(good["engagement"]["forward"] and good["engagement"]["backward"] and good["engagement"]["added"] == ["attn"],
       "engagement proven from the ADDED overlay's proof files on every rank")
    fb = run(ref, leg(root, "cand_fb", 0.97, overlay="attn", fallback=3))
    ok(fb["engagement"]["forward"] is False, "a non-zero fallback counter breaks the engagement proof")
    miss = run(ref, leg(root, "cand_miss", 0.97, overlay="attn", drop_rank=5))
    ok(miss["engagement"]["forward"] is False, "a rank without a proof breaks the engagement proof")
    none = run(ref, leg(root, "cand_none", 0.97))
    ok(none["engagement"]["forward"] is False and "no overlay proof" in none["engagement"].get("why", ""),
       "a cand that adds no proof is not engaged")
    # A wrapper that DELEGATES every call: wrapper forward/backward > 0 but the kernel's own counters 0.
    dlg = leg(root, "cand_dlg", 0.97, overlay="attn")
    for f in glob.glob(os.path.join(dlg, "attn", "engagement_fused.json.rank*")):
        r = json.load(open(f)); r["candidate_fwd"] = r["candidate_bwd"] = 0; json.dump(r, open(f, "w"))
    ok(run(ref, dlg)["engagement"]["forward"] is False,
       "a wrapper whose kernel never ran (candidate counters 0) is NOT engaged (wrapper counts are not summed in)")
    # HF naming: <slot>/engagement_<slot>_cand.json.rank* with wrapper-only counters.
    hf = leg(root, "cand_hf", 0.97)
    os.makedirs(os.path.join(hf, "qkvo"))
    for rk in range(8):
        json.dump({"rank": rk, "replacement": "/work/overlays/x/hf_attn_qkvo_proj/impl.py:entry", "forward": 56, "backward": 56},
                  open(os.path.join(hf, "qkvo", f"engagement_qkvo_cand.json.rank{rk}"), "w"))
    e = run(ref, hf)["engagement"]
    ok(e["forward"] and e["added"] == ["qkvo"], "HF proof naming (engagement_<slot>_cand.json.rank*) is read")
    # Same-slot REPLACEMENT: the ref already has an 'attn' proof from a different overlay.
    ref_inc = leg(root, "ref_inc", 1.0, overlay="attn")
    for f in glob.glob(os.path.join(ref_inc, "attn", "engagement.json.rank*")):
        r = json.load(open(f)); r["replacement"] = "/work/overlays/old/attn_v1/impl.py:entry"; json.dump(r, open(f, "w"))
    cand_new = leg(root, "cand_new", 0.97, overlay="attn")
    for f in glob.glob(os.path.join(cand_new, "attn", "engagement.json.rank*")):
        r = json.load(open(f)); r["replacement"] = "/work/overlays/new/attn_v2/impl.py:entry"; json.dump(r, open(f, "w"))
    e2 = run(ref_inc, cand_new)["engagement"]
    ok(e2["added"] == ["attn"] and e2["forward"], "a candidate replacing an incumbent overlay in the SAME slot is judged, not refused")
    via_probe = run(os.path.join(ref, "probe"), os.path.join(root, "cand", "probe"))
    ok(via_probe["verdict"]["klass"] == "win", "probe/ paths are accepted as leg paths")
    # Rebuilt compiled artifacts (*.so) are not identity: same sources, different .so hash -> unchanged.
    sys.path.insert(0, os.path.dirname(HERE))
    import verify_gate as _vg
    mk = lambda so: {"0": [{"replacement": "/work/overlays/a/attn/impl.py:f",
                            "kernel_sha256": {"kernel.py": "k1", "libx_123.so": so}}]}
    ok(_vg._identity(mk("aaa")) == _vg._identity(mk("bbb")), "identity ignores rebuilt .so hashes")
    ok(_vg._identity(mk("aaa")) != _vg._identity({"0": [{"replacement": "/x/attn/impl.py:f",
                                                          "kernel_sha256": {"kernel.py": "k2"}}]}),
       "...but a changed SOURCE hash is still a changed overlay")
    # Carried overlays (DECISIONS 4.55): a slot the ref runs must stay engaged in the cand, for any kind.
    ref_c = leg(root, "ref_carry", 1.0, overlay="attn")
    ok(run(ref_c, leg(root, "cand_carry_ok", 0.97, overlay="attn"))["carried"]["ok"] is True,
       "carried: the ref's overlay still engaged in the cand -> ok")
    lostc = run(ref_c, leg(root, "cand_carry_lost", 0.97, overlay="attn", fallback=3360))["carried"]
    ok(lostc["ok"] is False and lostc["lost"] == ["attn"],
       "carried: the ref's overlay falling back in the cand (a lever switched it off) is reported as lost")
    ok(run(ref, leg(root, "cand_carry_none", 0.97, overlay="attn"))["carried"] == {"slots": {}, "lost": [], "ok": True},
       "carried: a ref with no overlay carries nothing")
    # gate_eval's per-stack floor default (DECISIONS 4.54): Megatron 0.30%, HF 0.25%, explicit wins.
    import subprocess
    nul = run(ref, leg(root, "null", 1.0), floor=())
    def band(args):
        spec = {"ref": nul["ref"], "cand": nul["cand"], "comparison": nul["comparison"], "args": args}
        r = subprocess.run([node, os.path.join(os.path.dirname(HERE), "gate_eval.js")],
                           input=json.dumps([spec]), capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout)[0]["band_pct"]
    ok(band({"stack": "megatron"}) == 0.3 and band({"stack": "hf_trl"}) == 0.25 and
       band({"stack": "megatron", "across_job_floor_pct": 0.2}) == 0.2,
       "gate_eval: per-stack floor (Megatron 0.30%, HF 0.25%); an explicit floor overrides it")
    ok(nul["verdict"]["band_pct"] == 0.3, "verify_gate without --floor uses the per-stack default")
finally:
    shutil.rmtree(root, ignore_errors=True)
# Audit 2026-10-25 G-3: L1 is recomputed with the ORCHESTRATOR's kind / floor. A cand whose step-1 loss is
# 0.1% off at step 1 only passes kernel_swap (0.25%) and fails overlay (0.05%); no --l1-kind, no l1 field.
root = tempfile.mkdtemp(prefix="vgtest_")
try:
    ref = leg(root, "ref", 1.0)
    cand = leg(root, "cand", 0.97, overlay="attn", loss_rel=0.001)
    ov = run(ref, cand, ("--floor", "0.25", "--l1-kind", "overlay"))
    ks = run(ref, cand, ("--floor", "0.25", "--l1-kind", "kernel_swap"))
    sb = run(ref, cand, ("--floor", "0.25", "--l1-kind", "kernel_swap", "--l1-l0-strictly-better"))
    ok(ov["l1"]["pass"] is False and ov["l1"]["recomputed"]["kind"] == "overlay",
       f"step-1 loss +0.1% fails the overlay floor (pass={ov['l1'].get('pass')})")
    ok(ks["l1"]["pass"] is True, f"... and passes kernel_swap (pass={ks['l1'].get('pass')})")
    ok(sb["l1"]["checks"]["step1_loss_rel_pct"]["limit"] == 2.0, "--l1-l0-strictly-better is the 2.0% floor")
    ok("l1" not in run(ref, cand), "no --l1-kind -> no l1 field (the orchestrator then refuses)")
    bad = run(ref, cand, ("--floor", "0.25", "--l1-kind", "kernel_swap", "--l1-band", os.path.join(root, "nope.json")))
    ok(bad["l1"]["pass"] is False and "error" in bad["l1"], "an unreadable band is a refusal, not a pass")
finally:
    shutil.rmtree(root, ignore_errors=True)
# Audit 2026-10-25 ID-1, end to end: the full capture flows through probe_summarize into the workflow's own
# comparable(). An unclassified env knob differing between the legs makes the pair incomparable.
root = tempfile.mkdtemp(prefix="vgtest_")
try:
    C = {"v": 1, "argv": {"normalization": "RMSNorm", "train-iters": "30"}, "env": {"NCCL_ALGO": "Ring"}}
    ref = leg(root, "ref", 1.0, coverage=C)
    same = run(ref, leg(root, "cand", 0.97, overlay="attn", coverage={**C, "argv": {**C["argv"], "train-iters": "40"}}))
    ok(same["verdict"]["klass"] == "win", f"same unclassified settings (train-iters ignored) -> win ({same['verdict'].get('klass')})")
    diff = run(ref, leg(root, "cand_nccl", 0.97, overlay="attn", coverage={**C, "env": {"NCCL_ALGO": "Tree"}}))
    ok(diff["verdict"]["klass"] == "incomparable" and "UNCLASSIFIED" in diff["verdict"].get("reason", ""),
       f"NCCL_ALGO Ring vs Tree (moves no digest) -> incomparable ({diff['verdict'].get('klass')})")
    mixed = run(leg(root, "ref_old", 1.0), leg(root, "cand_new", 0.97, overlay="attn", coverage=C))
    ok(mixed["verdict"]["klass"] != "win", f"a leg with the capture vs one without is not a win ({mixed['verdict'].get('klass')})")
finally:
    shutil.rmtree(root, ignore_errors=True)
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
