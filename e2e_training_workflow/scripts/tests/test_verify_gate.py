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
def leg(root, name, speed, overlay=None, nranks=8, fallback=0, drop_rank=None):
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "probe"))
    impl = dict(IMPL, **({"GEAK_OVERLAY": "m.json", "GEAK_OVERLAY_SHA": "k1"} if overlay else {}))
    cfg = {"seq-length": "4096", "snapshot_keys": "v5", **impl}
    for rk in range(nranks):
        with open(os.path.join(d, "probe", f"probe_rank{rk}.jsonl"), "w") as f:
            f.write(json.dumps({"record": "header", "rank": rk, "world_size": nranks, "argv_hash": "a", "config": cfg}) + "\n")
            t = 0.0
            for i in range(30):
                dur = (40000 if i == 0 else 2300 + 7 * ((i * 37) % 5)) * speed
                t += dur / 1000 + 0.001
                f.write(json.dumps({"record": "step", "step": i + 1, "dur_ms": dur, "useful_tokens": 8000,
                                    "total_tokens": 8192, "peak_reserved_mb": 80000, "t_wall": t,
                                    "loss": 3.0 - 0.01 * i, "grad_norm": 5.0}) + "\n")
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
print("all checks passed" if not fails else f"{fails} check(s) FAILED")
sys.exit(1 if fails else 0)
