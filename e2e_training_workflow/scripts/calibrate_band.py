#!/usr/bin/env python3
"""Re-run the band's NULL CONTROL over every leg on disk. No GPU, no new measurements.

WHY THIS IS A SCRIPT AND NOT A NUMBER IN A DOCUMENT. The band formula's licence is one measurement:
"47 same-identity baseline pairs, true delta 0, 0 false wins (old formula: 6)". That number is in
`knowledge/sft_measurement.md` and nowhere else, so every proposal to change anything the band
touches has had to be declined on the grounds that re-validating it would be expensive. It is not
expensive. The legs are already on disk and the pairs can be rebuilt from their identity digests.

A pair is a NULL PAIR when both legs carry the same `config_digest` AND the same `impl_digest`: same
workload, same implementation, so the true throughput delta is zero by construction and ANY
"win" the band reports is false. Since 2026-10-23 (DECISIONS 4.54) the two legs must ALSO agree on
the framework's realized-argument dump in train.log: digests cover only what the snapshot captures,
and same-digest pairs that differ in an uncaptured argument are reported and excluded, not graded.

WHAT IT IS NOT. It is only the null half. A band that is infinitely wide scores 0 false wins and
gates nothing, so a null control alone can always be satisfied by making the band bigger. The
positive half -- that a real win still clears the band -- is approximated by INJECTION: each null
pair is re-judged with the candidate made exactly +0.5% / +1% faster (see _inject), which keeps the
pair's real noise and gives the gate's power at that effect. Read the two together or neither means
anything.

  calibrate_band.py                      # judge every null pair with THE GATE ITSELF (default)
  calibrate_band.py --floor 0.25         # force one across_job_floor_pct (default: the gate's per-stack floor)
  calibrate_band.py --legacy             # the old re-implemented per-leg band (kept for comparison)
  calibrate_band.py --quantile interp    # (legacy only) linear-interpolated quartiles
  calibrate_band.py --max-pairs 200      # bound the work (default: every pair; sampled, seed 0)
  runs dir: $GEAK_WORK_DIR/runs (default /home/treemann/geak_train/runs); node: $NODE or `node`.

THE GATE ITSELF (2026-10-22, audit TP-6). This script used to re-implement the band and drifted from
the gate: a different floor, per-leg medians where the gate quotes the PAIRED per-step ratio, and no
identity/implementation rules -- it reported 6/219 false wins the gate would not have produced. The
default now runs compare_legs.compare() on each null pair and hands it to scripts/gate_eval.js, which
executes gateVerdict() sliced from training_workflow.js. A false win is any pair the gate calls
`win` or `regression`. It can no longer disagree with the gate it certifies.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_summarize import detect_warmup_end, load_rank  # noqa: E402

# Mirrors training_workflow.js. Kept here as literals rather than parsed out of the JS: if the two
# drift, this script's answer stops describing the gate, and a silent drift is worse than a
# duplicate. The assertion that they match belongs in a test, not in a calibration run.
SE_SAFETY = 4.0
BAND_SAFETY = 2.0
FLOOR_PCT = 0.15


def _quartiles(vals, mode):
    s = sorted(vals)
    n = len(s)
    if n < 4:
        return None
    if mode == "nearest":
        # As shipped in probe_summarize.py. Jumps discontinuously when the quartile sits on a cliff:
        # measured 0.096% at n=25 and 2.607% at n=24 on the same leg, because nineteen values sat
        # inside 11642-11660 ms and the twentieth was 11952.
        return s[n // 4], s[(3 * n) // 4]
    # Linear interpolation between order statistics, the usual definition.
    def q(p):
        h = p * (n - 1)
        lo = int(math.floor(h))
        hi = min(lo + 1, n - 1)
        return s[lo] + (h - lo) * (s[hi] - s[lo])
    return q(0.25), q(0.75)


def _iqr_pct(vals, mode):
    qs = _quartiles(vals, mode)
    if not qs:
        return None
    s = sorted(vals)
    med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return 100.0 * (qs[1] - qs[0]) / med if med else None


def _usable(probe_dir):
    """Apply the gate's OWN usability rules before a leg may be a null pair.

    Without this the sweep silently includes legs the gate would never compare: capture runs
    (`captured_run`, where bench_train.sh itself prints "this run's throughput is NOT a valid
    measurement"), profiled runs, contaminated legs, and legs too short to estimate dispersion.
    Measured: including them gives 73 false wins out of 386 pairs, which says nothing about the band
    and everything about the inputs -- `attn_capture` against a clean leg is a 5.3% "delta" between
    a capture run and a measurement, exactly as designed. A calibration that silently grades the
    band on data the gate rejects is worse than no calibration, because the number looks like an
    indictment.
    """
    bs = os.path.join(os.path.dirname(probe_dir), "bench_summary.json")
    if not os.path.exists(bs):
        return False, "no bench_summary"
    try:
        j = json.load(open(bs))
    except Exception:
        return False, "unreadable bench_summary"
    if j.get("status") != "ok":
        return False, f"status={j.get('status')}"
    if j.get("captured_run"):
        return False, "capture run — not a valid throughput measurement"
    if j.get("profiled_run"):
        return False, "profiled run — the profiler perturbs the step"
    if j.get("hang_detected"):
        return False, "hang"
    if j.get("dispersion_reliable") is False:
        return False, "dispersion unreliable"
    occ = (j.get("occupancy_verdict") or {})
    if isinstance(occ, dict) and occ.get("verdict") not in (None, "clean"):
        return False, f"occupancy={occ.get('verdict')}"
    return True, None


def _leg(probe_dir):
    """Per-rank durations and useful tokens, plus the leg's identity. None when unusable."""
    files = sorted(glob.glob(os.path.join(probe_dir, "probe_rank*.jsonl")))
    if not files:
        return None
    ranks, header = [], None
    for f in files:
        try:
            h, steps = load_rank(f)
        except BaseException:
            # load_rank REFUSES a sidecar with two headers (two runs appended to one file) by
            # exiting. Catching BaseException keeps one corrupt directory from ending the sweep,
            # while still excluding it -- which is the behaviour that refusal exists to produce.
            return None
        if header is None:
            header = h
        ranks.append([(s.get("dur_ms"), s.get("useful_tokens")) for s in steps])
    if not ranks or len(ranks[0]) < 25:
        return None
    cfg = (header or {}).get("config", {})
    bs = json.load(open(os.path.join(os.path.dirname(probe_dir), "bench_summary.json")))
    return {"dir": probe_dir, "ranks": ranks, "cfg": cfg, "n": len(ranks[0]),
            "config_digest": bs.get("config_digest"), "impl_digest": bs.get("impl_digest"),
            "digest_schema": bs.get("digest_schema"), "argv_hash": (header or {}).get("argv_hash")}


def _identity(leg):
    """The AUTHORITATIVE identity pair, read from bench_summary.json rather than recomputed.

    An earlier version of this script re-derived the identity/implementation split from the config
    snapshot. It was wrong in a way that is invisible until you read the names in the output:
    probe_summarize injects `world_size` into the config before digesting, and this did not, so legs
    at different world sizes landed in the same "identity" group and produced 5.5% "false wins"
    between legs that were never the same measurement. Reading the digest the gate itself uses
    removes this script as a source of disagreement -- if the two ever differ, the digest is right
    and this file is wrong.
    """
    return (leg.get("config_digest"), leg.get("impl_digest"))


def _pair(a, b, mode):
    """Common-window delta and band for one pair, the way the gate computes them."""
    wa = max(detect_warmup_end([d for d, _ in a["ranks"][0]]),
             detect_warmup_end([d for d, _ in b["ranks"][0]]))
    n = min(a["n"], b["n"]) - wa
    if n < 10:
        return None

    def leg_stats(leg):
        # Worst IQR across ranks, and useful tok/s/GPU, both on the COMMON window -- the window the
        # delta is measured on. Estimating the band anywhere else is the defect this re-validates.
        iqrs, rates = [], []
        for r in leg["ranks"]:
            w = r[wa:wa + n]
            durs = [d for d, _ in w if d is not None]
            toks = [t for _, t in w if t is not None]
            if len(durs) < 4 or not toks:
                return None
            i = _iqr_pct(durs, mode)
            if i is None:
                return None
            iqrs.append(i)
            med = sorted(durs)[len(durs) // 2]
            rates.append(sum(toks) / len(toks) / (med / 1000.0))
        return max(iqrs), min(rates)

    sa, sb = leg_stats(a), leg_stats(b)
    if not sa or not sb:
        return None
    se = lambda i: 0.929 * i / math.sqrt(n)                       # noqa: E731
    band = max(FLOOR_PCT, SE_SAFETY * math.sqrt(se(sa[0]) ** 2 + se(sb[0]) ** 2))
    return {"delta_pct": 100.0 * (sb[1] - sa[1]) / sa[1], "band_pct": band,
            "n": n, "iqr_ref": sa[0], "iqr_cand": sb[0]}


def _node():
    import shutil
    n = os.environ.get("NODE") or shutil.which("node")
    if n:
        return n
    c = sorted(glob.glob(os.path.expanduser("~/.cursor-server/bin/*/node")) +
               glob.glob(os.path.expanduser("~/.cursor-server/bin/*/*/node")))
    if not c:
        sys.exit("no node found: set NODE=/path/to/node (the gate is JavaScript)")
    return c[0]


_DUMP = re.compile(r"^\s{2}([a-z0-9_]+) \.{3,}\s(.*)$")
# Length/naming args that legitimately differ between legs of ONE configuration.
_REALIZED_IGNORE = ("train_iters", "max_steps", "output_dir", "logging_dir", "run_name", "save", "load",
                    "tensorboard_dir", "wandb_exp_name")


def _realized(leg_dir):
    """The framework's own realized-argument dump from train.log ('  name ....... value'), minus the
    length/naming args. None when the log has no such dump. A null pair must agree on THIS table as
    well as on the digests: digests only cover what the snapshot chose to capture, and a pair that
    differs in an uncaptured argument (the rope-fusion gap, DECISIONS 4.54) is not a null."""
    t = {}
    try:
        with open(os.path.join(leg_dir, "train.log"), errors="replace") as f:
            for i, line in enumerate(f):
                if i > 6000:
                    break
                m = _DUMP.match(line.rstrip("\n"))
                if m and m.group(1) not in _REALIZED_IGNORE:
                    t[m.group(1)] = m.group(2).strip()
    except OSError:
        return None
    return t if len(t) >= 10 else None


def _spec(a_probe, b_probe, floor, stack):
    """compare_legs + fresh summaries -> the gate_eval spec for one pair, or None (no common window)."""
    from compare_legs import compare
    from probe_summarize import summarize
    cmp = compare(a_probe, b_probe)
    if not isinstance(cmp, dict) or cmp.get("error"):
        return None
    legs = []
    for pd in (a_probe, b_probe):
        s = summarize(pd)
        try:
            bs = json.load(open(os.path.join(os.path.dirname(pd), "bench_summary.json")))
        except Exception:
            bs = {}
        # dispatcher-added fields (occupancy, hang) come from bench_summary; the rest from a fresh summary
        for k in ("occupancy_verdict", "hang_detected", "status", "contamination_exhausted", "attempts"):
            if k in bs:
                s[k] = bs[k]
        s.setdefault("status", "ok")
        legs.append(s)
    args = {"stack": stack}
    if floor is not None:
        args["across_job_floor_pct"] = floor
    return {"ref": legs[0], "cand": legs[1], "comparison": cmp, "mode": "fast", "args": args}


def _inject(spec, pct):
    """The same pair with the candidate made exactly pct% faster on every step. Scaling every cand
    step by 1/(1+pct) scales every per-step ratio, the per-leg median and the rate by the same
    factor and leaves every relative dispersion (and the stall trim) unchanged -- so this IS the
    pair the gate would see, without re-reading the probes. Power = how often the gate calls it a win."""
    f = 1.0 + pct / 100.0
    s = json.loads(json.dumps(spec))
    c = s["comparison"]
    for key in ("delta_pct", "unaligned_delta_pct"):
        if isinstance(c.get(key), (int, float)):
            c[key] = 100.0 * (f * (1.0 + c[key] / 100.0) - 1.0)
    if isinstance(c.get("paired"), dict):
        c["paired"]["delta_pct"] = 100.0 * (f * (1.0 + c["paired"]["delta_pct"] / 100.0) - 1.0)
    s["cand"]["useful_tok_s_per_gpu"] = s["cand"]["useful_tok_s_per_gpu"] * f
    return s


def _gate_batch(specs, node):
    import subprocess
    r = subprocess.run([node, os.path.join(HERE, "gate_eval.js")], input=json.dumps(specs),
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"gate_eval.js failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


POWER_AT_PCT = (0.5, 1.0)


def _t0(probe_dir):
    """Wall-clock start of a leg (its first step record), for the session-gap label."""
    try:
        with open(os.path.join(probe_dir, "probe_rank0.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("record") == "step":
                    return r.get("t_wall")
    except (OSError, ValueError):
        return None
    return None


def main_gate(max_pairs, floor):
    """Every TRUE-NULL pair on disk through the gate, per stratum (stack x bare/overlay).

    A leg is counted once (realpath: bundles symlink their legs). A null pair has equal
    config_digest AND impl_digest (schema >= 3) AND an equal realized-argument table where both
    legs have one. Reports, per stratum: false wins/regressions (must be ~0), the inconclusive rate
    (how often the gate cannot resolve anything on a null), and POWER -- the same pairs with a
    +0.5% / +1% effect injected -- because a null control alone is satisfied by an infinite band."""
    import random
    node = _node()
    runs = os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"), "runs")
    groups, rejected, seen, meta = collections.defaultdict(list), collections.Counter(), set(), {}
    for pd in sorted(glob.glob(os.path.join(runs, "*", "probe")) + glob.glob(os.path.join(runs, "*", "*", "probe"))):
        rp = os.path.realpath(pd)
        if rp in seen:
            rejected["duplicate (symlinked) leg"] += 1
            continue
        seen.add(rp)
        ok, why = _usable(rp)
        if not ok:
            rejected[why] += 1
            continue
        leg = _leg(rp)
        if not leg or (leg.get("digest_schema") or 0) < 3 or not leg.get("impl_digest"):
            rejected["unreadable, short, or identity schema < v3"] += 1
            continue
        bs = json.load(open(os.path.join(os.path.dirname(rp), "bench_summary.json")))
        impl = bs.get("implementation") or {}
        stack = "hf_trl" if bs.get("step_shape") == "variable" or "geak-stack" in (bs.get("config") or {}) else "megatron"
        meta[rp] = (stack, "overlay" if impl.get("GEAK_OVERLAY_SHA") else "bare", _realized(os.path.dirname(rp)))
        groups[_identity(leg)].append(rp)
    pairs, split = [], 0
    for m in groups.values():
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                ta, tb = meta[m[i]][2], meta[m[j]][2]
                if ta is not None and tb is not None and ta != tb:
                    split += 1      # equal digests, different realized args: an identity gap, not a null
                    continue
                pairs.append((m[i], m[j]))
    random.Random(0).shuffle(pairs)   # a --max-pairs cap must not favour the alphabetically-first runs
    print(f"gate-based null control: {len(seen)} distinct legs, {len(pairs)} true-null pairs in "
          f"{len(groups)} identity groups; {split} same-digest pairs EXCLUDED for differing realized "
          f"args; evaluating {min(len(pairs), max_pairs)}; floor "
          f"{'per-stack default' if floor is None else str(floor) + '%'}")
    if rejected:
        print("  legs not used: " + ", ".join(f"{k}={n}" for k, n in rejected.most_common()))
    strata = collections.defaultdict(lambda: {"n": 0, "verdicts": collections.Counter(), "false": [],
                                              "power": collections.Counter(), "bands": []})
    work = []
    for a, b in pairs[:max_pairs]:
        sp = _spec(a, b, floor, meta[a][0])
        if sp is None:
            rejected["pair without a common window"] += 1
            continue
        work.append(((meta[a][0], meta[a][1]), a, b, sp))
    batch = []
    for _, _, _, sp in work:
        batch += [sp] + [_inject(sp, p) for p in POWER_AT_PCT]
    verdicts = _gate_batch(batch, node) if batch else []
    k = 1 + len(POWER_AT_PCT)
    for idx, (st, a, b, _) in enumerate(work):
        v, inj = verdicts[k * idx], verdicts[k * idx + 1: k * idx + k]
        S = strata[st]
        S["n"] += 1
        S["verdicts"][v.get("klass")] += 1
        if isinstance(v.get("band_pct"), (int, float)):
            S["bands"].append(v["band_pct"])
        if v.get("klass") in ("win", "regression"):
            S["false"].append((a, b, v))
        for p, w in zip(POWER_AT_PCT, inj):
            S["power"][p] += w.get("klass") == "win"
    total_false = 0
    for st in sorted(strata):
        S = strata[st]
        n, bands = S["n"], sorted(S["bands"])
        total_false += len(S["false"])
        p90 = bands[int(0.9 * (len(bands) - 1))] if bands else float("nan")
        print(f"\n[{st[0]} / {st[1]}] {n} pairs: false wins/regressions {len(S['false'])}; "
              f"inconclusive {100.0 * S['verdicts']['inconclusive_noise'] / max(n, 1):.1f}%; "
              f"band p90 {p90:.3f}%; " +
              "; ".join(f"power at +{p}% {100.0 * S['power'][p] / max(n, 1):.1f}%" for p in POWER_AT_PCT))
        print("  verdicts: " + ", ".join(f"{kk}={c}" for kk, c in S["verdicts"].most_common()))
        for a, b, v in S["false"][:10]:
            ta, tb = _t0(a), _t0(b)
            gap = f"{abs(ta - tb) / 3600:.1f} h apart" if ta and tb else "start time unknown"
            print(f"    FALSE {v.get('klass')}: {a.split('/')[-2]} vs {b.split('/')[-2]}: "
                  f"{v.get('delta_pct')}% vs band {v.get('band_pct')}% ({v.get('estimator')}; {gap})")
    # The gate only ever compares a SAME-SESSION pair (roles/integrator.md step 3). Cross-session
    # nulls are graded too, as the conservative case, but a false call between legs hours apart is
    # day-scale drift the gate's design keeps out, not a band failure -- the gap printed says which.
    print(f"\nFALSE WINS/REGRESSIONS (gate), all strata: {total_false} / {sum(S['n'] for S in strata.values())}")
    return 0


def main():
    if "--legacy" not in sys.argv:
        for tok in sys.argv[1:]:
            if tok.startswith("--") and tok not in ("--max-pairs", "--floor"):
                sys.exit(f"unknown flag {tok!r}; known: --floor, --max-pairs, --legacy")
        mp = int(sys.argv[sys.argv.index("--max-pairs") + 1]) if "--max-pairs" in sys.argv else 10 ** 9
        fl = float(sys.argv[sys.argv.index("--floor") + 1]) if "--floor" in sys.argv else None
        return main_gate(mp, fl)
    sys.argv = [a for a in sys.argv if a != "--legacy"]
    mode = "nearest"
    if "--quantile" in sys.argv:
        mode = sys.argv[sys.argv.index("--quantile") + 1]
    if mode not in ("nearest", "interp"):
        sys.exit(f"unknown --quantile {mode!r}; known: nearest, interp")
    max_pairs = 400
    if "--max-pairs" in sys.argv:
        max_pairs = int(sys.argv[sys.argv.index("--max-pairs") + 1])
    for tok in sys.argv[1:]:
        if tok.startswith("--") and tok not in ("--quantile", "--max-pairs"):
            sys.exit(f"unknown flag {tok!r}; known: --quantile, --max-pairs")

    legs, rejected = [], collections.Counter()
    for pd in sorted(glob.glob(os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"),
                                            "runs", "*", "probe"))):
        ok, why = _usable(pd)
        if not ok:
            rejected[why] += 1
            continue
        leg = _leg(pd)
        if leg:
            legs.append(leg)
        else:
            rejected["unreadable or <25 steps"] += 1

    # A NULL PAIR requires that the identity schema could have SEPARATED the two legs. v2 snapshots
    # carry 9 config fields and do not cover attention_dropout at all, so two v2 legs sharing a
    # digest prove nothing about the band.
    #
    # This is not hypothetical. Including them put 32 "false wins" on the board, almost all driven by
    # td_ref_legA_p0 running 7.3% faster than td_baselineA with a byte-identical config snapshot.
    # Diffing Megatron's own 647-line realized dump gives exactly ONE differing line --
    # attention_dropout 0.1 vs 0.0 -- which is precisely the gap v3 was built to close. Those legs
    # were never the same measurement; v2 simply could not say so. Grading the band on them would
    # have indicted it for a schema limitation that had been fixed weeks earlier.
    MIN_SCHEMA = 3
    groups = collections.defaultdict(list)
    schema_skipped = 0
    for leg in legs:
        if not leg.get("config_digest") or not leg.get("impl_digest"):
            continue
        if (leg.get("digest_schema") or 0) < MIN_SCHEMA:
            schema_skipped += 1
            continue
        groups[_identity(leg)].append(leg)
    print(f"  excluded {schema_skipped} more legs at identity schema < v{MIN_SCHEMA} "
          "(their digests cannot support a same-identity claim)")

    pairs, skipped = [], 0
    for key, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    pairs = pairs[:max_pairs]

    false_wins, results = [], []
    for a, b in pairs:
        r = _pair(a, b, mode)
        if not r:
            skipped += 1
            continue
        results.append(r)
        # Both directions: the pair is symmetric and "a win" is direction-free under a null.
        if abs(r["delta_pct"]) > r["band_pct"]:
            false_wins.append((os.path.basename(os.path.dirname(a["dir"])),
                               os.path.basename(os.path.dirname(b["dir"])), r))

    print(f"quartiles: {mode}")
    print(f"usable legs: {len(legs)}   identity groups: {len(groups)}")
    print("  excluded: " + ", ".join(f"{n}x {w}" for w, n in rejected.most_common()))
    print(f"NULL PAIRS evaluated: {len(results)}  (skipped {skipped} for too short a common window)")
    if not results:
        sys.exit("no null pairs -- nothing to calibrate against")
    bands = sorted(r["band_pct"] for r in results)
    deltas = sorted(abs(r["delta_pct"]) for r in results)
    med = lambda v: v[len(v) // 2]                                # noqa: E731
    print(f"  band   median {med(bands):.3f}%   max {bands[-1]:.3f}%")
    print(f"  |delta| median {med(deltas):.3f}%   max {deltas[-1]:.3f}%")
    print(f"\nFALSE WINS: {len(false_wins)} / {len(results)}")
    for a, b, r in false_wins[:10]:
        print(f"    {a} vs {b}: |{r['delta_pct']:+.3f}%| > band {r['band_pct']:.3f}% (n={r['n']})")
    if len(false_wins) > 10:
        print(f"    ... and {len(false_wins) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
