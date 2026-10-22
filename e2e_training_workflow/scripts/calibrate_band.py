#!/usr/bin/env python3
"""Re-run the band's NULL CONTROL over every leg on disk. No GPU, no new measurements.

WHY THIS IS A SCRIPT AND NOT A NUMBER IN A DOCUMENT. The band formula's licence is one measurement:
"47 same-identity baseline pairs, true delta 0, 0 false wins (old formula: 6)". That number is in
`knowledge/sft_measurement.md` and nowhere else, so every proposal to change anything the band
touches has had to be declined on the grounds that re-validating it would be expensive. It is not
expensive. The legs are already on disk and the pairs can be rebuilt from their identity digests.

A pair is a NULL PAIR when both legs carry the same `config_digest` AND the same `impl_digest`: same
workload, same implementation, so the true throughput delta is zero by construction and ANY
"win" the band reports is false. That is the whole control.

WHAT IT IS NOT. It is only the null half. A band that is infinitely wide scores 0 false wins and
gates nothing, so a null control alone can always be satisfied by making the band bigger. The
positive half -- that a real win still clears the band -- is not reconstructible from same-identity
legs and lives in the measured accepted candidates (+1.147%, +7.4348%, +12.36%). Read the two
together or neither means anything.

  calibrate_band.py                      # judge every null pair with THE GATE ITSELF (default)
  calibrate_band.py --floor 0.25         # the run's across_job_floor_pct (default 0.25)
  calibrate_band.py --legacy             # the old re-implemented per-leg band (kept for comparison)
  calibrate_band.py --quantile interp    # (legacy only) linear-interpolated quartiles
  calibrate_band.py --max-pairs 200      # bound the work
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


def _gate_pair(a_probe, b_probe, floor, node):
    """compare_legs + the workflow's own gateVerdict on one pair. Returns the verdict dict or None."""
    import subprocess
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
    spec = {"ref": legs[0], "cand": legs[1], "comparison": cmp, "mode": "fast",
            "args": {"across_job_floor_pct": floor}}
    r = subprocess.run([node, os.path.join(HERE, "gate_eval.js")], input=json.dumps(spec),
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise SystemExit(f"gate_eval.js failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


def main_gate(max_pairs, floor):
    node = _node()
    runs = os.path.join(os.environ.get("GEAK_WORK_DIR", "/home/treemann/geak_train"), "runs")
    groups, rejected = collections.defaultdict(list), collections.Counter()
    for pd in sorted(glob.glob(os.path.join(runs, "*", "probe"))):
        ok, why = _usable(pd)
        if not ok:
            rejected[why] += 1
            continue
        leg = _leg(pd)
        if not leg or (leg.get("digest_schema") or 0) < 3 or not leg.get("impl_digest"):
            rejected["unreadable, short, or identity schema < v3"] += 1
            continue
        groups[_identity(leg)].append(pd)
    pairs = [(m[i], m[j]) for m in groups.values() for i in range(len(m)) for j in range(i + 1, len(m))]
    print(f"gate-based null control: {len(pairs)} same-identity pairs in {len(groups)} groups "
          f"(evaluating up to {max_pairs}); floor {floor}%")
    counts, false_ = collections.Counter(), []
    for a, b in pairs[:max_pairs]:
        v = _gate_pair(a, b, floor, node)
        if v is None:
            counts["no common window"] += 1
            continue
        counts[v.get("klass")] += 1
        if v.get("klass") in ("win", "regression"):
            false_.append((a, b, v))
    print("  verdicts: " + ", ".join(f"{k}={n}" for k, n in counts.most_common()))
    judged = sum(n for k, n in counts.items() if k != "no common window")
    print(f"\nFALSE WINS/REGRESSIONS (gate): {len(false_)} / {judged}")
    for a, b, v in false_[:10]:
        print(f"    {a.split('/')[-2]} vs {b.split('/')[-2]}: {v.get('klass')} {v.get('delta_pct')}% "
              f"vs band {v.get('band_pct')}% ({v.get('estimator')})")
    return 0


def main():
    if "--legacy" not in sys.argv:
        for tok in sys.argv[1:]:
            if tok.startswith("--") and tok not in ("--max-pairs", "--floor"):
                sys.exit(f"unknown flag {tok!r}; known: --floor, --max-pairs, --legacy")
        mp = int(sys.argv[sys.argv.index("--max-pairs") + 1]) if "--max-pairs" in sys.argv else 400
        fl = float(sys.argv[sys.argv.index("--floor") + 1]) if "--floor" in sys.argv else 0.25
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
