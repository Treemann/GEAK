#!/usr/bin/env node
// End-to-end handoff check on REAL artifacts produced by real runs.
//
// The unit tests prove each piece works in isolation. This one proves the pieces INTERLOCK: that
// bench_train.sh's bench_summary.json actually carries every field gateVerdict() reads, with the
// right names and types. A field renamed on one side and not the other passes every unit test and
// breaks the whole loop.
//
// The sharpest check here: feed the gate TWO REAL BASELINE RUNS. The honest answer is `no_win` --
// 0.118% of genuine run-to-run variation against a 0.3% band. A gate that manufactures a win out of
// two identical configurations would be worse than no gate at all.
//
// Run: node scripts/tests/test_pipeline_integration.js <runA_dir> <runB_dir>
'use strict';
const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..', '..', 'training_workflow.js');
const src = fs.readFileSync(WF, 'utf8');
let failures = 0;
const ok = (c, m) => { if (!c) { console.error('  FAIL:', m); failures++; } else console.log('  ok:', m); };

function extract(name) {
  const start = src.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`${name} not found`);
  let p = src.indexOf('(', start), pd = 0, i = -1;
  for (let j = p; j < src.length; j++) {
    if (src[j] === '(') pd++;
    else if (src[j] === ')') { pd--; if (pd === 0) { i = src.indexOf('{', j); break; } }
  }
  let depth = 0;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error(`unterminated ${name}`);
}
const api = new Function(
  // This file drives REAL 0.5B legs, so it must use the 0.5B-class floor (the shipped default, 0.5),
  // not the 7B override. That is not bookkeeping: with the 7B floor of 0.15 the two real legs below
  // -- one configuration, measured twice, true delta zero -- get a 0.228% band against an observed
  // -0.254%, and the gate reports a REGRESSION that does not exist. It is the 0.5B false-positive
  // rate (6 of 37 same-identity pairs on disk) reproducing on live data, and it is why the aligned
  // floor is per workload rather than one constant.
  'const FLOOR_PCT = {aligned:0.5,unaligned:0.5}; const ACROSS_JOB_FLOOR_PCT = 0.5, ' +
  'BAND_SAFETY = 2.0, SE_SAFETY = 4.0; let runAcrossJobPct = 0;\n' +
  ['noiseBandPct', 'legUsable', 'comparable', 'windowMatchedLegs', 'alignedDelta', 'gateVerdict', 'expectedGainPct', 'stopRule']
    .map(extract).join('\n') +
  '\nreturn {noiseBandPct, legUsable, comparable, gateVerdict, expectedGainPct, stopRule};')();

const [dirA, dirB] = process.argv.slice(2);
if (!dirA || !dirB) { console.error('usage: <runA_dir> <runB_dir>'); process.exit(2); }
// A summary written before the digest definition last changed is STALE, not incomparable: the probe
// sidecars are on disk, so re-summarising answers the question. That is exactly what the gate tells
// a caller to do, so the test does it rather than pretending old artifacts are current.
const { execFileSync } = require('child_process');
const SUMMARIZE = path.resolve(__dirname, '..', 'probe_summarize.py');
const CURRENT_SCHEMA = 2;

function load(d) {
  const stored = JSON.parse(fs.readFileSync(path.join(d, 'bench_summary.json'), 'utf8'));
  if (stored.digest_schema === CURRENT_SCHEMA) return stored;
  const fresh = JSON.parse(execFileSync('python3', [SUMMARIZE, path.join(d, 'probe')],
                                        { encoding: 'utf8', maxBuffer: 64 << 20 }));
  // Keep the dispatcher-added fields; take the digest/dispersion half from the fresh computation.
  return Object.assign({}, stored, fresh, { _resummarised: true });
}
const A = load(dirA), B = load(dirB);
if (A._resummarised || B._resummarised) {
  console.log('  note: re-summarised a stale artifact (digest definition changed since it was written)');
}

// ---- 1. the contract: every field the gate reads is actually produced by the dispatcher ----
for (const f of ['status', 'useful_tok_s_per_gpu', 'worst_iqr_spread_pct', 'peak_reserved_mb',
                 'config_digest', 'digest_schema', 'ranks_agree_on_argv', 'occupancy_verdict',
                 'hang_detected', 'dispersion_reliable']) {
  ok(A[f] !== undefined, `bench_summary.json carries "${f}" (gateVerdict reads it)`);
}
ok(typeof A.occupancy_verdict === 'object' && 'verdict' in A.occupancy_verdict,
  'occupancy_verdict is the object shape legUsable expects, not a bare string');

// ---- 2. real legs are usable ----
ok(api.legUsable(A).ok && api.legUsable(B).ok, 'both real baseline legs are usable');
ok(api.comparable(A, B).ok, `both legs share a config digest (${A.config_digest})`);

// ---- 3. THE check: two identical baselines must NOT produce a win ----
// This is also the regression guard for the window-placement defect. On THESE two runs -- same
// configuration, same data, nothing changed -- the raw ratio is +0.329%, which clears the 0.3% band
// and reads as a win. It is not: 93% of it is where each leg's warmup detection happened to land,
// and `useful_tokens` swings +-25% step to step so a two-step shift moves the median.
const CMP = path.resolve(__dirname, '..', 'compare_legs.py');
const cmp = JSON.parse(execFileSync('python3', [CMP, path.join(dirA, 'probe'), path.join(dirB, 'probe')],
                                    { encoding: 'utf8', maxBuffer: 64 << 20 }));

const raw = api.gateVerdict({ ref: A, cand: B, mode: 'fast' });
const v = api.gateVerdict({ ref: A, cand: B, mode: 'fast', comparison: cmp });
const warmupsDiffer = cmp.window.own_warmup.ref !== cmp.window.own_warmup.cand;

ok(/NOT window-aligned/.test(raw.delta_source),
  'a verdict taken without an aligned comparison SAYS so, instead of presenting the raw ratio as ' +
  'if it were the delta');
ok(v.klass === 'no_win' && v.accepted === false,
  `two real runs of one configuration, window-aligned -> "${v.klass}" (${v.delta_pct}% inside ` +
  `${v.band_pct}%): the gate does not manufacture a win`);
ok(cmp.dispersion_reliable === true,
  `both legs are long enough to carry a band (${cmp.min_steady_steps} steady steps)`);

// The artifact only exists when the two legs detected DIFFERENT warmup ends. Both cases are real and
// the test adapts rather than demanding one of them: measured across six 7B pairs, the artifact is
// 0.0000% when the warmups agree and up to 0.3867% when they do not.
// SIZE depends on the step shape (2026-10-20): on VARIABLE-shape legs (unpacked; useful_tokens swings
// +-25% step to step) a shifted window moves the median by 0.05-0.4%; on FIXED-shape packed Megatron
// legs every step carries the same tokens and the artifact is ~0.01% (hl_1016_205202_ref1 vs ref2:
// 0.0139%). The 0.05% bar was calibrated on unpacked legs only, so it is applied only above it.
if (warmupsDiffer) {
  ok(cmp.window_artifact_pct > 0,
    `warmups differ (${cmp.window.own_warmup.ref} vs ${cmp.window.own_warmup.cand}) -> window ` +
    `placement alone accounts for ${cmp.window_artifact_pct}% (non-zero)`);
  if (cmp.window_artifact_pct > 0.05) {
    ok(Math.abs(v.delta_pct) < Math.abs(raw.delta_pct),
      `and alignment shrinks the delta (${raw.delta_pct}% -> ${v.delta_pct}%), which is the whole point`);
  } else {
    console.log(`  note: artifact ${cmp.window_artifact_pct}% <= 0.05% (fixed-shape legs): the shrink ` +
                `check is below noise and is not applied`);
  }
} else {
  ok(cmp.window_artifact_pct === 0,
    `warmups agree (both ${cmp.window.own_warmup.ref}) -> alignment is an identity, artifact 0%`);
  // Compare against compare_legs' WINDOW-ALIGNED delta, not the gate's quoted one: since DECISIONS
  // 4.13 the gate quotes the PAIRED per-step ratio when the legs saw the same batches, which is a
  // different (better) estimator and need not equal a ratio of medians even on matched windows.
  ok(Math.abs(raw.delta_pct - cmp.delta_pct) < 1e-9,
    `and the aligned and raw deltas coincide (${raw.delta_pct}% vs ${cmp.delta_pct}%), as they must ` +
    `when the windows already match; the gate quotes the paired ${v.delta_pct}%`);
}

// ---- 4. the band came from the legs themselves, not from a constant ----
// Read the band the GATE uses, not noiseBandPct(A, B): since DECISIONS 4.11/4.13 the gate prices a
// window-aligned delta with a band estimated on the same window (and, with a paired comparison, the
// paired ratio's IQR), so the per-leg helper is no longer what decides. Testing the helper's number
// against the gate's decision is what made this check fail once the band moved (2026-10-14 fix).
const aligned = (pct) => ({ delta_pct: pct, window_artifact_pct: 0, dispersion_reliable: true });
const band = api.gateVerdict({ ref: A, cand: A, mode: 'fast', comparison: aligned(0) }).band_pct;
ok(isFinite(band) && band >= 0.5 && band < 5.0,
  `band = ${band}% for this real pair: at least the 0.5% workload floor and derived from the legs ` +
  `(quiet 0.5B legs give ~0.5-0.9%, the noisier packed 7B legs ~2%), not a fixed 2-3% constant`);

// ---- 5. a synthetic candidate JUST above the band is accepted; just below is not ----
const bump = (leg, pct) => Object.assign({}, leg, {
  useful_tok_s_per_gpu: leg.useful_tok_s_per_gpu * (1 + pct / 100) });
// Synthetic candidates must carry an aligned comparison, or they are judged against the UNALIGNED
// floor (0.5%) — which is correct behaviour, and worth pinning in both directions.
ok(api.gateVerdict({ ref: A, cand: bump(A, band + 0.05), mode: 'fast',
                     comparison: aligned(band + 0.05) }).accepted,
  `a window-aligned +${(band + 0.05).toFixed(2)}% candidate clears the ${band}% band`);
ok(!api.gateVerdict({ ref: A, cand: bump(A, band - 0.05), mode: 'fast',
                      comparison: aligned(band - 0.05) }).accepted,
  `a window-aligned +${(band - 0.05).toFixed(2)}% candidate does not`);
ok(!api.gateVerdict({ ref: A, cand: bump(A, 0.4), mode: 'fast' }).accepted,
  'the SAME +0.40% without alignment is refused: the unaligned floor is 0.5%, because window ' +
  'placement alone was measured at up to 0.4456%');

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall handoffs check out');
process.exit(failures ? 1 : 0);
