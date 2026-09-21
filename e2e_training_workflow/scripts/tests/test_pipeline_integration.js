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
  'const ACROSS_JOB_FLOOR_PCT = 0.3, BAND_SAFETY = 2.0;\n' +
  ['noiseBandPct', 'legUsable', 'comparable', 'alignedDelta', 'gateVerdict', 'expectedGainPct', 'stopRule']
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
ok(raw.klass === 'win',
  `WITHOUT alignment the raw ratio (${raw.delta_pct}%) reads as a "${raw.klass}" against the ` +
  `${raw.band_pct}% band — this is the defect, pinned so it cannot come back silently`);
ok(/NOT window-aligned/.test(raw.delta_source),
  'and the verdict says so, rather than presenting the raw ratio as if it were the delta');

const v = api.gateVerdict({ ref: A, cand: B, mode: 'fast', comparison: cmp });
ok(v.klass === 'no_win' && v.accepted === false,
  `WITH a window-aligned comparison the same pair is "${v.klass}" (${v.delta_pct}% inside ` +
  `${v.band_pct}%): the gate does not manufacture a win out of window placement`);
ok(cmp.window_artifact_pct > 0.25,
  `window placement alone accounted for ${cmp.window_artifact_pct}% on this pair`);
ok(cmp.dispersion_reliable === true,
  `both legs are long enough to carry a band (${cmp.min_steady_steps} steady steps)`);

// ---- 4. the band came from the legs themselves, not from a constant ----
const band = api.noiseBandPct(A, B);
ok(band >= 0.3 && band < 1.0,
  `band derived from the legs' own IQR spread = ${band}% (floor 0.3%, not a fixed 2-3%)`);

// ---- 5. a synthetic candidate JUST above the band is accepted; just below is not ----
const bump = (leg, pct) => Object.assign({}, leg, {
  useful_tok_s_per_gpu: leg.useful_tok_s_per_gpu * (1 + pct / 100) });
ok(api.gateVerdict({ ref: A, cand: bump(A, band + 0.05), mode: 'fast' }).accepted,
  `a +${(band + 0.05).toFixed(2)}% candidate clears the band`);
ok(!api.gateVerdict({ ref: A, cand: bump(A, band - 0.05), mode: 'fast' }).accepted,
  `a +${(band - 0.05).toFixed(2)}% candidate does not`);

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall handoffs check out');
process.exit(failures ? 1 : 0);
