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
  ['noiseBandPct', 'legUsable', 'comparable', 'gateVerdict', 'expectedGainPct', 'stopRule']
    .map(extract).join('\n') +
  '\nreturn {noiseBandPct, legUsable, comparable, gateVerdict, expectedGainPct, stopRule};')();

const [dirA, dirB] = process.argv.slice(2);
if (!dirA || !dirB) { console.error('usage: <runA_dir> <runB_dir>'); process.exit(2); }
const load = (d) => JSON.parse(fs.readFileSync(path.join(d, 'bench_summary.json'), 'utf8'));
const A = load(dirA), B = load(dirB);

// ---- 1. the contract: every field the gate reads is actually produced by the dispatcher ----
for (const f of ['status', 'useful_tok_s_per_gpu', 'worst_iqr_spread_pct', 'peak_reserved_mb',
                 'config_digest', 'ranks_agree_on_argv', 'occupancy_verdict', 'hang_detected']) {
  ok(A[f] !== undefined, `bench_summary.json carries "${f}" (gateVerdict reads it)`);
}
ok(typeof A.occupancy_verdict === 'object' && 'verdict' in A.occupancy_verdict,
  'occupancy_verdict is the object shape legUsable expects, not a bare string');

// ---- 2. real legs are usable ----
ok(api.legUsable(A).ok && api.legUsable(B).ok, 'both real baseline legs are usable');
ok(api.comparable(A, B).ok, `both legs share a config digest (${A.config_digest})`);

// ---- 3. THE check: two identical baselines must NOT produce a win ----
const v = api.gateVerdict({ ref: A, cand: B, mode: 'fast' });
ok(v.klass === 'no_win' && v.accepted === false,
  `two real baseline runs -> "${v.klass}" (${v.delta_pct}% inside the ${v.band_pct}% band): ` +
  'the gate does not manufacture a win out of run-to-run variation');
ok(Math.abs(v.delta_pct) < 0.3,
  `measured run-to-run delta is ${v.delta_pct}%, consistent with the calibrated across-job band`);

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
