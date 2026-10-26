#!/usr/bin/env node
'use strict';
// gate_eval.js -- run the workflow's OWN gate (gateVerdict and its helpers, sliced out of
// training_workflow.js) on a JSON spec from stdin, so offline tools judge exactly as the gate does.
// calibrate_band.py used to re-implement the band and drifted (wrong floor, per-leg estimator
// instead of paired, no identity rules: 6/219 false wins; audit 2026-10-22 TP-6).
//
//   echo '{"ref":{...bench_summary...},"cand":{...},"comparison":{...compare_legs...},"mode":"fast",
//          "args":{"stack":"megatron"}}' | node gate_eval.js      (or a JSON array of specs)
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'training_workflow.js'), 'utf8');

function extract(name) {
  const start = src.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`function ${name} not found`);
  let pd = 0, i = -1;
  for (let j = src.indexOf('(', start); j < src.length; j++) {
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
// The constant block the gate reads, verbatim from the workflow (it parses them from `A`).
function constBlock() {
  const a = src.indexOf('const FLOOR_PCT = ');
  const b = src.indexOf('let runAcrossJobPct = 0;', a);
  if (a < 0 || b < 0) throw new Error('gate constant block not found');
  return src.slice(a, b + 'let runAcrossJobPct = 0;'.length);
}
const NAMES = ['windowMatchedLegs', 'noiseBandPct', 'legUsable', 'alignedDelta', 'contentAgree', 'comparable', 'gateVerdict'];
// STACK lives outside the constant block in the workflow (it is read from A.stack there too); the
// per-stack floor default needs it (DECISIONS 4.54).
const makeGate = (args, runAcross) => new Function('A', 'runAcrossJobPctIn',
  `const STACK = A.stack || 'megatron';\n${constBlock()}\nrunAcrossJobPct = runAcrossJobPctIn || 0;\n` +
  `${NAMES.map(extract).join('\n')}\nreturn { gateVerdict, legUsable };`)(args || {}, runAcross || 0);
// spec.op === 'usable': the workflow's own legUsable() on ref and cand (the packing path, audit ORCH-A4).
const judge = (spec) => {
  const g = makeGate(spec.args, spec.run_across_job_pct);
  if (spec.op === 'usable') return { ref: g.legUsable(spec.ref), cand: g.legUsable(spec.cand) };
  return g.gateVerdict({ ref: spec.ref, cand: spec.cand, comparison: spec.comparison, mode: spec.mode || 'fast',
                         l0: spec.l0, l1: spec.l1, allowedImpl: spec.allowedImpl });
};
// One spec -> one verdict; an ARRAY of specs -> an array of verdicts (calibrate_band batches them).
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(Array.isArray(input) ? input.map(judge) : judge(input)) + '\n');
