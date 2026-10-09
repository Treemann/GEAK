#!/usr/bin/env node
// Execute training_workflow.js END TO END with stubbed agents. No GPU, no model, no network.
//
// WHY. Three launches have now died on defects that are valid JavaScript, pass `node --check`, and
// throw before any useful work: a second top-level `export`, a sandbox clock call, and twice a
// temporal-dead-zone reference (`CONFIG_SCHEMA`, then `curOverlay`). Each time the answer was
// another text rule in preflight, and each rule covered exactly the one shape that had just bitten.
// The last one still let `curOverlay` through and cost 3 agents and 24 minutes.
//
// A text rule guesses at the failure. RUNNING the orchestrator finds it. This harness reproduces the
// Workflow runtime closely enough to execute every top-level statement, every phase, the kernel
// loop, the gate arithmetic and the result assembly -- so a TDZ, a typo, an undefined reference or a
// broken branch surfaces in under a second instead of 24 minutes into a 7B run.
//
// WHAT IT DOES NOT CHECK. Agent behaviour, prompt quality, or any number. The stubs return the
// minimum shape each call site reads. A green run here means "the orchestration executes", not "the
// result is right" -- those are the unit tests and the real runs.
'use strict';
const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..', '..', 'training_workflow.js');
let src = fs.readFileSync(WF, 'utf8');

// The runtime special-cases the leading `export const meta` and runs the rest as a function body.
const metaEnd = src.indexOf('};', src.indexOf('export const meta')) + 2;
src = src.slice(metaEnd);

let failures = 0;
const ok = (c, m) => { if (!c) { console.error('  FAIL:', m); failures++; } else console.log('  ok:', m); };

const leg = (o = {}) => Object.assign({
  status: 'ok', hang_detected: false, occupancy_verdict: { verdict: 'clean' },
  useful_tok_s_per_gpu: 850, worst_iqr_spread_pct: 0.08, peak_reserved_mb: 80578,
  config_digest: 'd1548a7a764bdada', digest_schema: 3, impl_digest: 'aaa', ranks_agree_on_argv: true,
  dispersion_reliable: true, min_steady_steps: 25, probe_dir: '/work/runs/x/probe',
}, o);

// One stub per role phase, returning the minimum shape its call site reads.
let CONFIG_L1 = { pass: true, checks: {} };   // overridden per test below
function stubFor(prompt) {
  const p = String(prompt);
  const is = (s) => p.includes(s);
  if (is('PHASE=setup') || is('training_director. PHASE=setup')) {
    return { baseline_legs: [leg(), leg({ worst_iqr_spread_pct: 0.11 }), leg()],
             across_job_band_pct: 0.07, trajectory_band_path: '/work/band.json',
             config_digest: 'd1548a7a764bdada', warmup_steps: 11, pad_fraction_pct: 70.1 };
  }
  if (is('PHASE=reprofile') || is('PHASE=profile')) {
    return { buckets: {}, top_n: [{ kernel: 'k1', short_name: 'k1' }],
             config_diagnosis: { verdict: 'kernel_work_can_pay', blocking: [], opportunity: [] },
             rank_skew_pct: 0.01 };
  }
  if (is('PHASE=strategize')) {
    return { head_candidates: [
      { kernel: 'head_a', short_name: 'head_a', critical_path_share_pct: 18, attainable_speedup: 1.5 },
      { kernel: 'head_b', short_name: 'head_b', critical_path_share_pct: 6, attainable_speedup: 2.0 }],
      stop: false };
  }
  if (is('PHASE=config')) {
    return { lever: 'comm_overlap', flags_added: ['--overlap-grad-reduce'], ceiling_pct: 2.27,
             ref: leg(), cand: leg({ useful_tok_s_per_gpu: 866 }),
             comparison: { delta_pct: 1.9, window_artifact_pct: 0.05 },
             config_digest_delta: { key: 'overlap-grad-reduce', ref: null, cand: 'true' },
             peak_reserved_mb: { ref: 80578, cand: 80578 },
             memory_conversion: { freed_mb: 0, would_buy: null },
             ...(CONFIG_L1 === undefined ? {} : { l1: CONFIG_L1 }) };
  }
  if (is('PHASE=extract')) return { task_dir: '/work/tasks/t', op: 'o', has_backward: true, engagement: {} };
  if (is('PHASE=integrate')) {
    return { ref: leg(), cand: leg({ useful_tok_s_per_gpu: 900, impl_digest: 'bbb' }),
             comparison: { delta_pct: 5.8, window_artifact_pct: 0.1 },
             engagement: { forward: true, backward: true }, overlay_path: '/work/ovl', notes: 'n' };
  }
  if (is('PHASE=gate')) {
    return { l0: { pass: true }, l1: { pass: true }, l2_recommended: false, risk_class: 'low',
             kb_rejection_card: null };
  }
  if (is('PHASE=finalize')) return { overlay: '/work/ovl', patch: 'p', launch: 'l', engagement_recheck: {} };
  if (is('PHASE=report')) return 'report text';
  if (is('PHASE=validate')) {
    return { validation_status: 'confirmed', baseline_useful_tok_s: 850, final_useful_tok_s: 900,
             delta_pct: 5.88, band_pct: 0.15 };
  }
  if (is('date +%s%3N') || is('epoch_ms')) return { epoch_ms: 1791468713613 + 1000 };
  return {};
}

async function dryRun(args) {
  const calls = { agent: 0, workflow: 0, phases: [] };
  const globals = {
    args,
    log: () => {},
    phase: (t) => calls.phases.push(t),
    agent: async (prompt) => { calls.agent++; return stubFor(prompt); },
    workflow: async () => { calls.workflow++; return { eval_dir: '/work/exp', final_patch: 'p', final_geomean: 2.0 }; },
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (it) => {
      let v = it; for (const s of stages) v = await s(v, it, 0); return v;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const names = Object.keys(globals);
  // eslint-disable-next-line no-new-func
  const fn = new Function(...names, `return (async () => { ${src} })();`);
  const result = await fn(...names.map((n) => globals[n]));
  return { result, calls };
}

(async () => {
  const base = {
    workflow_dir: path.resolve(__dirname, '..', '..'),
    workload: '/x/workloads/qwen2_5_math_7b.env',
    nproc: 8, gpu_ids: '0,1,2,3,4,5,6,7', iters: 30,
    budget: 2, max_gates: 4, kernel_budget: 3,
    t0_epoch_ms: 1791468713613, across_job_floor_pct: 0.15,
  };

  try {
    const { result, calls } = await dryRun(base);
    ok(true, `fast mode executes end to end (${calls.agent} agent calls, ${calls.workflow} lane calls)`);
    ok(calls.phases.includes('Setup') && calls.phases.includes('Validate'),
      `every phase runs: ${calls.phases.join(' -> ')}`);
    ok(result && result.status, `a result is assembled (status=${result.status})`);
    ok(!calls.phases.includes('ConfigSweep'), 'fast mode does NOT enter ConfigSweep');
  } catch (e) {
    ok(false, `fast mode threw: ${e && e.message}`);
  }

  // The branch that cost three launches: deep mode reaches ConfigSweep, which reads state declared
  // elsewhere in the file. Executing it is the only check that covers a temporal dead zone.
  try {
    const { result, calls } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
    ok(calls.phases.includes('ConfigSweep'), 'deep mode enters ConfigSweep');
    ok(result && result.config_sweep, 'ConfigSweep records a verdict in the result');
    const cs = result.config_sweep || {};
    // Shape is ALWAYS {levers:[...]}, never "an object when one and an array when several". The
    // report and validate phases read this; a shape that depends on the input is a reader bug
    // waiting to happen, and this assertion is what caught the change when the loop went in.
    ok(Array.isArray(cs.levers) && cs.levers.length === 1,
      `one lever by default, as a list (got ${Array.isArray(cs.levers) ? cs.levers.length : typeof cs.levers})`);
    ok(cs.levers[0] && cs.levers[0].verdict &&
       cs.levers[0].verdict.delta_source === 'window-aligned',
      'the config verdict carries an aligned delta, like a kernel verdict');
    ok('accepted' in cs, 'the sweep names its accepted lever (or null) rather than leaving it implied');
  } catch (e) {
    ok(false, `deep mode threw: ${e && e.message}`);
  }

  // A sweep that sweeps. Named levers are measured in ONE run so they share a baseline and a band.
  try {
    const { result } = await dryRun({ ...base, deep_mode: 'true', budget: 0, max_gates: 4,
                                      config_levers: 'recompute_granularity,attention_backend' });
    const cs = (result && result.config_sweep) || {};
    ok(Array.isArray(cs.levers) && cs.levers.length === 2,
      `two named levers produce two records (got ${Array.isArray(cs.levers) ? cs.levers.length : typeof cs.levers})`);
    ok(cs.levers[0].requested === 'recompute_granularity' &&
       cs.levers[1].requested === 'attention_backend',
      'each record remembers which lever was REQUESTED, not only what the Tuner returned');
    ok(Array.isArray(cs.not_measured) && cs.not_measured.length === 0,
      'nothing silently dropped: not_measured is explicit');
  } catch (e) {
    ok(false, `multi-lever sweep threw: ${e && e.message}`);
  }

  // The gate budget must bound the sweep, not be bounded by it.
  try {
    const { result } = await dryRun({ ...base, deep_mode: 'true', budget: 0, max_gates: 1,
                                      config_levers: 'recompute_granularity,attention_backend' });
    const cs = (result && result.config_sweep) || {};
    ok(cs.levers.length === 1 && cs.not_measured.includes('attention_backend'),
      'a spent gate budget stops the sweep and SAYS which lever went unmeasured');
  } catch (e) {
    ok(false, `budget-bounded sweep threw: ${e && e.message}`);
  }

  // A config lever is held to the same correctness gate as a kernel. Same +1.9% aligned win, three
  // L1 situations. Before 2026-10-09 the ConfigSweep call carried no l1 at all and all three would
  // have been ACCEPTED -- including Megatron's flash attention backend, whose gradient is 2.7x off.
  for (const [label, l1, expect] of [
    ['L1 passed', { pass: true, checks: {} }, true],
    ['L1 FAILED (step-1 grad_norm 60.7%)', { pass: false, reason: 'step-1 grad_norm differs by 60.74%' }, false],
    ['no L1 returned at all', undefined, false]]) {
    CONFIG_L1 = l1;
    try {
      const { result } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
      const lv = (((result || {}).config_sweep || {}).levers || [])[0] || {};
      const v = lv.verdict || {};
      ok(v.accepted === expect, `config lever with ${label}: accepted=${v.accepted} (${v.klass})`);
    } catch (e) {
      ok(false, `config lever with ${label} threw: ${e && e.message}`);
    }
  }
  CONFIG_L1 = { pass: true, checks: {} };

  console.log();
  if (failures) { console.error(`${failures} check(s) FAILED`); process.exit(1); }
  console.log('all checks passed');
})();
