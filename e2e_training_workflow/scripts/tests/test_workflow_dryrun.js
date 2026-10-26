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
  step_shape: 'fixed', workload_identity: { 'seq-length': '4096' },
  implementation: { 'attention-backend': 'fused' },
}, o);
// The carried overlay stack, read back from the REF_EXPECTED leg the orchestrator hands the
// Integrator: the stub's ref IS that stack and its cand extends it by one component, as a real
// Integrator's legs do (the gate now checks this, audit G3).
const refShaFrom = (prompt) => {
  const i = prompt.indexOf('"REF_EXPECTED"');
  const m = i >= 0 ? /"GEAK_OVERLAY_SHA": "([^"]*)"/.exec(prompt.slice(i, prompt.indexOf('## ', i + 1) > 0 ? prompt.indexOf('## ', i + 1) : undefined)) : null;
  return m ? m[1] : '';
};
// The carried stack's impl_digest, read back from REF_EXPECTED: a real gate ref IS that leg's implementation.
const refImplFrom = (prompt) => {
  const i = prompt.indexOf('"REF_EXPECTED"');
  const m = i >= 0 ? /"impl_digest": "([^"]*)"/.exec(prompt.slice(i)) : null;
  return m ? m[1] : 'aaa';
};
// The whole REF_EXPECTED leg object from a prompt (balanced braces), or null.
const refExpected = (prompt) => {
  const i = prompt.indexOf('"REF_EXPECTED": {');
  if (i < 0) return null;
  let j = prompt.indexOf('{', i), depth = 0;
  for (let k = j; k < prompt.length; k++) {
    if (prompt[k] === '{') depth++;
    else if (prompt[k] === '}') { depth--; if (!depth) { try { return JSON.parse(prompt.slice(j, k + 1)); } catch (e) { return null; } } }
  }
  return null;
};
let L0R_OVERRIDE;                          // the verifier's l0_verify result; undefined = correct, no evidence
let VDELTA_OVERRIDE;
let INTEG_CMP_SEQ = null;                  // queued `comparison` objects, one per integrate call                        // (refDir, candDir) -> the verifier's delta_pct for that pairing
let REFBIND_OVERRIDE;
let SETUP_BARE;                             // true: Setup ignores BASE_OVERLAY_CMD (a stacked run with a bare baseline)                       // (legOverrides) for the integrate ref leg; undefined = bound
let INTEG_N = 0;
let AUDIT_OVERRIDE;
let QUALITY_SEQ = null;                    // queued quality-phase replies (extension tests)
let L1V_OVERRIDE;                          // (refDir, candDir, prompt) -> the verifier's recomputed l1; undefined = pass
let QUALV_OVERRIDE;                        // (args) -> quality_ab_final JSON; undefined = echo the Director's last reply
let PAIRS_DROP;                            // (refDir, candDir) -> true to drop that pairing from the verifier reply
let LAST_QUALITY = null;
let L0X_OVERRIDE;                          // extra fields in the Auditor's l0 (L1-plan evidence)
let VALIDATE_OVERRIDE;                     // fields merged into the Director's Validate reply
let REUSEV_OVERRIDE;                       // quality_reuse.py check result (KIND quality_reuse); undefined = valid
let PACKV_OVERRIDE;                        // verify_gate --kind packing result; undefined = confirms
let ORACLE_OVERRIDE;                       // the Auditor's l0.oracle block; undefined = template v2
let MGCA_OVERRIDE;                         // the verifier's mgc_applicability; undefined = not applicable
let MGC_OVERRIDE;                          // the Auditor's main_grad_contract block; undefined = applicable + passed
let CARRIED_OVERRIDE;                      // the verifier's `carried` block; undefined = all carried overlays engaged
let VERIFY_OVERRIDE;                       // (refDir, candDir) -> verdict klass; undefined = 'win'                        // undefined = normal stub; null = the Auditor failed
let ENG_OVERRIDE;

// One stub per role phase, returning the minimum shape its call site reads.
let CONFIG_L1 = { pass: true, checks: {} };   // overridden per test below
let CONFIG_OVERRIDE = null;                     // replaces fields of the config stub per test
let STRAT_HEADS = null;                         // replaces the Architect's head_candidates per test
let EXTRACT_OVERRIDE = null;                    // replaces fields of the extract stub per test
function stubFor(prompt) {
  const p = String(prompt);
  const is = (s) => p.includes(s);
  if (is('PHASE=setup') || is('training_director. PHASE=setup')) {
    // With BASE_OVERLAY_CMD a real Director measures the STACKED baseline (the legs carry the overlay).
    const st = prompt.includes('"BASE_OVERLAY_CMD"') && !SETUP_BARE
      ? { implementation: { 'attention-backend': 'fused', GEAK_OVERLAY: 'b.json', GEAK_OVERLAY_SHA: 'base0' } } : {};
    return { baseline_legs: [leg(st), leg({ worst_iqr_spread_pct: 0.11, ...st }), leg(st)],
             across_job_band_pct: 0.07, trajectory_band_path: '/work/band.json',
             config_digest: 'd1548a7a764bdada', warmup_steps: 11, pad_fraction_pct: 70.1 };
  }
  if (is('PHASE=reprofile') || is('PHASE=profile')) {
    return { buckets: {}, top_n: [{ kernel: 'k1', short_name: 'k1' }],
             config_diagnosis: { verdict: 'kernel_work_can_pay', blocking: [], opportunity: [] },
             rank_skew_pct: 0.01 };
  }
  if (is('PHASE=strategize')) {
    return { head_candidates: STRAT_HEADS || [
      { kernel: 'head_a', short_name: 'head_a', critical_path_share_pct: 18, attainable_speedup: 1.5 },
      { kernel: 'head_b', short_name: 'head_b', critical_path_share_pct: 6, attainable_speedup: 2.0 }],
      stop: false };
  }
  if (is('PHASE=config')) {
    // A named lever gets its own flags, so two accepted levers are distinguishable downstream.
    const LEVER_FLAGS = { recompute_granularity: ['--recompute-granularity', 'selective'],
                          attention_backend: ['--attention-backend', 'flash'] };
    const req = Object.keys(LEVER_FLAGS).find((k) => is(`"REQUESTED_LEVER": "${k}"`));
    if (req) {
      // A real Tuner's ref IS the carried stack (REF_EXPECTED, incl. levers accepted earlier this run).
      const ex = refExpected(prompt) || {};
      const refImpl = ex.implementation || leg().implementation;
      return { lever: req, flags_added: LEVER_FLAGS[req], ceiling_pct: 2.0,
               ref: leg({ implementation: refImpl, impl_digest: ex.impl_digest || 'aaa' }),
               // An implementation lever: config_digest unchanged, the lever's own key in `implementation`.
               cand: leg({ useful_tok_s_per_gpu: 866, impl_digest: `after_${req}`,
                           implementation: Object.assign({}, refImpl,
                             { [LEVER_FLAGS[req][0].slice(2)]: LEVER_FLAGS[req][1] }) }),
               comparison: { delta_pct: 1.9, window_artifact_pct: 0.05 },
               ...(CONFIG_L1 === undefined ? {} : { l1: CONFIG_L1 }), ...(CONFIG_OVERRIDE || {}) };
    }
    return { lever: 'comm_overlap', flags_added: ['--overlap-grad-reduce'], ceiling_pct: 2.27,
             ref: leg(), cand: leg({ useful_tok_s_per_gpu: 866 }),
             comparison: INTEG_CMP_SEQ && INTEG_CMP_SEQ.length ? INTEG_CMP_SEQ.shift() : { delta_pct: 1.9, window_artifact_pct: 0.05 },
             config_digest_delta: { key: 'overlap-grad-reduce', ref: null, cand: 'true' },
             peak_reserved_mb: { ref: 80578, cand: 80578 },
             memory_conversion: { freed_mb: 0, would_buy: null },
             ...(CONFIG_L1 === undefined ? {} : { l1: CONFIG_L1 }), ...(CONFIG_OVERRIDE || {}) };
  }
  if (is('PHASE=extract')) return { task_dir: '/work/tasks/t', op: 'o', has_backward: true, engagement: {},
                                    ...(EXTRACT_OVERRIDE || {}) };
  if (is('PHASE=integrate')) {
    const rs = refShaFrom(prompt), add = `k${++INTEG_N}`;
    const impl = (sha) => (sha ? { 'attention-backend': 'fused', GEAK_OVERLAY: 'm.json', GEAK_OVERLAY_SHA: sha }
                               : { 'attention-backend': 'fused' });
    return { ref: leg({ implementation: impl(rs), impl_digest: refImplFrom(prompt), dir: `/work/runs/gref_${INTEG_N}`,
                        probe_dir: `/work/runs/gref_${INTEG_N}/probe`, ...(REFBIND_OVERRIDE || {}) }),
             cand: leg({ useful_tok_s_per_gpu: 900, impl_digest: `bbb${INTEG_N}`, implementation: impl(rs ? `${rs}+${add}` : add),
                         dir: `/work/runs/gcand_${INTEG_N}`, probe_dir: `/work/runs/gcand_${INTEG_N}/probe` }),
             legs_measured: { refs: [`/work/runs/gref_${INTEG_N}`, `/home/u/geak_train/runs/gref_${INTEG_N}/probe`,
                                     `/work/runs/gref2_${INTEG_N}`],
                              cands: [`/work/runs/gcand_${INTEG_N}`] },
             comparison: INTEG_CMP_SEQ && INTEG_CMP_SEQ.length ? INTEG_CMP_SEQ.shift() : { delta_pct: 5.8, window_artifact_pct: 0.1 },
             engagement: ENG_OVERRIDE !== undefined ? ENG_OVERRIDE : { forward: true, backward: true },
             overlay_path: '/work/ovl', notes: 'n' };
  }
  if (is('PHASE=verify')) {
    if (VERIFY_OVERRIDE === null) return null;
    if (prompt.includes('"KIND": "l0"')) {             // l0_verify.sh (DECISIONS 4.99)
      const task = (/"L0_TASK": "([^"]*)"/.exec(prompt) || [])[1] || '/work/tasks/t';
      return { pairs: [{ ref_dir: task, cand_dir: '/work/ovl', l0_recomputed: L0R_OVERRIDE !== undefined ? L0R_OVERRIDE
        : { correct: true, out_bit_identical: null, per_output: null, oracle: { kind: 'template', task_dir: task } } }] };
    }
    if (prompt.includes('"KIND": "quality"')) {        // quality_ab_final recomputed (audit ORCH-A8)
      const m = /"QUALITY_ARGS": (\[[^\]]*\])/.exec(prompt);
      const qa = m ? JSON.parse(m[1]) : [];
      const r = QUALV_OVERRIDE !== undefined ? (QUALV_OVERRIDE && QUALV_OVERRIDE(qa))
        : (LAST_QUALITY ? { quality: LAST_QUALITY.quality, stability: LAST_QUALITY.stability } : null);
      return { pairs: [{ ref_dir: `${qa[0]}_ref`, cand_dir: `${qa[0]}_cand`, quality_verdict: r }] };
    }
    if (prompt.includes('"KIND": "quality_reuse"')) {   // audit 2026-10-25 ID-3: the reuse proof is recomputed
      return { pairs: [{ ref_dir: '/p_ref', cand_dir: '/p_cand',
        reuse_check: REUSEV_OVERRIDE !== undefined ? REUSEV_OVERRIDE : { valid: true, seeds: [] } }] };
    }
    if (prompt.includes('"KIND": "packing"')) {       // verify_gate --kind packing (audit ORCH-A4)
      return { pairs: [{ ref_dir: '/work/runs/pr', cand_dir: '/work/runs/pc', packing: PACKV_OVERRIDE !== undefined
        ? PACKV_OVERRIDE : { delta_pct: 265.9, pack_gate_pass: true, data_matches: true, legs_usable: true, why: [] } }] };
    }
    const grab = (key) => { const m = new RegExp(`"${key}": \\[([^\\]]*)\\]`).exec(prompt); return m ? JSON.parse(`[${m[1]}]`) : []; };
    const refs = grab('REF_DIRS'), cands = grab('CAND_DIRS');
    const pairs = [];
    const l1k = (/"L1_KIND": "([a-z_]+)"/.exec(prompt) || [])[1];
    for (const r of refs) for (const c of cands) {
      if (PAIRS_DROP && PAIRS_DROP(r, c)) continue;
      const klass = VERIFY_OVERRIDE ? VERIFY_OVERRIDE(r, c) : 'win';
      pairs.push({ ref_dir: r, cand_dir: c, verdict: { klass, delta_pct: VDELTA_OVERRIDE ? VDELTA_OVERRIDE(r, c) : 2.0,
                                                         band_pct: 0.5, reason: 'stub' },
                   engagement: { forward: true, backward: true }, comparison: {},
                   carried: CARRIED_OVERRIDE !== undefined ? CARRIED_OVERRIDE : { ok: true, lost: [], slots: {} },
                   ...(l1k ? { l1: L1V_OVERRIDE !== undefined ? L1V_OVERRIDE(r, c, prompt)
                     : { pass: true, failures: [], recomputed: { kind: l1k } } } : {}),
                   error: null });
    }
    return { pairs, ...(prompt.includes('"MGC_TASK"') ? { mgc_applicability: MGCA_OVERRIDE !== undefined ? MGCA_OVERRIDE
      : { head: null, applicable: false, evidence: [] } } : {}) };
  }
  if (is('PHASE=gate')) {
    if (AUDIT_OVERRIDE !== undefined) return AUDIT_OVERRIDE;
    return { l0: { ...(L0X_OVERRIDE || {}), pass: true, oracle: ORACLE_OVERRIDE !== undefined ? ORACLE_OVERRIDE
                                      : { kind: 'template', template_version: 2, scores_every_output: true, task_dir: '/work/tasks/t' } },
             l1: { pass: true }, l2_recommended: false, risk_class: 'low',
             main_grad_contract: MGC_OVERRIDE !== undefined ? MGC_OVERRIDE : { applicable: true, passed: true },
             kb_rejection_card: null };
  }
  if (is('PHASE=finalize')) return { overlay: '/work/ovl', patch: 'p', launch: 'l',
                                     engagement_recheck: { forward: true, backward: true } };
  if (is('PHASE=report')) return 'report text';
  if (is('PHASE=quality')) {
    LAST_QUALITY = QUALITY_SEQ ? QUALITY_SEQ.shift()
      : { cleared: true, quality: 'PASS-quality: stub', stability: 'PASS-stability: stub', legs_prefix: '/work/runs/qgate_stub' };
    return LAST_QUALITY;
  }
  if (is('PHASE=validate')) {
    return { validation_status: 'confirmed', baseline_useful_tok_s: 850, final_useful_tok_s: 900,
             delta_pct: 5.88, band_pct: 0.15, estimator: 'paired per-step ratio', wall_delta_pct: 5.1,
             l1: { pass: true }, legs: [], ref_dir: '/work/runs/val_ref', final_dir: '/work/runs/val_final',
             ...(VALIDATE_OVERRIDE || {}) };
  }
  if (is('date +%s%3N') || is('epoch_ms')) return { epoch_ms: 1791468713613 + 1000 };
  return {};
}

// Every opts.phase seen across ALL dry runs (fast, deep, quality); checked once at the end.
const AGENT_GROUPS = new Set();

async function dryRun(args) {
  INTEG_N = 0;
  const calls = { agent: 0, workflow: 0, phases: [], prompts: [], agentLabels: [] };
  const globals = {
    args,
    log: () => {},
    phase: (t) => calls.phases.push(t),
    agent: async (prompt, opts) => {
      calls.agent++; calls.prompts.push(String(prompt)); calls.agentLabels.push(opts && opts.label);
      AGENT_GROUPS.add(opts && opts.phase);
      return stubFor(prompt);
    },
    workflow: async (ref) => {
      calls.workflow++; calls.lanePaths = (calls.lanePaths || []).concat([ref && ref.scriptPath]);
      return { eval_dir: '/work/exp', final_patch: 'p', final_geomean: 2.0 };
    },
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
    // The runtime refuses a child scriptPath it cannot verify as readable, and it refused one that
    // contained `..` -- killing a 7B run at its first lane call after five agents of good work.
    ok((calls.lanePaths || []).length > 0 && calls.lanePaths.every((p) => p && !/(^|\/)\.\.?(\/|$)/.test(p)),
      `lane scriptPath is normalized, no '.' or '..' segments (${(calls.lanePaths || [])[0]})`);
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
    const { result, calls } = await dryRun({ ...base, deep_mode: 'true', budget: 0, max_gates: 4,
                                      config_levers: 'recompute_granularity,attention_backend' });
    const cs = (result && result.config_sweep) || {};
    ok(calls.agentLabels.includes('config_tuner:recompute_granularity') &&
       calls.agentLabels.includes('config_tuner:attention_backend'),
      'each named lever gets its own agent label (the per-lever label used to be silently dropped)');
    ok(Array.isArray(cs.levers) && cs.levers.length === 2,
      `two named levers produce two records (got ${Array.isArray(cs.levers) ? cs.levers.length : typeof cs.levers})`);
    ok(cs.levers[0].requested === 'recompute_granularity' &&
       cs.levers[1].requested === 'attention_backend',
      'each record remembers which lever was REQUESTED, not only what the Tuner returned');
    ok(Array.isArray(cs.not_measured) && cs.not_measured.length === 0,
      'nothing silently dropped: not_measured is explicit');
    // P1-C (2026-10-22): accepted levers ACCUMULATE and are CARRIED into every later leg.
    const ac = result.accepted_config || {};
    ok(cs.accepted_all && cs.accepted_all.length === 2 && ac.levers && ac.levers.length === 2,
      `both accepted levers are reported, not just the first (accepted_all=${JSON.stringify(cs.accepted_all)})`);
    ok(JSON.stringify(ac.flags_added) === JSON.stringify(['--recompute-granularity', 'selective', '--attention-backend', 'flash']),
      `accepted flags are MERGED, not replaced (${JSON.stringify(ac.flags_added)})`);
    const later = calls.prompts.filter((p) => /PHASE=(integrate|validate|finalize)/.test(p));
    ok(later.length > 0 && later.every((p) => p.includes('--extra-train-args "--recompute-granularity selective --attention-backend flash"')),
      `every later leg-launching phase is told to carry BOTH levers (${later.length} prompts)`);
    const second = calls.prompts.filter((p) => p.includes('"REQUESTED_LEVER": "attention_backend"'));
    ok(second.length === 1 && second[0].includes('--extra-train-args "--recompute-granularity selective"'),
      'the second lever is measured ON TOP of the first (its legs carry the first lever)');
    ok(later.some((p) => p.includes('"impl_digest": "after_attention_backend"')),
      'the carried impl digest is refreshed from the last accepted lever\'s cand leg');
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

  // audit 2026-10-25 item 13: a flag+env lever (the deliverable's `fast`) is accepted, verified with
  // --lever-env, carried into later legs and into Validate's recomputation.
  {
    const fastCand = leg({ useful_tok_s_per_gpu: 866, impl_digest: 'fast',
      implementation: Object.assign({}, leg().implementation,
        { 'no-check-for-nan-in-loss-and-grad': 'true', 'env:GEAK_NAN_FROM_GRAD_NORM': '1' }) });
    const fast = { lever: 'nan_from_grad_norm', flags_added: ['--no-check-for-nan-in-loss-and-grad'], cand: fastCand };
    CONFIG_L1 = { pass: true, checks: { step1_loss_rel_pct: { candidate_kind: 'overlay' } } };
    try {
      CONFIG_OVERRIDE = { ...fast };
      const { result: e0 } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
      const v0 = ((((e0 || {}).config_sweep || {}).levers || [])[0] || {}).verdict || {};
      ok(v0.accepted === false && v0.klass === 'incomparable', `flag+env lever WITHOUT env_added is refused (${v0.klass})`);
      CONFIG_OVERRIDE = { ...fast, env_added: { GEAK_NAN_FROM_GRAD_NORM: '1' } };
      const { result: e1, calls: ec1 } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
      const lv1 = (((e1 || {}).config_sweep || {}).levers || [])[0] || {};
      ok(lv1.verdict && lv1.verdict.accepted === true && lv1.env_added && lv1.env_added.GEAK_NAN_FROM_GRAD_NORM === '1',
        `flag+env lever WITH env_added is accepted and recorded (${lv1.verdict && lv1.verdict.klass})`);
      ok(ec1.prompts.some((p) => p.includes('"KIND": "config"') && p.includes('"LEVER_ENV": "GEAK_NAN_FROM_GRAD_NORM"')),
        'the config verifier is told the lever env (--lever-env)');
      ok(ec1.prompts.some((p) => p.includes('PHASE=validate') && p.includes('--extra-env "GEAK_NAN_FROM_GRAD_NORM=1"')),
        'the carried env reaches later legs (Validate is told --extra-env)');
      ok(ec1.prompts.some((p) => p.includes('"KIND": "stack"') && p.includes('"LEVER_ENV": "GEAK_NAN_FROM_GRAD_NORM"')),
        'Validate\'s recomputation allows the lever env');
      ok(e1.status === 'ok' && e1.accepted_config && e1.accepted_config.env_added.GEAK_NAN_FROM_GRAD_NORM === '1',
        `a run whose accepted change is a flag+env lever can end ok (status=${e1.status})`);
      CONFIG_OVERRIDE = { ...fast, env_added: { 'BAD NAME': '1', GEAK_NAN_FROM_GRAD_NORM: '1; rm -rf /' } };
      const { result: e2 } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
      const v2 = ((((e2 || {}).config_sweep || {}).levers || [])[0] || {}).verdict || {};
      ok(v2.accepted === false, 'unsafe env names / values are dropped, so the lever is not accepted');
    } catch (e) {
      ok(false, `flag+env lever cases threw: ${e && e.stack}`);
    }
    CONFIG_OVERRIDE = null; CONFIG_L1 = { pass: true, checks: {} };
  }

  // PACKING is the one gateable Class-B lever. Real numbers: pack_gate.py PASS on fresh packs,
  // equal-useful-token gain +266% (833 -> 3,048 useful tok/s/GPU at 7B).
  const pk = { lever: 'packing', flags_added: ['--train-data-path', '/work/data/x_packed.jsonl', 'packing'],
               comparison_tokens: { delta_pct: 265.9 } };
  for (const [label, over, expect, klass] of [
    ['packing, gate PASS', { ...pk, packing_gate: { pass: true, verdict: 'PASS' } }, true, 'win'],
    ['packing, gate FAIL (wrong system prompt, 196/196 token mismatches)',
       { ...pk, packing_gate: { pass: false, verdict: 'FAIL', checks: { token_identity: { pass: false } } } }, false, 'packing_gate_failed'],
    ['packing, no gate result', { ...pk }, false, 'no_correctness_gate'],
    // audit 2026-10-25 ORCH-A4: the RECOMPUTED numbers decide, and unusable legs refuse.
    ['packing, verifier recomputes +1% (agent said +266%)', { ...pk, packing_gate: { pass: true, verdict: 'PASS' },
       __packv: { delta_pct: 1.0, pack_gate_pass: true, data_matches: true, legs_usable: true, why: [] } }, false, 'no_win'],
    ['packing, pack_gate verdict is about another dataset', { ...pk, packing_gate: { pass: true, verdict: 'PASS' },
       __packv: { delta_pct: 265.9, pack_gate_pass: true, data_matches: false, legs_usable: true, why: ['other data'] } }, false, 'verifier_refused'],
    ['packing, ref leg failed / contaminated', { ...pk, packing_gate: { pass: true, verdict: 'PASS' },
       __ref: { status: 'failed', occupancy_verdict: { verdict: 'contaminated' } } }, false, 'unusable_ref'],
    ['GBS change (ungateable Class B)', { lever: 'gbs', flags_added: ['--global-batch-size', '128'] }, false, null]]) {
    PACKV_OVERRIDE = over.__packv;
    CONFIG_OVERRIDE = Object.assign({}, over);
    delete CONFIG_OVERRIDE.__packv;
    delete CONFIG_OVERRIDE.__ref;
    if (over.__ref) CONFIG_OVERRIDE.ref = leg(over.__ref);
    try {
      const { result } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
      const lv = (((result || {}).config_sweep || {}).levers || [])[0] || {};
      const v = lv.verdict || {};
      const accepted = lv.refused ? false : v.accepted;
      ok(accepted === expect && (klass === null ? lv.refused === true : v.klass === klass),
        `${label}: accepted=${accepted} (${lv.refused ? 'REFUSED' : v.klass})`);
      ok(!!result.accepted_config === expect,
        `${label}: top-level accepted_config is ${result.accepted_config ? 'set' : 'null'}, matching the verdict`);
    } catch (e) {
      ok(false, `${label} threw: ${e && e.message}`);
    }
  }
  CONFIG_OVERRIDE = null;

  // ---- 2026-10-14: stacked baseline, carried overlay on re-profile, queue ranking ----------------
  const phasePrompts = (calls, ph) => calls.prompts.filter((p) => p.includes(`PHASE=${ph}`));
  try {
    const RAW = '/repo/e2e_training_workflow/scripts/stack_overlay_env.sh --cond s8';
    // --stack is appended so stack_overlay_env.sh refuses a condition of the other stack (2026-10-20).
    const CMD = `${RAW} --stack ${base.stack || 'megatron'}`;
    const { result, calls } = await dryRun({ ...base, baseline_overlay: RAW });
    ok(result.baseline_overlay === CMD, 'the result names the reference every delta is relative to, with --stack appended');
    const { result: rx } = await dryRun({ ...base, baseline_overlay: `${RAW} --stack megatron` });
    ok(rx.baseline_overlay === `${RAW} --stack megatron`, 'an explicit --stack is kept, not doubled');
    for (const ph of ['setup', 'reprofile', 'integrate', 'finalize', 'validate']) {
      const ps = phasePrompts(calls, ph);
      ok(ps.length > 0 && ps.every((p) => p.includes(`"BASE_OVERLAY_CMD": "${CMD}"`) &&
                                          p.includes('STACKED BASELINE')),
        `baseline_overlay reaches EVERY ${ph} leg (${ps.length} call(s)), with the stacking rules`);
    }
    const { result: r0, calls: c0 } = await dryRun(base);
    ok(r0.baseline_overlay === null && !c0.prompts.some((p) => p.includes('BASE_OVERLAY_CMD')),
      'without baseline_overlay nothing mentions it: the bare reference is unchanged');
  } catch (e) {
    ok(false, `baseline_overlay threw: ${e && e.message}`);
  }
  try {
    const { result, calls } = await dryRun(base);
    const rp = phasePrompts(calls, 'reprofile');
    ok(result.accepted.length > 0 && rp.length >= 2, `an acceptance triggers a re-profile (${rp.length} profiles)`);
    ok(!rp[0].includes('CARRIED_OVERLAY'), 'the first profile is of the reference: no carried overlay');
    ok(rp.slice(1).every((p) => /"CARRIED_OVERLAY": \[\s*"\/work\/ovl"/.test(p)),
      'the re-profile AFTER an acceptance carries the accepted overlay (it used to profile the bare stack)');
  } catch (e) {
    ok(false, `carried re-profile threw: ${e && e.message}`);
  }
  try {
    // Emitted in the WRONG order: head_b (6 x (1 - 1/2) = 3.0%) before head_a (18 x (1 - 1/1.5) = 6.0%).
    STRAT_HEADS = [
      { kernel: 'head_b', short_name: 'head_b', critical_path_share_pct: 6, attainable_speedup: 2.0 },
      { kernel: 'head_a', short_name: 'head_a', critical_path_share_pct: 18, attainable_speedup: 1.5 }];
    const { calls } = await dryRun({ ...base, budget: 1 });
    const ex = phasePrompts(calls, 'extract');
    ok(ex.length === 1 && ex[0].includes('"short_name": "head_a"'),
      'the queue is ranked by expected gain in code, not by the order the Architect emitted');
  } catch (e) {
    ok(false, `queue ranking threw: ${e && e.message}`);
  }
  STRAT_HEADS = null;

  // ---- 2026-10-15: an op outside autograd has no backward; only it may skip the fwd+bwd rule ----
  for (const [label, over, expectLane] of [
    ['autograd op without a backward', { has_backward: false }, false],
    ['op declared outside autograd (optimizer / grad tail)', { has_backward: false, autograd: false }, true],
    ['autograd op with its backward', { has_backward: true }, true]]) {
    EXTRACT_OVERRIDE = over;
    try {
      const { calls } = await dryRun({ ...base, budget: 1 });
      ok((calls.workflow > 0) === expectLane,
        `${label}: kernel lane ${calls.workflow > 0 ? 'RUNS' : 'is refused'}`);
    } catch (e) {
      ok(false, `${label} threw: ${e && e.message}`);
    }
  }
  EXTRACT_OVERRIDE = null;

  // ---- 2026-10-22: a lever that moves WORKLOAD identity is a recipe change, never accepted --------
  try {
    CONFIG_OVERRIDE = { lever: 'lr_decay', flags_added: ['--lr-decay-iters', '500'],
      cand: leg({ useful_tok_s_per_gpu: 900, config_digest: 'moved',
                  workload_identity: { 'seq-length': '4096', 'lr-decay-iters': '500' } }) };
    const { result } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
    const lv = (((result || {}).config_sweep || {}).levers || [])[0] || {};
    ok(lv.verdict && lv.verdict.klass === 'recipe_change' && lv.verdict.accepted === false &&
       JSON.stringify(lv.verdict.identity_moved) === '["lr-decay-iters"]',
      `an identity-moving lever is refused as recipe_change, named from the legs (${lv.verdict && lv.verdict.klass})`);
  } catch (e) {
    ok(false, `recipe_change case threw: ${e && e.message}`);
  }
  CONFIG_OVERRIDE = null;

  // ---- 2026-10-22 audit: fail closed on missing correctness / engagement evidence -------------
  try {
    AUDIT_OVERRIDE = null;   // what safeAgent returns after three failed attempts
    const { result: ra } = await dryRun(base);
    ok(ra.accepted.length === 0, `an Auditor that returned nothing cannot let a kernel through (accepted=${ra.accepted.length})`);
    AUDIT_OVERRIDE = { l0: { pass: true }, l1: { pass: null } };   // trajectory_gate's initial value
    const { result: rb } = await dryRun(base);
    ok(rb.accepted.length === 0, 'l1.pass=null is not a pass');
    AUDIT_OVERRIDE = undefined;
    ENG_OVERRIDE = {};
    const { result: rc } = await dryRun(base);
    ok(rc.accepted.length === 0, 'an empty engagement object is not a proof');
  } catch (e) {
    ok(false, `fail-closed cases threw: ${e && e.message}`);
  }
  AUDIT_OVERRIDE = undefined; ENG_OVERRIDE = undefined;

  // ---- 2026-10-22: ONE pre-registered seed extension of an INCONCLUSIVE outcome test ---------------
  try {
    QUALITY_SEQ = [{ cleared: false, quality: 'INCONCLUSIVE-quality: upper bound > margin', stability: 'PASS-stability: 0',
                     legs_prefix: '/work/runs/qgate_A' },
                   { cleared: true, quality: 'PASS-quality: n=10', stability: 'PASS-stability: 0', legs_prefix: '/work/runs/qgatex_B' }];
    const { result: rq, calls: cqx } = await dryRun({ ...base, quality_gate: 'true' });
    const qps = cqx.prompts.filter((p) => p.includes('PHASE=quality'));
    ok(qps.length === 2 && qps[1].includes('"EXTENSION_OF": "/work/runs/qgate_A"') && qps[1].includes('7013,8124,9235,1346,2457'),
      'an INCONCLUSIVE outcome test triggers exactly one extension with the pre-registered seeds');
    ok(rq.quality_gate.cleared === true && rq.quality_gate.extended === true && !!rq.quality_gate.first_verdict,
      'the combined verdict is final and the first verdict is kept on the record');
    QUALITY_SEQ = [{ cleared: false, quality: 'INCONCLUSIVE-quality: x', stability: 'PASS-stability: 0', legs_prefix: '/p' },
                   { cleared: false, quality: 'INCONCLUSIVE-quality: still', stability: 'PASS-stability: 0', legs_prefix: '/q' }];
    const { calls: cq2 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(cq2.prompts.filter((p) => p.includes('PHASE=quality')).length === 2, 'never more than ONE extension');
    // DECISIONS 4.63/4.75: reused REF/control legs count only when quality_reuse.py check, run by the
    // gate_verifier with orchestrator-built arguments, proved them valid. The Director's own reuse_check is ignored.
    const reuse = { quality_gate: 'true', quality_seeds: '1111,2222', quality_reuse: { prefix: 'qabhf8_old', ref_arm: 'cand', ctrl_arm: 'ctrl70' } };
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/p',
                     reuse_check: { valid: true } }];
    REUSEV_OVERRIDE = { valid: false, seeds: [{ why: ['implementation (overlay / env knobs) differs beyond --allow-impl'] }] };
    const { result: ru1, calls: cru } = await dryRun({ ...base, ...reuse });
    ok(ru1.quality_gate.cleared === false && /not proven/.test(ru1.quality_gate.quality),
      'quality_reuse: the verifier\'s invalid check makes the gate NOT CLEARED although the Director claimed valid');
    ok(cru.prompts.some((p) => p.includes('PHASE=quality') && p.includes('qabhf8_old') && p.includes('quality_reuse.py')),
      'quality_reuse: the Director gets the reused prefix and the tool');
    const rv = cru.prompts.find((p) => p.includes('"KIND": "quality_reuse"')) || '';
    ok(rv.includes('"--prefix"') && rv.includes('"/p"') && rv.includes('"--seeds"') && rv.includes('"1111,2222"') &&
       !rv.includes('--accept-unknown-harness'),
      'quality_reuse: the check is built by the orchestrator (prefix, seeds; unknown harness NOT accepted by default)');
    const rvKernel = /"REUSE_ARGS": (\[[^\]]*\])/.exec(rv);
    ok(rvKernel && JSON.parse(rvKernel[1]).join(' ').includes('GEAK_OVERLAY_SHA'),
      'quality_reuse: accepted kernels make the overlay a declared difference');
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/p' }];
    const { calls: cbo } = await dryRun({ ...base, ...reuse, deep_mode: 'true', budget: 0,
                                          baseline_overlay: '/x/stack_overlay_env.sh --cond s8a22' });
    const rvb = /"REUSE_ARGS": (\[[^\]]*\])/.exec(cbo.prompts.find((p) => p.includes('"KIND": "quality_reuse"')) || '');
    ok(rvb && JSON.parse(rvb[1]).join(' ').includes('GEAK_OVERLAY_SHA'),
      'quality_reuse: a stacked baseline (config-only accept) also declares the overlay (the cand arm runs the base stack)');
    CONFIG_OVERRIDE = { lever: 'nan_from_grad_norm', flags_added: ['--no-check-for-nan-in-loss-and-grad'],
      env_added: { GEAK_NAN_FROM_GRAD_NORM: '1' },
      cand: leg({ useful_tok_s_per_gpu: 866, impl_digest: 'fast', implementation: Object.assign({}, leg().implementation,
        { 'no-check-for-nan-in-loss-and-grad': 'true', 'env:GEAK_NAN_FROM_GRAD_NORM': '1' }) }) };
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/p' }];
    const { calls: cnan } = await dryRun({ ...base, ...reuse, deep_mode: 'true', budget: 0 });
    CONFIG_OVERRIDE = null;
    const rvn = /"REUSE_ARGS": (\[[^\]]*\])/.exec(cnan.prompts.find((p) => p.includes('"KIND": "quality_reuse"')) || '');
    const an = rvn ? JSON.parse(rvn[1]) : [];
    const allowArg = an[an.indexOf('--allow') + 1] || '', implArg = an[an.indexOf('--allow-impl') + 1] || '';
    ok(allowArg.split(',').includes('check_for_nan_in_loss_and_grad') && implArg.split(',').includes('env:GEAK_NAN_FROM_GRAD_NORM'),
      `quality_reuse: a --no-X lever allows realized X, and its env is an allowed impl key (${allowArg} | ${implArg})`);
    REUSEV_OVERRIDE = undefined;
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/p',
                     reuse_check: { valid: false } }];
    const { result: ru2 } = await dryRun({ ...base, ...reuse });
    ok(ru2.quality_gate.cleared === true && ru2.quality_gate.reuse_check && ru2.quality_gate.reuse_check.valid === true,
      'quality_reuse: a valid independent check keeps the verdict (the Director\'s self-report does not decide)');
    REUSEV_OVERRIDE = null;
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/p',
                     reuse_check: { valid: true } }];
    const { result: ru4 } = await dryRun({ ...base, ...reuse });
    ok(ru4.quality_gate.cleared === false, 'quality_reuse: no independent check result = NOT CLEARED');
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0' }];
    const { result: ru5 } = await dryRun({ ...base, ...reuse });
    REUSEV_OVERRIDE = undefined;
    ok(ru5.quality_gate.cleared === false && /no legs_prefix/.test(ru5.quality_gate.quality),
      'quality_reuse: a Director that returns no legs_prefix cannot be checked = NOT CLEARED');
    // audit 2026-10-25 ORCH-A3: an INCONCLUSIVE first verdict with an UNPROVEN reuse must not extend into CLEARED.
    REUSEV_OVERRIDE = { valid: false, seeds: [] };
    QUALITY_SEQ = [{ cleared: false, quality: 'INCONCLUSIVE-quality: upper > margin', stability: 'PASS-stability: 0',
                     legs_prefix: '/p' },
                   { cleared: true, quality: 'PASS-quality: n=10', stability: 'PASS-stability: 0', legs_prefix: '/q' }];
    const { result: ru3, calls: cru3 } = await dryRun({ ...base, ...reuse });
    REUSEV_OVERRIDE = undefined;
    ok(ru3.quality_gate.cleared === false && cru3.prompts.filter((p) => p.includes('PHASE=quality')).length === 1,
      'quality_reuse: an unproven reuse is NOT extended, and the verdict stays NOT CLEARED');
    let threw = false;
    try { await dryRun({ ...base, quality_gate: 'true', quality_reuse: { prefix: 'x', ref_arm: 'cand' } }); } catch (e) { threw = /quality_seeds/.test(String(e && e.message)); }
    ok(threw, 'quality_reuse without explicit quality_seeds fails loudly');
    QUALITY_SEQ = [{ cleared: false, quality: 'INCONCLUSIVE-quality: the positive control was not detected', stability: 'PASS-stability: 0', legs_prefix: '/p' }];
    const { calls: cqc } = await dryRun({ ...base, quality_gate: 'true', quality_ctrl: 'ctrl70' });
    ok(cqc.prompts.filter((p) => p.includes('PHASE=quality')).length === 1,
      'an undetected positive control is NOT extended (more seeds of a dead control prove nothing)');
    QUALITY_SEQ = [{ cleared: false, quality: 'FAIL-quality: worse', stability: 'PASS-stability: 0', legs_prefix: '/p' }];
    const { calls: cq3 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(cq3.prompts.filter((p) => p.includes('PHASE=quality')).length === 1, 'a FAIL is not extended');
  } catch (e) {
    ok(false, `extension cases threw: ${e && e.message}`);
  }
  QUALITY_SEQ = null;

  // ---- 2026-10-22 ORCH-1: the gate verifier recomputes every pairing from the leg dirs ---------
  try {
    const { result: r0, calls: c0 } = await dryRun(base);
    const vp = c0.prompts.filter((p) => p.includes('PHASE=verify') && p.includes('"KIND": "kernel"'));
    ok(vp.every((p) => { const m = /"REF_DIRS": \[([^\]]*)\]/.exec(p); const xs = m ? JSON.parse(`[${m[1]}]`) : [];
         return new Set(xs.map((x) => x.replace(/\/probe$/, '').split('/').slice(-2).join('/'))).size === xs.length; }),
      'the verifier is asked for each leg once, whatever path spelling the agent used');
    ok(r0.accepted.length > 0 && vp.length === r0.accepted.length && vp[0].includes('gref2_'),
      `every accepted kernel was recomputed by the verifier, over EVERY measured leg (${vp.length} verify calls)`);
    VERIFY_OVERRIDE = () => 'inconclusive_noise';
    const { result: r1 } = await dryRun(base);
    ok(r1.accepted.length === 0, 'a verifier that recomputes the reported pair as inconclusive blocks the accept');
    VERIFY_OVERRIDE = (r) => (r.includes('gref2_') ? 'regression' : 'win');
    const { result: r2 } = await dryRun(base);
    ok(r2.accepted.length === 0, 'ANY measured pairing that is a regression vetoes the accept (no cherry-picking)');
    VERIFY_OVERRIDE = null;
    const { result: r3 } = await dryRun(base);
    ok(r3.accepted.length === 0, 'no verifier result -> no accept (fail closed)');
    // DECISIONS 4.55: a candidate that switches off an overlay the reference runs is refused, and a
    // verifier reply without the carried check cannot accept (fail closed).
    VERIFY_OVERRIDE = undefined;
    CARRIED_OVERRIDE = { ok: false, lost: ['rope'], slots: { rope: { engaged: false, fallback_total: 3360 } } };
    const { result: r4 } = await dryRun(base);
    ok(r4.accepted.length === 0, 'a candidate that silently disables a carried overlay (rope fallback) is refused');
    CARRIED_OVERRIDE = null;
    const { result: r5 } = await dryRun(base);
    ok(r5.accepted.length === 0, 'a verifier reply without the carried-overlay check cannot accept');
    CARRIED_OVERRIDE = undefined;
    // DECISIONS 4.56: the fp32 main_grad accumulation contract gates GEMM overlays, fail closed.
    MGC_OVERRIDE = { applicable: true, passed: false, why: 'linearity 3638 ulp32 > 906' };
    const { result: m1 } = await dryRun(base);
    ok(m1.accepted.length === 0, 'an applicable main_grad contract that did not pass blocks the accept');
    MGC_OVERRIDE = null;
    const { result: m2 } = await dryRun(base);
    ok(m2.accepted.length === 0, 'no main_grad_contract result -> no accept (fail closed)');
    MGC_OVERRIDE = { applicable: false, passed: null, why: 'not a fused-wgrad GEMM site' };
    const { result: m3 } = await dryRun(base);
    ok(m3.accepted.length > 0, 'a not-applicable contract (decided by the script) does not block');
    // audit 2026-10-25 G-1: "not applicable" counts only when the verifier's independent check agrees.
    MGCA_OVERRIDE = { head: null, applicable: true, evidence: ['target: _LayerNormLinear', 'impl code: main_grad'] };
    const { result: m4 } = await dryRun(base);
    ok(m4.accepted.length === 0, 'a composite site the Auditor called not applicable but which touches main_grad is refused');
    MGCA_OVERRIDE = null;
    const { result: m5 } = await dryRun(base);
    ok(m5.accepted.length === 0, 'not applicable with no independent applicability check -> no accept (fail closed)');
    MGCA_OVERRIDE = undefined;
    MGC_OVERRIDE = { applicable: true, passed: false, head: null, why: 'touches main_grad, no head models this site' };
    const { result: m6 } = await dryRun(base);
    ok(m6.accepted.length === 0, 'an applicable site with no head is not passed');
    MGC_OVERRIDE = undefined;
    // DECISIONS 4.60: the L0 oracle must score every output.
    ORACLE_OVERRIDE = { kind: 'template', template_version: 1, scores_every_output: true };
    const { result: o1 } = await dryRun(base);
    ok(o1.accepted.length === 0, 'L0 from a pre-v2 template oracle (missing-output hole) cannot accept');
    ORACLE_OVERRIDE = { kind: 'custom', template_version: null, scores_every_output: false };
    const { result: o2 } = await dryRun(base);
    ok(o2.accepted.length === 0, 'a custom oracle not confirmed to score every output cannot accept');
    ORACLE_OVERRIDE = { kind: 'custom', template_version: null, scores_every_output: true, task_dir: '/work/tasks/t' };
    const { result: o3 } = await dryRun(base);
    ok(o3.accepted.length > 0, 'a custom oracle confirmed to score every output is accepted');
    ORACLE_OVERRIDE = null;
    const { result: o4 } = await dryRun(base);
    ok(o4.accepted.length === 0, 'L0 without its oracle provenance cannot accept (fail closed)');
    ORACLE_OVERRIDE = undefined;
  } catch (e) {
    ok(false, `verifier cases threw: ${e && e.message}`);
  }
  VERIFY_OVERRIDE = undefined;

  // ---- 2026-10-16: the optional outcome gate ---------------------------------------------------
  try {
    const { result: r0, calls: c0 } = await dryRun(base);
    ok(!c0.phases.includes('QualityGate') && r0.quality_gate && r0.quality_gate.cleared === null,
      'quality gate is opt-in: off by default, and the result SAYS the stack is not cleared for long training');
    const { result: r1, calls: c1 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(c1.phases.includes('QualityGate') && r1.quality_gate.cleared === true,
      'quality_gate=true runs the outcome test after Validate and records its verdict');
    const qp = c1.prompts.filter((p) => p.includes('PHASE=quality'));
    ok(qp.length === 1 && qp[0].includes('quality_ab.sh') && qp[0].includes('quality_ab_final.py')
       && qp[0].includes('"QUALITY_WORKLOAD": "/x/workloads/qwen2_5_math_7b.env"'),
      'the quality phase gets the protocol script, the pre-registered verdict script and THIS run\'s workload');
    const { calls: cc } = await dryRun({ ...base, quality_gate: 'true', quality_ctrl: 'ctrl70' });
    ok(cc.prompts.filter((p) => p.includes('PHASE=quality'))[0].includes('"QUALITY_CTRL": "ctrl70"'),
      'quality_ctrl passes an in-run positive-control arm to the outcome test');
    // HF runs default to the pre-registered HF protocol (P0-4g): _q descriptor, hf_trl stack, ctrl70 control.
    const { calls: ch } = await dryRun({ ...base, stack: 'hf_trl', workload: '/x/workloads/qwen2_5_math_7b_hf_cosine_zero2_nm.env', quality_gate: 'true' });
    const hq = ch.prompts.filter((p) => p.includes('PHASE=quality'))[0] || '';
    ok(hq.includes('"QUALITY_WORKLOAD": "/x/workloads/qwen2_5_math_7b_hf_cosine_zero2_nm_q.env"') &&
       hq.includes('"QUALITY_CTRL": "ctrl70"') && hq.includes('"QUALITY_STACK": "hf_trl"'),
      'an HF QualityGate defaults to the _q descriptor, the hf_trl stack and the in-run ctrl70 control');
    const { calls: cq } = await dryRun({ ...base, quality_gate: 'true', quality_workload: '/x/workloads/sched.env' });
    ok(cq.prompts.filter((p) => p.includes('PHASE=quality'))[0].includes('"QUALITY_WORKLOAD": "/x/workloads/sched.env"'),
      'an explicit quality_workload overrides the default');
    STRAT_HEADS = [];
    const { result: r2, calls: c2 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(!c2.phases.includes('QualityGate') && /nothing accepted/.test(r2.quality_gate.quality),
      'with nothing accepted the stack is unchanged, so the gate is skipped and says why');
    // Deep run, no kernel task, one Class-A lever accepted: the stack DID change. Before 2026-10-21
    // the gate counted kernels only and skipped this as "nothing accepted".
    const { result: r3, calls: c3 } = await dryRun({ ...base, deep_mode: 'true', budget: 0, quality_gate: 'true' });
    const qp3 = c3.prompts.filter((p) => p.includes('PHASE=quality'));
    ok(r3.accepted.length === 0 && r3.accepted_config && c3.phases.includes('QualityGate'),
      'a run whose only accepted change is a config lever still runs the outcome test');
    ok(qp3.length === 1 && qp3[0].includes('"ACCEPTED_CONFIG"') && qp3[0].includes('--overlap-grad-reduce'),
      'the quality phase is told the accepted lever, so the cand arm carries its flags');
    // A packing lever alone: both protocol arms already train packed, so ref == cand.
    CONFIG_OVERRIDE = { lever: 'packing', flags_added: ['--train-data-path', '/work/data/x_packed.jsonl', 'packing'],
                        comparison_tokens: { delta_pct: 265.9 }, packing_gate: { pass: true, verdict: 'PASS' } };
    const { result: r4, calls: c4 } = await dryRun({ ...base, deep_mode: 'true', budget: 0, quality_gate: 'true' });
    CONFIG_OVERRIDE = null;
    ok(r4.accepted_config && !c4.phases.includes('QualityGate') && /packing/.test(r4.quality_gate.quality),
      'a packing-only change skips the outcome test and says why (pack_gate.py is its gate)');
  } catch (e) {
    ok(false, `quality gate threw: ${e && e.message}`);
  }
  STRAT_HEADS = null;
  CONFIG_OVERRIDE = null;

  // ---- audit 2026-10-25 fix 12 (ORCH-A5 / A2 / G-3 / A8): code-derived status, verifier coverage,
  // recomputed L1 with a code-derived plan, recomputed outcome-test verdict ------------------------
  try {
    const kv = (calls) => calls.prompts.filter((p) => p.includes('PHASE=verify') && p.includes('"KIND": "kernel"'));
    // A2: a reply that skips a requested pairing (here every gref2_ leg) cannot accept.
    PAIRS_DROP = (r) => r.includes('gref2_');
    const { result: c1 } = await dryRun(base);
    PAIRS_DROP = undefined;
    ok(c1.accepted.length === 0, 'a verifier reply that skips a requested pairing cannot accept (regression veto intact)');
    // G-3: the verifier's own L1 decides; a failure rejects even when the Auditor's l1 passed.
    L1V_OVERRIDE = () => ({ pass: false, failures: ['step-1 loss differs by 0.3100% (limit 0.0500%)'] });
    const { result: c2 } = await dryRun(base);
    L1V_OVERRIDE = undefined;
    ok(c2.accepted.length === 0, 'a failed recomputed L1 refuses although the Auditor reported l1.pass');
    L1V_OVERRIDE = () => undefined;
    const { result: c2b } = await dryRun(base);
    L1V_OVERRIDE = undefined;
    ok(c2b.accepted.length === 0, 'a verifier reply without its L1 recomputation cannot accept');
    // G-3: the L1 plan comes from the L0 evidence, not the Auditor's choice.
    const { calls: c3 } = await dryRun(base);
    ok(kv(c3).length > 0 && kv(c3).every((p) => p.includes('"L1_KIND": "kernel_swap"') && p.includes('"L1_L0_STRICTLY_BETTER": false')),
      'no L0 evidence -> kernel_swap, never the 8x looser strictly-better floor');
    L0R_OVERRIDE = { correct: true, out_bit_identical: true, per_output: null, oracle: { task_dir: '/work/tasks/t' } };
    const { calls: c4 } = await dryRun(base);
    ok(kv(c4).length > 0 && kv(c4).every((p) => p.includes('"L1_KIND": "overlay"')), 'forward bit-identical in L0 -> the tight overlay floor');
    L0R_OVERRIDE = { correct: true, oracle: { task_dir: '/work/tasks/t' },
                     per_output: { out: { cand_err: 1, base_err: 2 }, dgrad: { cand_err: 1, base_err: 2 } } };
    const { calls: c5 } = await dryRun(base);
    ok(kv(c5).length > 0 && kv(c5).every((p) => p.includes('"L1_L0_STRICTLY_BETTER": true')),
      'every output strictly closer to fp32 -> strictly-better floor');
    L0R_OVERRIDE = { correct: true, oracle: { task_dir: '/work/tasks/t' },
                     per_output: { out: { cand_err: 1, base_err: 2 }, wgrad: { cand_err: 2, base_err: 2 } } };
    const { calls: c6 } = await dryRun(base);
    ok(kv(c6).length > 0 && kv(c6).every((p) => p.includes('"L1_L0_STRICTLY_BETTER": false')),
      'one output merely equal -> NOT strictly better');
    L0R_OVERRIDE = undefined;
    // The Auditor's transcript no longer plans L1: claiming bit-identity there changes nothing.
    L0X_OVERRIDE = { out_bit_identical: true };
    const { calls: c7 } = await dryRun(base);
    L0X_OVERRIDE = undefined;
    ok(kv(c7).length > 0 && kv(c7).every((p) => p.includes('"L1_KIND": "kernel_swap"')),
      "the Auditor's own out_bit_identical does not loosen or tighten the plan (the verifier's evidence does)");
    // DECISIONS 4.99: the verifier's own L0 decides.
    L0R_OVERRIDE = { correct: false, error: null, oracle: { task_dir: '/work/tasks/t' } };
    const { result: l0a } = await dryRun(base);
    L0R_OVERRIDE = null;
    const { result: l0b } = await dryRun(base);
    L0R_OVERRIDE = { correct: true, oracle: { task_dir: '/work/tasks/another' } };
    const { result: l0c } = await dryRun(base);
    L0R_OVERRIDE = undefined;
    ok(l0a.accepted.length === 0, "the verifier's own L0 not correct -> refused although the Auditor passed L0");
    ok(l0b.accepted.length === 0, 'no independent L0 result -> refused (fail closed)');
    ok(l0c.accepted.length === 0, "an L0 run on another task's oracle -> refused");
    // A5: status is code-derived.
    const { result: s0, calls: sc0 } = await dryRun(base);
    ok(s0.status === 'ok' && sc0.prompts.some((p) => p.includes('"KIND": "stack"') && p.includes('val_final')),
      `a confirmed Validate recomputed as a win is ok (status=${s0.status})`);
    VERIFY_OVERRIDE = (r, c) => (c.includes('val_final') ? 'no_win' : 'win');
    const { result: s1 } = await dryRun(base);
    VERIFY_OVERRIDE = undefined;
    ok(s1.status === 'unconfirmed' && /recomputation/.test(s1.status_reason || ''),
      `Validate says confirmed but the recomputed headline is no_win -> unconfirmed (${s1.status})`);
    VALIDATE_OVERRIDE = { ref_dir: undefined, final_dir: undefined };
    const { result: s2 } = await dryRun(base);
    VALIDATE_OVERRIDE = undefined;
    ok(s2.status === 'unconfirmed', 'a Validate without its headline leg dirs cannot make the run ok');
    VERIFY_OVERRIDE = (r, c) => (c.includes('val_final') ? 'win' : 'no_win');
    const { result: s3 } = await dryRun(base);
    VERIFY_OVERRIDE = undefined;
    ok(s3.accepted.length === 0 && s3.status === 'no_gain', `nothing accepted -> no_gain whatever Validate says (${s3.status})`);
    // A8: the outcome-test verdict is recomputed; cleared derives from the recomputed strings.
    QUALV_OVERRIDE = () => ({ quality: 'INCONCLUSIVE-quality: upper 95% bound +0.009 > margin', stability: 'PASS-stability: 0 severe' });
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/work/runs/qA' },
                   { cleared: true, quality: 'PASS-quality: n=10', stability: 'PASS-stability: 0', legs_prefix: '/work/runs/qB' }];
    const { result: q1, calls: qc1 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(q1.quality_gate.cleared === false && q1.quality_gate.reported && q1.quality_gate.reported.cleared === true,
      'a Director PASS that recomputes as INCONCLUSIVE is NOT CLEARED (the report is kept)');
    const qv = qc1.prompts.filter((p) => p.includes('"KIND": "quality"'));
    ok(qv.length === 2 && qv[1].includes('"/work/runs/qA"') && qv[1].includes('"/work/runs/qB"'),
      'the extension is recomputed over BOTH prefixes (the combined sample)');
    QUALV_OVERRIDE = () => ({ quality: 'PASS-quality: upper ok', stability: 'PASS-stability: 0 severe' });
    QUALITY_SEQ = [{ cleared: false, quality: 'whatever', stability: 'x', legs_prefix: '/work/runs/qA' }];
    const { result: q2 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(q2.quality_gate.cleared === true, 'cleared is derived from the recomputed PASS/PASS, not the Director\'s flag');
    QUALV_OVERRIDE = () => ({ quality: 'INCONCLUSIVE-quality: seed 7 cand did not finish', stability: 'FAIL-stability: seed 7 cand DIVERGED' });
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/work/runs/qA' }];
    const { result: q3, calls: qc3 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(q3.quality_gate.cleared === false && qc3.prompts.filter((p) => p.includes('PHASE=quality')).length === 1,
      'a recomputed FAIL-stability (diverged cand) is NOT CLEARED and is not extended');
    QUALV_OVERRIDE = () => null;
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/work/runs/qA' }];
    const { result: q4 } = await dryRun({ ...base, quality_gate: 'true' });
    ok(q4.quality_gate.cleared === false, 'no recomputed verdict -> NOT CLEARED');
    QUALV_OVERRIDE = undefined;
    QUALITY_SEQ = [{ cleared: true, quality: 'PASS-quality: ok', stability: 'PASS-stability: 0', legs_prefix: '/work/runs/qA' }];
    const { result: q5 } = await dryRun({ ...base, quality_gate: 'true', stack: 'hf_trl', quality_ctrl: 'none' });
    ok(q5.quality_gate && q5.quality_gate.cleared === false, 'an HF outcome test without the in-run control is NOT CLEARED (G-5)');
  } catch (e) {
    ok(false, `fix-12 cases threw: ${e && e.stack}`);
  }
  PAIRS_DROP = undefined; L1V_OVERRIDE = undefined; L0X_OVERRIDE = undefined; VALIDATE_OVERRIDE = undefined;
  QUALV_OVERRIDE = undefined; QUALITY_SEQ = null; VERIFY_OVERRIDE = undefined;

  // ---- audit 2026-10-25 #11: every gate's reference is bound to the carried stack ----------------------
  try {
    let threw = '';
    SETUP_BARE = true;
    try { await dryRun({ ...base, baseline_overlay: '/x/stack_overlay_env.sh --cond s8' }); } catch (e) { threw = String(e && e.message); }
    SETUP_BARE = undefined;
    ok(/runs no overlay/.test(threw), 'baseline_overlay with a bare Setup baseline fails loudly (#11 P8)');
    REFBIND_OVERRIDE = { impl_digest: 'other_impl' };
    const { result: b1 } = await dryRun(base);
    ok(b1.accepted.length === 0, 'a kernel gate whose ref is not the carried implementation is refused');
    REFBIND_OVERRIDE = { config_digest: 'other_workload' };
    const { result: b2 } = await dryRun(base);
    REFBIND_OVERRIDE = undefined;
    ok(b2.accepted.length === 0, 'a kernel gate whose ref ran another workload is refused');
    CONFIG_OVERRIDE = { ref: leg({ config_digest: 'other_workload' }) };
    const { result: b3 } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
    CONFIG_OVERRIDE = null;
    const v3 = ((((b3 || {}).config_sweep || {}).levers || [])[0] || {}).verdict || {};
    ok(v3.accepted === false, `a config gate whose ref is not the run's reference is refused (${v3.klass})`);
    const { result: b4, calls: c4 } = await dryRun({ ...base, deep_mode: 'true', budget: 1, config_levers: 'recompute_granularity' });
    const integ4 = c4.prompts.find((p) => p.includes('PHASE=integrate')) || '';
    ok(c4.prompts.some((p) => p.includes('PHASE=reprofile') || (p.includes('PHASE=profile') && p.includes('after_config'))) ||
       c4.prompts.filter((p) => p.includes('PHASE=profile') || p.includes('PHASE=reprofile')).length >= 2,
      'an accepted config lever re-profiles before the kernel track');
    ok(/"REF_EXPECTED": \{[^]*"impl_digest": "after_recompute_granularity"/.test(integ4) && b4.accepted.length > 0,
      `the kernel gate after a config accept is bound to the lever's cand leg, and still accepts (${b4.accepted.length})`);
  } catch (e) {
    ok(false, `#11 binding cases threw: ${e && e.stack}`);
  }
  REFBIND_OVERRIDE = undefined; SETUP_BARE = undefined; CONFIG_OVERRIDE = null;

  // ---- low-severity audit items (2026-10-26) --------------------------------------------------------
  try {
    ORACLE_OVERRIDE = { kind: 'template', template_version: 2, scores_every_output: true, task_dir: '/work/tasks/other_head' };
    const { result: a7 } = await dryRun(base);
    ORACLE_OVERRIDE = undefined;
    ok(a7.accepted.length === 0, 'ORCH-A7: L0 from another task\'s oracle cannot accept');
    // MS-6: the reported pair +2.0%, the other measured pairing -1.5%
    VERIFY_OVERRIDE = (r) => (r.includes('gref2_') ? 'no_win' : 'win');
    VDELTA_OVERRIDE = (r, c) => (r.includes('gref2_') ? -1.5 : 2.0);   // median of {2.0, -1.5} = 0.25 <= band 0.5
    const { result: m6 } = await dryRun(base);
    VDELTA_OVERRIDE = undefined;
    const { result: m6b } = await dryRun(base);
    VERIFY_OVERRIDE = undefined;
    ok(m6.accepted.length === 0, 'MS-6: a reported win whose median over all measured pairings is in band is refused (best-of-k)');
    ok(m6b.accepted.length > 0, 'MS-6: ...and accepted when the pairings agree');
  } catch (e) {
    ok(false, `low-severity cases threw: ${e && e.stack}`);
  }
  ORACLE_OVERRIDE = undefined; VERIFY_OVERRIDE = undefined;

  // ---- DECISIONS 4.88: median vs trimmed-mean disagreement -> ONE re-gate on legs twice as long ----------
  try {
    const cmp = (tm) => ({ delta_pct: 2.0, window_artifact_pct: 0.1,
      paired: { delta_pct: 2.0, ratio_iqr_pct: 0.4, steps: 22, same_batches: true, tmean_delta_pct: tm } });
    INTEG_CMP_SEQ = [cmp(-0.2), cmp(1.9)];
    const { result: e1, calls: ce1 } = await dryRun({ ...base, budget: 1 });
    const ip = ce1.prompts.filter((p) => p.includes('PHASE=integrate'));
    ok(ip.length === 2 && ip[1].includes('"ITERS": 60') && ip[1].includes('REGATE_OF'),
      `disagreeing estimators trigger ONE re-gate at 60 iterations (${ip.length} integrate calls)`);
    ok(e1.accepted.length === 1, 'the longer pair agrees -> accepted on the longer pair');
    INTEG_CMP_SEQ = [cmp(-0.2), cmp(-0.3)];
    const { result: e2, calls: ce2 } = await dryRun({ ...base, budget: 1 });
    ok(e2.accepted.length === 0 && ce2.prompts.filter((p) => p.includes('PHASE=integrate')).length === 2,
      'still disagreeing on the longer pair -> refused, and no third gate');
    INTEG_CMP_SEQ = [cmp(-1.4), cmp(1.8)];
    const { result: e3, calls: ce3 } = await dryRun({ ...base, deep_mode: 'true', budget: 0 });
    const cp3 = ce3.prompts.filter((p) => p.includes('PHASE=config'));
    const lv3 = ((((e3 || {}).config_sweep || {}).levers || [])[0] || {}).verdict || {};
    ok(cp3.length === 2 && cp3[1].includes('"ITERS": 60') && cp3[1].includes('REMEASURE_OF') && lv3.accepted === true,
      `a config lever whose estimators disagree is re-measured once on longer legs, then decided (${lv3.klass})`);
  } catch (e) {
    ok(false, `re-gate cases threw: ${e && e.stack}`);
  }
  INTEG_CMP_SEQ = null;

  // An opts.phase that is not a meta.phases title opens an ad-hoc group in /workflows and leaves the
  // declared phase reading "not started" though its agent ran (run wf_becee9a0-c6f, 2026-10-20).
  const metaTitles = new Set([...fs.readFileSync(WF, 'utf8').slice(0, metaEnd)
    .matchAll(/title: '([^']+)'/g)].map((m) => m[1]));
  const strays = [...AGENT_GROUPS].filter((g) => !metaTitles.has(g));
  ok(strays.length === 0,
    `every agent, in every mode, groups under a meta.phases title (strays: ${strays.join(', ') || 'none'})`);
  // Gate legs must keep bench_summary's `implementation` dict, or comparable()'s check that ref and
  // cand differ only in what is under test is skipped silently (wf_becee9a0-c6f's Integrator dropped it).
  const wfSrc = fs.readFileSync(WF, 'utf8');
  ok(/const LEG = obj\(\{ implementation:/.test(wfSrc) && /\['implementation', 'workload_identity'/.test(wfSrc) &&
     /INTEGRATE_SCHEMA = obj\(\{ ref: LEG, cand: LEG/.test(wfSrc) && /ref: LEG, cand: LEG, l1: VERDICT/.test(wfSrc),
    'integrate and config schemas REQUIRE an implementation dict on both gate legs');
  ok(AGENT_GROUPS.has('ConfigSweep') && AGENT_GROUPS.has('QualityGate'),
    'the optional phases were exercised by this check, not just the fast path');

  console.log();
  if (failures) { console.error(`${failures} check(s) FAILED`); process.exit(1); }
  console.log('all checks passed');
})();
