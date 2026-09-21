#!/usr/bin/env node
// Regression guard for training_workflow.js's DETERMINISTIC half (no GPU, no agent, no model).
//
// The functions are EXTRACTED FROM THE SHIPPED SOURCE, not reimplemented here: a reimplementation
// would keep passing while the workflow drifted, which is the failure mode this file exists to stop.
//
// What is pinned, and why each one is a real hazard:
//   1. The noise band is DERIVED from the legs' own robust dispersion. Measured on this box, the same
//      25-step run reports 38.6% raw spread and 0.072% IQR spread; a band built on raw spread gates
//      nothing, and a hard-coded 2-3% band (the inference layer's) throws away every real sub-1% win.
//   2. An unusable leg can never become a reference. Contaminated / hung / status!=ok legs are the
//      silent-bias path on a SHARED box (this node carries ~20 unrelated containers).
//   3. Memory is a CURRENCY, not a metric: in fast mode there is no config knob to spend it with, so
//      a memory-only result is BANKED, never accepted. Accepting it would inflate the headline.
//   4. L0/L1 can only ever REJECT, never accept.
//   5. The lane/gate barrier gives a gate the WHOLE box: a training A/B has no spare GPUs, unlike the
//      inference layer where lanes and the serving slot coexist.
//
// Run:  node e2e_training_workflow/scripts/tests/test_training_gate.js
'use strict';
const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..', '..', 'training_workflow.js');
const src = fs.readFileSync(WF, 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

/** Slice a top-level `function NAME(...) { ... }` out of the source by brace matching. */
function extract(name) {
  const start = src.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`function ${name} not found in training_workflow.js`);
  // The body brace is the first '{' AFTER the parameter list closes. Starting at the first '{'
  // would latch onto a destructured parameter list -- e.g. gateVerdict({ ref, cand, l0, l1, mode }).
  let p = src.indexOf('(', start), pd = 0, i = -1;
  for (let j = p; j < src.length; j++) {
    if (src[j] === '(') pd++;
    else if (src[j] === ')') { pd--; if (pd === 0) { i = src.indexOf('{', j); break; } }
  }
  if (i < 0) throw new Error(`could not find body of ${name}`);
  let depth = 0;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error(`unterminated function ${name}`);
}

const CONSTS = 'const ACROSS_JOB_FLOOR_PCT = 0.3, BAND_SAFETY = 2.0;';
const load = (names, extra = '') => new Function(
  `${CONSTS}\n${extra}\n${names.map(extract).join('\n')}\nreturn {${names.join(',')}};`)();

const leg = (o = {}) => Object.assign({
  status: 'ok', hang_detected: false, occupancy_verdict: { verdict: 'clean' },
  useful_tok_s_per_gpu: 4760, worst_iqr_spread_pct: 0.13, peak_reserved_mb: 14394,
  config_digest: 'abc123def456', ranks_agree_on_argv: true, dispersion_reliable: true,
  min_steady_steps: 20,
}, o);

// ---------------------------------------------------------------- 1. noise band
{
  const { noiseBandPct } = load(['noiseBandPct']);
  ok(noiseBandPct(leg(), leg()) === 0.3, 'tight legs fall back to the measured across-job floor (0.3%)');
  ok(noiseBandPct(leg({ worst_iqr_spread_pct: 1.2 }), leg()) === 2.4,
    'a noisy leg widens the band via its own IQR (1.2% -> 2.4%), rather than a constant');
  // The hazard this encodes: raw spread is NOT the input.
  ok(!/worst_spread_pct/.test(extract('noiseBandPct')),
    'the band is built on iqr spread only — raw spread (38.6% on a clean run) would gate nothing');
}

// ---------------------------------------------------------------- 2. leg usability
{
  const { legUsable } = load(['legUsable']);
  ok(legUsable(leg()).ok, 'a clean, completed leg is usable');
  ok(!legUsable(leg({ occupancy_verdict: { verdict: 'contaminated' } })).ok,
    'a leg with a foreign GPU process is NOT a measurement');
  ok(!legUsable(leg({ occupancy_verdict: { verdict: 'unknown' } })).ok,
    'unprovable isolation is treated as unusable, not as clean');
  ok(!legUsable(leg({ hang_detected: true })).ok, 'a hung leg is unusable');
  const blocked = legUsable(leg({ status: 'unusable', contamination_exhausted: true, attempts: 3 }));
  ok(!blocked.ok && blocked.fatal === true,
    'a leg contaminated on every retry is FATAL, not merely unusable (the node is busy; more gates ' +
    'cannot fix it)');
  ok(/all 3 attempts/.test(blocked.why), 'the fatal reason names how many attempts were made');
  ok(legUsable(leg({ status: 'unusable' })).fatal !== true,
    'an ordinary unusable leg stays NON-fatal — only exhausted contamination stops the run');
  // Order guard: a blocked leg also has status != ok, so the generic check must not shadow it.
  ok(/ORDER MATTERS/.test(extract('legUsable')),
    'the contamination check runs BEFORE the generic status check (it would otherwise be shadowed)');
  ok(!legUsable(leg({ status: 'unusable' })).ok, 'status!=ok is unusable');
  ok(!legUsable(leg({ useful_tok_s_per_gpu: 0 })).ok, 'no useful-token throughput is unusable');
  const short = legUsable(leg({ dispersion_reliable: false, min_steady_steps: 7 }));
  ok(!short.ok && /too few to estimate dispersion/.test(short.why),
    'a leg too short to estimate dispersion is refused (7 steady steps measured 1.243% IQR where ' +
    '24 measured 0.072% — small-n can err tight, which accepts changes that are not there)');
  ok(legUsable(leg({ dispersion_reliable: true })).ok, 'a reliable leg passes');
}

// ---------------------------------------------------------------- 3./4. the gate
{
  const { gateVerdict } = load(['gateVerdict', 'legUsable', 'noiseBandPct', 'comparable', 'alignedDelta']);
  const v = (o) => gateVerdict(Object.assign({ ref: leg(), cand: leg(), mode: 'fast' }, o));

  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4800 }) }).accepted,
    '+0.84% clears the 0.3% band and is accepted');
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4766 }) }).klass === 'no_win',
    '+0.13% is inside the band -> no_win (not a win)');
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4700 }) }).klass === 'regression',
    '-1.26% is a regression');
  ok(!v({ l0: { pass: false, reason: 'dgrad mismatch' } }).accepted,
    'L0 (op numerics incl. backward) can reject a throughput win');
  ok(!v({ cand: leg({ useful_tok_s_per_gpu: 4800 }), l1: { pass: false } }).accepted,
    'L1 (loss/grad-norm trajectory) can reject a throughput win');
  ok(v({ l0: { pass: true }, l1: { pass: true }, cand: leg({ useful_tok_s_per_gpu: 4800 }) }).accepted,
    'passing gates do not by themselves accept — throughput still has to clear the band');

  // memory is banked, never accepted, while no ConfigSweep exists to spend it
  const m = v({ cand: leg({ useful_tok_s_per_gpu: 4762, peak_reserved_mb: 12000 }) });
  ok(m.klass === 'banked_memory' && m.accepted === false,
    'throughput-neutral + 16% memory freed -> banked, NOT accepted (fast mode cannot convert it)');
  ok(m.banked_memory_pct > 16 && m.banked_memory_pct < 17, `banked_memory_pct recorded (${m.banked_memory_pct}%)`);
  ok(/fast mode has no config knob/.test(m.reason), 'the banked reason names why it is not a win');
  const md = gateVerdict({ ref: leg(), cand: leg({ useful_tok_s_per_gpu: 4762, peak_reserved_mb: 12000 }), mode: 'deep' });
  ok(/proposeMemoryConversion/.test(md.reason), 'in deep mode the same result routes to the converter');

  // The aligned delta must WIN over the raw ratio when both are present.
  const al = gateVerdict({ ref: leg(), cand: leg({ useful_tok_s_per_gpu: 4776 }), mode: 'fast',
                           comparison: { delta_pct: 0.02, window_artifact_pct: 0.31 } });
  ok(al.klass === 'no_win' && al.delta_source === 'window-aligned',
    'a +0.34% RAW ratio with a +0.02% ALIGNED delta is a no_win: 93% of it was window placement');
  const raw = gateVerdict({ ref: leg(), cand: leg({ useful_tok_s_per_gpu: 4776 }), mode: 'fast' });
  ok(/NOT window-aligned/.test(raw.delta_source),
    'without an aligned comparison the verdict says the number carries the artifact');

  const inc = v({ cand: leg({ useful_tok_s_per_gpu: 9999, config_digest: 'different99' }) });
  ok(inc.klass === 'incomparable' && !inc.accepted,
    'a +110% delta measured at a DIFFERENT config digest is refused as incomparable, not accepted');
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4800, ranks_agree_on_argv: false }) }).klass === 'incomparable',
    'ranks running different argv inside one leg make it incomparable');
  const fatal = v({ ref: leg({ status: 'unusable', contamination_exhausted: true, attempts: 3 }) });
  ok(fatal.klass === 'blocked' && fatal.fatal === true,
    'exhausted contamination surfaces as klass=blocked + fatal, so the loop can stop the run');
  ok(v({ ref: leg({ occupancy_verdict: { verdict: 'contaminated' } }),
         cand: leg({ useful_tok_s_per_gpu: 9999 }) }).klass === 'unusable_ref',
    'a huge delta against a contaminated reference is refused, not celebrated');
}

// ---------------------------------------------------------------- 5. critical-path routing + stop
{
  const { expectedGainPct, stopRule } = load(['expectedGainPct', 'stopRule']);
  ok(expectedGainPct({ critical_path_share_pct: 30, attainable_speedup: 1.0 }) === 0,
    'a head at its ceiling yields 0 expected gain even at 30% of the critical path');
  ok(Math.abs(expectedGainPct({ critical_path_share_pct: 30, attainable_speedup: 2 }) - 15) < 1e-9,
    '30% share x 2x attainable -> 15% expected gain');
  const s1 = stopRule([{ critical_path_share_pct: 5, attainable_speedup: 1.05 }], 0.3);
  ok(s1.stop === true && Math.abs(s1.best_expected_gain_pct - 0.238) < 0.01,
    `stop when the best remaining candidate (${s1.best_expected_gain_pct}%) cannot clear the band (0.3%)`);
  ok(stopRule([{ critical_path_share_pct: 40, attainable_speedup: 1.5 }], 0.3).stop === false,
    'keep going while a candidate can plausibly clear the band');
}

// ---------------------------------------------------------------- 6. the barrier
{
  const { makeBarrier } = load(['makeBarrier']);
  (async () => {
    const b = makeBarrier(8);
    const order = [];
    let releaseLane;
    const lane = b.lane(async () => {
      order.push('lane_start');
      await new Promise((r) => { releaseLane = r; });
      order.push('lane_end');
    });
    await new Promise((r) => setImmediate(r));
    const gate = b.gate(async () => { order.push('gate_run'); });
    await new Promise((r) => setImmediate(r));
    ok(!order.includes('gate_run'), 'a gate does NOT start while a lane still holds GPUs');
    releaseLane();
    await lane; await gate;
    ok(order.join(',') === 'lane_start,lane_end,gate_run',
      'the gate runs only after the box is drained (exclusive whole-box measurement)');

    const b2 = makeBarrier(8);
    let gateDone = false;
    const g2 = b2.gate(async () => { await new Promise((r) => setTimeout(r, 20)); gateDone = true; });
    const l2 = b2.lane(async () => {
      ok(gateDone, 'a lane admitted after a gate starts waits until the gate has finished');
    });
    await g2; await l2;

    // ---- structural guards that no arithmetic can cover ----
    ok(/FAST_SKIP = FAST_MODE \? new Set\(\['config'\]\)/.test(src),
      'fast mode skips ConfigSweep (and only ConfigSweep)');
    ok(/function requireRole/.test(src) && /refuses to run a phase whose agent is/.test(src),
      'a phase with no role FAILS LOUDLY instead of returning an empty result');
    ok(/MAX_GATES/.test(src) && /time_budget_ms/.test(src),
      'the run is bounded by BOTH a gate count and a wall clock');
    ok(/curConfig/.test(src) && /function benchInputs\(config, tag\)/.test(src),
      'ConfigSweep seam hooks 1-2 present: carried config IR, passed explicitly to every measurement');
    ok(/async function reprofile\(/.test(src) && /async function restrategize\(/.test(src),
      'seam hook 3 present: reprofile/restrategize are independently callable');
    ok(/function proposeMemoryConversion/.test(src) && /function kbIdentity/.test(src),
      'seam hooks 5-6 present: memory converter + config-hashed KB identity');

    // ---- pipeline invariants (structural; the loop itself needs the agent runtime) ----
    ok(/refLeg = integ\.cand;/.test(src),
      'an accepted candidate becomes the REFERENCE for the next gate (deltas compound on the ' +
      'carried stack, not on the bare baseline)');
    ok(/An accepted change moves which kernels dominate/.test(src) &&
       /profileAndRoute\(curConfig, `after_/.test(src),
      're-profile after every accept: the bottleneck moves, so the queue must be rebuilt');
    ok(/engagement\.forward === false \|\| integ\.engagement\.backward === false/.test(src),
      'engagement must be proven for forward AND backward separately, or the result is refused');
    ok(/has_backward === false/.test(src),
      'a task with no backward is refused: a forward-only task can call a slower kernel a win');
    ok(/ConfigSweep is not implemented yet/.test(src) && /fast-mode result under a/.test(src),
      'deep mode REFUSES rather than silently running the fast pipeline under a deep label');
    ok(/noImprove >= 2/.test(src), 'two consecutive non-wins stop the kernel track');
    ok(/if \(v\.fatal\)/.test(src) && /STOPPING: the box is not quiet enough/.test(src),
      'a fatal verdict STOPS the run rather than spending the rest of the gate budget on a busy node');
    ok(/gatesUsed < MAX_GATES/.test(src) && /tasksUsed < BUDGET/.test(src) && /outOfTime\(\)/.test(src),
      'the loop is bounded by gates, tasks AND wall clock');
    ok(/Setup produced no baseline/.test(src),
      'a missing baseline aborts: deltas against nothing are worse than no result');
    ok(/runFloor = Math\.max\(ACROSS_JOB_FLOOR_PCT/.test(src),
      "the run's MEASURED band replaces the default when it is wider (a noisy box must not inherit " +
      'a flattering band)');

    console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
    process.exit(failures ? 1 : 0);
  })();
}
