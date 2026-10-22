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

// Mirrors the shipped constants. The 7B-class floor (0.15) is used here rather than the 0.5 default
// so the band tests below still exercise the DERIVED terms; with the default floor the floor binds
// on every case and the se term would never be tested. Keep in step with training_workflow.js.
const CONSTS = 'const FLOOR_PCT = {aligned:0.15,unaligned:0.5}; const ACROSS_JOB_FLOOR_PCT = 0.15, '
  + 'BAND_SAFETY = 2.0, SE_SAFETY = 4.0; let runAcrossJobPct = 0;';
const load = (names, extra = '') => new Function(
  `${CONSTS}\n${extra}\n${names.map(extract).join('\n')}\nreturn {${names.join(',')}};`)();

const leg = (o = {}) => Object.assign({
  status: 'ok', hang_detected: false, occupancy_verdict: { verdict: 'clean' },
  useful_tok_s_per_gpu: 4760, worst_iqr_spread_pct: 0.13, peak_reserved_mb: 14394,
  config_digest: 'abc123def456', ranks_agree_on_argv: true, dispersion_reliable: true,
  digest_schema: 2,
  min_steady_steps: 20, step_shape: 'fixed',
}, o);

// ---------------------------------------------------------------- 1. noise band
{
  const { noiseBandPct } = load(['noiseBandPct']);
  const near = (a, b) => Math.abs(a - b) < 5e-4;
  // The dispersion term is the SAMPLING ERROR OF A MEDIAN over n steady steps, not the dispersion of
  // one step: se = 0.929 * iqr / sqrt(n), combined in quadrature, times SE_SAFETY. The old term
  // (iqr x 2) omitted the 1/sqrt(n) and so priced a single step's spread as the uncertainty of the
  // median of 20 of them. Measured consequence over 47 same-identity baseline pairs: it was 11x too
  // LOOSE at 7B (worst |delta| 0.0823% against a 0.914% band) while allowing 6/37 FALSE WINS at
  // 0.5B -- the widest band went to the most reproducible workload.
  ok(near(noiseBandPct(leg(), leg()), 0.1528),
    'two legs at iqr 0.13% over 20 steady steps -> 4 x se_pair = 0.153%, just over the 0.15% floor ' +
    '(the old formula charged 0.26% for the same legs)');
  const tight = leg({ worst_iqr_spread_pct: 0.05 });
  ok(noiseBandPct(tight, tight, true) === 0.15,
    'the ALIGNED floor still binds for quiet legs — here the 7B-class 0.15%, ~1.8x the worst aligned ' +
    'reproducibility measured over 10 clean 7B pairs (0.0823%)');
  ok(noiseBandPct(tight, tight, false) === 0.5,
    'the UNALIGNED floor is 0.5%. It no longer guards against a manufactured win — a raw ratio ' +
    'cannot be accepted at all now — it only governs the REGRESSION label');
  ok(near(noiseBandPct(leg({ worst_iqr_spread_pct: 1.2 }), leg()), 1.0031),
    'a noisy leg still widens the band via its own dispersion (1.2% iqr over 20 steps -> 1.003%), ' +
    'but at the correct scale: the old formula charged 2.4% for it');
  // n MUST divide the dispersion: a longer window is the cheapest way to tighten the band. Measured
  // at 0.5B: pairs with both legs >=29 steady steps have worst |delta| 0.2544%, against 0.4966% when
  // either leg had 19 -- ~10 extra iterations nearly halves the noise floor.
  const long = leg({ worst_iqr_spread_pct: 1.2, min_steady_steps: 80 });
  ok(noiseBandPct(long, long) < noiseBandPct(leg({ worst_iqr_spread_pct: 1.2 }), leg()),
    'the same dispersion over 4x the steps yields a TIGHTER band — steps buy precision');
  // A leg summarised by an older probe carries no step count; it must not get a 5x-tighter band for
  // the omission.
  const noN = leg({ worst_iqr_spread_pct: 1.2, min_steady_steps: undefined });
  ok(noiseBandPct(noN, noN) > noiseBandPct(leg({ worst_iqr_spread_pct: 1.2 }), leg()),
    'a leg with no step count falls back to the undivided term rather than being flattered by it');
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
  const { gateVerdict, legUsable, comparable } = load(['gateVerdict', 'legUsable', 'noiseBandPct', 'windowMatchedLegs', 'comparable', 'alignedDelta']);
  const v = (o) => gateVerdict(Object.assign({ ref: leg(), cand: leg(), mode: 'fast' }, o));

  // An ACCEPT now requires an aligned comparison, so every accept-asserting case must supply one.
  // That is the point of the fix, not an inconvenience: before it, these tests passed `comparison`
  // straight into gateVerdict() while the single production call site omitted it, so the suite
  // exercised the aligned path that production never took.
  const cmp = (d) => ({ comparison: { delta_pct: d, window_artifact_pct: 0 } });
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4800 }), ...cmp(0.84) }).accepted,
    '+0.84% aligned clears the band and is accepted');
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4766 }) }).klass === 'no_win',
    '+0.13% is inside the band -> no_win (not a win)');
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4700 }) }).klass === 'regression',
    '-1.26% is a regression');
  // THE BAND MUST COME FROM THE WINDOW THE DELTA CAME FROM. Real numbers, cfgrecompute_ref vs
  // cfgattn_flash, 2026-10-09: compare_legs forces a common window of 24 steps where the reference
  // has an IQR of 2.607%, while that leg's own bench_summary reports 0.096% over 25 steps. One step
  // of difference, 27x the dispersion, because 19 values sit inside 11642-11660 and then there is a
  // cliff to six slow steps, so v[3n/4] lands on the last tight value at n=25 and the first outlier
  // at n=24.
  const wmLegs = { ref: leg({ worst_iqr_spread_pct: 0.096, min_steady_steps: 25 }),
                   cand: leg({ worst_iqr_spread_pct: 0.07, min_steady_steps: 24,
                               useful_tok_s_per_gpu: 4727 }) };   // -0.73%, as measured
  const wmCmp = { window: { steps_compared: 24 },
                  ref: { iqr_spread_pct: 2.607 }, cand: { iqr_spread_pct: 0.07 },
                  delta_pct: -0.728, window_artifact_pct: 0.5243 };
  const vWindowed = v({ ...wmLegs, comparison: wmCmp });
  const vPerLeg = v({ ...wmLegs, comparison: { delta_pct: -0.728, window_artifact_pct: 0.5243 } });
  ok(vWindowed.band_pct > 1.9 && vWindowed.band_pct < 2.1,
    `the band is estimated on the COMMON window (${vWindowed.band_pct}%, expected ~1.98)`);
  ok(vPerLeg.band_pct < 0.6,
    `without window dispersion it falls back to the per-leg summaries (${vPerLeg.band_pct}%)`);
  ok(vWindowed.klass === 'inconclusive_noise' && vPerLeg.klass === 'regression',
    'and the verdict genuinely turns on it: inconclusive_noise on the matched band, regression on the narrow one');
  ok(/common/.test(vWindowed.band_window) && /per-leg/.test(vPerLeg.band_window),
    'each verdict says which sample its band came from');
  // The direction that matters. A false REJECT costs one candidate; a band that is too tight
  // ACCEPTS a win that is not there, which is what legUsable already warns about.
  ok(v({ ...wmLegs, cand: leg({ worst_iqr_spread_pct: 0.07, min_steady_steps: 24,
                                useful_tok_s_per_gpu: 4803 }),
         comparison: { ...wmCmp, delta_pct: 0.728 } }).accepted === false,
    'a +0.73% non-win is NOT ACCEPTED once the band is window-matched (the property that matters: ' +
    'a false reject costs one candidate, a band too tight ships a win that is not there)');

  // THE PAIRED ESTIMATOR. Real numbers, two IDENTICAL HF TRL legs, 2026-10-09: per-leg medians
  // read +1.1523% (a false win at any floor); the median of per-step ratios reads +0.1344%, ratio
  // IQR 0.2634% over 25 steps, while each leg's own step-time IQR is ~25%.
  const hfLegs = { ref: leg({ worst_iqr_spread_pct: 24.81, min_steady_steps: 25, step_shape: 'variable' }),
                   cand: leg({ worst_iqr_spread_pct: 25.36, min_steady_steps: 25, step_shape: 'variable' }) };
  const hfCmp = { window: { steps_compared: 25 }, ref: { iqr_spread_pct: 24.81 },
                  cand: { iqr_spread_pct: 25.36 }, delta_pct: 1.1523, window_artifact_pct: 0,
                  paired: { delta_pct: 0.1344, ratio_iqr_pct: 0.2634, steps: 25, same_batches: true } };
  const vHF = v({ ...hfLegs, comparison: hfCmp });
  ok(vHF.delta_pct === 0.134 && /paired/.test(vHF.estimator),
    `a valid paired comparison is what the gate reads (delta ${vHF.delta_pct}%, ${vHF.estimator})`);
  ok(!vHF.accepted, 'two identical HF runs are NOT a win once the paired estimator is used');
  ok(/paired/.test(vHF.band_window) && vHF.band_pct < 1,
    `and its band comes from the ratio series, not the ~25% content-driven per-leg IQR (${vHF.band_pct}%)`);
  const vBad = v({ ...hfLegs, comparison: { ...hfCmp, paired: { ...hfCmp.paired, same_batches: false } } });
  ok(vBad.delta_pct === 1.152 && /per-leg/.test(vBad.estimator),
    'if step i is NOT the same batch on both legs, the paired number is ignored, not trusted');
  ok(!vBad.accepted, `...and is not accepted on the per-leg number (${vBad.klass})`);
  const vBig = v({ ...hfLegs, comparison: { ...hfCmp, delta_pct: 40,
                                           paired: { ...hfCmp.paired, same_batches: false } } });
  ok(vBig.klass === 'no_aligned_delta' && !vBig.accepted,
    `even a +40% per-leg delta cannot ACCEPT a variable-shape pair without a valid paired comparison (${vBig.klass})`);
  ok(!legUsable(leg({ identity_vs_realized: { checked: 28, mismatches: [
       { key: 'lr', captured: '1.0e-5', realized: '0.0' }] } })).ok,
    'a leg whose captured identity disagrees with what ran is NOT usable (the real one-sample leg)');
  ok(legUsable(leg({ identity_vs_realized: null })).ok,
    'a stack with no realized table is "not checked", not refused');
  ok(legUsable(hfLegs.cand).ok,
    'a clean variable-shape leg at 25.36% per-leg IQR is USABLE (that spread is the dataset)');
  ok(!legUsable(leg({ worst_iqr_spread_pct: 25.36 })).ok,
    'while a FIXED-shape leg at the same IQR is still refused, exactly as before');
  // Accepted wins survive. tdop0928_ref2 vs _cand2: paired +1.3010%, ratio IQR 0.563% over ~20 steps.
  const vWin = v({ ref: leg(), cand: leg({ useful_tok_s_per_gpu: 4820 }),
                   comparison: { window: { steps_compared: 19 }, ref: { iqr_spread_pct: 2.9 },
                                 cand: { iqr_spread_pct: 0.1 }, delta_pct: 1.2857, window_artifact_pct: 0,
                                 paired: { delta_pct: 12.809, ratio_iqr_pct: 1.068, steps: 19, same_batches: true } } });
  ok(vWin.accepted, `a real +12.8% stack still clears the paired band (${vWin.band_pct}%)`);

  ok(!v({ l0: { pass: false, reason: 'dgrad mismatch' } }).accepted,
    'L0 (op numerics incl. backward) can reject a throughput win');
  ok(!v({ cand: leg({ useful_tok_s_per_gpu: 4800 }), l1: { pass: false } }).accepted,
    'L1 (loss/grad-norm trajectory) can reject a throughput win');
  ok(v({ l0: { pass: true }, l1: { pass: true }, cand: leg({ useful_tok_s_per_gpu: 4800 }), ...cmp(0.84) }).accepted,
    'passing gates do not by themselves accept — throughput still has to clear the band');

  // A raw ratio may REJECT but may never ACCEPT. Two identical-config 7B legs (true delta zero by
  // construction) measured +4.5925% unaligned against -0.0588% aligned at a 0.914% band, so an
  // unaligned number above the band is not evidence of anything.
  const noAl = v({ cand: leg({ useful_tok_s_per_gpu: 4800 }) });
  ok(noAl.accepted === false && noAl.klass === 'no_aligned_delta',
    '+0.84% with NO aligned comparison is refused as no_aligned_delta, not accepted');
  ok(/compare_legs\.py/.test(noAl.reason),
    'the refusal names the tool that would make it gateable');
  ok(v({ cand: leg({ useful_tok_s_per_gpu: 4700 }) }).klass === 'regression',
    'a raw ratio may still REJECT: a false reject costs one candidate, a false accept poisons the reference');
  ok(!v({ l0: { pass: false }, cand: leg({ useful_tok_s_per_gpu: 4800 }) }).klass.includes('aligned'),
    'L0/L1 keep first claim on the rejection reason — their klass is more actionable');

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

  const stale = gateVerdict({ ref: Object.assign(leg(), { digest_schema: 1 }),
                             cand: leg({ digest_schema: 2 }), mode: 'fast' });
  ok(stale.klass === 'stale_summary' && /Re-summarise/.test(stale.reason),
    'a summary written under an older digest definition is "stale", not "different workload" — ' +
    'the probe sidecars are on disk, so re-summarising answers it');

  // ...EXCEPT across schema 3. It widened workload identity to dropout/lr/seed/architecture, and a
  // v2 sidecar never recorded those, so the usual "re-summarise" remedy is impossible here. Telling
  // the reader to re-summarise would send them after a fix that cannot work.
  const v3 = gateVerdict({ ref: leg({ digest_schema: 2 }), cand: leg({ digest_schema: 3 }), mode: 'fast' });
  ok(v3.klass === 'stale_summary' && /RE-MEASURED/.test(v3.reason) && !/Re-summarise both/.test(v3.reason),
    'a schema-2 vs schema-3 pair says RE-MEASURE, not re-summarise: the v2 sidecar never recorded ' +
    'the fields schema 3 digests, so there is nothing on disk to recover them from');

  // Schema 5 is a second re-measure boundary: it captures --lr-decay-iters and the DDP reduction
  // flags (--ddp-average-in-collective was captured nowhere; a _cosine and a _cosine_avg leg shared
  // both digests). The probe header keeps argv_hash, not argv, so a v4 leg cannot be re-captured.
  const v5 = gateVerdict({ ref: leg({ digest_schema: 4 }), cand: leg({ digest_schema: 5 }), mode: 'fast' });
  ok(v5.klass === 'stale_summary' && /RE-MEASURED/.test(v5.reason) && /lr-decay-iters/.test(v5.reason),
    'a schema-4 vs schema-5 pair says RE-MEASURE and names what v4 never recorded');
  // v4 is a capture boundary too (it moved impl_digest on every Megatron leg, DECISIONS 4.10b), and
  // digest_schema is read from the sidecar, so re-summarising a v3 leg gives 3 again: RE-MEASURE.
  const v34 = gateVerdict({ ref: leg({ digest_schema: 3 }), cand: leg({ digest_schema: 4 }), mode: 'fast' });
  ok(v34.klass === 'stale_summary' && /RE-MEASURED/.test(v34.reason),
    'a schema-3 vs schema-4 pair says RE-MEASURE: re-summarising a v3 sidecar yields schema 3 again');

  // The reference must differ from the candidate ONLY in what is under test. Real implementation
  // dicts from wf_becee9a0-c6f's accepted gate (ref bare, cand = + the 1020 attention overlay).
  const IMPL = { 'attention-backend': 'fused', NVTE_FLASH_ATTN: '0', NVTE_FUSED_ATTN: '1',
                 NVTE_USE_HIPBLASLT: '1', 'use-distributed-optimizer': 'true' };
  const KERNEL = ['GEAK_OVERLAY', 'GEAK_OVERLAY_SHA'];
  const okK = gateVerdict({ ref: leg({ implementation: IMPL }), mode: 'fast', allowedImpl: KERNEL,
    cand: leg({ useful_tok_s_per_gpu: 4900, implementation: Object.assign({}, IMPL,
      { GEAK_OVERLAY: 'manifest_cand.json', GEAK_OVERLAY_SHA: '541b5ea21d92bd59' }) }) });
  ok(okK.klass !== 'incomparable', `a kernel gate whose legs differ only by the overlay passes the implementation check (${okK.klass})`);
  // _cosine reference (no average-in-collective) against a _cosine_avg candidate: same config_digest
  // since the flag is implementation, but the delta would include the lever's ~0.9%.
  const bad = gateVerdict({ ref: leg({ implementation: IMPL }), mode: 'fast', allowedImpl: KERNEL,
    cand: leg({ useful_tok_s_per_gpu: 4900, implementation: Object.assign({}, IMPL,
      { 'ddp-average-in-collective': 'true', GEAK_OVERLAY: 'm.json', GEAK_OVERLAY_SHA: '541b' }) }) });
  ok(bad.klass === 'incomparable' && /ddp-average-in-collective/.test(bad.reason),
    'a reference lacking a lever the candidate carries is refused, naming the key (was a silent ~0.9%)');
  const lever = gateVerdict({ ref: leg({ implementation: IMPL }), mode: 'deep',
    allowedImpl: ['ddp-average-in-collective'],
    cand: leg({ useful_tok_s_per_gpu: 4900, implementation: Object.assign({}, IMPL, { 'ddp-average-in-collective': 'true' }) }) });
  ok(lever.klass !== 'incomparable', 'a config lever may move exactly its own implementation key');
  const legacy = gateVerdict({ ref: leg(), cand: leg({ useful_tok_s_per_gpu: 4900 }), mode: 'fast', allowedImpl: KERNEL });
  ok(legacy.klass === 'incomparable' && /no implementation dict/.test(legacy.reason),
    'FAIL CLOSED: legs without an implementation dict are refused when the gate must check implementation');
  {
    const { overlayShaCheck } = load(['overlayShaCheck']);
    const L = (sha) => ({ implementation: sha ? { GEAK_OVERLAY_SHA: sha } : {} });
    ok(overlayShaCheck(L('a'), L('a'), L('a+b')).ok, 'cand = carried stack + one overlay passes');
    ok(overlayShaCheck(L('a'), L('a'), L('a+b+c')).ok, 'a two-manifest overlay (rmsnorm) adds two components and passes');
    ok(!overlayShaCheck(L('a+b'), L('a'), L('a+b+c')).ok, 'a BARE/older ref against the carried stack is refused (audit G3)');
    ok(!overlayShaCheck(L('a'), L('a'), L('x+b')).ok, 'a cand that does not extend the carried stack is refused');
    ok(!overlayShaCheck(L('a'), L('a'), L('a')).ok, 'a cand carrying no new overlay is refused');
    ok(overlayShaCheck(L(''), L(''), L('b')).ok, 'bare reference + first overlay passes');
  }
  // Audit 2026-10-22 G4: a leg trimmed to {status, throughput} used to be "clean".
  const trimmed = { status: 'ok', useful_tok_s_per_gpu: 1, config_digest: 'D', digest_schema: 5, implementation: {} };
  ok(!legUsable(trimmed).ok && /lacks the evidence/.test(legUsable(trimmed).why),
    `FAIL CLOSED: a leg missing occupancy/hang/dispersion/step_shape is unusable (${legUsable(trimmed).why.slice(0, 80)})`);
  // ranks_agree_on_argv must be TRUE, not merely not-false.
  ok(!comparable(leg({ ranks_agree_on_argv: undefined }), leg()).ok, 'a leg that does not say its ranks agree is not comparable');
  // The digest check runs last and is flagged, so ConfigSweep can re-gate only that case.
  const dOnly = comparable(leg(), leg({ config_digest: 'other' }));
  ok(!dOnly.ok && dOnly.digestOnly === true, 'a pair differing only in config_digest is flagged digestOnly');
  const notOnly = comparable(leg({ implementation: IMPL }), leg({ config_digest: 'other',
    implementation: Object.assign({}, IMPL, { 'manual-gc': 'true' }) }), ['ddp-average-in-collective']);
  ok(!notOnly.ok && !notOnly.digestOnly && /implementation differs/.test(notOnly.why),
    'an implementation mismatch is refused BEFORE the digest check, so a digest-moving lever cannot bypass it');

  // "we measured and it is not there" vs "we could not measure". A real candidate at +1.147% was
  // recorded as no_win against a 2.467% band -- 16.4x the floor -- because the reference leg's IQR
  // was 3.324% where this workload gives 0.05-0.11%. The legs were the problem; the record blamed
  // the candidate.
  const noisyPair = { ref: leg({ worst_iqr_spread_pct: 3.324, min_steady_steps: 29 }),
                      cand: leg({ worst_iqr_spread_pct: 1.316, min_steady_steps: 29,
                                  useful_tok_s_per_gpu: 4815 }),
                      comparison: { delta_pct: 1.147, window_artifact_pct: 0 }, mode: 'fast' };
  const nz = gateVerdict(noisyPair);
  ok(nz.klass === 'inconclusive_noise' && nz.remeasure === true,
    'a sub-band delta on legs far noisier than the workload is INCONCLUSIVE, not a no_win');
  ok(/NOT evidence against the candidate/.test(nz.reason),
    '...and says so, because the two must never be read the same way');
  // quiet legs, same delta -> an ordinary verdict again, so the rule is about the legs not the delta
  const quiet = gateVerdict({ ...noisyPair,
    ref: leg({ worst_iqr_spread_pct: 0.06, min_steady_steps: 29 }),
    cand: leg({ worst_iqr_spread_pct: 0.10, min_steady_steps: 29, useful_tok_s_per_gpu: 4815 }) });
  ok(quiet.klass !== 'inconclusive_noise',
    'the same delta on quiet legs is judged normally — the rule keys on the legs, not the number');
  ok(quiet.accepted === true,
    '...and on quiet legs +1.147% clears the floor, which is what the noisy pair could not show');

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

  // skip_heads must be applied at BOTH queue-construction sites. The re-profile path rebuilds the
  // queue from the Architect's fresh output, so a skip applied only at the initial build silently
  // re-introduces a head whose verdict is already known -- and the run then spends a gate on it.
  ok(/let headQueue = (rankQueue\()?applySkips\(/.test(src),
    'skip_heads is applied when the queue is first built');
  ok(/headQueue = (rankQueue\()?applySkips\(r\.strategy\.head_candidates/.test(src),
    'AND when the re-profile rebuilds it — otherwise the first accept re-introduces a skipped head ' +
    'and the run spends a gate on a verdict it already has');
  ok(/wfReturn\.skipped_heads/.test(src),
    'a filtered queue is recorded in the result — a silently shortened queue is indistinguishable ' +
    'from one that legitimately had nothing else in it');
    ok(/eng\.forward !== true \|\| eng\.backward !== true/.test(src),
      'engagement must be PROVEN true for forward AND backward (absent or malformed is refused)');
    ok(/audit\.l0\.pass === true && audit\.l1 && audit\.l1\.pass === true/.test(src),
      'a kernel needs L0 AND L1 explicitly passed; an absent Auditor result is not a pass');
    ok(/overlayShaCheck\(refLeg, integ\.ref, integ\.cand\)/.test(src),
      'the kernel gate checks the ref IS the carried stack and the cand extends it');
    ok(/has_backward === false/.test(src),
      'a task with no backward is refused: a forward-only task can call a slower kernel a win');
    // ConfigSweep now exists. What must be pinned is the line it may not cross: a CLASS B lever
    // changes what a step contains, so step i is not the same batch on both legs -- the step-1
    // probe and the step-aligned delta are both invalid, and L0 does not apply because no kernel
    // changed. Such a change would rest on throughput ALONE. The refusal lives in the ORCHESTRATOR,
    // not only in the role card: a rule only a prompt knows is a rule the next prompt can forget.
    ok(/CLASS_B_FLAG/.test(src) && /function classOfLever/.test(src),
      'ConfigSweep classifies levers in code, not only in the role prompt');
    for (const lever of ['packing', 'global-batch-size', 'micro-batch-size', 'seq-length']) {
      ok(new RegExp(`'${lever}'`).test(src),
        `'${lever}' is named as Class B — it changes what a step contains`);
    }
    ok(/refused: true/.test(src) && /rest on throughput/.test(src),
      'a Class B proposal is REFUSED and says why, rather than being silently downgraded');
    // Behaviour, not just presence: extract the classifier and run it.
    const { classOfLever } = load(['classOfLever'],
      ((src.match(/const GATEABLE_CLASS_B = \[[\s\S]*?\];/) || [''])[0]) + '\n' +
      ((src.match(/const CLASS_B_FLAG = \{[\s\S]*?\};/) || [''])[0]));
    ok(classOfLever(['--overlap-grad-reduce']).cls === 'A',
      'comm overlap classifies as A — it changes HOW a step is computed');
    ok(classOfLever(['--recompute-granularity', 'selective']).cls === 'A',
      'recompute granularity classifies as A, which is why it can be the memory converter');
    ok(classOfLever(['--global-batch-size', '128']).cls === 'B',
      'GBS classifies as B — it changes what a step contains');
    ok(classOfLever(['--overlap-param-gather']).cls === 'A' && classOfLever(['--attention-backend', 'fused']).cls === 'A',
      'classification is by flag name: no substring of an unrelated flag or value makes a lever Class B');
    ok(classOfLever(['--train-data-path', '/work/data/x_packed.jsonl', 'packing']).key === 'packing',
      'the packing proposal (data path + packing token) is the gateable Class-B packing lever');
    ok(classOfLever(['--sft-packing']).cls === 'B',
      'packing classifies as B even when the flag is spelled differently');
    ok(classOfLever(['--sft-packing']).gateable === true && classOfLever(['--global-batch-size', '128']).gateable === false,
      'packing is the ONE gateable Class-B lever; GBS stays ungateable (it changes the optimizer trajectory by design)');
    ok(/klass: 'recipe_change'/.test(src),
      'a lever that moves WORKLOAD identity is a recipe change and is refused (L1 cannot gate it, the ' +
      'fixed-recipe rule forbids tuning it); implementation levers keep config_digest and gate normally');
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
