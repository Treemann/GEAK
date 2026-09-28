export const meta = {
  name: 'training-workflow',
  description: 'End-to-end LLM SFT TRAINING-throughput optimizer for AMD Instinct MI GPUs (CDNA gfx942/gfx950). A system layer (Training Director / System Architect / Profiler / Kernel Extractor / Convergence Auditor / Integrator) wrapping the UNCHANGED single-kernel kernel_workflow. Measures useful (non-pad, loss-unmasked) tokens/s/GPU on a steady-state window via a stack-neutral torch-level probe, decomposes the step into compute / exposed-comm / bubble / straggler, extracts hot fwd+bwd kernels, optimizes them recursively, overlays them back reversibly, and gates on a measured throughput delta plus L0 (op numerics) and L1 (loss/grad-norm trajectory). Stack is pluggable via scripts/adapters/<stack>.sh (megatron shipped).',
  whenToUse: 'Raise the SFT training throughput of an LLM on AMD Instinct MI GPUs. Pass args.workload (a workloads/*.env descriptor) + args.workflow_dir. Use fast_mode for the fixed-config kernel track; deep_mode adds ConfigSweep.',
  phases: [
    { title: 'Setup', detail: 'Director pins the config IR, measures the TRUE baseline, and calibrates the band by repeating it' },
    { title: 'Profile', detail: 'Profiler decomposes the step across ranks + emits Top-N and the READ-ONLY config diagnosis' },
    { title: 'Strategize', detail: 'Architect routes kernels by critical-path share (NOT %GPU time)' },
    { title: 'ConfigSweep', detail: 'deep mode only: Config Tuner searches the Tier-0 lever pack' },
    { title: 'KernelTrack', detail: 'per head op: Extract(fwd+bwd) -> recursive kernel_workflow -> L0 -> Overlay -> L1 -> throughput gate' },
    { title: 'Finalize', detail: 'Integrator assembles the overlay bundle (+ optional L2 soak)' },
    { title: 'Report', detail: 'Architect writes the report and grows the playbook' },
    { title: 'Validate', detail: 'Director independently re-measures and arbitrates' },
  ],
};

const A = args || {};
const log0 = (m) => log(`[training] ${m}`);

// ===========================================================================
// SKELETON STATUS — read this before trusting a result from this file.
// The DETERMINISTIC half (mode gating, the lane/gate barrier, gate arithmetic, the noise band, the
// stop rule, the ConfigSweep seam) is implemented and unit-tested by scripts/test_training_gate.js.
// The AGENT half needs roles/*.md, which are not written yet. Phases whose role is missing FAIL LOUDLY
// (see requireRole) rather than returning a plausible-looking empty result — a training run that
// silently skips its gate would report a fabricated speedup.
// ===========================================================================
const IMPLEMENTED_ROLES = new Set([
  'training_director', 'system_architect', 'profiler',
  'kernel_extractor', 'convergence_auditor', 'integrator',
]);

function requireRole(role) {
  if (!IMPLEMENTED_ROLES.has(role)) {  // guards against a phase being wired before its prompt exists
    throw new Error(
      `roles/${role}.md is not written yet. This workflow refuses to run a phase whose agent is ` +
      `missing: skipping it would produce a result that looks complete and is not. ` +
      `Implemented roles: [${[...IMPLEMENTED_ROLES].join(', ') || 'none'}].`);
  }
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
const WORKFLOW_DIR = A.workflow_dir || '.';
const WORKLOAD = A.workload || '';            // workloads/<name>.env — REQUIRED, no default
const STACK = A.stack || 'megatron';
const GPU_IDS = String(A.gpu_ids || '0,1,2,3,4,5,6,7').split(',').map((s) => s.trim()).filter(Boolean);
const NPROC = parseInt(A.nproc != null ? A.nproc : GPU_IDS.length, 10);
const ITERS = parseInt(A.iters != null ? A.iters : 40, 10);
const KERNEL_WF_DIR = (A.kernel_workflow_dir || `${WORKFLOW_DIR}/../kernel_workflow`).replace(/\/+$/, '');
// Call the WORKER lane directly, not the dispatcher. kernel_workflow.js is a dispatcher: in
// mode=optimize/author (the default) it is a pure pass-through that forwards every arg unchanged to
// kernel_lane.js, and in mode=bakeoff it fans out to lanes itself. Either way it spends a nesting
// level, and the runtime allows exactly one: training -> dispatcher -> lane is three, and throws
// "workflow() cannot be called from within a child workflow" AFTER extraction has already run.
// e2e_workflow.js hit this and resolved it the same way (see its KERNEL_WF_SCRIPT comment); the
// training layer simply never got the same update. The worker's behaviour and args are unchanged,
// so this is a routing fix, not a semantic one — the kernel layer is still the UNCHANGED one.
// Consequence worth knowing: mode=bakeoff is NOT reachable from here, because the fan-out it does
// would need the level this layer has already spent. A bake-off has to be driven from the top.
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;
const EXP_ROOT = A.exp_root || '/work/runs';

// ---- modes (mirror of e2e_workflow.js; deep wins if both are set) ----------------------------
// Only TWO modes here, not three: a training gate costs two job launches and blocks every lane, so
// the useful distinction is "fixed config" vs "search the config too". Every deep-only knob is
// selected by a `DEEP_MODE ? deep : fast` ternary and every deep-only phase is gated by want(),
// so adding deep later must leave a fast run byte-identical.
const DEEP_MODE = String(A.deep_mode != null ? A.deep_mode : 'false') === 'true';
const FAST_MODE = !DEEP_MODE;
const PHASES = String(A.phases || 'all').split(',').map((s) => s.trim()).filter(Boolean);
const RUN_ALL = PHASES.includes('all');
const FAST_SKIP = FAST_MODE ? new Set(['config']) : null;
const DEEP_SKIP = null;
const want = (p) => (RUN_ALL || PHASES.includes(p)) && !(FAST_SKIP && FAST_SKIP.has(p)) && !(DEEP_SKIP && DEEP_SKIP.has(p));

// ---- budgets ---------------------------------------------------------------------------------
// Training is bounded by GATES, not only by wall clock: one A/B = two job launches (checkpoint load
// + warmup + N steady steps) and it blocks every kernel lane while it runs. Agent time is cheap by
// comparison. This is the opposite of the inference layer, where the gate is comparatively cheap.
const BUDGET = parseInt(A.budget != null ? A.budget : 4, 10);                  // kernel tasks
const MAX_GATES = parseInt(A.max_gates != null ? A.max_gates : (DEEP_MODE ? 24 : 8), 10);
const TIME_BUDGET_MS = parseInt(A.time_budget_ms != null ? A.time_budget_ms
  : (DEEP_MODE ? 86400000 : 18000000), 10);                                    // 24h / 5h
let gatesUsed = 0;
let tasksUsed = 0;

// ---- measurement policy ------------------------------------------------------------------------
// Floors come from MEASURED reproducibility, not convention: two independent jobs at identical
// config on this box agreed to 0.128% on useful_tok_s_per_gpu, and the within-job IQR spread is
// 0.07-0.13%. A 2-3% band (the inference layer's) would throw away every real sub-1% win here.
// The floor DEPENDS ON HOW THE DELTA WAS OBTAINED, and a single value is wrong in both directions.
// Six pairwise comparisons of one 7B configuration spanning four days:
//     window-aligned   worst |delta| 0.0589%   (typical 0.02-0.05%)
//     unaligned        worst |delta| 0.4456%   (window artifact up to 0.3867%)
// A single 0.3% floor is ~5x too loose for the aligned regime (it discards real sub-0.3% wins) AND
// too TIGHT for the unaligned fallback (0.4456% > 0.3%, so the gate manufactures a win out of window
// placement). Two floors, each above its own regime's worst observation with margin.
// NOTE the floor is only a floor: Setup measures this workload's own reproducibility and the larger
// of the two wins. That matters because reproducibility is workload-dependent — the same six-pair
// analysis gives 0.06% at 7B but 0.31-0.42% at 0.5B, where host-side jitter is a far bigger share of
// the step (idle 14% vs 2.4%) and does not cancel.
//
// RECALIBRATED 2026-09-23 against 47 same-identity baseline pairs on disk (true delta zero by
// construction), computed with compare_legs.py. The old formula, band = max(floor, max_iqr x 2),
// was not merely conservative -- it was BACKWARDS:
//     7B ws=8   n=10 pairs   worst |aligned delta| 0.0823%   assigned band 0.914%  -> 11x too LOOSE
//     0.5B ws=2 n=37 pairs   worst |aligned delta| 0.4966%   floor 0.15%           -> 6/37 FALSE WINS
// It handed the workload with the TIGHTEST reproducibility the WIDEST band, because
// worst_iqr_spread_pct is a WITHIN-leg dispersion and carries almost no information about across-job
// reproducibility: pearson(max_iqr, |aligned delta|) = +0.07 at 0.5B and NEGATIVE (-0.20) at 7B. The
// measured OLS slope where it is predictive at all is 0.147, not the 2.0 the formula assumed.
//
// The term was also dimensionally wrong: it priced the dispersion of ONE step as the uncertainty of
// a MEDIAN OF n STEPS, omitting the 1/sqrt(n). That single error explains both failure directions.
// Corrected: se(leg) = 0.929 x iqr / sqrt(steps)   (0.929 = 1.253/1.349, the SE of a median implied
// by an IQR), combined in quadrature across the pair.
//
// MEASURED, NOT ASSUMED: the dimensional fix ALONE is worse (15/47 false wins) -- the floor does the
// real work, and the se term only binds for an unusually noisy candidate leg. Both are kept: the
// floor for the common case, the se term for the case the old IQR term was reaching for.
// Validation on those 47 pairs: current formula 6 false wins, this formula 0.
//
// The floor is PER WORKLOAD and 0.5% is the safe default for the noisier 0.5B class. Pass
// --across_job_floor_pct 0.15 for the 7B ws=8 class (worst observed 0.0823%, so 0.15% keeps ~1.8x
// margin and recovers the sub-0.5% wins the default would discard). Cheapest lever of all is NOT the
// band: at 0.5B, pairs where BOTH legs had >=29 steady steps have worst |delta| 0.2544%, against
// 0.4966% when either leg had 19 -- ~10 extra iterations halves the noise floor.
// The ALIGNED floor is per-workload (default = the noisier 0.5B class). The UNALIGNED floor stays
// where it was: since a raw ratio can no longer ACCEPT at all (no_aligned_delta), its floor now only
// governs whether a raw number is labelled a REGRESSION, and keeping it sensitive is the useful
// choice there. Read such a label as a prompt to re-measure with alignment, not as evidence -- the
// unaligned artifact has been observed at 4.59%, far above this floor.
const FLOOR_PCT = { aligned: 0.5, unaligned: 0.5 };
const ACROSS_JOB_FLOOR_PCT = parseFloat(A.across_job_floor_pct != null ? A.across_job_floor_pct
                                                                      : FLOOR_PCT.aligned);
const BAND_SAFETY = parseFloat(A.band_safety != null ? A.band_safety : 2.0);
// Multiplier on the pair's sampling error. 3 and 4 both give 0/47 on the baseline pairs (the floor
// binds there); 4 is chosen by the noisy-candidate case, not by these pairs.
const SE_SAFETY = parseFloat(A.se_safety != null ? A.se_safety : 4.0);
// The run's own measured across-job reproducibility (max pairwise aligned |delta| among Setup's
// baseline legs). Set once, after Setup. It can only WIDEN the band, never narrow it below the
// floor: three back-to-back legs underestimate day-scale drift -- at 0.5B the run-local figure was
// 0.1002% against a 0.4966% worst over 37 pairs spanning several days.
let runAcrossJobPct = 0;

// ===========================================================================
// Deterministic decision logic (unit-tested; no agent, no GPU)
// ===========================================================================

/** Noise band for a pair of legs, DERIVED from their own dispersion rather than assumed.
 *  Uses the robust (IQR) spread, because the raw spread of a clean run is dominated by isolated
 *  stalls: the same 25-step run measured 38.6% raw and 0.072% IQR. Raw would gate nothing. */
function noiseBandPct(ref, cand, aligned = true) {
  // Sampling error of a MEDIAN over n steady steps, not the dispersion of one step. Falls back to
  // the old undivided term only when the step count is unknown, which keeps a leg summarised by an
  // older probe from silently getting a 5x-tighter band than it can support.
  const se = (leg) => {
    const iqr = Number(leg && leg.worst_iqr_spread_pct) || 0;
    const n = Number(leg && leg.min_steady_steps);
    return isFinite(n) && n > 0 ? 0.929 * iqr / Math.sqrt(n) : iqr;
  };
  const sePair = Math.sqrt(se(ref) ** 2 + se(cand) ** 2);
  const floor = aligned ? ACROSS_JOB_FLOOR_PCT
                        : Math.max(ACROSS_JOB_FLOOR_PCT, FLOOR_PCT.unaligned);
  return Math.max(floor, SE_SAFETY * sePair, BAND_SAFETY * runAcrossJobPct);
}

/** Is a leg usable at all? A leg that is unusable must never become a reference. */
function legUsable(leg) {
  // ORDER MATTERS. A blocked leg also has status != "ok", so the generic check must not run first:
  // it would report "leg status != ok" and the run would treat a busy node as an ordinary non-win,
  // quietly spending the rest of the gate budget on measurements that cannot mean anything.
  if (leg && leg.contamination_exhausted) {
    return { ok: false, fatal: true,
             why: `contaminated on all ${leg.attempts || '?'} attempts — the node is not quiet` };
  }
  if (!leg || leg.status !== 'ok') return { ok: false, why: 'leg status != ok' };
  if (leg.hang_detected) return { ok: false, why: 'hang detected' };
  // A leg too short to estimate dispersion is not a measurement either. Measured: 7 steady steps
  // gave an IQR of 1.243% where 24 steps of the same workload gave 0.072% -- the quartiles simply
  // land on different samples. Small-n can err in BOTH directions, and an artificially tight band
  // is the dangerous one: it accepts changes that are not there.
  if (leg.dispersion_reliable === false) {
    return { ok: false,
             why: `only ${leg.min_steady_steps} steady steps — too few to estimate dispersion; ` +
                  're-run with more iters' };
  }
  // An ABSOLUTE dispersion ceiling, independent of the band. The band is derived FROM this number,
  // so a leg whose IQR is comparable to its own step time defeats every downstream check by widening
  // the band past any possible delta: td_baseline_legA reported worst_iqr_spread_pct 91.69% (10684 ms
  // of IQR on an 11652 ms step) and passed every check here, while inducing a 183% band -- a gate
  // that cannot reject anything. That leg turned out to be two different runs appended to one probe
  // sidecar (now refused at the source, in train_probe.py and probe_summarize.load_rank), but the
  // ceiling stays: whatever the cause, a leg this dispersed is not a measurement of anything.
  // 25% is far above the worst legitimate value observed on a clean leg (0.533%) and far below the
  // corrupt one, so it discriminates without guessing at a tight threshold.
  const iqr = Number(leg.worst_iqr_spread_pct);
  if (isFinite(iqr) && iqr > 25) {
    return { ok: false,
             why: `worst_iqr_spread_pct ${iqr}% — the step-to-step spread is the same order as the ` +
                  'step itself, so this leg cannot bound anything (a band derived from it exceeds ' +
                  'any delta a candidate could show). Suspect two runs merged into one probe dir.' };
  }
  const occ = (leg.occupancy_verdict || {}).verdict;
  if (occ === 'contaminated') return { ok: false, why: 'another process held a GPU during this leg' };
  if (occ === 'unknown') return { ok: false, why: 'isolation could not be established (rocm-smi unavailable)' };
  if (!(Number(leg.useful_tok_s_per_gpu) > 0)) return { ok: false, why: 'no useful-token throughput' };
  return { ok: true };
}

/** Two legs are comparable only if they measured the SAME workload. Changing sequence length,
 *  global batch size, the parallel mapping or the data path changes the workload identity, and a
 *  ratio across two identities is fiction no matter how tight each leg is. The digest comes from
 *  what the PROBE observed inside the ranks, not from what the caller intended. */
/** The delta MUST come from a window-aligned comparison (scripts/compare_legs.py). Each leg detects
 *  its own warmup end, and `useful_tokens` swings +-25% step to step on SFT data, so two legs
 *  summarised on their OWN windows contain different samples. Measured on two runs of an identical
 *  configuration: unaligned +0.3294% (a "win" against a 0.3% band) versus aligned +0.0235% (correctly
 *  nothing) -- 93% of the apparent delta was window placement. */
function alignedDelta(integ) {
  const c = integ && integ.comparison;
  return c && typeof c.delta_pct === 'number' ? c : null;
}

function comparable(ref, cand) {
  if (!ref.config_digest || !cand.config_digest) return { ok: false, why: 'a leg carries no config digest' };
  // Distinguish "different workload" from "the digest DEFINITION changed between these summaries".
  // The second is not a real incomparability: re-summarise from the probe sidecars, which are on
  // disk, and the question answers itself. Reporting it as a workload difference sends the reader
  // looking for a configuration change that never happened.
  const rs = ref.digest_schema, cs = cand.digest_schema;
  if (rs !== cs) {
    // Schema 3 is the exception to the "re-summarise and the question answers itself" remedy above.
    // It widened workload identity to the regularization/optimizer/architecture set after
    // attention_dropout 0.0 vs 0.1 digested identically, and a v2 probe sidecar PHYSICALLY DOES NOT
    // CONTAIN those values -- there is nothing on disk to re-summarise them from. Saying
    // "re-summarise" here would send the reader after a fix that cannot work.
    const across3 = (rs === 3) !== (cs === 3);
    return { ok: false, stale: true,
             why: `digest schema differs (${rs} vs ${cs}) — one summary predates a change in how the ` +
                  (across3
                    ? 'digest is computed. Schema 3 captures fields (dropout, lr, seed, architecture) ' +
                      'that a schema-2 sidecar never recorded, so this pair must be RE-MEASURED; ' +
                      're-summarising cannot recover them.'
                    : 'digest is computed. Re-summarise both from their probe dirs, then compare.') };
  }
  if (ref.config_digest !== cand.config_digest) {
    return { ok: false, why: `config digest differs (${ref.config_digest} vs ${cand.config_digest}): ` +
      `these legs did not measure the same workload` };
  }
  if (ref.ranks_agree_on_argv === false || cand.ranks_agree_on_argv === false) {
    return { ok: false, why: 'ranks within one leg ran different argv' };
  }
  return { ok: true };
}

/** The gate. Throughput is the only thing that can ACCEPT in fast mode; memory is banked (there is
 *  no config knob to convert it with until ConfigSweep exists). L0/L1 can only ever REJECT. */
function gateVerdict(spec) {
  const { ref, cand, l0, l1, mode } = spec;
  const ru = legUsable(ref), cu = legUsable(cand);
  if (!ru.ok) return { accepted: false, klass: ru.fatal ? 'blocked' : 'unusable_ref',
                       fatal: !!ru.fatal, reason: `reference leg unusable: ${ru.why}` };
  if (!cu.ok) return { accepted: false, klass: cu.fatal ? 'blocked' : 'unusable_cand',
                       fatal: !!cu.fatal, reason: `candidate leg unusable: ${cu.why}` };
  const cmp = comparable(ref, cand);
  if (!cmp.ok) return { accepted: false, klass: cmp.stale ? 'stale_summary' : 'incomparable',
                        reason: cmp.why };
  if (l0 && l0.pass === false) return { accepted: false, klass: 'numerics', reason: `L0 failed: ${l0.reason || 'op-level numerics'}` };
  if (l1 && l1.pass === false) return { accepted: false, klass: 'trajectory', reason: `L1 failed: ${l1.reason || 'loss/grad-norm trajectory'}` };

  const r = Number(ref.useful_tok_s_per_gpu), c = Number(cand.useful_tok_s_per_gpu);
  // Prefer the aligned delta when the Integrator supplied one; fall back to the raw ratio only when
  // it did not, and say so, because the raw ratio carries the window artifact.
  const aligned = alignedDelta(spec);
  const deltaPct = aligned ? aligned.delta_pct : ((c - r) / r) * 100;
  const band = noiseBandPct(ref, cand, !!aligned);
  const memDeltaPct = (Number(ref.peak_reserved_mb) > 0 && Number(cand.peak_reserved_mb) > 0)
    ? ((Number(cand.peak_reserved_mb) - Number(ref.peak_reserved_mb)) / Number(ref.peak_reserved_mb)) * 100
    : null;

  const base = { delta_pct: +deltaPct.toFixed(3), band_pct: +band.toFixed(3),
    delta_source: aligned ? 'window-aligned' : 'raw ratio (NOT window-aligned — includes sample-composition drift)',
    window_artifact_pct: aligned ? aligned.window_artifact_pct : null, mem_delta_pct: memDeltaPct == null ? null : +memDeltaPct.toFixed(2) };
  // A raw ratio may REJECT but may never ACCEPT. Two IDENTICAL-config 7B legs (true delta zero by
  // construction) gave unaligned +4.5925% against aligned -0.0588% at a 0.914% band: window placement
  // alone manufactures a 5x-band "win". No floor can bound that -- the artifact scales with the data's
  // +-25% useful_tokens swing, not with a constant. The asymmetry is deliberate and matches L0/L1: a
  // false REJECT costs one candidate, which can be re-gated once the Integrator reports an aligned
  // comparison; a false ACCEPT installs the candidate as the reference every later delta compounds on
  // (see `refLeg = integ.cand` below) and puts the artifact in the headline permanently.
  if (deltaPct > band && !aligned) {
    return { accepted: false, klass: 'no_aligned_delta',
             reason: `+${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%, but this is a raw ratio: ` +
                     're-report with scripts/compare_legs.py before it can be a win', ...base };
  }
  if (deltaPct > band) return { accepted: true, klass: 'win', reason: `+${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%`, ...base };
  if (deltaPct < -band) return { accepted: false, klass: 'regression', reason: `${deltaPct.toFixed(2)}% < -band`, ...base };
  // In band. Memory is a CURRENCY, not a metric: it counts only once a config change spends it.
  if (memDeltaPct != null && memDeltaPct < -2.0) {
    return {
      accepted: false, klass: 'banked_memory', banked_memory_pct: +(-memDeltaPct).toFixed(2),
      reason: mode === 'deep'
        ? `throughput in band; ${(-memDeltaPct).toFixed(1)}% memory freed -> hand to proposeMemoryConversion()`
        : `throughput in band; ${(-memDeltaPct).toFixed(1)}% memory freed, but fast mode has no config knob to spend it -> banked`,
      ...base,
    };
  }
  return { accepted: false, klass: 'no_win', reason: `${deltaPct.toFixed(2)}% within band ${band.toFixed(2)}%`, ...base };
}

/** Amdahl-equivalent for training: only CRITICAL-PATH share is recoverable. A collective fully
 *  overlapped with compute can own 30% of GPU time and be worth exactly zero. */
function expectedGainPct(head) {
  const share = Number(head && head.critical_path_share_pct) || 0;
  const att = Number(head && head.attainable_speedup) || 1;
  return att > 1 ? share * (1 - 1 / att) : 0;
}

/** Stop when the best remaining candidate cannot clear the band even if fully realised. */
function stopRule(queue, band) {
  const best = (queue || []).reduce((m, h) => Math.max(m, expectedGainPct(h)), 0);
  return { stop: best <= band, best_expected_gain_pct: +best.toFixed(3), band_pct: +band.toFixed(3) };
}

// ===========================================================================
// The lane/gate barrier — the one concurrency rule that does NOT carry over from inference.
// Inference fans kernel lanes across spare GPUs WHILE the e2e gate runs on the serving slot. A
// TP1xDP8 training job occupies all 8 GPUs, so there are no spare GPUs during a gate. Reserving a
// subset instead (job on 6, lanes on 2) fails on measurement basis, not on performance: the parallel
// mapping is part of the workload identity, so a DP6 number is not about the DP8 configuration.
// Therefore: lanes may use the whole pool, and a gate DRAINS them first.
// ===========================================================================
function makeBarrier(poolSize) {
  let active = 0;
  let gateHeld = false;
  const waitersForSlot = [];
  const waitersForDrain = [];
  const pump = () => {
    if (gateHeld) return;
    while (waitersForSlot.length && active < poolSize) { active++; waitersForSlot.shift()(); }
  };
  return {
    get active() { return active; },
    get gateHeld() { return gateHeld; },
    /** Run kernel-lane work; blocked while a gate holds the box. */
    async lane(fn) {
      if (gateHeld || active >= poolSize) await new Promise((res) => waitersForSlot.push(res));
      else active++;
      try { return await fn(); } finally {
        active--;
        if (active === 0) while (waitersForDrain.length) waitersForDrain.shift()();
        pump();
      }
    },
    /** Run a measurement that needs the WHOLE box: stop admitting lanes, wait for the box to empty. */
    async gate(fn) {
      gateHeld = true;
      if (active > 0) await new Promise((res) => waitersForDrain.push(res));
      try { return await fn(); } finally { gateHeld = false; pump(); }
    },
  };
}

// ===========================================================================
// ConfigSweep seam (locked decision 10). Six hooks exist in FAST with trivial behaviour so DEEP is
// additive rather than a rewrite. Hooks 1-3 are the expensive ones to retrofit: threading a config
// through every measurement after the fact touches every call site, and that is exactly where a
// silently stale reference leg would hide.
// ===========================================================================
let curConfig = null;                 // hook 1: carried config IR; constant in fast
const bankedMemory = [];              // hook 5 input

function benchInputs(config, tag) {   // hook 2: config is ALWAYS an explicit argument
  return { WORKLOAD, STACK, NPROC, ITERS, EXP_ROOT, CONFIG: config, TAG: tag,
           GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts`, GPU_IDS: GPU_IDS.join(',') };
}

async function reprofile(config, round) {          // hook 3: independently callable
  requireRole('profiler');
  return safeAgentStub('profiler', 'reprofile', { ...benchInputs(config, `reprofile_${round}`) });
}
async function restrategize(config, profile) {      // hook 3
  requireRole('system_architect');
  return safeAgentStub('system_architect', 'strategize', { CONFIG: config, PROFILE: profile });
}
function proposeMemoryConversion(banked) {          // hook 5: no-op in fast
  if (!DEEP_MODE) return { proposed: false, reason: 'fast mode has no ConfigSweep to spend memory with', banked };
  return { proposed: false, reason: 'ConfigSweep not implemented yet', banked };
}
function kbIdentity(config) {                       // hook 6: config hash keys every record
  return { stack: STACK, workload: WORKLOAD, gpus: NPROC, config_digest: (config && config.digest) || null };
}

// ===========================================================================
// Agent plumbing (mirrors e2e_workflow.js)
// ===========================================================================
function roleAgent(role, phase, intro, inputs) {
  return `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## MEASUREMENT INVARIANT (do not violate — every number in this run must be comparable)
Take EVERY throughput number through ${WORKFLOW_DIR}/scripts/bench_train.sh with the SAME
--workload (${WORKLOAD}), --stack ${STACK} and --nproc ${NPROC}. The headline is
bench_summary.json:useful_tok_s_per_gpu (non-pad, loss-unmasked tokens), NEVER the stack's own
tokens/s: on this workload 71.6% of tokens are padding, so the nominal figure overstates useful work
by ~3.5x. Never change the parallel mapping, global batch size, sequence length or data order between
a reference leg and a candidate leg — that changes the workload identity and makes the delta fiction.
A leg whose occupancy_verdict is not "clean", or whose status is not "ok", is NOT a measurement:
re-run it, do not report it.

## INPUTS
${JSON.stringify(inputs, null, 2)}`;
}

async function safeAgentStub(role, phase, inputs, intro = '') {
  requireRole(role);   // unreachable until roles land; keeps the call sites honest meanwhile
  const opts = { phase, label: `${role}:${phase}` };
  const schema = PHASE_SCHEMA[phase];
  if (schema) opts.schema = schema;   // absent for 'report', which returns prose by design
  return safeAgent(roleAgent(role, phase, intro, inputs), opts);
}

// An agent() call made WITHOUT a schema returns the agent's final text — a string. Every role call
// site here dereferences its result as an object (setup.baseline_legs, profile.top_n, ...), so a
// string silently reads as a wall of `undefined`. That is how a Setup leg that had genuinely
// measured two clean baselines was thrown away as "produced no baseline": the agent did its job and
// the plumbing dropped it on the floor.
//
// The *_SCHEMA constants below were declared for this and never passed to agent(). They are wired
// in now. Coercion stays as a second line of defence for two reasons: it costs nothing, and a
// schema-less call (the 'report' phase legitimately returns prose) must still not be mistaken for
// an object. Anything that parses to a non-null object is accepted; a role that emits a bare string
// or a fenced block still lands as structured data instead of failing three phases downstream.
function coerceResult(r) {
  if (r == null || typeof r === 'object') return r;
  if (typeof r !== 'string') return r;
  const txt = r.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
  if (!(txt.startsWith('{') || txt.startsWith('['))) return r;   // genuine prose (e.g. 'report')
  try {
    const parsed = JSON.parse(txt);
    return (parsed && typeof parsed === 'object') ? parsed : r;
  } catch (e) { return r; }   // malformed: hand back the raw text, let the call site's guard speak
}

async function safeAgent(prompt, opts, tries = 3) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      const r = coerceResult(await agent(prompt, opts));
      if (r) return r;
      lastErr = 'null/empty result';
    } catch (e) { lastErr = String(e); }
    log0(`agent[${(opts && opts.label) || '?'}] attempt ${i + 1}/${tries} failed: ${String(lastErr).slice(0, 160)}`);
  }
  return null;
}

// ===========================================================================
// Pipeline
// ===========================================================================
const barrier = makeBarrier(GPU_IDS.length);
const wfReturn = {
  schema_version: 1, mode: DEEP_MODE ? 'deep' : 'fast', stack: STACK, workload: WORKLOAD,
  gates_used: 0, tasks_used: 0, accepted: [], banked_memory: [], status: 'incomplete',
};

if (!WORKLOAD) {
  throw new Error('args.workload is required (a workloads/*.env descriptor). There is no default: ' +
    'silently benching the wrong target is worse than failing.');
}

log0(`mode=${wfReturn.mode} stack=${STACK} workload=${WORKLOAD} nproc=${NPROC} ` +
     `budget=${BUDGET} max_gates=${MAX_GATES} band_floor=${ACROSS_JOB_FLOOR_PCT}%`);
log0(`phases: ${['setup', 'profile', 'strategize', want('config') ? 'config' : null, 'kernel', 'final']
  .filter(Boolean).join(' -> ')}`);

// ---------------------------------------------------------------------------
// Schemas — every agent returns structured JSON; judgement is the agent's, control flow is here.
// ---------------------------------------------------------------------------
const obj = (props, req) => ({ type: 'object', properties: props, required: req || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const SETUP_SCHEMA = obj({ baseline_legs: arrObj, across_job_band_pct: { type: 'number' },
  trajectory_band_path: { type: 'string' }, config_digest: { type: 'string' },
  warmup_steps: { type: 'number' }, pad_fraction_pct: { type: 'number' } }, ['baseline_legs']);
const PROFILE_SCHEMA = obj({ buckets: { type: 'object' }, top_n: arrObj,
  config_diagnosis: { type: 'object' }, rank_skew_pct: { type: 'number' } }, ['top_n']);
const STRATEGY_SCHEMA = obj({ head_candidates: arrObj, stop: { type: 'boolean' },
  reasoning: { type: 'string' } }, ['head_candidates']);
const EXTRACT_SCHEMA = obj({ task_dir: { type: 'string' }, op: { type: 'string' },
  has_backward: { type: 'boolean' }, engagement: { type: 'object' } }, ['task_dir']);
// `comparison` is REQUIRED, not optional: without it the gate cannot accept at all (see
// no_aligned_delta), so a return value that lacks it has wasted two job launches. Requiring it makes
// the model layer retry for it instead of the orchestrator discovering the omission after the A/B.
const INTEGRATE_SCHEMA = obj({ ref: { type: 'object' }, cand: { type: 'object' },
  comparison: { type: 'object' }, engagement: { type: 'object' },
  overlay_path: { type: 'string' } }, ['ref', 'cand', 'comparison']);
const AUDIT_SCHEMA = obj({ l0: { type: 'object' }, l1: { type: 'object' },
  l2_recommended: { type: 'boolean' }, risk_class: { type: 'string' },
  // Path of the KB card written back to the kernel layer on a reject, or null on a pass. Declared so
  // the Auditor is asked for it; a rejection the KB never hears about is re-derived from zero next run.
  kb_rejection_card: { type: ['string', 'null'] } }, ['l0', 'l1']);
const FINALIZE_SCHEMA = obj({ overlay: { type: 'string' }, patch: { type: 'string' },
  launch: { type: 'string' }, engagement_recheck: { type: 'object' } }, []);
const VALIDATE_SCHEMA = obj({ validation_status: { type: 'string' },
  baseline_useful_tok_s: { type: 'number' }, final_useful_tok_s: { type: 'number' },
  delta_pct: { type: 'number' } }, ['validation_status']);

// ---- wall clock -------------------------------------------------------------------------------
// The workflow sandbox has NO clock at all: the `Date.now` / argless `new Date` family, plus
// `performance` and `process`, are each either absent or throw, so that a resumed run replays
// identically. The previous `Date.now ? … : 0` feature-detect was worse than nothing — the binding
// EXISTS and throws only when CALLED, so the guard was always truthy and the run died at launch
// before its first agent. `new Date(<explicit ms>)` is pure and still works; only "now" is gone.
//
// A wall-clock budget therefore cannot be self-served. Two honest sources, in order:
//   T0  — passed in as args.t0_epoch_ms by the launcher, which does have a clock.
//   now — read through an agent, which runs bash and so can call `date`.
// On resume a cached stamp replays its ORIGINAL value. That is the behaviour we want: hours spent
// parked between sessions must not count against a budget that is meant to bound GPU work.
//
// If T0 is absent the budget is unenforceable. We then say so, once and loudly, and let the loop's
// two hard bounds (BUDGET, MAX_GATES) carry it — rather than fabricating a deadline from a clock we
// do not have, or silently reporting a bound that is not actually being applied.
// Phase -> schema. Passing this to agent() forces the role through the StructuredOutput tool, so a
// shape mismatch retries at the MODEL layer instead of arriving here as prose to be pattern-matched.
// 'report' is deliberately absent: it returns a written report, and prose is its correct output.
const PHASE_SCHEMA = {
  setup: SETUP_SCHEMA, profile: PROFILE_SCHEMA, reprofile: PROFILE_SCHEMA,
  strategize: STRATEGY_SCHEMA, extract: EXTRACT_SCHEMA, integrate: INTEGRATE_SCHEMA,
  gate: AUDIT_SCHEMA, finalize: FINALIZE_SCHEMA, validate: VALIDATE_SCHEMA,
};

const T0 = A.t0_epoch_ms != null ? parseInt(A.t0_epoch_ms, 10) : null;
if (T0 == null || !isFinite(T0)) {
  log0(`WARNING: no args.t0_epoch_ms, so the ${(TIME_BUDGET_MS / 3600000).toFixed(1)}h wall-clock ` +
       `budget is NOT enforced this run. The lane loop is still bounded by budget=${BUDGET} tasks ` +
       `and max_gates=${MAX_GATES}. Pass t0_epoch_ms (epoch ms at launch) to enforce it.`);
}
const CLOCK_SCHEMA = obj({ epoch_ms: { type: 'number' } }, ['epoch_ms']);
let clockFailed = false;
const outOfTime = async () => {
  if (T0 == null || !isFinite(T0) || clockFailed) return false;
  const r = await agent(
    'Run exactly `date +%s%3N` and return its output as epoch_ms. Do nothing else — no files, no ' +
    'analysis. This is a clock read for the orchestrator\'s wall-clock budget.',
    { label: 'clock', phase: 'KernelTrack', schema: CLOCK_SCHEMA });
  if (!r || typeof r.epoch_ms !== 'number' || !isFinite(r.epoch_ms)) {
    clockFailed = true;   // a broken clock must not end the run early; the hard bounds still hold
    log0('WARNING: clock read failed; wall-clock budget disabled for the rest of this run.');
    return false;
  }
  const elapsed = r.epoch_ms - T0;
  const over = elapsed > TIME_BUDGET_MS * 0.6;
  log0(`elapsed ${(elapsed / 3600000).toFixed(2)}h of ${(TIME_BUDGET_MS * 0.6 / 3600000).toFixed(2)}h` +
       ` lane deadline${over ? ' — OVER, draining' : ''}`);
  return over;
};

// ---------------------------------------------------------------------------
// Setup — the TRUE baseline, measured twice so this run's bands are its own.
// ---------------------------------------------------------------------------
phase('Setup');
// REPEAT is 3, not 2, and the third leg is the cheapest correctness this pipeline buys. With two
// legs the L1 envelope is a single abs(A-B) sample per step, so it collapses wherever those two legs
// happened to cross -- and BOTH real runs then had their L1 verdict rejected by its own null
// control, forcing the Director to hand-recalibrate mid-flight (0.5B from 5 legs, 7B likewise).
// trajectory_gate.MIN_BAND_LEGS is 3: below it the envelope is reported but may not vote. The cost
// is one extra baseline leg per run; the legs also feed the across-job throughput band, so it is not
// wasted. If the band's null control still fails at 3, raise this to 5 -- the band now SAYS so.
// REUSING a baseline is the single biggest speed lever available, and schema 3 is what makes it
// safe: before it, config_digest covered five fields and two different workloads could share one.
// Measured on the 2026-09-24 run, Setup cost 0.7h of a 10.6h run -- and it measured SIX legs when
// asked for three, so the real saving is larger than the nominal one. The Director still VALIDATES
// every reused leg (digest, schema, occupancy, dispersion); it just does not re-measure it.
const BASELINE_DIRS = String(A.baseline_dirs || '').split(',').map((x) => x.trim()).filter(Boolean);
if (BASELINE_DIRS.length) {
  log0(`baseline reuse: ${BASELINE_DIRS.length} leg(s) offered — the Director must verify identity ` +
       `and cleanliness before accepting them, and re-measure if any is unusable`);
}
const setup = await safeAgentStub('training_director', 'setup',
  { ...benchInputs(curConfig, 'baseline'), REPEAT: 3,
    ...(BASELINE_DIRS.length ? { BASELINE_DIRS } : {}) },
  'Pin the config IR, measure the TRUE baseline THREE times, and derive both bands from those legs. ' +
  'Measure EXACTLY REPEAT usable legs — if you take more, the extra legs must be named in notes with ' +
  'the reason, because an unexplained extra leg is 7 minutes of 8-GPU time nobody budgeted. ' +
  'If BASELINE_DIRS is given, verify and REUSE those legs instead of measuring. ' +
  'The L1 band MUST pass its own null control; if it does not, say so rather than using it.');
if (!setup || !(setup.baseline_legs || []).length) {
  throw new Error('Setup produced no baseline. Every later number is a ratio against it, so there ' +
    'is nothing to continue with — a run that proceeds here reports deltas against nothing.');
}
const baselineLegs = setup.baseline_legs;
const baseline = baselineLegs[0];
curConfig = { digest: setup.config_digest || (baseline && baseline.config_digest) || null,
              descriptor: WORKLOAD };
// The measured floor REPLACES the default when it is wider; a box noisier than the default must not
// inherit a band that flatters it.
const measuredFloor = Number(setup.across_job_band_pct) || 0;
// Feed the run's own across-job reproducibility into every later band. It can only WIDEN one: with
// REPEAT 3 this is the max pairwise aligned |delta| among three back-to-back legs, which measures
// this box today and systematically UNDERSTATES day-scale drift (0.5B: 0.1002% run-local against a
// 0.4966% worst over 37 pairs spanning days). So it is a widener, never a substitute for the floor.
runAcrossJobPct = measuredFloor;
const runFloor = Math.max(ACROSS_JOB_FLOOR_PCT, measuredFloor * BAND_SAFETY / 2);
log0(`baseline: ${baseline && baseline.useful_tok_s_per_gpu} useful tok/s/GPU; ` +
     `measured across-job ${measuredFloor}% -> run floor ${runFloor.toFixed(3)}%`);
wfReturn.baseline = baseline;
wfReturn.bands = { across_job_measured_pct: measuredFloor, floor_pct: runFloor,
                   trajectory_band_path: setup.trajectory_band_path || null };

// ---------------------------------------------------------------------------
// Profile + Strategize
// ---------------------------------------------------------------------------
async function profileAndRoute(config, round) {
  phase('Profile');
  const prof = await reprofile(config, round);
  if (!prof) return { profile: null, strategy: null };
  const diag = prof.config_diagnosis || {};
  if (diag.verdict === 'config_first') {
    // fast mode cannot treat this. Saying it plainly is what makes a later rejection attributable
    // to the configuration rather than blamed on the kernel.
    log0(`[diagnosis] config_first — kernel work is unlikely to show up end to end here:`);
    (diag.blocking || []).forEach((b) => log0(`  BLOCKING ${b.finding} -> ${b.knob}`));
    if (!DEEP_MODE) log0('  fast mode has no ConfigSweep; proceeding, but expect rejections.');
  }
  phase('Strategize');
  const strat = await restrategize(config, prof);
  return { profile: prof, strategy: strat };
}
let { profile, strategy } = await profileAndRoute(curConfig, 'baseline');
wfReturn.config_diagnosis = (profile && profile.config_diagnosis) || null;
// Heads whose verdict is already known from a previous run. Without this every run re-profiles the
// BARE stack, rebuilds the same queue, and spends its budget re-deriving decided answers: the 7B run
// of 2026-09-24 stopped on the wall clock having done lmhead (re-accepted) and attention (rejected),
// leaving the same 3 heads unexplored a second time. Skipping is by SHORT NAME and is logged, because
// a silently shortened queue looks identical to a queue that legitimately had nothing else in it.
const SKIP_HEADS = new Set(String(A.skip_heads || '').split(',').map((x) => x.trim()).filter(Boolean));
const applySkips = (q) => {
  if (!SKIP_HEADS.size) return q;
  const kept = q.filter((h) => !SKIP_HEADS.has(h.short_name) && !SKIP_HEADS.has(h.kernel));
  const dropped = q.length - kept.length;
  if (dropped) log0(`skip_heads: ${dropped} head(s) removed from the queue by args.skip_heads ` +
                    `[${[...SKIP_HEADS].join(', ')}] — their verdicts come from an earlier run, not this one`);
  return kept;
};
let headQueue = applySkips(((strategy && strategy.head_candidates) || []).slice());
log0(`heads queued: ${headQueue.length}${strategy && strategy.stop ? ' (architect says stop)' : ''}`);
// The Architect's `stop` was PRINTED and then ignored: it could report that no head is worth a gate
// and the loop would spend the whole budget regardless. A gate costs two job launches and blocks
// every lane, so an explicit "nothing here pays" is the cheapest signal in the run. Honour it, but
// only when it is not contradicted by the Architect's own numbers -- if it queued a head it expects
// to clear the band, the queue is the better evidence and the flag is treated as stale.
if (strategy && strategy.stop) {
  const best = headQueue.length ? Math.max(...headQueue.map((h) => expectedGainPct(h))) : 0;
  if (best <= runFloor) {
    log0(`architect says stop and no queued head beats the band (best ${best.toFixed(2)}% <= ` +
         `${runFloor.toFixed(2)}%) — skipping the kernel track rather than spending gates on it`);
    headQueue.length = 0;
  } else {
    log0(`architect says stop, but queued a head expected at ${best.toFixed(2)}% > band ` +
         `${runFloor.toFixed(2)}% — continuing, and recording the disagreement`);
    wfReturn.architect_stop_overruled = { best_expected_gain_pct: +best.toFixed(2), band_pct: +runFloor.toFixed(3) };
  }
}

// ---------------------------------------------------------------------------
// ConfigSweep — deep only. Refuses rather than silently skipping.
// ---------------------------------------------------------------------------
if (want('config')) {
  phase('ConfigSweep');
  throw new Error('ConfigSweep is not implemented yet (deep mode). The six seam hooks exist in fast ' +
    'mode, but the sweep itself does not: running deep now would report a fast-mode result under a ' +
    'deep label. Use fast_mode, or implement ConfigSweep.');
}

// ---------------------------------------------------------------------------
// KernelTrack — the fast-mode optimization loop.
// Lanes may use the whole GPU pool; a gate DRAINS them, because a training A/B needs every GPU.
// ---------------------------------------------------------------------------
phase('KernelTrack');
let curOverlay = null;
let refLeg = baseline;          // the carried accepted stack, not the bare baseline
let noImprove = 0;

// outOfTime() is async (it reads the clock through an agent) and is deliberately LAST in the
// condition: && short-circuits, so the free bounds are checked first and no clock agent is spawned
// once the queue is empty or the task/gate budget is already spent.
while (headQueue.length && tasksUsed < BUDGET && gatesUsed < MAX_GATES && !(await outOfTime())) {
  const stop = stopRule(headQueue, runFloor);
  if (stop.stop) { log0(`stop rule: best remaining ${stop.best_expected_gain_pct}% <= band ${stop.band_pct}%`); break; }
  const h = headQueue.shift();
  tasksUsed += 1;
  log0(`[head ${tasksUsed}/${BUDGET}] ${h.short_name || h.kernel} ` +
       `cp_share=${h.critical_path_share_pct}% expected=${expectedGainPct(h).toFixed(2)}%`);

  // ---- lane: extract + optimize (no training job is running, so lanes own the pool) ----
  const optimized = await barrier.lane(async () => {
    const ext = await safeAgentStub('kernel_extractor', 'extract',
      { ...benchInputs(curConfig, `extract_${h.short_name}`), HEAD: h, PROFILE: profile },
      'Capture real shapes AND grad_output; build an immutable fwd+bwd oracle.');
    if (!ext || !ext.task_dir) { log0('  extraction produced no task dir — skipping head'); return null; }
    if (ext.has_backward === false) {
      log0('  WARNING: task has no backward. A forward-only task can report a win for a kernel that ' +
           'is slower overall; treating as unusable.');
      return null;
    }
    // The UNCHANGED kernel layer, called exactly as a hand-written kernel task would be.
    // Carry SYSTEM-LEVEL rejections down as a starting constraint. The kernel layer gates on its own
    // frozen unittest and its own latency, and it is right to -- but those cannot see a defect that
    // only appears once the kernel is inside a real training step. Measured instance: the 7B
    // attention candidate was ACCEPTED by the lane (and its KB card still says so) while the
    // training layer rejected it for an 88.50% step-1 grad_norm collapse in the BACKWARD. Without
    // this, the next lane on the same op starts from a card that says "accepted" and re-derives the
    // rejection from zero -- or worse, does not.
    const priorRejections = (wfReturn.rejected || [])
      .filter((x) => x.head === h.short_name || x.head === h.kernel)
      .map((x) => `- ${x.verdict.klass}: ${String(x.verdict.reason || '').slice(0, 400)}`);
    const laneTask = priorRejections.length
      ? 'CONSTRAINT FROM THE TRAINING LAYER. A previous candidate for this op passed the kernel-level '
        + 'gate and was then REJECTED once it ran inside a real training step. The lane\'s unittest '
        + 'cannot see this class of defect, so treat it as a requirement, not a hint:\n'
        + priorRejections.join('\n')
        + '\nThe system gates on step-1 loss AND step-1 grad_norm against the unmodified stack on '
        + 'identical weights, so a backward that is fast and unittest-clean can still be refused.'
      : '';
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
      kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
      budget: parseInt(A.kernel_budget != null ? A.kernel_budget : 3, 10),
      gpu_ids: GPU_IDS.join(','),
      ...(laneTask ? { task: laneTask } : {}),
    });
    return { ext, kernel: r };
  });
  if (!optimized || !optimized.kernel) { noImprove += 1; continue; }

  // ---- gate: exclusive whole-box A/B ----
  const integ = await barrier.gate(async () => {
    gatesUsed += 1;
    return safeAgentStub('integrator', 'integrate',
      { ...benchInputs(curConfig, `gate_${h.short_name}`), HEAD: h, TASK: optimized.ext,
        KERNEL_RESULT: optimized.kernel, CARRIED_OVERLAY: curOverlay, REF_EXPECTED: refLeg },
      'Overlay reversibly, prove engagement for forward AND backward, take a paired same-session A/B.');
  });
  if (!integ || !integ.ref || !integ.cand) { noImprove += 1; continue; }
  if (integ.engagement && (integ.engagement.forward === false || integ.engagement.backward === false)) {
    log0('  engagement NOT proven on both paths — refusing the result (an overlay bound to only one ' +
         'half is the exact failure L1 exists to catch, and it is cheaper to catch here)');
    noImprove += 1; continue;
  }

  const audit = await safeAgentStub('convergence_auditor', 'gate',
    { TASK: optimized.ext, REF_PROBE: integ.ref.probe_dir, CAND_PROBE: integ.cand.probe_dir,
      BAND: wfReturn.bands.trajectory_band_path },
    'Run L0 (out/dgrad/wgrad vs fp32, magnitude AND sign) and L1 (step-indexed envelope).');

  // Forward the Integrator's window-aligned comparison. Omitting it here made alignedDelta() return
  // null on EVERY real gate -- the unit tests pass `comparison` straight into gateVerdict(), so they
  // exercised the aligned path while production never did, and the 7B run gated on a raw ratio.
  const v = gateVerdict({ ref: integ.ref, cand: integ.cand, comparison: integ.comparison,
                          l0: audit && audit.l0, l1: audit && audit.l1,
                          mode: DEEP_MODE ? 'deep' : 'fast' });
  // An unusable leg has a CAUSE, and the Integrator is the only thing that saw it. Flattening that
  // to "leg status != ok" threw away the actual finding once already: the mlp_fc1_swiglu candidate
  // died at iteration #2 with an inf gradient (reproduced twice, Megatron's rerun_state_machine
  // caught it and exited), and the run recorded an opaque `unusable_cand`. A candidate that DIVERGES
  // is a stronger result than one that is merely slow -- it must not read as a measurement failure.
  if (!v.accepted && /unusable|incomparable|stale/.test(v.klass || '') && integ.notes) {
    v.integrator_diagnosis = String(integ.notes).slice(0, 1200);
    v.reason = `${v.reason} — Integrator: ${String(integ.notes).split('\n')[0].slice(0, 220)}`;
  }
  log0(`  gate: ${v.klass} — ${v.reason}`);
  if (v.fatal) {
    // Retries are already exhausted inside bench_train.sh; continuing would spend the remaining
    // gate budget measuring a busy node and report the results as if they meant something.
    wfReturn.blocked = { reason: v.reason, head: h.short_name || h.kernel };
    log0('  STOPPING: the box is not quiet enough to measure on. Re-run when it is.');
    break;
  }
  const record = { head: h.short_name || h.kernel, verdict: v, audit: audit || null,
                   overlay: integ.overlay_path || null, task_dir: optimized.ext.task_dir };

  if (v.accepted) {
    curOverlay = integ.overlay_path || curOverlay;
    refLeg = integ.cand;                     // later candidates are gated ON TOP of this
    wfReturn.accepted.push(record);
    noImprove = 0;
    // An accepted change moves which kernels dominate — re-profile, exactly as a config win would.
    const r = await profileAndRoute(curConfig, `after_${h.short_name}`);
    if (r.profile) profile = r.profile;
    // Re-apply the skips: a re-profile rebuilds the queue from scratch and would otherwise
    // re-introduce a head whose verdict is already known.
    if (r.strategy && r.strategy.head_candidates) headQueue = applySkips(r.strategy.head_candidates.slice());
  } else {
    if (v.klass === 'banked_memory') {
      bankedMemory.push({ head: record.head, pct: v.banked_memory_pct });
      wfReturn.banked_memory.push({ head: record.head, pct: v.banked_memory_pct });
    }
    wfReturn.rejected = wfReturn.rejected || [];
    wfReturn.rejected.push(record);
    noImprove += 1;
  }
  if (noImprove >= 2) { log0('two consecutive non-wins — stopping the kernel track'); break; }
}
// What was NOT tried, and why the loop ended. The 7B run stopped on the wall clock with 3 of 4
// queued heads never attempted, and nothing in the result said so -- a reader seeing accepted=1,
// tasks=1/2 cannot tell "nothing else was worth trying" (the stop rule) from "we ran out of clock
// with the queue still full". Those call for opposite next actions, so the run has to say which.
wfReturn.unexplored_heads = headQueue.map((h) => ({
  kernel: h.short_name || h.kernel,
  critical_path_share_pct: h.critical_path_share_pct,
  expected_gain_pct: +expectedGainPct(h).toFixed(2),
}));
wfReturn.skipped_heads = [...SKIP_HEADS];
wfReturn.stop_reason = !headQueue.length ? 'queue_exhausted'
  : tasksUsed >= BUDGET ? 'task_budget'
  : gatesUsed >= MAX_GATES ? 'gate_budget'
  : 'wall_clock_or_stop_rule';
if (wfReturn.unexplored_heads.length) {
  log0(`${wfReturn.unexplored_heads.length} head(s) never attempted (${wfReturn.stop_reason}): ` +
       wfReturn.unexplored_heads.map((h) => `${h.kernel} exp=${h.expected_gain_pct}%`).join(', '));
}
wfReturn.memory_conversion = proposeMemoryConversion(bankedMemory);

// ---------------------------------------------------------------------------
// Finalize / Report / Validate
// ---------------------------------------------------------------------------
phase('Finalize');
const fin = await safeAgentStub('integrator', 'finalize',
  { ...benchInputs(curConfig, 'finalize'), ACCEPTED: wfReturn.accepted, OVERLAY: curOverlay },
  'Assemble the bundle and RE-CHECK engagement on the assembled artifact.');
wfReturn.final = fin || null;
// The Finalize intro ORDERS an engagement re-check on the assembled artifact, and FINALIZE_SCHEMA
// asks for the answer -- but nothing read it, so a bundle whose overlay stopped engaging shipped
// looking identical to one that engaged. That is the same class of bug as the *_SCHEMA constants
// that were never passed to agent(): the agent does the work and the orchestrator discards it.
// This does not re-open the gate (the win was already measured); it refuses to present an artifact
// as validated when its own re-check says the overlay is no longer bound.
if (fin && fin.engagement_recheck && wfReturn.accepted.length) {
  const er = fin.engagement_recheck;
  if (er.forward === false || er.backward === false) {
    wfReturn.finalize_engagement_failed = true;
    log0('FINALIZE RE-CHECK FAILED: the assembled bundle does not engage on ' +
         `${er.forward === false ? 'forward' : ''}${er.forward === false && er.backward === false ? ' and ' : ''}` +
         `${er.backward === false ? 'backward' : ''} — the measured win belongs to a binding this ` +
         'bundle does not reproduce. Do NOT ship it on these numbers.');
  }
}

phase('Report');
wfReturn.report = await safeAgentStub('system_architect', 'report',
  { BASELINE: baseline, ACCEPTED: wfReturn.accepted, REJECTED: wfReturn.rejected || [],
    BANKED: wfReturn.banked_memory, DIAGNOSIS: wfReturn.config_diagnosis, BANDS: wfReturn.bands },
  'Write architect_report.md. Say explicitly which numbers are ours and which are the stack\'s.');

phase('Validate');
const val = await safeAgentStub('training_director', 'validate',
  { ...benchInputs(curConfig, 'validate'), BASELINE: baseline, FINAL: fin, OVERLAY: curOverlay,
    BAND: wfReturn.bands.trajectory_band_path },
  'Re-measure the accepted stack against the TRUE baseline yourself and arbitrate.');
wfReturn.validation = val || null;

// ANNOTATE the accepted records against the independent re-measurement -- never rewrite them.
// In the 7B run accepted[0].verdict.delta_pct stayed 7.853% after the Director had PROVEN that
// figure was a window-placement artifact and the real number was +4.54%. Anything reading
// .accepted[] got the disproven one. Overwriting the gate's measurement would destroy the audit
// trail (and the disagreement between the two is itself the finding), so the original stays byte
// for byte and gains a pointer to what superseded it.
if (val && typeof val.delta_pct === 'number' && wfReturn.accepted.length) {
  const tol = Math.max(Number(wfReturn.bands && wfReturn.bands.floor_pct) || 0, 0.15);
  // Only meaningful when ONE candidate was accepted: with several, per-candidate deltas are each
  // measured against a different carried reference and are not expected to equal the headline.
  const single = wfReturn.accepted.length === 1;
  for (const a of wfReturn.accepted) {
    const gated = a.verdict && a.verdict.delta_pct;
    a.validation = {
      validated_delta_pct: val.delta_pct,
      validation_status: val.validation_status || null,
      gate_delta_pct: typeof gated === 'number' ? gated : null,
      gate_delta_source: (a.verdict && a.verdict.delta_source) || null,
      superseded: single && typeof gated === 'number'
        && Math.abs(gated - val.delta_pct) > tol,
      note: single
        ? 'The Director re-measured this stack independently. Where `superseded` is true, quote '
          + 'validated_delta_pct; the gate figure is kept for audit, not for reporting.'
        : 'Several candidates were accepted, so per-candidate deltas are measured against different '
          + 'carried references and do NOT sum to the headline. Compare them individually, not to '
          + 'validated_delta_pct.',
    };
    if (a.validation.superseded) {
      log0(`NOTE: ${a.head} gated at ${gated}% (${a.verdict.delta_source}); independent ` +
           `re-measurement says ${val.delta_pct}%. Report the latter.`);
    }
  }
}

wfReturn.gates_used = gatesUsed;
wfReturn.tasks_used = tasksUsed;
wfReturn.kb_identity = kbIdentity(curConfig);
wfReturn.status = val && val.validation_status === 'confirmed' ? 'ok'
  : (wfReturn.accepted.length ? 'unconfirmed' : 'no_gain');
log0(`done: status=${wfReturn.status} accepted=${wfReturn.accepted.length} ` +
     `gates=${gatesUsed}/${MAX_GATES} tasks=${tasksUsed}/${BUDGET}`);

// NOT exported. The Workflow loader special-cases only the leading `export const meta` and parses
// everything after it as a function body, so a second top-level `export` is a syntax error and the
// run never launches. The gate suite reads this file as TEXT and regexes it, so it never imported
// this binding; the manifest stays as documentation of which pure functions are under test.
const _test = { noiseBandPct, legUsable, comparable, gateVerdict, expectedGainPct, stopRule,
                makeBarrier, proposeMemoryConversion, kbIdentity, want, FAST_MODE, DEEP_MODE };
void _test;

return wfReturn;
