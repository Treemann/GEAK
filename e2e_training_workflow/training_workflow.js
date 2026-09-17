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
const KERNEL_WF_SCRIPT = (A.kernel_workflow_dir || `${WORKFLOW_DIR}/../kernel_workflow`) + '/kernel_workflow.js';
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
const ACROSS_JOB_FLOOR_PCT = parseFloat(A.across_job_floor_pct != null ? A.across_job_floor_pct : 0.3);
const BAND_SAFETY = parseFloat(A.band_safety != null ? A.band_safety : 2.0);

// ===========================================================================
// Deterministic decision logic (unit-tested; no agent, no GPU)
// ===========================================================================

/** Noise band for a pair of legs, DERIVED from their own dispersion rather than assumed.
 *  Uses the robust (IQR) spread, because the raw spread of a clean run is dominated by isolated
 *  stalls: the same 25-step run measured 38.6% raw and 0.072% IQR. Raw would gate nothing. */
function noiseBandPct(ref, cand) {
  const iqr = Math.max(
    Number(ref && ref.worst_iqr_spread_pct) || 0,
    Number(cand && cand.worst_iqr_spread_pct) || 0);
  return Math.max(ACROSS_JOB_FLOOR_PCT, iqr * BAND_SAFETY);
}

/** Is a leg usable at all? A leg that is unusable must never become a reference. */
function legUsable(leg) {
  if (!leg || leg.status !== 'ok') return { ok: false, why: 'leg status != ok' };
  if (leg.hang_detected) return { ok: false, why: 'hang detected' };
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
function comparable(ref, cand) {
  if (!ref.config_digest || !cand.config_digest) return { ok: false, why: 'a leg carries no config digest' };
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
function gateVerdict({ ref, cand, l0, l1, mode }) {
  const ru = legUsable(ref), cu = legUsable(cand);
  if (!ru.ok) return { accepted: false, klass: 'unusable_ref', reason: `reference leg unusable: ${ru.why}` };
  if (!cu.ok) return { accepted: false, klass: 'unusable_cand', reason: `candidate leg unusable: ${cu.why}` };
  const cmp = comparable(ref, cand);
  if (!cmp.ok) return { accepted: false, klass: 'incomparable', reason: cmp.why };
  if (l0 && l0.pass === false) return { accepted: false, klass: 'numerics', reason: `L0 failed: ${l0.reason || 'op-level numerics'}` };
  if (l1 && l1.pass === false) return { accepted: false, klass: 'trajectory', reason: `L1 failed: ${l1.reason || 'loss/grad-norm trajectory'}` };

  const r = Number(ref.useful_tok_s_per_gpu), c = Number(cand.useful_tok_s_per_gpu);
  const deltaPct = ((c - r) / r) * 100;
  const band = noiseBandPct(ref, cand);
  const memDeltaPct = (Number(ref.peak_reserved_mb) > 0 && Number(cand.peak_reserved_mb) > 0)
    ? ((Number(cand.peak_reserved_mb) - Number(ref.peak_reserved_mb)) / Number(ref.peak_reserved_mb)) * 100
    : null;

  const base = { delta_pct: +deltaPct.toFixed(3), band_pct: +band.toFixed(3), mem_delta_pct: memDeltaPct == null ? null : +memDeltaPct.toFixed(2) };
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
  return safeAgent(roleAgent(role, phase, intro, inputs), { phase, label: `${role}:${phase}` });
}

async function safeAgent(prompt, opts, tries = 3) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      const r = await agent(prompt, opts);
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
const INTEGRATE_SCHEMA = obj({ ref: { type: 'object' }, cand: { type: 'object' },
  engagement: { type: 'object' }, overlay_path: { type: 'string' } }, ['ref', 'cand']);
const AUDIT_SCHEMA = obj({ l0: { type: 'object' }, l1: { type: 'object' },
  l2_recommended: { type: 'boolean' }, risk_class: { type: 'string' } }, ['l0', 'l1']);
const FINALIZE_SCHEMA = obj({ overlay: { type: 'string' }, patch: { type: 'string' },
  launch: { type: 'string' }, engagement_recheck: { type: 'object' } }, []);
const VALIDATE_SCHEMA = obj({ validation_status: { type: 'string' },
  baseline_useful_tok_s: { type: 'number' }, final_useful_tok_s: { type: 'number' },
  delta_pct: { type: 'number' } }, ['validation_status']);

const T0 = Date.now ? Date.now() : 0;
const outOfTime = () => (Date.now ? (Date.now() - T0) > TIME_BUDGET_MS * 0.6 : false);

// ---------------------------------------------------------------------------
// Setup — the TRUE baseline, measured twice so this run's bands are its own.
// ---------------------------------------------------------------------------
phase('Setup');
const setup = await safeAgentStub('training_director', 'setup',
  { ...benchInputs(curConfig, 'baseline'), REPEAT: 2 },
  'Pin the config IR, measure the TRUE baseline TWICE, and derive both bands from those two legs.');
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
let headQueue = ((strategy && strategy.head_candidates) || []).slice();
log0(`heads queued: ${headQueue.length}${strategy && strategy.stop ? ' (architect says stop)' : ''}`);

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

while (headQueue.length && tasksUsed < BUDGET && gatesUsed < MAX_GATES && !outOfTime()) {
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
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
      kernel_path: ext.task_dir, workflow_dir: (A.kernel_workflow_dir || `${WORKFLOW_DIR}/../kernel_workflow`),
      budget: parseInt(A.kernel_budget != null ? A.kernel_budget : 3, 10),
      gpu_ids: GPU_IDS.join(','),
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

  const v = gateVerdict({ ref: integ.ref, cand: integ.cand,
                          l0: audit && audit.l0, l1: audit && audit.l1,
                          mode: DEEP_MODE ? 'deep' : 'fast' });
  log0(`  gate: ${v.klass} — ${v.reason}`);
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
    if (r.strategy && r.strategy.head_candidates) headQueue = r.strategy.head_candidates.slice();
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
wfReturn.memory_conversion = proposeMemoryConversion(bankedMemory);

// ---------------------------------------------------------------------------
// Finalize / Report / Validate
// ---------------------------------------------------------------------------
phase('Finalize');
const fin = await safeAgentStub('integrator', 'finalize',
  { ...benchInputs(curConfig, 'finalize'), ACCEPTED: wfReturn.accepted, OVERLAY: curOverlay },
  'Assemble the bundle and RE-CHECK engagement on the assembled artifact.');
wfReturn.final = fin || null;

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

wfReturn.gates_used = gatesUsed;
wfReturn.tasks_used = tasksUsed;
wfReturn.kb_identity = kbIdentity(curConfig);
wfReturn.status = val && val.validation_status === 'confirmed' ? 'ok'
  : (wfReturn.accepted.length ? 'unconfirmed' : 'no_gain');
log0(`done: status=${wfReturn.status} accepted=${wfReturn.accepted.length} ` +
     `gates=${gatesUsed}/${MAX_GATES} tasks=${tasksUsed}/${BUDGET}`);

export const _test = { noiseBandPct, legUsable, comparable, gateVerdict, expectedGainPct, stopRule,
                       makeBarrier, proposeMemoryConversion, kbIdentity, want, FAST_MODE, DEEP_MODE };

return wfReturn;
