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
    { title: 'QualityGate', detail: 'optional: one-epoch multi-seed outcome test of the final stack vs REF (DECISIONS 4.22)' },
  ],
};

const A = args || {};
const log0 = (m) => log(`[training] ${m}`);

// A STACKED BASELINE (added 2026-10-14). Without it every run gates its heads against the BARE
// reference, and a kernel's increment on the accepted stack has to be re-measured by hand afterwards
// -- increments change when stacked (attn_fmha_ck_varlen: +4.27% bare, +5.25% on the five-kernel
// stack). `baseline_overlay` is a COMMAND that, given `--out <leg dir>`, installs/verifies the accepted
// stack and prints the one `--extra-env` line a bench_train.sh leg needs (scripts/stack_overlay_env.sh
// is the reference implementation: "<wf>/scripts/stack_overlay_env.sh --cond s7"). It becomes part of
// EVERY leg of the run -- baseline, profile, gate ref AND cand, finalize, validate -- so the reference
// the whole run measures against is the accepted stack, not the bare workload.
// `--stack` is appended (unless the caller gave one) so stack_overlay_env.sh refuses a condition of
// the OTHER stack: HF targets resolve inside the Megatron image and are never called, so a mismatched
// baseline would run the original code under the stack's name (review 2026-10-20).
const _BASE_RAW = String(A.baseline_overlay || '').trim();
const BASE_OVERLAY_CMD = !_BASE_RAW ? null
  : (/stack_overlay_env\.sh/.test(_BASE_RAW) && !/--stack\b/.test(_BASE_RAW)
     ? `${_BASE_RAW} --stack ${A.stack || 'megatron'}` : _BASE_RAW);
// THE OUTCOME GATE (added 2026-10-16). Every per-kernel gate here judges 30 steps. P0-4b showed that a
// stack of individually gated kernels can still make a training run blow up hundreds of steps later
// (late catastrophic episodes on 4/5 seeds) -- an event no 30-step instrument can see. So a stack
// is not "safe to train with" until it passes a one-epoch, multi-seed outcome test. It costs ~2 legs
// x 20 min per seed, so it is opt-in: `quality_gate: true`. It runs only when this run accepted a
// kernel or a non-packing config lever (otherwise the stack is unchanged and was judged before).
const QUALITY_GATE = String(A.quality_gate || '').toLowerCase() === 'true' || A.quality_gate === true;
const QUALITY_SEEDS = String(A.quality_seeds || '5567,6678,7789,8890,9901');
// The ONE pre-registered seed extension (DECISIONS 4.45): when the first verdict is INCONCLUSIVE under
// the non-inferiority rule, these 5 fresh seeds are added and the combined verdict is FINAL -- fixed
// here, before any result, so it is not open-ended optional stopping. 'none' disables it.
const QUALITY_EXTEND_SEEDS = String(A.quality_extend_seeds || '7013,8124,9235,1346,2457');
// Optional IN-RUN positive control arm (audit G4): e.g. 'ctrl70' (the schedule scaled x0.7). The yardstick
// is then measured in this run instead of the fixed P0-4d value, at +50% legs. Off by default.
// HF has NO fixed yardstick (the P0-4d +0.0206 is Megatron's): its pre-registered protocol (P0-4g) measures
// the yardstick in-run from ctrl70, so on hf_trl the control arm is the DEFAULT (2026-10-23 review F1).
const QUALITY_CTRL = A.quality_ctrl ? (String(A.quality_ctrl) === 'none' ? null : String(A.quality_ctrl))
  : ((A.stack || 'megatron') === 'hf_trl' ? 'ctrl70' : null);
// Default = THIS RUN's workload: the outcome test must judge the stack the run built, on the recipe it
// built it on. A fixed default (`_cosine_avg`, 2026-10-20..22) broke the moment a run's own lever was
// --ddp-average-in-collective on `_cosine`: the REF arm would already carry the lever and REF == cand.
// The protocol needs a SCHEDULED recipe (warmup + decay, DECISIONS 4.22-4.24); a constant-LR run must
// pass quality_workload explicitly, and the Director says so if it is not scheduled.
// HF's outcome-test recipe is the run's descriptor with a `_q` suffix (same recipe, LR_DECAY_ITERS = one
// epoch of the train19k split = 296 iterations, P0-4g); Megatron's is the run's own descriptor.
// REUSE an earlier outcome test's REF (and control) legs (review P3, DECISIONS 4.63): {prefix, ref_arm,
// ctrl_arm}. Only the cand arm is run; scripts/quality_reuse.py links the reused legs and must prove,
// seed by seed, that their realized argument tables equal the new cand legs' apart from the change under
// test. The seeds are the reused test's, so quality_seeds must be given explicitly. Optional keys
// (DECISIONS 4.75): `allow` / `allow_impl` = extra realized / implementation keys the reused REF may
// differ in beyond the accepted levers and overlay (pre-declared, e.g. "env:GEAK_NAN_FROM_GRAD_NORM"),
// `accept_unknown_harness: true` = accept legs that predate the harness record (reported, not silent).
const QUALITY_REUSE = (() => {
  const r = A.quality_reuse;
  if (!r) return null;
  const o = typeof r === 'string' ? JSON.parse(r) : r;
  if (!o.prefix || !o.ref_arm) throw new Error('quality_reuse needs {prefix, ref_arm[, ctrl_arm]}');
  if (!A.quality_seeds) throw new Error('quality_reuse: pass quality_seeds = the reused test\'s seeds');
  return o;
})();
const QUALITY_WORKLOAD = String(A.quality_workload ||
  ((A.stack || 'megatron') === 'hf_trl' ? String(A.workload || '').replace(/(_q)?\.env$/, '_q.env') : (A.workload || '')));

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
  'config_tuner', 'gate_verifier',
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
// Collapse `.` and `..` segments. The runtime only loads a child scriptPath it can verify as readable,
// and it did not accept `<wf>/../kernel_workflow/kernel_lane.js` even though the normalized path is a
// readable file: a packed-stack fast run (wf_3593b41a-adb) died at the first lane call after five
// agents of good work. Pure string handling -- the sandbox has no path module.
function normPath(p) {
  const out = [];
  for (const seg of String(p).split('/')) {
    if (seg === '' || seg === '.') continue;
    if (seg === '..') { out.pop(); continue; }
    out.push(seg);
  }
  return (String(p).startsWith('/') ? '/' : '') + out.join('/');
}
const KERNEL_WF_DIR = normPath(A.kernel_workflow_dir || `${WORKFLOW_DIR}/../kernel_workflow`);
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
//
// PER-STACK default (DECISIONS 4.54), measured on 867 true-null pairs (realized-arg-table equal,
// equal impl_digest, schema >= 3) with the stall-trimmed paired dispersion: Megatron bare at 0.25%
// gave 2 false wins (+0.289% and +0.256%, cross-day and same-day drift); 0.30% gives 0 at the same
// +0.5%-effect power (81%). HF (paired ratio, n=201) has worst null |delta| 0.142%: 0.25% holds.
// An explicit --across_job_floor_pct still wins.
const FLOOR_PCT = { aligned: 0.5, unaligned: 0.5 };
const STACK_FLOOR_PCT = { megatron: 0.30, hf_trl: 0.25 };
const ACROSS_JOB_FLOOR_PCT = parseFloat(A.across_job_floor_pct != null ? A.across_job_floor_pct
                                        : (STACK_FLOOR_PCT[STACK] != null ? STACK_FLOOR_PCT[STACK] : FLOOR_PCT.aligned));
// Above this band a sub-band delta is `inconclusive_noise`, not `no_win`. It used to be 4 x the floor,
// which tied "what counts as a resolvable measurement" to a knob tuned for false wins: raising the
// floor for safety silently stopped flagging noisy pairs. It is the size of effect the workflow wants
// to be able to see: a band wider than 1% cannot resolve a 1% candidate (DECISIONS 4.54).
const INCONCLUSIVE_BAND_PCT = parseFloat(A.inconclusive_band_pct != null ? A.inconclusive_band_pct : 1.0);
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
/** The dispersion of the window the DELTA was actually measured on.
 *
 * The band estimates the uncertainty of the delta, so it has to come from the same sample as the
 * delta. It did not. `compare_legs.py` forces both legs onto a COMMON window and reports the delta
 * there; the band was being computed from each leg's bench_summary, i.e. from each leg's OWN
 * window. Normally the two barely differ and nobody noticed.
 *
 * MEASURED, on the cfgrecompute reference leg of 2026-10-09, all 8 ranks: own window [5:30], 25
 * steps, IQR 0.096%. Common window [6:30], 24 steps, IQR 2.607%. ONE step of difference, 27x the
 * dispersion. The cause is visible in the sorted durations -- 19 values inside 11642-11660 and then
 * a cliff to 11952, 12006, 12023, 12041, 12113, 13143 -- so the third quartile lands exactly on the
 * edge of it and `v[3n/4]` picks the last tight value at n=25 and the first outlier at n=24.
 *
 * The consequence was a flipped verdict: band 0.475% (own window) made the attention-backend leg a
 * `regression` at -0.728%, where the window-matched band of 1.978% makes it `inconclusive_noise`.
 * The direction that matters is the other one -- the same mechanism would have called a +0.728%
 * non-win a WIN, which is what `legUsable` already warns about: "an artificially tight band is the
 * dangerous one: it accepts changes that are not there."
 *
 * Returns null when the comparison does not carry per-leg window dispersion, and the caller falls
 * back to the per-leg summaries rather than inventing a number.
 */
function windowMatchedLegs(spec) {
  const c = spec && spec.comparison;
  if (!c || !c.window) return null;
  // With the paired estimator the uncertainty is that of the RATIO series, one series of n steps:
  // its IQR already contains both legs' noise. Passing it as the ref leg with a zero-dispersion
  // cand makes noiseBandPct's sePair collapse to exactly se(ratio). Per-leg IQR here would be the
  // content-driven spread the paired estimator exists to remove (HF: ~25% per leg, 0.26% paired).
  const p = c.paired;
  if (p && p.same_batches === true && isFinite(Number(p.ratio_iqr_pct)) && Number(p.steps) > 0) {
    return { ref: { worst_iqr_spread_pct: Number(p.ratio_iqr_pct), min_steady_steps: Number(p.steps) },
             cand: { worst_iqr_spread_pct: 0, min_steady_steps: Number(p.steps) },
             source: 'paired ratio' };
  }
  const n = Number(c.window.steps_compared);
  const ri = c.ref && Number(c.ref.iqr_spread_pct);
  const ci = c.cand && Number(c.cand.iqr_spread_pct);
  if (!(isFinite(n) && n > 0 && isFinite(ri) && isFinite(ci))) return null;
  return { ref: { worst_iqr_spread_pct: ri, min_steady_steps: n },
           cand: { worst_iqr_spread_pct: ci, min_steady_steps: n } };
}

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
  // The probe's useful-token count must agree with the trainer's own (HF, audit HFC-7): the headline's numerator.
  const tc = leg.token_crosscheck;
  if (tc && (Number(tc.mismatched) > 0 || Number(tc.steps_checked) === 0)) {
    return { ok: false, why: `the probe's useful-token count disagrees with the trainer's num_items_in_batch on ` +
                             `${tc.mismatched} step(s) (${JSON.stringify((tc.examples || [])[0] || {})})` };
  }
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
  // EXCEPT on a VARIABLE-shape leg, where per-leg IQR measures the dataset, not the measurement. A
  // dynamically padded stack's step time follows content -- under DDP, the content on the slowest
  // rank -- so two IDENTICAL HF TRL legs sit at 24.8% and 25.4%, straddling this ceiling: the
  // guard would have declared a coin-flip of clean legs unusable. For those legs the same guard is
  // applied where it means something, to the PAIRED ratio series in gateVerdict (0.26% on an
  // identical pair). The case it was written for -- two runs merged into one sidecar -- is refused
  // at the source by load_rank regardless of step shape.
  if (leg.step_shape !== 'variable' && isFinite(iqr) && iqr > 25) {
    return { ok: false,
             why: `worst_iqr_spread_pct ${iqr}% — the step-to-step spread is the same order as the ` +
                  'step itself, so this leg cannot bound anything (a band derived from it exceeds ' +
                  'any delta a candidate could show). Suspect two runs merged into one probe dir.' };
  }
  // A leg whose CAPTURED identity disagrees with what the framework says it RAN is a measurement of a
  // config other than the one its digest names. It cannot be paired or compared, because the digest
  // is the only thing that says what it is. (identity_vs_realized is null for a stack that prints no
  // realized table -- that is "not checked", not "checked and clean", and is not refused here.)
  const ivr = leg.identity_vs_realized;
  if (ivr && Array.isArray(ivr.mismatches) && ivr.mismatches.length) {
    return { ok: false,
             why: 'identity records a config that did not run: ' +
                  ivr.mismatches.map((m) => `${m.key} captured ${m.captured}, ran ${m.realized}`).join('; ') };
  }
  const occ = (leg.occupancy_verdict || {}).verdict;
  if (occ === 'contaminated') return { ok: false, why: 'another process held a GPU during this leg' };
  if (occ === 'unknown') return { ok: false, why: 'isolation could not be established (rocm-smi unavailable)' };
  if (!(Number(leg.useful_tok_s_per_gpu) > 0)) return { ok: false, why: 'no useful-token throughput' };
  // FAIL CLOSED on missing evidence (audit 2026-10-22, G4). Every check above fired only on a field
  // that was PRESENT and bad, so a leg an agent trimmed down to {status, useful_tok_s_per_gpu} was
  // "clean": no occupancy sample, no hang check, no dispersion, and -- without step_shape -- an HF leg
  // treated as fixed-shape so per-leg medians could accept. Absence of a check is not a pass.
  const missing = [];
  if (occ !== 'clean') missing.push('occupancy_verdict.verdict=clean');
  if (leg.hang_detected !== false) missing.push('hang_detected=false');
  if (leg.dispersion_reliable !== true) missing.push('dispersion_reliable=true');
  if (!isFinite(Number(leg.worst_iqr_spread_pct)) || leg.worst_iqr_spread_pct === null) missing.push('worst_iqr_spread_pct');
  if (!(Number(leg.min_steady_steps) > 0)) missing.push('min_steady_steps');
  if (leg.step_shape !== 'fixed' && leg.step_shape !== 'variable') missing.push('step_shape');
  if (missing.length) {
    return { ok: false, why: `leg summary lacks the evidence a measurement needs (${missing.join(', ')}); ` +
                             'pass bench_summary.json through unmodified' };
  }
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
/** The delta the gate reads. Prefers the PAIRED estimator when it is valid.
 *
 * compare_legs reports two aligned deltas. `delta_pct` is median(useful)/median(dur) per leg, each
 * median taken independently; `paired.delta_pct` is from the median of per-step duration ratios.
 * They agree whenever every step has the same shape (Megatron pads every sample to SEQ), and they
 * do not when step time follows content: on a dynamically padded stack two IDENTICAL HF TRL runs
 * read +1.1523% apart under the per-leg medians -- a false win at any floor -- and +0.1344% paired.
 *
 * Adopted 2026-10-09 against two conditions stated before checking them: no worse on the null
 * control (694 clean same-identity pairs: 0 false wins either way on Megatron, 1 -> 0 on HF) and no
 * accepted win flipped (+12.37 -> +12.81, +12.28 -> +12.25, +7.43 -> +7.51, +1.15 -> +1.33; the
 * known-null cold-start leg -0.024 -> -0.004).
 *
 * VALID ONLY IF step i is the same batch on both legs, which compare_legs checks step by step as
 * `same_batches`. When it is false the paired number is meaningless and the per-leg one is used.
 */
function alignedDelta(integ) {
  const c = integ && integ.comparison;
  if (!c) return null;
  const p = c.paired;
  if (p && p.same_batches === true && typeof p.delta_pct === 'number') {
    return Object.assign({}, c, { delta_pct: p.delta_pct, estimator: 'paired per-step ratio',
                                  per_leg_delta_pct: c.delta_pct });
  }
  return typeof c.delta_pct === 'number' ? Object.assign({}, c, { estimator: 'per-leg medians' }) : null;
}

// Implementation keys a gate's reference and candidate are ALLOWED to differ in: exactly the thing
// under test. Anything else differing means the reference is not the run's reference -- e.g. a
// `_cosine` leg (no --ddp-average-in-collective) reused against a `_cosine_avg` candidate shares the
// config_digest (the flag is implementation, DECISIONS 1.10b) and would hand the candidate ~0.9%.
const KERNEL_IMPL_DELTA = ['GEAK_OVERLAY', 'GEAK_OVERLAY_SHA'];
// A lever's env knobs: {NAME: value} with shell-safe names / values only (they go on an --extra-env line).
const leverEnv = (env) => Object.fromEntries(Object.entries(env && typeof env === 'object' ? env : {})
  .filter(([k, v]) => /^[A-Z][A-Z0-9_]*$/.test(k) && /^[A-Za-z0-9_.,:+\-/]*$/.test(String(v)))
  .map(([k, v]) => [k, String(v)]));
const envString = (env) => Object.entries(env || {}).map(([k, v]) => `${k}=${v}`).join(' ');
const leverImplDelta = (flags) => (Array.isArray(flags) ? flags : String(flags || '').split(/\s+/))
  .filter((t) => typeof t === 'string' && t.startsWith('--')).map((t) => t.replace(/^--/, '').split('=')[0]);

// Accepted config levers ACCUMULATE. `flags` is an argv token list; a later lever that names the
// same flag replaces its value (argparse: last wins), every other flag is kept. Before 2026-10-22 an
// accepted lever REPLACED config_flags, so a second accepted lever silently dropped the first.
function mergeFlags(prev, added) {
  const parse = (toks) => {
    const out = []; const t = (Array.isArray(toks) ? toks : String(toks || '').split(/\s+/)).filter(Boolean);
    for (let i = 0; i < t.length; i++) {
      if (!String(t[i]).startsWith('--')) continue;
      const [k, inline] = String(t[i]).split('=');
      const vals = inline !== undefined ? [inline] : [];
      while (inline === undefined && i + 1 < t.length && !String(t[i + 1]).startsWith('--')) vals.push(String(t[++i]));
      out.push([k, vals]);
    }
    return out;
  };
  const m = new Map(parse(prev));
  for (const [k, v] of parse(added)) { m.delete(k); m.set(k, v); }
  return [...m].flatMap(([k, v]) => [k, ...v]);
}
const flagsString = (flags) => (flags || []).map((t) => (/\s/.test(t) ? `'${t}'` : t)).join(' ');

// ref.GEAK_OVERLAY_SHA must equal the carried reference's, and cand's must EXTEND it by >= 1 component
// ('+'-joined, one per manifest; rmsnorm ships two). Skipped only when refLeg carries no implementation.
// A gate's reference must BE the run's current stack: same workload (config_digest) AND the same implementation
// (impl_digest = carried overlays + accepted levers + env knobs). comparable() only relates ref to cand, so a
// gate measured on another workload, or without an accepted lever, was accepted (audit 2026-10-25 #11 P2/P10).
function refBinding(expected, ref) {
  if (!expected || !ref) return { ok: false, why: 'no carried reference to bind the gate to (fail closed)' };
  if (!expected.config_digest || !expected.impl_digest) {
    return { ok: false, why: 'the carried reference leg carries no config / impl digest (fail closed)' };
  }
  if (ref.config_digest !== expected.config_digest) {
    return { ok: false, why: `the gate's reference ran workload ${ref.config_digest}, not this run's ` +
                             `${expected.config_digest}: it is not the run's reference` };
  }
  const cr = contentAgree(expected, ref);
  if (!cr.ok) return { ok: false, why: `the gate's reference differs from the carried stack in content: ${cr.why}` };
  if (ref.impl_digest !== expected.impl_digest) {
    return { ok: false, why: `the gate's reference implementation ${ref.impl_digest} is not the carried stack's ` +
                             `${expected.impl_digest} (an accepted lever, env knob or overlay is missing or extra)` };
  }
  return { ok: true };
}

function overlayShaCheck(expectedRef, ref, cand) {
  const sha = (l) => String(((l || {}).implementation || {}).GEAK_OVERLAY_SHA || '');
  if (!expectedRef || !expectedRef.implementation) {
    return { ok: false, why: 'the carried reference leg has no implementation dict, so the gate cannot check ' +
                             'that ref IS the carried stack (schemas require it; fail closed)' };
  }
  const e = sha(expectedRef), r = sha(ref), c = sha(cand);
  const ep = e ? e.split('+') : [], cp = c ? c.split('+') : [];
  if (r !== e) return { ok: false, why: `reference overlay stack ${r || '(none)'} is not the carried stack ${e || '(none)'}` };
  if (!(cp.length > ep.length && ep.every((p, i) => cp[i] === p))) {
    return { ok: false, why: `candidate overlay stack ${c || '(none)'} does not extend the carried stack ${e || '(none)'}` };
  }
  return { ok: true };
}

// The independent recomputation (audit ORCH-1). `ver` is the gate_verifier's output: verify_gate.py run
// on EVERY ref x cand leg the measuring agent took. Accept only if the REPORTED pair is a win on the
// recomputed numbers, its engagement is proven from the proof files (kernel gates), and NO pairing is
// a regression. Legs are matched by directory basename (agents mix host and container paths).
// The LAST TWO path components (agents mix host /home/... and container /work/... paths): a basename
// collided on the HF `<gate>/ref`, `<gate>/cand` layout (2026-10-23 review LEGBASE-COLLISION).
const legBase = (l) => {
  const d = typeof l === 'string' ? l : (l && (l.dir || l.probe_dir));
  return d ? String(d).replace(/\/+$/, '').replace(/\/probe$/, '').split('/').slice(-2).join('/') : null;
};
// `opts.expected` = {refs, cands}: the dirs the verifier was ASKED to cross (audit 2026-10-25 ORCH-A2: a
// reply holding only the primary pair silently disabled the regression veto). `opts.needL1`: the pair
// must carry the verifier's own L1 recomputation, passed (G-3).
function verifyDecision(ver, refLegOrDir, candLegOrDir, needEngagement, opts) {
  const o = opts || {};
  const pairs = (ver && Array.isArray(ver.pairs)) ? ver.pairs : null;
  if (!pairs || !pairs.length) return { ok: false, why: 'the gate verifier returned nothing: no independent recomputation, no accept' };
  const rb = legBase(refLegOrDir), cb = legBase(candLegOrDir);
  const prim = pairs.find((p) => legBase(p.ref_dir) === rb && legBase(p.cand_dir) === cb);
  const table = pairs.map((p) => ({ ref: legBase(p.ref_dir), cand: legBase(p.cand_dir),
    klass: p.verdict && p.verdict.klass, delta_pct: p.verdict && p.verdict.delta_pct,
    band_pct: p.verdict && p.verdict.band_pct, error: p.error || null,
    l1: p.l1 ? p.l1.pass : undefined }));
  if (!prim) return { ok: false, table, why: `the verifier did not recompute the reported pair ${rb} -> ${cb}` };
  if (o.expected) {
    const want = [];
    for (const r of o.expected.refs || []) for (const c of o.expected.cands || []) want.push([legBase(r), legBase(c)]);
    const miss = want.filter(([r, c]) => !pairs.some((p) => legBase(p.ref_dir) === r && legBase(p.cand_dir) === c));
    if (miss.length) {
      return { ok: false, table, why: `the verifier skipped ${miss.length} of ${want.length} requested pairing(s) ` +
        `(${miss.map(([r, c]) => `${r}->${c}`).join(', ')}): a skipped pairing can hide a regression` };
    }
    const bad = pairs.filter((p) => p !== prim && (p.error || !p.verdict || p.verdict.error));
    if (bad.length) {
      return { ok: false, table, why: `${bad.length} requested pairing(s) could not be judged ` +
        `(${bad.map((p) => `${legBase(p.ref_dir)}->${legBase(p.cand_dir)}`).join(', ')}): a pairing that was ` +
        'not judged can hide a regression' };
    }
  }
  if (prim.error || !prim.verdict || prim.verdict.error) {
    return { ok: false, table, why: `the verifier could not judge the reported pair: ${prim.error || (prim.verdict || {}).error}` };
  }
  if (prim.verdict.klass !== 'win') {
    return { ok: false, table, why: `recomputed from the legs, the reported pair is ${prim.verdict.klass}: ${prim.verdict.reason}` };
  }
  if (needEngagement) {
    const e = prim.engagement || {};
    if (e.forward !== true || e.backward !== true) {
      return { ok: false, table, why: `the proof files do not show engagement on every rank (forward=${e.forward}, ` +
                                      `backward=${e.backward}${e.why ? `: ${e.why}` : ''})` };
    }
  }
  // Every overlay the reference runs must still run in the candidate (DECISIONS 4.55: --rope-fusion on
  // s8 turned the accepted rope overlay into a 100% fallback while the pair still measured). FAIL
  // CLOSED: a primary pair without the check (an agent trimmed it) cannot show it was done.
  if (!prim.carried || typeof prim.carried.ok !== 'boolean') {
    return { ok: false, table, why: 'the verifier output for the reported pair lacks the carried-overlay check' };
  }
  const lost = pairs.filter((p) => p.carried && p.carried.ok === false);
  if (lost.length) {
    return { ok: false, table, why: `the candidate switches off accepted overlay(s) the reference runs ` +
      `(${[...new Set(lost.flatMap((p) => p.carried.lost || []))].join(', ')}: no longer engaged / falling back)` };
  }
  const reg = pairs.filter((p) => p.verdict && p.verdict.klass === 'regression');
  if (reg.length) {
    return { ok: false, table, why: `${reg.length} measured pairing(s) are a REGRESSION on the recomputed numbers ` +
                                    `(${reg.map((p) => `${legBase(p.ref_dir)}->${legBase(p.cand_dir)}`).join(', ')})` };
  }
  if (o.needL1 && !(prim.l1 && prim.l1.pass === true)) {
    return { ok: false, table, why: prim.l1
      ? `the verifier's own L1 recomputation (${(prim.l1.recomputed || {}).kind || '?'}) did not pass: ` +
        `${(prim.l1.failures || []).join('; ') || prim.l1.error || prim.l1.reason || 'pass=' + prim.l1.pass}`
      : 'the verifier output lacks its L1 recomputation for the reported pair' };
  }
  // Not best-of-k (audit MS-6): with several measured pairings the Integrator picked the primary, which raised
  // the null p95 from 0.11% to 0.21%. The MEDIAN delta over all judged pairings must clear the band too.
  const ds = pairs.filter((p) => p.verdict && typeof p.verdict.delta_pct === 'number').map((p) => p.verdict.delta_pct)
    .sort((x, y) => x - y);
  if (ds.length > 1) {
    const med = ds.length % 2 ? ds[(ds.length - 1) / 2] : (ds[ds.length / 2 - 1] + ds[ds.length / 2]) / 2;
    if (!(med > prim.verdict.band_pct)) {
      return { ok: false, table, why: `the reported pair wins (+${prim.verdict.delta_pct}%) but the median over all ` +
        `${ds.length} measured pairings is ${med.toFixed(3)}% <= band ${prim.verdict.band_pct}%: best-of-k, not a win` };
    }
  }
  return { ok: true, table, delta_pct: prim.verdict.delta_pct, band_pct: prim.verdict.band_pct,
           l1: prim.l1 ? { pass: prim.l1.pass, recomputed: prim.l1.recomputed || null } : null };
}

// The L1 kind and the --l0-strictly-better floor, derived by CODE from the L0 evidence (audit 2026-10-25
// G-3: the Auditor chose them, and --l0-strictly-better widens the step-1 floor 8x). `overlay` (0.05%)
// only when L0 shows the forward output bit-identical to the incumbent; `kernel_swap` (0.25%) otherwise;
// strictly-better (2.0%) only with a per-output error table in which EVERY output is strictly closer to
// the fp32 reference than the incumbent's.
function l1PlanFromL0(l0) {
  const po = l0 && l0.per_output && typeof l0.per_output === 'object' ? Object.values(l0.per_output) : [];
  const better = po.length > 0 && po.every((e) => e && typeof e.cand_err === 'number' &&
    typeof e.base_err === 'number' && e.cand_err < e.base_err);
  const overlay = !!(l0 && l0.out_bit_identical === true);
  return { kind: overlay ? 'overlay' : 'kernel_swap', l0_strictly_better: !overlay && better };
}

function contentAgree(a, b) {
  const ca = a && a.content, cb = b && b.content;
  if ((ca == null) !== (cb == null)) {
    return { ok: false, stale: true, why: 'one leg predates the content identity record (overlay trees / input ' +
                                         'fingerprints): re-measure it with the current probe' };
  }
  if (!ca) return { ok: true };
  const ia = ca.inputs || {}, ib = cb.inputs || {};
  const din = [...new Set([...Object.keys(ia), ...Object.keys(ib)])].filter((k) => ia[k] !== ib[k]);
  if (din.length) {
    return { ok: false, why: `the legs read different INPUT contents (${din.join(', ')}): same path, other bytes ` +
                             '(a re-written pack / checkpoint / tokenizer) -- not the same workload' };
  }
  const sa = String(((a.implementation || {}).GEAK_OVERLAY_SHA) || ''), sb = String(((b.implementation || {}).GEAK_OVERLAY_SHA) || '');
  const ta = String(ca.overlay_tree || ''), tb = String(cb.overlay_tree || '');
  const map = (s, t) => { const ss = s ? s.split('+') : [], tt = t ? t.split('+') : []; const m = {};
                          ss.forEach((x, i) => { m[x] = tt[i]; }); return m; };
  const ma = map(sa, ta), mb = map(sb, tb);
  const bad = Object.keys(ma).filter((k) => k in mb && ma[k] !== mb[k]);
  if (bad.length) {
    return { ok: false, why: `overlay component(s) ${bad.join(', ')} have the same impl.py but different files ` +
                             '(kernel sources / GEMM tables / helpers): the legs did not run the same kernel bytes' };
  }
  return { ok: true };
}

function comparable(ref, cand, allowedImpl) {
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
    // Schema 5 is the same kind of boundary: it captures --lr-decay-iters and the DDP reduction
    // flags, which v4 sidecars never recorded (the header keeps argv_hash, not argv).
    const RECAPTURE = { 3: 'dropout, lr, seed, architecture',
                        4: 'attention-backend, comm-overlap and recompute flags',
                        5: 'lr-decay-iters, per-token loss, bf16 grad reduction and the DDP flags',
                        6: 'the RMSNorm epsilon, RoPE fusion (moved to implementation) and CE fusion' };
    const crossed = Object.keys(RECAPTURE).map(Number)
      .filter((b) => Math.min(rs, cs) < b && b <= Math.max(rs, cs));
    return { ok: false, stale: true,
             why: `digest schema differs (${rs} vs ${cs}) — one summary predates a change in how the ` +
                  (crossed.length
                    ? `digest is computed. Schema ${crossed.join(' and ')} captures fields ` +
                      `(${crossed.map((b) => RECAPTURE[b]).join('; ')}) that the older sidecar never ` +
                      'recorded, so this pair must be RE-MEASURED; re-summarising cannot recover them.'
                    : 'digest is computed. Re-summarise both from their probe dirs, then compare.') };
  }
  if (ref.ranks_agree_on_argv !== true || cand.ranks_agree_on_argv !== true) {
    return { ok: false, why: 'ranks within one leg ran different argv (or the leg does not say they agree)' };
  }
  // Checked whenever the caller names what may differ. FAIL CLOSED: a leg without its implementation
  // dict cannot be checked, and the gate schemas require it (an Integrator once trimmed it away).
  if (allowedImpl && !(ref.implementation && cand.implementation)) {
    return { ok: false, why: 'a gate leg carries no implementation dict, so the gate cannot check that ' +
                             'ref and cand differ only in the change under test' };
  }
  if (allowedImpl) {
    const ri = ref.implementation, ci = cand.implementation;
    const extra = [...new Set([...Object.keys(ri), ...Object.keys(ci)])]
      .filter((k) => ri[k] !== ci[k] && !allowedImpl.includes(k)).sort();
    if (extra.length) {
      return { ok: false, why: `implementation differs beyond the change under test (${extra.map((k) =>
        `${k}: ${ri[k] === undefined ? '-' : ri[k]} vs ${ci[k] === undefined ? '-' : ci[k]}`).join('; ')}): ` +
        'the reference is not this run\'s reference, so the delta would include that difference' };
    }
  }
  // RUNTIME CODE that changes the computation without touching argv or env (audit OVL-6, DECISIONS
  // 4.59): rocm_compat patches Megatron at runtime (it changed the loss on packed data), overlay_train
  // installs every overlay, and the image carries every library kernel. Recorded per leg since v5 as
  // report-only provenance; a pair that ran different versions of any of them is not one A/B. Checked
  // here rather than digested, so no schema bump re-measures every leg. FAIL CLOSED from schema 6 (every
  // v6 leg records it); older legs cannot carry it and are refused by the schema check above anyway
  // unless both are old.
  // hf_sft.py: the HF training loop (recorded on HF legs only, DECISIONS 4.67); absent on both Megatron legs.
  // rocm_compat.py patches MEGATRON; the HF loop never imports it, so a rocm_compat change must not make an HF
  // pair incomparable (audit ID-10 / HFC-5). An HF leg is one whose harness records hf_sft.py.
  const hfPair = !!((ref.harness || {})['hf_sft.py'] || (cand.harness || {})['hf_sft.py']);
  const HARNESS_KEYS = hfPair ? ['overlay_train.py', 'hf_sft.py', 'image_id']
                              : ['rocm_compat.py', 'overlay_train.py', 'hf_sft.py', 'image_id'];
  const rh = ref.harness, ch = cand.harness;
  if (!(rh && ch)) {
    if (Math.max(Number(ref.digest_schema) || 0, Number(cand.digest_schema) || 0) >= 6) {
      return { ok: false, why: 'a schema-6 leg carries no `harness` record (runtime code + image), so the ' +
                               'gate cannot check that both legs ran the same rocm_compat / overlay_train / image' };
    }
  } else {
    const hd = HARNESS_KEYS.filter((k) => (rh[k] || null) !== (ch[k] || null));
    if (hd.length) {
      return { ok: false, why: `the legs ran different runtime code (${hd.map((k) => `${k}: ${rh[k] || '-'} vs ` +
        `${ch[k] || '-'}`).join('; ')}): re-measure the reference with the current harness` };
    }
  }
  // EVERY flag / env knob the snapshot does not classify (audit 2026-10-25 ID-1: --fp8-format, --optimizer,
  // NVTE_CK_USES_BWD_V3, PYTORCH_TUNABLEOP_ENABLED, NCCL_ALGO ... moved neither digest). probe_summarize
  // lists them as `unmapped`; a pair that differs in one is refused, whatever allowedImpl says -- classify
  // the key (workload or implementation) first. FAIL CLOSED on a mixed pair: a leg with the full capture
  // against one without cannot show the older leg did not differ. Two legs that both predate it (`null`)
  // are judged as before (historic pairs; every leg measured from now on carries it).
  const ru = ref.unmapped, cu = cand.unmapped;
  if ((ru == null) !== (cu == null)) {
    return { ok: false, stale: true, why: 'one leg predates the full argv/env capture (unmapped keys unknown): ' +
                                         're-measure it with the current probe' };
  }
  if (ref.unmapped_ranks_agree === false || cand.unmapped_ranks_agree === false) {
    return { ok: false, why: 'ranks within one leg ran different unclassified flags / env' };
  }
  if (ru && cu) {
    const ud = [];
    for (const part of ['argv', 'env']) {
      const a = ru[part] || {}, b = cu[part] || {};
      for (const k of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort()) {
        if (a[k] !== b[k]) ud.push(`${part === 'env' ? 'env ' : '--'}${k}: ${a[k] === undefined ? '-' : a[k]} vs ${b[k] === undefined ? '-' : b[k]}`);
      }
    }
    if (ud.length) {
      return { ok: false, why: `the legs differ in UNCLASSIFIED setting(s) (${ud.join('; ')}): classify each as ` +
        'workload or implementation (train_probe identity lists / probe_summarize IMPL_KEYS) before gating on it' };
    }
  }
  // CONTENT identity (audit 2026-10-25 #9 / #10): overlay components with the same impl.py SHA must have the
  // same directory tree (kernel sources, GEMM tables, helpers), and the data / tokenizer / checkpoint contents
  // must match. One leg with the record and one without cannot show they agree (stale).
  const contentCheck = contentAgree(ref, cand);
  if (!contentCheck.ok) return contentCheck;
  // The digest check runs LAST and says so (`digestOnly`), so ConfigSweep can re-gate a lever that is
  // supposed to move identity WITHOUT skipping the rank and implementation checks above. Before
  // 2026-10-22 it ran first and ConfigSweep re-gated on any 'incomparable' (audit G2/ORCH-6).
  // Number formatting is not identity (audit ID-9: '1e-5' vs '1.0e-5' digested differently): when the digests
  // differ, compare the workload identities with numerically equal values treated as equal.
  const numEq = (x, y) => x === y || (x != null && y != null && x !== '' && y !== '' &&
                                       Number.isFinite(Number(x)) && Number(x) === Number(y));
  const wiR = ref.workload_identity, wiC = cand.workload_identity;
  const sameWl = wiR && wiC && [...new Set([...Object.keys(wiR), ...Object.keys(wiC)])].every((k) => numEq(wiR[k], wiC[k]));
  if (ref.config_digest !== cand.config_digest && !sameWl) {
    return { ok: false, digestOnly: true, why: `config digest differs (${ref.config_digest} vs ${cand.config_digest}): ` +
      `these legs did not measure the same workload` };
  }
  return { ok: true };
}

/** The gate. Throughput is the only thing that can ACCEPT in fast mode; memory is banked (there is
 *  no config knob to convert it with until ConfigSweep exists). L0/L1 can only ever REJECT. */
function gateVerdict(spec) {
  const { ref, cand, l0, l1, mode, allowedImpl } = spec;
  const ru = legUsable(ref), cu = legUsable(cand);
  if (!ru.ok) return { accepted: false, klass: ru.fatal ? 'blocked' : 'unusable_ref',
                       fatal: !!ru.fatal, reason: `reference leg unusable: ${ru.why}` };
  if (!cu.ok) return { accepted: false, klass: cu.fatal ? 'blocked' : 'unusable_cand',
                       fatal: !!cu.fatal, reason: `candidate leg unusable: ${cu.why}` };
  const cmp = comparable(ref, cand, allowedImpl);
  if (!cmp.ok) return { accepted: false, klass: cmp.stale ? 'stale_summary' : 'incomparable',
                        digestOnly: !!cmp.digestOnly, reason: cmp.why };
  if (l0 && l0.pass === false) return { accepted: false, klass: 'numerics', reason: `L0 failed: ${l0.reason || 'op-level numerics'}` };
  if (l1 && l1.pass === false) return { accepted: false, klass: 'trajectory', reason: `L1 failed: ${l1.reason || 'loss/grad-norm trajectory'}` };

  const r = Number(ref.useful_tok_s_per_gpu), c = Number(cand.useful_tok_s_per_gpu);
  // Prefer the aligned delta when the Integrator supplied one; fall back to the raw ratio only when
  // it did not, and say so, because the raw ratio carries the window artifact.
  const aligned = alignedDelta(spec);
  const deltaPct = aligned ? aligned.delta_pct : ((c - r) / r) * 100;
  // The band must be estimated on the window the delta was measured on; see windowMatchedLegs.
  const wm = aligned ? windowMatchedLegs(spec) : null;
  const band = noiseBandPct(wm ? wm.ref : ref, wm ? wm.cand : cand, !!aligned);
  const memDeltaPct = (Number(ref.peak_reserved_mb) > 0 && Number(cand.peak_reserved_mb) > 0)
    ? ((Number(cand.peak_reserved_mb) - Number(ref.peak_reserved_mb)) / Number(ref.peak_reserved_mb)) * 100
    : null;

  const base = { delta_pct: +deltaPct.toFixed(3), band_pct: +band.toFixed(3),
    delta_source: aligned ? 'window-aligned' : 'raw ratio (NOT window-aligned — includes sample-composition drift)',
    window_artifact_pct: aligned ? aligned.window_artifact_pct : null,
    // Which sample the band came from. Not cosmetic: the two can differ by 27x on a leg whose
    // steps straddle a quartile cliff, and a reader comparing bands across runs needs to know
    // whether they were estimated on comparable samples.
    band_window: wm ? (wm.source === 'paired ratio' ? 'paired per-step ratio (same sample as the delta)'
                                                    : 'common (window-matched to the delta)')
                    : 'per-leg summaries (no window dispersion in comparison)',
    estimator: aligned ? aligned.estimator : 'raw ratio',
    mem_delta_pct: memDeltaPct == null ? null : +memDeltaPct.toFixed(2) };
  // A raw ratio may REJECT but may never ACCEPT. Two IDENTICAL-config 7B legs (true delta zero by
  // construction) gave unaligned +4.5925% against aligned -0.0588% at a 0.914% band: window placement
  // alone manufactures a 5x-band "win". No floor can bound that -- the artifact scales with the data's
  // +-25% useful_tokens swing, not with a constant. The asymmetry is deliberate and matches L0/L1: a
  // false REJECT costs one candidate, which can be re-gated once the Integrator reports an aligned
  // comparison; a false ACCEPT installs the candidate as the reference every later delta compounds on
  // (see `refLeg = integ.cand` below) and puts the artifact in the headline permanently.
  // A VARIABLE-shape leg (dynamic padding) has no usable per-leg estimator: two identical HF legs
  // read +1.15% apart on per-leg medians. Only the paired per-step ratio is a measurement there, so
  // without one -- or with one whose ratio series is itself wildly dispersed -- it may not ACCEPT.
  const variable = ref.step_shape === 'variable' || cand.step_shape === 'variable';
  const pairedOk = aligned && aligned.estimator === 'paired per-step ratio' &&
                   !(Number(spec.comparison.paired.ratio_iqr_pct) > 25);
  if (variable && deltaPct > band && !pairedOk) {
    return { accepted: false, klass: 'no_aligned_delta',
             reason: `+${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%, but a variable-step-shape ` +
                     'leg (dynamic padding) can only be judged by a PAIRED per-step comparison with ' +
                     'same_batches true; per-leg medians put two identical runs 1.15% apart', ...base };
  }
  if (deltaPct > band && !aligned) {
    return { accepted: false, klass: 'no_aligned_delta',
             reason: `+${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%, but this is a raw ratio: ` +
                     're-report with scripts/compare_legs.py before it can be a win', ...base };
  }
  // The median must not be the only witness (audit 2026-10-25 #6): a kernel that is slower on SOME steps
  // wins by median and loses in total time. The 10%-trimmed mean of the same per-step ratios (stalls cut,
  // a systematic minority of slow steps kept) must ALSO clear the band. On 42 same-identity null pairs the
  // median-minus-trimmed-mean gap is p95 0.62 / p99 0.94 pts, so this is a second, independent estimate.
  const tm = spec.comparison && spec.comparison.paired ? Number(spec.comparison.paired.tmean_delta_pct) : NaN;
  if (deltaPct > band && pairedOk && Number.isFinite(tm) && tm <= band) {
    return { accepted: false, klass: 'estimators_disagree', tmean_delta_pct: tm,
             reason: `median +${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%, but the 10%-trimmed mean of the same ` +
                     `per-step ratios is ${tm.toFixed(2)}% (<= band): either slower on a minority of steps, or a ` +
                     'leg too short for a mean-type estimate -- re-gate on longer legs', ...base };
  }
  // The band is an SE over the paired sample; fewer than 12 steps cannot carry it (audit MS-2/8).
  const pSteps = spec.comparison && spec.comparison.paired ? Number(spec.comparison.paired.steps) : NaN;
  if (deltaPct > band && pairedOk && Number.isFinite(pSteps) && pSteps < 12) {
    return { accepted: false, klass: 'inconclusive_noise',
             reason: `+${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%, but over only ${pSteps} paired steps (< 12)`, ...base };
  }
  if (deltaPct > band) return { accepted: true, klass: 'win', reason: `+${deltaPct.toFixed(2)}% > band ${band.toFixed(2)}%`,
                                ...(Number.isFinite(tm) ? { tmean_delta_pct: tm } : {}), ...base };
  if (deltaPct < -band) return { accepted: false, klass: 'regression', reason: `${deltaPct.toFixed(2)}% < -band`, ...base };
  // In band. Memory is a CURRENCY, not a metric: it counts only once a config change spends it.
  // 5%, not 2%: on 2,403 TRUE-NULL pairs (same config + impl digest) peak_reserved differs by up to 4.0%
  // at the 99th percentile and 78 pairs exceed 2%; the gate-based null control then called 24/400
  // identical pairs 'banked_memory' (2026-10-22). A freed-memory claim must clear the null spread.
  // PER STEP SHAPE since 2026-10-23 (DECISIONS 4.54), on 867 clean true-null pairs: Megatron (fixed
  // shape) |mem delta| p99 1.22%, max 1.33%, none > 2%; HF (variable) max 4.97%, 40/201 > 2%. One 5%
  // threshold made Megatron unable to bank a real 2-4% saving and left HF with no margin at all.
  const memThreshPct = variable ? 6.0 : 2.0;
  if (memDeltaPct != null && memDeltaPct < -memThreshPct) {
    return {
      accepted: false, klass: 'banked_memory', banked_memory_pct: +(-memDeltaPct).toFixed(2),
      reason: mode === 'deep'
        ? `throughput in band; ${(-memDeltaPct).toFixed(1)}% memory freed -> hand to proposeMemoryConversion()`
        : `throughput in band; ${(-memDeltaPct).toFixed(1)}% memory freed, but fast mode has no config knob to spend it -> banked`,
      ...base,
    };
  }
  // "We measured and it is not there" and "we could not measure" must not look alike. When the band
  // is set by the LEGS' dispersion rather than by this workload's reproducibility, a sub-band delta
  // says nothing about the candidate. Measured 2026-09-28: a candidate at +1.147% was recorded as
  // `no_win` against a 2.467% band -- 16.4x the floor -- because the reference leg had an IQR of
  // 3.324% where this workload normally gives 0.05-0.11%. On quiet legs that band would have been
  // the 0.15% floor and +1.147% a clear win. The legs were the problem and the record blamed the
  // candidate.
  //
  // The threshold is INCONCLUSIVE_BAND_PCT (an absolute resolution), no longer 4x the floor. It can
  // only ever fire where the verdict was already a non-win, so it cannot turn a win into a doubt.
  const bandDominatedByLegNoise = band > INCONCLUSIVE_BAND_PCT && band > ACROSS_JOB_FLOOR_PCT;
  if (bandDominatedByLegNoise) {
    return { accepted: false, klass: 'inconclusive_noise', remeasure: true,
             reason: `${deltaPct.toFixed(2)}% is inside a ${band.toFixed(2)}% band, wider than the ` +
                     `${INCONCLUSIVE_BAND_PCT}% this gate must resolve (floor ${ACROSS_JOB_FLOOR_PCT}%) — ` +
                     `it was set by leg dispersion, not by the workload. This is NOT evidence against the ` +
                     `candidate. Re-measure with LONGER legs (the band shrinks as 1/sqrt(steps)); if the ` +
                     `paired report shows many steps_dropped, the stalls are periodic (GC, recompiles) -- ` +
                     `run with --gc / a persistent TRITON_CACHE_DIR rather than simply repeating the pair.`,
             ...base };
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
// Declared HERE, with the other carried state, not inside KernelTrack. ConfigSweep runs BEFORE
// that block and passes it to the Tuner, and a `let` read above its declaration is a temporal
// dead zone error that costs a launch -- this one cost 3 agents and 24 minutes.
let curOverlay = [];                  // the carried accepted overlays, IN ORDER (each composes on the previous)
const bankedMemory = [];              // hook 5 input

function benchInputs(config, tag) {   // hook 2: config is ALWAYS an explicit argument
  return { WORKLOAD, STACK, NPROC, ITERS, EXP_ROOT, CONFIG: config, TAG: tag,
           GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts`, GPU_IDS: GPU_IDS.join(','),
           ...(BASE_OVERLAY_CMD ? { BASE_OVERLAY_CMD } : {}) };
}

async function reprofile(config, round, overlay) { // hook 3: independently callable
  requireRole('profiler');
  // The CARRIED overlay goes with it. Without it the "re-profile after an acceptance" profiled the
  // BARE stack again, so the re-ranked queue never reflected the shares the accepted kernel moved
  // (found 2026-10-14: the 10-13 Architect report said "measured here at their bare-reference shares").
  return safeAgentStub('profiler', 'reprofile', { ...benchInputs(config, `reprofile_${round}`),
                                                  ...(overlay && overlay.length ? { CARRIED_OVERLAY: overlay } : {}) });
}
async function restrategize(config, profile) {      // hook 3
  requireRole('system_architect');
  return safeAgentStub('system_architect', 'strategize', { CONFIG: config, PROFILE: profile });
}
function proposeMemoryConversion(banked) {          // hook 5: no-op in fast
  if (!DEEP_MODE) return { proposed: false, reason: 'fast mode has no ConfigSweep to spend memory with', banked };
  return { proposed: false, reason: banked.length ? 'ConfigSweep ran but banked memory was not spent' : 'no memory was banked to spend', banked };
}
function kbIdentity(config) {                       // hook 6: config hash keys every record
  return { stack: STACK, workload: WORKLOAD, gpus: NPROC, config_digest: (config && config.digest) || null };
}

// ===========================================================================
// Agent plumbing (mirrors e2e_workflow.js)
// ===========================================================================
function roleAgent(role, phase, intro, inputs) {
  const cf = inputs && inputs.CONFIG && inputs.CONFIG.config_flags;
  const ce = envString(inputs && inputs.CONFIG && inputs.CONFIG.config_env);
  const carried = (cf && cf.length) || ce ? `

## CARRIED CONFIG LEVERS (accepted earlier in this run: ${(inputs.CONFIG.accepted_levers || []).join(', ')})
EVERY leg of the current stack you launch passes them: bench_train.sh ... --extra-train-args "${flagsString(cf || [])}"${ce
  ? ` --extra-env "${ce}"` : ''}
(append to any other extra args / env, e.g. after the overlay's GEAK_OVERLAY=... line). A reference leg of a gate is a leg of the current stack too, so it
carries them as well. Only a leg explicitly defined as the run's TRUE/bare baseline (Validate's
baseline, the QualityGate REF arm) omits them. A leg without them is not this run's configuration:
check the lever's keys in bench_summary.json (\`config\` / \`implementation\`) before quoting it.` : '';
  return `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}${carried}

## MEASUREMENT INVARIANT (do not violate — every number in this run must be comparable)
Take EVERY throughput number through ${WORKFLOW_DIR}/scripts/bench_train.sh with the SAME
--workload (${WORKLOAD}), --stack ${STACK}, --nproc ${NPROC} and --gpus ${GPU_IDS.join(',')}. The headline is
bench_summary.json:useful_tok_s_per_gpu (non-pad, loss-unmasked tokens), NEVER the stack's own
tokens/s: on this workload 71.6% of tokens are padding, so the nominal figure overstates useful work
by ~3.5x. Never change the parallel mapping, global batch size, sequence length or data order between
a reference leg and a candidate leg — that changes the workload identity and makes the delta fiction.
A leg whose occupancy_verdict is not "clean", or whose status is not "ok", is NOT a measurement:
re-run it, do not report it.${BASE_OVERLAY_CMD ? `

## STACKED BASELINE (this run's reference is NOT the bare workload)
Every bench_train.sh leg you take -- baseline, profile, gate reference AND candidate, finalize,
validate -- runs on top of the accepted stack. For each leg, BEFORE launching it, run
    ${BASE_OVERLAY_CMD} --out <that leg's --out dir>
and pass its stdout (one line, GEAK_OVERLAY=... TORCH_EXTENSIONS_DIR=...) as --extra-env. It installs
or re-verifies the gated bytes and exits non-zero on any mismatch: then STOP, do not measure. A
candidate overlay (and any CARRIED_OVERLAY) is APPENDED to that GEAK_OVERLAY list, after the stack's
manifests -- never instead of them, and keep its TORCH_EXTENSIONS_DIR unless your overlay needs its own.
Check the stack's per-overlay engagement proofs under the leg dir (engagement_fused.json.rank*:
fallback 0) as well as your own: a stacked leg whose base overlays silently fell back is a bare leg.` : ''}

## INPUTS
${JSON.stringify(inputs, null, 2)}`;
}

// opts.phase must be a meta.phases title, or the agent lands in an ad-hoc group and the declared
// phase shows "not started" in /workflows. The role PHASE (lower-case) still goes into the prompt.
const PROGRESS_GROUP = {
  setup: 'Setup', reprofile: 'Profile', strategize: 'Strategize', config: 'ConfigSweep',
  extract: 'KernelTrack', integrate: 'KernelTrack', gate: 'KernelTrack',
  finalize: 'Finalize', report: 'Report', validate: 'Validate', quality: 'QualityGate',
  verify: 'KernelTrack',
};

async function safeAgentStub(role, phase, inputs, intro = '', extra = {}) {
  requireRole(role);   // unreachable until roles land; keeps the call sites honest meanwhile
  const opts = { phase: PROGRESS_GROUP[phase] || phase, label: extra.label || `${role}:${phase}` };
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
  // schema 2: config_sweep is {levers:[...], accepted}, not a single record.
  schema_version: 2,
  mode: DEEP_MODE ? 'deep' : 'fast', stack: STACK, workload: WORKLOAD,
  gates_used: 0, tasks_used: 0, accepted: [], banked_memory: [], status: 'incomplete',
  // Every delta in this result is relative to THIS reference. null = the bare workload.
  baseline_overlay: BASE_OVERLAY_CMD,
};

if (!WORKLOAD) {
  throw new Error('args.workload is required (a workloads/*.env descriptor). There is no default: ' +
    'silently benching the wrong target is worse than failing.');
}

log0(`mode=${wfReturn.mode} stack=${STACK} workload=${WORKLOAD} nproc=${NPROC} ` +
     `budget=${BUDGET} max_gates=${MAX_GATES} band_floor=${ACROSS_JOB_FLOOR_PCT}%`);
log0(BASE_OVERLAY_CMD
  ? `baseline_overlay: every leg runs ON TOP of [${BASE_OVERLAY_CMD}] -- deltas are increments on that stack`
  : 'baseline_overlay: none -- deltas are against the BARE workload');
log0(`phases: ${['setup', 'profile', 'strategize', want('config') ? 'config' : null, 'kernel', 'final']
  .filter(Boolean).join(' -> ')}`);

// ---------------------------------------------------------------------------
// Schemas — every agent returns structured JSON; judgement is the agent's, control flow is here.
// ---------------------------------------------------------------------------
const obj = (props, req) => ({ type: 'object', properties: props, required: req || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
// Every field legUsable/comparable read is REQUIRED (audit 2026-10-22: they failed open on absence).
// Setup's baseline legs too: the review found real Directors returned trimmed legs (occupancy as a
// string, no step_shape, no implementation), so the reference selection and the carried-stack check
// silently degraded on the first gate.
const LEG_REQUIRED = ['implementation', 'workload_identity', 'config_digest', 'digest_schema', 'status',
  'useful_tok_s_per_gpu', 'occupancy_verdict', 'hang_detected', 'dispersion_reliable', 'worst_iqr_spread_pct',
  'min_steady_steps', 'step_shape', 'ranks_agree_on_argv', 'probe_dir'];
const LEG = obj({ implementation: { type: 'object' }, workload_identity: { type: 'object' },
  config_digest: { type: 'string' }, digest_schema: { type: 'number' }, status: { type: 'string' },
  useful_tok_s_per_gpu: { type: 'number' }, occupancy_verdict: { type: 'object' },
  hang_detected: { type: 'boolean' }, dispersion_reliable: { type: 'boolean' },
  worst_iqr_spread_pct: { type: 'number' }, min_steady_steps: { type: 'number' },
  step_shape: { type: 'string', enum: ['fixed', 'variable'] }, ranks_agree_on_argv: { type: 'boolean' },
  probe_dir: { type: 'string' },
  // runtime code + image the leg ran (comparable() requires it from schema 6; copy it verbatim)
  harness: { type: ['object', 'null'] },
  // every flag / env knob nobody classified (probe_summarize, audit ID-1); null = predates the capture.
  // Not required (historic legs); the gate_verifier recomputes it from the probe either way.
  unmapped: { type: ['object', 'null'] }, unmapped_ranks_agree: { type: ['boolean', 'null'] },
  // overlay directory trees + input fingerprints (audit #9 / #10); null = predates the record
  content: { type: ['object', 'null'] } },
  LEG_REQUIRED);
const PROOF = obj({ forward: { type: 'boolean' }, backward: { type: 'boolean' } }, ['forward', 'backward']);
// Finalize runs even when no kernel was accepted; then there is nothing to re-check and null is honest.
const PROOF_OR_NULL = obj({ forward: { type: ['boolean', 'null'] }, backward: { type: ['boolean', 'null'] } },
                          ['forward', 'backward']);
const VERDICT = obj({ pass: { type: 'boolean' }, reason: { type: 'string' } }, ['pass']);
const SETUP_SCHEMA = obj({ baseline_legs: { type: 'array', items: LEG, minItems: 1 }, across_job_band_pct: { type: 'number' },
  trajectory_band_path: { type: 'string' }, config_digest: { type: 'string' },
  warmup_steps: { type: 'number' }, pad_fraction_pct: { type: 'number' } }, ['baseline_legs']);
const PROFILE_SCHEMA = obj({ buckets: { type: 'object' }, top_n: arrObj,
  config_diagnosis: { type: 'object' }, rank_skew_pct: { type: 'number' } }, ['top_n']);
// Each head must carry what the ranking and the stop rule read (audit ORCH-A10: a missing attainable_speedup
// silently ranked a head at 0 expected gain).
const HEAD = obj({ critical_path_share_pct: { type: 'number' }, attainable_speedup: { type: 'number' } },
                 ['critical_path_share_pct', 'attainable_speedup']);
const STRATEGY_SCHEMA = obj({ head_candidates: { type: 'array', items: HEAD }, stop: { type: 'boolean' },
  reasoning: { type: 'string' } }, ['head_candidates']);
const EXTRACT_SCHEMA = obj({ task_dir: { type: 'string' }, op: { type: 'string' },
  has_backward: { type: 'boolean' }, autograd: { type: 'boolean' }, engagement: { type: 'object' } },
  ['task_dir']);
// `comparison` is REQUIRED, not optional: without it the gate cannot accept at all (see
// no_aligned_delta), so a return value that lacks it has wasted two job launches. Requiring it makes
// the model layer retry for it instead of the orchestrator discovering the omission after the A/B.
// A gate leg must carry bench_summary's `implementation` dict, or comparable() cannot check that the
// reference differs from the candidate only in what is under test (DECISIONS 1.10b). The first real
// run's Integrator trimmed it away (kept impl_digest only), which would have skipped the check silently.
const INTEGRATE_SCHEMA = obj({ ref: LEG, cand: LEG,
  comparison: { type: 'object' }, engagement: PROOF,
  overlay_path: { type: 'string' },
  // EVERY leg dir measured for this gate (host paths), not just the reported pair (audit ORCH-1).
  legs_measured: obj({ refs: { type: 'array', items: { type: 'string' }, minItems: 1 },
                       cands: { type: 'array', items: { type: 'string' }, minItems: 1 } }, ['refs', 'cands']) },
  ['ref', 'cand', 'comparison', 'engagement', 'overlay_path', 'legs_measured']);
// main_grad_contract.sh's own fields, copied verbatim (DECISIONS 4.56): `applicable` is decided by the
// script from the task's rebind target, never by the agent.
const MGC = obj({ applicable: { type: 'boolean' }, passed: { type: ['boolean', 'null'] },
  head: { type: ['string', 'null'] }, why: { type: ['string', 'null'] }, result: { type: ['string', 'null'] } },
  ['applicable', 'passed']);
// The L0 oracle's provenance (DECISIONS 4.60): a pre-v2 TEMPLATE oracle silently skips a missing output.
const L0_VERDICT = obj({ pass: { type: 'boolean' }, reason: { type: 'string' },
  // L1 plan evidence (l1PlanFromL0): the forward `out` bit-identical to the incumbent, and the per-output
  // {cand_err, base_err} vs the fp32 reference, copied from the oracle's output.
  out_bit_identical: { type: ['boolean', 'null'] }, per_output: { type: ['object', 'null'] },
  oracle: obj({ kind: { type: 'string', enum: ['template', 'custom'] },
    template_version: { type: ['number', 'null'] }, scores_every_output: { type: 'boolean' },
    task_dir: { type: 'string' } }, ['kind', 'template_version', 'scores_every_output', 'task_dir']) }, ['pass', 'oracle']);
const AUDIT_SCHEMA = obj({ l0: L0_VERDICT, l1: VERDICT, main_grad_contract: MGC,
  l2_recommended: { type: 'boolean' }, risk_class: { type: 'string' },
  // Path of the KB card written back to the kernel layer on a reject, or null on a pass. Declared so
  // the Auditor is asked for it; a rejection the KB never hears about is re-derived from zero next run.
  kb_rejection_card: { type: ['string', 'null'] } }, ['l0', 'l1', 'main_grad_contract']);
const FINALIZE_SCHEMA = obj({ overlay: { type: 'string' }, patch: { type: 'string' },
  launch: { type: 'string' }, engagement_recheck: PROOF_OR_NULL }, ['engagement_recheck']);
const CONFIG_SCHEMA = obj({ lever: { type: 'string' }, flags_added: { type: 'array' },
  // env knobs the lever sets besides its flags, NAME -> value (audit 2026-10-25 item 13: a flag+env lever
  // such as --no-check-for-nan-in-loss-and-grad + GEAK_NAN_FROM_GRAD_NORM=1 could not be accepted or carried)
  env_added: { type: ['object', 'null'] },
  packing_gate: { type: 'object' }, comparison_tokens: { type: 'object' },
  ceiling_pct: { type: 'number' }, ref: LEG, cand: LEG, l1: VERDICT,
  comparison: { type: 'object' }, config_digest_delta: { type: 'object' },
  peak_reserved_mb: { type: 'object' }, memory_conversion: { type: 'object' } },
  ['lever', 'ref', 'cand', 'comparison']);

// Validate is accepted only with its numbers and its L1 (audit ORCH-7: a bare {validation_status:
// 'confirmed'} string used to make the run 'ok').
const VALIDATE_SCHEMA = obj({
  validation_status: { type: 'string', enum: ['confirmed', 'disputed', 'unusable'] },
  baseline_useful_tok_s: { type: 'number' }, final_useful_tok_s: { type: 'number' },
  delta_pct: { type: 'number' }, band_pct: { type: 'number' }, estimator: { type: 'string' },
  wall_delta_pct: { type: ['number', 'null'] }, l1: VERDICT, legs: { type: 'array' },
  // the headline pair's leg DIRECTORIES: the gate_verifier recomputes it (audit 2026-10-25 ORCH-A5)
  ref_dir: { type: 'string' }, final_dir: { type: 'string' } },
  ['validation_status', 'baseline_useful_tok_s', 'final_useful_tok_s', 'delta_pct', 'band_pct', 'estimator', 'l1', 'legs',
   'ref_dir', 'final_dir']);

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
const QUALITY_SCHEMA = obj({ cleared: { type: 'boolean' }, quality: { type: 'string' },
  stability: { type: 'string' }, legs_prefix: { type: 'string' }, notes: { type: 'string' },
  // quality_reuse.py check's JSON, verbatim, when QUALITY_REUSE was given
  reuse_check: { type: ['object', 'null'] } },
  ['cleared', 'quality', 'stability']);
const VERIFY_SCHEMA = obj({ pairs: { type: 'array', minItems: 1, items: obj({
  ref_dir: { type: 'string' }, cand_dir: { type: 'string' }, verdict: { type: ['object', 'null'] },
  packing: { type: ['object', 'null'] },   // verify_gate --kind packing
  reuse_check: { type: ['object', 'null'] },   // quality_reuse.py check (KIND quality_reuse)
  l0_recomputed: { type: ['object', 'null'] },   // l0_verify.sh (KIND l0)
  quality_verdict: { type: ['object', 'null'] },   // quality_ab_final.py's JSON line (KIND quality)
  l1: { type: ['object', 'null'] },   // verify_gate --l1-kind: the recomputed trajectory_gate verdict
  engagement: { type: ['object', 'null'] }, comparison: { type: ['object', 'null'] },
  carried: { type: ['object', 'null'] },
  error: { type: ['string', 'null'] } }, ['ref_dir', 'cand_dir']) },
  // mgc_applicability.py on the kernel's task + overlay (MGC_TASK / MGC_IMPL given), verbatim
  mgc_applicability: { type: ['object', 'null'] } }, ['pairs']);

const PHASE_SCHEMA = {
  quality: QUALITY_SCHEMA,
  setup: SETUP_SCHEMA, profile: PROFILE_SCHEMA, reprofile: PROFILE_SCHEMA,
  strategize: STRATEGY_SCHEMA, extract: EXTRACT_SCHEMA, integrate: INTEGRATE_SCHEMA,
  gate: AUDIT_SCHEMA, finalize: FINALIZE_SCHEMA, validate: VALIDATE_SCHEMA,
  config: CONFIG_SCHEMA, verify: VERIFY_SCHEMA,
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
// (Since DECISIONS 3.4b the envelope never votes at any leg count; the three legs still give the
// sign-balance null range three pairs instead of one, and feed the across-job throughput band.)
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
  'If BASELINE_DIRS is given, verify and REUSE those legs instead of measuring -- but only legs that carry ' +
  'the full argv/env capture (`unmapped` not null when re-summarised; DECISIONS 4.78); re-measure the others. ' +
  (BASE_OVERLAY_CMD
    ? 'With BASE_OVERLAY_CMD set, a reused leg is only valid if it ran the SAME stack: its probe ' +
      'identity must carry the same GEAK_OVERLAY manifest basenames and GEAK_OVERLAY_SHA as a leg ' +
      'taken through BASE_OVERLAY_CMD, and its per-overlay proofs must show fallback 0. A bare leg ' +
      'is NOT a baseline for a stacked run. ' : '') +
  'The L1 band MUST pass its own null control; if it does not, say so rather than using it. ' +
  'across_job_band_pct is the worst |paired.delta_pct| over the leg pairs whenever compare_legs.py ' +
  'reports paired.same_batches=true -- the estimator the gate quotes -- not the aligned median ratio, ' +
  'which isolated host stalls inflate (0.93% aligned vs 0.07% paired on the 2026-10-15 stacked legs).');
if (!setup || !(setup.baseline_legs || []).length) {
  throw new Error('Setup produced no baseline. Every later number is a ratio against it, so there ' +
    'is nothing to continue with — a run that proceeds here reports deltas against nothing.');
}
const baselineLegs = setup.baseline_legs;
// PICK THE QUIETEST LEG, do not take [0]. The baseline becomes `refLeg`, so its dispersion enters
// the 4*se_pair term of EVERY later band. Measured on 2026-09-28: a reused leg with
// worst_iqr_spread_pct 1.842% -- against 0.05-0.11% for its two siblings -- priced the gate band at
// 2.467%, and a candidate measuring +1.147% was recorded as `no_win`. The Director had already
// flagged that leg ("NOISY LEG ... do NOT use legA as the gate reference leg") and the orchestrator
// took it anyway, because it took the first one. A warning nothing consumes is not a warning.
const byQuiet = baselineLegs
  .filter((l) => legUsable(l).ok)
  .sort((a, b) => (Number(a.worst_iqr_spread_pct) || Infinity) - (Number(b.worst_iqr_spread_pct) || Infinity)
                  || (Number(b.min_steady_steps) || 0) - (Number(a.min_steady_steps) || 0));
if (!byQuiet.length) {
  // FAIL CLOSED: falling back to baselineLegs[0] silently disabled the quietest-leg selection AND the
  // carried-stack check on the first gate (review 2026-10-22). No usable baseline -> no reference.
  throw new Error('Setup returned no USABLE baseline leg: ' + baselineLegs.map((l) =>
    `${(l.dir || l.probe_dir || '?').split('/').slice(-2).join('/')}: ${legUsable(l).why}`).join('; '));
}
const baseline = byQuiet[0];
// A STACKED baseline must carry the stack (audit #11 P8): with BASE_OVERLAY_CMD set and no overlay in the
// baseline, every later gate measured "head vs bare" while the record said "on the stack" -- exactly the
// confound baseline_overlay exists to remove (attn +4.27% bare vs +5.25% stacked, 4.17).
if (BASE_OVERLAY_CMD && !((baseline.implementation || {}).GEAK_OVERLAY_SHA)) {
  throw new Error(`baseline_overlay is set but the baseline leg ${(baseline.dir || baseline.probe_dir || '?')} runs no ` +
                  'overlay (no GEAK_OVERLAY_SHA): Setup did not measure the stacked baseline');
}
// The CARRIED reference: the baseline, then the cand leg of every accepted config lever, then of every
// accepted kernel. Every gate's ref is bound to it (refBinding).
let stackRef = baseline;
if (baselineLegs.length > 1) {
  log0('baseline legs by dispersion: ' + baselineLegs
    .map((l) => `${(l.dir || l.leg || '?').split('/').pop()}=${l.worst_iqr_spread_pct}%`).join(' ') +
    ` -> reference is ${(baseline.dir || baseline.leg || '?').split('/').pop()}`);
  // A leg far noisier than its siblings is usually a symptom (host stall, contention), not a property
  // of the workload. Say so: it stays usable, but the operator should know the band it would imply.
  const iqrs = baselineLegs.map((l) => Number(l.worst_iqr_spread_pct)).filter((x) => isFinite(x) && x > 0);
  const med = iqrs.slice().sort((a, b) => a - b)[Math.floor(iqrs.length / 2)];
  for (const l of baselineLegs) {
    const q = Number(l.worst_iqr_spread_pct);
    if (isFinite(q) && med > 0 && q > 5 * med) {
      log0(`WARNING: baseline leg ${(l.dir || l.leg || '?').split('/').pop()} has IQR ${q}%, ` +
           `${(q / med).toFixed(0)}x the median of its own set (${med}%) — it is not the reference, ` +
           `but treat any band that uses it as inflated`);
    }
  }
}
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
// The SAME floor term noiseBandPct applies (max(floor, BAND_SAFETY x measured)): this was half of it, so the stop
// rule and the verifier's --floor ran on a band half as wide as the gate's (audit MS-5).
const runFloor = Math.max(ACROSS_JOB_FLOOR_PCT, measuredFloor * BAND_SAFETY);
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
  const prof = await reprofile(config, round, curOverlay);
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
// The queue is ordered HERE, by the same expectedGainPct the stop rule uses, not left to the order the
// Architect happened to emit: a rule that only a prompt follows is a rule the next prompt can forget.
// Stable: equal gains keep the Architect's order, which carries its rationale.
const rankQueue = (q) => q.map((h, i) => [h, i])
  .sort((a, b) => (expectedGainPct(b[0]) - expectedGainPct(a[0])) || (a[1] - b[1]))
  .map(([h]) => h);
let headQueue = rankQueue(applySkips(((strategy && strategy.head_candidates) || []).slice()));
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
// CLASS B levers change WHAT a step contains, so step i is not the same batch on both legs and BOTH
// trajectory instruments (the step-1 probe and the step-aligned delta) become invalid. L0 does not
// apply either -- no kernel changed. Such a change would rest on throughput ALONE, and this layer
// does not accept a number with no correctness gate behind it. Enforced here rather than left to the
// role card: a rule only a prompt knows is a rule the next prompt can forget.
// PACKING is the one Class-B lever with a gate, and the reason is specific: it is a LAYOUT change
// whose risks are structural (attention isolation, position restarts, which tokens are targets) and
// checkable at step 1 on identical weights -- scripts/pack_gate.py. GBS, MBS and sequence length
// change the optimizer's trajectory, which is what they are FOR; no equivalence check can call them
// correct, so they stay refused.
const GATEABLE_CLASS_B = ['packing', 'pack'];
const PACKING_MIN_GAIN_PCT = 5.0;   // cross-dataset comparison, no step pairing; measured effect 266%
// By FLAG NAME, not substring of free text (audit: `txt.includes('pack')` made any flag or path that
// merely contained the letters a Class-B lever). A bare `packing` token is how the Tuner names the
// pack transform; a changed --train-data-path is a data change, hence Class B too.
const CLASS_B_FLAG = { 'global-batch-size': 'global-batch-size', 'micro-batch-size': 'micro-batch-size',
  'seq-length': 'seq-length', 'sequence-length': 'seq-length', 'sft-packing': 'packing', 'packing': 'packing',
  'train-data-path': 'packing' };
function classOfLever(flags) {
  const toks = (Array.isArray(flags) ? flags : String(flags || '').split(/\s+/)).map((t) => String(t).toLowerCase());
  const names = toks.filter((t) => t.startsWith('--')).map((t) => t.replace(/^--/, '').split('=')[0]);
  const bare = toks.filter((t) => !t.startsWith('--') && CLASS_B_FLAG[t] === 'packing');
  const hitName = names.find((n) => CLASS_B_FLAG[n]) || (bare.length ? 'packing' : null);
  if (!hitName) return { cls: 'A', key: null };
  const key = CLASS_B_FLAG[hitName];
  return { cls: 'B', key, gateable: GATEABLE_CLASS_B.includes(key) };
}

// args.config_levers names the levers to measure, comma-separated, e.g.
// "recompute_granularity,attention_backend". Empty means one pass where the Tuner chooses from the
// Tier-0 pack itself, which is the original behaviour.
//
// Measuring several levers in ONE run is not just a time saving. Every lever is then judged against
// the SAME baseline and the SAME band, so the verdicts are comparable to each other; split across
// two runs they would carry two baselines and two bands, and the difference between two levers
// would be confounded with the difference between two days.
// The carried config after an accepted lever: flags MERGED (not replaced) and the digests taken from
// the accepted cand leg, which ran exactly this config. curConfig.digest was never refreshed before,
// so every later leg, report and KB identity carried the pre-lever digest.
function acceptLever(config, sweep) {
  const cand = sweep.cand || {};
  return { ...config, config_flags: mergeFlags(config.config_flags, sweep.flags_added),
           config_env: { ...(config.config_env || {}), ...leverEnv(sweep.env_added) },
           digest: cand.config_digest || config.digest, impl_digest: cand.impl_digest || config.impl_digest,
           accepted_levers: [...(config.accepted_levers || []), sweep.lever] };
}
const CONFIG_LEVERS = String(A.config_levers || '').split(',').map((x) => x.trim()).filter(Boolean);

if (want('config')) {
  phase('ConfigSweep');
  const sweepRecords = [];
  // One pass with `null` reproduces the single-shot behaviour exactly.
  for (const wantLever of (CONFIG_LEVERS.length ? CONFIG_LEVERS : [null])) {
    if (gatesUsed >= MAX_GATES) {
      log0(`ConfigSweep: gate budget spent (${gatesUsed}/${MAX_GATES}); ` +
           `${CONFIG_LEVERS.length - sweepRecords.length} lever(s) not measured`);
      break;
    }
    const directive = wantLever
      ? `Propose the "${wantLever}" lever SPECIFICALLY -- it is named by the caller and the other ` +
        'Tier-0 levers are out of scope for this pass. State its ceiling from the diagnosis, and ' +
        'measure it exactly as a kernel candidate is measured. If you believe this lever cannot ' +
        'pay on this workload, measure it anyway and say so in notes: a measured non-win on a ' +
        'named lever is the result being asked for, not a failure.'
      : 'Propose ONE Class-A lever from the Tier-0 pack, state its ceiling from the diagnosis, and ' +
        'measure it exactly as a kernel candidate is measured.';
    let sweep = await safeAgentStub('config_tuner', 'config',
      { ...benchInputs(curConfig, 'config'), DIAGNOSIS: wfReturn.config_diagnosis,
        BANDS: wfReturn.bands, CARRIED_OVERLAY: curOverlay, REQUESTED_LEVER: wantLever, REF_EXPECTED: stackRef,
        ALREADY_MEASURED: sweepRecords.map((r) => r.lever) },
      directive, { label: wantLever ? `config_tuner:${wantLever}` : undefined });
    // Same rule as a kernel gate (DECISIONS 4.88): median and trimmed mean disagree -> ONE re-measure of THIS
    // lever on legs twice as long. Needed most here: the NaN-from-grad-norm lever (+0.6%, validated) read trimmed
    // means of +0.33 / -1.44 / -0.58 on three 30-step pairs.
    if (sweep && sweep.ref && sweep.cand && !classOfLever(sweep.flags_added).cls.startsWith('B') && gatesUsed < MAX_GATES) {
      const se = leverEnv(sweep.env_added);
      const pre = gateVerdict({ ref: sweep.ref, cand: sweep.cand, comparison: sweep.comparison,
        l1: { pass: true }, mode: 'deep',
        allowedImpl: [...leverImplDelta(sweep.flags_added), ...Object.keys(se).map((k) => `env:${k}`),
                      ...Object.keys(se).filter((k) => k.startsWith('NVTE_'))] });
      if (pre.klass === 'estimators_disagree') {
        const longIters = Math.max(60, 2 * ITERS);
        log0(`ConfigSweep ${sweep.lever}: ${pre.reason}; re-measuring once at ${longIters} iterations`);
        gatesUsed += 1;
        const s2 = await safeAgentStub('config_tuner', 'config',
          { ...benchInputs(curConfig, 'config_long'), ITERS: longIters, DIAGNOSIS: wfReturn.config_diagnosis,
            BANDS: wfReturn.bands, CARRIED_OVERLAY: curOverlay, REQUESTED_LEVER: sweep.lever, REF_EXPECTED: stackRef,
            ALREADY_MEASURED: sweepRecords.map((r) => r.lever),
            REMEASURE_OF: { lever: sweep.lever, flags_added: sweep.flags_added, env_added: sweep.env_added || null } },
          `RE-MEASURE the "${sweep.lever}" lever exactly as before (same flags_added / env_added) on legs twice as ` +
          'long (ITERS is doubled): the first pair\'s median and trimmed-mean estimates disagreed.',
          { label: `config_tuner:${sweep.lever}:long` });
        if (s2 && s2.ref && s2.cand) sweep = Object.assign(s2, { remeasure_of: { first_comparison: sweep.comparison } });
      }
    }
  if (!sweep) {
    log0('ConfigSweep produced nothing; continuing to the kernel track.');
  } else {
    const kind = classOfLever(sweep.flags_added);
    if (kind.cls === 'B' && kind.gateable) {
      gatesUsed += 1;
      const pg = sweep.packing_gate || null;
      const ct = sweep.comparison_tokens || null;
      let d = ct && typeof ct.delta_pct === 'number' ? ct.delta_pct : null;
      let verdict;
      // INDEPENDENT RECOMPUTATION (audit 2026-10-25 ORCH-A4): this path used to accept on the Tuner's own
      // packing_gate.pass and comparison_tokens.delta_pct, with no legUsable / no verifier -- an
      // unusable (failed / contaminated / hung) leg pair could accept the largest lever there is.
      const ru = legUsable(sweep.ref || {}), cu = legUsable(sweep.cand || {});
      let pv = null;
      if (pg && pg.pass === true && ru.ok && cu.ok) {
        const ver = await safeAgentStub('gate_verifier', 'verify',
          { REF_DIRS: [sweep.ref.dir || sweep.ref.probe_dir].filter(Boolean),
            CAND_DIRS: [sweep.cand.dir || sweep.cand.probe_dir].filter(Boolean),
            KIND: 'packing', PACK_GATE_RESULT: pg.result || null, NPROC,
            GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
          'Run verify_gate.py --kind packing on the pairing and return its output verbatim. You are a script runner.');
        const pair = ver && Array.isArray(ver.pairs) ? ver.pairs[0] : null;
        pv = pair && pair.packing ? pair.packing : null;
        d = pv && typeof pv.delta_pct === 'number' ? pv.delta_pct : null;   // the RECOMPUTED delta decides
      }
      if (!ru.ok || !cu.ok) {
        verdict = { accepted: false, klass: 'unusable_' + (ru.ok ? 'cand' : 'ref'),
                    reason: `packing leg unusable: ${ru.ok ? cu.why : ru.why}` };
      } else if (pg && pg.pass === true && !(pv && pv.pack_gate_pass === true && pv.data_matches === true &&
                                             pv.legs_usable === true)) {
        verdict = { accepted: false, klass: 'verifier_refused',
                    reason: pv ? `the recomputation does not confirm the packing gate: ${JSON.stringify(pv.why || [])}`
                               : 'no independent recomputation of the packing gate (gate_verifier returned nothing)' };
      } else if (!pg || pg.pass !== true) {
        verdict = { accepted: false, klass: pg ? 'packing_gate_failed' : 'no_correctness_gate',
                    reason: pg ? `scripts/pack_gate.py verdict ${pg.verdict}: ${JSON.stringify(pg.checks || {})}`
                               : 'no packing_gate result (scripts/pack_gate.py): a packed dataset may not be ' +
                                 'accepted on throughput alone' };
      } else if (d == null) {
        verdict = { accepted: false, klass: 'no_aligned_delta',
                    reason: 'no equal-useful-token comparison (scripts/compare_legs_tokens.py)' };
      } else {
        const win = d > PACKING_MIN_GAIN_PCT;
        verdict = { accepted: win, klass: win ? 'win' : 'no_win', delta_pct: d, pack_gate_result: pg.result || null,
                    reported_delta_pct: ct && typeof ct.delta_pct === 'number' ? ct.delta_pct : null,
                    band_pct: PACKING_MIN_GAIN_PCT, delta_source: 'equal useful tokens (Class B)',
                    reason: `${d.toFixed(2)}% ${win ? '>' : '<='} ${PACKING_MIN_GAIN_PCT}% on an equal-useful-` +
                            'token basis; packing gate PASS' };
      }
      sweepRecords.push({ lever: sweep.lever, requested: wantLever, flags_added: sweep.flags_added,
        class: 'B (gated: packing)', verdict, packing_gate: pg,
        // Recorded, not decided here: GBS counts PACKS after packing, so samples-per-update changes
        // unless the caller divides it by the packing ratio. A recipe choice, made visibly.
        recipe_note: 'global-batch-size now counts packs; divide by conversations_per_pack to keep ' +
                     'samples per optimizer step' });
      log0(`ConfigSweep ${sweep.lever}: ${verdict.klass} (${verdict.reason})`);
      if (verdict.accepted) { curConfig = acceptLever(curConfig, sweep); stackRef = sweep.cand; }
    } else if (kind.cls === 'B') {
      // Refuse, do not silently downgrade: a Class-B result looks exactly like a Class-A one.
      sweepRecords.push({ refused: true, lever: sweep.lever, requested: wantLever, reason:
        `flags_added names a CLASS B lever (${kind.key}): it changes what a step contains, so step i ` +
        `is not the same batch on both legs. The step-1 probe and the step-aligned delta are both ` +
        `invalid, and L0 does not apply because no kernel changed — this would rest on throughput ` +
        `alone. scripts/compare_legs_tokens.py supplies a comparison basis for that class; it is ` +
        `not a gate.` });
      log0(`ConfigSweep REFUSED (${sweep.lever}): Class B lever ${kind.key}`);
    } else {
      gatesUsed += 1;
      // A config lever passes through the SAME correctness gate as a kernel, or it cannot ACCEPT.
      // Until 2026-10-09 this call carried no l1 at all, so a Class-A lever was judged on throughput
      // alone -- the exact objection used to refuse Class B. It mattered: Megatron's flash attention
      // backend on this stack has a forward within 0.11% of fused and a gradient 2.7x too small
      // (step-1 grad_norm 26.7 vs 68.0 in the production recipe). Had flash measured faster, it would
      // have been accepted.
      const l1 = sweep.l1 && typeof sweep.l1.pass === 'boolean' ? sweep.l1 : null;
      // A lever may move exactly its own implementation keys and the env knobs it DECLARES (env_added) --
      // an NVTE_* backend knob too, by name. Every NVTE_* used to be allowed wholesale (audit ID-4).
      const sweepEnv = leverEnv(sweep.env_added);
      const leverImpl = [...leverImplDelta(sweep.flags_added), ...Object.keys(sweepEnv).map((k) => `env:${k}`),
        ...Object.keys(sweepEnv).filter((k) => k.startsWith('NVTE_'))];
      const cv = gateVerdict({ ref: sweep.ref, cand: sweep.cand, comparison: sweep.comparison,
                               l1, mode: 'deep', allowedImpl: leverImpl });
      const cb = refBinding(stackRef, sweep.ref);
      if (cv.accepted && !cb.ok) Object.assign(cv, { accepted: false, klass: 'incomparable', reason: cb.why });
      if (!l1 && cv.accepted) {
        cv.accepted = false;
        cv.klass = 'no_correctness_gate';
        cv.reason = 'the Config Tuner returned no L1 result (scripts/trajectory_gate.py compare): a ' +
                    'config lever may not be accepted on throughput alone';
      }
      // A config change is EXPECTED to move config_digest -- that is what makes it a config change.
      // comparable() refuses that as "different workload", correctly, for a KERNEL candidate. Here
      // the identity difference is the subject of the experiment, so the digest check is reported
      // rather than enforced, and the Tuner must have named exactly which key moved.
      if (cv.klass === 'incomparable' && cv.digestOnly) {
        // A lever that moves WORKLOAD identity (lr schedule, precision recipe, loss normalization...) is
        // a recipe change, not an implementation lever: L1 refuses such a pair by design (different
        // workload, step i is not the same computation), and the fixed-recipe rule forbids tuning it.
        // Before 2026-10-22 this path re-gated ANY 'incomparable' on a Tuner-supplied digest delta,
        // skipping the rank and implementation checks (audit G2/ORCH-6); the review then showed the
        // re-gate could never pass L1 anyway. So: say what it is, from the legs, and do not accept.
        const wr = (sweep.ref || {}).workload_identity, wc = (sweep.cand || {}).workload_identity;
        const moved = wr && wc ? [...new Set([...Object.keys(wr), ...Object.keys(wc)])].filter((k) => wr[k] !== wc[k]) : null;
        Object.assign(cv, { accepted: false, klass: 'recipe_change', digestOnly: false, identity_moved: moved,
          reason: `the lever moves workload identity${moved ? ` [${moved.join(', ')}]` : ''}: a recipe change, ` +
                  'not an implementation lever -- not gateable by step-aligned L1 and not tunable under the ' +
                  'fixed-recipe rule' });
      }
      sweepRecords.push({ lever: sweep.lever, requested: wantLever, flags_added: sweep.flags_added,
        env_added: Object.keys(sweepEnv).length ? sweepEnv : null, ceiling_pct: sweep.ceiling_pct, verdict: cv,
        config_digest_delta: sweep.config_digest_delta || null,
        memory_conversion: sweep.memory_conversion || null });
      log0(`ConfigSweep ${sweep.lever}: ${cv.klass} (${cv.delta_pct}% vs band ${cv.band_pct}%, ` +
           `ceiling was ${sweep.ceiling_pct}%)`);
      if (cv.accepted) {
        // Same independent recomputation as a kernel gate (audit ORCH-1), on the Tuner's legs.
        // L1 is recomputed too (G-3). A config lever has no L0, so never --l0-strictly-better; its kind is
        // the Tuner's declared one (config_tuner.md 4b), recorded in the verifier's output.
        const s1 = ((sweep.l1 || {}).checks || {}).step1_loss_rel_pct || {};
        const cRefs = [sweep.ref.dir || sweep.ref.probe_dir].filter(Boolean);
        const cCands = [sweep.cand.dir || sweep.cand.probe_dir].filter(Boolean);
        const ver = await safeAgentStub('gate_verifier', 'verify',
          { REF_DIRS: cRefs, CAND_DIRS: cCands,
            KIND: 'config', LEVER_FLAGS: flagsString(sweep.flags_added), FLOOR: runFloor, NPROC,
            LEVER_ENV: Object.keys(sweepEnv).join(',') || null,
            L1_KIND: s1.candidate_kind === 'kernel_swap' ? 'kernel_swap' : 'overlay',
            L1_BAND: wfReturn.bands.trajectory_band_path || null, L1_L0_STRICTLY_BETTER: false,
            GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
          'Run verify_gate.py on EVERY ref x cand pairing and return its output verbatim. You are a script runner.');
        const vd = verifyDecision(ver, sweep.ref, sweep.cand, false,
                                  { expected: { refs: cRefs, cands: cCands }, needL1: true });
        cv.verifier = vd;
        if (!vd.ok) Object.assign(cv, { accepted: false, klass: 'verifier_refused', reason: vd.why });
      }
      if (cv.accepted) { curConfig = acceptLever(curConfig, sweep); stackRef = sweep.cand; }
    }
  }
  }
  // Shape: ALWAYS {levers:[...], accepted}. Not "an object when one lever and an array when
  // several" -- a result shape that depends on the input is a reader bug waiting to happen, and
  // this field is read by the report and the validate phases.
  if (sweepRecords.length) {
    // Surfaced at the TOP LEVEL too. `accepted` lists kernels; the first run that accepted a config
    // lever (packing, +258% equal-useful-token) returned `accepted: []` at the top, which reads as
    // "nothing was accepted" to anyone who does not dig into config_sweep.
    // ALL accepted levers, merged. Class-B (packing) flags are listed but kept out of flags_added:
    // they change the data, not the cand arm of a same-data comparison (QualityGate reads this).
    const accs = sweepRecords.filter((r) => r.verdict && r.verdict.accepted);
    const accA = accs.filter((r) => !String(r.class || 'A').startsWith('B'));
    wfReturn.accepted_config = accs.length ? {
      lever: accs.map((r) => r.lever).join('+'),
      class: accA.length ? 'A' : accs[0].class,
      levers: accs.map((r) => ({ lever: r.lever, class: r.class || 'A', flags_added: r.flags_added, env_added: r.env_added || null,
                                 delta_pct: r.verdict.delta_pct, delta_source: r.verdict.delta_source })),
      delta_pct: accs.length === 1 ? accs[0].verdict.delta_pct : null,
      delta_source: accs.length === 1 ? accs[0].verdict.delta_source : 'per lever, see levers[]',
      flags_added: (accA.length ? accA : accs).reduce((f, r) => mergeFlags(f, r.flags_added), []),
      env_added: (accA.length ? accA : accs).reduce((e, r) => ({ ...e, ...(r.env_added || {}) }), {}) } : null;
    wfReturn.config_sweep = {
      levers: sweepRecords,
      accepted: accs[0] || null,          // first accepted (kept for older readers)
      accepted_all: accs.map((r) => r.lever),
      requested: CONFIG_LEVERS.length ? CONFIG_LEVERS : null,
      not_measured: CONFIG_LEVERS.filter((l) => !sweepRecords.some((r) => r.requested === l)),
    };
  }
}

// ---------------------------------------------------------------------------
// KernelTrack — the fast-mode optimization loop.
// Lanes may use the whole GPU pool; a gate DRAINS them, because a training A/B needs every GPU.
// ---------------------------------------------------------------------------
// An accepted config lever moves which kernels dominate: re-profile ON the carried config before the kernel
// track, as after a kernel accept (audit #11: the queue was the pre-lever profile's).
if (stackRef !== baseline) {
  const r = await profileAndRoute(curConfig, 'after_config');
  if (r.profile) profile = r.profile;
  if (r.strategy && r.strategy.head_candidates) headQueue = rankQueue(applySkips(r.strategy.head_candidates.slice()));
}
phase('KernelTrack');
let refLeg = stackRef;          // the carried accepted stack (+ accepted levers), not the bare baseline
let noImprove = 0;

// outOfTime() is async (it reads the clock through an agent) and is deliberately LAST in the
// condition: && short-circuits, so the free bounds are checked first and no clock agent is spawned
// once the queue is empty or the task/gate budget is already spent.
let stopCause = null;   // set at the break that ends the kernel track
while (headQueue.length && tasksUsed < BUDGET && gatesUsed < MAX_GATES && !(await outOfTime())) {
  const stop = stopRule(headQueue, runFloor);
  if (stop.stop) { log0(`stop rule: best remaining ${stop.best_expected_gain_pct}% <= band ${stop.band_pct}%`); stopCause = 'stop_rule'; break; }
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
    // The forward-only refusal protects AUTOGRAD ops: a kernel 2x faster forward and 3x slower
    // backward is a regression a forward-only task reports as a win. An op OUTSIDE autograd -- the
    // optimizer step, the dist-opt grad scale + norm tail -- has no backward to hide one in, and was
    // refused by this rule anyway (2026-10-15: grad_tail_fused_scale_check, 2.6% of the stacked step,
    // extracted with 11 cases and 7 negative controls, then dropped here). The Extractor must SAY
    // the op is outside autograd (`autograd: false`); an omitted field still refuses.
    if (ext.has_backward === false && ext.autograd !== false) {
      log0('  WARNING: task has no backward. A forward-only task can report a win for a kernel that ' +
           'is slower overall; treating as unusable. (An op outside autograd must be returned with ' +
           'autograd: false.)');
      return null;
    }
    if (ext.has_backward === false) {
      log0('  op is outside autograd (autograd: false): it has no backward by construction, so the ' +
           'task objective is the op itself');
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
  let integ = await barrier.gate(async () => {
    gatesUsed += 1;
    return safeAgentStub('integrator', 'integrate',
      { ...benchInputs(curConfig, `gate_${h.short_name}`), HEAD: h, TASK: optimized.ext,
        KERNEL_RESULT: optimized.kernel, CARRIED_OVERLAY: curOverlay, REF_EXPECTED: refLeg },
      'Overlay reversibly, prove engagement for forward AND backward, take a paired same-session A/B.');
  });
  // The median and the 10%-trimmed mean of the per-step ratios disagree (DECISIONS 4.88): on 30-step legs that is
  // usually the trimmed mean's noise (re-measured at 60 steps, all seven s8 increments agree on every estimator),
  // sometimes a kernel slower on a minority of steps. ONE re-gate on legs twice as long decides; the second pair
  // replaces the first for everything downstream (audit, verifier, record).
  if (integ && integ.ref && integ.cand && gatesUsed < MAX_GATES) {
    const pre = gateVerdict({ ref: integ.ref, cand: integ.cand, comparison: integ.comparison,
                              l0: { pass: true }, l1: { pass: true }, mode: DEEP_MODE ? 'deep' : 'fast',
                              allowedImpl: KERNEL_IMPL_DELTA });
    if (pre.klass === 'estimators_disagree') {
      const longIters = Math.max(60, 2 * ITERS);
      log0(`  ${pre.reason}; re-gating once at ${longIters} iterations`);
      const integ2 = await barrier.gate(async () => {
        gatesUsed += 1;
        return safeAgentStub('integrator', 'integrate',
          { ...benchInputs(curConfig, `gate_${h.short_name}_long`), ITERS: longIters, HEAD: h, TASK: optimized.ext,
            KERNEL_RESULT: optimized.kernel, CARRIED_OVERLAY: curOverlay, REF_EXPECTED: refLeg,
            REGATE_OF: { ref: integ.ref.dir || integ.ref.probe_dir, cand: integ.cand.dir || integ.cand.probe_dir } },
          'RE-GATE on longer legs (ITERS is doubled): the first pair\'s median and trimmed-mean estimates disagreed. ' +
          'Same overlay, same protocol, fresh legs.');
      });
      if (integ2 && integ2.ref && integ2.cand) {
        integ2.regate_of = { first_comparison: integ.comparison, reason: pre.reason };
        integ = integ2;
      }
    }
  }
  if (!integ || !integ.ref || !integ.cand) { noImprove += 1; continue; }
  // FAIL CLOSED (audit OVL-1): engagement must be PROVEN true on both paths, not merely not-false.
  const eng = integ.engagement || {};
  if (eng.forward !== true || eng.backward !== true) {
    log0(`  engagement NOT proven on both paths (forward=${eng.forward}, backward=${eng.backward}) — ` +
         'refusing the result (an overlay bound to only one half, or not at all, is the exact failure ' +
         'L1 exists to catch, and it is cheaper to catch here)');
    noImprove += 1; continue;
  }

  // The Integrator's diagnosis travels to the Auditor, because the Auditor is what writes findings
  // back to the kernel layer's KB and the worst failures never reach L0/L1 at all. The
  // mlp_fc1_swiglu candidate died at iteration #2 with an inf gradient, reproduced twice: a stronger
  // result than any L1 verdict, diagnosed exactly by the Integrator, and recorded in no KB card
  // because the rule fired only on l0.pass/l1.pass being false. The most catastrophic failure mode
  // available was the one the knowledge base never heard about.
  const audit = await safeAgentStub('convergence_auditor', 'gate',
    { TASK: optimized.ext, REF_PROBE: integ.ref && integ.ref.probe_dir,
      CAND_PROBE: integ.cand && integ.cand.probe_dir,
      BAND: wfReturn.bands.trajectory_band_path, OVERLAY_IMPL: integ.overlay_path || null,
      GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts`,
      INTEGRATOR_NOTES: integ.notes || null,
      CAND_USABLE: !!(integ.cand && legUsable(integ.cand).ok) },
    'Run L0 (out/dgrad/wgrad vs fp32, magnitude AND sign), the main_grad accumulation contract ' +
    '(GEAK_SCRIPTS/main_grad_contract.sh, its fields copied verbatim) and L1 (step-indexed envelope). ' +
    'If CAND_USABLE is false the candidate never produced a measurable leg — read INTEGRATOR_NOTES ' +
    'and record THAT as the finding; a candidate that diverges is a stronger result than one that ' +
    'is merely slow.');

  // Forward the Integrator's window-aligned comparison. Omitting it here made alignedDelta() return
  // null on EVERY real gate -- the unit tests pass `comparison` straight into gateVerdict(), so they
  // exercised the aligned path while production never did, and the 7B run gated on a raw ratio.
  const v = gateVerdict({ ref: integ.ref, cand: integ.cand, comparison: integ.comparison,
                          l0: audit && audit.l0, l1: audit && audit.l1,
                          mode: DEEP_MODE ? 'deep' : 'fast', allowedImpl: KERNEL_IMPL_DELTA });
  // The verifier's OWN L0 (DECISIONS 4.99): the frozen oracle re-run on the shipped overlay (l0_verify.sh, one
  // GPU). Its `correct` is required to accept, and its evidence -- forward bit-identity against the captured
  // original, per-output error vs the original's -- is what plans L1. The Auditor's l0 can still only reject.
  let l0r = null;
  if (v.accepted) {
    const lv = await safeAgentStub('gate_verifier', 'verify',
      { KIND: 'l0', L0_TASK: optimized.ext.task_dir, L0_OVERLAY: integ.overlay_path || null,
        GPU: GPU_IDS[0], GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
      'Run l0_verify.sh on the task and overlay and return its result verbatim as pairs[0].l0_recomputed. ' +
      'You are a script runner.');
    const p0 = lv && Array.isArray(lv.pairs) ? lv.pairs[0] : null;
    l0r = p0 && p0.l0_recomputed && typeof p0.l0_recomputed === 'object' ? p0.l0_recomputed : null;
    v.l0_recomputed = l0r;
    const tdn = (x) => String(x || '').replace(/\/+$/, '').split('/').pop();
    if (!l0r || l0r.correct !== true || tdn((l0r.oracle || {}).task_dir || l0r.task) !== tdn(optimized.ext.task_dir)) {
      Object.assign(v, { accepted: false, klass: l0r && l0r.correct === false ? 'numerics' : 'no_correctness_gate',
        reason: !l0r ? 'no independent L0 (the verifier did not re-run the oracle): a kernel may not be accepted on ' +
                       "the Auditor's word"
              : l0r.correct !== true ? `the verifier's own L0 run of the oracle is not correct (${l0r.error || 'correct=' + l0r.correct})`
                : `the verifier ran the oracle of task ${tdn((l0r.oracle || {}).task_dir || l0r.task)}, not this head's` });
    }
  }
  // INDEPENDENT RECOMPUTATION of every pairing the Integrator measured (audit ORCH-1).
  let mgcAppl = null;   // the verifier's mgc_applicability.py (audit 2026-10-25 G-1)
  if (v.accepted) {
    const lm = integ.legs_measured || {};
    // One entry per LEG, not per spelling: agents report the same leg as /work/... and as a host path,
    // with or without /probe (run wf_e849d951-2cc recomputed one pairing 4 times).
    const uniqLegs = (xs) => { const seen = new Set(); return xs.filter(Boolean).filter((d) => {
      const k = legBase(d); if (seen.has(k)) return false; seen.add(k); return true; }); };
    const refDirs = uniqLegs([integ.ref.dir || integ.ref.probe_dir, ...(lm.refs || [])]);
    const candDirs = uniqLegs([integ.cand.dir || integ.cand.probe_dir, ...(lm.cands || [])]);
    const l1Plan = l1PlanFromL0(l0r);   // the verifier's recomputed evidence, not the Auditor's transcript
    const ver = await safeAgentStub('gate_verifier', 'verify',
      { REF_DIRS: refDirs, CAND_DIRS: candDirs,
        KIND: 'kernel', FLOOR: runFloor, NPROC, GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts`,
        L1_KIND: l1Plan.kind, L1_BAND: wfReturn.bands.trajectory_band_path || null,
        L1_L0_STRICTLY_BETTER: l1Plan.l0_strictly_better,
        MGC_TASK: optimized.ext.task_dir || null, MGC_IMPL: integ.overlay_path || null },
      'Run verify_gate.py on EVERY ref x cand pairing and return its output verbatim. You are a script runner.');
    const vd = verifyDecision(ver, integ.ref, integ.cand, true,
                              { expected: { refs: refDirs, cands: candDirs }, needL1: true });
    v.verifier = vd;
    mgcAppl = ver && ver.mgc_applicability && typeof ver.mgc_applicability.applicable === 'boolean'
      ? ver.mgc_applicability : null;
    if (!vd.ok) Object.assign(v, { accepted: false, klass: 'verifier_refused', reason: vd.why });
  }
  // The reference must BE the carried stack, and the candidate that stack plus this head's overlay
  // (audit G3): comparable() only checks that they differ in GEAK_OVERLAY*, not HOW. A bare ref
  // against a stacked cand would credit this head with the whole stack. SHA components, not names
  // (byte-identical overlays get different manifest names).
  const carried = overlayShaCheck(refLeg, integ.ref, integ.cand);
  if (v.accepted && !integ.overlay_path) {
    Object.assign(v, { accepted: false, klass: 'incomparable',
      reason: 'no overlay_path: an accepted overlay must be carried into every later gate and the bundle' });
  }
  if (v.accepted && !carried.ok) {
    Object.assign(v, { accepted: false, klass: 'incomparable', reason: carried.why });
  }
  const kb = refBinding(refLeg, integ.ref);
  if (v.accepted && !kb.ok) Object.assign(v, { accepted: false, klass: 'incomparable', reason: kb.why });
  // The oracle that produced L0 must score every output (DECISIONS 4.60): a template oracle from
  // build_task_dir >= v2 (output count / coverage / shape gated), or a custom one the Auditor confirms.
  const orc = audit && audit.l0 && audit.l0.oracle;
  // The oracle must be THIS head's task (audit ORCH-A7: task_dir was optional and never compared).
  const tdName = (x) => String(x || '').replace(/\/+$/, '').split('/').pop();
  const orcTaskOk = !!orc && !!orc.task_dir && tdName(orc.task_dir) === tdName(optimized.ext.task_dir);
  const orcOk = !!orc && orcTaskOk && orc.scores_every_output === true &&
    (orc.kind === 'custom' || (orc.kind === 'template' && Number(orc.template_version) >= 2));
  if (v.accepted && !orcOk) {
    Object.assign(v, { accepted: false, klass: 'no_correctness_gate',
      reason: orc && !orcTaskOk ? `L0 ran the oracle of task ${tdName(orc.task_dir) || '(none)'}, not this head's ` +
                                  `${tdName(optimized.ext.task_dir)}`
            : orc ? `L0 oracle cannot vouch for every output (kind=${orc.kind}, template_version=` +
                    `${orc.template_version}, scores_every_output=${orc.scores_every_output}): rebuild it ` +
                    'with scripts/rebuild_oracle.py' : 'L0 did not report which oracle it ran' });
  }
  // The fp32 main_grad accumulation contract (DECISIONS 4.50 / 4.56): L0 cannot see that path, and the
  // oracle-style limit passes a bf16-rounded wgrad. Fail closed on a missing result, an applicable
  // failure, or a run whose deliberately wrong controls did not fail (passed=false covers that too).
  // "Not applicable" is a pass only when the INDEPENDENT applicability check agrees (audit 2026-10-25
  // G-1: a composite overlay -- RMSNorm+fc1 GEMM+SwiGLU -- matched no head substring and came back
  // not applicable, though it accumulates into main_grad). An applicable site with no head is not passed.
  const mgc = audit && audit.main_grad_contract;
  if (v.accepted && !(mgc && typeof mgc.applicable === 'boolean' &&
                      (mgc.applicable === false || mgc.passed === true))) {
    const noHead = !!(mgc && mgc.applicable && !mgc.head);
    Object.assign(v, { accepted: false, klass: mgc && mgc.applicable && !noHead ? 'numerics' : 'no_correctness_gate',
      reason: mgc ? `main_grad accumulation contract not passed: ${mgc.why || 'passed=' + mgc.passed}`
                  : 'no main_grad_contract result: the fused-wgrad accumulation path was not checked' });
  }
  if (v.accepted && mgc && mgc.applicable === false && !(mgcAppl && mgcAppl.applicable === false)) {
    Object.assign(v, { accepted: false, klass: 'no_correctness_gate',
      reason: mgcAppl ? `main_grad_contract reported not applicable, but the site touches the fused-wgrad path ` +
                        `(${(mgcAppl.evidence || []).join(', ')}): it must be checked, not skipped`
                      : 'main_grad_contract reported not applicable and no independent applicability check confirmed it' });
  }
  // L0/L1 can only reject -- but ABSENT is not a pass (audit G1: an Auditor that failed three times
  // returned null, and the kernel path accepted on throughput alone; the config path never did).
  if (v.accepted && !(audit && audit.l0 && audit.l0.pass === true && audit.l1 && audit.l1.pass === true)) {
    Object.assign(v, { accepted: false, klass: 'no_correctness_gate',
      reason: `L0/L1 not proven (l0.pass=${audit && audit.l0 && audit.l0.pass}, ` +
              `l1.pass=${audit && audit.l1 && audit.l1.pass}): a kernel may not be accepted on throughput alone` });
  }
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
    wfReturn.blocked = { reason: v.reason, head: h.short_name || h.kernel }; stopCause = 'blocked';
    log0('  STOPPING: the box is not quiet enough to measure on. Re-run when it is.');
    break;
  }
  const record = { head: h.short_name || h.kernel, verdict: v, audit: audit || null,
                   overlay: integ.overlay_path || null, task_dir: optimized.ext.task_dir };

  if (v.accepted) {
    // COMPOSE, do not replace: the next head is gated on top of ALL accepted overlays (audit: the
    // single-path carry dropped the first accepted overlay as soon as a second was accepted).
    if (integ.overlay_path) curOverlay = [...curOverlay, integ.overlay_path];
    refLeg = integ.cand;                     // later candidates are gated ON TOP of this
    wfReturn.accepted.push(record);
    noImprove = 0;
    // An accepted change moves which kernels dominate — re-profile, exactly as a config win would.
    const r = await profileAndRoute(curConfig, `after_${h.short_name}`);
    if (r.profile) profile = r.profile;
    // Re-apply the skips: a re-profile rebuilds the queue from scratch and would otherwise
    // re-introduce a head whose verdict is already known.
    // ...and drop heads THIS run already decided (audit ORCH-A10: a re-profile re-queued them).
    const decided = new Set([...wfReturn.accepted, ...(wfReturn.rejected || [])].map((x) => x.head));
    if (r.strategy && r.strategy.head_candidates) {
      headQueue = rankQueue(applySkips(r.strategy.head_candidates.slice()))
        .filter((x) => !decided.has(x.short_name || x.kernel));
    }
  } else {
    if (v.klass === 'banked_memory') {
      bankedMemory.push({ head: record.head, pct: v.banked_memory_pct });
      wfReturn.banked_memory.push({ head: record.head, pct: v.banked_memory_pct });
    }
    wfReturn.rejected = wfReturn.rejected || [];
    wfReturn.rejected.push(record);
    // An inconclusive gate must not count toward the stop rule. noImprove means "candidates are not
    // paying off"; a leg too noisy to measure says nothing about payoff, and letting it stop the
    // track would end a run on the box's behaviour rather than on the work's.
    if (v.klass === 'inconclusive_noise') {
      wfReturn.inconclusive = wfReturn.inconclusive || [];
      wfReturn.inconclusive.push({ head: record.head, delta_pct: v.delta_pct, band_pct: v.band_pct });
      log0(`  INCONCLUSIVE (not counted as a non-win): ${v.reason}`);
    } else {
      noImprove += 1;
    }
  }
  if (noImprove >= 2) { log0('two consecutive non-wins — stopping the kernel track'); stopCause = 'two_consecutive_non_wins'; break; }
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
// The cause recorded at the break wins; the loop-condition causes are inferred only when no break fired
// (audit G10: 'two consecutive non-wins' and 'blocked' used to be reported as a budget or the clock).
wfReturn.stop_reason = stopCause || (!headQueue.length ? 'queue_exhausted'
  : tasksUsed >= BUDGET ? 'task_budget'
  : gatesUsed >= MAX_GATES ? 'gate_budget'
  : 'wall_clock');
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
if (wfReturn.accepted.length) {
  const er = (fin && fin.engagement_recheck) || {};
  if (er.forward !== true || er.backward !== true) {
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

// INDEPENDENT RECOMPUTATION of the headline pair (audit 2026-10-25 ORCH-A5: a Validate that said
// "confirmed" with delta -3% made the run 'ok'). verify_gate --kind stack allows exactly the accepted
// overlays and levers to differ; a run whose accepted change is a packing lever is judged on equal useful
// tokens (--kind packing) against the accepted lever's own pack_gate file.
let valRe = null;
const accCfgAll = (wfReturn.config_sweep && wfReturn.config_sweep.levers || []).filter((r) => r.verdict && r.verdict.accepted);
const accPack = accCfgAll.find((r) => String(r.class || '').startsWith('B'));
if (val && val.ref_dir && val.final_dir && (wfReturn.accepted.length || accCfgAll.length)) {
  const vRefs = [val.ref_dir], vCands = [val.final_dir];
  if (accPack) {
    const ver = await safeAgentStub('gate_verifier', 'verify',
      { REF_DIRS: vRefs, CAND_DIRS: vCands, KIND: 'packing', PACK_GATE_RESULT: accPack.verdict.pack_gate_result || null,
        NPROC, GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
      'Run verify_gate.py --kind packing on the pairing and return its output verbatim. You are a script runner.');
    const pv = ver && Array.isArray(ver.pairs) && ver.pairs[0] ? ver.pairs[0].packing : null;
    const ok = !!(pv && pv.pack_gate_pass === true && pv.data_matches === true && pv.legs_usable === true &&
                  typeof pv.delta_pct === 'number' && pv.delta_pct > PACKING_MIN_GAIN_PCT);
    valRe = { ok, kind: 'packing', delta_pct: pv ? pv.delta_pct : null, band_pct: PACKING_MIN_GAIN_PCT,
              why: ok ? null : (pv ? `packed stack not confirmed: ${JSON.stringify(pv.why || [])}, delta ${pv.delta_pct}`
                                   : 'the gate verifier returned nothing for the packed stack') };
  } else {
    const accA = accCfgAll.filter((r) => !String(r.class || 'A').startsWith('B'));
    const swap = wfReturn.accepted.length > 0 ||
      accA.some((r) => ((r.verdict.verifier || {}).l1 || {}).recomputed && r.verdict.verifier.l1.recomputed.kind === 'kernel_swap');
    const ver = await safeAgentStub('gate_verifier', 'verify',
      { REF_DIRS: vRefs, CAND_DIRS: vCands, KIND: 'stack',
        LEVER_FLAGS: flagsString(accA.reduce((f, r) => mergeFlags(f, r.flags_added), [])), FLOOR: runFloor, NPROC,
        LEVER_ENV: [...new Set(accA.flatMap((r) => Object.keys(r.env_added || {})))].join(',') || null,
        L1_KIND: swap ? 'kernel_swap' : 'overlay', L1_BAND: wfReturn.bands.trajectory_band_path || null,
        L1_L0_STRICTLY_BETTER: false, GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
      'Run verify_gate.py on EVERY ref x cand pairing and return its output verbatim. You are a script runner.');
    const vd = verifyDecision(ver, val.ref_dir, val.final_dir, false,
                              { expected: { refs: vRefs, cands: vCands }, needL1: true });
    valRe = { ok: vd.ok, kind: 'stack', delta_pct: vd.delta_pct != null ? vd.delta_pct : null,
              band_pct: vd.band_pct != null ? vd.band_pct : null, l1: vd.l1 || null, why: vd.ok ? null : vd.why,
              table: vd.table || null };
  }
}
wfReturn.validation_recomputed = valRe;

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

// ---------------------------------------------------------------------------
// QualityGate -- optional outcome test of the FINAL stack (see QUALITY_GATE above).
// ---------------------------------------------------------------------------
// The stack changed if a kernel OR a config lever was accepted. Counting kernels only skipped the
// gate as "nothing accepted" on a deep run whose only change was a config lever (review 2026-10-21).
// A Class-B packing lever is the exception: the protocol trains BOTH arms on the packed pack, so ref
// and cand would be the same recipe -- its correctness gate is pack_gate.py, not this test.
const qConfig = wfReturn.accepted_config &&
  !String(wfReturn.accepted_config.class || '').startsWith('B') ? wfReturn.accepted_config : null;
const stackChanged = wfReturn.accepted.length > 0 || !!qConfig;
// The verdict is RECOMPUTED (audit 2026-10-25 ORCH-A8 / G-3): quality_ab_final.py is pure host Python over
// the legs' train.log / probe files, so the gate_verifier re-runs it with arguments the orchestrator builds,
// and `cleared` is derived from its two verdict strings (/^PASS/ and /^PASS/), never taken from an agent.
// On hf_trl a verdict without the in-run control is refused (G-5: the fixed yardstick is Megatron's).
async function recomputeQuality(dq, prefixes) {
  if (!dq) return null;
  const reported = { cleared: dq.cleared, quality: dq.quality, stability: dq.stability };
  const refuse = (why) => Object.assign({}, dq, { cleared: false, reported,
    quality: `NOT CLEARED: ${why}; the Director reported: ${dq.quality}` });
  if (STACK === 'hf_trl' && !QUALITY_CTRL) return refuse('an HF outcome test needs the in-run control (quality_ctrl)');
  if (prefixes.some((x) => !x)) return refuse('no legs_prefix, so the verdict cannot be recomputed');
  // Two looks (Pocock 0.0294 at each) whenever the one pre-registered extension is possible (DECISIONS 4.86).
  const looks = QUALITY_EXTEND_SEEDS === 'none' ? '1' : '2';
  const qa = [...prefixes.map(String), '--cand', 'cand', '--base', 'ref', '--looks', looks,
              ...(QUALITY_CTRL ? ['--ctrl', String(QUALITY_CTRL)] : [])];
  const ver = await safeAgentStub('gate_verifier', 'verify',
    { KIND: 'quality', QUALITY_ARGS: qa, GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
    'Run quality_ab_final.py with exactly QUALITY_ARGS and return its LAST stdout line (JSON) verbatim as ' +
    'pairs[0].quality_verdict. You are a script runner.');
  const p0 = ver && Array.isArray(ver.pairs) ? ver.pairs[0] : null;
  const r = p0 && p0.quality_verdict;
  if (!r || typeof r.quality !== 'string' || typeof r.stability !== 'string') {
    return refuse('the gate verifier did not recompute the verdict');
  }
  return Object.assign({}, dq, { quality: r.quality, stability: r.stability,
    cleared: /^PASS/.test(r.quality) && /^PASS/.test(r.stability), reported, recomputed: r });
}
if (QUALITY_GATE && stackChanged) {
  phase('QualityGate');
  const qDir = await safeAgentStub('training_director', 'quality',
    { ...benchInputs(curConfig, 'quality'), FINAL: fin, OVERLAY: curOverlay,
      ACCEPTED_CONFIG: qConfig, QUALITY_SEEDS, QUALITY_WORKLOAD, QUALITY_CTRL, QUALITY_STACK: STACK,
      QUALITY_REUSE, QUALITY_REUSE_TOOL: `${WORKFLOW_DIR}/scripts/quality_reuse.py`,
      QUALITY_SCRIPT: `${WORKFLOW_DIR}/scripts/quality_ab.sh`,
      QUALITY_VERDICT: `${WORKFLOW_DIR}/scripts/quality_ab_final.py` },
    'Run the one-epoch outcome test of the final stack against REF and return the verdict of ' +
    'quality_ab_final.py VERBATIM (cleared, quality, stability). Do not reinterpret it.');
  const q = await recomputeQuality(qDir, [qDir && qDir.legs_prefix]);
  wfReturn.quality_gate = q || { cleared: false, quality: 'not run', stability: 'not run' };
  // Reused legs count only if the tool proved them the same recipe (fail closed). Computed ONCE and applied
  // to the first verdict, to the decision to extend, and to the extended verdict: the combined n = 10 sample
  // pools the first sample's reused legs, so an unproven reuse must not be laundered by an extension (audit
  // 2026-10-25 ORCH-A3: the extension replaced the guarded verdict wholesale and could report CLEARED).
  // The proof is the gate_verifier's run of quality_reuse.py check with arguments the ORCHESTRATOR builds
  // from what it accepted (audit 2026-10-25 ID-3/HFC-1): the Director's own `reuse_check` was a
  // self-report whose --allow it chose itself, and the old check compared the realized table only, so a
  // bare REF reused for a stacked cand passed. What may differ: the accepted levers' flags (realized and
  // implementation keys), the overlay content when kernels were accepted, and only what the user
  // pre-declared in quality_reuse.{allow, allow_impl}. Legs predating the harness record are refused
  // unless quality_reuse.accept_unknown_harness is set (recorded in the result).
  let reuseCheck = null;
  if (QUALITY_REUSE && q && q.legs_prefix) {
    const cfgKeys = qConfig ? [...leverImplDelta(qConfig.flags_added),
                               ...Object.keys(qConfig.env_added || {}).map((k) => `env:${k}`)] : [];
    const csv = (x) => (Array.isArray(x) ? x : String(x || '').split(',')).map((k) => String(k).trim()).filter(Boolean);
    // Realized-table names: a `--no-X` flag shows up as X (argparse dest), e.g. check_for_nan_in_loss_and_grad.
    const allow = [...new Set([...cfgKeys.filter((k) => !k.startsWith('env:'))
      .flatMap((k) => (k.startsWith('no-') ? [k, k.slice(3)] : [k])).map((k) => k.replace(/-/g, '_')),
      ...csv(QUALITY_REUSE.allow)])];
    // The cand arm runs the base stack (BASE_OVERLAY_CMD) and/or this run's accepted kernels, the REF arm no
    // overlay: either makes the overlay content a declared difference.
    const allowImpl = [...new Set([...cfgKeys, ...(wfReturn.accepted.length || BASE_OVERLAY_CMD ? ['GEAK_OVERLAY_SHA'] : []),
                                   ...csv(QUALITY_REUSE.allow_impl)])];
    const checkArgs = ['check', '--prefix', String(q.legs_prefix), '--base', 'ref', '--cand', 'cand',
      '--seeds', String(QUALITY_SEEDS), '--allow', allow.join(','), '--allow-impl', allowImpl.join(','),
      ...(QUALITY_REUSE.ctrl_arm && QUALITY_CTRL ? ['--ctrl', String(QUALITY_CTRL)] : []),
      ...(QUALITY_REUSE.accept_unknown_harness === true ? ['--accept-unknown-harness'] : [])];
    const ver = await safeAgentStub('gate_verifier', 'verify',
      { KIND: 'quality_reuse', REUSE_ARGS: checkArgs, GEAK_SCRIPTS: `${WORKFLOW_DIR}/scripts` },
      'Run quality_reuse.py with exactly REUSE_ARGS and return its JSON verbatim as pairs[0].reuse_check. ' +
      'You are a script runner.');
    const p0 = ver && Array.isArray(ver.pairs) ? ver.pairs[0] : null;
    reuseCheck = p0 && p0.reuse_check && typeof p0.reuse_check === 'object' ? p0.reuse_check : null;
    if (reuseCheck && reuseCheck.allow_impl && csv(reuseCheck.allow_impl).join() !== [...allowImpl].sort().join()) reuseCheck = null;
  }
  const reuseOk = !QUALITY_REUSE || !!(reuseCheck && reuseCheck.valid === true);
  const reuseRefusal = (v) => Object.assign({}, v, { cleared: false, reuse_check: reuseCheck,
    quality: `NOT CLEARED: the reused legs of ${QUALITY_REUSE.prefix} were not proven to train the cand's ` +
             `recipe (independent quality_reuse.py check valid=${reuseCheck && reuseCheck.valid}` +
             `${q && !q.legs_prefix ? '; the Director returned no legs_prefix' : ''})` });
  if (q && !reuseOk) wfReturn.quality_gate = reuseRefusal(q);
  else if (q && reuseCheck) wfReturn.quality_gate = Object.assign({}, q, { reuse_check: reuseCheck });
  // Extend only when the verdict is INCONCLUSIVE because the sample is too small -- NOT when the positive
  // control itself was not detected (more seeds of an undetected control prove nothing; review QEXT) and
  // NOT when the reused legs were not proven (above).
  if (q && reuseOk && /^INCONCLUSIVE-quality/.test(String(q.quality || '')) &&
      !/positive control was not detected/.test(String(q.quality || '')) && !/^FAIL/.test(String(q.stability || '')) &&
      QUALITY_EXTEND_SEEDS !== 'none' && q.legs_prefix) {
    log0(`quality gate INCONCLUSIVE (${q.quality}) -- running the ONE pre-registered extension: seeds ${QUALITY_EXTEND_SEEDS}`);
    const qxDir = await safeAgentStub('training_director', 'quality',
      { ...benchInputs(curConfig, 'quality_ext'), FINAL: fin, OVERLAY: curOverlay,
        ACCEPTED_CONFIG: qConfig, QUALITY_SEEDS: QUALITY_EXTEND_SEEDS, QUALITY_WORKLOAD, QUALITY_CTRL, QUALITY_STACK: STACK,
        EXTENSION_OF: q.legs_prefix,
        QUALITY_SCRIPT: `${WORKFLOW_DIR}/scripts/quality_ab.sh`,
        QUALITY_VERDICT: `${WORKFLOW_DIR}/scripts/quality_ab_final.py` },
      'This is the ONE pre-registered seed extension of an INCONCLUSIVE outcome test. Run the SAME protocol ' +
      '(same cand command, arms -- including QUALITY_CTRL if set -- stack and workload) for QUALITY_SEEDS only, then ' +
      'judge the COMBINED sample with `QUALITY_VERDICT EXTENSION_OF <your new prefix> --cand cand` plus ' +
      '`--ctrl QUALITY_CTRL` when it is set, and return that verdict VERBATIM. It is final.');
    const qx = await recomputeQuality(qxDir, [q.legs_prefix, qxDir && qxDir.legs_prefix]);
    wfReturn.quality_gate = Object.assign({}, qx || { cleared: false, quality: 'extension not run', stability: 'not run' },
      { first_verdict: q, extended: true });
    if (!reuseOk) wfReturn.quality_gate = reuseRefusal(wfReturn.quality_gate);   // belt and braces
  }
  log0(wfReturn.quality_gate.cleared
    ? 'quality gate: CLEARED for training (one-epoch outcome test)'
    : `quality gate: NOT CLEARED -- the stack must not be used for production training ` +
      `(${wfReturn.quality_gate.quality}; ${wfReturn.quality_gate.stability})`);
} else {
  wfReturn.quality_gate = QUALITY_GATE
    ? { cleared: null, stability: 'skipped',
        quality: wfReturn.accepted_config
          ? 'skipped: the only accepted change is a Class-B packing lever; the protocol already ' +
            'trains both arms packed, so ref and cand would be identical (gated by pack_gate.py)'
          : 'skipped: nothing accepted this run' }
    : { cleared: null, quality: 'not requested (quality_gate off)', stability: 'not requested',
        note: 'per-kernel gates judge 30 steps only; the stack is NOT cleared for long training ' +
              'until it passes the outcome test (DECISIONS 4.22)' };
}

wfReturn.gates_used = gatesUsed;
wfReturn.tasks_used = tasksUsed;
wfReturn.kb_identity = kbIdentity(curConfig);
// Status is CODE-derived (audit 2026-10-25 ORCH-A5): nothing accepted is `no_gain` whatever Validate says;
// `ok` needs the Director's confirmation AND the verifier's recomputation of the same pair as a win (delta
// above its band) with its own L1 passed. Anything else that accepted something is `unconfirmed`.
const acceptedAnything = wfReturn.accepted.length > 0 || accCfgAll.length > 0;
wfReturn.status = !acceptedAnything ? 'no_gain'
  : (val && val.validation_status === 'confirmed' && val.l1 && val.l1.pass === true && valRe && valRe.ok) ? 'ok'
    : 'unconfirmed';
if (wfReturn.status === 'unconfirmed') {
  wfReturn.status_reason = !val ? 'Validate returned nothing'
    : val.validation_status !== 'confirmed' ? `Validate: ${val.validation_status}`
      : !(val.l1 && val.l1.pass === true) ? 'Validate L1 did not pass'
        : !valRe ? 'Validate returned no headline leg dirs: the pair could not be recomputed'
          : `the independent recomputation refuses the headline: ${valRe.why}`;
}
// A bundle whose own re-check failed, or a stack the outcome test did NOT clear, is not 'ok' (audit G9).
if (wfReturn.status === 'ok' && wfReturn.finalize_engagement_failed) wfReturn.status = 'bundle_not_engaged';
if (wfReturn.status === 'ok' && wfReturn.quality_gate && wfReturn.quality_gate.cleared === false) wfReturn.status = 'not_cleared';
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
