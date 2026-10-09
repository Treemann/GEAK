#!/bin/bash
# Pre-flight for a full Workflow run. Fails EARLY and says exactly what is missing.
#
# A 7B fast run is hours of exclusive box time. Every check here corresponds to something that has
# actually gone wrong during bring-up, and each one costs seconds against the hours it protects.
#
#   bash scripts/preflight.sh                 # full check, including a real measurement leg
#   bash scripts/preflight.sh --no-bench      # skip the measurement leg (~2 min faster)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF="$(dirname "$HERE")"
REPO="$(dirname "$WF")"
WORK="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
MODELS="${GEAK_MODELS_DIR:-/home/treemann/models}"
IMAGE="${GEAK_IMAGE:-rocm/primus:v26.4}"
RUN_BENCH=1
[ "${1:-}" = "--no-bench" ] && RUN_BENCH=0

FAIL=0; WARN=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; WARN=$((WARN+1)); }
sec()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

sec "1. GPUs"
if command -v rocm-smi >/dev/null 2>&1; then
  N=$(rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l)
  [ "$N" -ge 1 ] && ok "$N GPUs visible" || bad "no GPUs visible to rocm-smi"
  # NOT `rocm-smi ... | grep -q`: with `set -o pipefail`, grep -q exits on the first match and the
  # SIGPIPE kills rocm-smi (141), so the pipeline reports failure and a QUIET box reads as BUSY.
  # This check said the opposite of the truth until it was captured to a variable first.
  PIDS_OUT="$(rocm-smi --showpids 2>/dev/null)"
  if printf '%s' "$PIDS_OUT" | grep -q "No KFD PIDs"; then
    ok "no GPU processes — the box is quiet"
  else
    # Not fatal: the occupancy guard re-measures. But a busy box wastes the whole run.
    warn "another process holds a GPU. A contaminated leg is re-measured (bounded) and then the run
        STOPS, so consider waiting. rocm-smi --showpids to see who."
  fi
else
  bad "rocm-smi not found"
fi

sec "2. Container image"
docker image inspect "$IMAGE" >/dev/null 2>&1 && ok "$IMAGE present" || bad "$IMAGE missing (docker pull it)"

sec "3. Workload descriptors and the data they name"
for envf in "$WF"/workloads/*.env; do
  # shellcheck disable=SC1090
  ( set +u; source "$envf"
    miss=""
    for hostpath in "${MODELS}${MODEL_CKPT#/models}" "${MODELS}${TOKENIZER_DIR#/models}"; do
      [ -d "$hostpath" ] || miss="$miss $hostpath"
    done
    if [ -z "$miss" ]; then
      echo "  ok    $(basename "$envf"): ckpt + tokenizer present (GBS=$GBS MBS=$MBS SEQ=$SEQ)"
    else
      echo "  FAIL  $(basename "$envf"): missing$miss"; exit 1
    fi ) || FAIL=$((FAIL+1))
done
DATA="$WORK/data/sft_ultrachat20k.jsonl"
if [ -s "$DATA" ]; then
  L=$(wc -l < "$DATA")
  ok "dataset $L lines ($(du -h "$DATA" | cut -f1))"
else
  bad "dataset missing: $DATA (see RUNBOOK.md step 1)"
fi

sec "4. The workflow's own pieces"
for f in training_workflow.js scripts/bench_train.sh scripts/adapters/megatron.sh \
         scripts/train_probe.py scripts/probe_summarize.py scripts/compare_legs.py \
         scripts/parse_train_profile.py scripts/trajectory_gate.py scripts/l0_gate.py \
         scripts/capture_train_shapes.py scripts/build_task_dir.py scripts/overlay_train.py \
         scripts/rocm_compat.py scripts/gpu_occupancy.py \
         scripts/controls/stale_state_control.py scripts/trajectory_tokens.py \
         scripts/pack_sft.py scripts/pack_gate.py scripts/calibrate_band.py; do
  [ -f "$WF/$f" ] || bad "missing $f"
done
[ "$FAIL" -eq 0 ] && ok "all scripts present"
MISSING_ROLES=""
for r in training_director system_architect profiler kernel_extractor convergence_auditor integrator config_tuner; do
  [ -f "$WF/roles/$r.md" ] || MISSING_ROLES="$MISSING_ROLES $r"
done
[ -z "$MISSING_ROLES" ] && ok "all 7 role prompts present" || bad "missing roles:$MISSING_ROLES"

sec "5. The kernel layer (the recursive callee)"
KWF="${GEAK_KERNEL_WF:-$REPO/kernel_workflow}"
# Check what is CALLED, not what is merely there. training_workflow.js calls the worker lane
# directly: the runtime allows exactly one nesting level and kernel_workflow.js is a dispatcher that
# would spend it (see the KERNEL_WF_SCRIPT comment over there). Both files exist, so the old check
# on kernel_workflow.js would have kept saying ok with kernel_lane.js deleted. The second grep
# re-pins the two to each other, so this section cannot silently drift out of date again.
[ -f "$KWF/kernel_lane.js" ] && ok "$KWF/kernel_lane.js (the recursive callee)" \
  || bad "kernel_lane.js not found at $KWF — this is the script training_workflow.js calls"
grep -qF 'const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;' "$WF/training_workflow.js" \
  && ok "training_workflow.js still calls the lane directly (one nesting level)" \
  || bad "training_workflow.js no longer names kernel_lane.js — section 5 checks the wrong file"
HARNESS="$REPO/e2e_workflow/scripts/harness_lib.py"
[ -f "$HARNESS" ] && ok "harness_lib.py (task dirs need it)" || bad "harness_lib.py not found at $HARNESS"

sec "6. Tests (no GPU)"
# Two LOADER rules that `node --check` CANNOT see, checked as text. Both failures below are valid
# JavaScript — node 22 auto-detects ESM, where a second top-level export is perfectly legal — which
# is exactly why they got past the syntax check and then killed the run at launch:
#   (i)  a SECOND top-level `export`. The Workflow loader special-cases only the leading
#        `export const meta` and parses everything after it as a FUNCTION BODY, where `export` is a
#        syntax error. node sees a well-formed module and says "parses".
#   (ii) a clock/random call. The sandbox's Date.now / argless new Date / Math.random bindings exist
#        and THROW when called, so the failure is at launch, after zero useful work.
WFJS="$WF/training_workflow.js"
EXPORTS=$(grep -cE '^[[:space:]]*export[[:space:]]' "$WFJS")
if [ "$EXPORTS" = 1 ] && head -1 "$WFJS" | grep -qE '^export const meta'; then
  ok "exactly one top-level export, and it is the leading 'export const meta'"
else
  bad "$EXPORTS top-level export statement(s); the loader accepts exactly one, the leading
        'export const meta'. A second one fails AT LAUNCH, before the first agent:"
  grep -nE '^[[:space:]]*export[[:space:]]' "$WFJS" | sed 's/^/        /'
fi
# `new Date(<explicit ms>)` is pure and stays legal; only the argless form reads a clock.
CLOCK=$(grep -nE 'Date\.now\(|new[[:space:]]+Date\([[:space:]]*\)|Math\.random\(|performance\.now\(' "$WFJS")
if [ -z "$CLOCK" ]; then
  ok "no clock/random calls (the budget is served by args.t0_epoch_ms and the clock agent)"
else
  bad "calls the sandbox rejects at runtime — a resumed run must replay identically:"
  printf '%s\n' "$CLOCK" | sed 's/^/        /'
fi
NODE="$(command -v node || ls ~/.cursor-server/bin/linux-x64/*/node 2>/dev/null | head -1)"
if [ -n "$NODE" ]; then
  "$NODE" --check "$WF/training_workflow.js" >/dev/null 2>&1 \
    && ok "training_workflow.js parses" || bad "training_workflow.js does not parse"
  "$NODE" "$WF/scripts/tests/test_training_gate.js" >/dev/null 2>&1 \
    && ok "gate/orchestration tests pass" || bad "gate tests FAIL — run them to see why"
  # EXECUTE the orchestrator with stubbed agents. Three launches have died on defects that are valid
  # JavaScript, pass `node --check`, and throw before any useful work -- a stray export, a sandbox
  # clock call, and twice a temporal dead zone. Each was answered with another text rule covering
  # exactly the shape that had just bitten, and the last rule still let `curOverlay` through at a
  # cost of 3 agents and 24 minutes. Running it finds the class; guessing at it does not.
  "$NODE" "$WF/scripts/tests/test_workflow_dryrun.js" >/dev/null 2>&1 \
    && ok "orchestrator dry-run executes (fast AND deep, stubbed agents, no GPU)" \
    || bad "orchestrator dry-run FAILS — run it to see the throw; this is what a launch would hit"
else
  warn "no node found; JS tests skipped (this box has none on PATH, see RUNBOOK)"
fi
python3 "$WF/scripts/tests/test_parse_train_profile.py" >/dev/null 2>&1 \
  && ok "profile-decomposition tests pass" || bad "profile tests FAIL"
python3 "$WF/scripts/tests/test_trajectory_gate.py" >/dev/null 2>&1 \
  && ok "L1 band/envelope tests pass (incl. the null control)" || bad "trajectory-gate tests FAIL"
# The L2 positive control's claim is that L0 cannot see it BY CONSTRUCTION. That claim is algebraic,
# so it is checkable here in milliseconds without a GPU -- and it must be, because the claim is what
# licenses spending 3 GPU-hours on a soak. See knowledge/gate_convergence.md, "L2 soak spec v3".
# The two identity key lists live in different files and must agree. Both halves of that failure
# were live at once until 2026-10-08: attention-backend was classified and captured by nothing, and
# no overlap-* key existed anywhere, so a ConfigSweep run produced two differently-computed legs
# carrying identical config_digest AND impl_digest. Comparing two lists costs milliseconds.
python3 "$WF/scripts/tests/test_identity_keys.py" >/dev/null 2>&1 \
  && ok "identity key lists agree (captured <-> classified, schema stamped)" \
  || bad "identity key tests FAIL — a digest does not cover what it claims to"
python3 "$WF/scripts/controls/stale_state_control.py" --self-test >/dev/null 2>&1 \
  && ok "L2 positive control: L0-invisibility holds (bit-exact at every alpha)" \
  || bad "L2 positive control self-test FAILS — do not run a soak licensed by it"

sec "7. Disk"
AVAIL=$(df -BG --output=avail "$WORK" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL" ]; then
  [ "$AVAIL" -ge 50 ] && ok "${AVAIL}G free under $WORK" \
    || warn "${AVAIL}G free under $WORK — a 7B trace is ~200MB per profiled rank"
fi

if [ "$RUN_BENCH" = 1 ]; then
  sec "8. A real measurement leg (0.5B, ~2 min) — the check that matters most"
  OUT=$(HIP_VISIBLE_DEVICES=0,1 bash "$WF/scripts/bench_train.sh" \
          --workload "$WF/workloads/qwen2_5_0_5b.env" --nproc 2 --iters 20 --tag preflight 2>&1)
  ST=$(printf '%s' "$OUT" | python3 -c "
import json,sys,re
t=sys.stdin.read(); i=t.find('{')
try:
    d=json.loads(t[i:t.rfind('}')+1])
    print('%s|%s|%s|%s'%(d.get('status'),d.get('useful_tok_s_per_gpu'),
                         d.get('worst_iqr_spread_pct'),d.get('warmup_steps_max')))
except Exception: print('parse_error||')" 2>/dev/null)
  IFS='|' read -r S TPUT IQR WARM <<< "$ST"
  if [ "$S" = "ok" ]; then
    ok "leg ok: ${TPUT} useful tok/s/GPU, IQR ${IQR}%, warmup ${WARM} steps"
  else
    bad "measurement leg returned status='$S' — fix this before spending a run"
    printf '%s\n' "$OUT" | tail -5 | sed 's/^/        /'
  fi
else
  sec "8. Measurement leg — SKIPPED (--no-bench)"
fi

sec "Verdict"
if [ "$FAIL" -eq 0 ]; then
  printf '  \033[32mGO\033[0m — %d warning(s). Launch with the Workflow tool; see RUN.md.\n' "$WARN"
  exit 0
fi
printf '  \033[31mNO-GO\033[0m — %d failure(s), %d warning(s).\n' "$FAIL" "$WARN"
exit 1
