#!/bin/bash
# Start a Claude Code session that actually HAS the Workflow tool.
#
# Two things stop a normal session from running this workflow, and neither announces itself — the
# Workflow tool is simply absent from the toolset and the agent will improvise something else:
#
#   1. CLAUDE_CODE_SIMPLE=1 (set in ~/.claude/settings.json on this box) strips the toolset down to
#      Bash/Edit/Read. Verified on claude 2.1.273: with it set the tools are exactly
#      "Bash, Edit, Read"; with it cleared they include "Workflow".
#   2. Public claude builds (>=2.1.x) REJECT `--effort ultracode`. The Workflow/parallel/phase
#      primitives are gated behind the `enableWorkflows` + `ultracode` SETTINGS keys instead. Having
#      the tool present is not the same as it executing the JS pipeline rather than merely
#      backgrounding it, which is why interface/run_e2e.py injects both -- this mirrors it.
#
# Usage:  bash scripts/launch_workflow_session.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF="$(dirname "$HERE")"

SETTINGS='{"enableWorkflows":true,"ultracode":true,"env":{"CLAUDE_CODE_SIMPLE":""}}'

cat <<TXT
Starting Claude Code with the Workflow tool enabled.
  workflow dir : $WF
  settings     : enableWorkflows + ultracode, CLAUDE_CODE_SIMPLE cleared

Once inside, paste ONE of these:

--- stage 1: shake the orchestration out on 0.5B (~20 min) ---
Run the training workflow: Workflow({scriptPath: "$WF/training_workflow.js", args: {workflow_dir: "$WF", workload: "$WF/workloads/qwen2_5_0_5b.env", nproc: 2, gpu_ids: "0,1", iters: 30, budget: 1, max_gates: 2, kernel_budget: 2}})

--- stage 2: the real run on 7B (~2-3 h) ---
Run the training workflow: Workflow({scriptPath: "$WF/training_workflow.js", args: {workflow_dir: "$WF", workload: "$WF/workloads/qwen2_5_math_7b.env", nproc: 8, gpu_ids: "0,1,2,3,4,5,6,7", iters: 30, budget: 2, max_gates: 4, kernel_budget: 3, time_budget_ms: 18000000}})

Check the tool is really there first by asking: "do you have a Workflow tool?"

TXT

# A custom API key that was once REJECTED sends an interactive session to the login screen instead
# of using it -- and it looks like an auth failure, not like a stored preference. Checked here
# because the fix is one dialog answer, and finding that out after launching wastes the launch.
python3 - <<'PY'
import json, os, sys
try:
    st = json.load(open(os.path.expanduser("~/.claude/settings.json"))).get("env", {})
    key = os.environ.get("ANTHROPIC_API_KEY") or st.get("ANTHROPIC_API_KEY", "")
    resp = json.load(open(os.path.expanduser("~/.claude.json"))).get("customApiKeyResponses", {})
    if key and key[-20:] in (resp.get("rejected") or []):
        print("\033[33m  warning\033[0m  this API key is in customApiKeyResponses.rejected, so an")
        print("           interactive session will show the LOGIN screen rather than use it.")
        print("           When the prompt appears, choose YES / 'use this key'.")
except Exception:
    pass
PY

cd "$WF"
exec env IS_SANDBOX=1 CLAUDE_CODE_SIMPLE= \
  claude --settings "$SETTINGS" --dangerously-skip-permissions "$@"
