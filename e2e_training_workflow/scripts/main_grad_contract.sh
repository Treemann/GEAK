#!/bin/bash
# main_grad_contract.sh -- run scripts/main_grad_contract.py inside the training image (host entry point).
#
#   main_grad_contract.sh <task_dir> <overlay impl.py> <out.json> [--gpu N] [--attr name]
#
# Paths may be host paths under $GEAK_WORK_DIR (mounted /work) or container paths (/work/...). One GPU,
# a named and labelled container (shared-node rule: own containers are identifiable). Extension builds
# go to <out.json's dir>/mgc_ext, never into the overlay dir. Prints the result JSON on stdout.
# Exit 0 = passed or not applicable; 1 = not passed; 2 = could not run.
set -uo pipefail
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEAK_WORK_DIR:-/home/treemann/geak_train}"
REPO="$(cd "$GEAK_ROOT/.." && pwd)"
IMAGE="${GEAK_IMAGE:-rocm/primus:v26.4}"
[ $# -ge 3 ] || { sed -n 2,9p "$0" >&2; exit 2; }
TASK="$1"; IMPL="$2"; OUT="$3"; shift 3
GPU=0; ATTR=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --attr) ATTR=(--attr "$2"); shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
ctr() {   # host path -> container path (a /work/... path is left alone)
  local p="$1"
  case "$p" in /work/*|/repo/*) echo "$p"; return;; esac
  [ -e "$p" ] || [ -e "$(dirname "$p")" ] || { echo "!!! $p does not exist" >&2; return 1; }
  p="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
  case "$p" in
    "$WORK"/*) echo "/work${p#$WORK}";;
    "$REPO"/*) echo "/repo${p#$REPO}";;
    *) echo "!!! $p is outside $WORK and $REPO: it would not exist in the container" >&2; return 1;;
  esac
}
host() { case "$1" in /work/*) echo "$WORK${1#/work}";; *) echo "$1";; esac; }
T_C="$(ctr "$TASK")" || exit 2
I_C="$(ctr "$IMPL")" || exit 2
mkdir -p "$(dirname "$(host "$OUT")")"
O_C="$(ctr "$(host "$OUT")")" || exit 2
EXT_C="$(dirname "$O_C")/mgc_ext"
docker run --rm --init --name "geak_mgc_$(basename "$(dirname "$I_C")")_$$" --label geak=1 \
  --label geak_owner="$(id -un)" --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --shm-size 16G --security-opt seccomp=unconfined \
  -v "$WORK":/work -v "$REPO":/repo -e HIP_VISIBLE_DEVICES="$GPU" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/Primus/third_party/Megatron-LM -e TORCH_EXTENSIONS_DIR="$EXT_C" \
  -e TRITON_CACHE_DIR=/work/triton_cache -w /tmp --entrypoint python "$IMAGE" -B \
  /repo/e2e_training_workflow/scripts/main_grad_contract.py --task "$T_C" --impl "$I_C" \
  --out "$O_C" "${ATTR[@]}" >&2
RC=$?
[ -s "$(host "$OUT")" ] || { echo "!!! no result written (container rc $RC)" >&2; exit 2; }
cat "$(host "$OUT")" | python3 -c "import json,sys; r=json.load(sys.stdin); print(json.dumps({k: r.get(k) for k in ('applicable','passed','head','combined','controls_all_fail','why','task','impl')}))"
exit $RC
