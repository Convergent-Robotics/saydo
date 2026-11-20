#!/usr/bin/env bash
# The grasp/put-down matrix (sim/grasp_matrix.py) on every lab task, one process per object, two
# at a time, then the summary (OUT/matrix.md, OUT/matrix.json). On the GPU machine, from the repo
# root:
#     sim/grasp_matrix.sh runs/matrix            # TASKS="tasks/lab/lab_30_biowaste_drain" for fewer
# Task variants that share another task's scene (lab_30_biowaste_drain__rinse's config is
# lab_30's) are run once, under the task that owns the config.
set -u
OUT=${1:-runs/matrix}
JOBS=${JOBS:-2}
source ${VENV:-venv-sim}/bin/activate   # VENV: another environment with Isaac Sim and cuRobo
export OMNI_KIT_ACCEPT_EULA=YES SPAI_FAST=${SPAI_FAST:-1}
mkdir -p "$OUT/logs"
SECONDS=0
declare -A seen
for t in ${TASKS:-tasks/lab/*/}; do
  t=${t%/}
  cfg=$(python -c "import json,sys,pathlib; p=pathlib.Path('$t'); print((p/json.load(open(p/'task.json'))['_base_config']).resolve())")
  [[ -n ${seen[$cfg]:-} ]] && { echo "$(basename $t): scene of ${seen[$cfg]}, skipped"; continue; }
  seen[$cfg]=$(basename $t)
  python sim/grasp_matrix.py "$t/task.json" --list | awk -v t="$t" '{print t, $1}' | sort -u
done > "$OUT/jobs.txt"
cat "$OUT/jobs.txt"
# One object per process; a stuck process is killed after 30 minutes.
xargs -P "$JOBS" -L 1 bash -c 'timeout 1800 python sim/grasp_matrix.py "$0/task.json" --object "$1" --out "'"$OUT"'" \
    > "'"$OUT"'/logs/$(basename $0)-$1.log" 2>&1; echo "$(basename $0) $1 exit $?"' < <(grep -v skipped "$OUT/jobs.txt")
python sim/grasp_matrix.py --summarise "$OUT" > /dev/null
echo "DONE in ${SECONDS}s: $OUT/matrix.md"
