#!/usr/bin/env bash
# Replay logged episodes through the current harness, no model: each listed episode's calls, at its
# own seed (the marks) and ablation, REPEATS times. Then audit the replays (ladder/audit_episodes.py)
# to see which harness faults remain. A list line, as audit_episodes.py --list writes it:
#     <task> <seed> <ablation> <episode.jsonl>
#     sim/replay_check.sh suspect.txt runs/replay-1112
#     REPEATS=1 JOBS=2 sim/replay_check.sh suspect.txt runs/replay-1112
# The replay stops following the log where the scene now answers a call differently (the logged
# calls are replayed as they are), so a fixed fault shows as the call succeeding, and what follows
# it may not make sense. Output: OUT/sim/replay/<task>/ablation-<n>/seed<seed>/ for each run, under
# OUT/r<i>-<k>/, and OUT/logs/.
set -u
LIST=${1:?usage: sim/replay_check.sh <list> <out>}
OUT=${2:?usage: sim/replay_check.sh <list> <out>}
JOBS=${JOBS:-3}
REPEATS=${REPEATS:-2}
STAGGER=${STAGGER:-30}
export SPAI_CPU_THREADS=${SPAI_CPU_THREADS:-$(( $(nproc) / JOBS > 4 ? $(nproc) / JOBS : 4 ))}
source ${VENV:-venv-sim}/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$OUT/logs"
i=0
while read -r task seed ablation file; do
    i=$((i + 1))
    found=(tasks/*/"$task"/task.json)
    for k in $(seq 1 "$REPEATS"); do echo "$i $k ${found[0]} $seed $ablation $file"; done
done < "$LIST" > "$OUT/jobs.txt"
echo "$(wc -l < "$OUT/jobs.txt") replays, $JOBS at a time"
xargs -r -P "$JOBS" -L 1 bash -c '
    ( flock 9; sleep '"$STAGGER"' ) 9>"'"$OUT"'/.launch.lock"
    timeout 3600 python sim/episode.py "$2" --seeds "$3" --ablation "$4" --pressure push --replay "$5" \
        --out "'"$OUT"'/r$0-$1" 2>&1 | grep -v -E "\[Error\]|\[Warning\]" > "'"$OUT"'/logs/r$0-$1.log"
    echo "$(date -u +%H:%M) r$0-$1 $(basename "$(dirname "$2")") seed$3 a$4 exit ${PIPESTATUS[0]}"' < "$OUT/jobs.txt"
echo REPLAYDONE
