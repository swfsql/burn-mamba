#!/usr/bin/env bash
# usage: extend.sh <src-run> <new-run> <N1> <N2> [--lr LR] [ts-cuda args for all phases]
# Extends a screen: copies the stable checkpoint runs/<src-run> (a screen's
# phase 1) to runs/<new-run>, continues it for N1 more steps (--resume, on its
# LR sequence), then decays a branch runs/<new-run>-d for N2 steps, as
# screen.sh does (cosine over 0.9·N2·ratio schedule steps).
# --lr LR: a step-down. stage.py sets the constant LR from the saved step on
# in runs/<new-run> (a planned later step-down of the schedule goes). A
# --max-lr is refused: on an LR sequence it scales the whole schedule.
# The same N1/N2 for every candidate keeps them step-matched.
# Foreground script: launch it with nohup … & (wait.sh waits on screen.pid).
set -euo pipefail
cd "$(dirname "$0")"
src=$1 run=$2 n1=$3 n2=$4; shift 4
lr=
if [ "${1:-}" = "--lr" ]; then lr=$2; shift 2; fi
for a in "$@"; do
  [ "$a" = "--max-lr" ] && { echo "extend.sh: use --lr (a step-down), not --max-lr"; exit 1; }
done
source ./args.sh; split_args "$@"  # adds --max-vram (see args.sh)
common=(--valid-every 0 --checkpoint-every 0 --valid-batches 1 "${app[@]}" -- "${ex[@]}")
dir=runs/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
cp -r "runs/$src" "$dir"
stamp "$dir"
echo $$ > "$dir/screen.pid"
[ -n "$lr" ] && python3 stage.py "$run" "$lr" >> "$dir/stdout.log"
echo "== $(date -Is) extend.sh phase1 $src -> $run -b $n1 ${common[*]} [$(bin_tag)]" >> "$dir/stdout.log"
bin/ts-cuda --training --resume -a "$dir" -b "$n1" "${common[@]}" >> "$dir/stdout.log" 2>&1
d=$(python3 - "$dir" "$n2" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/metrics.jsonl")]
t = [r for r in rows if r["event"] == "train"][-200:]
ratio = (t[-1]["step"] - t[0]["step"]) / (len(t) - 1)
print(max(1, int(0.9 * int(sys.argv[2]) * ratio)))
EOF
)
python3 decay.py "$run" "dec-$run" "$d" >> "$dir/stdout.log"
cp -r "$dir" "$dir-d"
stamp "$dir-d"
echo "== $(date -Is) extend.sh phase2 $run-d D=$d N2=$n2 [$(bin_tag)]" >> "$dir-d/stdout.log"
bin/ts-cuda --training --resume -a "$dir-d" -c "cfg/dec-$run/training_config.json" \
  -b "$n2" "${common[@]}" >> "$dir-d/stdout.log" 2>&1
echo "== $(date -Is) extend.sh done" >> "$dir-d/stdout.log"
