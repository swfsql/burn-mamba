#!/usr/bin/env bash
# usage: snapscreen.sh <run> <cfg> <T1> <T2> <E> [args [-- ex args]]
# screen.sh (runs/<run> at constant LR for T1 steps, then the decay branch
# runs/<run>-d of T2), with a checkpoint every E steps. Each checkpoint is
# copied to snap/<run>/<step>/ (model.bpk, the configs, progress.json,
# build.txt) after its progress.json is written, so the copy is whole. The
# decay branch copies into the same snap/<run>/. Use the 1-worker cfgs
# (`*-w1`) for a reproducible run. Launch with nohup … &.
set -euo pipefail
cd "$(dirname "$0")"
run=$1 cfg=$2 t1=$3 t2=$4 every=$5; shift 5
source ./args.sh; split_args "$@"  # adds --max-vram (see args.sh)
common=(--valid-every 0 --checkpoint-every "$every" --valid-batches 1 "${app[@]}" -- "${ex[@]}")
dir=runs/$run snap=snap/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
[ -e "$snap" ] && { echo "exists: $snap"; exit 1; }
mkdir -p "$dir" "$snap"
stamp "$dir"
echo $$ > "$dir/screen.pid"  # wait.sh waits on it

# Copies each new checkpoint of the directory $1, until the file $2 exists.
watch() {
  local d=$1 stop=$2 last=-1 step
  while :; do
    step=$(python3 -c "import json; print(json.load(open('$d/progress.json'))['step'])" 2>/dev/null || echo -1)
    if [ "$step" != "$last" ] && [ "$step" != -1 ] && [ ! -e "$snap/$step" ]; then
      mkdir -p "$snap/$step"
      cp "$d"/{model.bpk,model_config.json,training_config.json,progress.json} "$snap/$step/"
      stamp "$snap/$step"
      last=$step
    fi
    [ -e "$stop" ] && break
    sleep 5
  done
}

echo "== $(date -Is) snapscreen.sh phase1 $run $cfg -b $t1 ${common[*]} [$(bin_tag)]" >> "$dir/stdout.log"
watch "$dir" "$snap/.phase1-done" &
w1=$!
bin/ts-cuda --training -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" -b "$t1" "${common[@]}" >> "$dir/stdout.log" 2>&1
touch "$snap/.phase1-done"; wait "$w1"
# The decay schedule of screen.sh: D = 0.9 · T2 (packed: one step per row).
d=$(python3 -c "print(max(1, int(0.9 * $t2)))")
python3 decay.py "$run" "dec-$run" "$d" >> "$dir/stdout.log"
cp -r "$dir" "$dir-d"
stamp "$dir-d"
echo "== $(date -Is) snapscreen.sh phase2 $run-d D=$d N2=$t2 [$(bin_tag)]" >> "$dir-d/stdout.log"
watch "$dir-d" "$snap/.phase2-done" &
w2=$!
bin/ts-cuda --training --resume -a "$dir-d" -c "cfg/dec-$run/training_config.json" \
  -b "$t2" "${common[@]}" >> "$dir-d/stdout.log" 2>&1
touch "$snap/.phase2-done"; wait "$w2"
echo "== $(date -Is) snapscreen.sh done" >> "$dir-d/stdout.log"
