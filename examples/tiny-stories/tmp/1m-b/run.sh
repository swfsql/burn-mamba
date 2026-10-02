#!/usr/bin/env bash
# usage: run.sh <run-name> <cfg-name> <max-batches> [extra ts-cuda args...]
# Fresh run: runs/<run-name> must not exist yet. Stdout+stderr append to its
# stdout.log. Resume/branch: see branch.sh.
set -euo pipefail
cd "$(dirname "$0")"
run=$1 cfg=$2 nb=$3; shift 3
dir=runs/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
source ./args.sh; split_args "$@"
stamp "$dir"
echo "== $(date -Is) run.sh $run $cfg $nb ${app[*]} -- ${ex[*]} [$(bin_tag)]" >> "$dir/stdout.log"
nohup bin/ts-cuda --training -a "$dir" \
  -c "cfg/$cfg/training_config.json" -m "cfg/$cfg/model_config.json" \
  -b "$nb" "${app[@]}" -- "${ex[@]}" >> "$dir/stdout.log" 2>&1 &
echo "started $run (pid $!)"
