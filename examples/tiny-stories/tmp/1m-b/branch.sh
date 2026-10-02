#!/usr/bin/env bash
# usage: branch.sh <src-run> <new-run> <training-cfg-name|-> <max-batches> [ts-cuda args]
# Copies runs/<src-run> (weights, optim, progress, logs) to a fresh runs/<new-run>
# and continues it with --resume. `-` keeps the copied training config.
set -euo pipefail
cd "$(dirname "$0")"
src=$1 run=$2 cfg=$3 nb=$4; shift 4
dir=runs/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
cp -r "runs/$src" "$dir"
source ./args.sh; split_args "$@"
stamp "$dir"
echo "== $(date -Is) branch.sh $src $run $cfg $nb ${app[*]} -- ${ex[*]} [$(bin_tag)]" >> "$dir/stdout.log"
ccfg=()
[ "$cfg" != "-" ] && ccfg=(-c "cfg/$cfg/training_config.json")
nohup bin/ts-cuda --training --resume -a "$dir" "${ccfg[@]}" -b "$nb" "${app[@]}" \
  -- "${ex[@]}" >> "$dir/stdout.log" 2>&1 &
echo "started $run from $src (pid $!)"
