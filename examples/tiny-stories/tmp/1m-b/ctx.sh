#!/usr/bin/env bash
# usage: ctx.sh <run> [context batch, default 4]   — a host job (flex).
# Measures the context use of runs/<run> on flex: copies its weights and
# configs to ctx5/<run> (the app rewrites the configs at startup), runs
# `--context-use` there, and prints the join/far rows of the result.
# Run it between GPU runs: a flex job slows a GPU training.
set -euo pipefail
cd "$(dirname "$0")"
run=$1 cb=${2:-4}
dir=ctx5/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
cp "runs/$run"/{model.bpk,model_config.json,training_config.json} "$dir/"
source ./args.sh
stamp "$dir"
BURN_DEVICE=flex bin/ts-cuda -a "$dir" -- --context-use --context-batch "$cb" \
  > "$dir/context-use.log" 2>&1
echo "$run: $(grep -E '^  (join|far)' "$dir/context-use.log" | tr -s ' ' | paste -sd'|')"
