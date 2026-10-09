#!/usr/bin/env bash
# usage: params.sh <cfg-name>   — flex dry run (≤2 s) printing the param count.
# It runs the installed build on flex, so the count follows the current code.
cd "$(dirname "$0")"
source ./args.sh
cfg=$1
dir=params/$cfg-$(date +%s%N)
mkdir -p "$dir"
stamp "$dir"
BURN_DEVICE=flex timeout 2 bin/hn-cuda -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" --training -b 1 >> "$dir/stdout.log" 2>&1
echo "$cfg: $(grep -m1 'Number of parameters' "$dir/stdout.log" || echo 'NO COUNT (see '$dir')')"
