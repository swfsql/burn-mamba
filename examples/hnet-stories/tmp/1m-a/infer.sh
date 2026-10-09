#!/usr/bin/env bash
# usage: infer.sh <run> <dir> [temperature]
# Samples of runs/<run> on flex (a host job: run it between GPU runs): copies
# the run to <dir> (the app rewrites its configs), stamps it, and runs
# --inference there, eager (--no-graph). The output: the unprompted stories
# at 0.5 / 0.8 / 1.0 and 3 continuations of each of the user's two prompts at
# `temperature` (default 0.8), in <dir>/inference/ (sample-prompted-<prompt>-
# <continuation>.txt) and <dir>/inference.log. <dir>/README.txt names the
# checkpoint, its last valid and the build.
set -euo pipefail
cd "$(dirname "$0")"
run=$1 dir=$2 temp=${3:-0.8}
src=runs/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
cp -r "$src" "$dir"
source ./args.sh
stamp "$dir"
p0='Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked'
p1=$'Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, "Hi, I\'m Jane. What are your names?"'
bits=$(python3 -c "
import json, math
v = [json.loads(l) for l in open('$src/metrics.jsonl')]
v = [r for r in v if r.get('event') == 'valid']
print(f\"{v[-1]['loss'] / math.log(2):.4f}\")")
echo "Checkpoint: $src (valid $bits bits/char). The user's two prompts, 3 continuations each at temperature $temp (sample-prompted-<prompt>-<continuation>.txt). Flex, eager, $(bin_tag)." > "$dir/README.txt"
BURN_DEVICE=flex bin/hn-cuda -a "$dir" --inference --no-graph -- \
  --prompt "$p0" --prompt "$p1" --continuations 3 --temperature "$temp" > "$dir/inference.log" 2>&1
echo "$dir: $(ls "$dir/inference" | wc -l) samples, $(grep -o '[0-9.]* ms/char' "$dir/inference.log" | head -1)"
