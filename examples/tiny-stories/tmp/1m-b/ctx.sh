#!/usr/bin/env bash
# usage: ctx.sh [-g] [-n stories] [-b batch] [-r root] <run | snap/<run>/<step>>
#   -g  run on the GPU (the default device of bin/ts-cuda). Use it: on flex
#       (no -g), 2048 stories take ~36 min, too slow to be useful. Beside a
#       training, give -b 8: beside h16 × g2, the card total stayed at 4.7
#       GiB or less (at most ~0.9 GiB above the training alone), at 85–200 s
#       per checkpoint, and the training ran ~1.8× slower.
#   -n  read the first N validation stories (--context-stories, default: the
#       256 of the validation loss)
#   -b  the sequences per forward (default 4 on flex, 32 on the GPU)
#   -r  the copy root (default ctx5)
# Measures the context use of runs/<run>: copies its weights and configs to
# <root>/<run> (the app rewrites the configs at startup), runs `--context-use`
# there, and prints the join/far rows, the no-prompt row, the gain row and
# the two re-mention rows (`letters`, `first`: own, none, other, other2,
# gain, use) of the result. The log there also has the coherence boosting
# table, and context-stories.jsonl has the scores of each story (ctxstat.py
# reads it).
set -euo pipefail
cd "$(dirname "$0")"
gpu=0 stories= cb= root=ctx5
while getopts "gn:b:r:" opt; do
  case $opt in
    g) gpu=1 ;;
    n) stories=$OPTARG ;;
    b) cb=$OPTARG ;;
    r) root=$OPTARG ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
run=$1
[ -z "$cb" ] && { [ "$gpu" = 1 ] && cb=32 || cb=4; }
# A run name (runs/<run>) or a checkpoint directory (snap/<run>/<step>).
src=runs/$run
[ -d "$src" ] || src=$run
dir=$root/$(tr / _ <<< "${run#snap/}")
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
cp "$src"/{model.bpk,model_config.json,training_config.json} "$dir/"
source ./args.sh
stamp "$dir"
ex=(--context-use --context-batch "$cb")
[ -n "$stories" ] && ex+=(--context-stories "$stories")
if [ "$gpu" = 1 ]; then
  bin/ts-cuda -a "$dir" -- "${ex[@]}" > "$dir/context-use.log" 2>&1
else
  BURN_DEVICE=flex bin/ts-cuda -a "$dir" -- "${ex[@]}" > "$dir/context-use.log" 2>&1
fi
echo "$run: $(grep -E '^  (join|far|no prompt|gain|letters |first )' "$dir/context-use.log" | tr -s ' ' | paste -sd'|')"
