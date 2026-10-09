#!/usr/bin/env bash
# usage: long.sh [-g] [-t] [-b batch] [-r root] <run | snap/<run>/<step> | run dir>
#   -g  run on the GPU (the default device of the binary)
#   -t  a tiny-stories run (the ts-cuda of tiny-stories/tmp/1m-cased), given
#       as its directory, e.g.
#       ../../../tiny-stories/tmp/1m-cased/runs/h16-x12000-d
#   -b  the stories per forward (default 4)
#   -r  the copy root (default long)
# Scores the whole validation stories longer than 2048 characters
# (`--long-stories`) with the weights of the run: copies its weights and
# configs to <root>/<name> (the app rewrites the configs at startup), runs
# there, and prints the segment rows (0..1024, 1024..2048, the tail 2048..,
# all). long-stories.jsonl there holds the bits of each bin of each story
# (longstat.py reads it).
set -euo pipefail
cd "$(dirname "$0")"
gpu=0 bin=bin/hn-cuda batch=4 root=long
while getopts "gtb:r:" opt; do
  case $opt in
    g) gpu=1 ;;
    t) bin=../../../tiny-stories/tmp/1m-cased/bin/ts-cuda ;;
    b) batch=$OPTARG ;;
    r) root=$OPTARG ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
run=$1
# A run name (runs/<run>), or a directory (snap/<run>/<step>, a tiny-stories
# run).
src=runs/$run
[ -d "$src" ] || src=$run
name=$(basename "$src")
[[ "$bin" == */ts-cuda ]] && name=ts-$name
[[ "$src" == snap/* ]] && name=$(tr / _ <<< "${src#snap/}")
dir=$root/$name
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
cp "$src"/{model.bpk,model_config.json,training_config.json} "$dir/"
cp "$bin.txt" "$dir/build.txt"
ex=(--long-stories --context-batch "$batch")
if [ "$gpu" = 1 ]; then
  "$bin" -a "$dir" -- "${ex[@]}" > "$dir/long-stories.log" 2>&1
else
  BURN_DEVICE=flex "$bin" -a "$dir" -- "${ex[@]}" > "$dir/long-stories.log" 2>&1
fi
echo "$name: $(grep -E '^  (bits/char|per char|per story|later)' "$dir/long-stories.log" | tr -s ' ' | paste -sd'|')"
