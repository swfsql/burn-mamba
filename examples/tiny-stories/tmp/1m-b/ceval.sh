#!/usr/bin/env bash
# usage: ceval.sh [-g] [-n batches] [-r root] <run | snap/<run>/<step>> [-- contrast flags]
#   -g  run on the GPU (the default device of bin/ts-cuda): a GPU job. Without
#       -g, run on flex: a host job.
#   -n  batches per cut kind (default 64)
#   -r  the copy root (default ceval)
# How strongly the contrast pushes the checkpoint of runs/<run>
# (`--contrast-eval`), with no backward. It never touches runs/<run>: it
# copies the weights and the configs to <root>/<run> first, and runs there
# (the app rewrites the configs at startup). Prints the table (cut, J, loss,
# delta, push). The flags after `--` set the contrast (K, prompt,
# continuation; default K 4).
set -euo pipefail
cd "$(dirname "$0")"
gpu=0 batches=64 root=ceval
while getopts "gn:r:" opt; do
  case $opt in
    g) gpu=1 ;;
    n) batches=$OPTARG ;;
    r) root=$OPTARG ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
run=$1; shift
[ "${1:-}" = "--" ] && shift
# A run name (runs/<run>) or a checkpoint directory (snap/<run>/<step>).
src=runs/$run
[ -d "$src" ] || src=$run
dir=$root/$(tr / _ <<< "${run#snap/}")
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
cp "$src"/{model.bpk,model_config.json,training_config.json} "$dir/"
source ./args.sh
stamp "$dir"
ex=(--contrast-eval "$batches" "$@")
if [ "$gpu" = 1 ]; then
  bin/ts-cuda -a "$dir" -- "${ex[@]}" > "$dir/contrast-eval.log" 2>&1
else
  BURN_DEVICE=flex bin/ts-cuda -a "$dir" -- "${ex[@]}" > "$dir/contrast-eval.log" 2>&1
fi
echo "== $run"
sed -n '/^contrast eval/,/^saved/p' "$dir/contrast-eval.log" | grep -vE '^(contrast eval|saved)'
