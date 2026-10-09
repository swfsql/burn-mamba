#!/usr/bin/env bash
# usage: probe.sh <cfg> [hn-cuda args [-- example args]]  — memory/speed probe,
# not a screen.
# 10 training steps in a fresh runs/probe-<cfg>-<n>, then prints one line:
# - reserved: cubecl's `bytes_reserved` after step 10 (from the `prof` line):
#   every page the pools hold, the CUDA context excluded. It does not drift
#   with the host: rank configs by it.
# - total: the peak total VRAM (host included), and the host's idle usage.
# - ctx: total − idle − reserved, in MiB (the CUDA context, ~90–120 MiB).
#   (A per-process nvidia-smi query cannot work here: the container's PIDs
#   are not the host's.)
# - ms/step: the mean of steps 6..10 (`--profile 5`, the first block is JIT).
# A probe over 5400 MiB total is killed before the card runs out ("KILLED").
# Foreground: several probes run one after another from one caller.
set -uo pipefail
cd "$(dirname "$0")"
cfg=$1; shift
dir=runs/probe-$cfg-$(date +%s)
mkdir -p "$dir"
idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
source ./args.sh; split_args "$@"  # adds --max-vram (see args.sh)
stamp "$dir"
bin/hn-cuda --training -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" -b 10 --valid-every 0 --checkpoint-every 0 \
  --valid-batches 1 "${app[@]}" -- --profile 5 "${ex[@]}" >> "$dir/stdout.log" 2>&1 &
pid=$! peak=0 note=
while kill -0 "$pid" 2>/dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt 5400 ]; then note=KILLED; kill "$pid"; fi
  # Stop at the final validation: the training peak is already known.
  grep -q "running full validation" "$dir/stdout.log" && { kill "$pid"; break; }
  sleep 1
done
wait "$pid" 2>/dev/null
line=$(grep "^prof n=10 " "$dir/stdout.log")
ms=$(sed -n 's/.*total \([0-9.]*\) ms.*/\1/p' <<< "$line")
res=$(sed -n 's/.*reserved \([0-9.]*\) MB.*/\1/p' <<< "$line")
ctx=$(python3 -c "print(round($peak - $idle - ${res:-0} * 1e6 / 2**20))")
printf "%-20s reserved %6s MB  total %5s MiB (idle %4s, ctx %4s)  ms/step %6s  %s  [%s -- %s]\n" \
  "$cfg" "${res:--}" "$peak" "$idle" "$ctx" "${ms:--}" "$note" "${app[*]}" "${ex[*]}"
