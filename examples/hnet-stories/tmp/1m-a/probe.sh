#!/usr/bin/env bash
# usage: probe.sh <cfg> [hn-cuda args [-- example args]]  — memory/speed probe,
# not a screen.
# 10 training steps (a bucketed config: 2 per slot) in a fresh
# runs/probe-<cfg>-<n>, then prints one line:
# - reserved: cubecl's `bytes_reserved` after step 10 (from the `prof` line):
#   every page the pools hold, the CUDA context excluded. It does not drift
#   with the host: rank configs by it.
# - total: the peak total VRAM (host included), and the host's idle usage.
# - ctx: total − idle − reserved, in MiB (the CUDA context, ~90–120 MiB).
#   (A per-process nvidia-smi query cannot work here: the container's PIDs
#   are not the host's.)
# - ms/step: the mean of steps 6..10 (`--profile 5`, the first block is JIT).
#   A bucketed config: 2 steps per slot (`--bucket-probe 2`, `--profile 2`),
#   `slots a/b/…` the reserved MB after each slot, `reserved` their maximum,
#   ms/step that of the last slot (its first step is JIT). A run holds more:
#   R-N15-B5 probed 2903 (the 1024 slot), its run 3386 steady (× 1.17).
# A probe over 5400 MiB total (PROBE_KILL_MIB: the 6 GiB card here) is killed
# before the card runs out ("KILLED").
# Foreground: several probes run one after another from one caller.
set -uo pipefail
cd "$(dirname "$0")"
cfg=$1; shift
dir=runs/probe-$cfg-$(date +%s)
mkdir -p "$dir"
# The GPU of this process (Kaggle's runner sets CUDA_VISIBLE_DEVICES per lane).
gpu=${CUDA_VISIBLE_DEVICES:-0}
idle=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
source ./args.sh; split_args "$@"  # adds --max-vram (see args.sh)
stamp "$dir"
# A bucketed config: 2 batches of each slot, slot by slot (`--bucket-probe`),
# so the probe meets every batch shape (the memory of a shape depends on the
# model, not only on its positions); one prof line per slot.
nslots=$(python3 -c 'import json, sys
def find(o):
    if isinstance(o, dict):
        if isinstance(o.get("bucket"), dict):
            return len(o["bucket"]["slots"])
        return max([find(v) for v in o.values()] + [0])
    return 0
print(find(json.load(open(sys.argv[1]))))' "cfg/$cfg/training_config.json")
if [ "$nslots" -gt 0 ]; then
  steps=$((2 * nslots)) prof=2 bex=(--bucket-probe 2)
else
  steps=10 prof=5 bex=()
fi
bin/hn-cuda --training -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" -b "$steps" --valid-every 0 --checkpoint-every 0 \
  --valid-batches 1 "${app[@]}" -- --profile "$prof" "${bex[@]}" "${ex[@]}" >> "$dir/stdout.log" 2>&1 &
pid=$! peak=0 note=
while kill -0 "$pid" 2>/dev/null; do
  m=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt "${PROBE_KILL_MIB:-5400}" ]; then note=KILLED; kill "$pid"; fi
  # Stop at the final validation: the training peak is already known.
  grep -q "running full validation" "$dir/stdout.log" && { kill "$pid"; break; }
  sleep 1
done
wait "$pid" 2>/dev/null
line=$(grep "^prof n=$steps " "$dir/stdout.log")
ms=$(sed -n 's/.*total \([0-9.]*\) ms.*/\1/p' <<< "$line")
res=$(sed -n 's/.*reserved \([0-9.]*\) MB.*/\1/p' <<< "$line")
ctx=$(python3 -c "print(round($peak - $idle - ${res:-0} * 1e6 / 2**20))")
# The reserved MB after each slot. Not cumulative (`--max-vram` cleans the
# pool), so `reserved` is their maximum.
per=
if [ "$nslots" -gt 0 ]; then
  slotres=$(grep "^prof n=" "$dir/stdout.log" | sed -n 's/.*reserved \([0-9]*\)\.[0-9]* MB.*/\1/p')
  per="slots $(paste -sd/ <<< "$slotres")"
  res=$(sort -n <<< "$slotres" | tail -n 1)
fi
printf "%-20s reserved %6s MB  total %5s MiB (idle %4s, ctx %4s)  ms/step %6s  %s %s [%s -- %s]\n" \
  "$cfg" "${res:--}" "$peak" "$idle" "$ctx" "${ms:--}" "$per" "$note" "${app[*]}" "${ex[*]}"
