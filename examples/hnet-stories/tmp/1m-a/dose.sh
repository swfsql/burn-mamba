#!/usr/bin/env bash
# usage: dose.sh <cfg> [steps] [-- example args]   — the VRAM of a run.
# A probe (probe.sh) runs 10 steps and meets every slot, but a run holds
# more than a probe, by a factor that depends on the shape (× 1.04 to
# × 1.32: a router makes inner lengths that 10 steps do not meet). So
# dose.sh runs the first `steps` (default 520) steps of the real run (the
# shuffled order), with `--profile 100`, in a fresh runs/dose-<cfg>-<n>,
# and prints one line:
# - peak: the `peak` of each `prof` line (the largest `bytes_reserved` of
#   the pools after a backward over its 100 steps), and their maximum.
#   With the default cleanups of `--max-vram`, the process holds more: the
#   pages that the pools dropped wait for a cleanup (see ORIENTATION.md,
#   VRAM). With `--vram-cleanup-every 1`, they never wait past a window.
#   The CUDA context and the loaded kernels come on top (~140 MiB).
# - ms/step of each `prof` line.
# 520 steps meet a full gap of the default cleanups (256 → 512).
# Run it alone on the GPU. The kill guard (PROBE_KILL_MIB, 5400 MiB of the
# whole card from nvidia-smi) only protects the card: never decide by it.
set -uo pipefail
cd "$(dirname "$0")"
cfg=$1; shift
steps=520
if [ $# -gt 0 ] && [ "$1" != "--" ]; then steps=$1; shift; fi
[ "${1:-}" = "--" ] && shift
dir=runs/dose-$cfg-$(date +%s)
mkdir -p "$dir"
gpu=${CUDA_VISIBLE_DEVICES:-0}
source ./args.sh; split_args -- "$@"  # adds --max-vram and gc (see args.sh)
stamp "$dir"
bin/hn-cuda --training -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" -b "$steps" --valid-every 0 --checkpoint-every 0 \
  --valid-batches 1 -- --profile 100 "${ex[@]}" >> "$dir/stdout.log" 2>&1 &
pid=$! note=
while kill -0 "$pid" 2>/dev/null; do
  m=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$m" -gt "${PROBE_KILL_MIB:-5400}" ]; then note=KILLED; kill "$pid"; fi
  grep -q "running full validation" "$dir/stdout.log" && { kill "$pid"; break; }
  sleep 1
done
wait "$pid" 2>/dev/null
peaks=$(sed -n 's/^prof n=\([0-9]*\) .*peak \([0-9]*\)\.[0-9]* MB.*/\1:\2/p' "$dir/stdout.log" | paste -sd' ')
max=$(grep '^prof n=' "$dir/stdout.log" | grep -o 'peak [0-9]*' | awk '{print $2}' | sort -n | tail -n 1)
ms=$(sed -n 's/^prof n=.* total \([0-9]*\)\.[0-9]* ms.*/\1/p' "$dir/stdout.log" | paste -sd' ')
printf "%-20s dose %4s steps  peak max %5s MB [%s]  ms/step [%s]  %s [-- %s]\n" \
  "$cfg" "$steps" "${max:--}" "$peaks" "$ms" "$note" "${ex[*]}"
