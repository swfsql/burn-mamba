#!/usr/bin/env bash
# usage: speed.sh <run> <cfg> <steps> [ts-cuda args [-- example args]]
# A throughput run of packed training, not a screen: a fresh runs/<run> for
# <steps> batches (`--profile 50` for the reserved memory). Then it prints
# one line (also appended to speeds.txt):
# - rows × width, the story alignment and the fill (from the `corpus:` line),
# - chars/step: the train characters per row (after the cut of the stories
#   that are longer than a row) × rows per step,
# - ms/step: the mean over the steps after step 20 (the JIT is excluded),
# - kchars/s = chars/step / ms/step,
# - reserved: the last `prof` line (cubecl's `bytes_reserved`).
# Foreground: several runs go one after another from one caller.
set -uo pipefail
cd "$(dirname "$0")"
run=$1 cfg=$2 steps=$3; shift 3
source ./args.sh; split_args "$@"  # adds --max-vram (see args.sh)
dir=runs/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
stamp "$dir"
echo "== $(date -Is) speed.sh $run $cfg $steps ${app[*]} -- ${ex[*]} [$(bin_tag)]" >> "$dir/stdout.log"
bin/ts-cuda --training -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" -b "$steps" --valid-every 0 --checkpoint-every 0 \
  --valid-batches 1 "${app[@]}" -- --profile 50 "${ex[@]}" >> "$dir/stdout.log" 2>&1
code=$?
python3 - "$dir" "$run" "$code" "${ex[*]}" <<'EOF' | tee -a speeds.txt
import json, re, sys
d, run, code, ex = sys.argv[1:]
log = open(d + "/stdout.log").read()
m = re.search(r"corpus: (\d+) train characters in (\d+) packed rows of (\d+) \(([\d.]+) of the positions.*?multiples of (\d+)", log)
res = re.findall(r"^prof .*reserved ([\d.]+) MB", log, re.M)
rows_per_step = json.load(open(d + "/training_config.json"))["pack"]["rows"]
t = [json.loads(l) for l in open(d + "/metrics.jsonl")]
t = [r for r in t if r["event"] == "train" and r["step"] > 20]
if not m or len(t) < 2:
    print(f"{run:24s} FAILED (exit {code})  {ex}")
    sys.exit()
chars, rows, width, fill, align = int(m[1]), int(m[2]), int(m[3]), float(m[4]), int(m[5])
cps = chars / rows * rows_per_step
ms = (t[-1]["elapsed"] - t[0]["elapsed"]) / (len(t) - 1) * 1000
print(f"{run:24s} {rows_per_step}x{width:<5d} align {align:3d} fill {fill:.3f}  chars/step {cps:6.0f}"
      f"  ms/step {ms:6.1f}  kchars/s {cps / ms:6.2f}  reserved {res[-1] if res else '-':>7s} MB  exit {code}")
EOF
