#!/usr/bin/env bash
# usage: screen.sh <run> <cfg> <T1> <T2> [hn-cuda args for all phases]
# Phase 1: runs/<run>, fresh, at the cfg's (constant) LR, for T1.
# Phase 2: decay branch runs/<run>-d (a copy of phase 1, --resume), for T2.
# T1/T2 are optimizer steps, or minutes of training with an `m` suffix.
# - T1 = `7m`: `--max-seconds 420` (exact wall clock, counted from the first
#   step, so startup/JIT is excluded).
# - T2 = `2m`: a step budget N2 = 2·60·rate, where rate = rows/s over phase
#   1's last 200 train rows. A step budget, not --max-seconds: the cosine length
#   must be known up front. (A fresh process may run faster than the end of
#   phase 1; then phase 2 ends early, it never ends mid-decay.)
# Cosine length D (schedule steps) = 0.9 · N2 · (schedule/opt ratio over phase
# 1's last 200 rows), so the LR reaches min_lr (= max_lr/100) just before N2.
# Foreground script: launch it with nohup … & (see ORIENTATION.md).
set -euo pipefail
cd "$(dirname "$0")"
run=$1 cfg=$2 t1=$3 t2=$4; shift 4
source ./args.sh; split_args "$@"  # adds --max-vram (see args.sh)
common=(--valid-every 0 --checkpoint-every 0 --valid-batches 1 "${app[@]}" -- "${ex[@]}")
dir=runs/$run
[ -e "$dir" ] && { echo "exists: $dir"; exit 1; }
mkdir -p "$dir"
stamp "$dir"
echo $$ > "$dir/screen.pid"  # wait.sh waits on it
if [[ $t1 == *m ]]; then
  budget1=(--max-seconds "$(python3 -c "print(round(${t1%m} * 60))")")
else
  budget1=(-b "$t1")
fi
echo "== $(date -Is) screen.sh phase1 $run $cfg ${budget1[*]} ${common[*]} [$(bin_tag)]" >> "$dir/stdout.log"
bin/hn-cuda --training -a "$dir" -c "cfg/$cfg/training_config.json" \
  -m "cfg/$cfg/model_config.json" "${budget1[@]}" "${common[@]}" >> "$dir/stdout.log" 2>&1
read -r n2 d < <(python3 - "$dir" "$t2" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/metrics.jsonl")]
t = [r for r in rows if r["event"] == "train"][-200:]
ratio = (t[-1]["step"] - t[0]["step"]) / (len(t) - 1)
t2 = sys.argv[2]
if t2.endswith("m"):
    rate = (len(t) - 1) / (t[-1]["elapsed"] - t[0]["elapsed"])
    n2 = max(1, round(float(t2[:-1]) * 60 * rate))
else:
    n2 = int(t2)
print(n2, max(1, int(0.9 * n2 * ratio)))
EOF
)
python3 decay.py "$run" "dec-$run" "$d" >> "$dir/stdout.log"
cp -r "$dir" "$dir-d"
stamp "$dir-d"
echo "== $(date -Is) screen.sh phase2 $run-d D=$d N2=$n2 [$(bin_tag)]" >> "$dir-d/stdout.log"
bin/hn-cuda --training --resume -a "$dir-d" -c "cfg/dec-$run/training_config.json" \
  -b "$n2" "${common[@]}" >> "$dir-d/stdout.log" 2>&1
echo "== $(date -Is) screen.sh done" >> "$dir-d/stdout.log"
