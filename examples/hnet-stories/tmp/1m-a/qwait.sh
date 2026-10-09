#!/usr/bin/env bash
# usage: qwait.sh   (run it in the background; it prints once, then exits)
# Silent until queue.log gets its next `done` / `fail` / `empty` line (see
# queue.sh). Then prints that line, `summ.py` of the run (and of its `-d`
# branch, if any), the error lines of their logs and the peak total VRAM.
# OOM guard as wait.sh: total VRAM (host included) above $VRAM_MAX MiB
# (default 5400) for 3 samples in a row kills hn-cuda (the queue then goes on
# with its next line).
cd "$(dirname "$0")"
ends() { grep -cE '^(done|fail|empty)' queue.log; }
n0=$(ends) peak=0 over=0 max=${VRAM_MAX:-5400}
while [ "$(ends)" -le "$n0" ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt "$max" ]; then over=$((over + 1)); else over=0; fi
  if [ "$over" -ge 3 ]; then
    echo "VRAM GUARD: $m MiB > $max MiB, killed hn-cuda at $(date -Is)"
    pkill -x hn-cuda
    over=0
  fi
  sleep 10
done
line=$(grep -E '^(done|fail|empty)' queue.log | tail -1)
echo "$line"
run=$(awk '{print $4}' <<< "$line")
runs=()
for r in "$run" "$run-d" "$(awk '{print $5}' <<< "$line")"; do
  [ -n "$r" ] && [ -f "runs/$r/metrics.jsonl" ] && runs+=("$r")
done
[ ${#runs[@]} -gt 0 ] && python3 summ.py "${runs[@]}" 2>&1 | tail -30
for r in "${runs[@]}"; do
  grep -hE "panicked|VRAM LIMIT|rror|NaN" "runs/$r/stdout.log" | tail -3
  grep "^prof" "runs/$r/stdout.log" | tail -1
done
echo "peak total VRAM $peak MiB; pending: $(grep -vc '^#' queue.txt 2>/dev/null || echo 0)"
