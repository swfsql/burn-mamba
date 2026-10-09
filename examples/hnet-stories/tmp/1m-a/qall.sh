#!/usr/bin/env bash
# usage: qall.sh <run>…   (run it in the background; it prints once, then exits)
# Silent until queue.log gets its next `empty` line, or queue.sh is gone.
# Then prints the `done`/`fail` lines of the queue since the start, agg.py
# of the runs (those with a -d branch), and per run: the errors, the first,
# a middle and the last `chunks:` line, and the last `prof` line.
# OOM guard as qwait.sh: total VRAM (host included) above $VRAM_MAX MiB
# (default 5400) for 3 samples in a row kills hn-cuda.
cd "$(dirname "$0")"
n0=$(grep -c '^empty' queue.log) l0=$(wc -l < queue.log) over=0 peak=0 max=${VRAM_MAX:-5400}
while [ "$(grep -c '^empty' queue.log)" -le "$n0" ] && pgrep -fx 'bash ./queue.sh' > /dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt "$max" ]; then over=$((over + 1)); else over=0; fi
  if [ "$over" -ge 3 ]; then
    echo "VRAM GUARD: $m MiB > $max MiB, killed hn-cuda at $(date -Is)"
    pkill -x hn-cuda
    over=0
  fi
  sleep 15
done
tail -n +"$((l0 + 1))" queue.log | grep -E '^(done|fail)'
runs=()
for r in "$@"; do [ -f "runs/$r-d/metrics.jsonl" ] && runs+=("$r"); done
[ ${#runs[@]} -gt 0 ] && python3 agg.py "${runs[@]}" 2>&1
for r in "$@"; do
  [ -d "runs/$r" ] || continue
  echo "== $r"
  grep -hE "panicked|VRAM LIMIT|NaN" "runs/$r/stdout.log" "runs/$r-d/stdout.log" 2>/dev/null | head -3
  c=$(grep -h "^chunks" "runs/$r/stdout.log")
  [ -n "$c" ] && sed -n "1p;$(( ($(wc -l <<< "$c") + 1) / 2 ))p;\$p" <<< "$c"
  grep -h "^prof" "runs/$r/stdout.log" | tail -1
done
echo "peak total VRAM $peak MiB; pending: $(grep -vc '^#' queue.txt 2>/dev/null || echo 0)"
