#!/usr/bin/env bash
# usage: wait.sh <run> [more runs to summarize]
# Silent until hn-cuda is gone and runs/<run>/screen.pid (if any) has exited;
# tracks peak total VRAM (appended to runs/<run>/vram.txt), then prints
# summ.py + errors + peak.
# OOM guard: total VRAM (host included) above $VRAM_MAX MiB (default 5400) for
# 3 samples in a row kills the screen and hn-cuda before the 6 GB card runs
# out, and logs "VRAM GUARD" into vram.txt. The 3.6 GB rule is the process's
# own: hn-cuda enforces it (`-- --max-vram`, see args.sh), not this guard.
cd "$(dirname "$0")"
run=$1 peak=0 spid= over=0 max=${VRAM_MAX:-5400}
[ -f "runs/$run/screen.pid" ] && spid=$(cat "runs/$run/screen.pid")
tel="runs/$run/telemetry.csv" t0=$(date +%s) i=0
echo "sec,temp_c,sm_mhz,mem_mib,power_w,util_pct,rss_mib" >> "$tel"
while pgrep -x hn-cuda >/dev/null || { [ -n "$spid" ] && kill -0 "$spid" 2>/dev/null; }; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt "$max" ]; then over=$((over + 1)); else over=0; fi
  if [ "$over" -ge 3 ]; then
    echo "VRAM GUARD: $m MiB > $max MiB, killed at $(( $(date +%s) - t0 )) s" | tee -a "runs/$run/vram.txt"
    [ -n "$spid" ] && kill "$spid" 2>/dev/null
    pkill -x hn-cuda
    sleep 5
    over=0
  fi
  if [ $((i % 5)) -eq 0 ]; then
    g=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used,power.draw,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    rss=$(ps -o rss= -C hn-cuda | head -1)
    echo "$(( $(date +%s) - t0 )),$g,$(( ${rss:-0} / 1024 ))" >> "$tel"
  fi
  i=$((i + 1))
  sleep 2
done
echo "peak total VRAM MiB: $peak" | tee -a "runs/$run/vram.txt"
python3 summ.py "$@"
for r in "$@"; do grep -iE "panic|error|nan|VRAM LIMIT" "runs/$r/stdout.log" | head -3; done
for r in "$@"; do grep -h "^prof" "runs/$r/stdout.log" | tail -1; done
