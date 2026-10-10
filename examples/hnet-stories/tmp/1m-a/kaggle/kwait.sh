#!/usr/bin/env bash
# usage: kaggle/kwait.sh <budget-hours> <log>   (run it in the background)
# Pushes kaggle/kqueue.txt (kpush.sh), polls the kernel status until it
# ends, pulls and merges the batch (kpull.sh; its filtered output in
# <log>.pull), then queues long.sh for each pulled `-d` run (`>> long.txt`)
# in the local queue.
# Prints KWAIT-DONE at the end, PUSH FAILED if the push did not go out.
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/.."
hours=$1 log=$2
kernel=$(tr -d '\n' < "$here/owner.txt")/hnet-1m-a-runner
timeout 1500 kaggle/kpush.sh "$hours" 2>&1 | tr '\r' '\n' | grep --line-buffered -v "%|" > "$log"
tail -n 2 "$log"
grep -q "pushed batch" "$log" || { echo "PUSH FAILED"; exit 1; }
sleep 60
while :; do
  s=$(timeout 60 kaggle/kg.sh kernels status "$kernel" 2>&1 || true)
  case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) echo "$s"; break ;; esac
  sleep 120
done
kaggle/kpull.sh 2>&1 | grep -E "^(done|fail)|^==|valid@|merged" | tee "$log.pull"
# The long.sh runs go through the local queue (queue.txt), and the queue
# starts when it does not run. A GPU job beside a probe or a dose spoils its
# VRAM count.
for r in $(grep -o "^== [^:]*-d:" "$log.pull" | sed 's/^== //; s/:$//'); do
  [ -e "long/$r" ] || echo "./long.sh -g $r >> long.txt" >> queue.txt
done
if ! kill -0 "$(cat queue.pid 2>/dev/null)" 2>/dev/null; then
  setsid nohup ./queue.sh > /dev/null 2>&1 < /dev/null &
fi
echo KWAIT-DONE
