#!/usr/bin/env bash
# usage: nohup ./queue.sh > /dev/null 2>&1 &
# Runs the lines of queue.txt one at a time, the first line first. Each line is
# a shell command that runs in the foreground (screen.sh, extend.sh, a flex
# job, …). Logs `start` / `done` / `fail rc=N` lines into queue.log, and
# `empty` when queue.txt has no more lines. queue.pid holds its pid.
# To add, drop or reorder the pending runs, edit queue.txt (not this script).
# A line that starts with `#` is skipped.
set -uo pipefail
cd "$(dirname "$0")"
echo $$ > queue.pid
while :; do
  line=$(head -n1 queue.txt 2>/dev/null)
  [ -z "$line" ] && break
  sed -i 1d queue.txt
  [[ $line == \#* ]] && continue
  echo "start $(date -Is) $line" >> queue.log
  bash -c "$line"
  rc=$?
  if [ "$rc" = 0 ]; then echo "done $(date -Is) $line" >> queue.log
  else echo "fail $(date -Is) rc=$rc $line" >> queue.log; fi
done
echo "empty $(date -Is)" >> queue.log
