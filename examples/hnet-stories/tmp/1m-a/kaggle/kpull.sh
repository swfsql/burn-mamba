#!/usr/bin/env bash
# usage: kaggle/kpull.sh
# Prints the status of the kernel <owner>/hnet-1m-a-runner. When it is
# complete (or failed or cancelled), fetches its outputs into a fresh
# kaggle/out/<ts>/ and merges them into this workspace:
# - each runs/<name> to runs/<name> (a name that exists here is kept only in
#   kaggle/out/<ts>/, with a warning),
# - each ctxg/<run> (ctx.sh copies) to ctxg/<run>,
# - the text outputs (probes.txt, knobs.txt, ctxg.txt, speeds.txt) appended
#   to kaggle/<name> (the Kaggle GPU differs: they stay apart).
# Then prints kqueue.log of the kernel and summ.py of the merged runs. An
# output that was already merged (kaggle/pulled.txt) is not merged again.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ws=$(dirname "$here")
owner=$(tr -d '\n' < "$here/owner.txt")
kernel=$owner/hnet-1m-a-runner
status=$("$here/kg.sh" kernels status "$kernel" 2>&1 || true)
echo "$status"
case "$status" in
  *COMPLETE*|*ERROR*|*CANCEL*|*complete*|*error*|*cancel*) ;;
  *) exit 0 ;;
esac
ts=$(date +%Y%m%d-%H%M%S)
out=$here/out/$ts
mkdir -p "$out"
"$here/kg.sh" kernels output "$kernel" -p "$out" > /dev/null
# The kernel log identifies an output: skip one that was merged already.
klog=$out/hnet-1m-a-runner.log
key=$(cat "$out/kqueue.log" "$klog" 2>/dev/null | head -c 65536 | sha256sum | cut -c1-16)
if grep -qx "$key" "$here/pulled.txt" 2>/dev/null; then
  echo "already merged ($key); the fresh copy stays in $out"
  exit 0
fi
if [ ! -f "$out/kqueue.log" ]; then
  # The runner stopped before its queue: the end of the kernel log says why.
  echo "no kqueue.log: the end of the kernel log ($klog)"
  python3 -c "
import json, sys
for e in json.load(open(sys.argv[1]))[-15:]:
    print(e.get('stream_name', '?'), e.get('data', '').rstrip()[:400])
" "$klog" 2>/dev/null || tail -c 3000 "$klog"
  echo "$key" >> "$here/pulled.txt"
  exit 1
fi
merged=()
for d in "$out"/runs/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if [ -e "$ws/runs/$name" ]; then
    echo "exists: runs/$name (kept in $out/runs/$name)"
  else
    cp -r "$d" "$ws/runs/$name"
    merged+=("$name")
  fi
done
for d in "$out"/ctxg/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$ws/ctxg"
  [ -e "$ws/ctxg/$name" ] || cp -r "$d" "$ws/ctxg/$name"
done
for f in probes.txt knobs.txt ctxg.txt speeds.txt; do
  [ -f "$out/$f" ] && cat "$out/$f" >> "$here/$f"
done
echo "$key" >> "$here/pulled.txt"
cat "$out/kqueue.log" 2>/dev/null || echo "no kqueue.log (see $out)"
[ ${#merged[@]} -gt 0 ] && (cd "$ws" && python3 summ.py "${merged[@]}" | tail -40)
echo "merged ${#merged[@]} runs from $out"
