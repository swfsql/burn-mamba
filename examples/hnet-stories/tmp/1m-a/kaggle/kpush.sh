#!/usr/bin/env bash
# usage: kaggle/kpush.sh [budget-hours]   (default 10.5; a host job of ~1-3 min)
# Pushes one batch to Kaggle: the lines of kaggle/kqueue.txt, run by the
# private GPU kernel <owner>/hnet-1m-a-runner (runner.py) in a copy of this
# workspace. Refuses while the kernel is queued or running.
# - The payload (/tmp/kaggle-push-<ts>/): bin/hn-cuda (+ .txt), the *.sh /
#   *.py scripts, cfg/, the runs that the batch reads (the source of an
#   extend.sh / branch.sh line, the run of a ctx.sh / infer.sh line), the
#   batch as kqueue.txt, and the budget (seconds) as kaggle-budget.txt. It
#   goes up as a new version of the private dataset <owner>/hnet-payload.
# - Then the kernel is pushed (a new version runs at once). The lines move
#   from kqueue.txt to kbatches.txt, under a `== <ts>` header.
# The lines are those of queue.txt (screen.sh, extend.sh, probe.sh, …), and
# their runs/ names must be new on both hosts.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ws=$(dirname "$here")
owner=$(tr -d '\n' < "$here/owner.txt")
kernel=$owner/hnet-1m-a-runner
budget=$(python3 -c "print(round(${1:-10.5} * 3600))")
status=$("$here/kg.sh" kernels status "$kernel" 2>&1 || true)
case "$status" in
  *QUEUED*|*RUNNING*|*queued*|*running*) echo "busy: $status"; exit 1 ;;
esac
lines=$(grep -v '^#' "$here/kqueue.txt" | grep -v '^[[:space:]]*$' || true)
[ -n "$lines" ] || { echo "kqueue.txt has no lines"; exit 1; }
# The runs that the batch reads, less those that it creates itself (a ctx.sh
# after a screen.sh on the same line reads a run of the batch).
srcs=$(python3 -c '
import shlex, sys
reads, makes = set(), set()
for line in sys.stdin.read().splitlines():
    # A line can chain steps with &&: read each step alone.
    for step in line.split("&&"):
        w = shlex.split(step)
        if not w:
            continue
        # screen.sh <run> / extend.sh <src> <new> / branch.sh <src> <new>:
        # the new run and its decay branch.
        if w[0] == "./screen.sh" and len(w) > 1:
            makes |= {w[1], w[1] + "-d"}
        if w[0] in ("./extend.sh", "./branch.sh") and len(w) > 2:
            makes |= {w[2], w[2] + "-d"}
        # extend.sh / branch.sh / infer.sh: the source run is the first argument.
        if w[0] in ("./extend.sh", "./branch.sh", "./infer.sh") and len(w) > 1:
            reads.add(w[1])
        # ctx.sh: the run is the first argument that is no option (-g) and no
        # option value (-n N, -b N, -r ROOT).
        if w[0] == "./ctx.sh":
            args, i = w[1:], 0
            while i < len(args):
                if args[i] in ("-n", "-b", "-r"):
                    i += 2
                elif args[i].startswith("-"):
                    i += 1
                else:
                    reads.add(args[i])
                    break
print("\n".join(sorted(reads - makes)))
' <<< "$lines")
ts=$(date +%Y%m%d-%H%M%S)
pay=/tmp/kaggle-push-$ts
mkdir -p "$pay/ws/bin" "$pay/ws/runs" "$pay/data" "$pay/kernel"
cp "$ws/bin/hn-cuda" "$ws/bin/hn-cuda.txt" "$pay/ws/bin/"
cp "$ws"/*.sh "$ws"/*.py "$pay/ws/"
cp -r "$ws/cfg" "$pay/ws/"
for s in $srcs; do
  [ -d "$ws/runs/$s" ] || { echo "missing source run: runs/$s"; exit 1; }
  cp -r "$ws/runs/$s" "$pay/ws/runs/"
done
printf '%s\n' "$lines" > "$pay/ws/kqueue.txt"
echo "$budget" > "$pay/ws/kaggle-budget.txt"
tar czf "$pay/data/payload.tar.gz" -C "$pay/ws" .
cat > "$pay/data/dataset-metadata.json" <<EOF
{"title": "hnet-payload", "id": "$owner/hnet-payload", "licenses": [{"name": "CC0-1.0"}]}
EOF
echo "payload: $(du -h "$pay/data/payload.tar.gz" | cut -f1), sources: ${srcs:-none}"
if "$here/kg.sh" datasets status "$owner/hnet-payload" > /dev/null 2>&1; then
  "$here/kg.sh" datasets version -p "$pay/data" -m "batch $ts"
else
  "$here/kg.sh" datasets create -p "$pay/data"
fi
# The kernel must read the new version: wait until the dataset is ready.
for _ in $(seq 60); do
  s=$("$here/kg.sh" datasets status "$owner/hnet-payload" 2>&1 || true)
  [ "$s" = ready ] && break
  sleep 5
done
[ "$s" = ready ] || { echo "the payload is not ready: $s"; exit 1; }
cp "$here/runner.py" "$pay/kernel/"
cat > "$pay/kernel/kernel-metadata.json" <<EOF
{"id": "$kernel", "title": "hnet-1m-a-runner", "code_file": "runner.py",
 "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_tpu": false, "enable_internet": false,
 "dataset_sources": ["$owner/hnet-corpus", "$owner/hnet-payload"],
 "competition_sources": [], "kernel_sources": [], "model_sources": []}
EOF
"$here/kg.sh" kernels push -p "$pay/kernel" -t 43200
{ echo "== $ts $pay"; printf '%s\n' "$lines"; } >> "$here/kbatches.txt"
: > "$here/kqueue.txt"
echo "pushed batch $ts ($(wc -l <<< "$lines") lines)"
