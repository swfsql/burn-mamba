#!/usr/bin/env bash
# usage: stamp.sh <dir>…
# Writes <dir>/build.txt: the build info of bin/ts-cuda (see build.sh). The
# scripts stamp the directories that they create. Use this for a directory
# made by hand (a ctx*/ copy, an infer-*/ directory).
set -euo pipefail
cd "$(dirname "$0")"
source ./args.sh
for d in "$@"; do stamp "$d"; done
