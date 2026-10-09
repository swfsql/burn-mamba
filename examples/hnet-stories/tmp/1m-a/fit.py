#!/usr/bin/env python3
"""Fill the param budget: the largest value of one key that keeps the params
under the cap.

usage: fit.py <name> <key> [from=V] [step=S] [cap=C] k=v …
  e.g. `fit.py F-sr80 mmlp from=0 step=32 tie=0 r=80`

It walks `key` up from `from` (default 64) by `step` (default 8: the head
split of a width, `me·md / mp`) until the params reach the cap (default
1,000,000). Each try writes cfg/<name> with gen.py and counts its params
with params.sh (flex, ~2 s). A shape that does not build (no count) is
skipped. The best value stays in cfg/<name>, and the last line is
`<name>: <key>=<v> <params>`.

The fillers: `md` (~48k per 8 at md 136, VRAM too), `mmlp` (the main MLP,
~1.2k per unit at md 136 × 3, VRAM-free).
"""
import re
import subprocess
import sys

from lrseq import HERE

name, key, *rest = sys.argv[1:]
opts = dict(start=64, step=8, cap=1_000_000)
args = []
for kv in rest:
    k, _, v = kv.partition("=")
    if k in ("from", "step", "cap"):
        opts["start" if k == "from" else k] = int(v)
    elif k != key:
        args.append(kv)


def count(v):
    subprocess.run([sys.executable, f"{HERE}/gen.py", name, *args, f"{key}={v}"],
                   check=True, capture_output=True)
    out = subprocess.run([f"{HERE}/params.sh", name], capture_output=True, text=True).stdout
    m = re.search(r"Number of parameters: (\d+)", out)
    n = int(m.group(1)) if m else None
    print(f"  {key}={v}: {n if n is not None else 'no count'}", flush=True)
    return n


best, v = None, opts["start"]
while v <= opts["start"] + 64 * opts["step"]:
    n = count(v)
    if n is not None and n >= opts["cap"]:
        break
    if n is not None:
        best = (v, n)
    v += opts["step"]
if best is None:
    sys.exit(f"{name}: no {key} fits under {opts['cap']}")
count(best[0])
print(f"{name}: {key}={best[0]} {best[1]}")
