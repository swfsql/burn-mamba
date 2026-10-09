#!/usr/bin/env python3
"""The budget table of the knobs (see knobs.sh): one row per variant, with
its params and reserved MB, and their change against k-base.

usage: knobs.py   (reads knobs-params.txt and knobs.txt)

- params: the count of `params.sh` (the last line per config).
- reserved: cubecl's `bytes_reserved` after step 10 of `probe.sh` (the last
  line per config and flags), with --grad-checkpointing. `k-base nogc` is
  the base without it.
- MB/kparam: Δreserved per 1k Δparams, the price in VRAM of the params of
  the knob (large: a knob that targets the VRAM).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
params, reserved = {}, {}
for line in open(os.path.join(HERE, "knobs-params.txt")):
    m = re.match(r"(\S+): Number of parameters: (\d+)", line)
    if m:
        params[m[1]] = int(m[2])
for line in open(os.path.join(HERE, "knobs.txt")):
    m = re.match(r"(\S+)\s+reserved\s+(\S+) MB.*\[.*-- (.*)\]", line)
    if not m:
        continue
    name = m[1] + ("" if "--grad-checkpointing" in m[3] else " nogc")
    reserved[name] = float(m[2]) if m[2] != "-" else None
p0, r0 = params["k-base"], reserved["k-base"]
print(f"{'variant':<12} {'params':>9} {'Δparams':>9} {'resMB':>7} {'ΔMB':>7} {'MB/kparam':>9}")
for name in sorted(reserved, key=lambda n: (reserved[n] or 0) - r0):
    p = params.get(name.removesuffix(" nogc"))
    r = reserved[name]
    dp = p - p0 if p is not None else None
    dr = r - r0 if r is not None else None
    ratio = f"{dr / (dp / 1000):9.1f}" if dp and dr is not None and abs(dp) >= 500 else f"{'-':>9}"
    print(f"{name:<12} {p if p else '-':>9} {dp if dp is not None else '-':>+9} "
          f"{r if r is not None else '-':>7} {dr if dr is not None else '-':>+7.0f} {ratio}")
