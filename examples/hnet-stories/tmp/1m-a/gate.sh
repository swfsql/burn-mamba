#!/usr/bin/env bash
# usage: gate.sh <cfg> [lo] [hi]   — exit 0 when the last dose of <cfg> is
# in the band.
# Reads the last `doses.txt` line of <cfg> (`peak max N MB`). Exit 0 when
# lo ≤ N ≤ hi (default 3380–3670 MB: the dose target of ORIENTATION.md,
# VRAM, and one page over it). Exit 1 otherwise, or when <cfg> has no dose.
# A queue line uses it to start a screen only after a good dose:
#   `if ./gate.sh A; then ./screen.sh …A…; fi`
cd "$(dirname "$0")"
cfg=$1 lo=${2:-3380} hi=${3:-3670}
n=$(grep -E "^$cfg +dose " doses.txt | tail -n 1 | grep -oE 'peak max +[0-9]+ MB' | grep -oE '[0-9]+')
if [ -z "$n" ]; then echo "gate $cfg: no dose"; exit 1; fi
if [ "$n" -ge "$lo" ] && [ "$n" -le "$hi" ]; then echo "gate $cfg: $n MB in $lo-$hi"; exit 0; fi
echo "gate $cfg: $n MB not in $lo-$hi"
exit 1
