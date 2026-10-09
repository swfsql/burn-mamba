#!/usr/bin/env bash
# usage: knobs.sh   (a host job: ~2 s per variant)
# The budget table of the knobs: one change at a time from the base b
# (stage d128 2+2 r32, main d136 × 3 r64, N 2, untied head, init burn).
# For each variant k-<name>: writes cfg/k-<name> (gen.py), appends its param
# count to knobs-params.txt, and prints the queue line of its probe (10
# steps, `probe.sh`, into knobs.txt), with the gradient checkpointing (the
# default of args.sh), and one probe of the base without it. Then `knobs.py`
# joins both into the table of Δparams and Δreserved MB against k-base.
set -euo pipefail
cd "$(dirname "$0")"
base="tie=0 init=burn d=128 md=136 mL=3 N=2 lr=8e-3"
# name|gen.py keys|probe example flags
variants=(
  "base||"
  "sr64|r=64|"
  "mr128|mr=128|"
  "srotQ|srot=Quaternion4D|"
  "srotR|srot=Rotor4D|"
  "srot1|srot=Real1D|"
  "mrotQ|mrot=Quaternion4D|"
  "mrotR|mrot=Rotor4D|"
  "rf05|rf=0.5|"
  "se3|e=3|"
  "me3|me=3 mp=24|"
  "mv6|mV=6 muntie=1|"
  "sm2|sm=2|"
  "mm2|mm=2|"
  "mm4|mm=4|"
  "sp32|p=32|"
  "mp34|mp=34|"
  "mmlp|mmlp=256|"
  "mg4|mgr=4 mmgr=4|"
  "su2|su=2|"
  "mu2|mu=2|"
  "trapN|trap=None|"
  "N15|N=1.5|"
  "N3|N=3|"
)
for v in "${variants[@]}"; do
  IFS='|' read -r name keys flags <<< "$v"
  cfg=k-$name
  # shellcheck disable=SC2086
  python3 gen.py "$cfg" $base $keys > /dev/null
  ./params.sh "$cfg" >> knobs-params.txt
  echo "./probe.sh $cfg -- --chunk-log 0 $flags >> knobs.txt"
done
# The base without the gradient checkpointing (the default of args.sh).
echo "./probe.sh k-base -- --chunk-log 0 --no-gc >> knobs.txt"
