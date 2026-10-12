#!/usr/bin/env python3
"""The catalog: one line per model (the shape of its config), with its params
and its dose.

usage: catalog.py [name …] > catalog.txt   (no names: the models of NOTES)

The shape comes from cfg/<name>/args.txt on the defaults of gen.py (so it
follows the config), λ from its model_config.json. The params: the newest
params/<name>-<ts>/ count (params.sh). The dose: the newest `peak max` of
doses.txt (dose.sh), in MB. The role notes are the NOTES below (hand-kept).

The shape line:
- `S1 | stage d128 2+2 R1D r104 h16 | N 2 | main d128×3 e2 C2D r64 h16
  mlp8 | λ 1.41/1`. Per-stage values are joined by `/` (outermost first).
- `h` the heads (d·e/p), `e` the expand, `2+2` encoder + decoder layers,
  `R1D` / `C2D` / `Q4D` / `R4D` the rotation kind, `mlpN` the main MLP
  (hidden N), `u`/`m` micro_steps / MIMO when > 1.
- Then the keys off the recipe of the protocol (tie, init, lr, buckets,
  seed, …).
"""
import glob
import json
import os
import re
import sys

from gen import DEFAULTS, parse
from lrseq import HERE

# The models of the catalog (in this order) and their role notes. A run
# result belongs in notes.md, not here.
NOTES = {
    "R-N15-B5": "the best shape: R-B5 at N 1.5 (seed 0)",
    "R-N15-B5-s1": "R-N15-B5, seed 1",
    "R-B5": "the N 2 reference (the base of the S2 / wide-stage variants)",
    "R-N15-B": "R-N15-B5 at 7 length slots",
    "R-N15": "R-N15-B5 on the old data (first 1024 chars, invalid)",
    "M-s96r128-L5": "M family: stage d96, a main d112 × 5",
    "Nm4-s96r128-B5": "M family: stage d96, a main d128 × 4",
    "M-s96r128-L4r256": "M family: stage d96, a main d112 × 4 r 256",
    "M-s64r256-L4": "M family: stage d64 r 256, a main d144 × 4",
    "R-B5-s144": "wide stage: d144 at 16 heads, the main pays (expand 1)",
    "R-B5-s144b": "wide stage: d144 at R-B5's head dim (18 heads)",
    "S2a-B5": "two stages at total N 2 (stopped: ~1 GB under)",
    "S2a2-B5": "two stages at total N 2, refilled to the VRAM band",
    "S2x-s112r256-L4": "deep compression N 4/4, d112 (under the band)",
    "S2x-e21h16-L4": "deep compression N 4/4, d112, outer 2+1 at 16 heads",
    "S2x-e21h16-L4-lrm": "S2x-e21h16-L4 with λ capped (1.41/1.19/1)",
    "S2x-s64e2-L4": "deep compression N 4/4, d64 (under the band)",
    "S4x-s64e2-L4": "four stages at N 2, d64 (under the band)",
    "tmp-v-B2": "S2 N 4/4 d64, outer 3+3 + inner 2+2 (one page under)",
    "tmp-v-C2": "S4 N 2 d64, outer 3+3, r 256 ×4, main d64 (under, 1.012M)",
    "R-N15-r64V7": "main reuse: stage r 64, the main × 7 applications",
    "R-N15-r64V7-s1": "R-N15-r64V7, seed 1",
    "R-N15-r64sV4": "stage reuse at r 64: 4+4 applications (under the band)",
    "R-N15-r64sV5": "stage reuse at r 64: 5+5 applications",
    "R-N15-r128s11V2-m5": "stage reuse at r 128: 1+1 real × 2+2, main × 5 "
                          "(one page under)",
    "R-N125-r128s11V2-m5": "R-N15-r128s11V2-m5 at N 1.25 (the page)",
    "R-N125-r128s11V2-m5-s1": "R-N125-r128s11V2-m5, seed 1",
    "R-N15-r128s11V2-m5V6c-st2": "stage + main reuse: R-N15-r128s11V2-m5 "
                                 "× main V 6 (Cyclic), 2 class tokens",
    "R-N125-r128s11V2-m5V6c-st2": "the same at N 1.25 (over the band)",
}
# The flat reference: another workspace and another gen.py (by hand).
FLAT = ("h16-V7u", 995921, 3455,
        "flat | d160×5 V7 (Cyclic, untied) e2 C2D r64 h16 (p 20) | packed "
        "2 × 4224, gc",
        "the flat reference (../../../tiny-stories/tmp/1m-cased)")
# A model dosed under another config of the same VRAM (mmlp is VRAM-free).
DOSE_AS = {"S2x-e21h16-L4": "tmp-v-P4", "S2x-e21h16-L4-lrm": "tmp-v-P4"}
ROT = {"Real1D": "R1D", "Complex2D": "C2D", "Quaternion4D": "Q4D", "Rotor4D": "R4D"}
# The recipe of the protocol: a key off these values shows on the line.
RECIPE = dict(tie=0, init="burn", lr=4e-3, mlpmul=8)


def args(name):
    a = dict(DEFAULTS)
    a.update(RECIPE)
    with open(os.path.join(HERE, "cfg", name, "args.txt")) as f:
        for item in f.read().split():
            k, v = item.split("=", 1)
            a[k] = v if isinstance(DEFAULTS[k], str) else parse(v)
    return a


def stage_vals(a, key):
    vals = str(a[key]).split("/")
    return (vals * a["S"])[: a["S"]] if len(vals) == 1 else vals[: a["S"]]


def num(v):
    return f"{float(v):g}"


def shape(name):
    a = args(name)
    S = a["S"]
    d, e, p = (stage_vals(a, k) for k in ("d", "e", "p"))
    enc, dec, r = (stage_vals(a, k) for k in ("enc", "dec", "r"))
    rot, u, m = (stage_vals(a, k) for k in ("srot", "su", "sm"))
    heads = [str(int(d[s]) * int(e[s]) // int(p[s])) for s in range(S)]
    j = "/".join
    st = (f"{'stage' if S == 1 else 'stages'} d{j(d)} "
          f"{j(f'{enc[s]}+{dec[s]}' for s in range(S))} {j(ROT.get(x, x) for x in rot)} "
          f"r{j(r)} h{j(heads)}")
    if set(e) != {"2"}:
        st += f" e{j(e)}"
    if set(u) != {"1"}:
        st += f" u{j(u)}"
    if set(m) != {"1"}:
        st += f" m{j(m)}"
    if set(stage_vals(a, "smlp")) != {"0"}:
        st += f" mlp{j(stage_vals(a, 'smlp'))}"
    ev, dv = stage_vals(a, "eV"), stage_vals(a, "dV")
    if set(ev) != {"0"} or set(dv) != {"0"}:
        st += f" V{j(f'{ev[s]}+{dv[s]}' for s in range(S))}"
    mh = a["md"] * a["me"] // a["mp"]
    main = (f"main d{a['md']}×{a['mL']} e{a['me']} {ROT.get(a['mrot'], a['mrot'])} "
            f"r{a['mr']} h{mh}")
    if a["mmlp"]:
        main += f" mlp{a['mmlp']}"
    if a["mV"]:
        main += f" V{a['mV']} ({a['msched']})"
    for k, tag in (("mu", "u"), ("mm", "m"), ("mmgr", "mgr")):
        if a[k] not in (0, 1):
            main += f" {tag}{a[k]}"
    with open(os.path.join(HERE, "cfg", name, "model_config.json")) as f:
        lrm = json.load(f)["lr_multipliers"]
    parts = [f"S{S}", st, f"N {j(num(x) for x in stage_vals(a, 'N'))}", main,
             f"λ {j(f'{x:.3g}' for x in lrm)}"]
    extra = [f"{k}={a[k]}" for k in RECIPE if a[k] != RECIPE[k]]
    if a["buckets"] != DEFAULTS["buckets"]:
        extra.append("first 1024 chars (old)" if not a["buckets"]
                     else f"{len(str(a['buckets']).split(','))} slots")
    for k in ("seed", "trap", "bias", "alpha", "wd", "rr", "rf", "mgr", "start"):
        if a[k] != DEFAULTS[k]:
            extra.append(f"{k}={a[k]}")
    if extra:
        parts.append(" ".join(extra))
    return " | ".join(parts)


def params(name):
    dirs = [x for x in glob.glob(os.path.join(HERE, "params", name + "-*"))
            if re.fullmatch(re.escape(name) + r"-\d+", os.path.basename(x))]
    for d in sorted(dirs, key=lambda x: int(x.rsplit("-", 1)[1]), reverse=True):
        try:
            m = re.search(r"Number of parameters: (\d+)", open(os.path.join(d, "stdout.log")).read())
        except OSError:
            continue
        if m:
            return int(m[1])
    return None


def dose(name):
    best = "-"
    for line in open(os.path.join(HERE, "doses.txt")):
        m = re.match(re.escape(name) + r"\s+dose\s.*peak max\s+(\d+) MB", line)
        if m:
            best = m[1]
    return best


def main():
    names = sys.argv[1:] or list(NOTES)
    rows = []
    for n in names:
        if not os.path.isdir(os.path.join(HERE, "cfg", n)):
            print(f"{n}: no cfg/{n}", file=sys.stderr)
            continue
        p = params(n)
        rows.append((n, p if p else "-", dose(DOSE_AS.get(n, n)), shape(n), NOTES.get(n, "")))
    if not sys.argv[1:]:
        rows.append(FLAT)
    w = max(len(r[0]) for r in rows)
    print(f"{'model':<{w}}  {'params':>9}  {'dose MB':>7}  shape  — role")
    for n, p, d, s, note in rows:
        print(f"{n:<{w}}  {p:>9}  {d:>7}  {s}" + (f"  — {note}" if note else ""))


if __name__ == "__main__":
    main()
