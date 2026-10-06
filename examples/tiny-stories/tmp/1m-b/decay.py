#!/usr/bin/env python3
"""Write cfg/<cfg>/training_config.json: a decay branch of runs/<src>.

usage: decay.py <src-run> <cfg> <D-schedule-steps> [min_lr]

S = the step saved in runs/<src>/progress.json. The LR sequence of the run
(lrseq.py) keeps its segments before S. From S, a cosine runs from the rate
at S to min_lr over D steps, and holds min_lr after them. The segments of
the run from S on (planned step-downs) go. Default min_lr = the rate at S /
100.
"""
import os
import sys

from lrseq import HERE, cut, load, rate, run_path, save, saved_step, segments

src, cfg, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = saved_step(src)
tc = load(run_path(src, "training_config.json"))
# The rate of step S, as the run trained it: a segment that starts at S
# (a planned step-down) sets only the steps after S.
top = rate({"Sequence": [x for x in segments(tc) if x["from"] < s]}, s)
min_lr = float(sys.argv[4]) if len(sys.argv) > 4 else top / 100
cosine = {"max_lr": top, "min_lr": min_lr, "warmup_steps": 0, "total_steps": d}
cut(tc, s, {"CosineAnnealing": cosine})
out = os.path.join(HERE, "cfg", cfg)
os.makedirs(out, exist_ok=True)
save(os.path.join(out, "training_config.json"), tc)
with open(os.path.join(out, "args.txt"), "w") as f:
    f.write(f"decay of {src}: S={s} D={d} {cosine}\n")
print(out, cosine)
