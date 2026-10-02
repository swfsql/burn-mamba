#!/usr/bin/env python3
"""Write cfg/<cfg>/training_config.json: a decay branch of runs/<src>.

usage: decay.py <src-run> <cfg> <D-schedule-steps> [min_lr]

S = the step saved in runs/<src>/progress.json. The cosine becomes
warmup_steps = S, total_steps = S + D, so it decays max_lr → min_lr over [S, S+D]
(and stays at min_lr past it). Default min_lr = max_lr / 100.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src, cfg, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
run = os.path.join(HERE, "runs", src)
s = json.load(open(os.path.join(run, "progress.json")))["step"]
tc = json.load(open(os.path.join(run, "training_config.json")))
lr = tc["training"]["lr"]["CosineAnnealing"]
lr["warmup_steps"], lr["total_steps"] = s, s + d
lr["min_lr"] = float(sys.argv[4]) if len(sys.argv) > 4 else lr["max_lr"] / 100
out = os.path.join(HERE, "cfg", cfg)
os.makedirs(out, exist_ok=True)
json.dump(tc, open(os.path.join(out, "training_config.json"), "w"), indent=2)
open(os.path.join(out, "args.txt"), "w").write(f"decay of {src}: S={s} D={d} {lr}\n")
print(out, lr)
