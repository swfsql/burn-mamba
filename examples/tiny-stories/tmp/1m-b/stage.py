#!/usr/bin/env python3
"""Start a constant rate in the LR schedule of runs/<run>.

usage: stage.py <run> <lr> [from]

From the step `from` (default: the step in runs/<run>/progress.json), the
schedule in runs/<run>/training_config.json is the constant `lr`. The
segments that start at `from` or later go. A step-down of an extension:
`extend.sh --lr`, which runs this on the new run before it resumes. A
planned step-down: a `from` past the saved step.
"""
import sys

from lrseq import cut, load, run_path, save, saved_step

run, lr = sys.argv[1], float(sys.argv[2])
step = int(sys.argv[3]) if len(sys.argv) > 3 else saved_step(run)
path = run_path(run, "training_config.json")
tc = load(path)
cut(tc, step, {"Constant": {"lr": lr}})
save(path, tc)
print(f"stage {run}: {lr} from step {step}: {tc['training']['lr']}")
