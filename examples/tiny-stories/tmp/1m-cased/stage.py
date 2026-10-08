#!/usr/bin/env python3
"""Start a step-down in the LR schedule of runs/<run>.

usage: stage.py <run> <lr> [from | A-B]

From the step `from` (default: the step in runs/<run>/progress.json), the
schedule in runs/<run>/training_config.json is the constant `lr`. With
`A-B`, it is an anneal: a cosine from the rate at A to `lr` at B, which then
holds `lr`. The segments that start at `from` (or A) or later go. A
step-down of an extension: `extend.sh --lr`, which runs this on the new run
before it resumes. A planned step-down: a `from` past the saved step.
"""
import sys

from lrseq import cut, load, run_path, save, saved_step, segments, stage

run, lr = sys.argv[1], float(sys.argv[2])
span = sys.argv[3] if len(sys.argv) > 3 else str(saved_step(run))
path = run_path(run, "training_config.json")
tc = load(path)
start = int(span.split("-")[0])
kept = [s for s in segments(tc) if s["from"] < start]
_, segment = stage(kept, span, lr)
cut(tc, start, segment["lr"])
save(path, tc)
print(f"stage {run}: {lr} from {span}: {tc['training']['lr']}")
