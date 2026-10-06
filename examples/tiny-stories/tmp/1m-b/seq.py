#!/usr/bin/env python3
"""Rewrite the old LR form of a stable run or a cfg as an LR sequence.

usage: seq.py <run | cfg/<name>> ...

The old form is a `CosineAnnealing` over 10M steps: a warmup, then a rate
that does not change (min_lr = max_lr, or a leftover min_lr after a
`--max-lr`). The sequence: a `Linear` warmup to the first rate, then a
`Constant` segment from each resume (a `start` event of metrics.jsonl) where
the logged rate changes. A cfg has no log, so it gets the warmup and its
rate only. The old file stays as training_config.old.json. A decay branch
(a cosine of fewer than 10^6 steps) is refused: nothing extends a leaf.
"""
import json
import os
import shutil
import sys

from lrseq import HERE, load, save

LONG = 10**6


def clean(lr):
    """A logged rate without the drift of the old form: 4 significant
    digits (every rate of the protocol has 3 or fewer)."""
    return float(f"{lr:.4g}")


def history(metrics, warm, first):
    """The `Constant` segments after the warmup: one per resume where the
    logged rate changes by more than 1%."""
    rows = [json.loads(line) for line in open(metrics)]
    starts = sorted({r["step"] for r in rows if r.get("event") == "start"})
    train = [r for r in rows if r.get("event") == "train" and r["step"] > warm]
    out, current = [], first
    for row in train:
        if abs(row["lr"] - current) <= 0.01 * current:
            continue
        resume = max((s for s in starts if s < row["step"]), default=None)
        if resume is None or row["step"] != resume + 1:
            sys.exit(f"{metrics}: the rate changes at step {row['step']}, not at a resume")
        current = clean(row["lr"])
        out.append({"from": resume, "lr": {"Constant": {"lr": current}}})
    return out, current


def convert(target):
    is_cfg = target.startswith("cfg/")
    base = os.path.join(HERE, target if is_cfg else os.path.join("runs", target))
    path = os.path.join(base, "training_config.json")
    tc = load(path)
    lr = tc["training"]["lr"]
    cosine = lr.get("CosineAnnealing")
    if cosine is None:
        print(f"{target}: not the old form, left as is: {list(lr)}")
        return
    if cosine["total_steps"] < LONG:
        sys.exit(f"{target}: a decay branch (total_steps {cosine['total_steps']}), refused")
    warm = cosine["warmup_steps"]
    metrics = os.path.join(base, "metrics.jsonl")
    if is_cfg or not os.path.exists(metrics):
        if cosine["min_lr"] != cosine["max_lr"]:
            sys.exit(f"{target}: no log, and min_lr != max_lr: the rate is not known")
        first, stages, last = cosine["max_lr"], [], cosine["max_lr"]
    else:
        train = [json.loads(line) for line in open(metrics)]
        first = clean(next(r["lr"] for r in train if r.get("event") == "train" and r["step"] > warm))
        stages, last = history(metrics, warm, first)
        if abs(last - cosine["max_lr"]) > 0.01 * last:
            sys.exit(f"{target}: the last logged rate {last} is not max_lr {cosine['max_lr']}")
    warmup = {"from": 0, "lr": {"Linear": {"start": 0.0, "end": first, "steps": warm}}}
    tc["training"]["lr"] = {"Sequence": [warmup] + stages}
    shutil.copyfile(path, os.path.join(base, "training_config.old.json"))
    save(path, tc)
    print(f"{target}: {tc['training']['lr']}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for target in sys.argv[1:]:
        convert(target)


if __name__ == "__main__":
    main()
