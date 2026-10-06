#!/usr/bin/env python3
"""usage: cool.py <decay-run> [<decay-run> ...]

Reads runs/<run>/metrics.jsonl of decay branches (`-d`). The branch starts at
S (the step of the last `start` event) and has N2 steps. The train loss of
each step goes into one of 10 bins of the fraction (step - S) / N2. The
cosine of decay.py spans the first 90% (bins 0-8), and the last bin is the
tail at min_lr.

Prints per run: N2, the mean train bits of bin 0, then the mean of each bin
minus bin 0, and `tail` = bin 9 - bin 8 with its s.e. A tail drop well below
0 means the loss still falls when the LR is already at its minimum: the
weights had not settled, so the cooldown was too short.
"""
import json
import math
import os
import sys

LN2 = math.log(2)
BINS = 10


def decay_part(run):
    """S and the train rows of the last segment (the decay branch). The
    metrics hold the whole history, and each segment opens with a `start`
    event at its first step."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", run, "metrics.jsonl")
    rows = [json.loads(l) for l in open(path)]
    s = [r["step"] for r in rows if r.get("event") == "start"][-1]
    return s, [r for r in rows if r.get("event") == "train" and r["step"] > s]


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    print(f"{'run':<22} {'N2':>5} {'bin0':>7} " + " ".join(f"{b:>6}" for b in range(1, BINS)) + "    tail ± s.e.")
    for run in runs:
        s, seg = decay_part(run)
        n2 = seg[-1]["step"] - s
        bins = [[] for _ in range(BINS)]
        for r in seg:
            b = min(BINS - 1, (r["step"] - s - 1) * BINS // n2)
            bins[b].append(r["loss"] / LN2)
        means = [sum(x) / len(x) for x in bins]
        var = [sum((v - m) ** 2 for v in x) / (len(x) - 1) / len(x) for x, m in zip(bins, means)]
        tail = means[-1] - means[-2]
        se = math.sqrt(var[-1] + var[-2])
        cells = " ".join(f"{m - means[0]:>+6.3f}" for m in means[1:])
        print(f"{run:<22} {n2:>5} {means[0]:>7.4f} {cells}   {tail:+.4f} ± {se:.4f}")


if __name__ == "__main__":
    main()
