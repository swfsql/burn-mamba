#!/usr/bin/env python3
"""Compact summary of runs/<name>/metrics.jsonl (one or more run names).

Per run: steps, wall time (training only, excluding the start's corpus load),
steps/s, train bits/char averaged over trailing windows of steps, and every
validation (step, bits, acc, batches). `-c N` sets the number of train buckets.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2)


def load(name):
    path = os.path.join(HERE, "runs", name, "metrics.jsonl")
    rows = [json.loads(l) for l in open(path)]
    return rows


def summ(name, buckets):
    rows = load(name)
    train = [r for r in rows if r["event"] == "train"]
    valid = [r for r in rows if r["event"] == "valid"]
    starts = [r for r in rows if r["event"] == "start"]
    print(f"== {name}: {len(starts)} segment(s), {len(train)} train rows")
    if train:
        # elapsed restarts per segment; sum per-segment spans
        span, seg_first, prev = 0.0, train[0]["elapsed"], train[0]["elapsed"]
        for r in train[1:]:
            if r["elapsed"] < prev:  # new segment
                span += prev - seg_first
                seg_first = r["elapsed"]
            prev = r["elapsed"]
        span += prev - seg_first
        n = len(train)
        print(f"   steps {train[0]['step']}..{train[-1]['step']}  train-time {span/60:.1f} min"
              f"  {n/span if span else float('nan'):.2f} steps/s")
        k = max(1, n // buckets)
        line = []
        for i in range(0, n, k):
            chunk = train[i:i + k]
            b = sum(r["loss"] for r in chunk) / len(chunk) / LN2
            line.append(f"{chunk[-1]['step']}:{b:.3f}")
        print("   train bits: " + " ".join(line))
    for r in valid:
        print(f"   valid@{r['step']} ({r['elapsed']/60:.1f}m) {r.get('bits', float('nan')):.4f} bits"
              f"  acc {r.get('acc', float('nan')):.2f}  n={int(r.get('batches', 0))}")


if __name__ == "__main__":
    args, buckets = sys.argv[1:], 8
    if args and args[0] == "-c":
        buckets, args = int(args[1]), args[2:]
    for name in args:
        summ(name, buckets)
