#!/usr/bin/env python3
"""usage: ctxstat.py <dir> [<dir> ...]

Reads <dir>/context-stories.jsonl (a copy dir of ctx.sh, e.g. ctxg/<run>).
For each run, prints the mean ± s.e. over the stories of the far part
(pairing i+1): own, none, gain = none - own, penalty = other - none,
use = other - own. Then for each run after the first, prints the paired
difference (run - first run) on the stories that both scored. The pairing
i+1 depends on the set of stories, so compare runs with the same -n.
"""
import json
import math
import os
import sys

KEYS = ["own", "none", "gain", "penalty", "use"]


def load(d):
    rows = {}
    with open(os.path.join(d, "context-stories.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            own, none, other = r["own_far"], r["none_far"], r["other_far"]
            rows[r["story"]] = {
                "own": own,
                "none": none,
                "gain": none - own,
                "penalty": other - none,
                "use": other - own,
            }
    return rows


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def fmt(xs):
    m, se = mean_se(xs)
    return f"{m:+.4f} ± {se:.4f}"


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    runs = [(os.path.basename(d.rstrip("/")), load(d)) for d in dirs]
    width = max(len(name) for name, _ in runs)
    print(f"{'run':<{width}}  {'n':>5}  " + "  ".join(f"{k:>16}" for k in KEYS))
    for name, rows in runs:
        cols = [fmt([r[k] for r in rows.values()]) for k in KEYS]
        print(f"{name:<{width}}  {len(rows):>5}  " + "  ".join(f"{c:>16}" for c in cols))
    if len(runs) > 1:
        base_name, base = runs[0]
        print(f"paired difference to {base_name} (run - {base_name}):")
        for name, rows in runs[1:]:
            common = sorted(base.keys() & rows.keys())
            cols = [fmt([rows[s][k] - base[s][k] for s in common]) for k in KEYS]
            print(f"{name:<{width}}  {len(common):>5}  " + "  ".join(f"{c:>16}" for c in cols))


if __name__ == "__main__":
    main()
