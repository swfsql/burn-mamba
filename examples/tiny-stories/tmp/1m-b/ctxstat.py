#!/usr/bin/env python3
"""usage: ctxstat.py <dir> [<dir> ...]

Reads <dir>/context-stories.jsonl (a copy dir of ctx.sh, e.g. ctxg/<run>).
For each run, prints the mean ± s.e. over the stories of the far part
(pairing i+1): own, none, gain = none - own, penalty = other - none,
use = other - own. Then for each run after the first, prints the paired
difference (run - first run) on the stories that both scored. The pairing
i+1 depends on the set of stories, so compare runs with the same -n.

With the re-mention fields (`rem*`, the re-mentioned words of the own
prompt, see context.rs), it also prints them per character, pooled over the
stories (sum of bits / sum of characters, s.e. with the story as the
cluster): `letters` (every letter) and `first` (the first letter of each
word), own, none, gain = none - own, use = other - own. Then the paired
difference of own and gain to the first run.
"""
import json
import math
import os
import sys

KEYS = ["own", "none", "gain", "penalty", "use"]
REM = [("rem", "letters"), ("rem1", "first")]
REM_KEYS = ["own", "none", "gain", "use"]


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
            for p, _ in REM:
                if f"{p}_n" in r:
                    own, none, other = r[f"{p}_own"], r[f"{p}_none"], r[f"{p}_other"]
                    rows[r["story"]][p] = {
                        "n": r[f"{p}_n"],
                        "own": own,
                        "none": none,
                        "gain": none - own,
                        "use": other - own,
                    }
    return rows


def ratio_se(xs, ns):
    """Σ xs / Σ ns, and its s.e. with the stories as the clusters."""
    total = sum(ns)
    r = sum(xs) / total
    return r, math.sqrt(sum((x - r * n) ** 2 for x, n in zip(xs, ns))) / total


def fmt_ratio(xs, ns):
    r, se = ratio_se(xs, ns)
    return f"{r:+.4f} ± {se:.4f}"


def remention(runs, width):
    """The re-mention block, when every run has the fields."""
    if not all(all("rem" in r for r in rows.values()) for _, rows in runs):
        return
    for p, label in REM:
        print(f"re-mention {label} (bits/char, pooled over the characters):")
        print(f"{'run':<{width}}  {'chars':>6}  " + "  ".join(f"{k:>16}" for k in REM_KEYS))
        for name, rows in runs:
            rs = [r[p] for r in rows.values()]
            ns = [r["n"] for r in rs]
            cols = [fmt_ratio([r[k] for r in rs], ns) for k in REM_KEYS]
            print(f"{name:<{width}}  {sum(ns):>6}  " + "  ".join(f"{c:>16}" for c in cols))
        if len(runs) > 1:
            base_name, base = runs[0]
            print(f"paired difference to {base_name} (run - {base_name}):")
            for name, rows in runs[1:]:
                common = sorted(base.keys() & rows.keys())
                if any(rows[s][p]["n"] != base[s][p]["n"] for s in common):
                    print(f"{name:<{width}}  the marks differ (another -n?): no pairing")
                    continue
                ns = [base[s][p]["n"] for s in common]
                cols = [fmt_ratio([rows[s][p][k] - base[s][p][k] for s in common], ns) for k in REM_KEYS]
                print(f"{name:<{width}}  {sum(ns):>6}  " + "  ".join(f"{c:>16}" for c in cols))


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
    remention(runs, width)


if __name__ == "__main__":
    main()
