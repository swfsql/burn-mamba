#!/usr/bin/env python3
"""Compare the long-story bits of runs, story by story.

usage: longstat.py <dir> [<dir> …]   (each dir holds a long-stories.jsonl,
e.g. long/<run>; a bare name means long/<name>)

Per segment (0..1024, 1024..2048, the tail 2048.., all): the bits per
character of each run (all characters pooled), then the paired difference of
each later run against the first: the mean over the stories of the per-story
bits/char difference ± its standard error. Then the position profile of each
run (bits/char per 256 characters, pooled).
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENTS = [("0..1024", 0, 1024), ("1024..2048", 1024, 2048), ("2048..", 2048, None), ("all", 0, None)]


def load(arg):
    path = arg if os.path.isdir(arg) else os.path.join(HERE, "long", arg)
    rows = [json.loads(l) for l in open(os.path.join(path, "long-stories.jsonl"))]
    bin_ = json.load(open(os.path.join(path, "long-stories.json")))["bin"]
    return {r["story"]: r for r in rows}, bin_


def seg(row, bin_, a, b):
    """The bits and the characters of a story in a..b (b None: the end)."""
    n = row["len"]
    b = n if b is None else min(b, n)
    return sum(row["bins"][a // bin_ : -(-b // bin_)]), max(b - a, 0)


def mean_se(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v / len(xs))


def main():
    names = sys.argv[1:]
    runs = [load(a) for a in names]
    stories = sorted(set.intersection(*(set(r[0]) for r in runs)))
    width = max(len(os.path.basename(n.rstrip("/"))) for n in names)
    print(f"{len(stories)} stories in common")
    print(f"{'':{width}} " + "".join(f"{s:>20}" for s, _, _ in SEGMENTS))
    for k, ((rows, bin_), name) in enumerate(zip(runs, names)):
        cells = []
        for _, a, b in SEGMENTS:
            bits, chars = map(sum, zip(*(seg(rows[i], bin_, a, b) for i in stories)))
            cell = f"{bits / chars:.4f}"
            if k > 0:
                (rows0, bin0) = runs[0]
                d = []
                for i in stories:
                    x, n = seg(rows[i], bin_, a, b)
                    y, n0 = seg(rows0[i], bin0, a, b)
                    d.append(x / n - y / n0)
                m, se = mean_se(d)
                cell += f" {m:+.4f}±{se:.4f}"
            cells.append(f"{cell:>20}" if k == 0 else f"{cell:>28}")
        print(f"{os.path.basename(name.rstrip('/')):{width}} " + "".join(cells))
    for (rows, bin_), name in zip(runs, names):
        nb = max(len(rows[i]["bins"]) for i in stories)
        prof = []
        for k in range(nb):
            bits = chars = 0
            for i in stories:
                r = rows[i]
                if k < len(r["bins"]):
                    bits += r["bins"][k]
                    chars += min(bin_, r["len"] - k * bin_)
            prof.append(f"{bits / chars:.3f}")
        print(f"{os.path.basename(name.rstrip('/'))} profile/{bin_}: {' '.join(prof)}")


main()
