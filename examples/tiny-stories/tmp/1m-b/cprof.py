#!/usr/bin/env python3
"""usage: cprof.py <dir> [<dir> ...]

Reads <dir>/contrast-profile.jsonl (a copy dir of `ceval.sh ... --
--contrast-profile`). Each line is one contrast row: the last character of
the prompt, then the continuation, and per character the log-probability
(nats) after the own prompt and after the negative, and the entropy of each
prediction.

For each run, prints per character class and per position bin of the
continuation (all in bits per character):
  delta = own - other log-probability (what the contrast pushes),
  dH    = entropy after the negative - entropy after the own prompt,
  own   = the cost of the character after the own prompt.
The classes: `start` (the first letter of a word), `second`, `inner` (the
third letter of a word or later: the word prefix mostly fixes it), `space`,
`other` (punctuation, digits, newline).

Then for each run after the first, prints the paired difference (run - first
run) of delta and dH, with the s.e. over the rows. The runs must score the
same rows (the same flags of ceval.sh).
"""
import json
import math
import os
import sys

LN2 = math.log(2)
CLASSES = ["start", "second", "inner", "space", "other"]
BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256)]


def classes(text):
    """The class of each character of text[1:] (text[0] ends the prompt)."""
    out = []
    run = 1 if text[0].isalpha() else 0
    for c in text[1:]:
        if c.isalpha():
            out.append(CLASSES[min(run, 2)])
            run += 1
        else:
            out.append("space" if c == " " else "other")
            run = 0
    return out


def load(d):
    rows = []
    with open(os.path.join(d, "contrast-profile.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "text": r["text"],
                    "class": classes(r["text"]),
                    "delta": [(a - b) / LN2 for a, b in zip(r["own"], r["other"])],
                    "dh": [(b - a) / LN2 for a, b in zip(r["h_own"], r["h_other"])],
                    "own": [-a / LN2 for a in r["own"]],
                }
            )
    return rows


def groups(row):
    """The (group name, character indices) of a row: the classes, then the bins."""
    n = len(row["delta"])
    out = [(c, [j for j, k in enumerate(row["class"]) if k == c]) for c in CLASSES]
    out += [(f"@{a}..{b}", list(range(a, min(b, n)))) for a, b in BINS if a < n]
    return out + [("all", list(range(n)))]


def table(name, rows):
    sums = {}
    for row in rows:
        for g, idx in groups(row):
            s = sums.setdefault(g, [0, 0.0, 0.0, 0.0])
            s[0] += len(idx)
            for i, key in enumerate(["delta", "dh", "own"]):
                s[1 + i] += sum(row[key][j] for j in idx)
    print(f"== {name}: {len(rows)} rows")
    print(f"  {'group':<10} {'chars':>7} {'delta':>8} {'dH':>8} {'own':>7}  share of delta")
    total = sums["all"][1]
    for g, (n, d, h, o) in sums.items():
        if n == 0:
            continue
        share = d / total if total else float("nan")
        print(f"  {g:<10} {n:>7} {d / n:>+8.4f} {h / n:>+8.4f} {o / n:>7.4f}  {share:>6.1%}")


def paired(name, base, rows, key):
    """Per group: the mean over the rows of (run - base) of the row mean of key."""
    diffs = {}
    # The runs score the same texts, so the groups of a row are those of base.
    for b, r in zip(base, rows):
        for g, idx in groups(b):
            if idx:
                d = sum(r[key][j] - b[key][j] for j in idx) / len(idx)
                diffs.setdefault(g, []).append(d)
    cells = []
    for g, ds in diffs.items():
        m = sum(ds) / len(ds)
        se = math.sqrt(sum((x - m) ** 2 for x in ds) / (len(ds) - 1) / len(ds)) if len(ds) > 1 else float("nan")
        cells.append(f"{g} {m:+.4f}±{se:.4f}")
    print(f"  {name} {key}: " + "  ".join(cells))


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    runs = [(os.path.basename(os.path.normpath(d)), load(d)) for d in dirs]
    for name, rows in runs:
        table(name, rows)
    base_name, base = runs[0]
    for name, rows in runs[1:]:
        if [r["text"] for r in rows] != [r["text"] for r in base]:
            sys.exit(f"{name} and {base_name} do not score the same rows")
        print(f"== paired difference to {base_name} (run - {base_name}), bits/char:")
        for key in ["delta", "dh", "own"]:
            paired(name, base, rows, key)


if __name__ == "__main__":
    main()
