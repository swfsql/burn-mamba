#!/usr/bin/env python3
"""Compare the LR schedule of runs with the protocol.

usage: lrprof.py [-a] <run-dir>…   (a bare name means runs/<name>)
       lrprof.py --fill <run-dir>…
       lrprof.py --guard <run-dir> <N1>
       lrprof.py --batch <kqueue.txt>

For each run: the saved step S, and the first step k < S at which the
saved schedule differs from the protocol (PROTOCOL below). A trunk that
trained on the protocol prints `protocol`. A decay leaf (its last segment
is a CosineAnnealing that starts at its trunk's step T) is checked up to T
only, and prints the rate at which its cosine starts. Step S itself is not
checked: a resumed run trains from S + 1, and a step-down at S starts
there. `-a`: print the runs on the protocol too (default: only the others).

The configs of `gen.py` hold the protocol (`stages`), as the configs of
the flat do. So `extend.sh` follows it with no `--lr`, and `--lr` is only
for a deliberate step-down off the protocol.

`--fill`: give a trunk the rest of the protocol (a trunk from a config
without it). It changes a run only when it is not a leaf, it trained on
the protocol up to S, and it has no segment from S on (a planned
step-down stays). Its segments before S stay, and the protocol segments
from S on follow them.

`--guard` (`extend.sh` runs it when it has no `--lr`): a run that trained
on the protocol up to S must stay on it for the N1 steps after S. Exit 1
when it does not. A run off the protocol (a deliberate branch) passes.

`--batch` (`kpush.sh` runs it): the same check for each `extend.sh` step
of a Kaggle batch, before the push. It reads the source runs and configs
here, and follows the runs that the batch makes (`screen.sh`, `extend.sh`
and its `--lr` / `--span`). Exit 1 when a step fails.
"""
import json
import os
import shlex
import sys

from lrseq import HERE, rate, stage

PROTOCOL = {"Sequence": [
    {"from": 0, "lr": {"Linear": {"start": 0.0, "end": 4e-3, "steps": 128}}},
    {"from": 4630, "lr": {"Constant": {"lr": 1e-3}}},
    {"from": 8000, "lr": {"CosineAnnealing": {
        "total_steps": 4000, "max_lr": 1e-3, "min_lr": 5e-4, "warmup_steps": 0}}},
]}


def first_diff(lr, lo, hi):
    """The first step k in [lo, hi) where `lr` differs from the protocol,
    with its rate and the protocol rate, or None."""
    if "Sequence" not in lr:
        return lo, None, None
    for k in range(lo, hi):
        a, b = rate(lr, k), rate(PROTOCOL, k)
        if abs(a - b) > 1e-9 * max(a, b, 1e-12):
            return k, a, b
    return None


def load_run(path):
    """The LR schedule and the saved step of a run directory."""
    with open(os.path.join(path, "training_config.json")) as f:
        lr = json.load(f)["training"]["lr"]
    with open(os.path.join(path, "progress.json")) as f:
        step = json.load(f)["step"]
    return lr, step


def run_path(arg):
    return arg if os.sep in arg else os.path.join(HERE, "runs", arg)


def describe(diff):
    k, a, b = diff
    if a is None:
        return f"not a sequence from step {k}"
    return f"{a:.3g} (protocol {b:.3g}) from step {k}"


def profile(path):
    lr, step = load_run(path)
    end, leaf = step, None
    seq = lr.get("Sequence")
    if seq and path.rstrip("/").endswith("-d") and "CosineAnnealing" in seq[-1]["lr"]:
        end = seq[-1]["from"]
        leaf = seq[-1]["lr"]["CosineAnnealing"]["max_lr"]
    return step, leaf, first_diff(lr, 1, end)


def fill(path):
    """The `--fill` of one run: returns what it did."""
    if path.rstrip("/").endswith("-d"):
        return "a leaf: kept"
    step, _, diff = profile(path)
    if diff is not None:
        return f"off the protocol ({describe(diff)}): kept"
    cfg = os.path.join(path, "training_config.json")
    with open(cfg) as f:
        tc = json.load(f)
    seq = tc["training"]["lr"]["Sequence"]
    if any(s["from"] >= step for s in seq):
        return "has its own segments from its step: kept"
    rest = [s for s in PROTOCOL["Sequence"] if s["from"] >= step]
    if not rest:
        return "past the protocol: kept"
    tc["training"]["lr"] = {"Sequence": seq + rest}
    with open(cfg, "w") as f:
        json.dump(tc, f, indent=2)
    return f"filled from {step}: + {[s['from'] for s in rest]}"


def guard(name, lr, step, n1):
    """None when the N1 steps after `step` keep a protocol run on the
    protocol, else the error message."""
    if first_diff(lr, 1, step) is not None:
        return None
    diff = first_diff(lr, step + 1, step + n1 + 1)
    if diff is None:
        return None
    return (f"{name}: trained on the protocol up to step {step}, but its "
            f"schedule leaves it: {describe(diff)}. Give the trunk the "
            f"protocol (`lrprof.py --fill <src>`), or pass `--lr` for a "
            f"deliberate step-down.")


def staged(lr, step, new_lr, span):
    """`lr` after `stage.py <run> <new_lr> [span]` at the saved `step`."""
    span = span or str(step)
    start = int(span.split("-")[0])
    kept = [s for s in lr["Sequence"] if s["from"] < start]
    _, segment = stage(kept, span, new_lr)
    return {"Sequence": kept + [{"from": start, "lr": segment["lr"]}]}


def batch(path):
    """The `--batch` check: returns the number of failed steps."""
    runs, failed = {}, 0
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
    for line in lines:
        for step_text in line.split("&&"):
            w = shlex.split(step_text.split(">>")[0])
            if not w:
                continue
            if w[0] == "./screen.sh" and len(w) >= 5:
                cfg = os.path.join(HERE, "cfg", w[2], "training_config.json")
                with open(cfg) as f:
                    runs[w[1]] = (json.load(f)["training"]["lr"], int(w[3]))
            elif w[0] == "./extend.sh" and len(w) >= 5:
                src, new, n1 = w[1], w[2], int(w[3])
                if src in runs:
                    lr, step = runs[src]
                else:
                    lr, step = load_run(run_path(src))
                if len(w) > 6 and w[5] == "--lr":
                    span = w[8] if len(w) > 8 and w[7] == "--span" else None
                    lr = staged(lr, step, float(w[6]), span)
                    print(f"ok    {new}: --lr {w[6]} (a deliberate step-down)")
                else:
                    err = guard(new, lr, step, n1)
                    if err:
                        failed += 1
                        print(f"FAIL  {err}")
                    else:
                        print(f"ok    {new}: steps {step + 1}..{step + n1}")
                runs[new] = (lr, step + n1)
    return failed


def main():
    args = sys.argv[1:]
    if args[:1] == ["--fill"]:
        for arg in args[1:]:
            print(f"{os.path.basename(run_path(arg).rstrip('/')):34} {fill(run_path(arg))}")
        return
    if args[:1] == ["--guard"]:
        path = run_path(args[1])
        lr, step = load_run(path)
        err = guard(os.path.basename(path.rstrip("/")), lr, step, int(args[2]))
        if err:
            sys.exit(f"lrprof.py --guard: {err}")
        return
    if args[:1] == ["--batch"]:
        sys.exit(1 if batch(args[1]) else 0)
    show_all = args[:1] == ["-a"]
    for arg in args[show_all:]:
        path = run_path(arg)
        name = os.path.basename(path.rstrip("/"))
        try:
            step, leaf, diff = profile(path)
        except (OSError, KeyError, json.JSONDecodeError) as e:
            print(f"{name:34} unreadable: {e}")
            continue
        if diff is None and not show_all:
            continue
        what = "protocol" if diff is None else describe(diff)
        top = "" if leaf is None else f"  leaf from {leaf:.3g}"
        print(f"{name:34} step {step:5}  {what}{top}")


if __name__ == "__main__":
    main()
