"""Helpers for the LR schedules of the training configs (burn-stack `Lr`).

A schedule is the JSON form of `Lr`: {"Linear": {...}}, {"Constant": {...}},
{"CosineAnnealing": {...}}, or {"Sequence": [{"from": S, "lr": {...}}, ...]}.
The schedule of a segment counts the steps from its `from`. `rate` gives the
same value as `Lr::get_lr`.

The scripts start a segment at the saved step S of a run. The training of
step k reads the rate at k, and a resumed run trains from step S + 1. So the
segment sets every step after S, and step S itself had the rate before it.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rate(lr, step):
    """The rate of the schedule `lr` at the global `step`."""
    ((kind, p),) = lr.items()
    if kind == "Constant":
        return p["lr"]
    if kind == "Linear":
        if step >= p["steps"]:
            return p["end"]
        return p["start"] + (p["end"] - p["start"]) * step / p["steps"]
    if kind == "CosineAnnealing":
        if step < p["warmup_steps"]:
            return p["max_lr"] * step / p["warmup_steps"]
        if step >= p["total_steps"]:
            return p["min_lr"]
        progress = (step - p["warmup_steps"]) / (p["total_steps"] - p["warmup_steps"])
        return p["min_lr"] + 0.5 * (p["max_lr"] - p["min_lr"]) * (1 + math.cos(math.pi * progress))
    if kind == "Sequence":
        segment = [s for s in p if s["from"] <= step][-1]
        return rate(segment["lr"], step - segment["from"])
    sys.exit(f"unknown LR kind {kind}")


def segments(training):
    """The segments of the LR sequence of a training config. Exits on the
    old form (seq.py rewrites it)."""
    lr = training["training"]["lr"]
    if "Sequence" not in lr:
        sys.exit(f"not an LR sequence (seq.py rewrites the old form): {lr}")
    return lr["Sequence"]


def cut(training, step, lr):
    """Keep the segments that start before `step`, then start `lr` there."""
    kept = [s for s in segments(training) if s["from"] < step]
    training["training"]["lr"] = {"Sequence": kept + [{"from": step, "lr": lr}]}


def run_path(run, name):
    return os.path.join(HERE, "runs", run, name)


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def saved_step(run):
    """The step in runs/<run>/progress.json."""
    return load(run_path(run, "progress.json"))["step"]
