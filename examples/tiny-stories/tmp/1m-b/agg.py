#!/usr/bin/env python3
"""Aggregate the screens: one row per run, ranked by the decayed valid.

usage: agg.py [-b BUCKET] [run …]   (default: every runs/<x> with a runs/<x>-d)

Per run (read from runs/<x>-d, which holds both phases):
- params, ms/step and reserved MB (the last `prof` line of phase 1),
- train bits/char, mean of each BUCKET-step bucket of phase 1 (per-window
  train loss: noisy, but paired across runs with the same seed and data),
- slope: the change of train bits per 100 steps over the last 200 stable steps
  (a least-squares fit): how alive the run still is at the end of phase 1,
- sv / dv: the stable and the decayed valid bits/char (full validations),
- rank@k: the rank of the run by train bits in bucket k. A run whose rank
  improves from the early buckets to `dv` is a late bloomer.
"""
import json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2)


def rows(run):
    path = os.path.join(HERE, "runs", run, "metrics.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def info(run):
    """params, ms/step, reserved MB, from the stdout of phase 1."""
    params = ms = res = None
    log = os.path.join(HERE, "runs", run, "stdout.log")
    if os.path.exists(log):
        text = open(log, errors="replace").read()
        m = re.search(r"Number of parameters: (\d+)", text)
        params = int(m.group(1)) if m else None
        profs = re.findall(r"^prof n=\d+ .*total ([\d.]+) ms.*$", text, re.M)
        ms = float(profs[-1]) if profs else None
        r = re.findall(r"reserved ([\d.]+) MB", text)
        res = float(r[-1]) if r else None
    return params, ms, res


def summarize(run, bucket):
    d = rows(run + "-d")
    if not d:
        return None
    train = [r for r in d if r["event"] == "train"]
    valid = [r for r in d if r["event"] == "valid" and r.get("batches", 0) > 1]
    # The stable phase = the rows before the last "start" (the decay branch).
    # An extended run (extend.sh) has two stable segments before it.
    starts = [i for i, r in enumerate(d) if r["event"] == "start"]
    p1 = [r for r in d[: starts[-1]] if r["event"] == "train"] if len(starts) > 1 else train
    buckets = []
    for i in range(0, len(p1) - bucket + 1, bucket):
        chunk = p1[i : i + bucket]
        buckets.append(sum(r["loss"] for r in chunk) / len(chunk) / LN2)
    tail = p1[-200:]
    xs = [r["step"] for r in tail]
    ys = [r["loss"] / LN2 for r in tail]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    # The last full validation is the decayed one, the one before it the
    # stable one (an extended run also holds its source's stable validation).
    sv = valid[-2]["bits"] if len(valid) > 1 else float("nan")
    dv = valid[-1]["bits"] if len(valid) > 1 else float("nan")
    params, ms, res = info(run)
    return dict(run=run, params=params, ms=ms, res=res, buckets=buckets,
                slope=slope * 100, sv=sv, dv=dv)


def main():
    args, bucket = sys.argv[1:], 100
    if args[:1] == ["-b"]:
        bucket, args = int(args[1]), args[2:]
    runs = args or sorted(
        x for x in os.listdir(os.path.join(HERE, "runs"))
        if not x.endswith("-d") and os.path.isdir(os.path.join(HERE, "runs", x + "-d")))
    out = [s for s in (summarize(r, bucket) for r in runs) if s]
    out.sort(key=lambda s: s["dv"])
    nb = min(len(s["buckets"]) for s in out)
    # Rank by train bits per bucket (1 = best).
    for k in range(nb):
        order = sorted(out, key=lambda s: s["buckets"][k])
        for i, s in enumerate(order):
            s.setdefault("ranks", []).append(i + 1)
    head = (f"{'run':<18} {'params':>8} {'ms':>5} {'resMB':>5} "
            + " ".join(f"{'@' + str((k + 1) * bucket):>6}" for k in range(nb))
            + f" {'slope':>7} {'sv':>7} {'dv':>7}  ranks@buckets → dv")
    print(head)
    for i, s in enumerate(out):
        f = lambda v, fmt: format(v, fmt) if v is not None else "-"
        print(f"{s['run']:<18} {f(s['params'], '>8')} {f(s['ms'], '>5.0f')} "
              f"{f(s['res'], '>5.0f')} "
              + " ".join(f"{b:>6.3f}" for b in s["buckets"][:nb])
              + f" {s['slope']:>+7.3f} {s['sv']:>7.4f} {s['dv']:>7.4f}  "
              + " ".join(f"{r:>2}" for r in s["ranks"]) + f" → {i + 1}")


if __name__ == "__main__":
    main()
