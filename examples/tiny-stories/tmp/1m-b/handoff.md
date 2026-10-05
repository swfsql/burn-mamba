# Handoff: tiny-stories ≤1M-param search

Read this file, then `ORIENTATION.md` (goal, rules, protocol, tools, and
the map of the files). `notes.md` has the results and the findings (§3 = the
`v0.64.0` pool build, §4 = the far-use diagnostics and the 2048-story
context use, §5 = the contrast). `far-use.md` is the plan for auxiliary
objectives that target the far use. The paths are relative to
`examples/tiny-stories/tmp/1m-b/`.

## Goal (user)

The lowest valid bits/char **and** long-context coherence: the **far use**
of `--context-use`. Measure the context use of every screen and every
milestone (`ctx.sh -g -n 2048`), and rank by both. Judge the far use by the
own far bits paired with a control. Use the most of both ceilings (1M
params, VRAM). Virtual layers (V > L) always take the untied tensors of
mnist-class (`gen.py untie=1`).

## Guidelines (user)

- **Plan before long runs.** First run the checks of a few minutes that
  show whether a run of 40 minutes or more can be informative: evals on
  the existing checkpoints and snapshots, short probes, tiny smoke runs.
  Do not spend a long run that can be a null.
- **Models before seeds** (`ORIENTATION.md`, Working rules).

## State

Best loss: **0.7281**, h16 @14400 (`runs/p12k-h16-lr1-d`). Best far use:
**0.289**, h16 × g2 @14400 (`runs/p12k-h16g2-lr1-d2`). But its gain ties h16,
and its own bits are worse: the lead is in the penalty (`notes.md` §4).

The pool is a 2 × 2 on h16: {base, g2} × {L5, L5 × V6 untied}. Each cell:
decayed valid / far use (pairing i+1, 244 stories). Process memory on the
current build.

| id | cfg | params | ms/step | process | @3230 | @5630 | @9600 | @14400 | latest stable run |
|---|---|---|---|---|---|---|---|---|---|
| **h16** | `s1k-u1-wd1-h16` | 969,160 | ~720 | 3134 MiB | 0.8579 / 0.224 | 0.8014 / 0.253 | 0.7493 / 0.269 | **0.7281** / 0.284 | `p12k-h16-lr1` (12000) |
| h16 × g2 | `s1k-u1-wd1-h16g2` | 982,747 | ~680 | 3284 MiB | 0.8649 / 0.241 | 0.8083 / 0.261 | 0.7552 / 0.285 | 0.7327 / **0.289** | `p12k-h16g2-lr1` (12000) |
| h16 × V6 | `n-h16-V6` | 982,380 | 871 | ~3580 MiB | **0.8533** / 0.225 | – | – | – | `n03-h16-V6` (2630) |
| h16 × g2 × V6 | `n-h16g2-V6` | 995,318 | 880 | ~3844 MiB | 0.8589 / 0.222 | – | – | – | `n06-h16g2V6` (2630) |

- The pool runs used 2 workers: their dv has a data-order noise of ~0.01
  (`ORIENTATION.md`, Run noise). Parked: r128 (0.7590 / 0.267 @9600).

**The contrast** (`far-use.md` steps 2–3, `notes.md` §5): built, and tested
on h16 @3230 against the 1-worker control **b01-h16-w1** (dv 0.8680,
snapshots every 250 in `snap/b01-h16-w1/`).

- Random negatives do not push (the pair separates within a few words).
- Shuffled negatives push (0.4–0.8 of chance).
- **c03-h16-shuf**: shuffled (first and last sentence in place), sentence
  cut, J 0, λ 1, start 500, ramp 500. Order Δ ×5 (0.009 → 0.044
  bits/char). dv 0.8758 (+0.0078), own far +0.0080 ± 0.0010, none +0.018,
  use unchanged. The cost is real only if the 1-worker noise is small. The
  no-prompt loss suggests a shortcut ("odd context ⇒ predict worse").

## Queue

- Nothing runs, and `queue.txt` is empty.
- `queue.pending.txt` holds the parked pool work: n03 and n06 → 4630, each
  with its context use, then the probe + screen + context use of n07 (h16 +
  `trop=MaxPlus`). Its `ctx.sh` lines use the old 244-story flex form:
  change them to `ctx.sh -g -n 2048 -r ctxg … >> ctxg.txt` before a resume.
  To resume: `cp queue.pending.txt queue.txt && nohup ./queue.sh >
  /dev/null 2>&1 &`, then `qwait.sh` in the background.

## Binary

- `bin/ts-cuda` = `v0.64.0-4d90008-dirty-14058a87`: `4d90008` + the push
  readout, `--contrast-eval`, `--contrast-negative`. c03 used this build,
  and b01 used `…-dirty-8b4ea840`. Without the contrast, their training
  path is the same. After the commit of this session, `./build.sh` gives a
  clean tag of the same code.
- The pool runs used `v0.64.0-8dd4158-dirty-631e80dd` (`--pack-align N`).
- Keep `--max-vram` on every training run (the scripts add 3750).

## Next steps

1. **The 1-worker noise** (~3 min): `ctx.sh -g -n 2048 -r ctxg` on
   `snap/b01-h16-w1/500` and `snap/c03-h16-shuf/500`. They differ only by
   the float noise (the contrast starts at 500). About 0.001 apart: c03's
   cost is real. Much more: a b01 rerun (`b02`, ~40 min) for the noise of
   a final dv.
2. **Order or shortcut?** Compare the per-position `Δ` of c03 and b01 (a
   small change to `--contrast-eval`). Real order use sits on the chars
   that depend on the events. A "predict worse after odd prompts" penalty
   is flat over `C`.
3. Then decide on the shuffled contrast. The options: a lower λ (0.3), a
   later start (~2000: there the gain stalls and the penalty rises), J 64,
   or a natural-text negative. Every synthetic negative carries the
   shortcut risk (`far-use.md` step 3).
4. The parked pool work (Queue). More ideas:
   - the next LR point (5e-4 from 12000 at 14400 on h16),
   - M = 16000/20000, a clean quaternion test at B's shape (h4 + Q4D rf
     0.5), h32 r32,
   - a longer `dt` init range, the distance curve (`far-use.md` step 1.3).
