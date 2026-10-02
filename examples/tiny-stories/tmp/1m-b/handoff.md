# Handoff: tiny-stories ≤1M-param search

Read this file, then `ORIENTATION.md` (goal, rules, protocol, tools).
`notes.md` has the results and the findings (§3 = the `v0.64.0` build). The
paths are relative to `examples/tiny-stories/tmp/1m-b/`.

## Goal (user)

The lowest valid bits/char **and** long-context coherence: the **far use**
of `--context-use`. Measure the far use of every screen and every milestone
(`ctx.sh`, a flex job between GPU runs), and rank by both. Use the most of
both ceilings (1M params, VRAM). Virtual layers (V > L) always take the
untied tensors of mnist-class (`gen.py untie=1`).

## State

Best loss: **0.7281**, h16 @14400 (`runs/p12k-h16-lr1-d`). Best far use:
**0.289**, h16 × g2 @14400 (`runs/p12k-h16g2-lr1-d2`).

The pool is a 2 × 2 on h16: {base, g2} × {L5, L5 × V6 untied}. Each cell:
decayed valid / far use (pairing i+1). Process memory on the current build.

| id | cfg | params | ms/step | process | @3230 | @5630 | @9600 | @14400 | latest stable run |
|---|---|---|---|---|---|---|---|---|---|
| **h16** | `s1k-u1-wd1-h16` | 969,160 | ~720 | 3134 MiB | 0.8579 / 0.224 | 0.8014 / 0.253 | 0.7493 / 0.269 | **0.7281** / 0.284 | `p12k-h16-lr1` (12000) |
| h16 × g2 | `s1k-u1-wd1-h16g2` | 982,747 | ~680 | 3284 MiB | 0.8649 / 0.241 | 0.8083 / 0.261 | 0.7552 / 0.285 | 0.7327 / **0.289** | `p12k-h16g2-lr1` (12000) |
| h16 × V6 | `n-h16-V6` | 982,380 | 871 | ~3580 MiB | **0.8533** / 0.225 | running | – | – | `n03-h16-V6` (2630) |
| h16 × g2 × V6 | `n-h16g2-V6` | 995,318 | 880 | ~3844 MiB | 0.8589 / 0.222 | – | – | – | `n06-h16g2V6` (2630) |

- h16 × g2 × V6 is over the preferred limit (3686 MiB), and 17 MiB under
  the `--max-vram 3750` cap. It is inside the search limit.
- h16 with seed 1 (`n04-h16-s1`): 0.8596 / 0.241 @3230. So the h16 loss
  replicates. The **far use has a seed spread of ~0.017 @3230**, so a
  far-use gap @3230 decides nothing. The g2 far-use lead was +0.016 @9600 and
  only +0.005 @14400 (one seed each).
- Parked: r128 (0.7590 / 0.267 @9600). Evicted this session: n01 quaternion
  h16 g16 r16 L3 × V9 (0.9372 / 0.167), n02 h20 (0.8722 / 0.218), n05 h16 ×
  g4 d136 (0.8966 / 0.201).

## Running at the stop

- `queue.sh` started `extend.sh n03-h16-V6 p5k-h16V6 2000 1000` at
  05:32 UTC (2026-10-01). It ends at ~06:35 with `runs/p5k-h16V6` (4630) and
  `runs/p5k-h16V6-d` (5630). Then the queue runner exits (`queue.txt` is
  empty).
- `queue.pending.txt` holds the parked lines: the context use of
  `p5k-h16V6-d`, n06 → 4630 + its context use, the probe + screen + context
  use of n07 (h16 + `trop=MaxPlus`). To resume them:
  `cp queue.pending.txt queue.txt && nohup ./queue.sh > /dev/null 2>&1 &`,
  then `qwait.sh` in the background (it prints once per finished line).
- If somebody stopped the n03 extension, `runs/p5k-h16V6` is partial. Give
  the next attempt a new name.

## Binary

- `bin/ts-cuda` = `v0.64.0-8dd4158-dirty-631e80dd`: burn-mamba `main` at
  `8dd4158` + the new example flag `--pack-align N` (uncommitted in the
  checkout: `cli.rs`, `main.rs`, `training.rs` and two test constructors of
  `examples/tiny-stories`). Without the flag, it runs as `v0.64.0-8dd4158`.
  n01 used `v0.64.0-8dd4158`. The runs from n02 on use the dirty build.
- Keep `--max-vram` on every training run (the scripts add 3750).

## Findings of this session (detail: `notes.md` §3)

- The memory grows with heads × `r` (B/C per head), in steps at the pool
  page sizes. Chunk 64 does not help. h32 fits only at r32 (3507 MB, 943,190
  params). h32 at r48/r64 and h16 × r128 do not fit. Probe every new shape.
- Virtual layers V6 (untied, full params) pay −0.005/−0.006 of loss, on h16
  and on h16 × g2. They do not raise the far use. Cost: +17% step, +440 MiB.
- Width is expensive to give up: g4 at d136 loses +0.039. A quaternion with
  many heads costs `nheads · 3 · blocks` in-proj channels per application.
  So n01 had to take r16, and it lost +0.08.

## Next steps

1. Run the context use of `p5k-h16V6-d`, then resume the parked queue (n06
   → 4630).
2. Bring n03 and n06 to the milestones of the other two, in matched steps:
   8000 (`extend.sh … 3370 1600 --max-lr 1e-3`), then 12000 (`… 4000 2400
   --max-lr 1e-3`), with the context use after each. Then compare the 2 × 2
   at 9600 and 14400 on loss and far use. Each run is ~1–1.5 h.
3. n07: h16 + `trop=MaxPlus` (`n-h16-trop`, 996,520 params, probe first), a
   breadth test of a long-memory register.
4. More ideas: the next LR point (5e-4 from 12000 tied at 14400 on h16);
   M = 16000/20000 for the pool; a clean quaternion test at B's shape (h4 +
   Q4D rf 0.5, ~984k); h32 r32 (it fits, but it halves the state); a longer
   `dt` init range for longer timescales; more seeds for a far-use decision.
