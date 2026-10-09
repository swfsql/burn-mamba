# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the protocol, the tools).
`tmp.md` holds the results of the running session (they go to `notes.md`
at its end).

## State (2026-10-09 20:45)

- **The best H-Net**: R-N15 (`cfg/R-N15-lr4`: Real1D stages at r 104,
  stage d128 2+2, main d128 × 3 r64 Complex2D, N 1.5, mmlp 8, 999,425
  params, ~3.1 GB, `--ssd-path recalc:32`, lr 4e-3). dv@3230 0.8534 /
  0.8485 (two seeds). At the 1m-b milestones (the flat's LR plan) it
  leads the flat h16 at each: dv@3230 0.8534 / 0.8798, @5630 0.8038 /
  0.8137, @9600 0.7461 / 0.7597, @14400 **0.7207 / 0.7353** (the runs
  L-R-N15-lr4 → x4630 → x8000 → x12000). It also leads every recall row.
- **The noise**: a 600 + 200 screen of the R family ±0.009 (it cannot
  rank this family any more); a long run ±0.005.
- **Flat around R-N15** (long, ±0.005): N 1.25, α 0.1, stage r 128
  (R128c), lr 2.5e-3 / 8e-3 all tie or lose. The block alone (a flat
  model with the Real1D r 128 block, fr-d152) helps only early.
- **Use power-of-two state ranks**: another rank costs about as much VRAM
  as the next power of two, or more (the probe curve in tmp.md).
- **The best long run**: L-N15-lr4 (b × N 1.5, lr 4e-3, 2630 + 600): dv
  0.8915 (the flat h16-s2 0.8798). **lr 4e-3 for the long runs** (8e-3
  0.9070, 2.5e-3 0.8960). Less compression wins at 3230 too (lr 8e-3: N
  1.5 0.9070, N 2 0.9248, N 3 0.9482), and the flat model leads every
  recall row.
- **The protocol stays the first 1024 chars** (bs 10 × 1024). Whole
  stories (the length buckets of burn-stack, `gen.py buckets=…`) lose on
  this metric: −0.022 at 800, −0.020 at 3230. The flat model at the
  buckets (h16-B) equals the packed one (0.8847 / 0.8798).
- **The VRAM**: a Complex2D stage above r 64 does not fit (r 72 > 3750
  MiB, r 64 2761, Real1D r 104 2740). Lost: main virtual layers (also
  middle-heavy), stage r 64 Complex2D (ties), the bucketed protocol.

- **The whole-story metric** (`long.sh` / `longstat.py`, `long.txt`; 157
  valid stories longer than 2048 chars): the H-Net leads on chars 0..1024
  only. Past 1024 it gets worse, the flat model better: at 14400 the flat
  leads the whole long stories by 0.026 (tail 0.077). Cause: the H-Net
  protocol trains the first 1024 chars only. The buckets (whole stories)
  fix it in the b family (−0.043 on 1024..2048, −0.062 on the tail, 0..1024
  unchanged). 7.2% of the valid text is past char 1024.
- L-R-N15-wd05 (weight decay 0.05): dv 0.8492, a tie with R-N15.

## Running

- Local (`queue.txt`): L-W-m128r-L2 (started 20:16), then **L-R-N15-B**
  (R-N15 at the buckets, + long), then L-W-m64r (+ long).
- Kaggle batch 13 (v16, pushed 20:05): the N 2 extension to 8000 (two
  pieces) and L-W-s112-m. Batch 14 (a waiter pushes it after batch 13):
  L-Nm4-s96r64, L-Nm4-s96r128. Probe the pulled runs here with
  `long.sh -g <run>-d >> long.txt`.
- The heartbeat (a session cron, every 30 min) checks both runners.

## Next steps

1. **The protocol for the whole text**: if L-R-N15-B keeps R-N15's first
   1024 (dv ~0.851 ± 0.005) and wins the long stories, the buckets become
   the protocol (rank by dv and by `long` all). Then R-N15-B to the
   milestones, against the flat h16 (`long`: 1.0122 / 0.9425 / 0.8837 /
   0.8535).
2. **N 2 to the milestones** (user: a higher N should use the main stage
   more and remember entities better): L-R-x4630 → L-R-t6315 → L-R-x8000
   (Kaggle batch 13), then 12000 in two pieces as L-R-N15-x12000.
3. **The main network** (user: it should matter most). The main-heavy
   screens took depth from the stages (2+2 → 1+1): the loss can be the thin
   stages; L-A-s22-L2-N15 (main × 2) lost 0.027, so the main depth
   matters. The user: a larger main d and a larger main state rank. Long
   runs: L-W-m128r-L2 (main d168 × 2 r 128) and L-W-m64r (main d144 × 3 r
   64) here, L-W-s112-m (main d152 × 3 r 32) in batch 13; all with stage
   d112 2+2 Real1D r 128, N 2. The user: also the opposite trade (thin
   stages, a big main, a 4th main layer): Kaggle batch 14 (a waiter pushes
   it after batch 13): L-Nm4-s96r64 (stage d96 r 64, main d136 × 4 r 128,
   993k) and L-Nm4-s96r128 (stage d96 r 128, main d128 × 4 r 128, 996k).
   Also: a higher N, and the tail of the whole-story metric (where the
   main can matter).
4. Rank candidates with long runs (two seeds), not screens. A trunk longer
   than 1 h goes in pieces: `extend.sh … N1 0`, `--lr LR --span A-B`.
5. The VRAM confound (user): the flat h16-s2 3149 MB without gc, the H-Net
   ~3.1 GB with gc; a flat with gc could hold a larger block (fr-d152 is
   one: it held its edge only early). No gc-filled flat search was done.
