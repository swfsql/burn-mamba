# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the protocol, the tools).
`tmp.md` holds the results of the running session (they go to `notes.md`
at its end).

## State (2026-10-09 18:55)

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

## Running

- Local (`queue.txt`): L-A-s33-N15-lr4 (stage 3+3 at d112, main d112 × 3,
  the best screen 1.0898).
- Kaggle (batch 12): L-A-s33-N15-lr4-s1, L-A-s22-L2-N15-lr4 (stage 2+2
  at d136, main × 2).
- The heartbeat (a session cron, every 30 min) checks both runners.

## Next steps

1. Read the long allocation tests against R-N15 (0.851 at 3230).
2. Rank candidates with long runs (two seeds), not screens.
3. A trunk longer than 1 h goes in pieces: `extend.sh … N1 0` (no decay),
   `--lr LR --span A-B` for an anneal.
