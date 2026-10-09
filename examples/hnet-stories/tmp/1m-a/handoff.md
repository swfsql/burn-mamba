# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the protocol, the tools).
`tmp.md` holds the results of the running session (they go to `notes.md`
at its end).

## State (2026-10-09 10:15)

- **The best long run beats the flat model**: L-R-lr4 (`cfg/r1d-sr104-lr4`:
  Real1D stages at r 104, stage d128 2+2, main d128 × 3 r64 Complex2D, N
  2, mmlp 8, 999,425 params, 2916 MB, `--ssd-path recalc:32`, lr 4e-3,
  2630 + 600): dv 0.8617 (the flat h16-s2 0.8798), and every recall row.
- **The best screen**: R-N15 (the same × N 1.5) 1.0973 (F-r1d-sr104
  1.1076, b 1.1286, the flat 1.1314). The seed noise of a screen: ~±0.004.
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

- Local (`queue.txt`): F-r1d-sr64 (the rank or the kind?), R-trapN.
- Kaggle (batch 5): L-R-N15-lr4 (the winner × N 1.5, 2630 + 600 + ctx);
  the screens R128a (Real1D everywhere, stage r 128, Trapezoid None),
  R128b (main r 48 instead), R128a-N15.
- The heartbeat (a session cron, every 30 min) checks both runners.

## Next steps

1. Read L-R-N15-lr4 against L-R-lr4 (0.8617), and the r 128 screens.
2. Extend the best long run in < 1 h steps (does the lead grow?).
3. The VRAM of a non-power-of-two rank: find the padding (allocation or
   kernel) in the library. Complex2D costs ~1 GB per 64 ranks on top.
