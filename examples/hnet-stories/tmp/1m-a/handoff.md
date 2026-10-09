# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the protocol, the tools).
`tmp.md` holds the results of the running session (they go to `notes.md`
at its end).

## State (2026-10-09 20:50): the search restarts

- **The protocol changed** (user, 20:40): whole stories, full backprop, the
  length buckets 1024,1536,2048,3072,4160 (`gen.py` default). **Every
  H-Net run before it is invalid**: it trained the first 1024 chars only,
  and past char 1024 it got worse while the flat model got better
  (`long-v1.txt`). Restart from scratch; the flat reference stays.
- **The metric: the whole valid `wv`** (`long.sh <run>-d`: the 256 valid
  stories in full). The flat h16 (packed whole stories): **wv 0.8868 /
  0.8204 / 0.7660 / 0.7412** at 3230 / 5630 / 9600 / 14400 (dv 0.8798 /
  0.8137 / 0.7597 / 0.7353).
- **The insights that stay** (old protocol, `notes.md` / `tmp.md`):
  - Real1D stages at a large state rank (r 104, recalc:32) were the lever;
    a Complex2D stage above r 64 does not fit the VRAM.
  - Power-of-two state ranks (another rank costs the VRAM of the next
    power of two, or more).
  - lr 4e-3 for the long runs (8e-3 and 2.5e-3 lose). N 1.5 beat N 2 by
    0.008 (R-N15 0.8534, R 0.8617 at 3230); N 3 lost more.
  - The trapezoid and the B/C bias stay on (None lost 0.042).
  - The main depth matters: main × 2 (stages kept) lost 0.027. Main-heavy
    shapes that thinned the stages to 1+1 lost.
  - Main virtual layers lost; weight decay 0.05 ties 0.1; α 0.1 ties.
  - Noise: a 600 + 200 screen ±0.009 (cannot rank the strong shapes), a
    long run (2630 + 600) ±0.005.
  - The length grouping costs: 7 slots cost the flat model 0.006 of wv;
    the 512/768/1024 grouping cost the b family 0.008. Hence one slot up
    to 1024.

## Running

- Local (`queue.txt`): **L-R-N15-B5** (R-N15 at the 5 slots, the new
  baseline; started 20:46; + ctx + long).
- Kaggle batch 13 (v16, pushed 20:05; old protocol, its results are void):
  the N 2 extension and L-W-s112-m. Batch 14 (the waiter pushes it after
  batch 13): **L-R-N15-B** (7 slots: the protocol contrast) and **L-R-B5**
  (N 2). Probe the pulled runs here: `long.sh -g <run>-d >> long.txt`.
- The heartbeat (a session cron, every 30 min) checks both runners.

## Next steps

1. **The baseline**: L-R-N15-B5 `wv`, against the flat 0.8868 at 3230.
   L-R-N15-B (7 slots) checks the slot choice; L-R-B5 the N.
2. **The main network** (user: it should matter most; a larger main d and
   state rank; or thin stages, a big main, a 4th main layer), now in the
   whole stories, where the main can matter: cfg/W-m64r-B5 (stage d112 2+2
   Real1D r 128, main d144 × 3 r 64, N 2) and cfg/Nm4-s96r128-B5 (stage
   d96 r 128, main d128 × 4 r 128, N 2), against L-R-B5. Measure their
   VRAM at the buckets (the 1024 slot holds 11 × 1024 tokens).
3. Then the milestones of the best shape (`extend.sh`, pieces < 1 h:
   `… N1 0`, `--lr LR --span A-B`), against the flat `wv`.
4. Rank by long runs (two seeds where it matters), by `wv`, with dv and
   the recall (`ctx.sh`) beside it.
5. The VRAM confound (user): the flat h16-s2 3149 MB without gc, the H-Net
   ~3.1 GB with gc; a flat with gc could hold a larger block (fr-d152: wv
   0.8792 at 3230, h16-s2 0.8868). No gc-filled flat search was done.
