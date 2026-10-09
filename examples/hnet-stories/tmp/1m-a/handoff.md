# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the protocol, the tools).
`tmp.md` holds the results of the running session (they go to `notes.md`
at its end).

## State (2026-10-09 22:05): the search restarts

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

## Stopped (2026-10-09 22:05, user): restart in the new session

Everything is stopped: `queue.sh`, the GPU, the Kaggle kernel (v18
cancelled by the user), the waiters. `queue.txt` and `kaggle/kqueue.txt`
are empty. The config incident (tmp.md): every bucketed `cfg/` now has
`"probe": 0`; the binaries (`eb28b0bb`) and both smoke tests pass (a
regenerated config probes; the decay config of R-N15-B5 loads and trains).
To restart:

1. **L-R-N15-B5's decay leaf** (phase 1 is done: sv@2630 0.9809).
   `runs/L-R-N15-B5-d` holds a killed partial phase 2: `mv` it to
   `runs/fail2-L-R-N15-B5-d`, then as screen.sh phase 2: `cp -r
   runs/L-R-N15-B5 runs/L-R-N15-B5-d`, `cp bin/hn-cuda.txt
   runs/L-R-N15-B5-d/build.txt`, `bin/hn-cuda --training --resume -a
   runs/L-R-N15-B5-d -c cfg/dec-L-R-N15-B5/training_config.json -b 600
   --valid-every 0 --checkpoint-every 0 --valid-batches 1 -- --profile 100
   --ssd-path recalc:32 --grad-checkpointing --max-vram 3750 >>
   runs/L-R-N15-B5-d/stdout.log`, then `ctx.sh -g -n 2048 -r ctxg` and
   `long.sh -g` on it (a queue line).
2. **The VRAM probes** of the M family (`probes-B.txt`; R-N15-B5 is done:
   2903 MB at the 1024 slot, its run 3386): `./probe.sh <cfg> --
   --ssd-path recalc:32 --chunk-log 0 >> probes-B.txt` for Nm4-s96r128-B5
   and the 8 `M-*` (the list in tmp.md).
3. **Kaggle batch 14** again: `kaggle/kqueue.txt` = the screen.sh lines
   of L-R-B5 and L-Nm4-s96r128-B5 (+ ctx), then in the background
   `kaggle/kwait.sh 2 /tmp/kpush-19.log` (push, poll the status,
   `kpull.sh`, `long.sh` on each pulled `-d` run). The failed runs of v17
   are `runs/fail-*`.
- Old bucketed run dirs (L-B-*, B-*, probe-B-*, h16-B, the `ctxg/` and
  `long*/` copies) still lack `"probe"`: they do not load. They are
  invalid runs; add the field by hand if one must load again.

## Next steps

1. **The baseline**: L-R-N15-B5 `wv` (3386 MB reserved, ~1.3 s/step),
   against the flat 0.8868 at 3230. L-R-B5 gives the N 2 reference.
2. **The main network: explore a lot** (user hunch, 20:55: with whole
   stories, a higher N and a stronger main should matter more; a 4th
   layer, a larger main d and state rank, even at the cost of the stage d
   and rank; keep at it even if the first configs lose). The `M-*` family
   (stage 2+2 Real1D, thinner: d 64..96, r 32/64; main 4..5 layers, r 128
   or 256, Real1D or Complex2D; md fitted to 1M with `fit.py`; N 2, then
   N 3): VRAM probes here first (`probe.sh` visits every slot, user;
   `reserved` = the max over the slots; a run holds ~1.17 × that: dose a
   shape to a probe ≤ ~2990 MB for a run ≤ 3500), then long runs. Kaggle
   batch 14: L-R-B5 (the N 2 reference) and L-Nm4-s96r128-B5 (stage d96 r
   128, main d128 × 4 r 128).
3. Then the milestones of the best shape (`extend.sh`, pieces < 1 h:
   `… N1 0`, `--lr LR --span A-B`), against the flat `wv`.
4. Rank by long runs (two seeds where it matters), by `wv`, with dv and
   the recall (`ctx.sh`) beside it.
5. At the session end: the File Map line of burn-stack `dataset.rs`
   (`probe_order`, `BucketedStoriesDataset::new(…, probe)`) and of
   burn-mamba `examples/common/long_stories.rs` (examples/README.md has it).
6. The VRAM confound (user): the flat h16-s2 3149 MB without gc, the H-Net
   ~3.1 GB with gc; a flat with gc could hold a larger block (fr-d152: wv
   0.8792 at 3230, h16-s2 0.8868). No gc-filled flat search was done.
