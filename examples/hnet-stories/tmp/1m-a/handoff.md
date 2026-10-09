# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the rules, the protocol,
the tools), then `notes.md` (the findings).

## State (2026-10-09 22:15)

- **The search restarts** on the new protocol (whole stories, full
  backprop, 5 length slots). No H-Net has a valid result yet. The flat
  model to beat: **wv 0.8868 / 0.8204 / 0.7660 / 0.7412** at 3230 / 5630
  / 9600 / 14400.
- **The starting shape**: R-N15 (`cfg/R-N15-B5`: stage d128 2+2 Real1D r
  104, main d128 × 3 Complex2D r 64, N 1.5, mmlp 8, 999,425 params). Its
  first run, L-R-N15-B5, finished phase 1 (sv@2630 0.9809, 3386 MB, 1.28
  s/step). Its decay leaf is not done.
- **Everything is stopped** (user): no `queue.sh`, an idle GPU, no Kaggle
  kernel, no waiters. `queue.txt` and `kaggle/kqueue.txt` are empty. The
  binaries (`eb419b1f`) passed two smoke tests (on `eb28b0bb`, which
  differs only in a help line): a per-slot probe and the decay config of
  L-R-N15-B5.
- Old bucketed run dirs (L-B-*, B-*, probe-B-*, h16-B, their `ctxg/` and
  `long*/` copies) lack the `"probe"` config field, so they do not load.
  They are invalid runs. Add the field by hand if one must load again.

## Restart, in order

1. **The decay leaf of L-R-N15-B5.** `runs/L-R-N15-B5-d` holds a killed
   partial phase 2. Move it to `runs/fail2-L-R-N15-B5-d`. Then queue this
   line (the phase 2 of `screen.sh`, then the two probes):

   ```bash
   cp -r runs/L-R-N15-B5 runs/L-R-N15-B5-d && cp bin/hn-cuda.txt runs/L-R-N15-B5-d/build.txt && bin/hn-cuda --training --resume -a runs/L-R-N15-B5-d -c cfg/dec-L-R-N15-B5/training_config.json -b 600 --valid-every 0 --checkpoint-every 0 --valid-batches 1 -- --profile 100 --ssd-path recalc:32 --grad-checkpointing --max-vram 3750 >> runs/L-R-N15-B5-d/stdout.log 2>&1 && ./ctx.sh -g -n 2048 -r ctxg L-R-N15-B5-d >> ctxg.txt; ./long.sh -g L-R-N15-B5-d >> long.txt
   ```

2. **The VRAM probes** (queue lines after it), for Nm4-s96r128-B5,
   W-m64r-B5 and the 8 `M-*` of `notes.md` §8:
   `./probe.sh <cfg> -- --ssd-path recalc:32 --chunk-log 0 >> probes-B.txt`.
3. **Kaggle**: put the `screen.sh <run> <cfg> 2630 600 -- --profile 100
   --ssd-path recalc:32 && ./ctx.sh …` lines of L-R-B5 (the N 2
   reference) and L-Nm4-s96r128-B5 in `kaggle/kqueue.txt`. Then run
   `kaggle/kwait.sh 2 /tmp/kpush-19.log` in the background. The failed runs
   of the earlier pushes are `runs/fail-*`.

## Next steps

1. **The baseline**: the `wv` of L-R-N15-B5-d against the flat 0.8868.
   L-R-B5 gives the N 2 reference.
2. **The main network: explore a lot.** The user's hunch: with whole
   stories, a stronger main and a higher N should matter more. Try a 4th
   layer, a larger main d and state rank, even at the cost of the stage d
   and rank. Keep at it even if the first configs lose.
   - Long runs of the `M-*` shapes that fit the VRAM (a probe ≤ ~2990 MB),
     against L-R-B5 (N 2).
   - Then N 3.
   - A thinner stage frees VRAM: spend it on the main rank, MIMO or width.
3. The milestones of the best shape (`extend.sh`, pieces < 1 h), against
   the flat `wv`.
4. Deferred:
   - L-R-N15-B (`cfg/R-N15-B`, 7 slots): the slot choice at the long
     horizon.
   - The VRAM confound (user). The flat h16-s2 ran at 3149 MB without gc.
     With gc, a flat model could hold a larger block (fr-d152: wv 0.8792
     at 3230). Nobody ran a gc-filled flat search yet.
5. At the session end: drain `tmp.md` into `notes.md`.
