# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-10 04:30)

- **Nothing runs**: no `queue.sh`, an idle GPU, no Kaggle kernel, no
  waiters. `queue.txt` and `kaggle/kqueue.txt` are empty.
- **The binary**: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`. Its code is
  the committed code (burn-mamba `29b338d`, burn-stack `b0574bb`); only
  the tag is "dirty". The user cleaned `target/` for disk space, so the
  next `build.sh` compiles from scratch.
- **Every run** gets `--vram-cleanup-every 1` (`args.sh`): the dropped
  pages never wait past a window, so the process ≈ the `peak` of the
  `prof` lines + ~140 MiB. The dose target: a peak of ~3400–3550 MB
  (`notes.md` §8).
- **The results** (`notes.md` §3): the best shapes are still R-N15-B5 / R-B5
  (wv 0.8605–0.8723 at 3230, the flat 0.8868). N 1.5 and N 2 tie. The
  seed noise is 0.008. The M family (a stronger main, a thinner stage)
  loses by 0.03–0.10: the width of the char-level stage is the lever
  (`notes.md` §4).
- **The candidates of `notes.md` §5** (two and four stages, near the
  param ceiling) have their doses (`doses.txt`, peak max): **S2a2-B5 3532
  MB (on target)**, S2x-s112r256-L4 3029, S2x-s64e2-L4 2710, S4x-s64e2-L4
  2617. The last three are 400–900 MB under the VRAM target.

## Restart, in order

1. **Dose the candidates to the target.** S2a2-B5 is ready. The other
   three get a deliberate design change for +400 to +900 MB (strict rule
   14: never an auto-fill of the VRAM), keeping the params near the cap,
   then a new `dose.sh <cfg> -- --ssd-path recalc:32 --chunk-log 0`
   (~10 min). The width of the char-level stage is the lever with the
   evidence (`notes.md` §4), so prefer a change that keeps or widens the
   outer stage. Log each choice and its reason in `tmp.md`.
2. **Kaggle**: the long runs of the dosed candidates (`screen.sh <run>
   <cfg> 2630 600 -- --profile 100 --ssd-path recalc:32 && ./ctx.sh -g
   -n 2048 -r ctxg <run>-d >> ctxg.txt`), then `kaggle/kwait.sh <hours>
   /tmp/kpush-<n>.log` in the background. The order: S2x-s112r256-L4 and
   S2a2-B5 first (the widest outer stages), then the d64 pair
   S2x-s64e2-L4 / S4x-s64e2-L4.
3. **The horizon** (the user's hunch: the gain of a deep compression shows
   later, mainly on the long horizon): the S2x / S4x trunks go on to 4630
   (`extend.sh <run> <new> 2000 1000 …`), and so does the trunk of L-R-B5
   (the N 2 reference). Rank at 4630 by wv, the long-stories rows
   (`long.sh`, `longstat.py`) and the recall (`ctxstat.py`).

## Next steps

- **The stage width is the lever** (d64 +0.10, d96 +0.03–0.05 against
  d128). Open: a char-level stage **wider** than d128 (d144 / d160), paid
  by the main, at the VRAM ceiling. The old protocol had stage d144 tie
  d128 (a screen).
- A second seed of L-R-B5 (the N 2 reference), if a decision needs it.
- The M shapes were 500–950 MB under the ceiling: rerun one only if a
  reason comes up (the width result explains most of the loss).
- Deferred: the VRAM confound of the flat reference (the flat h16-s2 ran
  without gc at 3149 MB; a gc-filled flat model is untested).
- At the session end: drain `tmp.md` into `notes.md`.
