# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-12 03:00)

- **Kaggle** (kernel version 33, batch `20261012-013449`): R-N125-r128s11V2-m5
  on the protocol, `ctx.sh` after each leaf.
  - gpu0, seed 0 (from the trunk `L-R-N125-r128s11V2-m5`): `-x4630(-d)`,
    `-t6315`, `-x8000(-d)`. Ends ~05:00.
  - gpu1, seed 1: the screen `L-R-N125-r128s11V2-m5-s1(-d)`, then
    `-s1-x4630(-d)`, `-s1-t6315`, `-s1-x8000(-d)`. Ends ~06:20.
  - A `kwait.sh` (started in the old session, log `/tmp/kpush-13.log`)
    pulls it and queues `long.sh`. If no `kwait.sh` runs
    (`ps -ef | grep '[k]wait.sh'`), wait for the kernel status
    `COMPLETE` (`kaggle/kg.sh kernels status <owner>/hnet-1m-a-runner`),
    then run `kaggle/kpull.sh`.
- **Here**: nothing runs. The stage + main reuse screen is done (wv
  0.8733, `tmp.md` 04:12). Four candidates are dosed in the band and
  ready to screen (step 5): `R-N125-r128s11V2-m5-mg4` and `-mmg4` (3482
  MB), `R-N125-r128s11V2-m5-st2` (3482), `R-N15-r64V7c` (3592).
- **No new runs** (user, 2026-10-12): wait for these, then start the next
  session's plan.
- **The binaries** are unchanged: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`,
  the flat's `ts-cuda` = `v0.67.0-4dd4cb8-dirty-99dc0dcf`.
- No cron: a heartbeat is session-only. Start one again if you need it.
- `tmp.md` is empty: its findings went into `notes.md`.

## The result (`notes.md` §1)

- **R-N125-r128s11V2-m5** (stage 1+1 real R1D r 128 × 2+2 untied
  applications, N 1.25, main d128 × 5) beats both flat seeds at 3230: wv
  0.8533 against the flat mean 0.8602, also on the long stories and the
  recall. One seed.
- **R-N15-r64V7** (main reuse) ties the flat at 9600 (0.7501 / 0.7490
  against 0.7487).
- **The flat h16-V7u**: 0.7220 at 14400. Cyclic is its best virtual
  schedule (two seeds each against the middle placements).
- The trunk schedule is in the config now (`notes.md` §2, §9):
  `extend.sh` needs no `--lr` for the protocol.

## Next steps

1. **Collect the Kaggle batch.** Check each pulled `-d` run for its
   `long.sh` (`kwait.sh` misses some headers). Check the schedules:
   `python3 lrprof.py <runs>` (all should print nothing: on the protocol).
2. **Rank R-N125-r128s11V2-m5 (two seeds) against h16-V7u**: 3230 (flat
   mean 0.8602), 5630 (0.8037), 9600 (0.7487). Use `longstat.py -w` and
   `ctxstat.py` against the flat runs (`-t` dirs in `long/`; the flat's
   recall is in `../../../tiny-stories/tmp/1m-cased/ctxg/`).
3. **Read the stage + main reuse screen** (`L-R-N15-r128s11V2-m5V6c-st2-d`)
   against R-N125-r128s11V2-m5 at 3230. It has three changes from it:
   main V 6 (Cyclic), N 1.5, 2 class tokens (`start=2`, the param trim).
4. **If R-N125-r128s11V2-m5 holds at 9600**: take it to 14400 (the flat:
   0.7220) in pieces < 1 h (`-x10000` +2000 with no leaf, then `-x12000`
   +2000 and the leaf of 2400). The configs hold the whole protocol, so
   no `--lr`.
5. **Designs to try** (dose first, `gate.sh` in the queue line):
   - **Multi-Gate residuals** (`tmp.md`): every H-Net stack is `Standard`,
     but the flat h16-V7u is Multi-Gate 4, and in the flat the standard
     residual lost 0.046. Untested on the H-Net in the new protocol. The
     configs: `R-N125-r128s11V2-m5-mg4` (every stack, 993,485) and `-mmg4`
     (the main only, 992,965). Both dose 3482 MB (= the leader, in the
     band): screen both.
   - Stage + main reuse at N 1.25 is over the band (3771 MB). A VRAM trim
     could give the page back (8 stage heads at p 32: the same state
     size, less VRAM, −8k params).
   - The 225-param trim of a main application: `start=2` (user: keep
     it). **But** the stage + main reuse screen
     (`L-R-N15-r128s11V2-m5V6c-st2-d`) lost 0.019 against the leader, with
     three changes in it (`tmp.md` 04:12). Screen `R-N125-r128s11V2-m5-st2`
     (the leader with `start=2` only, dose in `doses.txt`) first: it
     tells if the start rows cost it. Then choose the trim again.
   - **The H-Net main schedule** (user): `R-N15-r64V7c` (Cyclic main,
     0,1,2,0,1,2,0) against R-N15-r64V7 (0,1,1,1,1,1,2), all else equal.
     Its dose runs here last (~04:45). Rank it on the same milestones,
     two seeds if the gap is < 0.01. The result sets the main schedule
     of the stage + main reuse designs.
6. At the session end: drain `tmp.md` into `notes.md`.
