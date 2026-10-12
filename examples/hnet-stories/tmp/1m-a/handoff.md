# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-12 05:00)

- **Kaggle** (kernel version 33, batch `20261012-013449`):
  R-N125-r128s11V2-m5 on the protocol, `ctx.sh` after each leaf.
  - seed 0: `L-R-N125-r128s11V2-m5-x4630(-d)`, `-t6315`, `-x8000(-d)`.
  - seed 1: `L-R-N125-r128s11V2-m5-s1(-d)` (3230), `-s1-x4630(-d)`,
    `-s1-t6315`, `-s1-x8000(-d)`. Ends ~06:20.
  - A `kwait.sh` (log `/tmp/kpush-13.log`) pulls it. If none runs
    (`ps -ef | grep '[k]wait.sh'`): wait for `COMPLETE`
    (`kaggle/kg.sh kernels status <owner>/hnet-1m-a-runner`), then
    `kaggle/kpull.sh`.
- **Here**: nothing runs (user, 2026-10-12: no new runs until the Kaggle
  batch ends). Four candidates are dosed in the band (step 3).
- The binaries: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`, the flat's
  `ts-cuda` = `v0.67.0-4dd4cb8-dirty-99dc0dcf`. No cron (session-only).

## The result (`notes.md` §1)

- **R-N125-r128s11V2-m5** (stage 1+1 real R1D r 128 × 2+2 untied
  applications, N 1.25, main d128 × 5) beats both flat seeds at 3230 (wv
  0.8533, the flat mean 0.8602), the long stories and the recall too. One
  seed so far.
- **R-N15-r64V7** (main reuse) ties the flat at 9600.
- The flat h16-V7u: 0.7220 at 14400; Cyclic is its best schedule.

## Next steps

1. **Collect the Kaggle batch**: a `long.sh` for each pulled `-d` run
   (`kwait.sh` misses some), and `python3 lrprof.py <runs>` (nothing
   printed = on the protocol).
2. **Rank R-N125-r128s11V2-m5 (two seeds) against h16-V7u** at 3230
   (0.8602), 5630 (0.8037), 9600 (0.7487): `longstat.py -w` against the
   flat's `long/ts-h16-V7u-*`, `ctxstat.py` against
   `../../../tiny-stories/tmp/1m-cased/ctxg/`. If it holds at 9600: to
   14400 (`-x10000` +2000 no leaf, then `-x12000` +2000 and the leaf of
   2400; no `--lr`).
3. **Screen the four dosed candidates** (one change each against its
   base; the doses are in `doses.txt`):

   | config | the change | dose MB |
   |---|---|---|
   | `R-N125-r128s11V2-m5-mg4` | Multi-Gate 4 in every stack (the flat has it; the standard residual lost 0.046 in the flat) | 3482 |
   | `R-N125-r128s11V2-m5-mmg4` | Multi-Gate 4 in the main only | 3482 |
   | `R-N125-r128s11V2-m5-st2` | 2 start rows (the param trim of a main application, user: keep it if it holds) | 3482 |
   | `R-N15-r64V7c` | a Cyclic main against R-N15-r64V7's 0,1,1,1,1,1,2 (user: the H-Net main schedule) | 3592 |

   Rank each against its base at 3230; two seeds if the gap is < 0.01.
4. **Then stage + main reuse again**: the first try lost 0.019 with three
   changes in it (`notes.md` §4). Use the results of step 3 (the trim, the
   schedule). At N 1.25 one main application is over the band: 8 stage
   heads (p 32, the same state size) could give the page back.
5. At the session end: drain `tmp.md` into `notes.md`.
