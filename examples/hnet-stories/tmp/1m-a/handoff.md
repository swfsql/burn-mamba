# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-12 06:30)

- **Nothing runs**: an idle GPU here, no Kaggle kernel, no waiter, no
  cron. `queue.txt` and `kaggle/kqueue.txt` are empty.
- Five candidates are dosed in the band and ready to screen (steps 1–2).
- The binaries: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`, the flat's
  `ts-cuda` = `v0.67.0-4dd4cb8-dirty-99dc0dcf`.

## The result (`notes.md` §1)

- **Two H-Nets tie the flat h16-V7u at 9600** (two seeds each, wv 0.7487):
  R-N125-r128s11V2-m5 (stage reuse at r 128) 0.7491, R-N15-r64V7 (main
  reuse) 0.7496. R-N15-B5 trailed by 0.0085. No H-Net beats the flat yet.
- R-N125-r128s11V2-m5 has a large seed spread (0.022 at 3230): its seed 0
  beat the flat everywhere, its seed 1 lost everywhere.
- The flat: 0.7220 at 14400; Cyclic is its best virtual schedule.

## Next steps

1. **Screen the four dosed one-change candidates** (`doses.txt`), each
   against its base at 3230, two seeds where the gap is < 0.01 (the seed
   spread of the r 128 leader is large: compare seed pairs, or means):

   | config | the change | dose MB |
   |---|---|---|
   | `R-N125-r128s11V2-m5-mg4` | Multi-Gate 4 in every stack (the flat has it; the standard residual lost 0.046 in the flat) | 3482 |
   | `R-N125-r128s11V2-m5-mmg4` | Multi-Gate 4 in the main only | 3482 |
   | `R-N125-r128s11V2-m5-st2` | 2 start rows (the param trim of a main application) | 3482 |
   | `R-N15-r64V7c` | a Cyclic main against 0,1,1,1,1,1,2 (user: the H-Net main schedule) | 3592 |

2. **The start rows** (user): the stage + main reuse screen lost 0.019,
   and two of its three changes were neutral alone (`notes.md` §4).
   - If `-st2` loses: the start rows matter. Screen
     `R-N125-r128s11V2-m5-st8` (8 rows, 992,177, 3482 MB), then consider
     start rows per stack (the main): a burn-stack change, as an H-Net
     stack takes no class markers today.
   - If `-st2` ties: the measured changes did not compose. Look at N 1.5 ×
     main reuse on the r 128 shape.
3. **Then stage + main reuse again**, with the results of steps 1–2 (the
   trim, the schedule, Multi-Gate). At N 1.25 one main application is over
   the band: 8 stage heads (p 32, the same state size) could give the page
   back.
4. **14400**: take the best tie to 14400 (the flat 0.7220) when a design
   is chosen: `-x10000` (+2000, no leaf), then `-x12000` (+2000, the leaf
   of 2400). The configs hold the whole protocol: no `--lr`.
5. At the session end: drain `tmp.md` into `notes.md`.
