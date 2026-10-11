# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-11 17:40)

- **Nothing runs**: an idle GPU, no Kaggle kernel, no waiters, no cron.
  `queue.txt` and `kaggle/kqueue.txt` are empty.
- **The binaries**: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3` (unchanged).
  The flat's `../../../tiny-stories/tmp/1m-cased/bin/ts-cuda` =
  `v0.67.0-4dd4cb8-dirty-99dc0dcf` (rebuilt 2026-10-11: it has
  `--vram-cleanup-every` and the `peak` measure).
- **The result** (`notes.md` §1): the flat reference h16 was under both
  ceilings. The flat at both ceilings is **h16-V7u**. It ties the best
  H-Nets at 3230 (0.8644) and at 5630 (each at its best schedule). It
  **leads by 0.0085 at 9600** (0.7487 against R-N15-B5's 0.7572).
- **The best H-Nets**: R-N15-B5 and R-N125-B5 (N 1.25 = N 1.5), and
  R-N15-r64V7 (stage r 64 + main V 7: a tie). Closed: deep compression
  (N 4/4), two stages at total N 2, a d144 stage paid by the main.
  Parked: S4 / d64 (no design near both ceilings).
- **The LR** (`notes.md` §2): the H-Net wants its step-down at about half
  of the budget. Compare two models on the same schedule.

## Next steps

1. **Can an H-Net beat h16-V7u?** That is now the question of the search.
   Rank every candidate against h16-V7u at the same milestone and
   schedule.
2. **Complete the flat reference** (here: the flat cannot run on Kaggle).
   - h16-V7u to 14400: `extend.sh h16-V7u-x8000 …` with the 8000–12000
     anneal, ~2 h in pieces < 1 h.
   - A second seed at 3230, if a decision needs it.
3. **H-Net ideas from the flat's lesson** (depth by reuse pays in the
   flat):
   - R-N15-r64V7 at 5630 and 9600: does main reuse keep up later?
   - Virtual layers in the stages (`gen.py` has `mV` for the main only:
     check if burn-stack takes them in a stage stack).
4. **Tools**: `kwait.sh` / `kpull.sh` can miss a `-d` run (`notes.md` §8).
   The flat could go to Kaggle with a payload and runner change, if its
   curve needs parallel runs.
5. At the session end: drain `tmp.md` into `notes.md`.
