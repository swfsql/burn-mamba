# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-18 07:10 UTC)

- **Nothing runs**: an idle GPU here, no Kaggle kernel, no waiter, no
  cron. `queue.txt` and `kaggle/kqueue.txt` are empty.
- The binaries: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`, the flat's
  `ts-cuda` = `v0.67.0-4dd4cb8-dirty-99dc0dcf`. Both run under the box's
  CUDA 13.0 unchanged.
- Kaggle: ~16 h of this week's 30 h used (the reset: Saturday 00:00 UTC).
- `runs/killed-*`: partial runs of a stopped queue line; ignore them.

## The result (`notes.md` §1)

- **R-N125-r128s11V2-m5-mg4 ties the flat h16-V7u, with a trend.** Two
  seeds each, wv: −0.0013 / −0.0013 / −0.0018 / −0.0028 (3230 / 5630 /
  9600 / 14400). At 14400: 0.7169 against 0.7197, ~1 s.e.
- The gain is only past char 1024 (−0.014 on the wv tail, −0.006 on the
  157 long stories past char 1024). On the first 1024 chars the H-Net ties
  the flat.
- Every variant of the leader that this session ran lost (`notes.md` §4):
  the leader is a local optimum of its neighbours.

## Open questions

These are the questions as this session saw them, each with its
evidence. They are not a plan: re-decide the order from `notes.md`.

1. **The N view (user, long-term)**: the leader's N 1.25 under-uses the
   H-Net; at N ≈ 1 it is a heterogeneous flat.
   - Rung a (the leader × N 2 × decoder 3 × st2), two-seed means
     against the leader's: +0.0174 / +0.0118 / +0.0068 / +0.0051 (3230 /
     5630 / 9600 / 14400); against the flat's +0.0023 at 14400.
   - So N 2 closes ~70% of its gap by 14400, but it still loses there. A
     longer budget could flip it; the step budget is a ceiling, though
     (ORIENTATION, Goal).
   - Rung b (N 2, the freed VRAM on main applications) needs a param
     trade: the leader has ~6.5k of headroom, one untied main application
     costs 8.6k.
   - N = 1 (the heterogeneous-flat test) needs a burn-stack change: the
     ratio loss divides by N − 1.
2. **A new axis.** The neighbours of the leader are spent: stage heads,
   stage placement, stage applications, main reuse, Multi-Gate on main
   reuse, N 2, 2 start rows (a tie to 14400: s0 +0.0021, s1 −0.0022 at
   5630; its s1 trunk `-t6315` waits, with no reason to go on).
   - Untested, each with a param trade: `rf 1` in the main, a C2D stage,
     a stage MLP.
   - The H-Net ties on the first 1024 chars and gains on the tail, so a
     gain on the first 1024 chars is the way to a clear lead.

The cost here: a 3230 screen ~80 min (the leader ~1.4 s/step), + ~70 min
to 5630, + ~2 h to 9600, + ~2.5 h to 14400. On Kaggle a chain to 14400
takes ~6–7.5 h per line.
