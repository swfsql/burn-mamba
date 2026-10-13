# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-13 08:30)

- **Nothing runs**: an idle GPU here, no Kaggle kernel, no waiter, no
  cron. `queue.txt` and `kaggle/kqueue.txt` are empty.
- **Kaggle is at its weekly GPU quota** (refused 2026-10-13 05:40). The
  reset time is unknown (likely 5–7 days). Plan the work for this card. A
  push is the only test of the quota.
- The binaries: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`, the flat's
  `ts-cuda` = `v0.67.0-4dd4cb8-dirty-99dc0dcf`.

## The result (`notes.md` §1)

- **R-N125-r128s11V2-m5-mg4 leads the flat h16-V7u** (wv, two-seed
  means). The lead: −0.0013 at 3230 and at 5630 (two flat seeds), −0.0047
  / −0.0051 at 9600 / 14400 (only the flat's seed 0). At 14400: 0.7174 / 0.7164
  against 0.7220. On the long stories past char 1024: −0.022 / −0.021.
- Multi-Gate 4 gains ~0.005 on its base at every milestone. It is the
  default now (ORIENTATION, Shapes and budgets).

## Open questions

These are the questions as this session saw them, each with its
evidence. They are not a plan: the plans of earlier handoffs steered the
sessions after them (user). Re-decide the order from `notes.md`.

1. **Does the lead hold against the flat's seed 1?** Seed 1 was better
   than seed 0 by 0.0085 / 0.0031 at 3230 / 5630. So the lead of ~0.005
   at 9600 / 14400 can shrink or go. Its trunk `h16-V7u-s1-x4630` is
   ready (the chain: ORIENTATION, Protocol). Until then, "the H-Net beats
   the flat" holds against one flat seed only.
2. **Multi-Gate on main reuse**: `R-N15-r64V7-mg4` (+ `-s1`), 997,533
   params, dosed 3592 MB, ready to screen. Its base R-N15-r64V7 ties the
   flat at 9600. Its main routes over 7 applications, as the flat does,
   where Multi-Gate gained 0.046.
3. **The 8 stage heads** (`p=32`) free ~630 MB at the same state size,
   but a stage application still costs ~480–590 MB.
   - Screened: 8 heads × 3+3 × main V 6: +0.007 / +0.006 at 3230 / 5630
     against the leader (one seed).
   - Dosed: `R-N125-r128s11V4-m5-sp32-mg4` (4+4, no main reuse,
     991,117) 3683 MB, 13 MB over the gate. Its N 1.5 twin is not made
     (expected ~3440).
   - The 8 heads alone were never measured: alone they sit under the
     band.
4. **Main reuse on the r 128 shape** lost ~0.006 twice (16 and 8 stage
   heads). It paid on the r 64 shape (R-N15-r64V7). Why it does not
   compose with stage reuse is open.
5. **A deeper stage** comes from applications, not from real layers: a
   real r 128 stage layer pair costs ~250k params (`notes.md` §4).

The cost here (the leader's speed, ~1.34 s/step): a 3230 screen ~80 min,
+ ~80 min to 5630, + ~2 h to 9600, + ~2 h to 14400. The flat: ~1.1
s/step, so its seed 1 needs ~3.5 h from 5630 to 14400.
