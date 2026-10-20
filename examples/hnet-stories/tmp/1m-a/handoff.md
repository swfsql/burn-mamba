# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (**its strict rules first**: they
come from the user and must survive every rewrite), then `notes.md` (the
findings).

## State (2026-10-20 04:30 UTC)

- **Nothing runs here**: the GPU is idle, `queue.txt` and
  `kaggle/kqueue.txt` are empty, and there is no waiter (the box was
  shut off).
- **A Kaggle batch ran past this session**: 20261020-032103 (kernel
  version 43), pushed 03:21, expected to end ~06:45–07:00 UTC. Its two
  lines:
  - `lrm2-s1`: `L-R-N125-r128s11V2-m5-mg4-lrm2-s1` → `-x4630` (+ `-d`) →
    `-t6315` → `-x8000` (+ `-d`), with `ctx.sh` on both leaves.
  - `sh32r64-s1`: the screen `L-R-N125-r128s11V2-m5-mg4-sh32r64-s1` (+
    `-d`) → `-x4630` (+ `-d`) → `-t6315`, with `ctx.sh` on both leaves.
- **First steps of the next session**:
  1. `kaggle/kpull.sh` (its waiter died with the box). Read only its
     filtered lines (`grep -E "^(done|fail)|^==|valid@|merged"`).
  2. Queue `./long.sh -g <run>-d >> long.txt` for each pulled `-d` run.
  3. Start `queue.sh`.
- The binaries: `bin/hn-cuda` = `6cf3c2b-dirty-cc8a67e3`, the flat's
  `ts-cuda` = `v0.67.0-4dd4cb8-dirty-99dc0dcf` (both run under CUDA
  13.0).
- Kaggle: ~22 h of this week's 30 h used, the batch above included. The
  reset: Saturday 2026-10-24 00:00 UTC.
- `runs/killed-*`: partial runs of a stopped queue line. Ignore them.
  `runs/killed-L-R-N125-r128s11V2-m5-mg4-sh32r64` is the s0 screen of
  `sh32r64`, stopped at ~step 1700 for the shutdown. Its run name is free
  again.

## The result (`notes.md` §1, §2)

- **The leader R-N125-r128s11V2-m5-mg4 ties the flat h16-V7u, with a
  trend**: −0.0028 at 14400 (two-seed means), ~1 s.e. Its gain is only
  past char 1024.
- **The new candidate: the leader × stage λ 2** (`lrm=2/1`). `lrm=auto`
  gives 1.12 at N 1.25. At 3230 its two-seed wv is 0.8558: **−0.0031
  against the leader, −0.0045 against the flat**.
  - It gains on chars 0..1024 (paired: s0 −0.0047, s1 −0.0021).
  - It loses past 1024 (+0.009 / +0.016, noisy).
  - λ 1.41 lost +0.0072, so part of the λ 2 gain can be noise.
- Closed this session: main d144 × 4 (+0.0070, and s0 +0.0097 at 5630),
  a C2D stage at 8 heads (+0.026), λ 1.41. `rf 1` in the main is shelved
  (one page over the band).

## Open questions

These are the questions as this session saw them, each with its
evidence. They are not a plan: re-decide the order from `notes.md`.

1. **Does λ 2 hold to 14400?** The leaves recover more and the trunks
   trail (+0.006 / +0.016 at 2630). After the pull:
   - s1 (Kaggle) is at 9600. Its next pieces: `extend.sh …-lrm2-s1-x8000
     …-lrm2-s1-t10000 2000 0`, then `extend.sh …-t10000 …-x12000 2000
     2400`.
   - s0 (here) is at 3230. Its chain left the queue before it started.
     The pieces: `extend.sh L-R-N125-r128s11V2-m5-mg4-lrm2
     L-R-N125-r128s11V2-m5-mg4-lrm2-x4630 2000 1000`, then `…-t6315 1685
     0`, `…-x8000 1685 1600`, `…-t10000 2000 0`, `…-x12000 2000 2400`
     (each leaf + `long.sh`).
   - λ is not monotone (1.12 / 1.41 / 2: 0 / +0.0072 / −0.0031). A point
     past 2 (2.83, 4) tells the trend from the noise.
2. **32 stage heads × r 64** (`sh32r64`). It keeps the leader's heads ×
   rank, with half the state and twice the decay timescales. 993,005
   params (`subc=1`), dose 3578 MB max.
   - s1 runs on Kaggle to t6315.
   - s0 needs its screen here (`gate.sh R-N125-r128s11V2-m5-mg4-sh32r64
     && screen.sh L-R-N125-r128s11V2-m5-mg4-sh32r64 … 2630 600`).
   - If it gains, the stage heads are the lever (§4: 8 heads lose 0.02
     against 16). Then λ 2 × `sh32r64` is the next composition.
3. **Counted, never run**:
   - Main 32 heads × r 32 + `mmlp 8` (993,805): the main twin of
     `sh32r64`. A main at r 32 lost before.
   - Stage expand 1 frees 99k params and VRAM, but no fill near both
     ceilings was found.
4. **The N view (user, long-term)**: no N = 1 test for now (user,
   2026-10-19).

The speed here: the leader ~1.4 s/step, `sh32r64` ~1.46. The cost: a
3230 screen ~80 min, + ~70 min to 5630, + ~2 h to 9600, + ~2.5 h to
14400. On Kaggle a
chain to 14400 takes ~6–7.5 h per line.
