# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-16

- **The environment**: the box CUDA is 13.0 now (driver 580.178.05; was
  12.2). The old binaries load it through `libnvrtc.so` (no rebuild): a
  probe of the leader ran (2809 MB reserved). The Kaggle CLI is 2.2.4 now
  (was 1.7.4.5); `kg.sh` works (`kernels status`, `datasets status`).
- **Decision: calibrate the doses first.** `dose.sh` of the leader
  (CUDA 12.2: 3482 MB, ~1340 ms/step). Reason: if CUDA 13 moves the peak
  or the speed, every old dose is stale for `gate.sh`.
- **Decision: the flat's seed 1 runs here, to 14400** (handoff question
  1): it decides if the lead holds against the flat's two-seed mean. The
  flat cannot run on Kaggle. Its `-x8000` piece (3370 1600) is split as
  the H-Net chain (`-t6315` 1685 0, `-x8000` 1685 1600): pieces < ~1 h.
- **Decision: Kaggle gets R-N15-r64V7-mg4 (+ `-s1`) to 5630** (handoff
  question 2), also as the quota test. Reason: its base ties the flat at
  9600 over 7 main applications, where Multi-Gate gained 0.046 in the
  flat. To 5630 only (~3.7 h): a partial result decides on 9600. Budget
  5.5 h.
- **Made R-N15-r128s11V4-m5-sp32-mg4** (handoff question 3): the N 1.5
  twin of `R-N125-…V4-m5-sp32-mg4` (3683 MB, 13 MB over the gate). 991,117
  params; the configs differ only in N and λ. Its dose goes before the
  flat chain (~13 min): a gated shape can go into the next Kaggle batch.
- **Kaggle is still at its quota** (2026-10-16 20:48 UTC: "Maximum weekly
  GPU quota of 30.00 hours reached"). The batch stays in `kqueue.txt`; the
  orphan payload dataset `hnet-pay-20261016-204732` stays (the CLI cannot
  delete it). A one-shot retry is set for 2026-10-17 00:23 (a weekly reset
  at Saturday 00:00 UTC is a guess, not a fact).
- **The calibration under CUDA 13**: the leader doses 3482 MB again (the
  same pages: 100:3385, then 3482), so the old doses hold. ms/step [1572
  1397 1406 1420 1406] against [1646 1474 1397 1339 1346]: the tail is
  ~4% slower, near the noise of the host load. Watch the speed of the
  flat chain against its seed 0.
- **R-N15-r128s11V4-m5-sp32-mg4 doses 3539 MB** (in the band; ms/step
  ~1500, the leader ~1406): N 1.5 saves 144 MB on its N 1.25 twin (3683).
  Made `-s1`. **Decision: screen both seeds here after the flat chain**
  (the gate reads the dose of the shape, the seed does not change it).
  Reason: it is the leader × 8 stage heads × 4+4 applications × N 1.5,
  the deeper stage that §4 says must come from applications. Kaggle keeps
  R-N15-r64V7-mg4 (a different question: Multi-Gate on main reuse).
- **The flat's seed 1 at 9600: wv 0.7430** (seed 0: 0.7487; mean
  0.74585). The leader's mean 0.74405: **Δ means −0.0018** (was −0.0047
  against seed 0 only). So at 9600 the lead is the same as at 3230 / 5630
  (−0.0013): a tie within the seed noise. The 157 long stories (paired
  against the flat's s1, `longstat.py`, means of two seeds): all chars
  −0.0034, 1024..2048 −0.0056, past 2048 −0.0062. So the H-Net keeps a
  small lead on the long tails only. ms/step of the flat under CUDA 13:
  1049 (CUDA 12.2: 1063).
- **The flat's seed 1 at 14400: wv 0.7173** (seed 0: 0.7220; mean
  0.71965). The leader's mean 0.7169: **Δ means −0.0028**. The lead over
  the milestones: −0.0013 / −0.0013 / −0.0018 / −0.0028 (3230 / 5630 /
  9600 / 14400). It grows, but it is ~1 s.e. of a two-seed difference
  (the flat's seeds differ by 0.0047): a tie with a trend. By the
  position (means of two seeds): wv chars 0..1024 −0.0018, past 1024
  −0.0139 (32 stories); the 157 long stories (paired against the flat's
  s1): 0..1024 +0.0008, 1024..2048 −0.0072, past 2048 −0.0061, all
  −0.0041. So the H-Net gains only past char 1024. On the first 1024
  chars it ties the flat.
- **Kaggle is back**: refused 2026-10-16 20:48 UTC, accepted 2026-10-17
  00:23 UTC (batch `20261017-002316`, kernel version 37, RUNNING). So the
  weekly quota likely resets on Saturday 00:00 UTC. The batch:
  R-N15-r64V7-mg4 + `-s1` to 5630 (budget 5.5 h, ~3.7 h expected).
  `kwait.sh` waits in the background (log `/tmp/kwait-1017a.log`).
