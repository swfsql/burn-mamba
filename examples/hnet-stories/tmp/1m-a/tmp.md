# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-19 (session start 19:40 UTC)

- **Decision: the first new axis is the main width (d144 × 4 for d128 ×
  5), with `rf 1` as its second arm.** Reason: the leader ties the flat on
  chars 0..1024, and the flat's main is wider (d160) with `rf 1`. The
  param costs (flex counts, the leader 993,485):
  - `rf 1`: +10,320 (16 more angles per main layer). Over the cap at d128
    × 5, and no clean 4k trade exists (st2 −256, mg4 in total 1.8k; the
    untied copies and the B/C bias stay).
  - main d144 × 4: 988,961; × `rf 1`: 998,241. d160 × 3: 933k / 941k
    (far under); d152 × 4: 1,054,757 (over).
  - stage MLP (`smlp=8`, hidden 8): +6,656. A filler, not a design.
  - stage r 256 × main 4: 942k (16 heads), 922k (8 heads): far under.
  - The flat evidence for `rf 1` (`../../../tiny-stories/tmp/1m-b`):
    rf 0.5 lost ~0.009 at u2 (f29), and h16 × r128 × rf 0.5 fell behind
    in the leaves (+0.003 / +0.005 at 5630 / 9600).
- The configs: `R-N125-r128s11V2-m4d144-mg4` (988,961: ~1k under 990k,
  accepted: no param filler for 1k) and `R-N125-r128s11V2-m4d144-mg4-rf1`
  (998,241). Both are doses in the local queue.
- User (19:50): no N = 1 test for now. The local GPU can also run
  extensions (long runs) when it is fully idle.
- Dose `m4d144-mg4-rf1` (18 main heads, p 16): **3675 MB**, over the gate
  (3670). **Decision**: 16 main heads (`mp=18`), as the leader and the flat
  (h16 at p 20): `R-N125-r128s11V2-m4d144p18-mg4-rf1`, 993,729 params, a
  dose in the queue. The `rf 0.5` twin at p 18 has 984,449 (too far under
  990k): screen the combination first, and run the one-change arms only if
  it surprises (ORIENTATION, Working notes).
- Dose `m4d144-mg4` (`rf 0.5`, 18 main heads): **3482 MB**, ~1340
  ms/step (the leader's). In the band, 988,961 params.
- **Plan**: if the p 18 `rf 1` dose is in the band, Kaggle runs its two
  seeds (dose, gate, screen, x4630: one line each, equal lengths), and this
  GPU screens `m4d144-mg4` (s0) as the main-width arm. Reason: a two-seed
  mean is the ranking unit, and two lines on one host type avoid the host
  noise (0.0045 at 3230).
- Dose `m4d144p18-mg4-rf1` (16 main heads): **3675 MB** again. So the
  main heads do not set this peak, and `rf 1` costs one page (3482 →
  3675). **Decision**: pay the page with N 1.5 (it ties N 1.25, `notes.md`
  §4; −144…−290 MB at the r 128 shapes), at 18 main heads:
  `R-N15-r128s11V2-m4d144-mg4-rf1` (998,241) = `m4d144-mg4` × `rf 1` ×
  N 1.5. Its dose is in the queue.
- **Revised plan (host-matched seeds)**: the leader's s0 ran here and its
  s1 on a T4. So each new model runs s0 here and s1 on Kaggle. Kaggle:
  `m4d144-mg4-s1` + `R-N15-…-rf1-s1` (dose, gate, screen, x4630). Here:
  the s0 screens, then x4630.
- Dose `R-N15-…-m4d144-mg4-rf1`: **3675 MB** max (3579 in 4 of 5
  windows). **Decision: `rf 1` is shelved**: one page over the band at N
  1.25 and at N 1.5, and its evidence at u 1 is weak (the flat: the
  rotation ties at u 1; rf 0.5 lost at u 2). Its screen left the queue.
- 20:36: `L-R-N125-r128s11V2-m4d144-mg4` (s0) screens here.
- **Next candidate: a C2D stage at 8 stage heads** (p 32, `rf 0.5`):
  `R-N125-r128s11V2-m5-mg4-sp32c`, **993,517** params. Reason: 8 stage
  heads free 16.5k params (977,005) and ~630 MB, and a C2D stage costs
  16.5k (its angles are in the untied tail of every application). So the
  untested C2D stage is param-neutral. A mixed change (heads × kind):
  8 heads alone at 2+2 was never screened in the band. Its dose follows the
  screen here. The Kaggle push waits for it (the quota is the limit, not
  the wall time).
- **`L-R-N125-r128s11V2-m4d144-mg4-d` (s0, here): wv 0.8715** against
  the leader's s0 0.8625: **+0.0090** (sv at 2630 +0.013). `longstat.py`
  (paired, 256 stories): chars 0..1024 **+0.0096 ± 0.0016**, past 1024
  +0.0029 ± 0.0119. So the wider, shallower main loses on the first 1024
  chars (the target), and the tail ties. Under the 0.01 closing gap (and
  against the leader's weaker seed), so its s1 gets a screen only if a
  Kaggle line has room. A reading: the main wants depth (as main × 3 → ×
  5), and d128 is its floor (the stage is d128): the main is at a local
  optimum too.
- Dose `sp32c` (C2D stage, 8 stage heads): **3385 MB** (gate band; one
  page under the leader), ~1250 ms/step (the leader ~1340).
- 22:03: `L-R-N125-r128s11V2-m5-mg4-sp32c` (s0) screens here. Kaggle
  batch (screens only, ~1.5 h): `sp32c-s1` and `m4d144-mg4-s1`, no dose
  on Kaggle (both shapes dosed here; a seed twin has the same shape).
  Reason for screens only: decide the extensions at 3230.
- **`L-R-N125-r128s11V2-m5-mg4-sp32c-d` (s0, here): wv 0.8843, +0.0218**
  against the leader's s0 (sv at 2630 1.0214, +0.042). Paired: chars
  0..1024 +0.0225 ± 0.0023, past 1024 +0.0217 ± 0.0077. **Closed** (> 0.01).
  With the earlier 8-head losses (4+4 +0.011, 3+3 × V6 +0.007), the stage
  wants its 16 heads: 8 heads cost more than the C2D kind can return.
- 23:16: the local GPU is idle. **Decision**: extend `m4d144-mg4` (s0) to
  5630 (the user: extensions when idle; a shape change at 0 params and 0
  MB, +0.009 at 3230, so it gets the next milestone before a verdict).
- 32 stage heads (p 8): 1,026,541 params (+33k, the per-(head, rank) B/C
  bias and the per-head tails); at r 64: 984,813 (half the stage state).
  No param trade without a library change: not now.
- **Next: the stage LR multiplier λ** (0 params, 0 MB: the leader's
  shape and dose). `lrm=auto` gives λ = √N: 1.12 at N 1.25, 1.41 at N 2
  (where "λ auto +0.019" was measured, against λ 1). So the leader's stage
  lost its λ through the heuristic, unmeasured. Configs
  `R-N125-r128s11V2-m5-mg4-lrm141` (λ 1.41/1) and `-lrm2` (λ 2/1), with
  `-s1` twins. Here: both s0 screens after the extension (gated on the
  leader's dose). Kaggle: the s1 twins in the next batch.
- Kaggle batch 20261019-220331 pulled 23:29 (kwait queued both `long.sh`
  lines). At 3230, against the leader's s1 (0.8552):
  - `m4d144-mg4-s1` **0.8602** (+0.0050). **Two-seed mean 0.8659,
    +0.0070** against the leader's 0.8589.
  - `sp32c-s1` **0.8850** (+0.0298). Two-seed mean 0.8847, **+0.0258**:
    closed for good.
- 23:33: Kaggle batch with `lrm141-s1` and `lrm2-s1` (screens).
- **`L-R-N125-r128s11V2-m4d144-mg4-x4630-d` (s0): wv 0.8137, +0.0097**
  against the leader's s0 (0.8040); paired chars 0..1024 +0.0094 ±
  0.0016, past 1024 +0.0215 ± 0.0118. The gap holds (+0.0090 at 3230).
  **Main d144 × 4: closed** (two-seed +0.0070 at 3230; no closing trend).
- 00:27: `lrm141` (s0) screens here, then `lrm2`.
