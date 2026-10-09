# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-09 22:40 restart

- Here: the decay leaf of L-R-N15-B5 (the failed partial leaf is
  `runs/fail2-L-R-N15-B5-d`), then the probes. Their order: the three
  extremes of the M family first (M-s64r64-L4, M-s96r64-L5,
  M-s96r64-L4r256), then Nm4-s96r128-B5, the other M shapes, W-m64r-B5.
  The probe of R-N15-B5 is done (2903 MB), so it does not run again.
- Kaggle batch 19: L-R-B5 (the N 2 reference) and **L-R-N15-B5-s1** (the
  baseline at `seed=1`: the seed noise of the new protocol, user
  suggestion 2). L-Nm4-s96r128-B5 moves out of this batch: it has no
  probe yet, and it belongs with the M family (the next batch, on the new
  dv binary).
- L-R-N15-B5-d at step 100: 1.32 s/step, reserved 3384.8 MB (the phase 1
  3386).

## The validation scores whole stories (user suggestion 1)

- burn-stack `lm.rs` `dataloaders_for`: a config with `bucket` validates
  on the same slots and `chars`: the same 256 stories, each whole, in one
  window (`BucketedStoriesDataset` of the valid split, the order of
  `training.seed`). No new config field: the `bucket` field selects it.
  An unbucketed config validates as before (windows of `seq_len`).
- So dv of a bucketed run is `wv`. A run of an older binary still has the
  1024-char dv: read the binary tag in `build.txt`.
- `gen.py`: only the comment changes (`seq`/`runlen` do nothing under the
  buckets). The configs do not change.
- **Check** (binary `4e42bba3`, `runs/vchk-L-R-N15-B5-d`: `-b 0` on a copy
  of the leaf): 28 batches, 207,187 chars, bits 0.86880287. `long.sh`:
  wv 0.86880286. Equal. So from this binary on, dv = wv, and `long.sh` is
  needed only for the long stories and the per-position profile.

## The baseline (new protocol)

- **L-R-N15-B5-d**: **wv 0.8688** (chars 0..1024 0.8612, past 1024
  0.9529), the old dv (first 1024) 0.861. The flat h16 at 3230: 0.8868, so
  the H-Net leads by **0.018**.
- Against old-protocol R-N15 at 3230 (wv 0.8659, dv 0.8534): chars
  0..1024 +0.008, past 1024 about −0.07 (computed from the 7.2% share),
  all +0.003. So the 5 slots cost the story heads about as much as the
  grouping did before (B-c1024 +0.008).
- The long stories: 0..1024 0.9938, 1024..2048 1.0105, 2048.. 0.9779,
  all 0.9981. The recall (`ctxg.txt`): gain 0.1426 ± 0.0017, penalty
  0.0908.
- The leaf: 600 steps in 13 min here (1.32 s/step).

## Kaggle batch 19 (old binary: dv cut at 1024, wv by long.sh)

| run | wv | 0..1024 | past 1024 | long stories all | dv (1024) | sv@2630 |
|---|---|---|---|---|---|---|
| L-R-N15-B5-d (seed 0) | 0.8688 | 0.8612 | 0.9529 | 0.9981 | 0.861 | 0.9809 |
| L-R-N15-B5-s1-d | **0.8605** | 0.8536 | 0.9377 | 0.9889 | 0.8536 | 0.9748 |
| L-R-B5-d (N 2) | 0.8723 | 0.8650 | 0.9525 | 1.0051 | 0.8650 | 0.9870 |

- **The seed noise of the new protocol: 0.0083** between the two seeds
  of R-N15-B5 (paired −0.0084 ± 0.0019; the ± is the valid-set noise
  only). The old protocol had ±0.005 per long run. So one run per shape
  can rank only a gap of ≳ 0.01. Below that, a decision needs a second
  seed. The seed spreads every row the same way (0..1024 −0.008, past 1024
  −0.011, the long stories −0.009), and the recall too (re-mention letters
  own −0.037 ± 0.005, own −0.006).
- **N 2 against N 1.5**: L-R-B5 +0.0035 against the seed mean of R-N15-B5
  (0.8647), inside the seed noise. The recall of N 2 (re-mention gain
  +0.064 against seed 0) is also within the seed spread (seed 1: +0.041).
  So N 1.5 and N 2 tie, as at 4e-3 in the old protocol.
- The flat h16 at 3230 is 0.8868: the H-Net leads by 0.015–0.026.
- Speed on a T4: phase 1 51–54 min (R shapes), the leaf ~12.5 min.

## The M family: probes (binary 4e42bba3)

| cfg | probe MB | per slot | ms/step (4160 slot) |
|---|---|---|---|
| R-N15-B5 (ref) | 2903 | 2903/2614/2421/2710/2421 | 1876 |
| M-s64r64-L4 | 1538 | 1538/1255/1142/1199/1037 | 868 |
| M-s96r64-L5 | 1813 | 1813/1497/1317/1452/1317 | 1077 |
| M-s96r64-L4r256 | 1947 | 1947/1771/1595/1595/1595 | 2628 |

- **The M shapes leave ~1000–1450 MB of the probe budget (2990) unused.**
  The ceiling rule (every shape uses the most of both ceilings) says to
  spend it. Decision: fill each extreme with the cheapest known lever, a
  stage Real1D **r 128** (§4: the rank pays; r 128 costs less than r 104),
  and refit the params (`fit.py`): M-s64r128-L4 (md 152, mmlp 8, 986,829),
  M-s96r128-L5 (md 112 × 5, mmlp 24, 996,463), M-s96r128-L4r256 (md 112,
  mmlp 16, 995,113). Their probes run next. A filled shape that fits
  replaces its original in the long runs.

| cfg | probe MB | per slot | ms/step |
|---|---|---|---|
| M-s96r64-L4 | 1771 | 1771/1520/1318/1419/1268 | 918 |
| Nm4-s96r128-B5 | 2285 | 2285/1991/1771/1918/1697 | 988 |
| M-s64r128-L4 | 1897 | 1897/1645/1520/1582/1457 | 893 |
| M-s96r128-L5 | 2285 | 2285/1991/1771/1991/1771 | 1577 |
| M-s96r128-L4r256 | 2545 | 2545/2299/2054/2217/2054 | 1121 |
| M-s64r256-L4 | 2614 | 2614/2325/2132/2228/2035 | 1832 |

- The 1024 slot is the peak of every shape. The stage rank sets most of
  it: r 64 → 128 costs +360 MB at d64 and +470 MB at d96 (L5), r 128 →
  256 at d64 +720 MB. The ms/step of a probe (2 steps of the 4160 slot,
  the first one JIT) is noise: r256 main read 2628 at stage r 64 and 1121
  at r 128.
- M-s64r128-L4 left 1.1 GB, so its stage went to **r 256** (md 144, mmlp
  8, 991,977): M-s64r256-L4, 2614 MB.
- **Kaggle batch 2 (staged, the new binary)**: L-M-s64r256-L4,
  L-M-s96r128-L4r256, L-M-s96r128-L5, L-Nm4-s96r128-B5 (a middle shape:
  main d128 × 4 r 128). Against L-R-B5 (N 2, batch 19).
- Here: the N 3 probes of those four (`<cfg>-N3`: only N and λ change),
  so that the N 3 step is ready.
- **The fill, against the knob table** (`knobs.py`, notes §7; old
  protocol, Complex2D base, ±150 MB). `fit.py` fits only the params, so
  the probes choose the VRAM knob, and `fit.py` then refits the params.
  The stage state rank is the most VRAM per param (17.5 MB/kparam, with
  the stage MIMO 2 at 15.9). So it fills the VRAM gap of a main-heavy
  shape and costs the main the fewest params. At stage d96, r 256 would
  cost ~+940 MB (twice the r 64 → 128 step): 2285 + 940 > 2990. So
  M-s96r128-L5 and Nm4-s96r128-B5 keep ~700 MB unused. The knobs that
  could take it: the stage MIMO 2 (+690 MB, but 3.3× the ms/step), or a
  main knob (main MIMO 2: +64k params, ~0 MB). Deferred until a d96
  shape wins.
- **N 3 probes**: M-s64r256-L4-N3 2614, M-s96r128-L4r256-N3 2626,
  M-s96r128-L5-N3 2285, Nm4-s96r128-B5-N3 2285 MB: the N 2 peaks. The
  stages set the peak (the 1024 slot), so N frees no VRAM there. A larger
  N costs nothing in VRAM, and the main knobs stay cheap at either N.
- The rest of the old probes: M-s96r32-L4 1520, M-s112r64-L4 1768,
  W-m64r-B5 2303 MB.
- 23:17: the local GPU was idle until batch 19 ends, so **L-M-s64r256-L4
  runs here** (new binary). It also gives the probe-to-run factor of a
  main-heavy shape (suggestion 5). Its Kaggle lane takes the deferred
  7-slot contrast **L-R-N15-B** (R-N15-B5's model at 7 slots). Batch 2 is
  then: L-M-s96r128-L4r256, L-M-s96r128-L5, L-Nm4-s96r128-B5, L-R-N15-B.
- L-M-s64r256-L4 at step 100: **reserved 2711 MB, probe 2614: × 1.04**
  (R-N15-B5: × 1.17). 1744 ms/step (prof), 0.67 steps/s: phase 1 ≈ 65
  min, a little over the 1 h piece. Left to run. The shapes this heavy
  go to Kaggle. Step 300: still 2711 MB, 1314 ms/step (phase 1 ≈ 58
  min).

## Two stages (user, 23:30)

- Every shape so far has S = 1. `Nˢ` is per stage (the router of stage s
  keeps ~1/Nˢ of its positions), so the main sees ~1/ΠNˢ. The one S = 2
  test (b-S2b, an old screen) lost 0.050, but at a total compression of 4
  and with one main layer fewer. So two stages are untested.
- The second stage boundary costs ~50k params (d 128/128 1+1 / 1+1 r 104:
  1,039,361 against R-B5's 989,825 at mmlp 0).
- **S2a-B5**: R-B5 with its 4 stage layers split over two levels: outer
  d112 1+1, inner d128 1+1 (both Real1D r 128), N 1.41/1.41 (total 1.99,
  against R-B5's 2), main d128 × 3 Complex2D r 64, mmlp 0, 995,821. λ
  1.51 / 1.19. Its probe waits for the GPU here.
- **The extreme (user hunch)**: S = 2 at N 4/4, so the main sees 1/16 of
  the chars, with a main of ≥ 4 layers (a second hunch, uncertain). The
  user expects no gain at a short run (the stages still learn), but a
  gain at medium-long runs, mainly on the long horizon (the long stories,
  the recall).
  - **S2x-s64-L4**: outer d64 1+1 Real1D r 256, inner d96 1+1 r 128,
    main d136 × 4 Real1D r 128, mmlp 8, 998,797. λ 5.83 / 2.38.
  - **S2x-s96-L4**: outer d96 1+1 r 128, inner d112 1+1 r 128, main d112
    × 4 Real1D r 128, mmlp 48, 996,877. λ 4.32 / 2.0.
  - Risk: λ auto puts the outer stage at lr × 5.8 (2.3e-2). Old protocol:
    a stage 4+4 at d96 was unstable at its stage LR, and 1.6e-2 diverged
    in a screen. Decision: keep λ auto (the protocol), read the train
    bits early, and cap λ (`lrm=3/2/1`) if a run diverges.
  - Decision (the user's horizon point): judge them at 3230 **and** at
    the 4630 milestone. The trunk of each goes on to 4630 (`extend.sh`,
    + a 1000-step leaf), and so does its reference L-R-B5. Rank by wv, the
    long-stories rows and the recall at 4630.
  - Their probes are queued here after S2a-B5.
- **The contrast (user)**: S = 4 at N 2 each, the same total compression
  (16). **S4x-s64-L4**: 4 stages d64 1+1 Real1D r 128, main d120 × 4
  Real1D r 128, mmlp 40, 989,405. λ 5.48 / 3.87 / 2.74 / 1.94. The four
  boundaries eat params: at stages d64/64/96/96 the main fits only d96
  (979,585, `cfg/tmp-S4x-d96`), so all stages are d64. The same horizon
  plan as S2x (to 4630). Its probe is queued after the S2x ones.
