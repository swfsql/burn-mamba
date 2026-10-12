# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-11 session 2

- 17:56 **Decision**: h16-V7u to 14400 on the protocol (its 8000–12000
  anneal is in the saved schedule), in pieces < 1 h: `h16-V7u-x10000`
  (+2000, no leaf, ~33 min), then `h16-V7u-x12000` (+2000, ~33 min, then the
  leaf of 2400, ~39 min), then `long.sh -t`. Reason: the rank at 14400
  needs the reference there (R-N15-B5 0.7308). The flat's `extend.sh` got
  the `N2 = 0` trunk piece of the H-Net copy.
- 17:58 **Decision**: Kaggle batch: R-N15-r64V7, seeds 0 and 1, on the
  protocol to 5630 (`-x4630`) and 9600 (`-t6315`, `-x8000`), `ctx.sh` after
  each leaf. Reason: 9600 is the milestone where the flat leads (0.0085),
  and the seed noise (0.008) needs two seeds to rank a gap of that size.
  The step-down at 2630 (the best 5630 schedule) waits: 9600 decides.
- 18:00 **Stage virtual layers**: burn-stack gives each stage stack its own
  `NetworkShape`, so they need no library change. `gen.py` got `eV`/`dV`
  (per stage), `esched`/`dsched`, `suntie` (`auto`: on when a stack of the
  stage has virtual layers; the stage block is shared by its encoder and
  decoder, so both untie). The old configs regenerate byte-identical. The
  model builds (`params.sh`). Params: tied stage applications keep 999,425;
  each untied extra application costs ~6.6k.
- 18:01 **Decision**: probe 5 stage-reuse shapes (between the flat pieces),
  then dose the ones that a probe places in the band:
  - `R-N15-sV3t`: stage 3+3 applications of 2+2 real, tied, r 104 (999,425).
  - `R-N15-eV3t`: encoder only, 3 applications, tied, r 104 (999,425).
  - `R-N15-r64sV3` / `-r64sV4`: stage r 64 (the VRAM step below r 104),
    3+3 / 4+4 untied (965,697 / 978,689: a param rest to fill).
  - `R-N15-s11V2-m5`: 1+1 real × 2+2 untied applications, the freed ~260k
    params to the main (× 5) (1,003,873: over). Trimmed by the stage r
    104 → 96 (the same r 128 VRAM bucket): `R-N15-s11V2-m5r96`, 999,169.

  Reason: the flat gained by reuse (h16 → h16-V7u: −0.022 at 3230). The
  main reuse of the H-Net (R-N15-r64V7) only draws with R-N15-B5. The
  stage is where the H-Net spends its VRAM, so reuse there is the untested
  lever.
- 18:15 **User**: virtual layers are untied by preference (cheap in
  params, usually better). So the two tied shapes became untied, trimmed
  to the cap first by the main MLP filler (`mmlp=0`, neutral on a
  screen), then by the stage r inside the r 128 VRAM bucket: `R-N15-sV3u`
  (3+3, r 100, 998,289), `R-N15-eV3u` (encoder 3, r 104, 996,401). The
  tied `-sV3t` / `-eV3t` are dropped.
- 18:34 **Probes** (MB; R-N15-B5 2906 today, dose 3385): `-eV3u` 3678
  (over), `-sV3u` stopped at `--max-vram` (3875 MiB), `-s11V2-m5r96` 3485
  (over), `-r64sV3` 2279, `-r64sV4` 2616. At r 64 a stage application
  costs ~170 MB. At r 104 the one encoder application of `-eV3u` cost
  772 MB.
- 18:37 **User**: a power-of-2 state rank can help, and a higher rank can
  need *less* VRAM (the flat: h16 × r 96 3995 MB > h16 × r 128 3776 MB).
  So the r 100 / r 96 trims were the wrong knob, and the 772 MB of one
  r 104 application can be a page jump. **Decision**: probe the r 128
  class (`tmp-p2-*`, over the param cap, VRAM only): R-N15-B5 × r 128,
  `eV3` × r 128, `s11V2-m5` × r 128. Then dose the two r 64 shapes (power
  of 2 already), each filled with the main MLP (the design choice is the
  stage reuse): `R-N15-r64sV4` (mmlp 24, 997,121), `R-N15-r64sV54`
  (encoder 5, decoder 4, mmlp 16, 994,401).
- 18:41 **`gate.sh`** (user: yes): exit 0 when the last dose of a cfg is
  in 3380–3670 MB. The queue screens `R-N15-r64sV54` if its dose passes,
  else `R-N15-r64sV4` if its dose passes, else nothing. Then `long.sh` +
  `ctx.sh` on the leaf. Reason: the 5+4 shape has more reuse.
- 19:12 **The flat trunk** (the in-loop valid, the first 1024 chars):
  h16-V7u 0.7646 at 10000, 0.7438 at 12000 (h16 at 12000: 0.7632). The
  leaf to 14400 ends ~19:46, then `long.sh -t` gives its wv.
- 19:56 **h16-V7u at 14400: wv 0.7220.** Paired (`longstat.py -w`):
  R-N15-B5 +0.0086 ± 0.0016, R-B5 +0.0099, h16 +0.0194. The lead of the
  flat at 14400 equals its lead at 9600 (0.0085).
- 19:56 **The r 128 probes** (MB, VRAM only): a stage at r 128 needs
  *less* than at r 104 / r 96. B5 × r 128 2617 (r 104: 2906, −289),
  `eV3` × r 128 3099 (r 104: 3678, −579), `s11V2-m5` × r 128 2906 (r 96:
  3485, −579). This confirms the user's finding on the H-Net.
- 19:58 **User**: test the flat with its extra applications in the
  middle. h16-V7u is Cyclic: 0,1,2,3,4,0,1 (the 2 extras on layers 0, 1).
  The flat's `gen.py` got `sched=<list>` (Custom). The configs:
  `h16-V7u-m2` (0,1,2,2,2,3,4) and `h16-V7u-m13` (0,1,1,2,3,3,4), both
  995,921 params. **Decision**: this GPU runs them (the flat runs only
  here): a probe each (for the flat a probe = a dose, × 1.0), a screen
  only if the probe is in 3290–3670, then `long.sh -t` and `ctx.sh`. Also
  `ctx.sh` of h16-V7u-s2-d (the reference had no recall run).
- 19:58 **Decision**: drop the r64sV54 dose and the r 64 gated screen
  here. Reason: r 128 is cheaper than r 104, and the stage rank pays, so
  the r 128 class leads the H-Net line. The H-Net screens go to the next
  Kaggle batch (a dose, `gate.sh`, then the screen, on one line).
- 20:00 **The r 128 candidate**: `R-N15-r128s11V2-m5` = 1+1 real stage
  layers × 2+2 untied applications at r 128, main d128 × 5, `mmlp=0`,
  `rf=0.5` (the main only: the stage is Real1D): 991,665. Reason for
  `rf=0.5`: it trims the last 10k; it is the reference model's setting,
  and it tied in the flat (`1m-b`: 0.8584 against 0.8579).
- 20:12 The flat probes: `h16-V7u-m2` and `-m13` 3454.8 MB (= h16-V7u).
  The `m2` screen runs from 20:05. `R-N15-r64sV4` dose: **3000 MB**, under
  the band (× 1.15 of its probe).
- 20:14 **Decision**: the next Kaggle batch (staged in `kqueue.txt`, it
  goes up when the r64V7 batch ends): `R-N15-r128s11V2-m5` and
  `R-N15-r64sV5` (stage 5+5 untied applications of 2+2 real at r 64,
  `mmlp=8`, 991,681), each as dose → `gate.sh` → screen → `ctx.sh`.
  Reason for r64sV5: one more application per stack adds ~390 MB to the
  3000 MB of r64sV4, so it lands in the band, and it completes the r 64
  stage-reuse line.
- 21:12 **`h16-V7u-m2`** (0,1,2,2,2,3,4) at 3230, paired against the
  Cyclic h16-V7u-s2-d: wv 0.8688, **+0.0043 ± 0.0018** (worse). The long
  stories +0.0030 ± 0.0014. The recall: far own +0.0005 ± 0.0011 (equal),
  re-mention letters own **−0.044 ± 0.004** (better). So: a little worse
  in bits (~2σ, one seed, the flat's seed noise is ~0.002), better in the
  re-mention letters. `m13` screens now (ends ~22:00).
- 21:40 **R-N15-r64V7 on the protocol** (Kaggle, dv = wv, seeds 0 / 1):
  5630 **0.8088 / 0.8087** (h16-V7u 0.8037, R-N15-B5 0.8143); 9600
  **0.7699 / 0.7674**, mean 0.7687 (h16-V7u 0.7487: +0.020; R-N15-B5
  0.7572: +0.0115). So the main reuse at a stage r 64 leads R-N15-B5 at
  5630, then falls behind it by 9600. **Closed**: it does not keep up.
  Paired against h16-V7u-x8000-d (`longstat.py`): wv +0.0196 / +0.0175
  (R-N15-B5 +0.0076), the long stories +0.0224 / +0.0241. `kwait.sh`
  missed the two seed-1 leaves again: their `long.sh` lines went into the
  queue by hand.
- 21:42 **Decision**: push the staged batch (`R-N15-r128s11V2-m5`,
  `R-N15-r64sV5`, each dose → gate → screen → ctx), budget 3 h.
- 22:01 **`h16-V7u-m13`** (0,1,1,2,3,3,4) at 3230, paired against Cyclic:
  wv 0.8694, **+0.0050 ± 0.0016**. The long stories +0.0092 ± 0.0015. The
  recall: far own +0.0037 ± 0.0011 (worse), re-mention letters own
  −0.043 ± 0.005 (better, as `m2`). **Result**: both middle schedules lose
  ~0.005 wv against Cyclic (0,1,2,3,4,0,1: the extras at the top of the
  stack, on the first real layers). Both gain ~0.044 on the re-mention
  letters.
- 22:02 **Decision**: this GPU runs a second seed of h16-V7u at 3230
  (`h16-V7u-s1`, only the seed differs, so the dose carries over), then
  `long.sh -t` and `ctx.sh`. Reason: it measures the flat's seed noise,
  so it tells if the 0.005 of the middle schedules is real. It also
  firms the 0.8644 that ranks every H-Net at 3230.
- 23:00 **h16-V7u seed 1** at 3230: wv **0.8559**, −0.0078 ± 0.0022
  against seed 0 (0.8644). The long stories −0.0079. The recall: far own
  −0.0115, re-mention letters own −0.062. So **the flat's seed noise is
  ~0.008 too**, as the H-Net's. The two-seed mean of h16-V7u at 3230 is
  **0.8602**. The H-Net two-seed means: R-N125-B5 0.8631, R-N15-B5 0.8647,
  R-N15-r64V7 0.8660: the flat leads all of them at 3230 too (0.003 to
  0.006). The middle schedules (0.8688, 0.8694) are worse than both flat
  seeds (~0.009 against the mean), but one seed each is inside the noise,
  and so is their re-mention gain.
- 23:00 **Decision**: a second seed of `h16-V7u-m2` (`h16-V7u-m2-s1`)
  here, then `long.sh -t` and `ctx.sh`. Reason: a two-seed mean of the
  middle schedule against the two-seed mean of Cyclic decides the user's
  question.
- 23:20 **Error (Claude): the 9600 runs of R-N15-r64V7 are not on the
  protocol.** The H-Net config holds no step-down: the protocol adds it
  with `extend.sh … --lr 1e-3` on the x4630 → t6315 piece (R-N15-B5 did
  so). The r64V7 lines (17:58) omitted `--lr 1e-3`, so their trunk stayed
  at 4e-3 to 8000 and the 9600 leaf decayed from 4e-3 (the saved
  `training_config.json`). A step-down at 82% costs the H-Net ~0.01
  (`notes.md` §2), about the whole +0.0115 against R-N15-B5. So the 21:40
  "Closed" is **withdrawn**: `L-R-N15-r64V7(-s1)-t6315/-x8000` are the
  "no step-down" schedule, not the protocol. The 5630 result stands (the
  x4630 trunks and leaves have the same schedule as R-N15-B5's): r64V7
  leads R-N15-B5 by 0.0055 there (two seeds).
- 23:20 **Decision**: the protocol rerun of both seeds from the x4630
  trunks (`-pt6315`, then `-px8000` + the leaf of 1600, `ctx.sh`), staged
  as the next Kaggle batch.
- 23:30 **The audit** (`lrprof.py`, every run of both workspaces against
  the protocol, step by step): the only error is the 6 r64V7 runs
  (`-t6315`, `-x8000`, `-x8000-d`, both seeds). The other new-protocol
  deviations are the deliberate `sd2630` (9 H-Net, 4 flat) and `h6315` (6)
  runs. 112 more are the old-protocol screens (another peak LR from step
  1). Every ranked comparison in `notes.md` is on matched schedules.
- 23:30 **The cause**: the flat's configs hold the trunk schedule
  (`stages=4630:1e-3,8000-12000:5e-4`), so its extensions follow it with
  no flag. The H-Net's configs held only the ramp (`stages=""`): its
  protocol needed `--lr 1e-3` at the 4630 → 6315 piece, and nothing
  checked it. The user expected the flat's behavior.
- 23:30 **The fix** (user: a trunk scheduler as the flat's):
  - `gen.py`: `stages` defaults to the protocol (byte-equal to the flat's
    schedule). `cfg/` regenerated from `args.txt`: 149 configs now hold
    the protocol. 156 old-protocol configs went back to their old files
    (a regeneration added today's `bucket` default to them). Nothing else
    changed (checked against a backup).
  - `lrprof.py --fill`: 50 on-protocol trunks got the rest of the
    protocol (their segments before the saved step stay; the audit is
    unchanged after it). So every extension follows the protocol with no
    `--lr`, and the staged r64V7 rerun has no `--lr`.
  - Guards: `extend.sh` (both workspaces) runs `lrprof.py --guard` on the
    source when it has no `--lr`. `kpush.sh` runs `lrprof.py --batch` and
    does not push a failing batch. Tested on the 17:58 batch (fails at
    step 4631) and on protocol, chained and deliberate lines (pass).
  - The batch on Kaggle now (`r128s11V2-m5`, `r64sV5`) went up with the
    old configs: `lrprof.py --fill` its two trunks after the pull.
- 23:31 **The Kaggle batch** (dose → gate → screen):
  - `R-N15-r128s11V2-m5`: dose **3193 MB**, under the band: the gate
    stopped it (no screen). It needs ~1 page more.
  - `R-N15-r64sV5`: dose 3390–3522 MB; the screen at 3230: **wv 0.8670**
    (R-N15-B5 two-seed mean 0.8647, h16-V7u 0.8602). The stage reuse at
    r 64 does not pay. Its trunk got the protocol (`--fill`).
- 23:33 **Decision**: push the r64V7 rerun (kernel version 32; the
  `kpush.sh` LR check passed).
- 23:35 **Decision**: spend the missing page of `r128s11V2-m5` by N 1.5 →
  1.25 (`R-N125-r128s11V2-m5`, 991,665, λ 1.12/1): the main then runs on
  80% of the tokens (67% at N 1.5). Reason: no params, and N 1.25 = N 1.5
  on R-N15-B5 (two seeds each). A main V6 (one untied application, +8.6k)
  was 225 params over the cap. Dose here after the flat, then a gated
  screen.
- 00:12 **`h16-V7u-m2` seed 1**: wv 0.8571 (Cyclic seed 1: 0.8559,
  +0.0012). Two-seed means: Cyclic **0.8602**, `m2` **0.8630** (+0.0028,
  inside the noise). The re-mention letters: `m2` gains ~0.017 within each
  seed pair (seed 0 −0.044 + 0.000, seed 1 −0.075 against −0.062).
  **Result**: the middle placement does not beat Cyclic in bits. Cyclic
  stays the flat reference.
- 00:12 `R-N125-r128s11V2-m5`: dose **3482 MB** (in the band, the gate
  passed). Its screen runs here from 00:09 (~70 min).
- 00:20 **User hypothesis**: two layers applied twice (`m13`,
  0,1,1,2,3,3,4) fare better than one layer applied three times (`m2`).
  Seed 0: `m13` − `m2` = +0.0006 wv (equal), the long stories +0.0062,
  far own +0.0032, re-mention letters ≈ 0. One seed cannot decide it (the
  paired schedule gaps move ~0.003 between seeds). **Decision**: `m13`
  seed 1 here after the H-Net screen, for a two-seed mean of all three
  schedules.
- 00:35 **Re-mention letters against wv** (user: the letter gain of `m2`
  looks larger than its wv loss). The letters are ~12% of the chars of
  the far window (44,338 of 1968 × 192). Weighted, the gain is worth
  −0.005 / −0.002 per char of the window (seeds 0 / 1), and the measured
  far own of `m2` is +0.0005 / +0.0003: the other 88% of the window lose
  what the names gain. So `m2` trades the local prediction for the entity
  recall; the net is wv +0.0028. The letters metric moves ~0.06 between
  seeds of one model.
- 01:12 `R-N125-r128s11V2-m5` trunk at 2630: sv **0.9698**, the best H-Net
  trunk (R-N15-B5 0.9809 / 0.9748, R-N125-B5 0.9794 / 0.9855, R-N15-r64V7
  0.9864, R-N15-r64sV5 1.0027). The leaf (wv) ends ~01:25.
- 01:25 **`R-N125-r128s11V2-m5` at 3230: wv 0.8533**, the first H-Net
  ahead of the flat (h16-V7u 0.8644 / 0.8559, mean 0.8602: −0.0069; R-N15-B5
  mean 0.8647: −0.0114). Paired against flat seed 0 (flat seed 1 in
  brackets): wv −0.0102 ± 0.0019 (−0.0078), the long stories −0.0108 ±
  0.0016 (−0.0079), far own −0.0171 ± 0.0013 (−0.0115), re-mention letters
  own −0.126 ± 0.006 (−0.062). It beats both flat seeds on every row. One
  seed. Shape: stage d128, 1+1 real R1D layers at r 128 × 2+2 untied
  applications, N 1.25, main d128 × 5 C2D r 64 rf 0.5, 991,665 params, a
  3482 MB dose, 1.37 s/step here.
- 01:30 **Decision**: the next Kaggle batch (staged, the LR check
  passes): seed 0 to 5630 and 9600 on the protocol, and seed 1 (a screen,
  the dose carries over) to 3230, 5630 and 9600, `ctx.sh` after each leaf.
  Reason: the flat used to pull ahead later (9600, 14400), and one seed
  ranks only a gap ≳ 0.01.
