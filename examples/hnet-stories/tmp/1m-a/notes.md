# Notes: hnet-stories ≤1M-param search (1m-a)

The findings of the search, by topic. §4–§7 come from the **old protocol**
(the first 1024 chars of each story, before 2026-10-09 20:40). Those runs
are invalid as results: they never trained past char 1024. Their findings
are the starting points of the new protocol, not its numbers.

Terms:

- **wv**: the whole valid (the 256 valid stories, each in full). The
  ranking metric.
- **dv**: the decayed valid of the training loop. From binary `4e42bba3`
  (2026-10-09 22:44) on, the dv of a bucketed run **is** wv (checked to
  1e-8). Before, dv cut each story at 1024 chars, and wv came from
  `long.sh`.
- **sv**: the stable valid, before the decay.
- A **long run**: 2630 + 600 steps at lr 4e-3 (`screen.sh`). A **screen**
  (old protocol): 600 + 200 steps at lr 8e-3.
- A variant has the name of its change from its base (`b × N 1.5`).

## 1. The reference and the metrics

- **The flat reference**: h16 of `tiny-stories/tmp/1m-cased` (packed 2 ×
  4224 rows: whole stories with full backprop, 3149 MB without gc).

  | step | wv | dv (first 1024) |
  |---|---|---|
  | 3230 | 0.8868 | 0.8798 |
  | 5630 | 0.8204 | 0.8137 |
  | 9600 | 0.7660 | 0.7597 |
  | 14400 | 0.7412 | 0.7353 |

  At 3230 also: fr-d152 (the flat with the Real1D r 128 block of the
  H-Net) wv 0.8792, h16-B (the flat at the 7 buckets) 0.8924.
- **wv**: 207,187 chars, 32 stories longer than 1024. 12% of the stories
  are longer than 1024, and 7.2% of the valid text is past char 1024
  (0.8% past 2048). `long.sh` also splits it (0..1024, past 1024).
- **The long stories** (`long.sh`): the 157 valid stories longer than 2048
  chars (2056..4064, mean 2463), from the whole valid split, each in one
  `forward`. The right padding is exact (batch 1 = batch 4).
- **The recall** (`ctx.sh -g -n 2048`, 1968 stories, paired): the s.e. of
  a difference is ~0.001 on the far bits, ~0.01 on the re-mention first
  letter. The valid bits hide the recall.
- **The seed noise** (new protocol): R-N15-B5 at seeds 0 and 1 differ by
  **0.0083** in wv (paired −0.0084 ± 0.0019; the ± is the valid-set noise
  only). The shift is the same in every row (0..1024, past 1024, the long
  stories), and in the recall (re-mention letters own −0.037). So one run
  per shape ranks only a gap ≳ 0.01. Below that, add a seed.

## 2. The protocol (2026-10-09)

- **Why it changed**: the old H-Net (first 1024 chars only) got worse past
  char 1024, and the flat model got better. At 14400, the bits per 256
  chars of the H-Net rose from 0.866 to ~1.0 along a story, those of the
  flat fell to ~0.83.
- **Whole stories fix the tail**: L-B-N15-lr4 (the 7 buckets) against
  L-N15-lr4 (first 1024), on the long stories: 1024..2048 −0.043, past
  2048 −0.062, chars 0..1024 −0.001.
- **The length grouping costs the story heads**: the new baseline against
  the old R-N15 at 3230: chars 0..1024 +0.008, past 1024 about −0.07, all
  +0.003. (B-c1024, the old data only grouped: +0.008. The flat at 7
  buckets: +0.006.)
- **5 slots** (1024,1536,2048,3072,4160): every story up to 1024 chars
  (88%) mixes in one slot. Batches × stories per slot: 20960 × 11, 3353 ×
  6, 1879 × 4, 1199 × 3, 177 × 2. **7 slots tie**: L-R-N15-B 0.8697
  against R-N15-B5 0.8605 / 0.8688.
- **The validation** of a bucketed config takes the same slots and `chars`
  (burn-stack `dataloaders_for`): each valid story whole, in one window.

## 3. The results of the new protocol (wv at 3230)

| run | shape | wv | 0..1024 | past 1024 | long all | run MB |
|---|---|---|---|---|---|---|
| L-R-N15-B5-d | R-N15 (N 1.5), seed 0 | 0.8688 | 0.8612 | 0.9529 | 0.9981 | 3386 |
| L-R-N15-B5-s1-d | seed 1 | **0.8605** | 0.8536 | 0.9377 | 0.9889 | 3386 |
| L-R-B5-d | R-B5 (N 2) | 0.8723 | 0.8650 | 0.9525 | 1.0051 | ~3390 |
| L-R-N15-B-d | R-N15, 7 slots | 0.8697 | 0.8614 | 0.9621 | | 2799 |
| L-M-s96r128-L5-d | stage d96, main d112 × 5 | 0.9022 | 0.8949 | 0.9838 | 1.0308 | 2440 |
| L-Nm4-s96r128-B5-d | stage d96, main d128 × 4 | 0.9094 | 0.9025 | 0.9854 | 1.0397 | 2505 |
| L-M-s96r128-L4r256-d | stage d96, main d112 × 4 r 256 | 0.9240 | 0.9167 | 1.0057 | | 2887 |
| L-M-s64r256-L4-d | stage d64 r 256, main d144 × 4 | 0.9723 | | | | 2711 |

- **The H-Net leads the flat** (0.8868) by 0.015–0.026.
- **N 1.5 and N 2 tie**: R-B5 +0.0035 against the seed mean of R-N15-B5
  (0.8647). The recall of N 2 is also within the seed spread.
- The run MB are the pool reserved MB, under the old cleanup schedule (see
  §7): the M runs were 500–950 MB under the VRAM ceiling, so they are
  weak evidence (strict rule 1).

## 4. The stage width is the lever (new protocol)

- **The M family** (the user's hunch: with whole stories, a stronger main
  should matter more) **loses** at 3230: +0.030 (L5), +0.038 (Nm4),
  +0.052 (L4r256), +0.10 (stage d64), on every row. The long stories too
  (L5 +0.024 past 2048, ± 0.007): no long-horizon gain at 3230.
- The loss follows the width of the char-level stage: d64 +0.10, d96
  +0.03 to +0.05.
- **The stage LR is not the cause** (D-M-s64r256-L4-lrm, 656 steps, the
  same batches): λ 2.12 (auto) and λ 1.41 (R-B5's stage LR) tie by step
  656 (train bits 1.428 / 1.418), both 0.13 behind R-B5 (1.291). So
  `lrm=auto` stays.
- Not tested yet: the M shapes at the VRAM ceiling, the M shapes at the
  longer milestones.

## 5. Two and four stages (new protocol, not run yet)

- `Nˢ` is per stage (the router of stage s keeps ~1/Nˢ of its positions),
  so the main sees ~1/ΠNˢ of the chars. The one old S = 2 test (b-S2b, a
  screen) lost 0.050, but at a total compression of 4 and with one main
  layer fewer.
- **The params**: a stage boundary costs ~50k at d128 (d128/128 1+1 / 1+1
  r 104: 1,039,361 against R-B5's 989,825). The widths must not shrink
  inward (a narrower inner stage does not build). S2 N 4/4 at d128/128
  with a main d128 × 4 r 128: 1.25M. **S4 fits 1M only with d64 stages**
  (at d96 ×4: 1.17M). A second outer layer pair (2+2) costs ~160k at d112,
  ~50k at d64. Outer r 256 + main r 64 costs the same params as outer r
  128 + main r 128.
- **The VRAM**: a 1+1 outer stage holds about half the VRAM of a 2+2 one.
- **S2a-B5** (R-B5's 4 stage layers over two levels: outer d112 1+1,
  inner d128 1+1, r 128, N 1.41/1.41, main d128 × 3): it trailed R-B5 by
  0.03–0.045 in train bits up to step 1424 (the same batches), then it
  stopped: its run held 2473 MB, ~1 GB under the ceiling.
- **The user's hunches**: S = 2 at N 4/4 (the main sees 1/16), against S =
  4 at N 2 (the same total), with a main of ≥ 4 layers. The gain should
  show at medium-long runs, mainly on the long horizon. So judge them at
  3230 **and** at 4630 (the trunk and its reference L-R-B5 go on to
  4630), by wv, the long-stories rows and the recall.
- λ auto puts an outer stage of these shapes at lr × 4.3–5.8. Watch the
  early train bits. Cap λ (`lrm=3/2/1`) if a run diverges.
- **The refilled candidates** (2026-10-10, near both ceilings):

  | cfg | shape | params | dose peak max (steady) MB | ms/step |
  |---|---|---|---|---|
  | S2x-s112r256-L4 | d112/112 1+1/1+1, r 256/128, N 4/4, main d112 × 4 r 64 | 995,425 | 3029 (2861) | 805 |
  | S2a2-B5 | d112/120 1+1/1+1, r 256/128, N 1.41/1.41, main d120 × 3 C2D r 64 | 992,028 | **3532 (3364)** | 1118 |
  | S2x-s64e2-L4 | d64/64 2+2/1+1, r 256/256, N 4/4, main d128 × 4 r 128 | 989,073 | 2710 (2324–2421) | 927 |
  | S4x-s64e2-L4 | d64 ×4, outer 2+2, r 256 then 128, N 2, main d104 × 4 r 128 | 998,213 | 2617 (2429–2526) | 1246 |

  The doses: 520 steps, `--vram-cleanup-every 1` (§8); R-N15-B5 has 3386.
  **Only S2a2-B5 is at the target** (~3400–3550). The others need a
  deliberate design change for +400 to +900 MB (strict rule 14). The
  first `prof` block holds the largest peak of a run. Real1D everywhere
  unless noted. The S2x-s64e2 / S4x-s64e2 pair has equal stage widths
  (d64): it ranks S against N, and carries the d64 handicap (§4).

## 6. The recipe (old protocol)

- **The head**: the untied head (`tie=0`) won by 0.036.
- **The init**: the Burn init beat the reference init by 0.032.
- **λ** (the stage LR multiplier) auto beat 1/1 by 0.019.
- **The LR**: long runs at **4e-3** (8e-3 0.9070, 4e-3 0.8915, 2.5e-3
  0.8960).
- **The base b**: stage d128 2+2, main d136 × 3 r 64, untied head, Burn
  init, N 2, λ auto. Stage 4+4 at d96 lost 0.08 (unstable). One main layer
  lost.
- **The param filler**: the main MLP at `mlpmul=8` (`mmlp`) costs no VRAM
  and is neutral on a screen. Use it only for a small rest (strict rule
  14).

## 7. The block, N and the main (old protocol)

- **Real1D stages at a large state rank**: the rank pays, not the kind.
  F-r1d-sr104 screen 1.1076 (F-base 1.1294). Long: R-N15 0.8534 / 0.8485.
- **The trapezoid** (and the B/C bias) replace the deleted short conv:
  Trapezoid None lost 0.042 and 0.064.
- **Lost**: main virtual layers (Cyclic +0.020, middle-heavy +0.010), two
  stages (+0.050, not budget-matched), main × 2 (+0.027), stage 1+1 with a
  heavier main (+0.025–0.06), a main at rank 32 (+0.027), stage 3+3 at
  d112 (+0.007).
- **Tied**: Multi-Gate 4, a Real1D main, weight decay 0.05, α 0.1, N 1.25.
- **Compression trades local bits for entity recall**: N 2 had the better
  re-mention letters and far use at 3230.

## 8. VRAM

- **The ceiling** is on the VRAM of the process (strict rule 1). Never
  decide by `nvidia-smi` (strict rule 2).
- **How cubecl spends it** (48301b5, the `Adaptive` layout, the `Sync`
  allocator):
  - `bytes_reserved` (`Device::memory_pool_usage`) is the sum of the pages
    that the pools hold.
  - A pool drops a page on its own (idle past its period, or left behind
    by a growth of the arena pages). The storage only queues it
    (`GpuStorage.deallocations`). `cuMemFree` runs at `storage.flush()`,
    and only an explicit cleanup (`Device::memory_cleanup`) flushes. The
    periodic cleanup of each `reserve` never flushes. The storage counter
    (`GpuStorage.allocated`) is exact, but the public `MemoryReport` does
    not carry it.
  - The CUDA context and the loaded kernels are outside the pools (~140
    MiB).
- **The measure** (burn-stack, 2026-10-10): `--vram-cleanup-every 1`
  cleans up before every window, so the queue never outlives a window.
  `vram::check` records the high-water mark of the pools after every
  backward, and the `prof` line prints it (`peak X MB`). So **the process
  ≈ peak + ~140 MiB**. `args.sh` adds the flag to every run.
- **R-N15-B5, 520 steps**: the old schedule and `--vram-cleanup-every 1`
  have the same peak (3386 MB). The cleanup before every window is ~10%
  faster (1018–1063 against 1134–1188 ms/step).
- **The dose target**: a peak of ~3400–3550 MB. `probe.sh` (10 steps)
  only for the coarse search: a run holds more, by a shape-dependent
  factor (× 1.04 M-s64r256-L4, × 1.17 R-N15-B5, × 1.32 S2a-B5).
  `dose.sh` (520 steps, ~10 min) gives the peak of the run.
- **What moves the peak**: the stages set it (the 1024 slot). The stage
  state rank is the most VRAM per param (r 64 → 128: +360 MB at d64,
  +470 at d96; r 128 → 256 at d64 +720). N does not move it (the N 3
  probes of four M shapes: the N 2 peaks). The main knobs are cheap.
- **Power-of-two state ranks**: another rank costs the VRAM of the next
  power of two, or more. A Complex2D stage above r 64 does not fit.
- The budget table of the knobs (old protocol, Complex2D base, ±150 MB):
  `knobs.py`. VRAM-heavy and param-cheap (~16–18 MB per kparam): the
  stage state rank, the stage MIMO 2 (but 3.3× the ms/step), the main
  virtual layers. Param-heavy and VRAM-free: the main MLP, the main
  expand.

## 9. Kaggle

- Two T4s, the same pool accounting as here. A T4 trains ~1.4× this card
  (R shapes: phase 1 51–54 min, the leaf ~12.5 min).
- Kaggle's unpacking of a `.tar.gz` stalls for hours. The payload goes up
  as `payload.tgz.bin` in a new private dataset per batch.
- The CLI sends the `KGAT_` token as a Bearer token (`kg.sh`). The CLI
  cannot cancel a kernel: the user does it on the web.
- `kwait.sh` queues `long.sh` for the pulled runs in the local queue (a
  GPU job beside a probe spoils it, and killed two probes).

## 10. Lessons

- **A new config field** (strict rule 9): never an `Option`. A Burn config
  default does not apply when a config loads.
- A run that starts in the second of a kill can fail in cubecl: wait ~20 s.
- The bucket summary counts per slot.
- **A waiter on `pgrep -f`** matches itself and never ends (strict rule 4).
- **`build.sh`**: `cp … && mv …` does not stop `set -e`. On a full disk,
  `bin/hn-cuda.txt` then named a binary that did not install. Fixed: one
  command per line and a `cmp`.
- **The disk** (627 GB, shared with every project) filled up on
  2026-10-10. `cargo`'s `target/debug` was 115 GB.
- **The Edit tool** drops the trailing space of an edit string: two queue
  lines lost their `-- ` that way.
- **Never edit a script while a queue line runs it**: bash reads a script
  as it runs.
- `nvidia-smi` counts the host, and the host usage moves (strict rule 2).
