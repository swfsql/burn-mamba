# Notes: hnet-stories ≤1M-param search (1m-a)

The findings of the search, by topic. §3–§7 come from the **old protocol**
(the first 1024 chars of each story, before 2026-10-09 20:40). Those runs
are invalid as results: they never trained past char 1024. Their findings
are the starting points of the new protocol, not its numbers.

Terms:

- **dv**: the decayed valid (the 256 valid stories, their first 1024
  chars).
- **wv**: the whole valid (the same stories in full).
- **sv**: the stable valid, before the decay.
- A **screen**: 600 + 200 steps at lr 8e-3. A **long run**: 2630 + 600
  steps at lr 4e-3.
- A variant has the name of its change from its base (`b × N 1.5`).

## 1. The reference and the metrics

- **The flat reference**: h16 of `tiny-stories/tmp/1m-cased` (packed 2 ×
  4224 rows: whole stories with full backprop, 3149 MB without gc).

  | step | wv | dv |
  |---|---|---|
  | 3230 | 0.8868 | 0.8798 |
  | 5630 | 0.8204 | 0.8137 |
  | 9600 | 0.7660 | 0.7597 |
  | 14400 | 0.7412 | 0.7353 |

  At 3230 also: fr-d152 (the flat with the Real1D r 128 block of the
  H-Net) wv 0.8792, h16-B (the flat at the 7 buckets) 0.8924.
- **wv** (`long.sh`): 207,187 chars, 32 stories longer than 1024. Its
  chars 0..1024 reproduce dv exactly. 12% of the stories are longer than
  1024, and 7.2% of the valid text is past char 1024 (0.8% past 2048).
- **The long stories** (`long.sh`): the 157 valid stories longer than 2048
  chars (2056..4064, mean 2463), from the whole valid split. Each one runs
  in one `forward`. Batch 1 and batch 4 give the same bits: the right
  padding is exact.
- **The recall** (`ctx.sh -g -n 2048`, 1968 stories, paired): the s.e. of
  a difference is ~0.001 on the far bits, ~0.01 on the re-mention first
  letter. The valid bits hide the recall.

## 2. Why the protocol changed (2026-10-09)

- **The old H-Net got worse past char 1024, and the flat model got
  better.**
  At 14400 (bits/char pooled, R-N15 / flat): chars 0..1024 0.8458 /
  0.8495, 1024..2048 0.9023 / 0.8636, past 2048 0.9061 / 0.8387, the long
  stories all 0.8794 / 0.8535. The bits per 256 chars of the H-Net rose
  from 0.866 to ~1.0 along a story. Those of the flat fell to ~0.83.
  The cause is the old protocol: it trained the first 1024 chars only.
- **Whole stories fix the tail**: L-B-N15-lr4 (b at the 7 buckets) against
  L-N15-lr4 (b, first 1024), on the long stories: 1024..2048 −0.043, past
  2048 −0.062, chars 0..1024 −0.001.
- **The length grouping costs**: on wv, L-B-N15-lr4 lost 0.010 overall
  (chars 0..1024 +0.015, past 1024 −0.044). B-c1024 (the old data, only
  grouped 512/768/1024) cost +0.008. The flat model at the 7 buckets lost
  0.006 of wv against its packed rows.
- So the protocol has **5 slots** (1024,1536,2048,3072,4160): every story
  up to 1024 chars (88%) mixes in one slot. Batches × stories per slot:
  20960 × 11, 3353 × 6, 1879 × 4, 1199 × 3, 177 × 2 (27,568 batches). The
  1024 slot holds 11 × 1024 tokens (fill 0.69). The old protocol held 10 ×
  1024.
- **The first run** (R-N15-B5, R-N15 at the 5 slots): sv@2630 0.9809,
  3386 MB steady, 1.28 s/step (the old protocol ~1.0 s/step).

## 3. The recipe (old protocol)

- **The head**: the tied head starts at 117 bits/char (it predicts the
  current char). The untied head (`tie=0`) won by 0.036.
- **The init**: the Burn init beat the reference init (N(0, 0.02) + the
  residual rescale) by 0.032.
- **λ** (the stage LR multiplier) auto beat 1/1 by 0.019.
- **The LR**: screens 8e-3 (4e-3 +0.016, 1.2e-2 +0.017, 1.6e-2 diverges,
  5e-3 ties). Long runs **4e-3**: at N 1.5, 8e-3 dv 0.9070, 4e-3 0.8915,
  2.5e-3 0.8960. A hot trunk gains more in the decay but loses overall.
- **The base b**: stage d128 2+2 (Complex2D r 32), main d136 × 3 r 64,
  untied head, Burn init, N 2, λ auto. Stage d144 tied d128. Stage 4+4 at
  d96 lost 0.08 (unstable at the stage LR). One main layer lost.
- **The param filler**: the main MLP at `mlpmul=8` (`mmlp`) costs no VRAM
  and is neutral on the screen (F-base 1.1294, b 1.1286).

## 4. The block (old protocol)

- **The lever: Real1D stages at a large state rank**, budget-matched.
  F-r1d-sr104 (Real1D r 104, recalc:32) screen 1.1076 (F-base 1.1294).
  Real1D r 64 1.1231 and Complex2D r 64 1.1282: the rank pays, not the
  kind. Real1D r 32 with the freed budget unspent lost 0.036. Long:
  L-R-lr4 (N 2) 0.8617, **R-N15** (N 1.5) 0.8534 / 0.8485 (two seeds).
  R128c (Real1D everywhere, stage r 128, main r 32) tied at ~0.8 GB less.
- **The trapezoid** (and the B/C bias) replace the deleted short conv:
  Trapezoid None lost 0.042 (R-trapN) and 0.064 (R128a).
- **Lost**: main virtual layers (Cyclic +0.020, middle-heavy +0.010), two
  stages (+0.050, not budget-matched).
- **Tied**: Multi-Gate 4, a Real1D main (R-all), weight decay 0.05 (against
  0.1), the ratio weight α 0.1 (against 0.3), N 1.25 (against 1.5).

## 5. The compression N and the main network (old protocol)

- **Less compression won the bits**: screens N 1.5 / 2 / 3 1.1175 /
  1.1286 / 1.1404. Long at 8e-3 0.9070 / 0.9248 / 0.9482. At 4e-3 N 1.5
  and N 2 tied in b (0.8915 / 0.8927). In the R family N 1.5 led by 0.008
  at 3230 and by 0.005 at 9600 (0.7461 / 0.7515).
- **Compression trades local bits for entity recall**: at 800 steps N 2
  had the best re-mention first-letter gain (0.697, N 1.5 0.467, flat
  0.631). At 3230 (lr 4e-3) N 2 had the better far use (0.217 / 0.203)
  and re-mention letters (own 1.070 / 1.135).
- **The allocation**: R-N15 holds 54% of its params in the 4 stage layers,
  39% in the 3 main layers (the H-Net paper does the reverse). The main
  runs at 1/N of the positions, so its VRAM per param is about half.
  - The main depth matters: main × 2 with the stages kept lost 0.027
    (L-A-s22-L2-N15).
  - Main-heavy shapes that thin the stages to 1+1 lost 0.025–0.06
    (screens, against R128c).
  - A main at rank 32 lost 0.027 (L-W-s112-m, N 2).
  - Stage 3+3 at d112 lost 0.007 (long, two seeds).
- **The user's hunch** (2026-10-09): with whole stories, a stronger main
  and a higher N should matter more. A stronger main has a 4th layer, a
  larger d and a larger state rank, at the cost of the stage d and rank.
  The `M-*` family (§8) tests it. It has not run yet.

## 6. The H-Net against the flat model (old protocol)

- **dv**: R-N15 led by 0.026 at 3230, 0.010 at 5630, 0.014 at 9600, 0.015
  at 14400. It also led every recall row at 3230 and 5630.
- **The block control**: fr-d152 (the flat with the Real1D r 128 block)
  gained 0.008 at 3230 and nothing at 5630 (+0.003). So the gain was the
  H-Net, not its block.
- **wv** (computed after the switch): R-N15 0.8659 / 0.8164 / 0.7580 /
  0.7322 at the four milestones. It still led the flat, by less.

## 7. VRAM

- **The budget table** (`knobs.txt`, 10-step probes of b with gc, ΔMB /
  Δparams). VRAM-heavy, param-cheap (~16–18 MB per kparam): the stage state
  rank (r 64: +799 / +46k), the stage MIMO 2 (+690 / +43k), the main
  virtual layers (+585 / +35k). Param-heavy, VRAM-free: the main MLP, the
  main expand. The head count is a hidden cost: stage head dim 32 frees
  603 MB. Real1D stages free ~600 MB. gc saves 360 MB (−15%).
- **Power-of-two state ranks**: another rank costs as much as the next
  power of two, or more. Complex2D stage r 48 3130 MB, 64 2675, 72 4351,
  96 5127, 128 4426. Real1D r 64 1671, 72 2829, 128 2300. The user: the
  cubecl JIT (its tiling follows the dimensions). A Complex2D stage above r
  64 does not fit. Real1D r 128 costs less than r 104.
- **The chunk**: the default chunk is √(r·p) rounded up to a multiple of
  32, so it doubles at r ≥ 80. `--ssd-path recalc:32` pins it (no value
  changes).
- **The probe against a run**: unbucketed, a run held ~1.06–1.12 × its
  step-10 probe. At the buckets, `probe.sh` runs 2 batches of each slot
  (`--bucket-probe 2`). R-N15-B5 per slot: 2903 / 2614 / 2421 / 2710 /
  2421 MB, so the 1024 slot is the peak. Its run held 3386 steady (1.17 ×
  the maximum). The pool is not cumulative across the slots (`--max-vram`
  cleans it). Probe noise: ±150 MB.

## 8. The M family (new protocol, not run yet)

N 2, stage 2+2 Real1D, main Real1D r 128 unless noted. `md` fitted to 1M
with `mmlp 8` (`fit.py`), then `mmlp` filled.

| cfg | stage d / r | main | params |
|---|---|---|---|
| M-s96r64-L4 | 96 / 64 | d136 × 4 | 992,533 |
| M-s80r64-L4 | 80 / 64 | d144 × 4 | 996,753 |
| M-s64r64-L4 | 64 / 64 | d152 × 4 | 992,717 |
| M-s96r64-L5 | 96 / 64 | d120 × 5 | 999,624 |
| M-s96r64-L4r256 | 96 / 64 | d120 × 4, r 256 | 994,829 |
| M-s96r64-L4c | 96 / 64 | d128 × 4, Complex2D | 997,553 |
| M-s96r32-L4 | 96 / 32 | d136 × 4 | 990,485 |
| M-s112r64-L4 | 112 / 64 | d120 × 4 | 995,253 |

Also: Nm4-s96r128-B5 (stage d96 r 128, main d128 × 4 r 128, 996,273) and
W-m64r-B5 (stage d112 r 128, main d144 × 3 r 64, 997,543).

## 9. Kaggle

- Two T4s, the same reserved MB as here (the b probe: 1962 MB on both).
  The bucketed b family ran at 1.14 steps/s there.
- Kaggle's unpacking of a `.tar.gz` stalls for hours. The payload goes up
  as `payload.tgz.bin` in a new private dataset per batch (live in ~20 s).
- The CLI sends the `KGAT_` token as a Bearer token (`kg.sh`). The CLI
  cannot cancel a kernel: the user does it on the web.

## 10. Lessons

- **A new config field** (user): add it to every saved config that loads
  again, never make it an `Option` for compatibility. A Burn config default
  does not apply when a config loads. Stop the runners first: the next
  phase of a running run loads its config with the new binary.
- A run that starts in the second of a kill can fail in cubecl (the GPU
  memory is not free yet): wait ~20 s.
- The bucket summary must count per slot (a probe order breaks the runs of
  equal slots).
