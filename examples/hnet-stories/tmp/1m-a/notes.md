# Notes: hnet-stories ≤1M-param search (1m-a)

The findings of the search, by topic. §5–§6 come from the **old protocol**
(the first 1024 chars of each story, before 2026-10-09 20:40). Those runs
never trained past char 1024, so their numbers are invalid. Their findings
were the starting points of the new protocol.

Terms:

- **wv**: the whole valid (the 256 valid stories, each in full). The
  ranking metric. The dv of a bucketed run is wv (binary `4e42bba3` on).
  The in-loop valid of the flat (and of the H-Net before `4e42bba3`)
  scores only the first 1024 chars of each story: wv ≈ that + 0.007.
- **sv**: the stable valid of a trunk. A **leaf** (`-d`): the decay branch
  of a trunk.
- A **long run**: a trunk of 2630 steps at lr 4e-3 + a leaf of 600
  (`screen.sh`). The later milestones: §2.
- **Both ceilings** (strict rule 1): params ≥ ~990k and < 1M, a dosed peak
  of ~3400–3550 MB.
- A variant has the name of its change from its base. `catalog.txt` has
  the shape of each named model.
- **The long stories** (`long.sh`): the 157 valid stories longer than 2048
  chars (2056..4064, mean 2463), each in one `forward`. The right padding
  is exact (batch 1 = batch 4). **The recall** (`ctx.sh -g -n 2048`, 1968
  stories, paired): the s.e. of a difference is ~0.001 on the far bits,
  ~0.01 on the re-mention first letter. The valid bits hide the recall.
- **The seed noise**: R-N15-B5 at seeds 0 and 1 differ by 0.008 in wv (the
  same shift in every row and in the recall). One run per shape ranks only
  a gap ≳ 0.01.

## 1. The result: a flat at both ceilings ties the H-Net, then leads

| milestone | best H-Net (wv) | h16-V7u (flat, both ceilings) | h16 (old flat) |
|---|---|---|---|
| 3230 | R-N15-B5 0.8688 / 0.8605, R-N125-B5 0.8638 / 0.8624, R-N15-r64V7 0.8644 / 0.8675 | **0.8644** | 0.8868 |
| 5630, the protocol | R-N15-B5 0.8143, R-B5 0.8204 | **0.8037** | 0.8204 |
| 5630, step-down at 2630 | 0.7962 / 0.8020 / 0.8046 | **0.7968** | 0.8197 |
| 9600 | R-N15-B5 0.7572, R-B5 0.7608 | **0.7487** | 0.7660 |
| 14400 | R-N15-B5 0.7308, R-B5 0.7320 | not run | 0.7412 |

- **h16** (the flat reference until 2026-10-11) is under both ceilings:
  969,481 params, no gc, a 3149 MB peak. Every H-Net lead over it (3230:
  0.015–0.026, 9600: 0.005–0.009, 14400: 0.009–0.010) was this gap.
- **h16-V7u** = h16 (d160 × 5, C2D r 64, 16 heads) × 7 virtual
  applications of its 5 layers (Cyclic, untied: 13.2k params per
  application), under gc: 995,921 params, a 3455 MB peak, 1.03 steps/s
  here. It gains 0.022 on h16 at 3230. Other flats at 3230: fr-d152 (the
  H-Net's Real1D r 128 block) 0.8792, h16-B (7 slots) 0.8924.
- Against h16-V7u, the H-Net ties at 3230 (6 runs at N 1.25–1.5, mean
  0.8646). It ties at 5630 when each model has its best schedule (§2).
  **At 9600 the flat leads by 0.0085.**
- The long stories, paired against h16-V7u: 3230 R-N15-B5 +0.005 /
  −0.004, R-N125-B5 −0.002 / −0.007, R-N15-r64V7 +0.002 / +0.003, R-B5
  +0.012. 5630 (the protocol): R-N15-B5 +0.011. 9600: R-N15-B5 +0.009 ±
  0.0015.

## 2. The LR schedule: the H-Net steps down at about half of its budget

- The protocol (the flat's plan): 4e-3 to 4630, then 1e-3, then a cosine
  1e-3 → 5e-4 over 8000–12000. The leaves of 600 / 1000 / 1600 / 2400 give
  3230 / 5630 / 9600 / 14400.
- The step-down to 1e-3 (`extend.sh --lr`), Δwv at the next leaf against
  the protocol run of the same trunk:

  | step-down at | 5630: H-Net | 5630: flat | 9600: H-Net |
  |---|---|---|---|
  | 2630 | −0.012 (R-N15-B5), −0.016 (R-B5), seed 1 at 0.7962 | h16 −0.0007, h16-V7u −0.007 | +0.010 |
  | 4630 (the protocol) | 0 | 0 | 0 |
  | 6315 | | | +0.001 / +0.002 |

- So the H-Net wants the step-down at 47–66% of the budget. At 27% or 82%
  it loses ~0.01. The protocol's 5630 milestone (82% at 4e-3) is bad for
  the H-Net. The h16-V7u flat gains from the early step-down too.
- The trunks at 1e-3 from 2630 stall (0.8425 / 0.8284 / 0.8216 at 4630 /
  6315 / 8000). At 4e-3 the R-B5 trunk progresses slower than the flat's
  (2630 → 4630: −0.039 against −0.064). The R-N15-B5 trunk does not (4630:
  0.9331, h16 0.9364).
- The leaf gain (leaf − trunk) of the H-Net: −0.122 (3230), −0.135
  (5630). The leaf gain of h16: −0.113, −0.115.

## 3. The results of the new protocol (wv at 3230)

| run | shape (change from its base) | wv | peak MB |
|---|---|---|---|
| L-R-N15-B5-d / -s1-d | stage d128 2+2 R1D r 104, main d128 × 3 C2D r 64, N 1.5 | 0.8688 / 0.8605 | 3386 |
| L-R-N125-B5-d / -s1-d | R-N15-B5 × N 1.25 | 0.8638 / 0.8624 | 3386 |
| L-R-N15-r64V7-d / -s1-d | R-N15-B5 × stage r 64 × main V 7 (0,1,1,1,1,1,2, untied) | 0.8644 / 0.8675 | 3452 (max 3592) |
| L-R-B5-d | R-N15-B5 × N 2 | 0.8723 | ~3390 |
| L-R-N15-B-d | R-N15-B5 × 7 slots | 0.8697 | 2799 |
| L-R-B5-s144b-d | R-B5 × stage d144 (18 heads) × main d144 × 3 expand 1 r 44 | 0.8838 | 3390 |
| L-S2a2-B5-d | two stages d112/120 1+1/1+1 r 256/128, N 1.41/1.41, main d120 × 3 | 0.8965 | 3532 |
| L-M-s96r128-L5-d | stage d96, main d112 × 5 | 0.9022 | 2440 * |
| L-Nm4-s96r128-B5-d | stage d96, main d128 × 4 | 0.9094 | 2505 * |
| L-M-s96r128-L4r256-d | stage d96, main d112 × 4 r 256 | 0.9240 | 2887 * |
| L-M-s64r256-L4-d | stage d64 r 256, main d144 × 4 | 0.9723 | 2711 * |
| L-S2x-e21h16-L4-lrm-d | two stages d112 2+1/1+1 (outer 16 heads), N 4/4, main d112 × 4 expand 1, λ 1.41/1.19/1 | 1.0048 | 3663 |
| L-S2x-e21h16-L4-d | the same at λ auto (4/2/1) | 1.0099 | 3663 |

\* Under the band (the old cleanup schedule): weak evidence.

## 4. The shapes (new protocol)

- **N**: N 1.25 = N 1.5 (two-seed means 0.8631 / 0.8647). N 2 is ≈ +0.008
  at 3230, +0.004–0.006 at 5630 / 9600 (one seed each), and ties at 14400
  (0.7320 / 0.7308). N does not move the peak.
- **The stage width is the lever, up to d128**: d64 +0.10, d96 +0.03 to
  +0.05 on every row (the M family: a stronger main, a thinner stage). A
  d144 stage paid by the main (R-B5-s144b, the main at expand 1) loses
  0.0115, mostly in the recall (re-mention letters gain −0.060 ± 0.006,
  far use −0.009). The old plateau (d144 = d128 with a main × 2) agrees.
- **The main carries the recall**: a weaker main (R-B5-s144b) and deep
  compression lose it.
- **Stage rank against main reuse**: R-N15-r64V7 (stage r 64, the freed
  VRAM and params on 7 main applications) ties R-N15-B5 at 3230.
- **Two stages at total N 2** (S2a2-B5): +0.024 at 3230, +0.020 at 5630
  (the long stories +0.020). Its recall equals R-B5's at 5630.
- **Deep compression** (S2 at N 4/4, d112): +0.13 at 3230, +0.108 at 5630
  (against R-B5). A λ cap (4/2 → 1.41/1.19) gains only 0.005, so the LR is
  not the cause. The recall deficit closes faster than the bits (far use
  −0.064 → −0.042, re-mention letters own +0.405 → +0.239) but stays. The
  user's hunch (the gain shows later) is judged at 3230 and 5630:
  **closed**.
- **The d64 / S4 line is boxed in**: S4 fits 1M only at d64 (d80:
  1.11–1.23M). An outer d64 stage at r 256 is under the band at 2+2 and
  3+3 (S2 3201, S4 ~3008 MB). It is over the band at r 512 (2+1 already
  3850) and at 16 heads (p 8: > 4000). A per-layer rank in a stage would
  give a design (a library change). **Parked.**
- **The param costs**: a stage boundary ~50k at d128. One outer layer at
  d112 r 256: 146k. S2 at d128/128 + a main d128 × 3–4: 1.13–1.25M, so d112
  is the widest S2 with a main of ≥ 4 layers. The widths must not shrink
  inward (a narrower inner stage does not build).
- λ auto puts an outer stage of N 4/4 at lr × 4 (d112) to × 5.7 (d64).
- The M family loses at the stage LR too: λ 2.12 (auto) and λ 1.41 tie by
  step 656 (D-M-s64r256-L4-lrm). So `lrm=auto` stays.

## 5. The recipe (old protocol)

- The untied head (`tie=0`) won by 0.036. The Burn init beat the reference
  init by 0.032. λ auto beat 1/1 by 0.019.
- The LR of the long runs: **4e-3** (8e-3 0.9070, 4e-3 0.8915, 2.5e-3
  0.8960).
- The main MLP at `mlpmul=8` (`mmlp`) costs no VRAM and is neutral on a
  screen: a filler for a small rest only (strict rule 14).
- Whole stories fix the tail: the H-Net on the first 1024 chars got worse
  past char 1024 (0.866 → ~1.0 per 256 chars). Whole stories gained
  −0.043 (1024..2048) and −0.062 (past 2048). The 5 length slots tie the 7
  slots.

## 6. The block, N and the main (old protocol)

- **Real1D stages at a large state rank**: the rank pays, not the kind
  (F-r1d-sr104 1.1076 against F-base 1.1294).
- **The trapezoid** (and the B/C bias) replace the deleted short conv:
  Trapezoid None lost 0.042 and 0.064.
- **Lost**: main virtual layers (Cyclic +0.020, middle-heavy +0.010), main
  × 2 (+0.027), stage 1+1 with a heavier main (+0.025–0.06), a main at r 32
  (+0.027), stage 3+3 at d112 (+0.007), stage 4+4 at d96 (+0.08,
  unstable).
- **Tied**: Multi-Gate 4, a Real1D main, weight decay 0.05, α 0.1.

## 7. VRAM

- **The ceiling** is on the VRAM of the process (strict rule 1). Never
  decide by `nvidia-smi` (strict rule 2).
- **How cubecl spends it** (48301b5, the `Adaptive` layout, the `Sync`
  allocator): a pool drops a page on its own, and the storage only queues
  it (`GpuStorage.deallocations`). `cuMemFree` runs at `storage.flush()`,
  and only an explicit cleanup (`Device::memory_cleanup`) flushes.
  `--vram-cleanup-every 1` cleans up before every window: the same pool
  peak, ~10% faster than the old schedule. The process ≈ peak + ~140 MiB.
- **The pages are ~168 MB**: a peak moves in these steps.
- **What moves the peak**: the stages set it (the 1024 slot). The
  [s, h, r] tensors dominate: the heads × the state rank. A head dim of 8
  doubles the heads and the stage VRAM (d64, d112: over the band). At d144,
  18 heads and 16 heads differ by < 1 page. A state rank costs the VRAM of
  the next power of two: at d144, r 64 → r 104 is +60%. N moves the peak
  little. A main at expand 1 is cheap (R-B5-s144: −200 MB against R-B5).
- **The probe → run factor** is × 1.0 (S4, the flat), × 1.06–1.08 (S2 at
  N 4/4), × 1.06–1.34 (S = 1: R-B5-s144 × 1.06, R-N15-B5 × 1.17,
  R-N15-r64V7 × 1.34). Dose every shape.
- **The flat**: gc saves ~690 MB at h16 (3149 → 2460). Each untied virtual
  application adds ~460 MB under gc (V6u 2996, V7u 3455).
- The knob table of the old protocol (Complex2D base, ±150 MB) is
  `knobs.py`. VRAM-heavy and param-cheap (~16–18 MB per kparam): the
  stage state rank, the stage MIMO 2 (but 3.3× the ms/step). Param-heavy
  and VRAM-free: the main MLP, the main expand.

## 8. Kaggle

- Two T4s, the same pool accounting as here. A T4 runs the R shapes at
  ~1.17 s/step (this card ~1.05), R-N15-r64V7 at ~1.42.
- Kaggle's unpacking of a `.tar.gz` stalls for hours. The payload goes up
  as `payload.tgz.bin` in a new private dataset per batch.
- The CLI sends the `KGAT_` token as a Bearer token (`kg.sh`). The CLI
  cannot cancel a kernel: the user does it on the web. An environment
  reset can remove the CLI: the user installs it again.
- `kwait.sh` queues `long.sh` for the pulled runs in the local queue (a
  GPU job beside a probe spoils it). It missed the `== <run>-d:` header of
  one pulled `-d` run twice (L-R-B5-x12000-d, L-R-B5-sd2630-d), so it
  queued no `long.sh` for it. Check each pulled `-d` run.
- Only `probes.txt`, `knobs.txt`, `ctxg.txt` and `speeds.txt` come back
  as text. A dose on Kaggle: read the peaks in the pulled `runs/dose-*`.
- The flat cannot run on Kaggle: the runner and the payload know only
  this workspace.

## 9. Lessons

- **A reference must be at both ceilings too**: h16 was 3% under on the
  params and ~300 MB under on the VRAM. That gave the H-Net a false lead of
  0.015–0.026.
- **Compare two models on the same schedule**: the step-down position
  moves a result by ~0.01, and differently for the H-Net and the flat.
- **A new config field** (strict rule 9): never an `Option`. A Burn config
  default does not apply when a config loads.
- A run that starts in the second of a kill can fail in cubecl: wait ~20 s.
- **A waiter on `pgrep -f`** matches itself and never ends (strict rule 4).
- **`build.sh`**: `cp … && mv …` does not stop `set -e`. Fixed: one
  command per line and a `cmp`.
- **The disk** (627 GB, shared with every project) filled up on
  2026-10-10. `cargo`'s `target/debug` was 115 GB.
- **The Edit tool** drops the trailing space of an edit string.
- **Never edit a script while a queue line runs it**: bash reads a script
  as it runs.
- `queue.sh` changes `queue.txt`: read it again before a Write.
