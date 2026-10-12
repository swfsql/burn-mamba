# Notes: hnet-stories ≤1M-param search (1m-a)

The findings, by topic. §5–§6 come from the **old protocol** (the first
1024 chars of each story, before 2026-10-09 20:40): invalid numbers, kept
only as the starting points of the new protocol.

Terms:

- **wv**: the whole valid (256 stories, each in full), the ranking metric.
  The dv of a bucketed run is wv (binary `4e42bba3` on). The flat's in-loop
  valid scores the first 1024 chars only (wv ≈ that + 0.007).
- **sv**: the stable valid of a trunk. A **leaf** (`-d`): its decay branch.
- **Both ceilings** (strict rule 1): params ≥ ~990k and < 1M, a dosed peak
  of ~3400–3550 MB.
- **The long stories** (`long.sh`): the 157 valid stories > 2048 chars,
  each in one `forward`. **The recall** (`ctx.sh -g -n 2048`, 1968 stories,
  paired): far own / none / gain, and the re-mention letters (s.e. ~0.001
  on the far bits, ~0.005 on the letters). The valid bits hide the recall.
- **The seed noise**: 0.008 in wv between two seeds (R-N15-B5), 0.0085 for
  the flat h16-V7u. One run ranks only a gap ≳ 0.01.

## 1. The result: stage reuse at r 128 leads the flat at 3230

| milestone | best H-Nets (wv) | h16-V7u (flat) |
|---|---|---|
| 3230 | **R-N125-r128s11V2-m5 0.8533** (one seed); R-N15-B5 0.8688 / 0.8605, R-N125-B5 0.8638 / 0.8624, R-N15-r64V7 0.8644 / 0.8675 | 0.8644 / 0.8559 (mean **0.8602**) |
| 5630, the protocol | R-N15-r64V7 0.8088 / 0.8087, R-N15-B5 0.8143 | **0.8037** |
| 5630, step-down at 2630 | 0.7962 / 0.8020 / 0.8046 | **0.7968** |
| 9600 | R-N15-r64V7 0.7501 / 0.7490, R-N15-B5 0.7572 | **0.7487** |
| 14400 | R-N15-B5 0.7308, R-B5 0.7320 | **0.7220** |

- **h16-V7u** (the reference): d160 × 5 C2D r 64, 16 heads, × 7 Cyclic
  untied applications, Multi-Gate 4, gc: 995,921 params, 3455 MB. It gains
  0.022 on the old reference h16 (969k, 3149 MB: under both ceilings, so
  every earlier H-Net lead of 0.005–0.026 was that gap).
- **The R-B5 family trails the flat**: 0.003–0.006 at 3230 (two-seed
  means), 0.0085 at 9600, 0.0086 ± 0.0016 at 14400 (R-N15-B5). It ties at
  5630 only when each model has its best schedule (§2).
- **R-N125-r128s11V2-m5** (§4) at 3230: −0.0069 against the flat mean.
  Paired against flat seed 0 (flat seed 1 in brackets): wv −0.0102 ±
  0.0019 (−0.0078), the long stories −0.0108 (−0.0079), far own −0.0171
  (−0.0115), re-mention letters −0.126 (−0.062). It beats both flat seeds
  on every row. One seed.
- **R-N15-r64V7** (main reuse) ties the flat at 9600: paired wv +0.0006 /
  +0.0002 ± 0.0016, the long stories +0.0014 / −0.0002.
- **The flat's virtual schedule** (two seeds each, wv at 3230): Cyclic
  (0,1,2,3,4,0,1) **0.8602**, `m2` (0,1,2,2,2,3,4) 0.8630 (+0.0028),
  `m13` (0,1,1,2,3,3,4) 0.8699 (+0.0097, worse in both seeds). So the
  middle placement does not help the flat. `m2` gains ~0.017 on the
  re-mention letters per seed pair, but the letters are ~12% of the far
  window, and there the gain and the loss on the other chars cancel.

## 2. The LR schedule: the H-Net steps down at about half of its budget

- The protocol (the flat's plan): 4e-3 to 4630, then 1e-3, then a cosine
  1e-3 → 5e-4 over 8000–12000. Leaves of 600 / 1000 / 1600 / 2400 give
  3230 / 5630 / 9600 / 14400.
- Δwv of the step-down position against the protocol run of the same
  trunk:

  | step-down at | 5630: H-Net | 5630: flat | 9600: H-Net |
  |---|---|---|---|
  | 2630 | −0.012 / −0.016 | h16-V7u −0.007 | +0.010 |
  | 4630 (the protocol) | 0 | 0 | 0 |
  | 6315 | | | +0.001 / +0.002 |
  | none (4e-3 to 8000) | | | +0.019 (R-N15-r64V7, two seeds) |

- So the H-Net wants the step-down at 47–66% of the budget, and 27% or 82%
  costs ~0.01. Compare two models on the same schedule.
- **The trunk schedule is in the config** (as in the flat): `gen.py`
  writes the protocol, `extend.sh` follows it with no `--lr`, and
  `lrprof.py` guards it (§9).
- The leaf gain (leaf − trunk): the H-Net −0.122 / −0.135 (3230 / 5630),
  h16 −0.113 / −0.115. Trunks at 1e-3 from 2630 stall.

## 3. The results of the new protocol (wv at 3230)

| run | shape (change from its base) | wv | peak MB |
|---|---|---|---|
| L-R-N125-r128s11V2-m5-d | stage d128 1+1 R1D r 128 × 2+2 untied applications, N 1.25, main d128 × 5 C2D r 64 rf 0.5 | **0.8533** | 3482 |
| L-R-N125-B5-d / -s1-d | R-N15-B5 × N 1.25 | 0.8638 / 0.8624 | 3386 |
| L-R-N15-r64V7-d / -s1-d | R-N15-B5 × stage r 64 × main V 7 (0,1,1,1,1,1,2, untied) | 0.8644 / 0.8675 | 3452 (max 3592) |
| L-R-N15-r64sV5-d | R-N15-B5 × stage r 64 × 5+5 untied applications | 0.8670 | 3522 |
| L-R-N15-B5-d / -s1-d | stage d128 2+2 R1D r 104, main d128 × 3 C2D r 64, N 1.5 | 0.8688 / 0.8605 | 3386 |
| L-R-N15-B-d | R-N15-B5 × 7 slots | 0.8697 | 2799 |
| L-R-B5-d | R-N15-B5 × N 2 | 0.8723 | ~3390 |
| L-R-N15-r128s11V2-m5V6c-st2-d | R-N125-r128s11V2-m5 × N 1.5 × main V 6 Cyclic × 2 start rows | 0.8733 | 3482 |
| L-R-B5-s144b-d | R-B5 × stage d144 × main d144 × 3 expand 1 | 0.8838 | 3390 |
| L-S2a2-B5-d | two stages d112/120, N 1.41/1.41, main d120 × 3 | 0.8965 | 3532 |
| L-M-s96r128-L5-d | stage d96, main d112 × 5 | 0.9022 | 2440 * |
| L-M-s64r256-L4-d | stage d64 r 256, main d144 × 4 | 0.9723 | 2711 * |
| L-S2x-e21h16-L4-lrm-d | two stages d112, N 4/4, main d112 × 4 | 1.0048 | 3663 |

\* Under the band: weak evidence.

## 4. The shapes (new protocol)

- **N**: N 1.25 = N 1.5 (two-seed means 0.8631 / 0.8647). N 2 is +0.004
  to +0.008 up to 9600 and ties at 14400. N costs no params. It moves the
  peak little where the stage sets it (R-N15-B5), but +290 MB at the r 128
  shape (§7): there a lower N spends a page.
- **The stage width is the lever, up to d128**: d64 +0.10, d96 +0.03 to
  +0.05. A d144 stage paid by the main loses 0.0115, mostly in the recall.
- **The main carries the recall**: a weaker main and deep compression
  lose it.
- **Stage reuse** (virtual layers in a stage, `eV`/`dV`, untied; no
  library change):
  - At r 64 it does not pay (5+5: 0.8670).
  - At r 128 it pays: 1+1 real × 2+2 applications free ~260k params for a
    main × 5 (R-N125-r128s11V2-m5: 0.8533, §1).
  - At r 104 / r 100 an extra stage application is over the band (§7).
- **Main reuse**: R-N15-r64V7 (stage r 64, main × 7 untied, 0,1,1,1,1,1,2)
  ties R-N15-B5 at 3230, then leads it (−0.0055 at 5630, −0.0076 at 9600)
  and ties the flat at 9600. Its schedule against Cyclic is untested
  (R-N15-r64V7c, dosed).
- **Stage + main reuse**: one untied main application of the r 128 shape
  costs 8.6k params (225 over the cap) and ~290 MB at N 1.25 (over).
  R-N15-r128s11V2-m5V6c-st2 (N 1.5, main V 6 Cyclic, `start=2` as the
  trim) loses 0.019 against the leader: wv +0.0190 ± 0.0016, the long
  stories +0.022, far own +0.021, and the same with no prompt (the recall
  gain is equal). So it models the text worse. Three changes: not
  separable yet (the start rows untested: R-N125-r128s11V2-m5-st2, dosed).
- **The residuals**: every H-Net stack is `Standard` (`mgr=0`, `mmgr=0`),
  the flat is Multi-Gate 4. In the flat (`1m-b`) the standard residual
  lost 0.046 (Multi-Gate 3 / 6: +0.011 / +0.017). The H-Net's one test
  was old protocol (tied). Untested now: `-mg4` / `-mmg4` are dosed.
- **Closed**: two stages at total N 2 (+0.024 at 3230, +0.020 at 5630);
  deep compression N 4/4 (+0.13 / +0.108, not the LR); the d64 / S4 line
  (no design near both ceilings; a per-layer stage rank would need a
  library change).
- **The param costs**: a stage boundary ~50k at d128; one outer layer at
  d112 r 256 146k. The widths must not shrink inward. `lrm=auto` stays.

## 5. The recipe (old protocol)

- The untied head (`tie=0`) +0.036, the Burn init +0.032, λ auto +0.019.
- The LR **4e-3** (8e-3 +0.016, 2.5e-3 +0.005).
- The main MLP (`mmlp`) costs no VRAM and is neutral: a filler for a small
  rest only (strict rule 14).
- Whole stories fix the tail past char 1024 (−0.043 / −0.062). 5 length
  slots = 7.

## 6. The block, N and the main (old protocol)

- **Real1D stages at a large state rank**: the rank pays, not the kind.
- **The trapezoid** (and the B/C bias) replace the deleted short conv:
  Trapezoid None lost 0.042 and 0.064.
- **Lost**: main virtual layers (Cyclic +0.020 on b × N 3, middle-heavy
  +0.010 on F with a main r 96 too: two bases, so not a schedule
  comparison), main × 2 (+0.027), stage 1+1 with a heavier main
  (+0.025–0.06), a main at r 32 (+0.027).
- **Tied**: Multi-Gate 4, a Real1D main, weight decay 0.05, α 0.1.

## 7. VRAM

- **The ceiling** is on the process (strict rules 1, 2). cubecl (48301b5):
  a dropped page waits for an explicit cleanup; `--vram-cleanup-every 1`
  flushes before every window. The process ≈ peak + ~140 MiB. A page is
  ~168 MB: a peak moves in these steps.
- **What moves the peak**: the stages (the 1024 slot), the heads × the
  state rank. A head dim of 8 doubles the heads (over the band). A main at
  expand 1 and the main MLP are cheap.
- **Use a power-of-2 state rank** (user): another rank jumps in the pages
  and can cost more than the next power of 2. Stage r 128 against r 104 /
  r 96 (probes): R-N15-B5 2617 against 2906, `eV3` 3099 against 3678,
  `s11V2-m5` 2906 against 3485. The flat: r 96 3995, r 128 3776.
- **The cost of one more step of a knob** (MB): a stage application at
  r 64 ~170 (probe), at r 128 ~480 (probe); a main application at N 1.25
  ~290 (dose); N 1.5 → 1.25 at the r 128 shape +290 (dose); Multi-Gate 4
  (stage or main) 0 (dose 3482 = the leader); 2 start rows 0.
- **The probe → run factor**: × 1.0 (the flat, S4), × 1.06–1.08 (S2 at
  N 4/4), × 1.06–1.34 (S = 1). Dose every shape; `gate.sh` screens a shape
  only after a dose in the band.
- **The flat**: gc saves ~690 MB; each untied virtual application adds
  ~460 MB under gc.

## 8. Kaggle

- Two T4s, the same pool accounting. A T4: ~1.17 s/step for the R shapes,
  ~1.46 for R-N15-r64V7 (this card ~1.05 / 1.37 for the r 128 leader). The
  pieces of a chain cost no time between them.
- The payload goes up as `payload.tgz.bin` in a new private dataset per
  batch (a `.tar.gz` stalls Kaggle's unpacking for hours).
- `kpush.sh` runs the LR check (`lrprof.py --batch`) before a push.
- `kwait.sh` queues `long.sh` for the pulled `-d` runs here, but it misses
  some `== <run>-d:` headers (5 so far): check each pulled `-d` run.
- Only `probes.txt`, `knobs.txt`, `ctxg.txt` and `speeds.txt` come back as
  text. A Kaggle dose: read the peaks in the pulled `runs/dose-*`.
- The CLI cannot cancel a kernel (the user does it on the web). The flat
  cannot run on Kaggle.

## 9. Lessons

- **A reference must be at both ceilings too**: h16 gave the H-Net a false
  lead of 0.015–0.026.
- **Compare two models on the same schedule**, and check it before a
  verdict (`lrprof.py <run>`). The H-Net configs held only the ramp, an
  R-N15-r64V7 batch omitted `--lr`, and nothing checked: 4e-3 to 8000,
  +0.019 at 9600, a wrong "closed". The fix: the protocol in the config,
  `lrprof.py --fill` for 50 trunks, `--guard` (`extend.sh`) and `--batch`
  (`kpush.sh`).
- **One change per screen**: the stage + main reuse screen changed three
  things, so its loss cannot be placed.
- **A regeneration of `cfg/`** can change old configs through a newer
  default (156 got `bucket`): compare against a backup, keep the old files.
- **The queue**: read `queue.txt` again before a Write (`queue.sh` changes
  it); never start `queue.sh` in the same parallel call as the Write (it
  starts empty and exits); never edit a script while a queue line runs it.
- **A new config field** (strict rule 9): never an `Option`.
- **Waiters**: never on `pgrep -f` (strict rule 4). A run that starts in
  the second of a kill can fail in cubecl: wait ~20 s.
- `build.sh`: one command per line and a `cmp` (`cp … && mv …` does not
  stop `set -e`). The disk is shared (627 GB): `target/debug` reached
  115 GB. The Edit tool drops the trailing space of an edit string.
