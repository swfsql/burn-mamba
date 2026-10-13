# Notes: hnet-stories ≤1M-param search (1m-a)

The findings, by topic. §5–§6 come from the **old protocol** (the first
1024 chars of each story, before 2026-10-09 20:40): invalid numbers, kept
only as the starting points of the new protocol.

Terms:

- **wv**: the whole valid (256 stories, each in full), the ranking metric.
  The dv of a bucketed run is wv (binary `4e42bba3` on). The flat's in-loop
  valid scores the first 1024 chars only: its wv comes from `long.sh -t`.
- **sv**: the stable valid of a trunk. A **leaf** (`-d`): its decay branch.
- **Both ceilings** (strict rule 1): params ≥ ~990k and < 1M, a dosed peak
  of ~3400–3550 MB.
- **The long stories** (`long.sh`): the 157 valid stories > 2048 chars,
  each in one `forward`. **The recall** (`ctx.sh -g -n 2048`, 1968 stories,
  paired): far own / none / gain, and the re-mention letters (s.e. ~0.001
  on the far bits, ~0.005 on the letters). The valid bits hide the recall.
- **The seed noise** (wv, two seeds): R-N15-B5 0.008; the flat h16-V7u
  0.0085 / 0.0031 (3230 / 5630); R-N125-r128s11V2-m5 0.022 / 0.013 /
  0.0085 (3230 / 5630 / 9600), but its seed 1 is the outlier of its family
  (the other seeds at 3230: 0.8533–0.8676); `-mg4` 0.007 / 0.006 / 0.003 /
  0.001 (up to 14400). **Rank by two-seed means** (§9).

## 1. The result: R-N125-r128s11V2-m5-mg4 leads the flat

| milestone | `-mg4`: s0 / s1 (mean) | the flat h16-V7u: s0 / s1 (mean) | Δ means |
|---|---|---|---|
| 3230 | 0.8625 / 0.8552 (0.8589) | 0.8644 / 0.8559 (0.8602) | −0.0013 |
| 5630 | 0.8040 / 0.7978 (0.8009) | 0.8037 / 0.8006 (0.8022) | −0.0013 |
| 9600 | 0.7455 / 0.7426 (0.7441) | 0.7487 / – | −0.0047 \* |
| 14400 | **0.7174 / 0.7164 (0.7169)** | 0.7220 / – | −0.0051 \* |

\* Against the flat's seed 0 only.

- **R-N125-r128s11V2-m5-mg4** (the leader): stage d128 1+1 real R1D r 128,
  16 heads, × 2+2 untied applications, N 1.25, main d128 × 5 C2D r 64 rf
  0.5, Multi-Gate 4 in every stack. 993,485 params, 3482 MB, ~1340
  ms/step here.
- Paired against the flat (seed 0) on the long stories at 14400: −0.0037
  / −0.0040 ± 0.0016 on all chars, **−0.022 / −0.021** (± ~0.012) past
  char 1024. So the lead is largest on the long tails.
- **Open: the flat's seed 1 past 5630.** It was better than seed 0 by
  0.0085 / 0.0031 at 3230 / 5630. Against the flat's two-seed mean, the
  lead is −0.0013 at 3230 and at 5630. So the lead at 9600 / 14400 can
  shrink when seed 1 gets there (`h16-V7u-s1-x4630` is its trunk).
- Its base (no Multi-Gate): 0.8645 / 0.8062 / 0.7491 (two-seed means at
  3230 / 5630 / 9600). The other tie of the flat at 9600: R-N15-r64V7
  (main reuse) 0.7496. R-N15-B5: +0.0086 against the flat at 14400.
- **The flat h16-V7u**: d160 × 5 C2D r 64, 16 heads, × 7 Cyclic untied
  applications, Multi-Gate 4, gc: 995,921 params, 3455 MB. It gains 0.022
  on the old reference h16 (969k, 3149 MB: under both ceilings, so every
  earlier H-Net lead of 0.005–0.026 was that gap). Its virtual schedule
  (two seeds each, 3230): Cyclic 0.8602, `m2` +0.0028, `m13` +0.0097.

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
| L-R-N125-r128s11V2-m5-mg4-d / -s1-d | the leader (§1) | 0.8625 / 0.8552 | 3482 |
| L-R-N125-r128s11V2-m5-mmg4-d | R-N125-r128s11V2-m5 × Multi-Gate 4 in the main only | 0.8586 | 3482 |
| L-R-N125-r128s11V2-m5-st2-d / -s1-d | R-N125-r128s11V2-m5 × 2 start rows | 0.8566 / 0.8657 | 3482 |
| L-R-N125-r128s11V2-m5-d / -s1-d | stage d128 1+1 R1D r 128 × 2+2 untied applications, N 1.25, main d128 × 5 C2D r 64 rf 0.5 | 0.8533 / 0.8756 | 3482 |
| L-R-N125-B5-d / -s1-d | R-N15-B5 × N 1.25 | 0.8638 / 0.8624 | 3386 |
| L-R-N15-r64V7-d / -s1-d | R-N15-B5 × stage r 64 × main V 7 (0,1,1,1,1,1,2, untied) | 0.8644 / 0.8675 | 3452 (max 3592) |
| L-R-N15-r64V7c-d | R-N15-r64V7 × a Cyclic main | 0.8636 | 3592 |
| L-R-N15-r128s11V3-m5V6c-st2-sp32-mg4-d | 8 stage heads × 3+3 × main V 6 Cyclic × 2 start rows × Multi-Gate 4, N 1.5 | 0.8660 (5630: 0.8069) | 3442 |
| L-R-N15-r64sV5-d | R-N15-B5 × stage r 64 × 5+5 untied applications | 0.8670 | 3522 |
| L-R-N15-B5-d / -s1-d | stage d128 2+2 R1D r 104, main d128 × 3 C2D r 64, N 1.5 | 0.8688 / 0.8605 | 3386 |
| L-R-N15-B-d | R-N15-B5 × 7 slots | 0.8697 | 2799 |
| L-R-N15-r128s11V2-m5V6c-st2-d / -s1-d | R-N125-r128s11V2-m5 × N 1.5 × main V 6 Cyclic × 2 start rows | 0.8733 / 0.8676 | 3482 |
| L-R-B5-d | R-N15-B5 × N 2 | 0.8723 | ~3390 |
| L-R-B5-s144b-d | R-B5 × stage d144 × main d144 × 3 expand 1 | 0.8838 | 3390 |
| L-S2a2-B5-d | two stages d112/120, N 1.41/1.41, main d120 × 3 | 0.8965 | 3532 |
| L-M-s96r128-L5-d | stage d96, main d112 × 5 | 0.9022 | 2440 * |
| L-M-s64r256-L4-d | stage d64 r 256, main d144 × 4 | 0.9723 | 2711 * |
| L-S2x-e21h16-L4-lrm-d | two stages d112, N 4/4, main d112 × 4 | 1.0048 | 3663 |

\* Under the band: weak evidence.

## 4. The shapes (new protocol)

- **N**: N 1.25 = N 1.5 (two-seed means 0.8631 / 0.8647). N 2 is +0.004
  to +0.008 up to 9600 and ties at 14400. N costs no params. It moves the
  peak little where the stage sets it (R-N15-B5), but ~240–290 MB at the
  r 128 shapes (§7).
- **The stage width is the lever, up to d128**: d64 +0.10, d96 +0.03 to
  +0.05. A d144 stage paid by the main loses 0.0115, mostly in the recall.
- **The main carries the recall**: a weaker main and deep compression
  lose it.
- **Stage reuse** (virtual layers in a stage, `eV`/`dV`, untied): at r 64
  it does not pay (5+5: 0.8670). At r 128 it pays: 1+1 real × 2+2
  applications free ~260k params for a main × 5 (the leader's base). A
  real stage layer is expensive: 2+2 real at r 128 (8 heads) is
  1,245,557 params, 1,119,233 with a main × 4. So a deeper stage comes
  from applications, not from real layers.
- **Main reuse**:
  - It pays on the r 64 stage: R-N15-r64V7 (main × 7 untied,
    0,1,1,1,1,1,2) ties R-N15-B5 at 3230, then leads it (−0.0055 at
    5630, −0.0076 at 9600) and ties the flat at 9600. Its schedule:
    Cyclic ties 0,1,1,1,1,1,2 at 3230 (−0.0010 ± 0.0019, one seed).
  - It does not pay on the r 128 shape: 16 stage heads, N 1.5, main V 6
    Cyclic: +0.006 at 3230 against the base mean (two seeds). 8 stage
    heads × 3+3 × main V 6 × Multi-Gate: +0.0071 / +0.0060 at 3230 /
    5630 against the leader's mean (one seed). Its recall gain led at
    3230 (0.146 against 0.137 / 0.143) and was gone at 5630.
- **Multi-Gate 4** (against the standard residual): −0.0056 / −0.0053 /
  −0.0050 at 3230 / 5630 / 9600 (two-seed means against the base); in
  the main only −0.0059 at 3230 (one seed). 0 MB (three doses), +1.8k
  params. The same gain at every milestone: no late payoff against its
  base. It is far from the −0.046 of the flat (`1m-b`). A reading (the
  user): each stack of the H-Net has few real layers (a stage stack: one
  real layer over 2 applications), so a gate has little to route. It is
  the default now (ORIENTATION).
- **The start rows**: 2 rows = 4 (−0.0033, two seeds each). 8 rows: not
  run.
- **8 stage heads (p 32)**: the same state size, ~630 MB less (§7). Never
  measured alone: alone it sits under the band.
- **Closed**: two stages at total N 2 (+0.024 at 3230, +0.020 at 5630);
  deep compression N 4/4 (+0.13 / +0.108, not the LR); the d64 / S4 line
  (no design near both ceilings; a per-layer stage rank would need a
  library change).
- **The param costs**: a stage boundary ~50k at d128; one outer layer at
  d112 r 256 146k; one untied main application (r 128 shape) 8.6k; one
  untied stage application pair at 8 heads ~7k. The widths must not
  shrink inward. `lrm=auto` stays.

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
- **What moves the peak**: the stages (the 1024 slot), the stage **heads**
  × the state rank. A head dim of 8 doubles the heads (over the band). A
  main at expand 1 and the main MLP are cheap.
- **The stage heads**: 8 heads at p 32 (the same state size as 16 at
  p 16) cost ~630 MB less (3771 → 3144 at the N 1.25 stage + main reuse
  shape) and run ~22% faster. A likely cause: B/C after the per-head bias
  are `[tokens, heads, r]`. But the cost of one more stage application
  stays ~480–590 MB. So 8 heads lower the floor only.

  | 8 stage heads, r 128 | N 1.25 | N 1.5 |
  |---|---|---|
  | 3+3, main × 5 (+ mmlp 8) | 3096 (T4) | |
  | 4+4, main × 5, mg4 | 3683 | not made (~3440?) |
  | 2+2, main V 6, st2 | 3144 | |
  | 3+3, main V 6, st2, mg4 | 3683 | 3442 |
  | 4+4, main V 6, st2, mg4 | > `--max-vram` | 3924 |

- **Use a power-of-2 state rank** (user): another rank jumps in the pages
  and can cost more than the next power of 2. Stage r 128 against r 104 /
  r 96 (probes): R-N15-B5 2617 against 2906, `eV3` 3099 against 3678,
  `s11V2-m5` 2906 against 3485. The flat: r 96 3995, r 128 3776.
- **The cost of one more step of a knob** (MB, dose unless noted): a stage
  application at r 64 ~170 (probe), at r 128 / 16 heads ~480 (probe), at
  r 128 / 8 heads ~480–590; a main application at N 1.25 ~290 (16 stage
  heads) or ~590 (8 stage heads, with 3+3); N 1.5 → 1.25 +240–290 at the
  r 128 shapes; Multi-Gate 4: 0; 2 start rows: 0.
- **The probe → run factor**: × 1.0 (the flat, S4), × 1.06–1.08 (S2 at
  N 4/4), × 1.06–1.34 (S = 1). Dose every shape; `gate.sh` screens a shape
  only after a dose in the band.
- **The flat**: gc saves ~690 MB; each untied virtual application adds
  ~460 MB under gc.

## 8. Kaggle

- **At its quota** (2026-10-13 05:40): "Maximum weekly GPU quota of 30.00
  hours reached." The reset time is unknown (likely 5–7 days). The CLI
  shows no quota: a push is the only test. Assume Kaggle is at its quota
  or near it (user).
- Two T4s, the same pool accounting. A T4: ~1.17 s/step for the R shapes,
  ~1.46 for R-N15-r64V7 (this card ~1.05 / 1.37). A `-mg4` chain with its
  valids: ~1.5 s per step of wall time. The pieces of a chain cost no
  time between them. One batch of this session: 2.7–3.4 h of kernel wall
  time.
- A kernel line that stops early (a failed gate, a short chain) leaves its
  T4 idle while the other line runs: one gated candidate per line, and
  lines of equal length.
- The payload goes up as `payload.tgz.bin` in a new private dataset per
  batch (a `.tar.gz` stalls Kaggle's unpacking for hours).
- `kpush.sh` runs the LR check (`lrprof.py --batch`) before a push, and
  stops when the CLI refuses the kernel (the CLI exits 0 on a refusal).
- `kwait.sh` queues `long.sh` for the pulled `-d` runs here, but it misses
  some `== <run>-d:` headers (11 so far): check each pulled `-d` run.
- Only `probes.txt`, `knobs.txt`, `ctxg.txt` and `speeds.txt` come back as
  text. A Kaggle dose: read the peaks in the pulled `runs/dose-*`.
- The CLI cannot cancel a kernel (the user does it on the web). The flat
  cannot run on Kaggle. After an environment reset the CLI can be gone
  (the user installs it with pip).

## 9. Lessons

- **A reference must be at both ceilings too**: h16 gave the H-Net a false
  lead of 0.015–0.026.
- **Rank by two-seed means.** The stage + main reuse screen read +0.019
  against the leader's seed 0 (its lucky seed). Against the base mean it
  was +0.009, and its second seed made it +0.006. A lead against one seed
  of the reference is not a lead: the flat's second seed cut the lead of
  `-mg4` at 5630 from −0.0028 to −0.0013.
- **Compare two models on the same schedule**, and check it before a
  verdict (`lrprof.py <run>`). The H-Net configs held only the ramp, an
  R-N15-r64V7 batch omitted `--lr`, and nothing checked: 4e-3 to 8000,
  +0.019 at 9600, a wrong "closed". The fix: the protocol in the config,
  `lrprof.py --fill` for 50 trunks, `--guard` (`extend.sh`) and `--batch`
  (`kpush.sh`).
- **A combined screen tests the composition** (user): it can join changes
  measured alone with a cheap untested one. A surprise then says that the
  untested change matters or that the measured ones do not compose, and a
  one-change screen tells which.
- **A cheap change gets the long schedule before it is closed** (user):
  Multi-Gate tied at 3230 and led the flat (seed 0) at 9600 and 14400.
- **A regeneration of `cfg/`** can change old configs through a newer
  default (156 got `bucket`): compare against a backup, keep the old
  files. So `gen.py` keeps `mgr=0` as its default.
- **The queue**: read `queue.txt` again before a Write (`queue.sh` changes
  it); never start `queue.sh` in the same parallel call as the Write (it
  starts empty and exits); never edit a script while a queue line runs it.
- **A new config field** (strict rule 9): never an `Option`.
- **Waiters**: never on `pgrep -f` (strict rule 4). A run that starts in
  the second of a kill can fail in cubecl: wait ~20 s.
- `build.sh`: one command per line and a `cmp` (`cp … && mv …` does not
  stop `set -e`). The disk is shared (627 GB): `target/debug` reached
  115 GB. The Edit tool drops the trailing space of an edit string.
