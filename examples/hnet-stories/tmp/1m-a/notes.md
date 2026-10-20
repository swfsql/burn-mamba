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
  0.0085 / 0.0031 / 0.0057 / 0.0047 (3230 / 5630 / 9600 / 14400);
  R-N125-r128s11V2-m5 0.022 / 0.013 / 0.0085 (its seed 1 is the outlier of
  its family); `-mg4` 0.007 / 0.006 / 0.003 / 0.001. **The host noise**:
  the same seed and config give 0.8636 here and 0.8681 on a T4 (3230).
  **Rank by two-seed means** (§9).

## 1. The result: R-N125-r128s11V2-m5-mg4 ties the flat, with a trend

| milestone | `-mg4`: s0 / s1 (mean) | the flat h16-V7u: s0 / s1 (mean) | Δ means |
|---|---|---|---|
| 3230 | 0.8625 / 0.8552 (0.8589) | 0.8644 / 0.8559 (0.8602) | −0.0013 |
| 5630 | 0.8040 / 0.7978 (0.8009) | 0.8037 / 0.8006 (0.8022) | −0.0013 |
| 9600 | 0.7455 / 0.7426 (0.7441) | 0.7487 / 0.7430 (0.7459) | −0.0018 |
| 14400 | **0.7174 / 0.7164 (0.7169)** | 0.7220 / 0.7173 (0.7197) | **−0.0028** |

- The lead grows, but at 14400 it is ~1 s.e. of a two-seed difference: a
  tie with a trend.
- **It is a long-context gain only.** At 14400 (means of two seeds): wv
  chars 0..1024 −0.0018, past 1024 −0.014 (32 stories); the 157 long
  stories (paired) 0..1024 +0.0008, 1024..2048 −0.0072, past 2048
  −0.0061. On the first 1024 chars the H-Net ties the flat.
- **R-N125-r128s11V2-m5-mg4** (the leader): stage d128 1+1 real R1D r 128,
  16 heads, × 2+2 untied applications, N 1.25, main d128 × 5 C2D r 64 rf
  0.5, Multi-Gate 4 in every stack. 993,485 params, 3482 MB, ~1400
  ms/step here.
- At N 1.25 the main sees ~80% of the chars. So the leader is close to a
  heterogeneous flat (R1D r 128 around C2D r 64): its lead can come from
  the mixed state sizes, not from compression (§4, N).
- **The flat h16-V7u**: d160 × 5 C2D r 64, 16 heads, × 7 Cyclic untied
  applications, Multi-Gate 4, gc: 995,921 params, 3455 MB, ~1050
  ms/step. It gains 0.022 on the old reference h16 (969k, 3149 MB: under
  both ceilings, so every earlier H-Net lead of 0.005–0.026 was that
  gap). Its virtual schedule (two seeds each, 3230): Cyclic 0.8602, `m2`
  +0.0028, `m13` +0.0097.
- **The candidate: the leader × stage λ 2** (`-lrm2`, §2). At 3230 its
  two-seed wv is 0.8558: −0.0031 against the leader, −0.0045 against the
  flat. It gains on chars 0..1024 and loses past 1024. Its chains to 9600
  run (`handoff.md`).

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
- **The stage LR multiplier λ** (`lrm`: 0 params, 0 MB). `lrm=auto` gives
  λ = √N: 1.12 at N 1.25, 1.41 at N 2. So the leader's stage got its λ
  from the heuristic, not from a measurement. Two seeds each at 3230,
  against the leader (auto):

  | λ | wv s0 / s1 (mean) | Δ mean | chars 0..1024 s0 / s1 | past 1024 s0 / s1 |
  |---|---|---|---|---|
  | 1.41 | 0.8642 / 0.8680 (0.8661) | +0.0072 | +0.0008 / +0.0113 | +0.024 / +0.019 |
  | 2 | 0.8576 / 0.8539 (0.8558) | **−0.0031** | **−0.0047 / −0.0021** | +0.009 / +0.016 |

  The bands are paired against the leader's seed (± 0.0016 on 0..1024,
  ± 0.006–0.014 past 1024). Not monotone in λ, so part of it can be
  noise. At λ 2 the trunks trail (+0.006 / +0.016 at 2630), and the
  leaves recover more.

## 3. The results of the new protocol (wv at 3230)

| run | shape (change from its base) | wv | peak MB |
|---|---|---|---|
| L-R-N125-r128s11V2-m5-mg4-lrm2-d / -s1-d | the leader × stage λ 2 (§2) | 0.8576 / 0.8539 | 3482 |
| L-R-N125-r128s11V2-m5-mg4-d / -s1-d | the leader (§1) | 0.8625 / 0.8552 | 3482 |
| L-R-N125-r128s11V2-m5-mg4-st2-d / -s1-d | the leader × 2 start rows (T4 / here) | 0.8657 / 0.8572 | 3482 |
| L-R-N125-r128s11V2-m5-mmg4-d | R-N125-r128s11V2-m5 × Multi-Gate 4 in the main only | 0.8586 | 3482 |
| L-R-N125-r128s11V2-m5-st2-d / -s1-d | R-N125-r128s11V2-m5 × 2 start rows | 0.8566 / 0.8657 | 3482 |
| L-R-N125-r128s11V2-m5-d / -s1-d | stage d128 1+1 R1D r 128 × 2+2 untied applications, N 1.25, main d128 × 5 C2D r 64 rf 0.5 | 0.8533 / 0.8756 | 3482 |
| L-R-N125-r128s11V13-m5-mg4-d / -s1-d | the leader × encoder 1, decoder 3 applications | 0.8636 (T4 0.8681) / 0.8696 | 3482 |
| L-R-N125-B5-d / -s1-d | R-N15-B5 × N 1.25 | 0.8638 / 0.8624 | 3386 |
| L-R-N125-r128s11V2-m5-mg4-lrm141-d / -s1-d | the leader × stage λ 1.41 (§2) | 0.8642 / 0.8680 | 3482 |
| L-R-N125-r128s11V2-m4d144-mg4-d / -s1-d | the leader × main d144 × 4 (18 heads) | 0.8715 / 0.8602 | 3482 |
| L-R-N15-r64V7-d / -s1-d | R-N15-B5 × stage r 64 × main V 7 (0,1,1,1,1,1,2, untied) | 0.8644 / 0.8675 | 3452 (max 3592) |
| L-R-N15-r64V7c-d | R-N15-r64V7 × a Cyclic main | 0.8636 | 3592 |
| L-R-N15-r128s11V3-m5V6c-st2-sp32-mg4-d | 8 stage heads × 3+3 × main V 6 Cyclic × 2 start rows × Multi-Gate 4, N 1.5 | 0.8660 (5630: 0.8069) | 3442 |
| L-R-N15-r64sV5-d | R-N15-B5 × stage r 64 × 5+5 untied applications | 0.8670 | 3522 |
| L-R-N15-r128s11V4-m5-sp32-mg4-d / -s1-d | the leader × 8 stage heads × 4+4, N 1.5 | 0.8719 / 0.8685 | 3539 |
| L-R-N15-B5-d / -s1-d | stage d128 2+2 R1D r 104, main d128 × 3 C2D r 64, N 1.5 | 0.8688 / 0.8605 | 3386 |
| L-R-N15-B-d | R-N15-B5 × 7 slots | 0.8697 | 2799 |
| L-R-N15-r128s11V2-m5V6c-st2-d / -s1-d | R-N125-r128s11V2-m5 × N 1.5 × main V 6 Cyclic × 2 start rows | 0.8733 / 0.8676 | 3482 |
| L-R-B5-d | R-N15-B5 × N 2 | 0.8723 | ~3390 |
| L-R-N15-r64V7-mg4-s1-d | R-N15-r64V7 × Multi-Gate 4 (s0: over the gate on the T4) | – / 0.8740 | 3455–3733 |
| L-R-N125-r128s11V31-m5-mg4-d | the leader × encoder 3, decoder 1 applications | 0.8752 | 3482 |
| L-R-N2-r128s11V23-m5-mg4-st2-d / -s1-d | the leader × N 2 × decoder 3 × 2 start rows | 0.8812 / 0.8714 | 3385 |
| L-R-B5-s144b-d | R-B5 × stage d144 × main d144 × 3 expand 1 | 0.8838 | 3390 |
| L-R-N125-r128s11V2-m5-mg4-sp32c-d / -s1-d | the leader × 8 stage heads × a C2D stage (rf 0.5) | 0.8843 / 0.8850 | 3385 |
| L-S2a2-B5-d | two stages d112/120, N 1.41/1.41, main d120 × 3 | 0.8965 | 3532 |
| L-M-s96r128-L5-d | stage d96, main d112 × 5 | 0.9022 | 2440 * |
| L-M-s64r256-L4-d | stage d64 r 256, main d144 × 4 | 0.9723 | 2711 * |
| L-S2x-e21h16-L4-lrm-d | two stages d112, N 4/4, main d112 × 4 | 1.0048 | 3663 |

\* Under the band: weak evidence.

## 4. The shapes (new protocol)

- **The leader is a local optimum of its neighbours.** Every one-change
  variant of this search lost (two-seed means unless noted, against the
  leader): 8 stage heads × 4+4 +0.011; encoder 1 + decoder 3 +0.010 /
  +0.009 (3230 / 5630); encoder 3 + decoder 1 +0.013 (one seed);
  N 2 × decoder 3 +0.017 (3230; +0.005 at 14400); 2 start rows +0.0026 /
  −0.0013 (3230 / 5630), s0 +0.0001 / +0.0021 (9600 / 14400; the long
  stories at 14400 +0.0007): a tie. Main d144 × 4 lost +0.0070, and 8
  stage heads × a C2D stage +0.026 (3230). Only the stage λ 2 gains (§2).
- **N**: N 1.25 = N 1.5 (0.8631 / 0.8647). N costs no params. It moves
  the peak little where the stage sets it (R-N15-B5), ~240–290 MB per
  0.25 at the r 128 shapes (N 1.25 → 2 frees about one decoder
  application). N 2 loses, and the gap closes only part of the way:
  - R-B5 (N 2) against R-N15-B5 (N 1.5), the same shape: +0.0035 /
    +0.0061 / +0.0036 / +0.0012 (3230 / 5630 / 9600 / 14400); the long
    stories past 2048 +0.016 / +0.007 (9600 / 14400).
  - **The N ladder, rung a** (the leader × N 2 × decoder 3 × st2): wv s0
    / s1 0.8812 / 0.8714, 0.8153 / 0.8100, 0.7541 / 0.7476, 0.7250 /
    0.7190. Against the leader's two-seed means: **+0.0174 / +0.0118 /
    +0.0068 / +0.0051**; against the flat's +0.016 / +0.0105 / +0.005 /
    +0.0023. Seed 0 on the long stories (paired, against the leader's
    s0): +0.024 / +0.012 / +0.0077 / +0.0068; at 14400 the two-seed means
    against the leader's: all +0.0042, 0..1024 +0.0032, past 2048 +0.0025.
  - So N 2 closes ~70% of its gap by 14400 and still loses by 14400: on
    this protocol, at 1M params, the compression pays late, if at all.
    N = 1 is undefined: the ratio loss (burn-stack
    `routing.rs::ratio_loss`) scales by N / (N − 1).
- **The stage placement** (the 4 stage applications of the leader):
  2+2 is the best. Encoder 1 + decoder 3 ties on chars 0..1024 and loses
  the tail and some recall; encoder 3 + decoder 1 loses everywhere. So
  the decoder needs ≥ 2 applications, and a third decoder application is
  worth about one encoder application.
- **The stage width is the lever, up to d128**: d64 +0.10, d96 +0.03 to
  +0.05. A d144 stage paid by the main loses 0.0115, mostly in the recall.
- **The main carries the recall**: a weaker main and deep compression
  lose it.
- **The main wants depth**: main d144 × 4 (988,961, 3482 MB) for d128 × 5
  loses +0.0070 (two seeds, 3230). Seed 0: +0.0090 / +0.0097 (3230 /
  5630), and +0.0096 / +0.0094 on chars 0..1024 (paired). The main cannot
  go narrower than the stage (d128), so the main is at a local optimum
  too. Closed. (`lrm=auto` scales with the main width: this arm ran at
  stage λ 1.19.)
- **`rf 1` in the main** (the flat's value) costs +10,320 params (16 more
  angles per layer) and one page (3482 → 3675 MB). It fits the params
  only at main d144 × 4 (998,241). There it doses 3675 at N 1.25 and at
  N 1.5, at 16 or 18 main heads. Shelved: it is over the band. Its
  evidence at u 1 is weak (the flat: the rotation ties at u 1, and rf 0.5
  lost at u 2).
- **Stage reuse** (virtual layers in a stage, `eV`/`dV`, untied): at r 64
  it does not pay (5+5: 0.8670). At r 128 it pays: 1+1 real × 2+2
  applications free ~260k params for a main × 5 (the leader's base). A
  real stage layer is expensive (2+2 real at r 128: 1,245,557 params). So
  a deeper stage comes from applications, and more than 2+2 did not pay
  (8 stage heads × 3+3 or 4+4).
- **Main reuse**:
  - It pays on the r 64 stage: R-N15-r64V7 (main × 7 untied,
    0,1,1,1,1,1,2) ties R-N15-B5 at 3230, then leads it (−0.0055 at
    5630, −0.0076 at 9600) and ties the flat at 9600.
  - It does not pay on the r 128 shape: +0.006 at 3230 (16 stage heads,
    N 1.5, main V 6, two seeds); +0.007 / +0.006 at 3230 / 5630 (8 stage
    heads × 3+3, one seed).
- **Multi-Gate 4** (against the standard residual): −0.0056 / −0.0053 /
  −0.0050 at 3230 / 5630 / 9600 on the leader's base (two seeds); 0 MB,
  +1.8k params. Far from the −0.046 of the flat (`1m-b`). On main reuse
  (R-N15-r64V7, its s1) it **loses** +0.0065 / +0.0046 (3230 / 5630): it
  does not pay more where the main routes over 7 applications. A reading
  (the user): each H-Net stack has few real layers, so a gate has little
  to route.
- **8 stage heads (p 32)**: the same state size, ~630 MB less (§7), but
  the freed VRAM bought only stage applications that lose (+0.007 to
  +0.011). A C2D stage at 8 heads (`sp32c`, 993,517) is param-neutral: 8
  heads free 16.5k, and the C2D stage costs 16.5k. It loses +0.026 (two
  seeds, 3230), on chars 0..1024 and past them. So the stage wants at
  least 16 heads. Closed.
- **32 stage heads × r 64** (`sh32r64`, p 8). It keeps the leader's heads
  × rank (the VRAM, §7), with half the state and twice the decay
  timescales. 984,813 params, + `subc` (8.2k) = 993,005. Its screens run
  (`handoff.md`).
- **Closed**: two stages at total N 2 (+0.024 / +0.020 at 3230 / 5630);
  deep compression N 4/4 (+0.13 / +0.108); the d64 / S4 line (no design
  near both ceilings).
- **The param costs**: a stage boundary ~50k at d128; one outer layer at
  d112 r 256 146k; one untied main application (r 128 shape) 8.6k; one
  untied stage application at 16 heads ~6.6k, at 8 heads ~7k per pair.
  On the leader:
  - `rf 1` +10.3k.
  - A C2D stage +16.5k (its angles are in the untied tail of every
    application).
  - 8 stage heads −16.5k, 32 heads +33k (the per-(head, rank) B/C bias
    and the per-head tails).
  - Stage expand 1 −99k (+ main × 6: 1,020,961, + r 256: 969,901).
  - `subc`: +2·h·r per extra stage application.

  The leader has ~6.5k of headroom. The widths must not shrink inward.

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
  ~97–168 MB: a peak moves in these steps.
- **CUDA 13.0** (the box, 2026-10-16; was 12.2): the old binaries run
  unchanged, the leader doses 3482 MB again, the flat runs 1049 ms/step
  (1063 before). So the doses of CUDA 12.2 hold.
- **What moves the peak**: the stages (the 1024 slot), the stage **heads
  × the state rank**. The per-(head, rank) bias makes B/C per head
  (`[b, s, h, r]`), so their memory follows h × r. 8 heads × r 128: −630
  MB. 32 heads × r 128: over the band. 32 heads × r 64 (the leader's h ×
  r): 3578 max, 3482 in 4 of 5 windows. A main at expand 1 and the main
  MLP are cheap. The stage placement does not move it (1+3, 2+2, 3+1 all
  3482). The main heads do not move it either (16 or 18 at d144 × 4).
- **`rf 1` in the main costs one page**: main d144 × 4 3482 → 3675. N 1.5
  does not pay it back (3675 max, 3579 in 4 of 5 windows).
- **The peak varies by seed and host**: R-N15-r64V7-mg4 dosed 3592 here,
  3733 (s0) and 3455 (s1) on the T4s. A shape one page under the gate can
  fail it on another seed.
- **The stage heads**: 8 heads at p 32 cost ~630 MB less than 16 at p 16
  (3771 → 3144 at the N 1.25 stage + main reuse shape) and run ~22%
  faster. One more stage application still costs ~480–590 MB.

  | 8 stage heads, r 128 | N 1.25 | N 1.5 |
  |---|---|---|
  | 3+3, main × 5 (+ mmlp 8) | 3096 (T4) | |
  | 4+4, main × 5, mg4 | 3683 | 3539 |
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
  heads) or ~590 (8 stage heads, with 3+3); N 1.5 → 1.25 +144–290 at the
  r 128 shapes; N 1.25 → 2 with one more decoder application −97;
  Multi-Gate 4: 0; 2 start rows: 0.
- **The probe → run factor**: × 1.0 (the flat, S4), × 1.06–1.08 (S2 at
  N 4/4), × 1.06–1.34 (S = 1). Dose every shape; `gate.sh` screens a shape
  only after a dose in the band.
- **The flat**: gc saves ~690 MB; each untied virtual application adds
  ~460 MB under gc.

## 8. Kaggle

- **The weekly quota (30 h) resets on Saturday 00:00 UTC**: refused on
  Friday 2026-10-16 20:48, accepted on Saturday 00:23. The CLI shows no
  quota: a push is the only test.
- **The CLI is 2.2.4** (was 1.7.4.5); `kg.sh` and the scripts work
  unchanged.
- A pushed kernel can stay **QUEUED for hours** (version 39: ~2.5 h, a
  Saturday evening); earlier pushes started at once.
- **A cancelled kernel leaves no output** (`kpull.sh` merges 0 runs).
- Two T4s, the same pool accounting. A T4: ~1.17 s/step for the R shapes,
  ~1.46 for R-N15-r64V7 (this card ~1.05 / 1.37). A chain with its valids:
  ~1.5 s per step of wall time. The pieces of a chain cost no time
  between them.
- A kernel line that stops early (a failed gate, a short chain) leaves its
  T4 idle while the other line runs: one gated candidate per line, and
  lines of equal length. The kernel runs as long as its longest line, so
  more steps on the shorter line cost no quota.
- A seed twin of a shape dosed here screens on Kaggle with no dose (no
  `doses.txt` there, and the shape is the same).
- `kwait.sh` dies with the box. After a shutdown, pull with
  `kaggle/kpull.sh`, then queue `long.sh` for each pulled `-d` run.
- The week of 2026-10-17: ~16 h by 10-18. Then three batches: ~1.5 h,
  ~1.5 h, and ~3.3 h (pushed 2026-10-20 03:21). In total ~22 h.
- The payload goes up as `payload.tgz.bin` in a new private dataset per
  batch (a `.tar.gz` stalls Kaggle's unpacking for hours). A refused push
  leaves its dataset behind (the CLI cannot delete it).
- `kpush.sh` runs the LR check (`lrprof.py --batch`) before a push, and
  stops when the CLI refuses the kernel (the CLI exits 0 on a refusal).
- `kwait.sh` queues `long.sh` for the pulled `-d` runs here, but it misses
  some `== <run>-d:` headers (13 so far): check each pulled `-d` run. It
  also starts the local queue: in a wind-down, use a pull-only waiter.
- A run name that exists here is not merged (it stays in `kaggle/out/`):
  move a killed local run aside (`runs/killed-<name>`) before the pull.
- Only `probes.txt`, `knobs.txt`, `ctxg.txt` and `speeds.txt` come back as
  text. A Kaggle dose: read the peaks in the pulled `runs/dose-*`.
- The CLI cannot cancel a kernel (the user does it on the web). The flat
  cannot run on Kaggle. After an environment reset the CLI can be gone
  (the user installs it with pip).

## 9. Lessons

- **A reference must be at both ceilings too**: h16 gave the H-Net a false
  lead of 0.015–0.026.
- **Rank by two-seed means.** The stage + main reuse screen read +0.019
  against the leader's seed 0 (its lucky seed); its second seed made it
  +0.006. The flat's second seed cut the lead of `-mg4` at 9600 / 14400
  from −0.0047 / −0.0051 to −0.0018 / −0.0028.
- **Models before seeds**: a one-seed gap > 0.01 against the same seed of
  the leader closes a candidate (V31); a gap under that gets its second
  seed.
- **Compare two models on the same schedule**, and check it before a
  verdict (`lrprof.py <run>`). An R-N15-r64V7 batch without `--lr` ran
  4e-3 to 8000 and gave a wrong "closed" (+0.019 at 9600).
- **A combined screen tests the composition** (user): it can join changes
  measured alone with a cheap untested one. A surprise then says that the
  untested change matters or that the measured ones do not compose, and a
  one-change screen tells which.
- **A cheap change gets the long schedule before it is closed** (user):
  Multi-Gate tied at 3230 and led at 9600 / 14400; N 2 closed ~70% of its
  3230 gap by 14400.
- **Host-matched seeds**: the leader's s0 ran here and its s1 on a T4. So
  a candidate runs s0 here and s1 on Kaggle: each per-seed comparison then
  stays on one host type (the host noise: 0.0045 at 3230).
- **A heuristic default drifts with its inputs**. `lrm=auto` lowered the
  stage λ from 1.41 to 1.12 when N moved from 2 to 1.25. No run measured
  that. After every shape neighbour lost, λ 2 gained. Measure a derived
  default again when its input moves.
- **A regeneration of `cfg/`** can change old configs through a newer
  default (156 got `bucket`): compare against a backup, keep the old
  files. So `gen.py` keeps `mgr=0` as its default, and a new key
  (`emgr`/`dmgr`) defaults to the old behaviour (checked byte for byte).
- **The queue**: read `queue.txt` again before a Write (`queue.sh` changes
  it); never start `queue.sh` in the same parallel call as the Write (it
  starts empty and exits); never edit a script while a queue line runs it.
- **A new config field** (strict rule 9): never an `Option`.
- **Waiters**: never on `pgrep -f` (strict rule 4). A run that starts in
  the second of a kill can fail in cubecl: wait ~20 s. In a `Monitor`
  pipe, every stage must flush per line (`cut` does not: no events).
- `build.sh`: one command per line and a `cmp` (`cp … && mv …` does not
  stop `set -e`). The disk is shared (627 GB): `target/debug` reached
  115 GB. The Edit tool drops the trailing space of an edit string.
