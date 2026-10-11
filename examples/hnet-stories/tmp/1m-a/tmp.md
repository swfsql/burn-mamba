# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-11

- 03:55 The `kaggle` CLI was gone (an environment reset). The user
  installed it again (`/usr/local/bin/kaggle`, 1.7.4.5).
- 03:55 `kqueue.txt` held three stale lines (the pre-refill configs
  S2x-s112-L4, S2x-s64e-L4, S4x-s64-L4). Replaced.
- 03:55 **Kaggle batch 1** (`/tmp/kpush-1.log`): L-S2a2-B5 (screen, the
  only candidate at the dose target) + L-R-B5-x4630 (the trunk of the N 2
  reference to 4630 + a leaf of 1000). Why the reference now: it is ready,
  and every 4630 comparison of the S2/S4 trunks needs it.
- **Param counts** (flex) of the redesign options:
  - S2 at N 4/4, d128/128 1+1/1+1: 1.13M (main d128 × 3) to 1.25M (× 4). So
    **d112 is the widest S2 with a main of ≥ 4 layers**. One more outer
    layer at d112 r 256 costs 146k.
  - S4 at d80 (every stage ≥ the outer): 1.11–1.23M. So **S4 stays at d64**.
  - S = 1 wider than d128 (N 2): stage d144 2+2 r 64 + main d144 × 2 r 64:
    1.012M; r 104: 1.064M. Stage d160 + main d160 × 1: 1.037M.
- 04:00 12 probes (`probes-B.txt`, the VRAM response of each knob): S2-d112
  + inner r 256 / outer 2+1 / outer p 8; the wide S = 1 stage (d144 r 104,
  d144 r 64, d160 r 64); S4-d64 + ranks 256/256/128/128, 256 ×4, outer
  3+3; S2-d64 + outer 3+3, inner 2+2, d80.
- **Probe round 1** (reserved MB of the 1024 slot; the run factor: × 1.0
  S4, × 1.06–1.08 S2 at N 4/4, × 1.17 S = 1):
  - S2-d112 (base 2861): inner r 256 3032, outer 2+1 3200 (but +146k
    params), outer p 8 > 5 GB (killed).
  - S = 1: d144 r 104 + main × 2 3176 (≈ 3715 in a run: over), d144 r 64
    1985, d160 r 64 + main × 1 2105. The rank tier (64 → 128) is +60% at
    d144: no rank between them fits the target.
  - S4-d64 (base 2622): ranks 256/256/128/128 2713, 256 × 4 2912, outer 3+3
    2912.
  - S2-d64 (base 2517): outer 3+3 2912, inner 2+2 2815, d80/80 3148 (≈
    3400 in a run).
  - The pages of the pools are ~168 MB: the VRAM moves in these steps.
- `expand` is a `usize`: a fractional stage expand (d144 at d_inner 256)
  needs a library change. Not done.
- **The VRAM follows heads × rank** (the [s, h, r] tensors): p 8 doubles
  the heads and blew up. So a d144 stage at **p 18** has R-B5's 16 heads at
  the same rank, and its VRAM should stay near R-B5's. Probe round 2 checks
  it (tmp-v-Wf: main d144 × 3 me 1; tmp-v-Wg: main d144 × 2), with R-B5 as
  the reference probe, and the second designs of the S2/S4 candidates.
- **Probe round 2**: R-B5 2906 (its run 3390). Wf 3003 (≈ 3510 in a run:
  on target). Wg (main d144 × 2, expand 2) 3197 (over). W8 (d136 p 16 r
  104, main d136 × 2) 2993. S2-d112 + outer 2+1 + main expand 1 (A1, 977k)
  3032 (≈ 3215 dosed: under). A3 (outer 2+2 r 128, inner r 64) 2305. S2
  d80/80 (B1, main d112 r 96, 999k) 3027. S2-d64 outer 3+3 + inner 2+2 (B2,
  main d88, 979k) 3201 (on target). S4 d80 (C1, r 256/64/64/64, main d80
  r 64) 2791. S4-d64 outer 3+3, r 256 × 4, main d64 (C2, 1.012M) 3008.
- **Decision: R-B5-s144** = R-B5 × stage d144 p 18 (16 heads, as R-B5) ×
  main d144 × 3 at expand 1, r 48 (C2D), 996,696 params. Why: the user's
  lever (a char-level stage wider than d128, paid by the main). The main
  pays by its expand, not by a layer: the main × 2 version (Wg) is over
  the VRAM, and the main keeps its 3 layers. The main rank 48 trims the
  params (the VRAM tier of 64 stays). One change against R-B5 besides the
  payment.
- **A rule for the char-level stage: 16 heads** (as R-B5). The d64 stages
  have 8 heads (p 16), the d112 ones 14. Maybe part of the width deficit
  is the head count (fewer Δ timescales). The head count is also the
  VRAM-heavy, param-cheap knob that the under-target candidates need.
  Probe round 3: the outer stage at 16 heads (p 14 at d112, p 8 at d64) on
  S2x-s112r256-L4 (P1, 999k), S2x-s64e2-L4 (P2, 1.012M), S4x-s64e2-L4 (P3,
  1.021M), and A1 (P4, 982k). Then the dose of R-B5-s144.
- **Probe round 3**: P1 (S2-d112, outer 16 heads) 3082. P4 (A1 + outer 16
  heads) 3658. P2/P3 (the d64 outer at p 8, 16 heads) > 3855 MiB: **16
  heads at d64 doubles the outer VRAM**, so the head rule does not apply
  at d64. The outer r 512 (2+2, d64) of both S2 and S4: > 3855 MiB too.
- **R-B5-s144 dose: 3193 MB** (steady; × 1.06 of its probe, not × 1.17).
  That is one page under the band. The main at expand 1 is cheap in VRAM.
  So R-B5-s144 does **not** go to Kaggle as it is (strict rule 1). Probe
  round 4 looks for one deliberate knob that adds one page: the stage at
  18 heads (p 16) or 24 heads (p 12), main virtual layers (mV 5,
  0,1,1,1,2), the stage at Complex2D rf 0.5. Also the d64 pair at outer
  2+1 r 512 (main fitted: S2 d120, S4 d96), and the doses of P4 and B2.
- **Probe round 4**: R-B5-s144 + 18 heads (p 16) 3066 (+63), + 24 heads
  (p 12) over, + mV 5 3196 (+193, but a known loser and 1.012M), + C2D rf
  0.5 over. The d64 pair at outer 2+1 r 512: S2 3850, S4 3855 (over). So
  the d64 outer is stuck between r 256 (2+2: under) and r 512 (over).
- **P4 dose 3663 MB** (× 1.0). ~110 over the band, but the process (≈ 3663
  + 147 = 3810 MB) is under 3.6 GiB (3865 MB). **Taken as the S2-d112
  candidate: S2x-e21h16-L4** = S2x-s112r256-L4 × outer 2+1 × outer 16
  heads (p 14) × main expand 1 (pays the outer layer), mmlp 8 (the small
  rest), 993,443 params. Why: the outer stage gets the VRAM (depth and
  heads), and the main of 4 layers stays (the user's hunch). Flag: it has
  ~270 MB more than R-B5 (3390).
- **Kaggle batch 1** (pulled 05:03):
  - **L-S2a2-B5-d wv 0.8965** at 3230: +0.024 behind R-B5 (0.8723), +0.010
    behind the flat (0.8868).
  - **L-R-B5-x4630-d wv 0.8204** at 5630 = **the flat's 0.8204** at 5630.
    The H-Net leads by 0.0145 at 3230 and ties at 5630. (Its stable sv:
    0.9870 at 2630, 0.9556 at 4630.)
- 05:08 **Kaggle batch 2** (`/tmp/kpush-2.log`): L-S2x-e21h16-L4 (screen)
  + L-S2a2-B5-x4630. Why the S2a2 trunk: the only multi-stage trunk at
  the target; at 5630 it tests the "the multi-stage gain shows later"
  hunch (it trails by 0.024 at 3230).
- **R-B5-s144b** = R-B5 × stage d144 at R-B5's own p 16 (18 heads) × main
  d144 × 3 expand 1 r 44: 997,258 params. The natural widening (the head
  dim of R-B5). Dose queued after B2 and two `long.sh`.
- **B2 dose 3201 MB** (× 1.0 of its probe): one page under the band. Not
  sent.
- **R-B5-s144b dose 3390 MB** = R-B5's 3390: on the band, 997,258 params.
  05:18 its long run starts **here** (Kaggle is busy, this GPU idles):
  L-R-B5-s144b (screen) + ctx + long.sh.
- **The long stories** (`longstat.py`, paired, 157 stories), against the
  flat:
  - 3230: R-B5 −0.0069 ± 0.0019 (all; past 2048 +0.004 ± 0.009). S2a2-B5
    +0.0205 ± 0.0021 (every row +0.016 to +0.024).
  - 5630: R-B5 (L-R-B5-x4630-d) **+0.0024 ± 0.0016** (0..1024 +0.004,
    1024..2048 +0.000, past 2048 +0.005 ± 0.005). The H-Net lead of 3230
    is gone at 5630, on the long stories too.
- **The d64 pair is boxed in** between the VRAM tiers: the outer d64 stage
  at r 256 is under the band at 2+2 and 3+3 (S2 B2 3201, S4 C2 3008), and
  over it at r 512 (2+1 already 3850) or at 16 heads (p 8). One more r 256
  outer layer costs ~58k params, which only a main of d64–72 can pay. S4
  at d64 has no design near both ceilings with these knobs (a per-layer
  rank in a stage would give one; that is a library change). So the S4 /
  d64 line waits; the deep-compression question goes on with S2x-e21h16-L4
  (N 4/4, d112).
