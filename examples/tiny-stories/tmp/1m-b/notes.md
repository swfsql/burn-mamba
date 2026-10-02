# notes: tiny-stories ≤1M search, part b (results and findings)

`archive/notes.md` is the full chronological log (search it for a detail, do
not read it whole). `../1m/` is the older history.

Notation: sv → **dv** = the stable valid → the decayed valid, in bits/char.
ms = ms/step. MB = cubecl `reserved`. The noise of dv is ±0.005–0.008 (one
seed per run). A run name is its directory in `runs/`, and a decay branch
adds `-d`.

## 1. The 256-char regime (bs32 seq256 runlen4, 600 + 200 steps)

Start: `../1m` s04 = 1.2092. Base **b01-s04** = `cfg/d128L8` (d128 L8 e2 p64
r64, MultiGate n4, Complex2D, LR 4e-3, 990,784 params): dv 1.2042, 590 ms,
2501 MB. Each variant pays for its feature with a layer or with `r`, so it
stays under 1M params. All rows are step-matched.

| run | change | dv | ms |
|---|---|---|---|
| f01-q4d | Quaternion4D, L7 r60 | 1.2065 | 714 |
| f02-real1d | Real1D | 1.2089 | 557 |
| f03-l4v8 | L4 × V8 (502k params) | 1.2651 | 522 |
| f04-m2 | MIMO m2, L7: 4.9 GB, over the cap | 1.2132 | 1727 |
| f05-mlp | L6 + MLP 96 | 1.2074 | 514 |
| f06-kalman | `gain=Kalman` | 1.2130 | 720 |
| f09 | L4 × V8, 2 heads r256, Real1D, untied (690k) | 1.2485 | 678 |
| f10 | f09 + g2 (955k) | 1.2594 | 734 |
| f11 | d96 L13 | 1.2529 | 1018 |
| **f12** | **d160 L5 p80** (932,680) | **1.1994** | 433 |
| f13 | d184 L4 | 1.2083 | 332 |
| f14 | f12, seed 1 | 1.1915 | – |
| f15 | u2 d156 L4, HorizontalCarryOver | 1.1897 | 440 |
| **f16** | **f15 with Vertical** | **1.1825** | 498 |
| f17 / f23 | f16 with VPH-CarryOver / VPH-Reset | 1.1981 / 1.1965 | ~475 |
| f18 / f19 | u4 d144 L3 / u3 d136 L4 (Vertical) | 1.2320 / 1.1978 | 541 / 570 |
| f20 | f16 at e3 d128 | 1.1818 | 476 |
| f21 / f29 | f16 + Real1D / + rf 0.5 | 1.1954 / 1.1919 | 412 / 441 |
| f22 | f21 at L4 × V6 | 1.1922 | 598 |
| f24 / f25 | f16 at LR 8e-3 / 2e-3 | 1.1904 / 1.2135 | – |
| f26 | f16 + standard residual | 1.1838 | 423 |
| f27 / f28 | f16 with 8 heads / 2 heads | 1.1987 / 1.1925 | 617 / 389 |
| f30 | f16 at bs16 seq512 runlen2 | 1.1531 | 449 |
| **f31** | **f16 at bs8 seq1024 runlen1** | **1.1351** | 451 |
| f32 | f31 + `trop=MaxPlus` | 1.1410 | 479 |
| f33 | f12 at bs8 seq1024 (`cfg/s1k-u1`) | 1.1458 | 387 |

The extensions x01–x04 (f16, f12, f13, f18 to 1200 + 200) gave 1.0986,
1.1052, 1.1189, 1.1424: the short screens predicted the order 4/4.

- **The BPTT window is the largest lever** (the same chars per step):
  256 × 32 1.1825, 512 × 16 1.1531, 1024 × 8 1.1351. The valid set holds
  `seq × runlen` chars per story, so keep `seq × runlen = 1024`.
- Params matter: the weight sharing of f03 costs 0.06. More virtual compute
  at full params (f22) does not pay.
- VRAM grows with heads × `r`: B/C are per head after the per-(head, rank)
  bias, and the rotation multiplies them. A large `r` is not cheap.
- Depth L4–L8 is flat step-matched, and L13 loses. Shallow shapes are faster.
- The `u` dial peaks at u2 (u3/u4 take params from the width). At u2 the
  single Vertical tap wins, and both two-tap members lose.
- The rotation ties at u1. At u2 it pays ~0.013 (full > half > none).
- LR 4e-3 is the peak.
- The cheap knobs (standard residual, 2 heads, Real1D, rf 0.5, e3, MLP) tie or
  lose ~0.01, and each saves 10–20% of the step.

The first pool came from the s1k regime: A = `u2v-s1024` (f31), B =
`s1k-u1` (f33), C = `s1k-u2lean` (A + standard residual + 2 heads), D =
`s1k-u1lean` (B + standard residual + 2 heads).

## 2. Packed training

Packing sweep (`s1k-u1lean`, 200 batches, rows × width + chunk → kchars/s):
**2×4224 c128 23.66** · 2×4096 c128 22.52 · 4×2048 c128 22.36 · 4×2112 c64
21.63 · 4×2176 c128 21.16 · 2×4224 c64 20.91 · 2×4352 c256 20.18 · 1×8448
c128 18.52 · 8×1024 c128 18.47. Packing gives +11–13% chars/s over unpacked.

**Chars, not steps, set the loss**: a fresh packed B at 2630 + 600 (k02,
0.8713) equals a B with an unpacked history at the same chars (0.8710).
@3600, A and D trailed B and C by 0.012–0.017.

### Screens (fresh, packed, 2630 + 600 → dv @3230)

**B-wd** = `k05-B-wd1` (`cfg/s1k-u1-wd1`: d160 L5 p80 r64, 4 heads, MultiGate
n4, Complex2D, u1, wd 0.1, 932,680 params). k01–k04 use B without wd.

| run | change | params | dv | ms | MB |
|---|---|---|---|---|---|
| k02-B-fresh | B (wd 1e-4) | 932,680 | 0.8713 | 410 | 2065 |
| k05-B-wd1 | **B-wd** | 932,680 | 0.8705 | 430 | 2065 |
| k01-B-kalman | B + `gain=Kalman` | 932,720 | 0.8714 | 566 | 2259 |
| k03-B-mgr0 | B + standard residual | 931,060 | 0.9175 | 375 | 2041 |
| k04-B-h2 | B with 2 heads p160 | 926,950 | 0.8988 | 403 | 1896 |
| k06 / k07 | B-wd, MultiGate n6 / n3 | 932,690 / 932,675 | 0.8877 / 0.8812 | ~400 | – |
| k08-Cwd | C + wd (u2, `recalc:256`) | 969,980 | 0.8813 | 387 | 2211 |
| k09-Bwd-r128 | d152 p76 r128 | 975,372 | 0.8680 | 480 | 2623 |
| k10 | d152 L4 r256 | 984,328 | 0.9014 | 514 | 2962 |
| k13 | d144 L6 p72 | 921,936 | 0.8829 | 501 | 2356 |
| k14-q4d | Quaternion4D g2 rf 0.5, L3 × V6 Stretched d164, untied | 965,680 | 0.8900 | 987 | 3543 |
| k15 | 8 heads p40 | 944,740 | 0.8772 | 520 | 2453 |
| **k16-Bwd-h16** | **16 heads p20** | 969,160 | **0.8579** | 742 | 3447 |
| k17-h16g2 | 16 heads, g2, d152 p19 | 982,747 | 0.8649 | 771 | 3496 |

The probe skipped r192 (`d140r192`, 3593 MB, no margin) and MIMO m2
(`d152m2`, over 3480 MiB at the first step).

### Milestones (fresh histories, decayed valid)

Stable 2630 / 4630 / 8000 / 12000, then a decay of M/5. The LR column is the
constant LR after step 4630.

| model | LR | @3230 | @5630 | @9600 | @14400 |
|---|---|---|---|---|---|
| B-wd | 4e-3 / 2e-3 / 1e-3 | 0.8705 | 0.8156 | 0.7748 / 0.7641 / 0.7607 | |
| B, no wd | 4e-3 | 0.8713 | 0.8138 | 0.7744 | |
| r128 | 4e-3 / 2e-3 / 1e-3 | 0.8680 | 0.8113 | 0.7754 / 0.7641 / 0.7590 | |
| C-wd | 1e-3 | 0.8813 | 0.8310 | 0.7805 | |
| Kalman, no wd | 1e-3 | 0.8714 | 0.8154 | 0.7791 | |
| q4d | – | 0.8900 | 0.833 | | |
| **h16** | 1e-3 | 0.8579 | 0.8014 | 0.7493 | **0.7281** |
| h16 | 1e-3, 5e-4 from 8000 | | | | 0.7283 |
| h16 × g2 | 1e-3 | 0.8649 | 0.8083 | 0.7552 | 0.7327 |

### Context use (far use, pairing i+1)

The second pairing (i+2) differs by ≤ 0.003 (the noise).

| model | @3230 | @5630 | @9600 | @14400 |
|---|---|---|---|---|
| B-wd | 0.217 | 0.241 | 0.259 (4e-3), 0.261 (1e-3) | |
| r128 | 0.231 | 0.251 | 0.267 | |
| Kalman | | 0.239 | | |
| C-wd | | | 0.247 | |
| h8 | 0.210 | | | |
| h16 | 0.224 | 0.253 | 0.269 | 0.284 (1e-3 and 5e-4) |
| **h16 × g2** | **0.241** | **0.261** | **0.285** | **0.289** |

The 256 regime (800 steps) had 0.04–0.065.

### Findings

- **wd 0.1** ties no wd to 9600 at 4e-3. With wd the decay recovers more
  (0.113 against 0.087 @3230): wd holds the weight norms, so the effective
  step stays large. Without wd, the step-down recovers less at the decay
  (Kalman 0.7791). User: every candidate has wd 0.1.
- **The LR step-down** at 4630 pays −0.011 (2e-3) and −0.014 … −0.016
  (1e-3), on B and on r128. A halving to 5e-4 at 8000 ties at 14400. The
  constant 4e-3 phase is flat, and its decay recovers 0.07–0.11.
- **Heads lower the loss**: h16 is −0.011 … −0.015 against B at every
  milestone (80 decay timescales). h8 does not help (not monotone). h16 is at
  the VRAM cap.
- **Groups raise the context use**: h16 × g2 has the highest far use at every
  milestone (+0.015 over h16 @9600) for +0.006 of loss (d152 to fit 1M).
  But only +0.005 @14400, and the seed spread of the far use is ~0.017
  @3230 (§3). One seed per run: not proven.
- The far use grows ~+0.03 per doubling of the steps. The LR step-down buys
  local modelling only. The 1024 windows were a context-use win.
- r128 learns faster early and uses more context. But its loss meets B's by
  9600, at 1.19× the step. r256 (L4) loses, and r192 does not fit.
- The corner of B is a local optimum: MultiGate n4 × 4 heads × d160 L5. The
  standard residual costs +0.046 at 4 heads, but it wins at 2 heads (D 0.888
  against k04 0.899). L4 and L6 lose.
- u2 (C) trails at every milestone, also at 1e-3. MIMO does not fit the cap.
- Kalman ties B at 3230 and 5630, at 1.35× the step. Its no-wd run @9600
  cannot separate Kalman from wd.
- The quaternion L3 × V6 (a user request) trails by ~0.02 at 3230 and 5630,
  at 2.2× the step, with B's context use.

### Inference

- `infer-r128-lr1-9600*/` hold samples of `p8k-r128-lr1-d`: built-in prompts,
  the user's two prompts, T 0.8 and 0.5. The unprompted text is coherent. The
  prompted continuations are fluent, but they ignore the prompt.
- The decode is exact (the example tests check it, also on a real checkpoint:
  `TS_CHECKPOINT=<run>`). So the drift comes from the model: its plot-level
  memory is weak. This is what the far use measures.

### Ideas not tried

- mnist-class (`examples/mnist-class/model.rs`, 1k params) is a reference for
  when the screens stop separating: L1 × V4 `Stretched`, `ngroups = nheads`,
  Quaternion4D rf 0.5, no proj bias, out-norm, MultiGate n4, untied small
  tensors + layer `Norm`, no class latents, no final norm.
- h8 × g2 (d152, 959,282 params).

## 3. The `v0.64.0` build: fill the ceilings (from 2026-09-30)

User direction: use the most of both ceilings (1M params, VRAM). Where a
shape has VRAM left, add cheap capacity (expand, virtual layers, …). Virtual
layers (V > L) always take the mnist-class untied tensors (`untie=1`: the
block tail + small tensors, and the layer `Norm`). Also try a shot in the
dark: many heads × many groups × many layers × Quaternion4D (rf 0.5).

- A quaternion costs `nheads · 3 · blocks` in-proj channels (`blocks =
  r·rf/4`), and each untied application holds its own tail. So heads ×
  quaternion × applications forces a small `r` or few real layers.
- `queue.sh` runs the lines of `queue.txt` in order (edit the data file, not
  the script). `qwait.sh` (in the background) prints once when the next line
  ends. A cron heartbeat (~30 min) checks the queue.
- New flag `--pack-align N` (the example): the stories start at multiples of
  N, a multiple of the chunk. So chunk 64 trains on the rows of chunk 128.

### Probes (10 steps, packed, `recalc:128`)

Process ≈ reserved (MiB) + ~135 MiB. Limits: process 3686 (preferred) /
3891 (search).

| cfg | shape | params | reserved MiB | process | ms |
|---|---|---|---|---|---|
| `s1k-u1-wd1-h16` | h16 (calibration) | 969,160 | 3003 | 3134 | 678 |
| `n-h20` | d160 L5 p16, 20 heads | 981,400 | 3104 | 3228 | 742 |
| `n-h32-d144` | d144 L5 p9, 32 heads | 849,489 | 4795 | 4920 | 1039 |
| `n-h16r128-d144` | d144 L5 h16 r128 | 931,214 | 4399 | 4539 | 903 |
| `n-h16-V6` | h16, L5 × V6 Cyclic, untied | 982,380 | 3441 | 3578 | 807 |
| `n-q-h16g16r16-L3V9` | Q4D rf 0.5, h16 g16 r16, L3 × V9 Stretched | 940,016 | 3659 | 3800 | 1351 |
| `n-q-h16g8r32-L3V6` | Q4D rf 0.5, h16 g8 r32, L3 × V6 | 964,196 | 3221 | 3356 | 992 |
| `n-q-h16g16r32-L2V8` | Q4D rf 0.5, h16 g16 r32, L2 × V8 | 968,712 | 4389 | 4540 | 1386 |

h32 at d160 has 1,018,150 params (over). h16 → h20 costs +100 MiB, but h32
costs +1800 MiB. Chunk 64 (`--pack-align 128`) does not help: h32 4752 MB
(against 5028), h16 2986 MB (against 3149). So the per-head memory is not the
`l²` chunk tensors. It is B/C per head (`[b, s, h, r]`, after the bias and
the rotation), ∝ heads × `r`. The state is `d_inner × r` at any head count,
so h32 at r32/r48 keeps the B/C memory of h16 r64 or less: `n-h32r32`
943,190, `n-h32r48` 980,670. Probes: h32 r48 > 5400 MiB total (killed,
4876 MB reserved at step 5). **h32 r32 3507 MB** (process ~3460 MiB), 856
ms. The memory jumps in steps (r32 → r48: +1370 MB), probably at the pool
page sizes. So a probe is the only reliable size check.

### Screens (fresh, packed, 2630 + 600 → dv @3230)

- **n01-q-g16r16** (`n-q-h16g16r16-L3V9`): the user's shot in the dark. All
  four levers at once: 16 heads, `ngroups = nheads`, 9 applications, Q4D.
  The process is ~3800 MiB (over the preferred limit, under the search
  limit). sv 1.0634 → **dv 0.9372**, 1402 ms. Last at every bucket, with a
  constant gap of +0.10 to h16 (not a late bloomer). Evicted. The suspects
  (not separated): r16 (state 20 × 16 per head), the 9 applications at LR
  4e-3, the B/C width of g16, the quaternion. n03 (h16 V6) tests the virtual
  layers alone.
- **n02-h20** (d160 p16): sv 0.9888 → **dv 0.8722**, 768 ms. It tracks
  h16 to @1052, then falls back (+0.019 @2630). The head ladder @3230 is not
  monotone: h4 0.8705, h8 0.8772, h16 0.8579, h20 0.8722. h16 can be a lucky
  seed (one seed per run). Control: **n04-h16-s1** (k16 with seed 1): sv
  0.9717 → **dv 0.8596** (seed 0: 0.8579). h16 replicates. The ladder is
  really not monotone (h20 +0.013), and n03's V6 margin (−0.005 against
  both seeds) holds.
- **n03-h16-V6** (h16, L5 × V6 Cyclic, untied, 982,380): sv 0.9694 →
  **dv 0.8533**, 871 ms, process ~3580 MiB. The best @3230 (−0.005 against
  h16, inside the noise), 1st at 3 of 5 buckets. Virtual layers did not pay
  in the 256 regime (tied, at fewer params). Here, untied at full params,
  they tie or win. Promoted to the 4th pool slot.
- Far use @3230 (i+1): k17 h16 × g2 **0.241**, k09 r128 0.231, n03 h16 × V6
  0.225, k16 h16 0.224, n02 h20 0.218, B-wd 0.217, h8 0.210, n01 0.167. So V6
  buys loss, not context. Groups and `r` buy context. Next to test: g4 (n05),
  then V6 × g2.
- **The seed spread of the far use is ~0.017 @3230**: n04 (h16, seed 1)
  0.2405 against k16 (seed 0) 0.2235. The pairing spread (≤ 0.003) is only
  the evaluation noise. So the far-use gaps @3230 above (≤ 0.017) are inside
  the training noise. Decide context use at the later milestones (h16 × g2
  kept +0.016 @9600, one seed each).
- Probe `n-h16g2-V6` (995,318): 3914 MB reserved (3733 MiB, process ~3844
  MiB: over the preferred limit, 17 MiB under the `--max-vram 3750` cap),
  901 ms. Queued as n06.
- **n05-h16g4** (d136 p17, 16 heads, 4 groups, 990,721): sv 1.0188 → **dv
  0.8966**, 748 ms, process ~3280 MiB. Last at every bucket (+0.039 against
  h16). Far use 0.201 (the lowest of the h16 family). The width cost of 4
  groups (d160 → d136) is far too high. Evicted. Groups pay only where they
  are nearly free in width (g2 at d152).
- **n06-h16g2V6** (h16 × g2, L5 × V6 untied, 995,318): sv 0.9787 → **dv
  0.8589**, 880 ms, process ~3844 MiB (over the preferred limit). V6 pays
  −0.006 on g2 too (k17 0.8649). The steepest end slope (−0.011). Far use
  0.222 (in the noise).
- **Pool (2026-10-01)**: a 2 × 2 on h16, {base, g2} × {L5, L5 × V6}: h16,
  h16 × g2, n03 h16 × V6, n06 h16 × g2 × V6. r128 is parked (the weakest at
  9600 on both: dv 0.7590, far use 0.267). At the long milestones the 2 × 2
  separates the context lever (g2) from the loss lever (V6).
- **h16 × g2 @14400** (`p12k-h16g2-lr1-d2`, a fresh decay branch from the
  stable 12000): **dv 0.7327** (h16 0.7281), far use **0.289** (h16 0.284).
  The far-use lead of g2 shrank from +0.016 @9600 to +0.005, inside the
  noise.
- User (2026-10-01): the goal is also **long-context coherence**. Measure
  the far use of every screen and milestone (`ctx.sh` → `ctx.txt`). Groups
  raised it most so far (h16 × g2), so **n05-h16g4** (`n-h16g4-d136`: d136
  p17, 16 heads, 4 groups, 990,721 params) tests whether the trend goes on.
