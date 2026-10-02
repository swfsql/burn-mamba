# notes — tiny-stories ≤1M search, part b (chronological log)

## Start state (from `../1m/handoff.md`)

- Best: `../1m` `s04-nofr-rl4` = **1.2092** decayed valid bits/char, ~9 min
  of training. Config `cfg/d128L8`: d128, 8 real layers (V0), e2 p64 r64,
  untied head, bs32, seq256, Muon + AdamW, LR 4e-3, frontier `Always`,
  `run_len 4`, 600 constant + 200 decay steps. 990,784 params.
- Findings that carry over: the frontier gate was the bottleneck (−0.15 bits),
  rank only after a decay, bs32 beats bs16 per wall clock, LR is shallow,
  `tie=0` always.
- Slowdown bug (burn-bug `issue6.md`): cubecl's metadata cache fills with
  single-use entries when shapes vary per step. The window shapes are now
  fixed (burn-stack's masked loss), so the slowdown may be gone. Check it with
  `--profile`.

## Log

- Setup: `1m-b/` with the scripts of `../1m`. `screen.sh` rewritten: T1 in
  minutes → `--max-seconds`, T2 → a step budget from phase 1's rate.
  `--valid-batches 1` in every screen phase. Binaries rebuilt at `b58a0ab`.
- **b01-s04** (re-baseline): `cfg/d128L8`, screen 600 + 200 steps,
  `-- --profile 100`. Stable 1.3545 → **decayed 1.2042** (s04: 1.3486 →
  1.2092). 6.0 + 2.0 train-min (s04: 8.9). Peak VRAM 2.99 GB (s04: 3.40).
  Profile flat at ~590 ms/step (fwd 47, bwd 347, gap 71, opt 124) from window
  200 to 600 ⇒ **the slowdown is gone**. No segments needed.
- User (2026-09-24): speed is secondary. The logged it/s is an EMA, wrong after
  a full validation (use `summ.py`, which reads `elapsed`). Only SGD replays
  the training step from a captured graph (check its speed later, with its own
  LR). **No 1 h extensions yet**: first an early (~10 min) picture of the
  features: rotation kinds, Kalman, MultiGate, real vs virtual layers, MLP,
  nheads, ngroups, other block knobs.
- Screen protocol from here: **step-matched 600 + 200** (b01 is the baseline),
  ms/step reported beside. Each variant = the ≤1M config that holds the
  feature: the base loses a layer or some `r` to pay for it.
- `gen.py`: new key `trop` (tropical). String-default keys keep `None` as a
  string (`trap=None` wrote `null` before).
- Params (base 990,784): Real1D 957,760 · Kalman 990,848 · KalmanPN 994,976 ·
  mgr0 988,704 · trap=None 986,656 · cls0 990,272 · L4V8 501,856 ·
  L4V8 d176 p88 891,568 · trop r60 990,448 · p32 r60 996,176 · p128 983,024 ·
  L7 Q4D r60 994,692 · L6 Rotor4D r60 993,760 · L7 m2 998,472 · L7 g2 984,136 ·
  L6 u2 r48 977,560 · L6 mlp96 968,272 · r32 905,664. Over 1M: L7 r128
  1,017,512 · L9 r32 1,017,256 · L6 mlp128 1,042,000.
- User: a captured SGD step can be 2–4× faster, but it uses more VRAM. Here:
  tiny-stories does **not** capture its training step yet
  (`tmp/handoff-graph-capture.md`, "Next" 1: the window-0 class splice, the
  host `scored` upload in `lm_output`, the carried caches). So `--sgd` runs
  eagerly now. VRAM room: b01 peaks at 2.99 GB of ~3.6 GB. A capture at bs32
  L8 may not fit. bs16 peaked at 1.99 GB (`../1m` s05).
- **f01-q4d**: `cfg/q4d-L7r60` (Quaternion4D, L7, r60). Stable 1.3464 →
  **decayed 1.2065** (b01 1.2042): a tie at 7/8 of the depth. 714 ms/step
  (fwd 121: the quaternion scan), 1.22× b01. **Peak VRAM 3571 MiB**: at the
  ceiling (no NaNs seen). Rotor4D (two factors) needs a memory check first.
- **f02-real1d**: `cfg/real1d` (Real1D, no rotation; 957,760). Brackets the
  rotation ladder: a tie too ⇒ the rotation does not matter at this horizon.
  Stable 1.3606 → **decayed 1.2089**. 557 ms/step, peak VRAM 2766 MiB.
  ⇒ **Rotation ladder: all three tie** (1.2042 / 1.2065 / 1.2089, ~noise).
  Real1D frees 33k params and ~0.2 GB. Rotor4D goes to the back of the queue.
- **f03-l4v8**: `cfg/L4V8` (4 real layers cycled to 8; 501,856, same compute
  as b01). A tie ⇒ this horizon is compute-bound: spend params on width.
  Stable 1.4087 → **decayed 1.2651** (+0.061 vs b01). 522 ms/step, peak VRAM
  3341 MiB (more than b01). ⇒ **params matter at this horizon**: the weight
  sharing costs 0.06. A feature that costs params must earn them.
- **f04-m2**: `cfg/m2-L7` (MIMO m=2, L7; 998,472). **INVALID: VRAM ~4.5–4.9
  GB for the whole training** (peak 4952 MiB). Face value: stable 1.3482 →
  decayed 1.2132 (b01 1.2042), 1727 ms/step (2.9× b01), 23 min. ⇒ MIMO at
  bs32 is out: over the VRAM budget, ~3× slower, no gain even step-matched.
  Only a bs16 A/B (vs a bs16 baseline) could revisit it.
- `wait.sh` VRAM guard: > 3650 MiB (`VRAM_MAX`) for 3 samples in a row kills
  the screen + ts-cuda, logs "VRAM GUARD" into `vram.txt`. Risky next: u=2
  (2× writes), Rotor4D, r128.
- **f05-mlp**: `cfg/mlp96-L6` (L6 + gated MLP 96 per layer; 968,272).
  Stable 1.3617 → **decayed 1.2074**: a tie. 514 ms/step (0.87× b01), peak
  VRAM 3410 MiB. (Peaks drift: b01 2993, f03 3341, f05 3410.)
  ⇒ At 600 + 200 steps, every same-budget swap ties within ~0.005. Only the
  weight sharing lost.
- **f06-kalman**: `cfg/kalman` (`gain=Kalman`; 990,848). Stable 1.3632 →
  **decayed 1.2130** (+0.009, edge of noise). 720 ms/step (1.22× b01, fwd
  162), peak VRAM 3559 MiB. ⇒ no early gain. It may pay late: not rejected.
- User asked: f03 was at half the params (it answered "same compute, half the
  params", not "virtual at the ceiling"). Virtual at ~1M params: L8V16 990,784
  (2× compute) · L4V8 d184 p92 967,272 · L5V10 d160 p80 932,680 · L6V12 d144
  p72 921,936. VRAM: L8V16 ≈ 5 GB and L4V8 d184 ≈ 4 GB (estimates), over.
- **f07-l5v10**: `cfg/L5V10d160` (5 real layers cycled to 10, d160 p80;
  932,680). **VRAM GUARD kill at 34 s (4600 MiB)**, in the first training
  step. The first step is representative: b01 reaches its steady 2993 MiB at
  41 s and stays flat. ⇒ at bs32, width (d, d_inner) × applications is capped
  near b01's.
- User: keep virtual at 8, lower the reals, spend the params on width /
  bigger states (mnist-class ended with one real layer). The state rank `r`
  is cheap in activation memory (B/C and chunk states only), ~+21k params per
  layer per +64. d128 e2 p64 keeps f03's activations. Params: L4V8 r128
  586,976 · r256 757,216 · **r384 927,456** · r448 1,012,576 · L2V8 r256
  385,072 · r512 555,312 · r1024 895,792 · **r1152 980,912** · L1V8 r1024
  454,360 · r2048 794,840. g2: L4V8 r128 719,072 · L2V8 r256 517,168.
- **f08-l4v8r384**: `cfg/L4V8r384` (f03 + r384; 927,456). vs f03 (1.2651):
  does the state win back the weight sharing? Next rung: L2V8 r1152.
  **VRAM GUARD kill at 35 s (5814 MiB, CUDA OOM panics).** ⇒ `r` is NOT
  cheap in memory: after QK-norm + bias + rotation (and the GQA expansion),
  B/C are per head, `[b, s, h, r]`, several saved per application.
- `probe.sh <cfg>`: 10 steps, peak VRAM (1 s samples, kill > 5400 MiB),
  ms/step of steps 6..10. Results append to `probes.txt`. Calibration: d128L8
  (b01, 2993 MiB) and L4V8 (f03, ~3.1–3.3 GB), then L4V8 r128 / r256, each
  Complex2D and Real1D.
  Peak MiB / ms/step: d128L8 3256 / 596 · L4V8 3256 / 530 · L4V8 r128 C2D
  3832 / 667 · r128 Real1D 3160 / 545 · r256 C2D KILLED (> 5.4 GB) · r256
  Real1D 4422 / 702. (The probe reads ~260 MiB above b01's screen: the host's
  VRAM drifts. `probe.sh` now prints idle and peak − idle.)
  ⇒ The memory grows with **heads × r** (B/C are per head after the
  per-(head, rank) bias). The rotation multiplies it. The state per head is
  `p × r`, so fewer, wider heads grow the state at a fixed `h·r`.
- User: virtual layers may gain from untying the small tensors (mnist-class:
  block `InProjTail DtBias D BNorm CNorm OutNorm`, layer `Norm`). `gen.py`:
  `untie=1` (that list, + `Norm2` with an MLP), `mgrpv=1` (MultiGate per
  virtual layer), `sched=Cyclic|Stretched`. Defaults unchanged (`_check2`).
- Params (Real1D, untie=1 unless noted): L4V8 r128 553,952 (tied) · 561,968 ·
  L4V8 p128 r256 690,400 · **L4V8 p256 r512 956,600** · L3V8 p256 r512
  724,276 · L2V8 p256 r512 491,952 · L2V8 p256 r512 e4 694,744 · L1V8 p256
  r512 e4 362,580.
- Probes: L4V8h2r256, L4V8h1r512, L2V8h1r512e4 (h·r = 512 each). The auto
  chunk at p256 r512 is ≈ √(r·p) → 384 > seq 256: each window pads to 384.
  Total (idle 766) / ms/step: h2r256 3510 / 683 · h1r512 3958 / 1080 ·
  L2 h1r512 e4 KILLED. L4V8h2r256g2 (954,592): 3800 / 745 (g2 adds ~290 MiB,
  so B/C are not per head everywhere).
- User: read the held VRAM from burn/cubecl for a precise account.
  `training.rs` profiler: the `prof` line now also prints `reserved` =
  `Device::memory_pool_usage().bytes_reserved` (all pool pages, the CUDA
  context excluded). `bin/ts-cuda` rebuilt (old: `ts-cuda-v0-b58a0ab`).
  `probe.sh` v2 prints: proc (the process's own peak, `nvidia-smi
  --query-compute-apps`: context + pools, no host drift), reserved, total,
  idle. At idle, the host holds 766 MiB (one foreign compute app 136 MiB, the
  rest graphics).
  v2 results (the proc column read 0: container PIDs ≠ host PIDs; replaced by
  ctx = total − idle − reserved): reserved MB / total MiB / ms/step: d128L8
  **2501** / 3254 / 583 · L4V8h2r256 2767 / 3510 / 679 · L4V8h2r256g2 3059 /
  3800 / 740 · L4V8h1r512 3252 / 3958 / 1064. ctx ≈ 90–120 MiB each.
  ⇒ **cubecl `reserved` is the precise, host-independent measure.**
- User (VRAM rule, new): **the process itself (host excluded) stays under
  3.6 GB**, and the process enforces it. New example flag `-- --max-vram
  <MiB>` (`cli.rs`, `training::Run`): after each backward, it reads the pools'
  `bytes_reserved`, and above the cap it prints "VRAM LIMIT" and exits with
  code 3. The CUDA context (~90–120 MiB) is outside the pools, so the scripts
  pass `--max-vram 3480`. Binary: `bin/ts-cuda` (old: `ts-cuda-v1-reserved`).
- **f09-v8h2r256**: `cfg/L4V8h2r256` (L4V8, Real1D, p128 r256 (2 heads),
  untie=1; 690,400). vs f03 (1.2651) and b01 (1.2042). Three changes vs f03
  at once (state, untie, Real1D): first ask whether the V8 family can compete.
  Stable 1.3953 → **decayed 1.2485** (f03 −0.017, b01 +0.044). 678 ms/step,
  reserved 2767 MB.
- Scripts (after f09): `args.sh` `split_args` splits the args at `--` and adds
  `--max-vram $VRAM_MAX_MIB` (default 3480) unless given. `run.sh`,
  `branch.sh`, `screen.sh` use it. `wait.sh`: the total guard is an OOM guard
  only (5400 MiB), it greps "VRAM LIMIT", prints the last `prof` line.
  Checked: `t01-maxvram` (d128L8, `--max-vram 1000`) stops at the first
  backward: "pools reserve 2385 MiB > 1000".
- **f10-v8h2r256g2**: `cfg/L4V8h2r256g2` (f09 + g2; 954,592, 2917 MiB
  reserved). Do per-head B/C projections close f09's gap?
  Stable 1.4157 → **decayed 1.2594** (f09 +0.011). 734 ms/step, reserved
  3059 MB. ⇒ g2's params do not pay. **V8 family set aside at this horizon**:
  f03 1.2651 (502k) · f09 1.2485 (690k) · f10 1.2594 (955k) vs b01 1.2042.
  Unshared depth wins at 10 min. It may pay at a longer horizon.
- Next axis: real-layer shape at ~1M. Params: d96 p48 L12 891,072 · L13
  (≈964k) · L14 1,037,968 · d112 p56 L10 974,352 · L11 1,070,656 · d160 p80
  L5 932,680 · L6 1,115,984 · d128 L6 e3 1,048,972 · d192 p96 L4 r48
  1,014,528. Probes: d96L13, d160L5, + u2-L6r48, rotor-L6r60, h8-r60.
  Reserved MB / ms/step (cap 3480 MiB ≈ 3649 MB; all fit): d96L13 (964,520)
  3205 / 944 · d160L5 2162 / 406 · u2-L6r48 2477 / 575 · rotor-L6r60 3325 /
  864 · h8-r60 3447 / 847.
- **f11-d96L13**: `cfg/d96L13` (d96 p48 L13; 964,520). Deeper and narrower
  than b01. Step-matched, so its 1.6× step time is not charged.
  Stable 1.4095 → **decayed 1.2529** (+0.049 vs b01). 1018 ms/step (1.7×),
  reserved 3205 MB. ⇒ deeper + narrower loses.
- **f12-d160L5**: `cfg/d160L5` (d160 p80 L5; 932,680). Wider + shallower.
  Stable 1.3361 → **decayed 1.1994** (first < 1.2; −0.005 vs b01, ~noise).
  433 ms/step (0.73× b01), reserved 2162 MB. ⇒ same or better loss, 1.35×
  faster, fewer params: **width wins**. Shape so far: L13 d96 1.2529 · L8
  d128 1.2042 · L5 d160 1.1994.
- Wider params (e2, p = d/2 ⇒ 4 heads, r64): d176 L4 891,568 · d184 L4
  967,272 · d192 L4 r40 998,768 · d208 L3 914,864 · d224 L3 1,049,560 ·
  d256 L2 906,352 · d288 L2 1,130,032. d160 L5 p40 (8 heads) 944,740 · p160
  (2 heads) 926,950.
- **f13-d184L4**: `cfg/d184L4` (d184 p92 L4; 967,272). Stable 1.3427 →
  **decayed 1.2083**. 332 ms/step (0.56× b01), reserved 1871 MB.
  ⇒ Step-matched, L4–L8 is flat (1.2083 / 1.1994 / 1.2042). The step time
  falls with depth, so the shallow shapes win at equal wall clock.
  **New base: d160L5** (best score, 0.73× b01's step).
- **f14-d160L5-s1**: `cfg/d160L5` with `--seed 1`: the noise of the new base
  (only one estimate so far: s04 vs b01, 0.005).
  Stable 1.3360 (seed 0: 1.3361) → **decayed 1.1915** (seed 0: 1.1994).
  ⇒ **Noise ≈ ±0.005–0.008 decayed; base mean ≈ 1.195.** A feature must move
  the decayed score by > ~0.015 to count.
- u=2 on d160 (base cfg `d=160 L=5 p=80`): L5 1,277,220 · L4 1,025,008 ·
  d144 L5 1,057,804 · d136 L5 955,776 · d152 L4 935,032 · d156 L4 979,508.
- **f15-u2**: `cfg/u2-d156L4` (u=2, d156 p78 L4, HorizontalCarryOver;
  979,508). Paid with depth (L4–L5 flat), not width.
  Stable **1.3182** (base 1.3361 / 1.3360) → **decayed 1.1897** (base mean
  1.195). 440 ms/step (= base), reserved 2453 MB. ⇒ mild positive: the
  decayed gain is inside the noise, the stable gain (−0.018) is not.
- User: at u > 1 the trapezoid variant changes the behavior. Params at u2
  d156 L4: Vertical / HorizontalReset 979,508 · VerticalPlusHorizontal
  {CarryOver, Reset} 984,532 · None 974,484. Plan: Vertical, then
  VerticalPlusHorizontalCarryOver (it contains both single-tap members), then
  HorizontalReset / None only if these separate.
- **f16-u2-vert**: `cfg/u2v-d156L4` (f15 with `trap=Vertical`). Stable
  **1.3123** → **decayed 1.1825** (best so far; −0.0125 vs base mean, −0.007
  vs f15). 498 ms/step (1.13× base), reserved 2550 MB. ⇒ u=2 beats the base
  in both phases with both taps. Vertical ≥ HorizontalCarryOver.
- **f17-u2-vhco**: `cfg/u2vh-d156L4` (f15 with
  `trap=VerticalPlusHorizontalCarryOver`; 984,532). Stable 1.3273 →
  **decayed 1.1981** (≈ base). 481 ms/step, reserved 2550 MB.
  ⇒ At u=2: **Vertical 1.1825 < HorizontalCarryOver 1.1897 < two-tap
  1.1981 ≈ base 1.195.** The two-tap mix does worse than either tap (`μ`
  may need longer).
- u ≥ 3 with Vertical: u3 L4 d136 975,608 · u3 L3 d160 979,520 · u4 L4 d120
  952,952 · u4 L3 d144 985,020 · u4 L3 d136 891,320.
- **f18-u4-vert**: `cfg/u4v-d144L3` (u=4, Vertical, d144 p72 L3; 985,020).
  Fallback if over the cap: u3 L4 d136.
  Stable 1.3542 → **decayed 1.2320** (+0.037 vs base mean, +0.050 vs u2
  Vertical). 541 ms/step, reserved 3277 MB. Confounded: L3 + d144 pay for
  the 4× `x` segment.
- **f19-u3-vert**: `cfg/u3v-d136L4` (u=3, Vertical, d136 p68 L4; 975,608).
  The u curve at L4: a real optimum at u=2, or the L3 penalty?
  Stable 1.3310 → **decayed 1.1978** (≈ base). 570 ms/step, reserved 3568
  MB. ⇒ **u dial peaks at u=2** at 1M params: u1 1.195 · u2 1.1825 · u3
  1.1978 · u4 1.2320. Past u=2, the `x·u` segment takes params from width.
- User: the free VRAM allows more expand and/or virtual layers. On u2
  Vertical (f16, 2550 MB): L4V6 / L4V8 979,508 · L4V8 untie 1,036,252 ·
  e3 d128 953,488 · e4 d112 950,912 · e3 d136 1,065,072 · e4 d120
  1,080,008. Probes: u2v-L4V6, u2v-L4V8, u2v-e3d128, u2v-e4d112.
  Reserved MB / ms/step: L4V6 3665 / 708 (over the 3649 MB cap by 16) · L4V8
  KILLED · e3 d128 2622 / 490 · e4 d112 2671 / 500. ⇒ expand is nearly free
  in memory and speed. V6 + Real1D may fit (probe later).
- **f20-u2v-e3**: `cfg/u2v-e3d128` (u2 Vertical, e3, d128 p96 L4; 953,488).
  Stable 1.3173 → **decayed 1.1818** (= f16 1.1825). 476 ms/step, reserved
  2622 MB. ⇒ d_model ↔ expand is neutral. e4 skipped.
- Probes (Real1D): u2v-L4V6-real (939,316) 3253 MB / 643 ms (fits) ·
  u2v-real (V4) 2259 MB / 417 ms (vs Complex2D 2550 / 498).
- **f21-u2v-real**: `cfg/u2v-real` (f16 with Real1D). Does the rotation
  matter at u=2 (2× angle reach)? Also the clean baseline for V6 + Real1D.
  Stable 1.3236 → **decayed 1.1954** (f16 Complex2D: 1.3123 → 1.1825).
  412 ms/step, reserved 2259 MB. ⇒ at u=2 the rotation seems to pay (~0.013,
  just above the noise), as the angle-reach argument predicts. At u=1 it tied.
- **f22-u2v-V6-real**: `cfg/u2v-L4V6-real` (L4 × V6, u2 Vertical, Real1D;
  939,316, 3253 MB). vs f21: does virtual compute help at full params? If
  yes: V6 + Complex2D with `rf=0.5` to fit the cap.
- f22 result: stable 1.3357 → **decayed 1.1922** (f21 V4: 1.3236 → 1.1954).
  598 ms/step (1.45× f21), reserved 3253 MB. ⇒ **extra virtual compute
  does not pay at this horizon** (dv within noise, sv worse).
- User: check Vertical vs `VerticalPlusHorizontalReset` (the carry-over
  two-tap member lost). **f23-u2-vhr**: `cfg/u2vhr-d156L4` (f16's config,
  `trap=VerticalPlusHorizontalReset`; 984,532). Stable 1.3349 → **decayed
  1.1965**. 472 ms/step, reserved 2550 MB. ⇒ trapezoid at u=2: **Vertical
  1.1825** < HorizontalCarryOver 1.1897 < VPH-Reset 1.1965 ≈ VPH-CarryOver
  1.1981. Both two-tap members lose to the single vertical tap here.
- User: extend the best by ~10 min? Stagnated? Aggregate the runs (for "starts
  worse, alive later"). `agg.py [-b N] [runs]` → one ranked row per screen:
  params, ms, reserved, train-bits buckets, slope (last 200 stable steps, per
  100), sv, dv, rank per bucket → dv rank. Output kept in `agg.txt`.
  ⇒ **Not stagnated**: every slope is −0.03 … −0.06 bits/100 steps at step
  600. Ranks settle from @200 on (the @100 rank is init noise). Late climber:
  f18 (u4: train rank 21 → 16 → 12 → 10 → 10 → 11, dv rank 16). Steepest:
  f13 (d184L4, −0.062, 332 ms/step).
- `extend.sh <src> <new> <N1> <N2>`: copy a screen's stable checkpoint,
  +N1 steps at the same constant LR (--resume), then a decay branch of N2
  (as screen.sh). `agg.py` takes the last `start` as the decay branch.
  Plan (after f22 + the VPHR check): +600 / 200 for f16, f12, f13, f18.
- User: ~4 extending candidates is a good breadth/depth balance. Chained
  after f23 (log `x-chain.log`, result `agg-x.txt`): **x01-f16**, **x02-f12**,
  **x03-f13**, **x04-f18** (each: +600 constant, then 200 decay; 1200 + 200).
  Stable@1200 → **decayed@1400** (short screen dv): x01-f16 1.1823 →
  **1.0986** (1.1825) · x02-f12 1.1914 → 1.1052 (1.1994) · x03-f13 1.2042 →
  1.1189 (1.2083) · x04-f18 1.2275 → 1.1424 (1.2320). ms/step 509 / 437 /
  365 / 508.
  ⇒ **The short screens predicted the order 4/4** (train ranks constant over
  all six 200-step buckets). u4 is not a late bloomer. The u2 lead over u1
  shrank 0.017 → 0.007 (~noise) while u2 costs 1.16× per step: at equal wall
  clock, u1 and u2 are likely tied (dv falls ~0.016/100 steps between 800
  and 1400). Screens stay step-matched for breadth. The final pick needs a
  wall-clock-matched comparison. Best so far: **1.0986** in ~14 train-min.
- Next: LR on f16's config (optimizer before architecture; 4e-3 came from
  d128L8). **f24-u2v-lr8**: `cfg/u2v-lr8` (f16 model, LR 8e-3). Then 2e-3.
  Stable 1.3393 → **decayed 1.1904** (4e-3: 1.3123 → 1.1825). Faster in the
  first 100 steps, slower after. 455 ms/step. ⇒ 4e-3 ≥ 8e-3.
- **f25-u2v-lr2**: `cfg/u2v-lr2` (f16 model, LR 2e-3). Stable 1.3359 →
  **decayed 1.2135** (+0.031). ⇒ **LR curve peaks at 4e-3** (2e-3 1.2135 ·
  4e-3 1.1825 · 8e-3 1.1904). Keep 4e-3.
- **f26-u2v-mgr0**: `cfg/u2v-mgr0` (f16 with the standard residual;
  978,244). Stable 1.3255 → **decayed 1.1838** (= f16 1.1825). 423 ms/step
  (~0.9× f16). ⇒ MultiGate does not pay at 4 layers / this horizon.
  Standard is the better pick per wall clock. Screens keep MultiGate (f16
  stays the comparable base).
- Heads at d156 (d_inner 312): p39 (8 heads) 996,504 · p156 (2 heads)
  971,244. **f27-u2v-h8**: `cfg/u2v-h8`. Stable 1.3294 → **decayed 1.1987**
  (+0.016 vs 4 heads). 617 ms/step (1.3×), reserved 3617 MB (near the cap).
  ⇒ 8 heads lose.
- **f28-u2v-h2**: `cfg/u2v-h2` (2 heads, p156). Stable 1.3211 → **decayed
  1.1925** (+0.010). 389 ms/step (0.8×), reserved 2284 MB. ⇒ 4 heads best
  step-matched, 2 heads close and faster.
- Pattern: cheaper variants tie or lose ~0.01 step-matched but save 10–20%
  of the step each: standard residual 1.1838 / 423 ms · 2 heads 1.1925 /
  389 · Real1D 1.1954 / 412 (f16 1.1825 / ~470–498). A "lean" combination
  may win at equal wall clock. Test after the last breadth knobs.
- **f29-u2v-rf05**: `cfg/u2v-rf05` (f16 with `rf=0.5`; 959,412). Stable
  1.3235 → **decayed 1.1919** (+0.009). 441 ms/step, reserved 2405 MB.
  ⇒ Rotation at u=2: full 1.1825 < half 1.1919 < none 1.1954.
- **f30-u2v-s512**: `cfg/u2v-s512` (f16 with bs16 seq512 runlen2: the same
  chars per step and per story, 2× the BPTT window). Probe: 2478 MB, 457 ms.
  Stable **1.2803** → **decayed 1.1531** (−0.029 vs f16, best by far). 449
  ms/step, reserved 2478 MB. Checked: not an artifact. Both validate the
  same 190,070 chars (valid is capped at `run_len` windows: seq × runlen =
  1024 in both) and carry the state across windows (`epoch_valid`).
  ⇒ **512-char BPTT windows × 16 stories beat 256 × 32** at equal chars/step.
  **Keep seq × runlen = 1024**, or the valid set changes.
- **f31-u2v-s1024**: `cfg/u2v-s1024` (bs8 seq1024 runlen1: one window per
  story, no carried state in training). Probe: 2453 MB, 460 ms.
  Stable **1.2584** → **decayed 1.1351** (−0.018 vs s512, −0.047 vs f16).
  451 ms/step, reserved 2453 MB. ⇒ **the BPTT window is the biggest lever so
  far** (256×32: 1.1825 · 512×16: 1.1531 · 1024×8: 1.1351). New regime:
  bs8 seq1024 runlen1. Past 1024 the valid set changes (valid is capped at
  seq × runlen chars per story): decoupling it is a burn-stack dataset change.
  Architecture ranks came from the 256 regime and may shift here.
- **f32-s1k-trop**: `cfg/s1k-trop` (f31 + `trop=MaxPlus`; 990,804). Stable
  1.2764 → **decayed 1.1410** (+0.006, noise; sv +0.018). 479 ms/step.
  ⇒ no early gain.
- **f33-s1k-u1**: `cfg/s1k-u1` (the u=1 base d160L5 model = f12, in the
  1024 regime). u2 vs u1 again: the regime changed a lot, and u2 costs 1.16×.
  Stable 1.2746 → **decayed 1.1458** (u2 f31: 1.2584 → 1.1351). 387 ms/step
  (f31 451), reserved 2065 MB. ⇒ u2 keeps a step-matched lead (~0.011–0.016)
  in the new regime. u1 is 1.16× faster.
- Breadth summary (step-matched 600 + 200): wins = BPTT 1024×8, u2 Vertical,
  wide-shallow d156–160 L4–5, LR 4e-3. Cheaper ties = standard residual,
  2 heads, rf 0.5, Real1D, e3, MLP. Losses = virtual layers, big state / g2
  at V8, deep-narrow, 8 heads, u3/u4, two-tap trapezoids, Kalman, tropical,
  LR 2e-3/8e-3, MIMO (VRAM). (`agg.txt` mixes regimes and extensions: its
  train-bits ranks do not compare across regimes.)
- **Wall-clock phase** (the goal is loss per training time): `screen.sh …
  12m 3m` in the 1024 regime, chained (`w-chain.log`, `agg-w.txt`):
  **w01-A-u2v** (f31 cfg, 451 ms) · **w02-B-u1** (f33 cfg, 387) ·
  **w03-C-u2lean** (`cfg/s1k-u2lean`: A + mgr0 + 2 heads (p156); 969,980;
  probe 2211 MB / 378 ms) · **w04-D-u1lean** (`cfg/s1k-u1lean`: B + mgr0 +
  2 heads (p160); 925,330; 1871 MB / 314 ms).
  Results (~14.5 train-min each; stable steps + decay steps; sv → **dv**):
  D u1lean 300 ms, 2286 + 600: 1.0260 → **0.9267** · C u2lean 356 ms, 1906
  + 502: 1.0317 → 0.9333 · B u1 386 ms, 1790 + 466: 1.0531 → 0.9468 · A u2v
  448 ms, 1520 + 402: 1.0843 → 0.9848.
  ⇒ **At equal wall clock the fastest per step wins**: the step-matched
  order reverses. At equal steps, A and B tie by step 1500 (train 1.084
  both): the u2 lead fades with steps. D ≈ C (0.007, noise). Best:
  **0.9267 in ~14.5 min** (s04: 1.2092). The constant-LR phase flattens
  (late slopes ≈ 0), and the decay gives ~0.1 bits: long runs need a
  proportional decay (~20%).
- Proposal to the user: 2–3 more wall-clock screens of speed levers on D
  (Real1D, rf 0.5, d184 L4, fewer layers), then ~1 h for the winner + C.
- **User (end of the session): no wall-clock ceiling.** The ceilings are
  training steps, params and VRAM (wall clock counts again only if much
  worse). Keep extending a **pool of 4 candidates** (A, B, C, D above),
  compared at matched steps. Replace a clearly worse one by a promoted
  candidate (it may lag in extensions). Step-matched @1500 (train, 300-step
  buckets): C 1.074 < A 1.084 = B 1.084 < D 1.095. `handoff.md` and
  `ORIENTATION.md` now describe this protocol. The speed-lever proposal above
  is dropped.
- User: mnist-class (1k-param ceiling, `examples/mnist-class/model.rs`) is a
  reference for later, when the coarse screens stop separating. Its choices:
  L1 × V4 `Stretched`, d6 r8 e2 p6 (2 heads), `ngroups = nheads`,
  Quaternion4D with `rope_fraction 0.5`, no proj bias, out-norm, MultiGate 4
  streams, untied small tensors + layer `Norm`, no class latents, no final
  norm, no grad horizon.

## Part c (2026-09-26): packed training

- Binaries: `bin/ts-cuda-v3-4b35cb3` (HEAD `4b35cb3`, packed training) and
  `bin/ts-cuda-v4-chunk` (v3 + `--ssd-path recalc:<L>`: the chunk length in
  folded positions, uncommitted in `examples/tiny-stories/cli.rs`).
  `bin/ts-cuda` = v4. `bin/ts-cuda-v2-eaaedd6` = the old `ts-cuda`.
- Check: v3 validates `w04-D-u1lean-d` again to loss 0.6423 (0.927
  bits/char), the same as before. The checkpoints and the metric carry over.
- `probe.sh` now takes example flags after `--` (and adds `--max-vram`).
  `speed.sh <run> <cfg> <steps> [args]`: a throughput run (fill,
  chars/step, ms/step after step 20, kchars/s, reserved). Lines in
  `speeds.txt`.
- Packed probes (2 rows, each model at its own chunk; `probes.txt`): A
  2×4160 fill 0.928, 514 ms, 2574 MB · B 2×4224 0.904, 423 ms, 2065 MB ·
  C 2×4160 0.919, 411 ms, 2114 MB · D 2×4224 0.888, 329 ms, 1871 MB. The
  alignment is the chunk of the model (A 48, B 96, C 64, D 128 tokens).
  Packed vs unpacked: +9–15% ms/step, +26–29% chars/step, so +11–13%
  chars/s for each. kchars/s: A 15.0 · B 18.1 · C 18.6 · D 22.3.
- **User: packed training only from now on.** The step ceiling stays.
- **p3k-D-u1lean**: D (`w04-D-u1lean`, 2286) +714 packed (2×4224) → 3000,
  then 600 decay: dv **0.888** (0.9267 at 2286 + 600 unpacked). 336
  ms/step. `p3k-C-u2lean`: stopped by the user (it used 4160, not D's
  windowing).
- **User: one packing config for every candidate**, chosen by chars per
  wall-clock second on D's model: runs from zero, 200 batches each,
  `SerialRecalculated` only, the chunk length is a knob. Then the pool (4
  model configs) continues with that packing. `sp01`: stopped at startup.
- Packing sweep (`sp11`–`sp17`, `s1k-u1lean`, 200 batches, one run each;
  rows × width, chunk in tokens → fill, ms/step, **kchars/s**): 2×4224 c128
  0.888, 317, **23.66** · 4×2112 c64 0.871, 340, 21.63 · 4×2176 c128
  0.843, 347, 21.16 · 2×4224 c64 0.920, 372, 20.91 · 2×4352 c256 0.832,
  359, 20.18 · 1×8448 c128 0.908, 414, 18.52 · 8×1024 c128 0.737, 327,
  18.47. Reserved 1871–2065 MB.
  ⇒ **Packing for every candidate: `--pack 4224 --pack-rows 2` + a
  128-token chunk** (`--ssd-path recalc:128` at u1, `recalc:256` at u2).
  The chunk has an optimum at 128. Rows of ~2k are slower per step than
  2×4224 (against the estimate in the handoff) and lose fill.
- M = 3000 chain (`p-chain.log`, `agg-p3k.txt`): `p3k-C2-u2lean` (the
  second attempt of C), `p3k-B-u1`, `p3k-A-u2v`, each +(3000 − its step)
  packed, then 600 decay. D = `p3k-D-u1lean` (the same packing).
  Decayed valid @3600 (train @3000, packed kchars/s, packed steps before
  3000): **B 0.8710** (0.944, 17.8, 1210) · **C 0.8721** (0.948, 18.1,
  1094) · A 0.8838 (0.951, 15.1, 1480) · D 0.8878 (0.961, 22.3, 714).
  (`agg-p3k.txt`: its ms/resMB columns are phase 1 of the first process,
  unpacked. The packed rates are in the chain output.)
  ⇒ Two tiers: B ≈ C, then A ≈ D (+0.012 … +0.017). D is rank 4 in all 10
  train buckets. The packed steps carry ~26% more chars, so the histories
  differ: at 3000, A trained ~20.2M chars, B 19.8M, C 19.6M, D 19.0M. A
  trained the most and still trails. The u2 lead of the 1500 bucket is gone
  (A/C vs B). The lean changes hurt u1 (D vs B) and help u2 (C vs A).
  Constant-phase slopes ≈ 0 (−0.003 … −0.021 per 100); the decay gives
  ~0.08 bits.
- User: power-of-two widths. `sp18` 2×4096 c128: fill 0.888, 7272
  chars/step, 322.9 ms, 22.52 kchars/s · `sp19` 4×2048 c128: 0.826, 6766,
  302.6 ms, 22.36 (sp11 2×4224: 23.66, an earlier sitting). ⇒ No speed-up
  from a power of two. **2×4224 stays.**
- **User: Claude decides** when to evict a candidate (an evicted one can
  return later) and when to start new ones, also when they start behind
  ("the track is infinite"). A 20-min heartbeat (session cron) wakes the
  session. Semi-supervised from here.
- A fresh packed candidate trains all its steps packed. So at equal steps it
  has more chars than a pool member with an unpacked history (+14% vs B at
  3000). Compare such a candidate at **matched chars**: B trained 19.76M
  chars by step 3000 (1790 × 5964 unpacked + 1210 × 7505 packed) = 2630
  packed steps.
- **k01-B-kalman** (promoted into A's slot): `cfg/s1k-u1-kalman` (B +
  `gain=Kalman`; 932,720). Fresh, packed, 2630 + 600 → compare with B's
  0.8710. Stable 0.9563 → **decayed 0.8714** (B 0.9506 → 0.8710 at 3000
  + 600). ~566 ms/step steady (1.34× B), reserved 2259 MB.
  ⇒ A tie at matched chars, with 10% fewer steps. But the histories differ
  (fresh packed vs 1790 unpacked + packed), so this does not isolate the
  Kalman gain. Kalman stays in A's slot.
- **k02-B-fresh**: `cfg/s1k-u1` (B), fresh, packed, 2630 + 600: the control
  for k01, and a check of the matched-chars rule (fresh B vs B 0.8710).
  Stable 0.9578 → **decayed 0.8713**. ~410 ms/step, reserved 2065 MB.
  ⇒ **The matched-chars rule holds** (fresh 0.8713 = mixed 0.8710): chars,
  not steps, set the loss here. ⇒ **Kalman = B** (0.8714 vs 0.8713, the
  same history). Its train buckets equal B's to ±0.003 (last bucket: B
  0.959, Kalman 0.966), at 1.34× the step. It stays one more milestone
  (the "may pay late" test), then goes if it still ties.
- Pool: B (`p3k-B-u1`), C (`p3k-C2-u2lean`), D (`p3k-D-u1lean`), Kalman
  (`k01-B-kalman`, 2630 fresh ≈ 3000 mixed). `k02-B-fresh` is a reference
  twin of B, not a pool member.
- **k03-B-mgr0**: `cfg/s1k-u1-mgr0` (B + standard residual, 4 heads p80;
  931,060). Fresh, packed, 2630 + 600. It splits D's lean pair at u1 (D =
  B + standard residual + 2 heads, −0.017), while the pair helped at u2
  (C > A). A tie gives a cheaper B, and it replaces D if it beats D.
  Stable 1.0000 → **decayed 0.9175** (B 0.8713). Behind from the first
  bucket (train @328 2.233 vs 2.133, @2624 1.010 vs 0.959). ~375 ms/step,
  2041 MB. ⇒ **At u1, the standard residual costs +0.046 at 4 heads**, and
  2 heads buy back ~0.030 of it (D 0.888). D is a trade, not a free lean.
  (The 256-regime tie of f26 was at u2.) Dropped (never in the pool).
- **k04-B-h2**: `cfg/s1k-u1-h2` (u1, MultiGate, 2 heads p160 = D +
  MultiGate; 926,950). The untested corner of the pair. Fresh, packed,
  2630 + 600. If it beats B, the matching u2 test is C + MultiGate.
  Stable 0.9873 → **decayed 0.8988**. ~403 ms/step, 1896 MB.
  ⇒ The u1 2×2 (residual × heads, decayed at ~3000-equivalent + 600):
  MultiGate 4 heads **0.871** (B) · MultiGate 2 heads 0.899 · Standard 4
  heads 0.9175 · Standard 2 heads 0.888 (D, ~0.882 at matched chars). An
  interaction: MultiGate pairs with 4 heads, Standard with 2 heads. B's
  corner is the best. Dropped (never in the pool). No more local knob
  grids around B for now.
- **M = 5000 chain** (`p-chain.log`, `agg-p5k.txt`): the pool D, C, B,
  Kalman, each +2000 packed constant steps from its latest stable run, then
  1000 decay (M/5). Kalman goes 2630 → 4630 (the chars of 5000 mixed).
  Questions: does Kalman pay late, does u2 (C) keep pace with u1 (B), does
  D close its gap?
  Stable → **decayed** (ms/step, this sitting): Kalman 0.8887 → **0.8154**
  (611; fresh 4630 + 1000 ≈ the chars of 6000) · B 0.8874 → **0.8160**
  (452) · C 0.8893 → 0.8209 (432) · D 0.9036 → 0.8312 (345). Reserved as
  before. Gain 3000 → 5000: −0.051 … −0.057 for each.
  ⇒ Kalman = B again (it does not pay late by ~6000) at 1.35× the step. D
  is last in every bucket of both milestones, +0.015 (~+0.012 after its
  char deficit). C +0.005 (edge of the noise). The ranks are stable, and
  the gains come from the length.
  ⇒ **Evicted: Kalman, D** (they can return). Pool: B (`p5k-B-u1`), C
  (`p5k-C-u2lean`) + 2 open slots.
- The constant phase is flat at both milestones, and the decay recovers
  ~0.07. So the next lever is the optimizer, not the architecture. The
  weight decay is 1e-4 (Muon and AdamW): effectively off, never screened.
- **k05-B-wd1**: `cfg/s1k-u1-wd1` (B + `wd=0.1` for both groups,
  decoupled: 4e-4 per step at LR 4e-3). Fresh, packed, 2630 + 600, vs the
  fresh-B control k02 (0.8713). If it wins: an LR step-down from
  `p5k-B-u1` (stable @5000) is next.
- **User: every candidate should end up with weight decay.** Find its value
  (bracket 0.1), then give it to the whole pool.
- k05 result: stable 0.9837 (k02 0.9578) → **decayed 0.8705** (k02
  0.8713): a tie. The decay recovers 0.113 (k02 0.087). ~430 ms/step,
  2065 MB. Train buckets equal to ~1300, then higher in the stable phase
  (@2624 0.986 vs 0.959).
  ⇒ The WSD + wd pattern: wd holds the weight norms, so the effective step
  of the scale-invariant weights stays large (a noisier stable phase, more
  to recover). Without wd the norms grow, an implicit slow LR decay (the
  flat stable phases). The gain of wd should grow with the horizon.
  ⇒ k05 takes an open slot (**B-wd**). **p5k-Bwd-u1**: +2000 → 4630, then
  1000 decay (the chars of M = 5000), vs B 0.8160 / Kalman 0.8154.
  Stable 0.9284 (B 0.8874) → **decayed 0.8156** (B 0.8160). ~415 ms/step.
  The decay recovers 0.113 (B 0.071). ⇒ wd 0.1 ties at both horizons (no
  late gain yet). **wd 0.1 is the standard now** (user directive). B-wd
  replaces B in the pool (B stays as the no-wd reference). C gets wd at
  its next extension. Pool: B-wd (`p5k-Bwd-u1`), C (`p5k-C-u2lean`) + 2
  open slots.
- MultiGate is the one architecture knob with a large measured effect
  (+0.046 over Standard at 4 heads). Its `n_stream` (4) was never varied.
  At L5, n4 keeps `[x, F₀, F₁, F₂]` and mixes the last two layers into
  them. n6 keeps every layer output as a stream: full depth attention, no
  mixing.
- **k06-Bwd-mg6**: `cfg/s1k-u1-wd1-mg6` (B-wd + `n_stream = 6`; 932,690).
  Fresh, packed, 2630 + 600, vs k05 (0.8705).
  Stable 0.9994 → **decayed 0.8877** (+0.017). Train buckets behind k05 by
  0.012–0.015 from ~1200 on. ~400 ms/step. ⇒ Full depth attention with no
  mixing loses: the mixing phase does work. Dropped.
- **k07-Bwd-mg3**: `cfg/s1k-u1-wd1-mg3` (B-wd + `n_stream = 3`: keeps
  `[x, F₀, F₁]`, layers 2–4 mix; 932,675). Fresh, 2630 + 600, vs k05. If
  n3 beats n4, n2 is next.
- **User (protocol):** no limit on M. Each extension stays bounded (no
  multi-hour run), and 4 candidates are always in play (never one candidate
  extended alone). Some effects may show only by M ≈ 10000: read the
  trends with `agg.py` (buckets, slopes, ranks). Claude decides fresh
  candidates and evictions. **Do not ask the user questions that block
  progress.**
- Plan: rebuild the pool on **fresh packed histories with wd 0.1**, so
  `agg.py` compares every member step by step (a mixed history shifts the
  buckets by ~370–470 steps of chars). Members: B-wd (`p5k-Bwd-u1`, stable
  4630), C-wd (fresh), and two more (k07 if it holds, then new ideas).
  Milestones in fresh steps: 2630 + 600, 4630 + 1000, then 8000 + 1600,
  … Each extension ≤ ~40 min.
- k07 result: stable 0.9969 → **decayed 0.8812** (+0.011 vs k05). ⇒
  `n_stream`: n3 0.8812 · **n4 0.8705** · n6 0.8877. n4 is a local optimum.
  B's corner is locally optimal on residual × heads × streams. Dropped.
  (Note: wd raises the stable-phase train loss, so `agg.py` buckets compare
  only between runs with the same wd.)
- New pool (fresh, packed): **B-wd** (`p5k-Bwd-u1`, stable 4630), **C-wd**
  (fresh), **B-nowd** (`k02`, fresh; the long-horizon test of wd for a
  milestone or two), and a 4th for diversity: a larger state.
- Chain `k-chain.log` (3 independent runs): **k08-Cwd**
  (`cfg/s1k-u2lean-wd1`, C + wd 0.1; 969,980; `recalc:256`) 2630 + 600 ·
  **k09-Bwd-r128** (`cfg/s1k-u1-wd1-d152r128`: d152 p76 r128, MultiGate,
  wd 0.1; 975,372; 1.9× B's state per layer) 2630 + 600, vs k05 0.8705 ·
  **p5k-B-fresh** (k02 → 4630 + 1000).
  Results (stable → decayed, ms/step, reserved): **k09 r128** 0.9798 →
  **0.8680** (480, 2623 MB) · k05 B-wd 0.9837 → 0.8705 · k08 C-wd 0.9827
  → 0.8813 (387, 2211 MB) · **p5k-B-fresh** @5630 0.8831 → **0.8138**
  (B-wd @5630: 0.8156).
  ⇒ **The larger state is the first lever that moves**: dv only −0.0025
  (noise), but the train buckets pull ahead of k05 from @1200 (@1600 1.053
  vs 1.068, @2400 0.989 vs 1.001; the same wd). A growing gap, at 1.12×
  the step. ⇒ wd still ties on B at 5630. ⇒ C-wd +0.011 behind B-wd: the
  fastest early (best @800/@1200), then it falls back. Kept one more
  milestone (the u2 line), evicted if it stays behind at 4630.
- Pool: B-wd (4630), B-nowd (`p5k-B-fresh`, 4630), r128 (`k09`, 2630),
  C-wd (`k08`, 2630).
- State-size options < 1M (state per layer = 2d·r): d140 L5 r192 964,610 ·
  d128 L5 r256 943,288 · **d152 L4 r256 984,328** (2× k09 per layer, 1.6×
  in total, k09's width). L4 ≈ L5 step-matched (earlier screens), so the
  lost layer should cost little. **k10-Bwd-L4r256**:
  `cfg/s1k-u1-wd1-d152L4r256`, after a memory probe, 2630 + 600, vs k09 /
  k05.
  Probe 2962 MB, 526 ms. Stable 1.0178 → **decayed 0.9014** (+0.033 vs
  k09). Behind from the first bucket. 514 ms/step. ⇒ The state trend does
  not continue in this shape: the 4th layer matters more than in the old
  screens, or r256 is past the optimum (one run cannot tell). Dropped. A
  middle point (d140 L5 r192, 964,610) is the next breadth screen if r128's
  lead holds.
- **p5k chain 2** (`k-chain.log`): **p5k-r128** (k09 → 4630 + 1000), then
  **p5k-Cwd** (k08 → 4630 + 1000). Questions: does r128's bucket lead hold,
  does C-wd stay behind (then evict)?
  Decayed @5630 (fresh; ms/step, reserved): **r128 0.8113** (491, 2521 MB)
  · B-nowd 0.8138 · B-wd 0.8156 (414) · C-wd 0.8310 (393). Train buckets
  (wd runs) @2000 / @3000 / @4500: r128 1.018 / 0.962 / 0.924 · B-wd 1.030
  / 0.967 / 0.927 · C-wd 1.025 / 0.974 / 0.939.
  ⇒ r128 leads, but its bucket lead over B-wd shrinks (0.012 @1500 → 0.003
  @4500): the larger state buys faster early learning so far. ⇒ **C-wd
  evicted** (+0.015, behind in every bucket from @2500; the u2 line lost
  twice). ⇒ wd still does not pay on B. B-nowd stays to M = 8000 (the
  long-horizon wd test), then goes.
- Pool: r128-wd (`p5k-r128`), B-wd (`p5k-Bwd-u1`), B-nowd (`p5k-B-fresh`),
  all at stable 4630, + 1 open slot.
- **k11-Bwd-r192**: `cfg/s1k-u1-wd1-d140r192` (d140 L5 p70 r192, wd 0.1;
  964,610). Fresh 2630 + 600, vs k09 0.8680 / k05 0.8705. It separates
  k10's loss (the lost layer or r256) and tests the state trend at L5.
- Then the M = 8000 chain, with a paired **LR step-down** branch (r128 at
  2e-3 from 4630): the optimal LR usually falls with the training length,
  and the stable phases are flat.
- k11 probe: **3593 MB reserved** (3426 MiB of the 3480 cap), 603 ms/step:
  more than k10 (r256 L4, 2962 MB) with a smaller state (the pool size
  classes, p70). **Skipped** (54 MiB of margin: the validation or the
  decay could hit the cap). With k10's loss and r128's shrinking lead, the
  state direction is parked.
- **M = 8000 chain** (`k-chain.log`, `agg-p8k.txt`), stable 4630 → 8000
  (+3370), then 1600 decay: `p8k-r128` (4e-3) · **`p8k-r128-lr2`** (the
  4th member: r128 at `--max-lr 2e-3` from 4630, paired with p8k-r128) ·
  `p8k-Bwd` · `p8k-B-fresh` (B-nowd, its last milestone as the wd
  reference).
  Stable @8000 → **decayed @9600** (ms/step): **r128-lr2 0.8411 →
  0.7641** (489) · B-nowd 0.8354 → 0.7744 (411) · B-wd 0.8936 → 0.7748
  (412) · r128 0.8886 → 0.7754 (491). (`agg-p8k.txt`.)
  ⇒ **The LR step-down pays: −0.011 vs its paired twin** (the same history
  to 4630), the largest single gain since packing. The stable valid drops
  by 0.048: 4e-3 is too high for long constant phases. ⇒ **r128's early
  lead is gone** by 9600 (= B-wd), at 1.19× the step: a larger state buys
  early speed, not a lower loss. ⇒ **wd ties again** (0.7748 vs 0.7744):
  neither helps nor hurts to 9600. It stays the standard.
  ⇒ **Evicted: B-nowd** (the reference is answered), **r128 @4e-3** (the
  paired control is answered). Pool: r128-lr2 (`p8k-r128-lr2`, stable 8000
  at 2e-3), B-wd (`p8k-Bwd`, stable 8000 at 4e-3) + 2 open slots.
- **p8k-Bwd-lr2**: B-wd at `--max-lr 2e-3` from `p5k-Bwd-u1` (4630) →
  8000 + 1600. It replicates the step-down on B (paired with p8k-Bwd) and
  compares B with r128 under the better schedule (the same history as
  r128-lr2).
  Stable 0.8403 → **decayed 0.7641** (= r128-lr2 0.7641; B-wd @4e-3
  0.7748). 410 ms/step.
  ⇒ **The step-down replicates** (−0.0107 on B, −0.0113 on r128). ⇒ **B =
  r128 under the better schedule**, and B is 1.19× faster: B-wd-lr2 is the
  pool reference. ⇒ Evicted: B-wd @4e-3 (the paired job is done). Pool:
  B-lr2 (`p8k-Bwd-lr2`), r128-lr2 (`p8k-r128-lr2`), both stable 8000 at
  2e-3, + open slots.
- **p8k-Bwd-lr1**: B-wd at `--max-lr 1e-3` from `p5k-Bwd-u1` (4630) →
  8000 + 1600, paired with B-lr2: the level of the step-down. If 1e-3
  loses, 2e-3 is about right, and M = 12000 tests a halving at 8000.
  Stable 0.8019 → **decayed 0.7607**. ⇒ The step-down level on B from 4630
  (stable @8000 → decayed @9600): 4e-3 0.8936 → 0.7748 · 2e-3 0.8403 →
  0.7641 · **1e-3 0.8019 → 0.7607**. The gains shrink (−0.0107, −0.0034):
  1e-3 is near the optimum for this phase. The stable LR should keep
  falling with the steps (a slow decay, in effect).
- The 4 candidates are **model configs** (user, earlier). The LR schedule
  is a shared setting, like packing and wd: **4e-3 until 4630, then
  1e-3**. Further halvings are tested as pairs on B inside the milestones.
  Pool (4 model configs on that schedule): **B** (`p8k-Bwd-lr1`, stable
  8000) · **r128** (catch-up: `p5k-r128` → 1e-3 → 8000 + 1600) · **C-wd
  returns** (evicted at 4e-3 before any step-down; the fastest early
  learner, its late behaviour under a low LR is untested; catch-up from
  `p5k-Cwd`) · **B + MIMO m2** (Mamba-3's quality lever, untested in this
  regime; params + memory probe first). r128-lr2 is superseded by the
  catch-up.
- Chain `k-chain.log`: `p8k-r128-lr1`, `p8k-Cwd-lr1` (each: +3370 at
  `--max-lr 1e-3`, then 1600 decay).
- Catch-up results (decayed @9600): **r128-lr1 0.7590** (491 ms) · B-lr1
  0.7607 · **C-wd-lr1 0.7805** (405 ms): behind from @3000, ~0.017 @8000.
  ⇒ The low LR does not rescue u2. **C-wd evicted again.** Pool: r128, B
  (stable 8000 at 1e-3) + 2 open slots.
- Flex inference of `p8k-r128-lr1-d` (user request):
  `infer-r128-lr1-9600/` (3 unprompted at T 0.5/0.8/1.0, 3 prompted; 800
  chars each; `BURN_DEVICE=flex bin/ts-cuda --inference --no-graph`, 9 s).
- Screen chain `k-chain.log`: **k12-Bwd-m2** (`cfg/s1k-u1-wd1-d152m2`: d152
  L5 p76 r64 MIMO m2; 959,372; memory probe first) · **k13-Bwd-L6**
  (`cfg/s1k-u1-wd1-d144L6`: d144 L6 p72; 921,936; k10's L4 lost badly, so
  depth may matter more in this regime). Fresh 2630 + 600, vs k05 0.8705 /
  k09 0.8680.
- **k12 (MIMO m2) skipped**: its probe hit the cap at the first step (the
  pools reserve 3634 MiB > 3480). MIMO is out at this packing (a smaller
  chunk would shrink the fused axis, but it changes the story alignment,
  so the chars per step).
- **User request: a longer quaternion run** ("it trails at the start, maybe
  it catches up by ~2000–4000"). p8k-r128's shape with mnist-class-like
  settings: Quaternion4D, `ngroups = 2`, `rope_fraction = 0.5`, reduced
  when over 1M params or the VRAM cap, for example with virtual layers
  and untied tensors. Use the budget (user: 830k "is quite low").
  Params: d152 L5 r128 = 1,269,132 (over) · d136 L5 1,071,316 (over) ·
  L3 × V5 untied: d152 831,028, d168 966,156 · **d144 L4 × V5 untied
  967,612** (chosen: 4 real layers, one reused, the closest to L5; the
  width −5%). `cfg/s1k-q4d-g2-d144L4V5`: Stretched, the mnist-class untied
  list (InProjTail, DtBias, D, BNorm, CNorm, OutNorm) + layer Norm,
  MultiGate n4, r128, wd 0.1. Queued after k13: probe → **k14-q4d**
  (2630 + 600) → **p5k-q4d** (→ 4630 + 1000). Then 1e-3 from 4630 as the
  pool.
  **User: L3 × V6 Stretched instead** (L4 × V5 reuses only one layer; 3
  untied real layers leave more params). L3 × V6: d160 930,788 · **d164
  965,680** · d168 1,001,148 (over). `cfg/s1k-q4d-g2-d164L3V6` (d164 p82,
  each real layer applied twice). It replaces the d144 L4 × V5 queue (the
  queue was stopped before it ran).
- k13 result: probe 2356 MB, 478 ms. Stable 1.0014 → **decayed 0.8829**
  (+0.012 vs k05), 501 ms/step. ⇒ With k10 (L4): **d160 L5 is the sweet
  spot** of depth against width. Dropped.
- Pool: B, r128 (8000 at 1e-3), q4d (running), and a 4th: **Kalman**, a
  different mechanism that tied B at 3230 and 5630 ("may pay late").
  **p8k-K-lr1**: `p5k-K-kalman` (fresh, 4630, **no wd**) → `--max-lr
  1e-3` → 8000 + 1600. No wd saves the ~45 min of a fresh wd run. wd tied
  at every horizon, and its effect is smaller at 1e-3: a minor confound.
  Queued after the q4d chain. Then M = 12000 for the 4, with the paired
  halving (5e-4 from 8000) on B.
- q4d probe: **3543 MB reserved** (3379 MiB, 101 MiB under the 3480 cap),
  **913 ms/step (2.2× B)**. My 3300 MB margin skipped it. But the reserved
  bytes stay flat after step 10 in every run (k10: 2962 in the probe and
  the run), and the validation is forward-only. So it runs as configured
  (the user's shape), with the enforced cap as the guard. The 2.2× step is
  "much slower": noted for the wall-clock rule, but the user asked for this
  longer run. Chain: **k14-q4d** (2630 + 600) → **p5k-q4d** (4630 + 1000)
  → **p8k-K-lr1** (Kalman).
- User: the prompted samples "start differently". Not a bug: the three
  files continue three different built-in prompts (`PROMPTS`), each file =
  prompt + continuation (checked: each continues its own prompt). New
  example flags (uncommitted: `cli.rs`, `inference.rs`, `main.rs`, README
  table): `--prompt <TEXT>` (repeatable, replaces the built-in prompts),
  `--continuations <N>` (per prompt, own seeds; files
  `sample-prompted-<i>-<j>.txt`). The defaults give byte-identical output.
  `bin/ts-cuda-v5-prompt`. The user's two milestone prompts (lily/dog;
  alice, jack and jane), 3 continuations each:
  `infer-r128-lr1-9600-prompts/`. Rerun them on each new best checkpoint.
- User: are the start latents run before the prefill? **Yes.** New
  uncommitted example test `examples/tiny-stories/inference/tests.rs`
  (`the_latents_run_before_the_prompt`, a tiny jittered Mamba-3 net, not
  `model_config`): at prompt lengths 1, 255, 256, 257, the chunked prefill
  = `prime` then `forward` = the one-pass training splice (logits, and one
  `step` after), and shifted latent values move the logits. The opening of
  `generate` is now `inference::open` (the test calls it). Note: `forward`
  with `class: None` still splices the latents (it covers the whole
  sequence in one call). `bin/ts-cuda-v6-open`: inference output
  byte-identical to v5.
- User: the unprompted t0.8 sample is somewhat coherent, the prompted
  continuations are not. The test now also teacher-forces 64 `step`s after
  both openings (4 latents, like the real model) against one `forward` over
  prompt + continuation (the path of the training and valid loss): equal at
  every step. ⇒ **The decode is exact; the gap is the model** (local
  structure learned, plot-level tracking weak). (The one-pass `forward`
  returns the 4 latent rows first.) Keep the two prompts as a qualitative
  bar for each new best checkpoint.
- User showed the samples: unprompted is coherent, the prompted
  continuations are fluent but ignore the prompt. Suspected a forward→step
  cache mismatch. Tests extended: the tiny net now has the search models'
  settings (L5, MultiGate n4, proj bias, out-norm, own LM head, rf 1, SISO
  decode); plus `#[ignore]` **`a_checkpoint_decodes_as_it_scores`**
  (`TS_CHECKPOINT=<run> cargo test --example tiny-stories -- --ignored
  --nocapture`). On `p8k-r128-lr1-d`: the decode after the prefill = one
  `forward` at every step (1e-3 rel.), and on-topic continuations cost
  **0.6527** (lily → cat) / **0.7388** (alice/jane) bits/char on both paths.
  ⇒ The pipeline is exact. The drift is sampling (T 0.8) plus weak
  plot-level memory. Options offered: `--temperature` for the prompted
  samples; a "prompt use" number (continuation cost after its own vs an
  unrelated prompt).
- User: add `--temperature`, try 0.5. Added (uncommitted; the prompt flags
  are now `inference::Prompting`; default 0.8, output byte-identical
  without the flag). `bin/ts-cuda-v7-temp`. The two prompts, 3
  continuations at T 0.5: `infer-r128-lr1-9600-prompts-t0.5/`.
- User: still off at 0.5, so a model limit ("the loss may be too high").
  **Context use** (user: build it, run it on the pool): new example mode
  `--context-use` (`examples/tiny-stories/context.rs`, uncommitted; test
  `context/tests.rs`; the tiny test net moved to `test_net.rs`). Validation
  stories with ≥ 512 chars: `P` = chars 0..256, `C` = 256..512. `C` is
  scored as the valid loss scores (one `forward`, latents in front) after
  its own `P` and after the `P` of story `i + 1`; `use = bits(other) −
  bits(own)`, in two segments of `C`: the join (0..64, local syntax) and
  the far part (64..256, plot-level). Writes `context-use.json`.
  `bin/ts-cuda-v8-ctx`. Run on flex, on copies in `ctx/<run>/` (the app
  rewrites its configs at startup). 244 of 256 stories qualify; ~80 s per
  checkpoint on flex.
  Results (valid · far own → other · **far use** · join use): B-wd @3230
  0.871 · 0.919 → 1.136 · **0.217** · 0.969 | Kalman @5630 0.815 · 0.862 →
  1.101 · 0.239 · 1.058 | B-wd @5630 0.816 · 0.858 → 1.099 · **0.241** ·
  1.055 | C @9600 lr1 0.781 · 0.821 → 1.068 · 0.247 · 1.080 | B-wd @9600
  4e-3 0.775 · 0.812 → 1.071 · 0.259 · 1.115 | B-wd @9600 lr1 0.761 · 0.796
  → 1.057 · **0.261** · 1.136 | r128 @9600 lr1 0.759 · 0.798 → 1.065 ·
  **0.267** · 1.140.
  ⇒ Far use grows slowly with training (+0.044 over ~3× the steps, ~+0.03
  per doubling; 19% → 25% of the far loss). Most of the gains are local:
  the other-prompt far loss drops too (1.136 → 1.057). The LR step-down
  buys local modelling only (use 0.259 → 0.261). Architectures are close
  (r128 +0.006, Kalman = B, u2 lowest). Track far use next to the valid
  loss at each milestone. Noise not estimated yet: add a second pairing
  (i + 2) next time.
- `context_use` now also scores a second pairing (i + 2): the spread of the
  two uses is the noise. `bin/ts-cuda-v9-ctx2`. Runs in `ctx2/`: q4d @3230
  / @5630 vs B-wd and r128 at the same horizons, and the old rotation
  groups (256 regime, 600 + 200): u2 f16 Complex2D / f21 Real1D / f29
  rf 0.5, and L8 b01 Complex2D / f02 Real1D / f01 Quaternion4D (L7).
- p5k-q4d (L3 × V6 d164): decayed @5630 **0.833** (B-wd 0.8156, r128
  0.8113): +0.017, the same gap as at 3230. No catch-up yet.
- **User hypothesis: more heads/groups.** Heads = distinct decay
  timescales per layer (Δ, A per head: B has 4 × 5 = 20). Groups = distinct
  B/C (write keys, read queries), for recall. Params: h8 (d160 p40)
  944,740 · h16 (d160 p20) 969,160 · h4 × g2 d152 947,692 · h8 × g2 d152
  959,282 · h4 × g2 d160 1,035,720 (over). The old "8 heads lose" was at
  u2 in the 256 regime. Queued after the Kalman step-down: **k15-Bwd-h8**,
  **k16-Bwd-h16** (probe, 2630 + 600, then context use); h8 × g2 next if
  heads help.
- **Host OOM** (user report): the flex context use of `p5k-q4d-d` (batch 16
  × 516 positions of the quaternion model) + 7 `params.sh` dry runs (each
  starts the corpus load) + the training process, on the 16 GB host. The
  OS killed the flex job. Then my `pkill -f bin/ts-cuda-v9-ctx2` also
  killed the waiting heads queue and my own shell (`-f` self-matches: use
  `pgrep -x`/exact PIDs only). The Kalman training survived. Fix:
  `context.rs` BATCH 16 → 4 (`bin/ts-cuda-v10-ctx4`). Rules: one host
  job at a time (no dry runs, builds or flex jobs in parallel); the
  context-use chain (`ctx3/`) has a guard that kills its job by PID below
  2 GB of available memory and logs the peak RSS. `ctx/`, `ctx2/`: the
  first runs (ctx2 partial: the killed jobs).
- `ctx3/` results (far use, pairing i+1 / i+2; the pairing spread ≤ 0.002):
  B-wd @3230 0.217 / 0.219 · r128 @3230 **0.231** / 0.229 · B-wd @5630
  0.241 / 0.243 · r128 @5630 **0.251** / 0.251 (far loss vs the other
  prompt: r128 higher, it leans on context more). With the @9600 pair
  (0.261 / 0.267): **r128 > B at every horizon (+0.014, +0.010, +0.006)**.
  Old 256-window regime (800 steps): far use only 0.04–0.065. u2: Complex2D
  0.065 · rf 0.5 0.055 · Real1D 0.050; L8: Complex2D 0.049 · Real1D 0.048
  · Q4D 0.041. ⇒ The long-window win was a context-use win. Peak RSS
  91–124 MiB for all of them.
- **The L3 × V6 q4d models: 8.3 GB RSS in 29 s at batch 4** (the guard
  killed them), against 91 MiB for the old L7 Q4D (g1, rf 1, no virtual
  layers). A ~×80 blow-up in this config (virtual layers? g2? rf 0.5 +
  quat scan?): a library question, and maybe the cause of its 3.5 GB / 2.2×
  GPU step. `--context-batch <N>` added (`bin/ts-cuda-v11-ctxb`); q4d
  measured at batch 1 in `ctx4/`. Next: profile one flex forward, one
  setting at a time.
- **Batch 1: the same 8.5 GB in ~30 s. It is a leak, not a big batch.**
  RSS traces of `--inference` on flex (+ ~150–170 MB/s, linear, also in the
  unprompted `prime` + `step` decode, ~2.7 MB per step): q4d L3 × V6 untied
  leaks (18 ms/char) · f22 (L4 × V6 Cyclic, nothing untied, Real1D): flat,
  4 ms/char · leakA (L3 × V6 Stretched **untied**, Complex2D g1 rf 1,
  untrained): leaks, 12.5 ms/char · leakB (the same, **not untied**):
  flat, 4.2 ms/char. ⇒ **The untied-parameter path leaks on flex** (and
  is ~3× slower). `burn_stack::utils::untied::view` clones the whole layer
  and re-wraps each untied param (`Param::initialized(id, narrow)`) on
  every application of every call. Views should drop after use, so
  something retains them (burn `Param` bookkeeping or the flex allocator?).
  Not chased further: a burn-bug-session matter (minimal repro: a tiny
  untied virtual net, a loop of `step`s, RSS). Workaround: the q4d
  context use runs on the GPU (training there showed flat memory), under a
  GPU-total guard (4000 MiB) and the host guard, before the heads screens
  (queue → `/tmp/queue2.out`, `ctx5/`).
- **p8k-K-lr1** (Kalman, no wd, 1e-3 from 4630): stable 0.7988 → **decayed
  0.7791** (B-lr1 0.7607, r128-lr1 0.7590). Best train buckets throughout
  (no wd lowers the stable loss), but its decay recovers only 0.020 (B-lr1
  0.041). ⇒ The no-wd confound is **not** minor under the step-down: the
  norms grew during the 4e-3 phase, so the 1e-3 phase is partly "decayed"
  in effect already. The run cannot separate Kalman from wd. **Kalman
  evicted** (it tied B at 3230 and 5630; 1.35–2.4× the step) rather than
  paying ~40 min for a no-wd B control.
- q4d @5630 (agg): buckets 3rd of 3 at every bucket, +0.017 / +0.022 vs
  B-wd / r128; 993 ms/step. No catch-up by 4630. **Parked** until its GPU
  context use arrives (and until the untied path is fixed).
- Pool: B (`p8k-Bwd-lr1`), r128 (`p8k-r128-lr1`), both stable 8000 at 1e-3,
  + open slots (the h8/h16 screens are next).
- q4d context use on the GPU: the guard killed it (GPU total 4.1–4.3 GB in
  30 s, host 270 MiB). ⇒ The untied leak is in the no-autodiff path on
  **any** device (training showed flat memory: the backward may consume
  what the views hold; a clue for the burn-bug session). Workaround: 16
  stories on flex (`--valid-stories 16`, 14 qualify; `ctx6/`), @5630, far
  use i+1 / i+2: B-wd 0.259 / 0.254 · **r128 0.285 / 0.276** · q4d 0.261 /
  0.250 (4.9 GB peak in 13 s). ⇒ q4d ≈ B in context use, below r128; its
  far loss is the worst (0.893). The quaternion adds nothing measurable so
  far. Stays parked.
- **Heads screens** (fresh 2630 + 600, wd; far use i+1 / i+2): **k16-Bwd-h16**
  (d160 p20; 969,160) stable 0.9681 → **decayed 0.8579** (best screen so
  far; −0.013 vs B-wd 0.8705, −0.010 vs r128 0.8680), far use 0.224 /
  0.228, 742 ms/step (1.73× B), **reserved 3447 MB (at the cap)** ·
  k15-Bwd-h8 (d160 p40; 944,740) 0.9893 → 0.8772 (+0.007, edge of noise),
  far use 0.210 / 0.212, 520 ms, 2453 MB.
  ⇒ **The user's heads hypothesis holds at 16 heads** (80 decay
  timescales). Not monotone (h8 ≥ B): the extension checks h16's lead.
  No room for more (h32, h16 + r128 exceed the VRAM cap). h8 dropped.
- Pool: B, r128 (stable 8000 at 1e-3), **h16** (2630). Chain: **p5k-h16**
  (k16 → 4630 + 1000), then **p8k-h16-lr1** (1e-3 from 4630 → 8000 +
  1600), context use after each (`ctx5/`).
  Decayed: **h16 0.8014 @5630** (r128 0.8113, B 0.8156) · **0.7493 @9600**
  (r128 0.7590, B 0.7607): **new best**, ahead at every milestone
  (−0.013 / −0.015 / −0.011 vs B), train buckets rank 1 almost everywhere
  (@8000 0.780 vs 0.792 / 0.794). Far use (i+1 / i+2): @5630 0.253 / 0.259
  (r128 0.251, B 0.241) · @9600 0.269 / 0.272 (r128 0.267, B 0.261). ⇒
  h16 wins mostly on the loss itself (far own 0.789 vs B 0.796); its
  context use equals r128's. 748 ms/step (1.8× B), reserved 3443 MB.
  ⇒ **B evicted** (clearly behind at every milestone; the reference, it can
  return). Pool: **h16**, **r128**, **h16 × g2** (the groups half of the
  user's hypothesis; `cfg/s1k-u1-wd1-h16g2`: d152 p19 g2, 982,747), and
  the LR halving test moves onto h16 (5e-4 vs 1e-3 from 8000, at M =
  12000).
- **k17-h16g2**: probe, then 2630 + 600, then context use (vs k16 0.8579).
  Probe 3496 MB (3334 MiB: over my 3450 MB margin, under the 3480 MiB
  cap; run with the enforced cap). Stable 0.9800 → **decayed 0.8649**
  (h16 0.8579, r128 0.8680, B 0.8705), 771 ms/step. **Far use 0.241 /
  0.240: the highest at 3230** (h16 0.224 / 0.228, r128 0.231 / 0.229, B
  0.217 / 0.219); far own 0.915 (h16 0.905).
  ⇒ Heads lower the loss; groups raise the context use (distinct write keys
  and read queries, as the recall papers say), at some local-modelling cost
  (d152). Context use is what the samples lack: h16 × g2 joins the pool.
  Catch-up chain: **p5k-h16g2** (4630 + 1000), **p8k-h16g2-lr1** (1e-3 →
  8000 + 1600), context use after each. Then M = 12000 for h16, h16 × g2,
  r128, and the halving pair on h16.
  Decayed: h16 × g2 **0.8083 @5630** (h16 0.8014, r128 0.8113) · **0.7552
  @9600** (h16 0.7493, r128 0.7590); 777 ms/step, 3491 MB. Far use
  (i+1 / i+2): @5630 **0.261 / 0.259** · @9600 **0.285 / 0.283** (h16 0.269
  / 0.272, r128 0.267, B 0.261). ⇒ **h16 × g2 trades a steady +0.006–0.007
  of loss for the highest context use at every milestone** (+0.015 over
  h16 @9600, beyond the pairing noise ≤ 0.003). Groups are the lever for
  coherence; heads for the loss.
- **M = 12000 chain** (`k-chain.log`, `/tmp/p12k.out`), stable 8000 →
  12000, then 2400 decay: **p12k-h16-lr05** (`--max-lr 5e-4` from 8000:
  the halving) · **p12k-h16-lr1** · **p12k-h16g2-lr1** · **p12k-r128-lr1**.
  Context use on flex after each GPU run (sequential: no overlap).
- Side effect: the flex jobs slow a running GPU training. Its host side
  needs the CPU (the Kalman decay ran at ~1.55 s/step against ~0.6).
  Wall clock is no ceiling, but run flex evaluations between GPU runs when
  possible.
- **User: the LR per milestone tells what a single run from zero needs**
  (a cosine anneal). **Only when the search ends** (a settled config, user):
  fit a single-run schedule to the step-down points, then run it from zero.
  Not during the search. Until then, keep collecting the LR points in the
  milestones.
- **M = 12000 results** (stable @12000 → decayed @14400; far use i+1 / i+2):
  **p12k-h16-lr1** 0.7804 → **0.7281** (0.284 / 0.285) · p12k-h16-lr05
  (5e-4 from 8000) 0.7538 → 0.7283 (0.284 / 0.285). ⇒ The halving at 8000
  ties: 1e-3 holds to 12000 (the stable valid is lower at 5e-4, the decay
  closes the gap). **Best: 0.7281.** p12k-h16g2-lr1: stable 0.786 @12000
  (saved); its decay was stopped at 12078 (unsaved). p12k-r128-lr1: not run.
- **Wrap-up (user, 2026-09-27):** everything stopped (the chain, the
  training, the heartbeat cron). `bin/ts-cuda` = `bin/ts-cuda-v11-ctxb`.
  `handoff.md` and `ORIENTATION.md` rewritten for the next session. The
  doc overview of the uncommitted example code is in the repo root
  `tmp.md`.
