# tmp: the notes of the running session (2026-10-08)

## Setup

- The goal (user): the lowest valid bits/char of the cased text with an
  H-Net of Mamba-3 stacks, under the constraints of `tiny-stories/tmp/1m-*`
  (< 1M params, the process under 3.6 GiB, 3.8 GiB for the search). Far use:
  not ranked (no `--context-use` in hnet-stories yet).
- Runs: a sequence of extensions of < ~1 h each (user), so that the partial
  results can stop a run. Library changes in burn-stack are allowed.
- Protocol: unpacked (an H-Net takes no packed rows), seq 1024, run_len 1,
  frontier Always. The valid metric is that of tiny-stories (256 stories,
  their first 1024 chars). bs 10 ≈ 7450 scored chars/step (the packed
  2×4224 of tiny-stories: ~7.5k). The fill is 0.73. 12% of the stories are
  longer than 1024 and lose their tail.
- The port: `--profile`, `--max-vram`, `--grad-checkpointing` (burn-stack
  `examples::{prof, vram}`, `cli::StepFlags`), `--ssd-path`,
  `--main-ssd-path`, `--continuations`.
- The flat reference: `tiny-stories/tmp/1m-cased/runs/h16-s2` (h16, packed,
  screen 2630 + 600).
- Mamba-3 in every stack (user). The Mamba-2-like stage of the reference is
  Mamba-3 with `rot=Real1D` (a per-stage knob for the block search).
- The reference (paper): E/D = 4 Mamba-2 layers each (M4-M4), no MLP in
  E/D, MLP in the main network; α = 0.03; N = 6 (1 stage), (3, 3) (2
  stages); λ ≈ 2 / 1.5 / 1.
- `gen.py` default `cmult=64`: the inner length rounds up to a multiple of
  64 (few distinct shapes, few JIT compilations).
- **The flat reference** h16 (packed 2×4224, recalc:128,
  `v0.67.0-4930bbc-dirty-eb298dd6`): sv@2630 0.9931, **dv@3230 0.8798**
  (primary 0.8666 + case 0.0131). 1.39 steps/s (a build and the tests ran
  beside it), 3149 MB reserved.
- Probes (10 steps, bs 10, `v0.67.0-4930bbc-dirty-a9c7869b`): a2 (stage
  d96 × 2+2 r32, main d176 × 3 r64, 995k) reserves 1924 MB (default SSD
  chunk 32), 2179 MB (recalc 128/64), 2700 MB (256/128); a4 (d112, main
  d160 × 3, 985k) 2273 MB at 128/64. Steady ~0.7 s/step between the JIT
  spikes (~5 s per new inner shape). The 10-step ms/step is JIT, not speed.
  The default chunk (32) is the fastest and the smallest.
- **LR screens (600 + 200), a2, tied head:** lr 4e-3 dv 1.3047, lr 2e-3 dv
  1.3520 (4e-3 leads at every bucket). 663 ms/step, 2056 MB reserved. The
  router holds the target: L1/L0 0.25–0.28, mean p ~0.44 (α 0.3, N 4).
  `runs/a2-lr8` is a stopped start (killed at 30 s: low value, see below).
- **The tied head at init**: valid@0 117 bits/char, train ~36 bits/char
  over the first 100 steps. The tied head reads the N(0, 1) embedding, the
  residual stack passes the input embedding, so the init predicts the
  *current* character with a logit ≈ |e|² ≈ d. The flat h16 has an untied
  head (valid@0 6.7). Next: a2 × untied head (`tie=0`, 999,723 params).
- **a2 × untied head** (a2t, lr 4e-3): dv 1.2686 (tied 1.3047, −0.036),
  valid@0 6.37. The base from here: `tie=0`. Peak total VRAM 3948 MiB
  (host included).
- **a2t × lr 8e-3**: dv 1.2526 (lr 4e-3: 1.2686, −0.016). lr 4e-3 leads
  the stable train buckets 400–500, lr 8e-3 gains more in the decay.
  a2t × lr 1.6e-2 diverges (train ~3.3, dv 2.69). a2t × λ 1/1 (lr 4e-3):
  dv 1.2878 (λ auto 1.2686: the stage multiplier helps). The screen recipe:
  **lr 8e-3, λ auto**.
- **The flat h16 at 600 + 200** (`1m-cased/runs/h16-q`, 969,481 params):
  **dv 1.1314** (sv 1.2592). The best H-Net (a2t × lr 8e-3) is +0.12
  behind, and 0.17–0.2 behind at every train bucket: not only the slow
  start of the router. Hypothesis: the char-level capacity (flat: 5 × d160
  per char. a2t: 4 × d96, and within a chunk the main network adds only
  what it knew at the chunk start).
- Queued (lr 8e-3, λ auto): al-s128 (stage d128 2+2, main d136 × 3, 959k),
  al-e4 (stage d96 4+4, main d160 × 2, 967k), a2t × N 2, a2t × init burn,
  a2t × α 0.03.
- **al-s128** (stage d128 2+2, main d136 × 3, 959k): dv **1.2084** (a2t
  1.2526, −0.044), 742 ms, 2166 MB. **al-e4** (stage d96 4+4, main d160 ×
  2): dv 1.3320 (+0.08), a bump at @300 (2.29 > 1.95: 8 char-level layers
  at the stage LR 2.1e-2 may be unstable), 1003 ms, 2624 MB. The width at
  char level pays, the depth does not (at this LR). Queued: al-s144 (stage
  d144 2+2, main d144 × 2, 979k), al-s160 (stage d160 2+2, main d160 × 1
  r48, 993k: nearly the flat h16 with one chunk-level layer).
- Against a2t × lr 8e-3 (1.2526): **× init burn 1.2210 (−0.032)**, × N 2
  1.2308 (−0.022; 858 ms, 2504 MB), × α 0.03 1.2529 (tie; the router
  still holds 0.28 against 0.25). The reference init (N(0, 0.02) + the
  residual rescale) hurts at this size; the flat h16 has the Burn init.
  The base from here: `init=burn`. Queued: al-s128 × init burn (do the
  gains stack?).
- **The stage width plateaus**: al-s144 (stage d144 2+2, main d144 × 2)
  dv 1.2076 = al-s128 1.2084. al-s160 (stage d160, main d160 × 1 r48)
  1.2377: one main layer loses. Queued on al-s128 × init burn: × Multi-Gate
  4 streams (stages + main, 961k), × N 2.
- **The gains stack**: al-s128 × init burn dv 1.1660 (−0.042); × Multi-Gate
  4 1.1660 (tie, +1.9k params, 778 ms: no gain); **× init burn × N 2
  1.1286** (−0.037 more; 958,776 params, 847 ms, 2652 MB reserved, the
  router at 0.51 against 0.50). **It matches the flat h16 at 600 + 200
  (1.1314).** The base from here, "b": stage d128 2+2, main d136 × 3, init
  burn, untied head, N 2, lr 8e-3, λ auto (1.46 / 1). A card peak of 5447
  MiB (host included) was seen while these ran; their pools peaked at
  ≤ 2652 MB, so the host share spiked (not these processes).
- Queued on b: × N 3, × N 1.5, × lr 5e-3, × lr 1.2e-2, × Real1D stages,
  × 2 stages (b-S2b: stage d112 2+2, stage d128 1+1, main d136 × 2, N 2/2,
  982k).
- **The compression**: b × N 3 dv 1.1404 (782 ms, 2373 MB), b (N 2)
  1.1286, **b × N 1.5 1.1175** (960 ms, 2804 MB; the router at 0.66
  against 0.667). Less compression wins at every step. At a fixed param
  budget, the model wants compute at every char: the compression pays only
  if its savings go somewhere. Queued: b × N 1.25 (does the trend go to the
  degenerate N 1, a flat 7-layer net?), b × N 3 × main virtual 6 (muntie,
  994k: the savings of N 3 spent on a second application of the main
  network, ~the compute of N 1.5).
- `gen.py muntie=1`: the untied list of the main (block + stack), for
  `mV`.
- **The user's hypothesis**: compression may pay later. Early training is
  local (spelling, short grammar), where per-char compute pays. Later the
  range matters (names, phrases), where the main network's state spans N×
  more text per update. The short-screen gap of N 3 to N 1.5 does not
  shrink over buckets 300–600 (+0.035 → +0.041), but 600 steps is ~4.5M
  chars. Queued after the batch: L-N15 and L-N3 (b × N 1.5, b × N 3 at
  2630 + 600, against the flat h16 dv 0.8798). If the gap shrinks, extend
  both in < 1 h steps. Proposed: port `--context-use` (the re-mention
  probe) to hnet-stories, the direct measure of the recall of names.
- The tiny-stories example tests pass after the move of prof/vram (14
  passed, 1 ignored).
- **The LR on b**: lr 5e-3 dv 1.1282, lr 8e-3 1.1286 (tie), lr 1.2e-2
  1.1457 (+0.017, leads only @100–200). The optimum is flat over 5e-3 –
  8e-3. Use 5e-3 for the long runs (the 1m-b lesson: the best constant LR
  falls as the run goes on)? L-N15 / L-N3 keep the 8e-3 of their cfg.
- **b × Real1D stages** (the Mamba-2-like stage, 951k): dv 1.1649 (b
  1.1286, +0.036), 764 ms, 2086 MB. The rotation at char level pays: the
  stages keep Complex2D. **Not budget-matched** (user): Real1D frees 8k
  params and 566 MB, unspent (e.g. a larger stage state rank). Not rerun
  for now. The general point: every screen so far leaves ~0.9–1.1 GB of
  pools unused (the base b: 2652 MB, the search cap 3750 MB, the preferred
  3545 MB). After the long runs: a "fill the VRAM" series on the base
  (state rank, main MIMO, main virtual layers).
- The fill knobs are not free in params here: b has 41k left, the stage r
  64 alone costs +46k, main MIMO 2 +64k. Queued after the long runs (all
  < 1M): f-sr48 (stage r 48, 982k), f-sr64 (stage r 64, main d128, 960k),
  f-mr96 (main r 96, 995k), f-mm2 (main MIMO 2, main d128, 974k), f-mv6
  (main virtual 6, untied, 994k).
- **b × 2 stages** (b-S2b: stage d112 2+2, stage d128 1+1, main d136 × 2,
  N 2/2, 982k): dv 1.1782 (b 1.1286, +0.050), 824 ms, 2144 MB (−508 MB).
  Both routers hold their 0.5. Not budget-matched (the main sees 1/4 of the
  chars, as at N 4); its recall is not probed yet.
- **Virtual layers** (user): they lost in every earlier test. If tested,
  use a schedule that repeats the middle real layers more than the edge
  layers (the first and last layers as the entry and exit). `gen.py
  msched=0,1,1,1,1,2` (burn-stack `Schedule::Custom`). The fill screen
  f-mv6m (main mL 3 → 6 applications, middle-heavy, untied, 994k) replaces
  the Cyclic f-mv6. b-N3-v6 (running) keeps Cyclic: the contrast.
- **b × N 3 × main virtual 6** (Cyclic `0,1,2,0,1,2`, untied, 994k, gc):
  dv 1.1601 (b × N 3 1.1404, +0.020), 977 ms, 2547 MB. The savings of N 3
  spent on a Cyclic second pass of the main network lose. The
  middle-heavy f-mv6m (on b, N 2) is still queued.
- **The budget table** (`knobs.py`, 10-step probes with gc; the base b
  1962 MB, 958,776 params). ΔMB / Δparams from b: stage Real1D −614 /
  −8k; stage head dim 32 −603 / −14k; rope fraction 0.5 −480 / −11k; main
  head dim 34 −455 / −15k; N 3 −206; Trapezoid None −191 / −15k; main MLP
  ~0 / +314k; main expand 3 ~0 / +167k; main Quaternion4D +74 / +322k;
  Multi-Gate 4 +103 / +2k; main micro-steps 2 +184 / +172k; stage expand 3
  +323 / +212k; no gc +360; stage Quaternion4D +463 / +190k; main r 128
  +489 / +73k; main virtual 6 +585 / +35k; stage MIMO 2 +690 / +43k; stage
  r 64 +799 / +46k; main Rotor4D +874 / +658k; main MIMO 4 +1096 / +187k;
  stage Rotor4D +1395 / +388k. **VRAM-heavy, param-cheap** (~16–18 MB per
  kparam): the stage state rank, the stage MIMO, the main virtual layers.
  **Param-heavy, VRAM-free**: the main MLP, the main expand. The head count
  is a large hidden VRAM cost (stage p 32: −603 MB). gc saves 360 MB
  (−15%). Caveats: the step-10 probe reads ~12% low against the steady
  state (b without gc: probe 2322, screen 2652); |ΔMB| < ~150 is noise
  (2 workers: each probe sees other batches, and the inner length follows
  the batch; main MIMO 2 reads −122); the N 3 probe has not reached its
  chunk ratio by step 10.
- **Kaggle as a second runner** (user: Kaggle first, Colab later).
  `kaggle/`: kg.sh (the CLI 1.7.4.5 with the repo's `KGAT_` token: kagglesdk
  sends KAGGLE_API_TOKEN as a Bearer token before it looks at the
  username/key, which only pass the CLI check), kcorpus.sh (the private
  dataset swfsql/hnet-corpus: the cased caches), runner.py (the kernel),
  kpush.sh / kpull.sh, kqueue.txt (the Kaggle queue), kbatches.txt (the
  pushed batches), out/<ts>/ (the fetched outputs). The smoke: Kaggle
  unpacks the payload archive (runner.py takes either form). **T4 × 2**
  (sm_75, driver 580); the k-base probe reserves **1962.3 MB, the same as
  here**; the 100 + 50 smoke screen ran at 756 ms/step. So runner.py runs
  one lane per GPU (dependent steps on one line with `&&`), and probe.sh
  queries its own GPU (`CUDA_VISIBLE_DEVICES`). The first batch (05:00):
  L-N15 and L-N3 (+ their ctx probes) on the two T4s; L-N2 runs here.
- **gc by default** (user): `args.sh` adds `--grad-checkpointing` unless
  `--no-gc`. From b-N3-v6 on, every run has it (b-S2b and earlier: none).
- **The budget table** (user: which knobs target the VRAM): `knobs.sh` /
  `knobs.py`, 24 one-knob variants of b (stage/main state rank, rotation
  kind per stack, rope fraction 0.5, expand 3, main virtual 6, MIMO 2/4,
  head dim, main MLP, Multi-Gate, micro-steps 2, Trapezoid None, N 1.5 /
  3), with gc, + the base without gc. Params (`knobs-params.txt`): base
  958,776. Queued as 10-step probes before the long runs. `gen.py su/mu`:
  micro-steps per stack. k-me3 takes mp 24 and the head-dim variant is mp 34
  (main d136: 408 / 16 and 272 / 32 do not split into heads).
- **`--context-use` in hnet-stories** (the re-mention probe): the measure
  moved to `examples/common/context.rs`, generic over a logits closure (the
  lead rows: the class latents of tiny-stories, the start rows of the
  H-Net). tiny-stories keeps a thin `context.rs` (the load, the closure,
  the tests: both pass). `ctx.sh` / `ctxstat.py` here run `bin/hn-cuda`
  (`v0.67.0-7bc782e-dirty-37fe66fe`). Smoke (flex, 32 stories, b-N15-d,
  38 s): far use 0.102, re-mention letters own 1.96 / none 2.16 / other
  2.26. Queued: `ctx.sh -g -n 2048` of b-N15-d, s128-N2-d, b-N3-d, the
  flat h16-q-d (→ `ctxg.txt`), and of L-N15-d, L-N3-d after their screens.
- **The context probe at 800 steps** (`ctxg.txt`, 1968 stories, paired;
  s.e. of a difference ~0.001 far, ~0.01 first letter). Own far bits: N 1.5
  1.1746, flat h16 1.1770, N 2 1.1804, N 3 1.1996. **The re-mention first
  letter** (the choice of the entity), own: N 1.5 3.890, N 2 **3.763**,
  N 3 3.782, flat 3.774. Its gain over no prompt: N 1.5 0.467, N 2
  **0.697**, N 3 0.625, flat 0.631. **Compression trades local modeling for
  entity recall**: N 1.5 wins the bits and is the worst at the recall
  (+0.11–0.13 against all others). N 2 recalls best, ahead of the flat
  model, at +0.006 far bits. The user's hypothesis has support already at
  800 steps. The long comparison: L-N2 (first), L-N15, L-N3, each with its
  probe. b × N 1.25 dropped (the direction that hurts the recall).
- Committed: burn-stack `c94299e`, burn-mamba `7bc782e`, `03731e8` (the
  probe). A session rewind
  restored the pre-edit `burn-stack/CLAUDE.md` (the File Map entries of
  the commit, put back) and `queue.txt` (b-r1d twice, removed).
