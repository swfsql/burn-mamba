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
- **L-N2** (b, 2630 + 600, gc): sv@2630 1.0934, **dv@3230 0.9248**, 881
  ms, 2086 MB. The flat h16-s2 (packed, lr 4e-3, 3149 MB): sv 0.9931, dv
  **0.8798**. At 600 + 200 they tied (1.1286 / 1.1314). At 3230 the H-Net
  is +0.045 behind. Its decay gains 0.169 (the flat 0.113): the trunk at
  lr 8e-3 (picked on the short screens, 5e-3 tied) is likely too hot for
  a long trunk. Next: L-N2 × lr 4e-3. The recall of L-N2-d (`ctxg.txt`):
  far own 0.9544; the re-mention first letter own 2.925, gain **1.229**
  (0.697 at 800 steps). The flat h16-s2-d probe is queued.
- **The full-budget shapes** (user, again: every shape uses both
  ceilings). `fit.py` fills the params (`mmlp` at `mlpmul=8`). The probes
  (`fills.txt`): F-base 1809 MB, F-mv6m-mr96 3121, F-r1d-sr104 2715.
  F-sr80, F-sr88, F-sm2r48 stopped at the VRAM limit (F-sr80: 4499 MiB):
  at a stage r ≥ 80 the default chunk goes from 32 to 64 (√(r·p) rounded
  up to a multiple of 32), so the intra-chunk memory doubles. The chunk
  length changes no value: `--ssd-path recalc:32` pins it (re-probed).
  The F-base probe reads 1809 (k-base 1962): the probes have a ±150 MB
  noise from the inner length. Re-probes with the chunk pinned at 32:
  F-sr80 4295 MiB (still over), F-sr88 and F-sr72 over, F-sm2r48 over;
  F-sr64 2589, F-r1d-sr104 2740, F-mv6m-mr96 3129. **A Complex2D stage
  above r 64 costs far more than linear** (r 64 2589, r 72 > 3750), while
  Real1D r 104 fits. To investigate (the rotation path?).
- **The recall of the flat model at 3230** (h16-s2-d, paired, 1968
  stories): re-mention first letter own **2.685** (L-N2-d 2.925), gain
  **1.622** (L-N2-d 1.229, −0.39 ± 0.02); far own 0.899 (0.954), far use
  0.217 (0.187). At 800 the H-Net led the gain by +0.07; at 3230 the flat
  leads by 0.39. Without a prompt the H-Net is better (4.154 / 4.307). The
  user's hypothesis (the compression pays later) has no support here. The
  confounds: the LR (8e-3 / 4e-3), the VRAM (2086 / 3149 MB), and the
  window: the flat trains on packed rows of 4224 (whole stories), the
  H-Net on the first 1024 chars of each story.
- **Length buckets** (user: a window of 4096; a few batch-length slots, by
  the longest story; batch similar lengths together). burn-stack
  `dataset.rs` `bucket_batches` / `BucketedStoriesDataset` /
  `BucketedStoriesBatcher`, `lm.rs` `Bucket {slots, chars}` (+
  `--buckets`, `--bucket-chars`, `--no-bucket`), a unit test. Each story
  goes to the shortest slot that holds it; a slot takes `round(chars / its
  mean)` stories, so one shape per slot; the leftovers move up. The
  validation is unchanged (bs 10, the first 1024 chars). `gen.py
  buckets=512,768,1024,1536,2048,3072,4160 bchars=7450`: 512 16 × 844,
  768 11 × 13308, 1024 9 × 7851, 1536 6 × 3352, 2048 4 × 1879, 3072 3 ×
  1198, 4160 2 × 176; fill 0.838 (unbucketed 0.73), 8.2–9.2k positions
  per batch (now 10,240), every story whole (max 4148). The user's three
  slots 1024/2048/4160 would fill 0.679 with up to 12.5k positions. The
  train chars per epoch grow 7.7% (the tails of the 12% long stories).
  The story train-length stats: median 720, p88 1026, p99 2231, max 4148.
- **The long N comparison** (2630 + 600, lr 8e-3; L-N15 and L-N3 on
  Kaggle T4s, 1.14 / 1.37 steps/s): dv@3230 L-N15 **0.9070**, L-N2 0.9248,
  L-N3 0.9482 (flat h16-s2 0.8798). The order of the short screens holds,
  and the gaps grow (N 1.5 − N 2: −0.011 at 800, −0.018 at 3230; N 3 − N 2:
  +0.012, +0.023). The recall at 3230 (paired, 1968 stories): far own N
  1.5 0.934, N 2 0.954, N 3 0.982 (flat 0.899); re-mention letters own
  1.182 / 1.278 / 1.358 (flat 0.950); re-mention first letter own 2.907 /
  2.925 / **2.847** (flat 2.685), gain 1.319 / 1.229 / 1.246 (flat 1.622).
  Less compression wins the bits and most recall rows. Compression still
  helps the entity choice (N 3 has the best own first letter of the
  H-Nets), but not enough to pay. The flat model leads every row.
- **B-base** (b at the buckets): the corpus line matches the simulation
  (28,612 batches, fill 0.838); prof n=100: 1585 ms (JIT included), 1822
  MB. Kaggle (batch 2): L-B-N15 and L-B-N15-lr4 (2630 + 600 + ctx): the
  buckets at the long horizon (against L-N15) and the trunk LR.
- **B-base** (b at the buckets 512…4160, 600 + 200): **dv 1.1504** (b
  1.1286, +0.022), 984 ms, 1822 MB. The buckets lose at the short
  horizon. A likely cause: the metric scores the first 1024 chars, and
  whole stories give fewer openings per step (9.2 stories per batch, not
  10) and spend steps on positions that the metric never scores. The
  window gains of 1m-b (256 → 1024) were inside the scored span. The
  diagnostic B-c1024 (slots 512/768/1024: the old data, only grouped by
  length; 16 × 844, 11 × 13308, 8 × 12780, fill 0.871) separates the tails
  from the grouping. The bucketed probes (10 steps): B-Fbase 1706,
  B-sr64 2245, B-r1d-sr104 2476, B-mv6m-mr96 2767 MB.
- The Kaggle push of L-B-N15 / L-B-N15-lr4 failed: the new version of
  the payload dataset never went live (`lastUpdated` stayed at 05:14,
  `datasets files` listed the old marker). kpush.sh stopped before the
  kernel (the marker check worked). Retried with the full log
  (`/tmp/kpush-retry.log`): the same (the CLI reports no error, Kaggle
  accepts the version and does not process it: 41 min for the first).
  A test dataset `hnet-ping`: a new dataset and a version of it are live
  at once (tiny files). A new dataset with the payload (`hnet-pay-<ts>`)
  stalled too (403 for 10 min). The same 46 MB archive as
  `payload.tgz.bin` (`hnet-ping2`) was live in 20 s: **Kaggle's unpacking
  of a `.tar.gz` stalls** (it ran at 05:14). kpush.sh uploads
  `payload.tgz.bin` into a new dataset per batch; runner.py unpacks it.
- **h16-B** (the flat h16 at the buckets, lr 4e-3, no gc): 816 ms/step,
  **3707 MB** reserved, flat from step 100 (packed h16-s2: 3149 MB). Under
  the 3.8 GiB search ceiling, over the 3.6 GiB preference.
- **h16-B-d** (the flat at the buckets, 2630 + 600): sv 0.9978, **dv
  0.8847** (packed h16-s2: 0.9931 / 0.8798, +0.005). The recall: far own
  0.909 (0.899), re-mention first letter own 2.719 (2.685), gain 1.457
  (1.622). For the flat model the buckets ≈ the packed rows.
- **B-c1024** (b at the buckets 512/768/1024, the first 1024 chars,
  fill 0.871): dv 1.1370 (b 1.1286, +0.008), 804 ms, 1671 MB. So at 800
  steps the grouping by length costs ~0.008 and the tails past 1024 a
  further ~0.013 (B-base 1.1504). The seed noise is not known: the
  repeat s128-N2-s1 (b, seed 1) is queued.
- **The seed noise**: s128-N2-s1 (b, seed 1) dv 1.1326 (seed 0 1.1286):
  ~±0.004 on a 600 + 200 screen. B-c1024 (+0.008) is ~2σ; B-base
  (+0.022) is real.
- **F-sr64** (stage r 64, md 128, mmlp 32, 996,865): dv 1.1282 (F-base
  1.1294, b 1.1286), 1082 ms, 2761 MB. The stage state rank 64 ties.
- **The bucketed long runs** (Kaggle v5, payload 20261009-070847):
  L-B-N15 (lr 8e-3) sv 1.0965, **dv 0.9272**; L-B-N15-lr4 sv **1.0301**,
  **dv 0.9061**. Against L-N15 (the first 1024 chars, lr 8e-3, 0.9070):
  the buckets lose 0.020 at the same LR also at 3230. The trunk LR: 4e-3
  beats 8e-3 by 0.066 at 2630 and 0.021 after the decay (the decay gain
  0.124 against 0.169). Matched with the flat (h16-B, the same buckets,
  lr 4e-3, 0.8847): the H-Net is +0.021 behind. The recall: far own
  L-B-N15-lr4 0.934 (h16-B 0.909); re-mention letters own 1.223 (0.991).
  **Decision**: the protocol stays the first 1024 chars (the metric's
  span); the trunk LR goes down. Kaggle batch 3: L-N15-lr4 and
  L-N15-lr25 (2630 + 600 + ctx).
- **F-r1d-sr104** (Real1D stages at r 104, recalc:32, md 128, mmlp 8,
  999,425): **dv 1.1076**, the best screen (F-base 1.1294: −0.022, ~5σ;
  b × N 1.5 1.1175), 1058 ms, 2916 MB. The budget-matched Real1D test
  reverses b-r1d (Real1D r 32, unspent: 1.1649). The stage state rank
  pays once it is large: Complex2D r 64 ties (F-sr64 1.1282), and
  Complex2D above r 64 does not fit (the anomaly), while Real1D r 104
  fits at 2916 MB. So the anomaly now matters.
- **F-mv6m-mr96** (main virtual 6 middle-heavy + main r 96): dv 1.1395
  (+0.010), 1272 ms, 3763 MB. The main virtual layers lose again.
- **The long LR** (N 1.5, the first 1024 chars, Kaggle v6): L-N15-lr4 sv
  1.0203, **dv 0.8915**; L-N15-lr25 sv 1.0052, dv 0.8960 (lr 8e-3: sv
  1.0676, dv 0.9070). **lr 4e-3 is the LR of the long runs** (2.5e-3 has
  the better trunk, a smaller decay gain). The gap to the flat h16-s2
  (0.8798): +0.012 (at lr 8e-3: +0.027).
- **L-N2-lr4** (b, lr 4e-3, 2630 + 600): sv 1.029, **dv 0.8927** (L-N15-lr4
  0.8915: a tie; at lr 8e-3 N 1.5 led by 0.018). The recall (paired, 1968
  stories): far use N 2 **0.217** / N 1.5 0.203 (flat h16-s2 0.217);
  re-mention letters own **1.070** / 1.135 (flat 0.950). At the right LR
  N 2 ties the bits and recalls better: the hot LR hurt the compression
  more. The user's hypothesis gets part of its support back.
- **R-p32** (Real1D stages, head dim 32, r 128, md 128, 997,281): dv
  1.1136 (F-r1d-sr104 1.1076: +0.006, ~1.5σ), 819 ms, 1774 MB.
- Queued here: F-r1d-sr64 (Real1D r 64 + mmlp 48: the rank or the kind?)
  and R-trapN (Trapezoid None, r 124).
- **L-R-lr4** (F-r1d-sr104 at lr 4e-3, 2630 + 600, Kaggle v7): **dv@3230
  0.8617**. **The H-Net beats the flat model** (h16-s2 0.8798: −0.018;
  the best Complex2D H-Net L-N15-lr4 0.8915: −0.030). 998 ms, 2916 MB.
  It also beats the flat model on every recall row (paired, 1968
  stories): far own 0.878 / 0.899, far use 0.249 / 0.217; re-mention
  letters own 0.847 / 0.950, use 1.322 / 1.127; first letter own 2.479
  / 2.685, use 2.246 / 1.964.
- **Around the winner** (600 + 200, lr 8e-3, Kaggle v7): R-N15 (× N 1.5)
  **1.0973** (−0.010); R-all (Real1D main too, stage r 120) 1.1101 (tie);
  R-all-N15 1.0981.
- **The stage-rank VRAM curve** (10-step probes on a T4, recalc:32, b's
  shape, MB): Complex2D r 48 3130, 64 2675, 68 4206, 72 4351, 80 4609, 96
  5127, 128 4426; Real1D r 64 1671, 72 2829, 128 2300. **A rank that is no
  power of two costs about as much as the next power of two, or more** (48
  > 64, 96 > 128, Real1D 72 > 128): allocation or kernel padding, not the
  model. Complex2D adds ~1 GB per 64 ranks on top (the rotated B/C).
  So: Real1D r 128 (cheaper than r 104), power-of-two ranks only. The
  chunk pin (recalc:32) does not remove it, and it moved F-r1d-sr104 by
  only 25 MB. The user: likely the cubecl JIT (the line sizes and the
  tiling follow the dimensions, also without fusion and autotune). A
  multiple of 32 is not enough (r 96 is the worst point).
- **F-r1d-sr64** (Real1D r 64 + mmlp 48, 998,785): dv 1.1231 (Complex2D
  r 64 F-sr64 1.1282, Real1D r 104 1.1076), 806 ms, 1881 MB. The gain is
  the rank (r 64 → 104: −0.016), not the kind (a tie at r 64).
- **R-trapN** (Trapezoid None, r 124): dv **1.1501** (+0.042), 911 ms,
  2916 MB. The trapezoid pays much more than its params (R128a has it
  off, R128b keeps it).
- **The split of the winner** (user: are the stage and main blocks the
  same? bigger main blocks?): one stage layer ~136k (d128 e2 p16 Real1D
  r 104), one main layer ~131k (d128 e2 p16 Complex2D r 64 + MLP 8): the
  4 stage layers hold 543k (54%), the 3 main layers 394k (39%), the rest
  62k. The H-Net paper does the reverse (thin encoders, a big main). The
  main runs at 1/N of the positions, so its VRAM per param is half. The
  old allocation tests (a2 d96 + md176) lost at Complex2D r 32, before the
  stage rank paid. Kaggle batch 6 moves params from the stages to the
  main, with the stage block kept: M-s11-L5 (stage 1+1, main × 5, 991k),
  M-s12-L4 (1+2, × 4, 995k), M-s11-w (1+1, main d168 × 3, 992k),
  M-s11-e3r128 (1+1, main × 3 at expand 3 and r 128, 997k).
- **L-R-N15-lr4** (the winner × N 1.5, lr 4e-3, Kaggle v8): sv 0.9685, **dv
  0.8534** (L-R-lr4 0.8617: −0.008; the flat 0.8798: −0.026), 3092 MB.
  The new best long run.
- **The r 128 screens** (Kaggle v8): R128b (Real1D everywhere, stage r
  128, main r 48, 991k) **1.1078** (ties F-r1d-sr104 1.1076), 2476 MB;
  R128a (Trapezoid None) 1.1718, R128a-N15 1.1140: the trapezoid again.
- **The flat at the milestone 4630** (h16-x4630-d, 4630 + 1000): **dv@5630
  0.8137**.
- **L-R-x4630-d** (L-R-lr4 → 4630 + 1000, lr 4e-3 constant): **dv@5630
  0.8096** (the flat h16-x4630-d 0.8137: −0.004). The lead shrinks: −0.018
  at 3230, −0.004 at 5630. The recall still leads (paired, 1968 stories):
  far own 0.823 / 0.831, use 0.263 / 0.248; re-mention letters own 0.743
  / 0.778; first letter own 2.291 / 2.406, use 2.416 / 2.266. Next here:
  L-R-N15-x4630 (the best long run, −0.026 at 3230).
- **Kaggle batch 6** (600 + 200, lr 8e-3): R128c (Real1D everywhere,
  stage r 128, main r 32, mmlp 16, 996,257) **1.1001**, 914 ms, 2212 MB;
  R128c-N15 **1.0964** (the best screen), 978 ms, 2300 MB; R-N15-s1
  1.1081 (R-N15 seed 0 1.0973: **a seed gap of 0.011**, so the noise of
  the R family is ±0.005–0.008, more than b's ±0.004; gaps < ~0.01 need
  long runs). **The main-heavy shapes lose** (the user's question): stage
  1+1 / main × 5 1.1609; 1+2 / × 4 1.1341; 1+1 / main d168 1.1477; 1+1 /
  main expand 3 + r 128 1.1570 (3810 MB). At this scale the char-level
  stages carry the modeling and need their 2+2 depth. Kaggle batch 7:
  L-R128c-N15-lr4 and L-R128c-lr4 (2630 + 600 + ctx).
- **L-R-N15-x4630-d** (L-R-N15-lr4 → 4630 + 1000): **dv@5630 0.8038** (the
  flat 0.8137: −0.010; L-R-x4630-d 0.8096). The lead over the flat
  shrinks: −0.026 at 3230, −0.010 at 5630. Next: the milestone 8000 +
  1600 (the flat h16-x8000 here, on its sequence: 1e-3 from 4630; the
  H-Net on Kaggle with `extend.sh --lr 1e-3`).
- **The R128c long runs** (lr 4e-3, Kaggle v10): L-R128c-N15-lr4 **dv
  0.8581** (L-R-N15-lr4 0.8534: +0.005), 981 ms, 2300 MB; L-R128c-lr4 dv
  0.8606 (L-R-lr4 0.8617: −0.001), 911 ms, 2212 MB. R128c ties R at the
  long horizon, with ~0.8 GB less VRAM. Kaggle batch 8: L-R-N15-x8000
  (4630 → 8000 + 1600 at 1e-3, as the flat) and the long seed repeat
  L-R-N15-lr4-s1.
- **The flat at the milestone 8000** (h16-x8000-d, 8000 + 1600, 1e-3 from
  4630): **dv@9600 0.7597**.
- **The block control** (is the gain the H-Net or the Real1D large-rank
  block?): fr-d152, the flat h16 with the stage block of the winner
  (Real1D, r 128, head dim 16, d152 × 5, Multi-Gate 4, 980,192), packed
  as h16-s2, lr 4e-3, 2630 + 600 + ctx (here): sv 0.9807, **dv 0.8715**
  (h16-s2 0.8798: −0.008), 802 ms, 3021 MB. **The H-Net still leads the
  block-matched flat** (L-R-N15-lr4 0.8534: −0.018): of its −0.026 against
  h16, −0.008 is the block and −0.018 the H-Net. The recall (paired):
  far own 0.870 / 0.891 (fr-d152), re-mention letters own 0.812 / 0.849,
  first letter own 2.423 / 2.475, gain 1.717 / 1.597: the H-Net leads
  every row. Next here: fr-x4630 (4630 + 1000), the trend against the
  block-matched flat.
- **L-R-N15-x8000-d** (4630 → 8000 at 1e-3 + 1600, Kaggle v11): sv@8000
  0.7902, **dv@9600 0.7461** (the flat h16-x8000-d 0.7597: **−0.014**). The
  lead over the flat: −0.026 at 3230, −0.010 at 5630, −0.014 at 9600.
- **The long seed noise**: L-R-N15-lr4-s1 dv 0.8485 (seed 0 0.8534): ~±0.005
  at 3230. The R128c / R gaps (≤ 0.005) are noise.
- **fr-x4630-d** (the block-matched flat → 4630 + 1000): **dv@5630
  0.8164** (h16-x4630-d 0.8137: +0.003). The large-rank Real1D block helps
  a flat model only early; the H-Net (L-R-N15-x4630-d 0.8038) leads both.
  So the gain is the H-Net, not the block.
- **The stage-heavy allocation** (the block of R128c; queued here):
  A-s33-d112 (stage 3+3 d112, main d112 × 3, 997k), A-s22-d136-L2 (stage
  2+2 d136, main d136 × 2, mmlp 40, 994k), and R128c-s1 (the seed noise of
  the base). Stage 3+2 at d128 does not fit (> 1M even at main × 2).
- **Kaggle batch 9** (lr 4e-3, 2630 + 600): L-R-N125-lr4 **0.8515**, 1122
  ms, 3180 MB (chunk ratio 0.797, target 0.8); L-R-N15-a01-lr4 (α 0.1)
  **0.8517** (ratio 0.667 = target). Both tie R-N15 (seeds 0.8534 /
  0.8485, mean 0.851): the compression and the ratio weight are flat
  around N 1.25–1.5, α 0.1–0.3. Kaggle batch 10: the deeper stages
  A-s44-d96 (stage 4+4 d96, main d96 × 3, mmlp 64, 993k), A-s33-d112-N15,
  A-s44-d96-N15, A-s33-d112-L2 (main × 2 + mmlp 136, 1M).
- **The stage-heavy screens** (here): A-s33-d112 1.1091, 1094 ms, 2493 MB;
  A-s22-d136-L2 1.1073, 918 ms, 2623 MB; **R128c-s1 1.1177** (seed 0
  1.1001: **a seed gap of 0.018**). The R family's short screens have
  ~±0.009 of seed noise: they cannot rank it. Both allocations tie the
  R128c mean (~1.109). From here, long runs decide (±0.005).
- **Kaggle batch 10** (600 + 200): A-s44-d96 1.1105 (1318 ms, 2545 MB),
  A-s44-d96-N15 1.1061, **A-s33-d112-N15 1.0898** (1194 ms, 2493 MB; the
  best screen value, R128c-N15 1.0964: −0.007, inside the noise),
  A-s33-d112-L2 1.1103. N 1.5 helps every shape here (−0.004 … −0.019).
  The long test of A-s33-d112-N15 goes to batch 12.
- **The flat at the milestone 12000** (h16-x12000-d, 8000-12000 cosine
  1e-3 → 5e-4, + 2400): **dv@14400 0.7353**.
- **L-R-N15-x12000-d** (8000 → 12000 at the flat's cosine, in two trunk
  pieces, + 2400; Kaggle v14): sv@12000 0.7500, **dv@14400 0.7207** (the
  flat 0.7353: **−0.015**). The lead over the flat by milestone: −0.026 at
  3230, −0.010 at 5630, −0.014 at 9600, −0.015 at 14400. It holds at
  ~0.015 after the first milestone; the flat does not catch up.
- **L-R128c-N15-lr4-s1**: dv 0.8517 (seed 0 0.8581). Two-seed means at
  3230: R128c-N15 0.8549, R-N15 (r 104) 0.8510: r 104 stays the reference
  (Δ 0.004, ~1 s.e.).
- **L-A-s33-N15-lr4** (stage 3+3 at d112, main d112 × 3, N 1.5, lr 4e-3):
  sv 0.9821, **dv 0.8597** (R-N15 mean 0.851: +0.009). The best screen
  value (1.0898) did not hold at the long horizon. Next here: L-R-N15-wd05
  (weight decay 0.05).
- **User notes (19:50)**: extend N 2 too (the main stage, the entities);
  add a whole-story metric (the long valid stories); the main network
  should matter most; the VRAM of the flat reference (3149 MB, no gc;
  matched within ~2% of the H-Net's ~3.1 GB, but without gc). Plan in
  handoff.md. Kaggle batch 13: L-R-x8000 (N 2, two pieces) and
  L-W-s112-m (stage d112 2+2 Real1D r 128, main d152 × 3 Real1D r 32,
  N 2, lr 4e-3, 981,710: the stages keep their depth, the main grows).
- **Kaggle batch 12** (long, N 1.5, lr 4e-3): L-A-s33-N15-lr4-s1 0.8562
  (seed 0 0.8597, mean 0.858: +0.007 against R-N15); **L-A-s22-L2-N15-lr4
  0.8780** (stage 2+2 at d136, main × 2: +0.027). **Taking a main layer
  away costs much** (with the stages kept deep): the main depth matters
  (the user's intuition). The user: the main wants both a larger d and a
  larger state rank. Here: L-W-m128r-L2 (stage d112 2+2 Real1D r 128,
  main d168 × 2 Real1D r 128, mmlp 32, 996k) and L-W-m64r (main d144 × 3
  r 64, mmlp 24, 998k), long. Batch 13 has L-W-s112-m (main d152 × 3, r
  32: the low-rank contrast) and the N 2 extension.
- **L-R-N15-wd05** (R-N15 at weight decay 0.05, not 0.1): sv 0.9549 (R-N15
  0.9685 / 0.9675: −0.013), **dv 0.8492** (0.8534 / 0.8485, mean 0.851:
  −0.002, a tie). The lower decay helps the stable phase only. ctx: use
  far 0.252, re-mention letters gain 0.868.
- **The whole-story metric** (`--long-stories`, `examples/common/
  long_stories.rs`; `long.sh` / `longstat.py` here): the valid stories
  longer than 2048 chars of the whole split, each in one `forward`, the
  bits per char of 0..1024, 1024..2048, the tail 2048.., all, and a
  profile per 256 chars. On the train cache: 1.5% of the stories are
  longer than 2048, 0.13% longer than 3072 (max 4149). The valid split:
  157 stories (2056..4064, mean 2463); 12% of the stories are longer than
  1024, **7.2% of the valid text is past char 1024** (0.8% past 2048).
  Batch 1 = batch 4 exactly (the right padding is exact). ~40 s per run.
- **The long stories (`long.txt`), bits/char pooled, H-Net R-N15 / flat
  h16:**

  | step | 0..1024 | 1024..2048 | 2048.. | all |
  |---|---|---|---|---|
  | 3230 | **1.0010** / 1.0100 | 1.0551 / **1.0216** | 1.0482 / **0.9946** | 1.0315 / **1.0122** |
  | 5630 | **0.9427** / 0.9412 | 0.9950 / **0.9517** | 0.9945 / **0.9227** | 0.9732 / **0.9425** |
  | 9600 | **0.8768** / 0.8808 | 0.9308 / **0.8942** | 0.9349 / **0.8648** | 0.9090 / **0.8837** |
  | 14400 | **0.8458** / 0.8495 | 0.9023 / **0.8636** | 0.9061 / **0.8387** | 0.8794 / **0.8535** |

  (5630: the flat leads 0..1024 by 0.0015.) Paired at 14400 (flat − H-Net,
  per story): +0.0037 ± 0.0020, −0.0387 ± 0.0032, −0.0769 ± 0.0063, all
  −0.0259 ± 0.0019. **Past char 1024 the H-Net gets worse and the flat
  better**: the profile per 256 (14400) H-Net 0.866 0.875 0.894 0.905
  0.911 0.902 0.891 … 1.02, flat 0.872 0.881 0.882 0.887 0.874 0.864 0.830
  … 0.93. The H-Net protocol trains the first 1024 chars only; the flat's
  packed rows hold whole stories. **The buckets fix it**: L-B-N15-lr4 (b at
  the buckets, whole stories) against L-N15-lr4 (b, first 1024): 0..1024
  −0.0012 ± 0.0024, 1024..2048 −0.0427 ± 0.0032, tail −0.0620 ± 0.0071, all
  −0.0292 ± 0.0022; its profile falls past 1024 as the flat's does. The
  rest at 3230: h16-B-d 1.0172 / 1.0261 / 1.0008 / 1.0181, fr-d152-d
  1.0078 / 1.0188 / 0.9829 / 1.0082. On the whole valid text the past-1024
  loss weighs ~0.072 × 0.04 ≈ 0.003, against the H-Net's first-1024 lead
  (0.015 at 14400). Next here: **L-R-N15-B** (R-N15 at the buckets
  512…4160, bchars 7450, 2630 + 600 + ctx + long), ahead of L-W-m64r.
- The long allocation tests: L-A-s33-N15-lr4 (here, 18:14), Kaggle batch
  12: L-A-s33-N15-lr4-s1 and L-A-s22-L2-N15-lr4.
- **The milestone 12000 + 2400** (the flat plan: 8000-12000 cosine 1e-3 →
  5e-4): the flat h16-x12000 here (one trunk phase of ~50 min); the H-Net
  on Kaggle batch 11 in two trunk pieces (`extend.sh … 2000 0 --lr 5e-4
  --span 8000-12000`, then `… 2000 2400`). extend.sh: `N2 = 0` (a trunk
  piece) and `--span A-B` (stage.py's anneal; dry test: the same sequence
  as the flat). Lane 2: L-R128c-N15-lr4-s1 (two seeds of R128c-N15 long).
- Kaggle batch 9: L-R-N125-lr4 (N 1.25: the trend of the long runs) and
  L-R-N15-a01-lr4 (the ratio-loss weight α 0.1, not 0.3).
- **The extensions to the milestone 4630 + 1000** (the 1m-b plan): the
  flat h16-s2 → h16-x4630 and L-R-lr4 → L-R-x4630 (+ ctx), here.
- **The flat reference is not matched** (user: packed data, the hot
  trunk LR, the H-Net also learns its tokenizer). The volume per step
  is matched: h16-s2 scores 7505 chars/step (2 × 4224 × 0.888), L-N* 7452
  (10 × 1024 × 0.73). The composition is not: the flat rows hold whole
  stories (the positions past 1024 train), the old H-Net protocol only
  the first 1024 chars. The LR: the flat trunk at 4e-3 (its peak in 1m-b),
  the H-Net at 8e-3. The tokenizer: the gap to the flat grows with the
  steps (−0.003 at 800, +0.045 at 3230), not the sign of a slow start
  alone, but 3230 steps is ~11% of an epoch. The matched reference:
  **h16-B** (the flat h16 at the same buckets, lr 4e-3, 2630 + 600 +
  ctx; queued here, `bin/ts-cuda` rebuilt with the buckets).
- **F-base** (b + main MLP 32 at `mlpmul=8`, 998,352): dv 1.1294 (b
  1.1286), 942 ms, 2086 MB. The param filler is neutral: it fills the
  budget without moving the screen.

## The protocol switch (2026-10-09 20:40): whole stories

- **User**: train on full stories with full backprop (length slots); every
  earlier H-Net run is invalid, the flat reference stays, the insights
  stay. L-W-m128r-L2 killed at step 1371. Old-protocol probes:
  `long-v1/`, `long-v1.txt`.
- **The whole valid `wv`** (`long.sh`, `--long-stories`): the 256 valid
  stories in full (207,187 chars; 32 longer than 1024). Its chars 0..1024
  reproduce `dv` exactly (R-N15 0.8534 = dv@3230).
- **The flat reference** (packed whole stories), `wv` / dv:

  | step | flat h16 wv | dv | old H-Net R-N15 wv (invalid) |
  |---|---|---|---|
  | 3230 | **0.8868** | 0.8798 | 0.8659 |
  | 5630 | **0.8204** | 0.8137 | 0.8164 |
  | 9600 | **0.7660** | 0.7597 | 0.7580 |
  | 14400 | **0.7412** | 0.7353 | 0.7322 |

  At 3230 also: fr-d152-d 0.8792, h16-B-d (the flat at the 7 buckets)
  0.8924 (+0.006 against packed, on both parts: 0..1024 +0.005, past
  1024 +0.013).
- **The length grouping costs** (old-protocol b family, `wv`): L-B-N15-lr4
  (7 slots, whole stories) 0.9130 against L-N15-lr4 (first 1024) 0.9033:
  chars 0..1024 0.9061 / 0.8915 (+0.015), past 1024 0.9896 / 1.0340
  (−0.044). B-c1024 (the 1024 data grouped 512/768/1024) cost +0.008.
  So the protocol takes **5 slots: 1024,1536,2048,3072,4160** (`gen.py`
  default): the 88% of stories up to 1024 chars mix in one slot (11 ×
  1024 tokens, fill 0.69; the old protocol 10 × 1024), the long slots stay
  fine (≤ 9.2k tokens). 27,568 batches (7 slots: 28,612).
- L-R-N15-B (7 slots) crashed at start (a cubecl memory failure: it
  started in the second of the kill, beside a probe). Now: **L-R-N15-B5**
  here (20:46), Kaggle batch 14 (after batch 13): L-R-N15-B (7 slots: the
  protocol contrast) and L-R-B5 (N 2).
- **L-R-N15-B5** at the 5 slots: reserved 3386 MB from step 100 to 600
  (flat), 1.28 s/step here (the old protocol ~1.0).
- **User hunch (20:55)**: explore a lot the main-strong settings (a 4th
  layer, a larger main d and state rank, at the cost of the stage d and
  rank); with whole stories, a higher N and a stronger main should matter
  more. Kaggle batch 14 is now L-R-B5 + L-Nm4-s96r128-B5 (the 7-slot
  contrast deferred). The `M-*` family (N 2, stage 2+2 Real1D, main Real1D
  r 128 unless noted; md fitted with mmlp 8, then mmlp filled):

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

  Their VRAM probes (`probes-B.txt`) follow L-R-N15-B5 here.
- **Kaggle batch 13** (old protocol, knowledge only): L-R-x8000-d (N 2)
  dv@9600 0.7515 (N 1.5: 0.7461; N 1.5 still leads by 0.005 at 9600);
  L-W-s112-m-d (stage d112 r 128, main d152 × 3 r 32, N 2) dv@3230 0.8887
  (R at N 2: 0.8617, +0.027: a main at rank 32 loses).
- **probe.sh** visits every slot (burn-stack `--bucket-probe 2`, user: the
  VRAM of a slot depends on the model, not in proportion to its
  positions): `slots a/b/…` = the reserved MB after each slot.
- **Incident (21:42–21:48)**: the first `Bucket.probe` was a `usize` with a
  Burn config default, and a Burn default does not apply when a config
  loads: every saved bucketed config failed ("missing field `probe`").
  Lost: Kaggle batch 14 (both lines at start; re-pushed 21:50), the
  decay phase of L-R-N15-B5 (re-run from its phase-1 checkpoint, 21:48),
  the 10 probes. A first fix made it an `Option` (burn-stack 21b859b);
  **the user: update the configs instead** (reverted, 7ba5ee7). `gen.py`
  writes `"probe": 0`; the bucketed `cfg/` were regenerated from `args.txt`
  (only that field changed); dec-L-R-N15-B5, runs/L-R-N15-B5 and its -d by
  hand. Everything stopped at 22:00 (user), Kaggle v18 too.
- **L-R-N15-B5** phase 1: sv@2630 0.9809 (58.4 min, 0.75 steps/s).
- **The per-slot probe** (smoke test, R-N15-B5): reserved after each slot
  2903 / 2614 / 2421 / 2710 / 2421 MB (1024 / 1536 / 2048 / 3072 / 4160):
  the 1024 slot (11 × 1024) is the peak. Not cumulative (`--max-vram`
  cleans the pool), so `reserved` = their maximum. The run held 3386
  steady (× 1.17 the probe); the probe's total peak 4168 MiB (idle 692).
  The bucket summary of a probe run was wrong (it counted runs of
  consecutive slots; burn-stack ee201dd counts per slot).
