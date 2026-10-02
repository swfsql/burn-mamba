# tmp: session notes (2026-10-02), to merge at the end of the session

## Data sizes

- Train: the first 262,144 stories of the train rows (rows 20,000.., of
  2,712,634), 210M chars, packed into 56,032 rows of 4224. One epoch is
  28,016 steps (2 rows per step).
- A `--resume` (each extension and each decay branch) trains the rest of
  the epoch from a **new** shuffle of all rows. So the draws of the segments
  overlap.
  - h16 @14400 (2630 + 2000 + 3370 + 4000 + 2400): 28,800 row draws (0.51
    epoch), about 23,500 unique rows (42%), about 110k unique stories.
  - A screen @3230 (2630 + 600): about 11% of the rows, about 30k stories.
- Valid loss: the first 256 stories of the valid rows (rows 10,000..20,000),
  190,070 chars.
- Context use: the 244 of those 256 with ≥ 512 chars. 46,848 far chars per
  score. The 9,744 other valid stories and the 10,000 test stories are not
  used.

## Context use on 2048 valid stories (GPU)

- New: `--context-stories N` (the example), `context-stories.jsonl` (the
  α = 0 scores of each story, by its index in the valid split),
  `ctx.sh [-g] [-n N] [-b B] [-r root] <run>`, `ctxstat.py` (mean ± s.e.,
  and paired differences on the same stories). Build
  `v0.64.0-6ca4e79-dirty-afd6a618`.
- The GPU gives the flex numbers to 4 decimals (256 stories: 15 s). 2048
  stories (1968 with ≥ 512 chars): ~78 s per checkpoint. Copies in `ctxg/`,
  summary lines in `ctxg.txt`.
- The s.e. over the stories: 0.006 (gain) and 0.008 (use) on 244 stories,
  0.002 and 0.003 on 1968. A paired difference of two models: ~0.001.

The far part, 1968 stories (pairing i+1):

| run | step | own | none | gain | penalty | use |
|---|---|---|---|---|---|---|
| k16 h16 (seed 0) | 3230 | 0.8765 | 1.0174 | 0.1409 | 0.0848 | 0.2257 |
| n04 h16 (seed 1) | 3230 | 0.8757 | 1.0267 | 0.1509 | 0.0879 | 0.2388 |
| k17 h16 × g2 | 3230 | 0.8832 | 1.0264 | 0.1432 | 0.0963 | 0.2395 |
| h16 | 14400 | 0.7401 | 0.9165 | 0.1764 | 0.1072 | 0.2835 |
| h16 × g2 | 14400 | 0.7478 | 0.9262 | 0.1784 | 0.1102 | 0.2887 |

Paired differences (± s.e.):

| pair | own | none | gain | penalty | use |
|---|---|---|---|---|---|
| seed 1 − seed 0 (h16 @3230) | −0.0007 ± 0.0012 | +0.0093 ± 0.0014 | +0.0100 ± 0.0011 | +0.0031 ± 0.0014 | +0.0131 ± 0.0014 |
| g2 − h16 @3230 | +0.0067 ± 0.0011 | +0.0090 ± 0.0013 | +0.0023 ± 0.0010 | +0.0115 ± 0.0014 | +0.0138 ± 0.0015 |
| g2 − h16 @14400 | +0.0077 ± 0.0010 | +0.0098 ± 0.0013 | +0.0021 ± 0.0011 | +0.0031 ± 0.0017 | +0.0051 ± 0.0016 |

- **The seed noise is now the limit, not the stories.** The two h16 seeds
  differ by 0.010 in gain and 0.013 in use, ~10 s.e. More stories do not
  help. 2048 is enough.
- **The seeds differ in the no-prompt score, not in the own score** (own
  −0.0007, none +0.0093). So `bits(C | own)` is the stable metric (one pair
  of seeds). The gain and the use take the noise of their baselines: `C`
  read as a story start (none) and after a wrong prompt (other).
- **This corrects `notes.md` §4:** on 244 stories, the seed spread looked
  like gain 0.006 / penalty 0.011. On 1968 stories it is gain 0.010 /
  penalty 0.003. The gain is the noisy part.
- **g2 against h16:** own is worse at both milestones (+0.007, +0.008, far
  above the seed difference of own). The gain ties (+0.002). The penalty is
  higher @3230 (+0.0115, above the seed difference of 0.003). So the
  reading of `notes.md` §4 holds: g2 commits harder to any prompt, and it
  does not use its own prompt better.
- For step 2: judge by `bits(C | own)` of the far part (paired, 2048
  stories) and the valid loss. Read the gain and the use against a seed
  noise of ~0.01.

## Step 2: a design for the contrast data (proposal, not built)

What the pipeline gives (read in burn-stack `examples/tiny_stories/{lm,dataset}.rs`
and `training.rs`):

- burn-stack's `epoch_train` drives the loop and calls
  `Wrap::train_window(batch, …)` once per step. A packed batch is one
  window: `[2, 4224]` token ids, a `Packed` layout (resets and opening
  slots), and `score_bs`. The batch has no story identity and no cut
  positions. Its workers build it on the host.
- burn-stack is a git dependency at a pinned rev. A change there needs a
  commit there and a rev bump here. The design below changes only the
  example.

The design (agreed with the user, 2026-10-02; runs from scratch):

1. **A contrast feed in the example.** `Wrap` holds it (interior
   mutability, because `train_window` takes `&self`). It draws K train
   stories per step from its own seeded shuffle (the resume seed, as the LM
   loader). It cuts each story into P and C, and skips a story that is too
   short. It builds the batch on the main thread (K rows are cheap).
2. **Configuration** (example flags after `--`, also saved in the artifacts
   dir, so that a bare `--resume` keeps them): `--contrast K`,
   `--contrast-prompt MIN:MAX` (a random P length), `--contrast-cut
   char|word|sentence` (move the cut back to the last boundary of that
   kind), `--contrast-continuation N`, `--contrast-skip J`,
   `--contrast-weight λ`, `--contrast-lm` (also the CE on the C chars of
   the contrast rows: more LM data, off at first). P is always padded to
   MAX with the `pad` mask, so the shapes do not change from step to step
   (burn#5751).
3. **One P pass, two C passes.** `forward([lat|P], None, path, cursor,
   pad)` with right padding: the cache of each slot is exact at its own
   cut. Then `forward(C, caches)` (own) and `forward(roll(C, 1),
   caches.clone())` (other: row i−1 reads C_i after P_{i−1}). The roll is
   cyclic. This is the forward over concat(P, C), split at the cut so that
   the other pass shares the P part of the graph. It stays **shared
   suffix**: C_i is scored after P_i (own, row i) and after P_{i−1} (other,
   row i−1). The cursor emits the latents in the P pass only.
4. **The loss.** Gather log p at the C targets of both passes. Roll the
   other pass back by one row, so that both terms of row i score C_i. Then
   Δ_i,s (own − other, per char), the prefix sums Δ_≤t from C[J] (a
   `[c, c]` lower-triangular matmul, not `cumsum`), and
   λ · mean_t −log σ(Δ_≤t). Every `[lat|P]` position of both passes, the
   padding and C[0..J) get no contrast loss. The gradient goes back into
   P_i and P_{i−1} through both caches (no detach).
5. **First runs:** h16 from scratch to 3230 (`screen.sh … 2630 600`), one
   seed (user: models before seeds, now in `ORIENTATION.md`), against k16
   (own far bits, 2048 stories, paired). Char cut, J = 0 (the join
   saturates Δ early: a gentle start, most of the gradient near the cut),
   K = 4, λ = 1 (normalized), start 500, ramp 500. Next: J = 64 (the
   contrast on the far part of the metric), then the sentence cut.
6. **VRAM: two backward passes, one optimizer step.** First the LM loss:
   forward, backward, `GradientsParams` (this frees the LM activations).
   Then the contrast: forward, backward, accumulate
   (`GradientsAccumulator`, `TrainOutput` has public fields). The peak is
   the larger pass, not the sum. K = 4 at P ≤ 384 is ~4k tokens against the
   8.4k of the LM pass, so h16 stays at ~3134 MiB. The cost is time: ~+40%
   per step at K = 4 (the user accepts it).
7. **Logs.** The contrast loss and the mean Δ per char, per step. Evaluate
   with `ctx.sh -g -n 2048`: own far, gain, penalty, and the valid loss.

Rejected for a start: the own term from the packed LM rows (it saves one C
pass). It needs the story bounds and the cuts inside a packed row, so it
needs a change to burn-stack's batcher.

## Step 2: implemented (example only, no burn-stack change)

- `examples/tiny-stories/contrast.rs` (new): `Cut` (`allows`),
  `ContrastConfig` (`#[derive(Config)]`, saved as `contrast.json` in the
  artifacts dir; `resolve` = saved + flags), `ContrastOverrides` (the
  `--contrast*` flags and their help), `ContrastBatch::new` (the P pass
  without the last P char, right-padded to `prompt_max − 1`; the own and
  other C inputs start with the last P char, so the C passes hold the row of
  every C char at a fixed shape), `loss` (one P pass, two C passes, the
  roll back, `Δ` in nats, the prefix sums by a `triu` matmul,
  `−log_sigmoid`, `λ`; `--contrast-lm` adds the CE of the own C chars over
  the positions of the packed batch), `cut` (a random length, moved back to
  the cut kind), `Feed` (the train stories ≥ `prompt_min + continuation`,
  truncated to `prompt_max + continuation`; its own shuffle; per step one
  line in `contrast.jsonl`; every 100 steps a `contrast step=… loss …
  delta … bits/char` line), `FEED` (a static, as `prof`).
- `contrast/tests.rs`: the cut kinds, the cut bounds, and `loss` against
  one `forward` over each `concat(P, C)` (values and the embedding grad;
  with detached caches the grad check fails: max rel diff 0.43).
- `training.rs`: `train(…, contrast)` starts the feed (seed = the LM
  shuffle seed ^ `CONTRAST_SEED`, step = `progress.step + 1`).
  `train_window`: LM backward → `GradientsParams`, then the contrast pass
  and its backward, `GradientsAccumulator`, `TrainOutput { grads, item }`.
  The `--max-vram` check is after both. `prof::contrast` adds the contrast
  to `fwd`/`bwd`. The readouts are read after `window_end` (in `gap`).
- `main.rs`: `mod contrast`; `ContrastConfig::resolve` when training.
  `cli.rs`: the flags. (Also `--context-stories`, see above.)
- Smoke config: `cfg/t-tiny` (`gen.py t-tiny d=32 L=2 V=0 r=16 e=2 p=16
  lr=4e-3 wd=0.1 seq=1024 runlen=1 stories=262144 frontier=always`, 18,304
  params).

### Smoke tests (build `v0.64.0-6ca4e79-dirty-6b24e9f4`, 300 + 100 steps)

- No panic, no NaN. `contrast.json` is saved, and a bare `--resume`
  (`s02-tiny-c4-b`, no flags) keeps the contrast. `contrast.jsonl`
  continues its steps across the resumes. (The `contrast step=` lines of a
  `-d` log repeat phase 1: `screen.sh` copies the dir.)
- h16 probe (10 steps): reserved 3149 → **3157 MB** (+8), ms/step 697 →
  **948** (+36%). So the peak memory does not change. Tiny: 107 → 222 ms
  (launch-bound).
- The contrast loss is ~5 over the first 100 steps at every λ (an untrained
  model gives very different log-probs after two prompts), then 0.65 →
  0.35 → 0.18. The train Δ grows to +0.03 bits/char at step 400.

| run (t-tiny) | λ | contrast | dv @400 | cost | train Δ @400 |
|---|---|---|---|---|---|
| s01-tiny-ctl | – | – | 2.5102 | – | – |
| s04 w0.001 | 0.001 | char, J 0 | 2.5197 | +0.010 | +0.0305 |
| s04 w0.01 | 0.01 | char, J 0 | 2.5371 | +0.027 | +0.0303 |
| s02-tiny-c4 | 0.1 | char, J 0 | 2.7368 | +0.227 | +0.0244 |
| s03-tiny-c4-all | 1 | sentence, J 64, lm | 3.1073 | +0.597 | ~0 (loss ≈ ln 2) |

- **The train Δ does not depend on λ** (+0.0305 at λ = 0.001, +0.0244 at
  0.1). At this size, the LM learns the separation, and the contrast term
  only costs loss. At J = 64 the tiny model stays at chance.
- **The scale of λ.** `∂L/∂Δ_i,s = −(1/(K·m)) Σ_{t ≥ s} σ(−Δ_i,≤t)`. At
  chance (σ = ½), the push on all own log-probs sums to `λ·(m + 1)/4`
  (= 64λ at m = 256, for any K), against 1 for the LM loss. The push on
  the other log-probs is the same, with the other sign. So λ = 0.1 pushes
  6.4× as hard as the LM loss at chance. The prefix sums amplify by ~m/4.

### Normalization and schedule of λ (user, 2026-10-02)

- **Normalized:** the loss is `λ · 4/(m + 1) · mean −log σ(Δ_≤t)`. At
  λ = 1, the contrast pushes as hard as the LM loss at chance. The old
  λ = 0.01 is ~0.64 now, and the old 0.1 is ~6.4.
- **λ = 0 at init** (user: no point in topics before grammar).
  `--contrast-start S` (default 1000), `--contrast-ramp R` (default 1000):
  `λ(step) = λ · clamp((step − S)/R, 0, 1)`. While λ(step) = 0 and
  `--contrast-lm` is off, the pass does not run (no time cost). The weight
  is a `[1]` tensor, not a kernel scalar, because it changes every step
  of the ramp (burn#5751).
- `contrast.jsonl` lines now hold `weight`, and only the steps where the
  pass ran. Test `weight_is_zero_then_ramps`.
- Smoke `s05-tiny-c4-ramp` (build `…-dirty-81b30239`, λ 1, start 150, ramp
  100): no pass before 150 (~100 ms/step, as the control), weight 0.01 @151,
  0.5 @200, 1 @250, ~210 ms/step after. Loss 0.48 over 151–250 (no early
  ~5). dv @400 2.5187 (+0.0085 against the control: the cost of the old
  λ = 0.001). Train Δ +0.0305 @400, as every other tiny run.
- h16 train curve (k16, window means): 3.05 bits @50, 1.98 @100, 1.54
  @200, 1.28 @500, 1.11 @1000, 0.95 @2630. Seed 1 is the cfg
  `s1k-u1-wd1-h16-s1` (n04).
