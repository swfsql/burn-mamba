# TinyStories (character-level LM)

An auto-regressive Mamba-3 language model over single **characters** of
[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean),
a cleaned 2.7M-story subset of [TinyStories](https://arxiv.org/abs/2305.07759)
(GPT-4-generated children's stories, plain ASCII).

The model is deliberately tiny: two Mamba-3 blocks (`d_model = 32`,
`state_rank = 64`, `expand = 4`), cycled to an 8-deep virtual stack over
Multi-Gate residuals, between a tied character embedding and its transpose.
39,632 parameters, of which the embedding is 1,536.

## Vocabulary

The dataset's cleaning pipeline guarantees exactly 74 distinct ASCII characters:
the 52 cased letters plus ``\n !"$',-.0123456789:;?``. Case-folding the letters
leaves **48** tokens, and every one of them actually occurs — so the alphabet is
the corpus's own inventory, not a slice of ASCII:

```text
\n !"$',-.0123456789:;?abcdefghijklmnopqrstuvwxyz
```

There is no `<unk>`, no `<bos>` and no padding class (`pad_vocab_size_multiple =
1`), so every logit the model emits is a character the decoder understands. The
embedding is **tied** (`missing_lm_head = true`): one table answers both "which
character is this" and "which character comes next".

Stories are joined with `"\n\n"` — a blank line, which never occurs *inside* a
story (single `\n` separates its paragraphs), so it is an unambiguous document
boundary, and it is also the prompt used for unconditional sampling.

## Data

The dataset ships as a single 673MB parquet file, which is absurd for an example
this size, so instead of the `HuggingfaceDatasetLoader` path (python + the
`datasets` library + a full sqlite import) the corpus is paged out of the public
[datasets-server](https://huggingface.co/docs/datasets-server) `/rows` endpoint,
100 stories per request (its hard maximum, ~2s each). The normalized text is
cached in `~/.cache/burn-dataset/tinystories-gpt4-clean/<split>-<n>.txt`, so the
download happens once per `(split, story count)`. The loader, the windowing
and the epoch loops are `burn_stack::examples::tiny_stories`, one copy shared
with `burn-deltanet`'s counterpart of this example — including that cache.

The endpoint is rate limited — the measured budget is ~28 requests per two
minutes, after which CloudFront answers `429` with an HTML body for ~15s at a
time — so the pager paces itself at one page per 4s and retries a failed one with
exponential backoff (30s, doubling, 6 attempts). The default corpus is 43
requests, roughly three minutes; `--train-stories 32768` is 329 requests, closer
to half an hour. All of it is one-time: the normalized text is then read from the
cache.

Splits follow the dataset card's suggested row ranges (the rows are pre-shuffled,
so a contiguous range is already a random sample): rows `0..10k` are test,
`10k..20k` validation, `20k..` training. The defaults pull 4,096 train and 256
validation stories (~3.4MB of text, ~43 requests); `--train-stories` scales that
up.

The character stream is cut into non-overlapping windows of `seq_len + 1`, each
scored at **every** position against its next character (so one window
contributes `seq_len` classification examples, and the reported accuracy is per
character). One training **item** is a *run* of `run_len` consecutive windows —
see [Runs and the frontier](#runs-and-the-frontier).

## Runs and the frontier

A window is `seq_len` characters, but the stories are not: the stream continues
past the cut, and so does the state that generation would have there. Training
each window from a **zero** state — the obvious tiling, and what `--run-len 1`
still does — therefore trains a regime inference is never in after its first
`seq_len` characters.

So the loop walks the `run_len` windows of an item in order, takes one optimizer
step per window, and **carries the final state into the next window**:

- The carry is *earned*. After each window the **frontier gate** scores it, and a
  failing window ends the run — the rest of the item is discarded rather than
  trained on a state the model got lost in. The gate is trainer-side: it reads
  one scalar and decides whether a cache is passed on; no gradient goes near it.
  The default is relative — advance while the window scored at most `1 + tol`
  times the running EMA of **opening** (zero-state) window losses, i.e. *did the
  carried state do at least as well as starting fresh would have?* The baseline
  rides the training curve down, so the question means the same thing in epoch 1
  and epoch 16.
- The carry is **detached** (a round trip through the inner backend, not
  `Tensor::detach`, which frees nothing): gradients never cross a window
  boundary, and peak memory is one window's activations regardless of `run_len`
  — measured flat at 312MB RSS for `run_len` 1 and 8 (flex, `seq_len = 256`,
  batch 8). Back-propagation *within* the window is untouched.

Every slot of a mini-batch walks its own run in lockstep, so one training
iteration is still one batch and the log line stays `Batch b/N` — with
`Windows k/run_len (mean m)` added, `m` being the epoch's mean depth so far,
which is the number that says whether the curriculum is moving. `1.0` is
a fully stalled frontier (stateless training); `run_len` is a gate that never
fires (`--no-frontier`, plain stateful TBPTT).

Validation reports both regimes, `[fresh state]` (every window from zero — the
`run_len`-independent number, and the one the [Results](#results) table is in)
and `[carried state]` (threaded through the run, ungated — what generation has).

## Usage

```bash
# debug check in flex (fp32)
cargo check --example tiny-stories

# train and then sample (downloads ~3.4MB of stories on the first run)
cargo run --release --example tiny-stories --features "backend-cuda" -- --training --inference

# a bigger corpus and a longer window
cargo run --release --example tiny-stories --features "backend-cuda" -- --training \
    -- --train-stories 32768 --seq-len 512
```

With the defaults (`seq_len = 256`, `batch_size = 8`) training needs ~1.2GB of
vram. Downstream flags, all forwarded after the trailing `--` and persisted into
the artifacts' `training_config.json`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--seq-len <n>` | 256 | characters per window (the BPTT length) |
| `--run-len <n>` | 8 | windows per item, i.e. how far the carried state may reach (`1` ⇒ stateless) |
| `--frontier-tol <f>` | 0.05 | slack of the frontier gate over its opening-window baseline |
| `--no-frontier` | off | carry the state through the whole run, ungated |
| `--train-stories <n>` | 4096 | stories pulled from the train split |
| `--valid-stories <n>` | 256 | stories pulled from the validation split |
| `--epochs <n>` | 16 | passes over the corpus |
| `--batch-size <n>` | 8 | windows per optimizer step |
| `--no-muon` | off | keep the hidden weight matrices on AdamW instead of [Muon](https://kellerjordan.github.io/posts/muon/) (see `mnist-class`'s README) |

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Results

16 epochs over the default corpus (3.36M characters), measured on the held-out
validation split from a **zero** state (the `[fresh state]` line). Uniform
baseline: `log2(48) = 5.58` bits/char.

Every number below was measured with stateless windows, i.e. at what is now
`--run-len 1`; carried state and the frontier gate are not in them.

| Setting | Valid bits/char | Valid char accuracy |
|---|---|---|
| **the default** | **1.386** | **70.2%** |
| the same model, 4 epochs | 1.475 | 68.3% |
| batch 16, `lr = 2e-3`, no Muon, 4 epochs | 1.749 | 63.2% |

The last row is what the obvious defaults give. Closing that gap took no extra
parameters worth mentioning (39,496 → 39,632, still under 40K) — this model is
**optimization-limited, not capacity-limited**, and the cheapest evidence is that
two unrelated optimizer changes each beat *every* architectural reallocation that
fits the budget. In decreasing order the levers were:

| Lever | Effect |
|---|---|
| `batch_size` 16 → 8 | the largest single win; worth more than the whole LR ladder |
| `max_lr` 2e-3 → 12e-3 | monotone to 16e-3, flat to 24e-3, turns over at 32e-3 |
| virtual layers 4 → 8 | free in parameters; peaks at 8 (12 is worse) |
| `MultiGate` residuals, `n_stream = 4` | +136 parameters; peaks at 4 (8 is worse) |
| 4 → 16 epochs | still improving at epoch 14, flat by 15-16 |
| Muon | the smallest, but it stacks with the other two optimizer changes |

Depth and `MultiGate` are worth a note: both are essentially free in parameters,
and both *lost* when screened at the original `lr = 2e-3` (8 virtual layers gave
1.758 against 4 layers' 1.749). They only pay once the optimizer can use them —
which is the same finding as the LR ladder, seen from the architecture side.
Nothing that trades one part of the budget for another ever won: a SwiGLU MLP,
more real layers at lower `expand`, `Quaternion4D`, and the library's reference
`rotation_range = 1` + `rope_fraction = 0.5` all scored at or below the default.
The model also never overfits — at epoch 14 validation is *ahead* of the epoch's
running training average — so `--train-stories` is not the lever either.

### Truncated BPTT

`grad_horizon` back-propagates only the top `K` virtual layers. It is a bad deal
here, at the same parameter count (measured at the older 4-epoch, 4-layer setting):

| Virtual layers | `grad_horizon` | Valid bits/char | Valid char accuracy | it/s |
|---|---|---|---|---|
| 4 | `None` | 1.749 | 63.2% | ~7 |
| 16 | 4 | 3.208 | 36.1% | ~3.2 |

A language model is scored at *every* position, so leaving 12 of the 16
applications of a shared weight undifferentiated biases every one of those
readouts — unlike a task that reads out once, at the end of the sequence. The
16-layer arm also plateaued at epoch 2 and then regressed, on training loss as
well as validation.

At 1.39 bits/char the samples are real words with real spelling, a consistent
character, and clauses that mostly parse — about what 39K parameters buys. Four
consecutive **unprompted** samples from the last epoch (seeded only with the
document boundary, `sample_temperature = 0.8`, first 100 characters of each):

```text
once upon a time, there was a little girl named lucy. she loved to drop and saw a small fruit gold c
once upon a time, there was a little girl named looks. she liked to play with her toys and walked on
once upon a time, there was a little girl named kitty. she liked to play with it. she climbed in her
once upon a time, there was a little girl named lucy. she loved to cry ahead and started to eat it.
```

Nothing supplies that opening — the model reconstructs the corpus's stock first
sentence from a start-of-document token alone, then keeps one subject and its
pronoun consistent to the end of the sample. Its grip is on syntax rather than
sense: the clauses parse and the sentence boundaries land, but "loved to drop",
"loved to cry ahead", and the name "looks" show it is still assembling plausible
shapes rather than meanings. That is the honest ceiling for 39K parameters.

## Sampling

`inference.rs` shows the library's two execution modes back to back: the prompt
is consumed by one chunkwise `forward()` (prefill), and every generated character
then costs one `step()` against that same cache — O(state) per token, with no
growing KV cache. Sampling is temperature-scaled multinomial over the full 48-way
softmax (`temperature <= 0` is greedy), seeded by `ChaCha8Rng` so a run is
reproducible.

`--inference` writes one story per temperature (0.5 / 0.8 / 1.0) plus one
continuation of a fixed prompt into `<artifacts>/inference/`. Training samples a
short story at every small validation check into
`<artifacts>/sample-epoch-{e}-batch-{b}.txt`, so the text can be watched turning
from noise into words into sentences. The checks are spaced in optimizer steps
(every 100), so their cadence does not move with `--run-len`.

## Notes

- Loss is reported both in nats (Burn's cross-entropy) and as **bits per
  character**; the uniform baseline is `log2(48) = 5.58` bits.
- The tied head starts *badly*: Burn initialises an `Embedding` from `N(0, 1)`,
  so at `d_model = 32` the initial logits have variance ~32 and the first batches
  score 25-40 bits/char instead of 5.58. It is a transient — the opening steps
  are spent shrinking the embedding — but it does eat the start of the LR
  schedule. An untied head (`missing_lm_head = false`) does not have it, and the
  proper fix would be an initializer knob on the library's `VocabNetworkBuilder`.
