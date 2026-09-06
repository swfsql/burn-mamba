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

A story's start is marked out of band, by four learnable **class latents**, not by
a character — see [Story boundaries](#story-boundaries).

## Data

The dataset is a single 673MB parquet file: one column (`text`), one row per
story, 2,669 ZSTD row groups of 1,024 rows. It is downloaded **whole**, once, the
same way `mnist-*` downloads its IDX files, into
`~/.cache/burn-dataset/tinystories-gpt4-clean/`; only the row groups a request
touches are decompressed. The stories that come out are normalized and cached
again as text, one file per `(split, story count)` — so every later run reads a
few MB of text and never opens the parquet at all. The loader, the windowing and
the epoch loops are `burn_stack::examples::tiny_stories`.

Splits follow the dataset card's suggested row ranges (the rows are pre-shuffled,
so a contiguous range is already a random sample): rows `0..10k` are test,
`10k..20k` validation, `20k..` training. The defaults pull 4,096 train and 256
validation stories (~3.4MB of text); `--train-stories` scales that up at no extra
download.

## Story boundaries

**One item is one story** (303–4,149 characters, median 724), stripped of its
surrounding whitespace, and nothing is spliced between two of them: a story is a
self-contained example. What marks its start is four `ClassLatent::Start`
registers — learnable `d_model`-wide rows the stack prepends to the sequence — and
the **last of them is scored against the story's first character**. So the model
is trained to answer "what does a story open with?" from the latents alone.

That is what unconditional sampling then does: `prime()` replays the latents
against a zero cache, with no input token, and hands back the first character's
distribution; generation continues from there with plain `step()`s. The
alternative — the `"\n\n"` that used to join the stories, fed in as a seed — is out
of distribution, because that sequence only ever occurred *between* two stories,
i.e. always on a state still carrying the previous one.

One `generate()` call is therefore one story. A second story wants a second call
against a **reset** cache, which is the one place these examples genuinely reset
one.

Every position is scored against its next character (so the reported accuracy is
per character), and a story is walked in windows — see
[Runs and the frontier](#runs-and-the-frontier). Stories differ in length, so a
batch is padded to a whole number of windows of its longest one; the batch
carries how many positions of each slot are real, and the padding is gathered
away before the loss, never reaching it or the accuracy.

## Runs and the frontier

A window is `seq_len` characters, but a story is not: it continues past the cut,
and so does the state that generation would have there. Dropping the rest — what
`--run-len 1` does — trains the model only on story openings, so it never sees the
state a story is in past its first `seq_len` characters.

So the loop walks a story's windows in order — the *run* — takes one optimizer
step per window, and **carries the final state into the next window**:

- The run's length is the **story's**. `--run-len` is only a cap on it, and its
  default (`usize::MAX`) imposes none, leaving the depth entirely to the gate.
- The carry is *earned*. After each window the **frontier gate** scores it, and a
  failing window ends the run — the rest of the story is discarded rather than
  trained on a state the model got lost in. The gate is trainer-side: it reads
  one scalar and decides whether a cache is passed on; no gradient goes near it.
  The default is absolute: advance while the window scored at most
  `--frontier-bits` (1.6) bits per character. That acts from the first window of
  the first epoch, and depth grows out of the training curve by itself. A closed
  gate is not a *wrong* regime — window 0 is the story's own beginning, so its
  zero state is the right one and the recurrence still runs the whole window;
  what a closed gate costs is **reach**, the model seeing only each story's first
  `seq_len` characters. The threshold has to sit *above* where the model settles
  (this one reaches ~1.4-1.5 bits/char), or the curriculum never starts.
- The carry is **detached** (a round trip through the inner backend, not
  `Tensor::detach`, which frees nothing): gradients never cross a window
  boundary, and peak memory is one window's activations regardless of how deep
  the run goes. Back-propagation *within* the window is untouched.

Every slot of a mini-batch walks its own story in lockstep, so one training
iteration is still one batch and the log line stays `Batch b/N` — with
`Windows k/n (mean m)` added, `n` being the windows the batch's longest story
spans and `m` the epoch's mean depth so far, which is the number that says
whether the curriculum is moving. `1.0` is a frontier that never opens (each
story trained to its first window and no further); `n` is a gate that never fires
(`--no-frontier`, plain stateful TBPTT).

Validation runs the one regime that exists: the state threaded through each whole
story, ungated, which is exactly what generation has.

## Usage

```bash
# debug check in flex (fp32)
cargo check --example tiny-stories

# train and then sample (downloads the 673MB parquet once, if it is not cached yet)
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
| `--run-len <n>` | `usize::MAX` | cap on the windows one story may spend (`1` ⇒ openings only) |
| `--frontier-bits <f>` | 1.6 | the frontier gate's threshold, in bits per character |
| `--no-frontier` | off | carry the state through the whole story, ungated |
| `--train-stories <n>` | 4096 | stories pulled from the train split |
| `--valid-stories <n>` | 256 | stories pulled from the validation split |
| `--epochs <n>` | 16 | passes over the corpus |
| `--batch-size <n>` | 8 | windows per optimizer step |
| `--no-muon` | off | keep the hidden weight matrices on AdamW instead of [Muon](https://kellerjordan.github.io/posts/muon/) (see `mnist-class`'s README) |

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Results

16 epochs over the default corpus (3.36M characters), measured on the held-out
validation split. Uniform baseline: `log2(48) = 5.58` bits/char.

Every number below was measured on the **previous** corpus layout: one continuous
`"\n\n"`-joined character stream, cut into stateless windows (what is now
`--run-len 1`), with no class latents. Story-per-item scoring changes what the
average is over, so the table is a record of the *levers*, not a current
measurement; the ranking is what it is here for.

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
sentence from the sequence's own start alone, then keeps one subject and its
pronoun consistent to the end of the sample. Its grip is on syntax rather than
sense: the clauses parse and the sentence boundaries land, but "loved to drop",
"loved to cry ahead", and the name "looks" show it is still assembling plausible
shapes rather than meanings. That is the honest ceiling for 39K parameters.

## Sampling

`inference.rs` shows the library's three execution modes back to back: the class
latents are replayed by one `prime()` (no input token, and it already answers with
the first character's distribution), a prompt — when there is one — is consumed by
one chunkwise `forward()` (prefill), and every generated character then costs one
`step()` against that same cache — O(state) per token, with no growing KV cache.
Sampling is temperature-scaled multinomial over the full 48-way softmax
(`temperature <= 0` is greedy), seeded by `ChaCha8Rng` so a run is reproducible.

`--inference` writes one story per temperature (0.5 / 0.8 / 1.0), each primed and
unprompted, plus one continuation of a fixed prompt into
`<artifacts>/inference/`. Training samples a short story at every small validation
check into `<artifacts>/sample-epoch-{e}-batch-{b}.txt`, so the text can be
watched turning from noise into words into sentences. The checks are spaced in
optimizer steps (every 300), so their cadence does not move with the run lengths
the corpus happens to hand out.

## Notes

- Loss is reported both in nats (Burn's cross-entropy) and as **bits per
  character**; the uniform baseline is `log2(48) = 5.58` bits.
- The tied head starts *badly*: Burn initialises an `Embedding` from `N(0, 1)`,
  so at `d_model = 32` the initial logits have variance ~32 and the first batches
  score 25-40 bits/char instead of 5.58. It is a transient — the opening steps
  are spent shrinking the embedding — but it does eat the start of the LR
  schedule. An untied head (`missing_lm_head = false`) does not have it, and the
  proper fix would be an initializer knob on the library's `VocabNetworkBuilder`.
