# TinyStories (character-level LM)

An auto-regressive Mamba-3 language model over single **characters** of
[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
That dataset is a cleaned 2.7M-story subset of
[TinyStories](https://arxiv.org/abs/2305.07759): children's stories that GPT-4
wrote, in plain ASCII.

The model is small: a few Mamba-3 blocks, cycled to a virtual stack over
Multi-Gate residuals, between a tied character embedding and its transpose.

This example is a work in progress. It works, but its sizes and hyperparameters
are placeholders until a parameter search. `model.rs` and `main.rs` hold the
current values.

## Vocabulary

The cleaning pipeline of the dataset guarantees exactly 74 distinct ASCII
characters: the 52 cased letters plus ``\n !"$',-.0123456789:;?``. Case folding
leaves **48** tokens, and each of them occurs in the corpus. So the alphabet is
the inventory of the corpus, not a slice of ASCII:

```text
\n !"$',-.0123456789:;?abcdefghijklmnopqrstuvwxyz
```

There is no `<unk>`, no `<bos>` and no padding class
(`pad_vocab_size_multiple = 1`). So every logit of the model is a character that
the decoder knows. The embedding is **tied** (`missing_lm_head = true`): one table
answers both "which character is this" and "which character comes next".

Four learnable **class latents** mark the start of a story, not a character (see
[Story boundaries](#story-boundaries)).

## Data

The dataset is a single 673MB parquet file: one column (`text`), one row per
story, 2,669 ZSTD row groups of 1,024 rows.

- The first run downloads it **whole**, once, into
  `~/.cache/burn-dataset/tinystories-gpt4-clean/` (as `mnist-*` downloads its IDX
  files).
- A request decompresses only the row groups that it reads.
- The normalized stories go into a second cache as text, one file per
  `(split, story count)`. So every later run reads a few MB of text and never
  opens the parquet.

`burn_stack::examples::tiny_stories` holds the loader, the windowing and the
epoch loops.

The splits use the row ranges that the dataset card suggests. The rows are
pre-shuffled, so a contiguous range is already a random sample:

- rows `0..10k`: test,
- rows `10k..20k`: validation,
- rows `20k..`: training.

The defaults take 4,096 train and 256 validation stories (~3.4MB of text).
`--train-stories` takes more, with no extra download.

## Story boundaries

**One item is one story** (303–4,149 characters, median 724), without its
surrounding whitespace. Nothing goes between two stories: a story is a
self-contained example.

- Four `ClassLatent::Start` registers mark its start. They are learnable
  `d_model`-wide rows that the stack puts in front of the sequence.
- The **last of them is scored against the first character of the story**. So
  the model learns "what does a story start with?" from the latents alone.

Unconditional sampling uses exactly that. `prime()` replays the latents against a
zero cache, with no input token, and returns the distribution of the first
character. Generation then continues with plain `step()`s. The alternative is a
separator character between stories, given as a seed. That seed is out of
distribution: in a joined stream it occurs only *between* two stories, so always
on a state that still carries the previous story.

So one `generate()` call is one story. A second story needs a second call
against a **reset** cache. That is the one place where these examples really
reset a cache.

On CUDA, the decode `step`s replay from one captured graph (burn-stack's
`CapturedStep`), and are not launched again. A model this small is bound by the
host that enqueues its launches, not by the GPU. The prefill chunks of a prompt
also replay, from one graph that every prompt shares. The text is the same in
both modes. `--no-graph` runs both eagerly.

Every position is scored against its next character, so the reported accuracy is
per character. A story is walked in windows (see
[Runs and the frontier](#runs-and-the-frontier)).

- Stories differ in length. So a batch is padded to a whole number of windows of
  its longest story.
- The batch carries the number of real positions of each slot. The loss and the
  accuracy mask the padding out, at the fixed shape of the window. A shape that
  changes per window slows every later CUDA allocation (tracel-ai/burn#5751).
- `-- --profile <N>` prints the mean ms of each phase per `N` windows, and the
  live device allocations. These stay flat while no launch shape changes.

## Runs and the frontier

A window is `seq_len` characters, but a story continues past the cut. So does
the state that generation would have there. `--run-len 1` drops the rest of the
story. Then the model trains only on story openings, and it never sees the state
of a story past its first `seq_len` characters.

So the loop walks the windows of a story in order (the *run*). It takes one
optimizer step per window, and **carries the final state into the next window**:

- The length of the run is the length of the **story**. `--run-len` is only a
  cap. Its default (`usize::MAX`) sets no cap, and leaves the depth to the gate.
- The carry is *earned*. After each window, the **frontier gate** scores it. A
  window that fails ends the run: the rest of the story is discarded, and not
  trained on a state where the model is lost.
  - The gate is on the trainer side. It reads one scalar and decides if a cache
    goes to the next window. No gradient goes through it.
  - The default is absolute: continue while the window scored at most
    `--frontier-bits` (1.6) bits per character. That applies from the first
    window of the first epoch, and the depth grows with the training curve.
  - A closed gate is not a *wrong* regime. Window 0 is the start of the story, so
    its zero state is correct, and the recurrence still runs the whole window. A
    closed gate costs **reach**: the model sees only the first `seq_len`
    characters of each story.
  - The threshold must be *above* the level where the model settles, or the
    curriculum never starts.
- The carry is **detached**. It makes a round trip through the inner backend, not
  `Tensor::detach` (which frees nothing). Gradients never cross a window
  boundary, and the peak memory is the activations of one window, however deep
  the run goes. Back-propagation *within* the window does not change.

Every slot of a mini-batch walks its own story in lockstep. So one training
iteration is still one batch, and the log line stays `Batch b/N`. It adds
`Windows k/n (mean m)`:

- `n` is the number of windows of the longest story of the batch.
- `m` is the mean depth of the epoch so far. It shows if the curriculum moves.
  `1.0` is a frontier that never opens (each story trained to its first window
  only). `n` is a gate that never closes (`--no-frontier`, plain stateful TBPTT).

Validation runs the only regime that generation has: the state goes through each
whole story, ungated.

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

The flags of the example go after the trailing `--` (`-- --help` lists them).
The corpus knobs persist in the `training_config.json` of the artifacts. The
shared CLI, before the `--`, sets:

- the number of epochs,
- the batch size (`--batch-size`, the number of windows per optimizer step),
- the optimizer. The default is Muon + AdamW. `--adamw` keeps the hidden weight
  matrices on AdamW instead of [Muon](https://kellerjordan.github.io/posts/muon/)
  (see the README of `mnist-class`).

| Flag | Default | Meaning |
|------|---------|---------|
| `--seq-len <n>` | 256 | characters per window (the BPTT length) |
| `--run-len <n>` | `usize::MAX` | cap on the windows that one story can use (`1` ⇒ openings only) |
| `--frontier-bits <f>` | 1.6 | the threshold of the frontier gate, in bits per character |
| `--no-frontier` | off | carry the state through the whole story, ungated |
| `--train-stories <n>` | 4096 | stories taken from the train split |
| `--valid-stories <n>` | 256 | stories taken from the validation split |
| `--ssd-path <p>` | `recalc` | the SSD path of every chunkwise `forward`: `recalc`, `serial` or `minimal` (not persisted) |
| `--profile <n>` | off | print the mean ms of each training-step phase once per `n` windows (not persisted) |
| `--profile-sync` | off | with `--profile`: also sync the device after each phase |

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Sampling

`inference.rs` shows the three execution modes of the library, one after the
other:

1. One `prime()` replays the class latents. It has no input token, and it already
   returns the distribution of the first character.
2. Chunkwise `forward()`s read a prompt, if there is one. This is the prefill:
   right-padded 256-character chunks after the latents. The latents run once, and
   their state is kept.
3. Every generated character then costs one `step()` against that same cache:
   O(state) per token, with no growing KV cache.

Sampling is temperature-scaled multinomial over the full 48-way softmax
(`temperature <= 0` is greedy). `ChaCha8Rng` seeds it, so a run is reproducible.

`--inference` writes into `<artifacts>/inference/`:

- one primed, unprompted story per temperature (0.5 / 0.8 / 1.0),
- one continuation of each of three fixed prompts.

Training samples a short story at every small validation check, into
`<artifacts>/sample-epoch-{e}-batch-{b}.txt`. So you can see the text change from
noise to words to sentences. The checks are spaced in optimizer steps (every
300), so their rate does not change with the run lengths that the corpus gives.

## Notes

- The loss is reported in nats (the cross-entropy of Burn) and as **bits per
  character**. The uniform baseline is `log2(48) = 5.58` bits.
- The tied head starts *badly*. Burn initialises an `Embedding` from `N(0, 1)`. So
  at a `d_model` of 32, the initial logits have a variance of ~32, and the first
  batches score 25–40 bits/char instead of 5.58. This is a transient: the first
  steps shrink the embedding. But it uses the start of the LR schedule. An untied
  head (`missing_lm_head = false`) does not have this problem. The correct fix
  would be an initializer knob on the `VocabNetworkBuilder` of the library.
