# TinyStories (character-level LM)

An auto-regressive Mamba-3 language model over single **characters** of
[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean),
a cleaned 2.7M-story subset of [TinyStories](https://arxiv.org/abs/2305.07759)
(GPT-4-generated children's stories, plain ASCII).

The model is deliberately tiny: two Mamba-3 blocks (`d_model = 32`,
`state_rank = 64`, `expand = 4`), cycled to a 4-deep virtual stack, between a
tied character embedding and its transpose. 39,496 parameters, of which the
embedding is 1,536.

## Vocabulary

The dataset's cleaning pipeline guarantees exactly 74 distinct ASCII characters:
the 52 cased letters plus ``\n !"$',-.0123456789:;?``. Case-folding the letters
leaves **48** tokens, and every one of them actually occurs — so the alphabet in
`dataset.rs` is the corpus's own inventory, not a slice of ASCII:

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
download happens once per `(split, story count)`.

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

The character stream is cut into non-overlapping windows of `seq_len + 1`; each
window is one training item, scored at **every** position against its next
character (so one window contributes `seq_len` classification examples, and the
reported accuracy is per character).

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

With the defaults (`seq_len = 256`, `batch_size = 16`) training needs ~1.3GB of
vram. Downstream flags, all forwarded after the trailing `--` and persisted into
the artifacts' `training_config.json`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--seq-len <n>` | 256 | characters per window (the BPTT length) |
| `--train-stories <n>` | 4096 | stories pulled from the train split |
| `--valid-stories <n>` | 256 | stories pulled from the validation split |
| `--epochs <n>` | 4 | passes over the corpus |
| `--batch-size <n>` | 16 | windows per optimizer step |
| `--muon` | off | hidden weight matrices on [Muon](https://kellerjordan.github.io/posts/muon/) instead of AdamW (see `mnist-class`'s README) |

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Results

4 epochs over the default corpus (3.36M characters, 3252 optimizer steps),
measured on the held-out validation split:

| Virtual layers | `grad_horizon` | Valid bits/char | Valid char accuracy | it/s |
|---|---|---|---|---|
| **4** (the default) | `None` | **1.749** | **63.2%** | ~7 |
| 16 | 4 | 3.208 | 36.1% | ~3.2 |

(uniform baseline: `log2(48) = 5.58` bits/char.)

The parameter count is identical — only how many times the 2 real weight sets are
applied, and how much of that is differentiated, differs. Truncated BPTT is a bad
deal for a language model specifically: it is scored at *every* position, so
leaving 12 of the 16 applications of a shared weight undifferentiated biases every
one of those readouts — unlike a task that reads out once, at the end of the
sequence. The 16-layer arm also plateaued at epoch 2 and then regressed, on
training loss as well as validation.

At 1.75 bits/char the samples are real words with real spelling and story
structure, and broken grammar — about what 39K parameters buys:

```text
once upon a time, there was a little girl named button to the fish and had a big
looked at the girl and said, "i said. the bird came to the boy named to play with
his friends. they like to play with his truck saw a big buy and had a big sing.
```

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
from noise into words into sentences.

## Notes

- Loss is reported both in nats (Burn's cross-entropy) and as **bits per
  character**; the uniform baseline is `log2(48) = 5.58` bits.
- The tied head starts *badly*: Burn initialises an `Embedding` from `N(0, 1)`,
  so at `d_model = 32` the initial logits have variance ~32 and the first batches
  score 25-40 bits/char instead of 5.58. It is a transient — the opening steps
  are spent shrinking the embedding — but it does eat the start of the LR
  schedule. An untied head (`missing_lm_head = false`) does not have it, and the
  proper fix would be an initializer knob on the library's `VocabNetworkBuilder`.
