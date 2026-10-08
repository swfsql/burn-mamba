# hnet-stories

A character-level [H-Net](https://arxiv.org/abs/2507.07955) on the
TinyStories corpus of the [`tiny-stories`](../tiny-stories/README.md)
example. This example is a work in progress: its sizes and hyperparameters
are placeholders until a parameter search.

An H-Net does not read a fixed tokenization. Each **stage** runs a stack at
the resolution of its input, and a router decides where a chunk starts. The
stage keeps only the rows that start a chunk, the next level runs on that
shorter sequence, and the stage expands the result back. The innermost level
is the **main network**. The H-Net itself is block-generic and lives in
`burn_stack::modules::hnet` (its module header holds the math). This example
selects the families through `MambaHNetVocabNet`.

```bash
# train and then sample (downloads the 673MB parquet once, if not cached yet)
cargo run --release --example hnet-stories --features backend-flex -- --training --inference
# sample the trained model, with the chunk starts marked
cargo run --release --example hnet-stories --features backend-flex -- --inference \
    --artifacts-path <DIR> -- --show-chunks
```

## The model

`model.rs` holds the config. One stage is the default:

| Part | Family | Width | Stack |
|---|---|---|---|
| stage 0 (characters) | Mamba-2 | 32 | 2 encoder layers, 2 decoder layers, no MLP |
| stage 1 (`--stages 2`) | Mamba-2 | 48 | 2 encoder layers, 2 decoder layers, no MLP |
| main network | Mamba-3 | 64 | 4 layers with a SwiGLU MLP |

- Mamba-2 at character resolution and a different mixer inside follow the
  reference (which uses attention inside). `--stage-family mamba3` and
  `--main-family mamba2` change the families of a new model.
- The embedding is tied to the LM head, and a token is the pair (character,
  case flag), as in `tiny-stories`.
- Four learnable **start rows** open every story (`n_start_tokens`). They go
  in front of the H-Net, so it routes them like characters. The output of the
  last start row predicts the first character. So an unprompted sample needs
  no seed character. The stacks of an H-Net take no class latents.

## The training recipe

The model config (`HNetStoriesConfig`) also holds the recipe, so a resumed
run keeps it:

- **The ratio loss** of each stage steers the share of rows that start a
  chunk toward `1/Nˢ` (`ratio_targets`, 4 characters per chunk at stage 0).
  Its weight is `ratio_weight`: `0.3`, ten times the reference value. Early
  in training, the router of a model this small can start almost no chunk.
  The metrics report the next-character loss alone.
  `-- --chunk-log N` prints the chunk ratio of each stage every `N` windows.
- **A learning rate per stage.** Each stage trains at the rate of the
  schedule times `λˢ` (`lr_multipliers`). The default is the heuristic of the
  paper, normalised to `1` at the main network. A stage sees more rows per
  step and is narrower, so it takes a larger rate.
- **Runs of windows.** A story trains window by window, and the state carries
  from one window to the next (see "Runs and the frontier" in the
  `tiny-stories` README). The first window of a story opens with the start
  rows. An H-Net takes no packed rows (`--pack`): its chunks would move the
  resets.

## Sampling

`--inference` prints one story per temperature and a continuation of each
prompt (`-- --prompt <TEXT>`, repeatable).

- `-- --show-chunks` marks each character where a stage starts a chunk: `|`
  for stage 0, `‖` where stage 1 also starts one.
- `-- --step-mode masked` (the default) steps the inner network of a stage on
  every row. It keeps the old inner state on the rows that start no chunk.
  Every step then runs the same kernels, so on CUDA the decode replays one
  captured graph. `-- --step-mode gathered` steps the inner network only on
  the rows that start a chunk. It reads the decisions on the host, so it runs
  eagerly.

## Scaling up

The defaults are small enough for a CPU (`backend-flex`). For a larger model,
change the widths and depths in `model.rs` (a new artifacts directory takes
them), and train on `backend-cuda`.
