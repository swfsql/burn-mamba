# ORIENTATION: tiny-stories ≤1M-param search, the cased world

Workspace: `examples/tiny-stories/tmp/1m-cased/` (the paths below are
relative to it). Read `handoff.md` first.

## The cased world (user, 2026-10-08)

- The corpus keeps its case. A character is a pair: its **id** (the 48 ids
  of the case-folded alphabet) and its **case flag** (1 for `A-Z`, 0 for
  every other character). The vocabulary does not grow (user).
- The model reads the flag at its input: a learned vector (zero at init)
  that is added to the embedding where the flag is 1. A symbol always has
  the flag 0. The model also predicts the flag: one logit beside the 48
  vocab logits. A symbol has no case, so its flag prediction is not scored.
- The loss of a character is the cross-entropy of its id, plus the binary
  cross-entropy of its flag when it is a letter. So the loss is the
  information of the cased text.
- **No number of `../1m-b` compares with a number here** (user: a new
  world). The baselines start again.

## Metric

- The valid bits/char of the cased text: `bits` in `metrics.jsonl` (the
  log line: `bits = primary + case`). `primary_bits` is the part of the ids,
  `case_bits` the part of the flags (summed over the letters, divided by
  all the characters). Rank by the decayed `bits`. Report all three.
- The context use (`ctx.sh`) scores the cased text too: the bits of the id,
  plus the bits of the case of a letter. Coherence boosting acts on each of
  the two softmaxes. A re-mention word matches without regard to the case.
- The samples have capitals. `infer.sh` uses the user's two prompts with
  capitals.

## The same as `../1m-b`

The goal, the ceilings (1M params, VRAM), the working rules, the protocol
and the tools are those of `../1m-b/ORIENTATION.md`. Read it. These points
differ or are new:

- `build.sh` also records the local burn-stack. During this work,
  `Cargo.toml` patches burn-stack to `../burn-stack` (the user commits it).
- `--grad-checkpointing` (an example flag): about −700 MB at h16 for 2–5%
  more time, with the same losses. Use it where a shape needs the VRAM.
- `gen.py stages=` and `stage.py` take an anneal: `A-B:LR` is a cosine from
  the rate at A to LR at B. The LR plan of a new trunk:
  `stages=4630:1e-3,8000-12000:5e-4` (user: cosine anneals between rates).
- **Name a variant by its full change from the pool base**, in chat and in
  the notes: "h16 × r128 × rf 0.5 (d152)", never a bare shape.
- The text caches: `~/.cache/burn-dataset/tinystories-gpt4-clean/
  <split>-<n>-cased.txt` (the case-folded caches stay for the old builds).
- The files: `handoff.md` (the state and the next steps), `notes.md` (the
  results), `tmp.md` (the notes of the running session), `runs/`, `cfg/`,
  `bin/`, `ctx/` (the copies of `ctx.sh`). Git tracks the `*.md`, `*.sh`
  and `*.py` files only.
