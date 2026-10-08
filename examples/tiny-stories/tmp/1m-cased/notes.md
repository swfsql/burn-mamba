# notes: tiny-stories ≤1M search, the cased world (results and findings)

The results of the cased world (see `ORIENTATION.md`). No number here
compares with a number of `../1m-b`.

## 1. The case channel (2026-10-08)

The design (user): a character is the pair (id, case flag). The input adds
a learned vector (zero at init) where the flag is 1, and a symbol always has
the flag 0. The output adds one case logit beside the 48 vocab logits. The
loss adds the binary cross-entropy of the flag on a letter. The vocabulary
does not grow. The aim: a mark at each name, against "jack and jack" (the
binding failure of `../1m-b`, `notes.md` §7).

- The code: burn-stack (the cased tokens and caches, the pair batches,
  `VocabNetwork`'s flag channel, the loss and its valid split, the
  sampler) and burn-mamba (`MambaVocabNet`, the example: `context.rs`
  scores the cased bits with one boost per softmax and matches the
  re-mention words without the case; `contrast.rs` is ported).
- The tests (flex): burn-stack 96 (3 ignored), with the loss against a
  host reference; burn-mamba lib packed + class 37; the example 14 (1
  ignored). `cargo doc` of both crates: no warnings.
- The GPU smoke (`v0.65.0-76b714f-dirty-2fdd53d8`): h16 3149 MB, 664 ms
  (with gc 2460 MB, 684 ms), as in `../1m-b`. 30 steps: valid `3.904
  bits/char: primary 3.749 + case 0.155`. A base rate of capitals (~4% of
  the letters, ~3/4 of the characters are letters) costs ~0.19, so the
  case already reads the context. The samples keep the capitals of the
  prompt, and capitals come only on letters.
- `runs/h16-s` is a partial baseline (235 steps, stopped by the user).
