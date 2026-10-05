# far use: next steps (auxiliary objectives)

The plan for the second half of the goal: long-context coherence, measured as
the **far use** of `--context-use` (`ORIENTATION.md`, Metric). This file
holds the ideas, the order of the work, and the rejected ideas with their
reasons. The results go to `notes.md`.

## The problem

- The best far use is 0.289 bits/char (h16 × g2 @14400). The far part costs
  0.768 bits/char after its own prompt and 1.057 after another prompt.
- The samples show the failure. In
  `infer-r128-lr1-9600-prompts/inference/sample-prompted-1-0.txt`:
  - The model keeps "jack", but it loses "alice" and "jane" after one line.
  - It writes "jick": the first letter is correct, the rest is not.
  - "jack was a very funny little girl": it does not bind attributes to names.
  - Its own phrase "part your words" becomes the topic. Two lines later, "the
    dirty words" are characters. Its own recent text wins over the prompt.
- Without a prompt (`…/sample-t0.8.txt`), "box", "tim" and "sam" hold for the
  whole story. So the model can hold a topic that it chose itself.
- The hard prompt is also out of distribution (three names, "alice", a
  question). Judge progress with in-distribution prompts, for example the
  first 256 chars of valid stories. Keep the hard prompt as a stress test.
- Topic-level far use is also word meaning in the distributional sense: what
  a word predicts many chars later. A win on topic is a win on meaning.

## The frame

- The LM loss already contains the far-use signal. Far use is the part of the
  LM loss that the prompt pays for. So an auxiliary loss does not change the
  best model. It changes where the gradient goes. Most of the gradient is
  local (spelling, grammar).
- An auxiliary loss must have the optimum of the LM loss (it is
  **consistent**). If it does not, it gets a small weight.
- **The bar.** The far use grows ~+0.03 per doubling of the steps, and its
  seed spread is ~0.01–0.017 @3230. An auxiliary loss pays if it gives more
  than +0.03 of far use at the same valid loss.
- **Judge by `bits(C | own)`** of the far part (2048 stories, paired with
  the control: the stable metric), by the valid loss and by the samples.
  Read the **gain** `bits(C | none) − bits(C | own)` and the far use against
  their noise (~0.01). The contrast trains `Δ`, so `Δ` can grow because
  `bits(C | other)` grows (the penalty), or because odd contexts score
  worse.

## Step 1: diagnostics (no training)

These are flex jobs, like `ctx.sh`, between GPU runs. Each one is a small
addition to `examples/tiny-stories/context.rs`.

1. **Coherence boosting** (Malkin et al. 2022). Score `C` with
   `log_softmax((1 + α)·z_full − α·z_noP)`. `z_noP` are the logits of `C`
   with no prompt (the class latents only). **Done** (`notes.md` §4):
   - No α > 0 lowers `bits(C | own)`. The readout is calibrated. Training
     makes α = 0 nearly stationary, so this does not decide capacity, and it
     does not test the writes (step 2).
   - The no-prompt pass splits the use into a **gain** (`none − own`) and a
     **penalty** (`other − none`). The far-use lead of g2 is all penalty.
   - Parked: the boosted samples of the hard prompt (α ≈ 0.3).
2. **The shuffle probe.**
   `Δ_shuffle = bits(C | own P, sentences shuffled) − bits(C | own P)`.
   Split `P` at ". " and keep the last sentence in place (the join). The bag
   of words stays the same. Only the order of the events changes. A
   `Δ_shuffle` near 0 means that the far use is a bag of sentences. This
   probe is the gate of step 3. **Answered** by `--contrast-eval` (the
   shuffled negative, `notes.md` §5): Δ_shuffle is 0.01–0.03 bits/char,
   ~6–8% of the far use.
3. **The distance curve** (Khandelwal et al. 2018). Replace the text farther
   back than `d` chars with the text of another story, for `d` ∈ {16, 32,
   64, 128, 256, 512}. The curve shows which timescales the model uses. It
   ties in with the heads hypothesis and with a longer `dt` init range.
4. Optional: **re-mention bits.** The bits of the words of `C` (3 letters or
   more) that also occur in `P`. This probe catches "jick". It needs no word
   list.

## Step 2: the shared-suffix contrast (stage 1: topic)

Score the same continuation `C` after two prompts:

```text
Δ    = log p(C | P_own) − log p(C | P_other)
L    = L_LM + λ · mean_t[ −log σ(Δ_≤t) ]
Δ_≤t = Σ_{s≤t} Δ_s          (Δ_s: the term of char s of C)
```

- **The spelling cancels.** `log p(C | P) = log p(C) + PMI(C; P)`. PMI
  (pointwise mutual information) is the bits that `P` saves on `C`. Both
  terms of `Δ` score the same chars, so the fluency term `log p(C)` cancels
  exactly. Only the part that depends on the prompt stays.
- **Consistent.** The true model is the best classifier of "which prompt
  wrote `C`". This holds only with a scale of 1 on `Δ`.
- **Bounded.** For one pair, the gradient is
  `σ(−Δ)·[∇log p(C | own) − ∇log p(C | other)]`. The weight `σ(−Δ)` goes to
  0 when the model separates the pair. A fixed weight ("LM × 2 + ascent")
  pushes without end.
- **The prefix form.** The sequence form saturates: `Δ ≈ 0.289 × 192 ≈ 55`
  bits per pair, so `σ(−Δ) ≈ 0`. The prefix terms give a gradient in the
  first words after the cut, where the classifier is not sure yet.
- **Cut at a sentence end**, or put a gap of unknowns after both prompts
  (step 4). Local syntax must not show the cut. If it does, the gradient
  trains "find the break" (the join costs +1.2 bits/char).
- **Do not detach the caches.** The prompt chars get no contrast loss, so
  nothing unlearns them. The gradient that goes back into the prompt trains
  what the model writes into its state.
- **Layout.** Use a small contrast batch per step, beside the packed LM
  batch, with the layout of `context.rs` (prompt, then continuation).
  - The own term comes from the forward of the contrast batch.
  - The other term runs `C` again from the caches at the cut, rolled by one
    row across the batch.
  - The cost is one more forward and backward of `C`.
  - Sentence cuts give prompts of different lengths. The `pad` mask (right
    padding) keeps the cache of each slot exact.
- **λ** is the knob, normalized so that λ = 1 pushes as hard as the LM loss
  at chance. It is 0 until `--contrast-start`, then ramps.
- **Built** in the example (`contrast.rs`, `notes.md` §5). The other term
  comes from the C rows rolled by one row, so the shared suffix needs no
  cache permutation. The LM backward runs first, then the contrast
  backward, and the gradients add up: the peak memory does not change.
- **Result: random negatives do not push.** At J 0 the join separates the
  pair at once (c01: a null run). On the far part (J 64) h16 tells a random
  other story apart within ~20–30 characters at every checkpoint from 250
  on (push 0.41 @250, 0.08 @1000, 0.02 @14400). A later start does not
  help. The limit is the easy negative, so step 3.

A model that learns only the topic is already a win. The current samples
change topic: a meeting, then "mummy", then "dirty words".

## Step 3: beyond the topic (stage 2: bag-matched negatives)

The gate: step 2 raises the far use, but `Δ_shuffle` stays near 0.

- Use negatives with the bag of words of `P`. Then only order and binding
  can separate the pair. Both kinds come from the data, with no word lists:
  - the own prompt with its sentences shuffled (the last sentence in place),
  - the nearest story by word overlap (computed once on the host).
- The precedent. BERT's next-sentence prediction used random negatives. The
  topic alone separated them, so it taught no coherence, and RoBERTa removed
  it with no loss. ALBERT's sentence-order prediction swapped the two
  segments (the same topic), and it helped.
- Use a curriculum: random negatives first, then bag-matched ones.
- **Built** (`--contrast-negative`): `shuffled` (the middle sentences move,
  the first and the last stay: a moved opening is a cheap sign),
  `shuffled-all`, and `nearest:M` (the most similar prompt of M, the
  prompts compared up to their own cuts).
- **Result.** Shuffled negatives push 0.4–0.8 of chance at every
  checkpoint. nearest:64 only ~2× random. c03 (shuffled, λ 1, against b01
  on 1 worker): the order Δ grows ×5 (0.009 → 0.044 bits/char), at an LM
  cost of ~0.008 (own far and dv). The use against random prompts does not
  move, and the no-prompt score gets worse (+0.018). That fits a shortcut:
  "unnatural context ⇒ predict worse".
- **Risk of every synthetic negative:** a shuffled prompt never occurs as
  an own prompt. So nothing stops the model from scoring all odd contexts
  lower. Judge by the own bits paired against the control, never by `Δ`.

## Step 4: unknown gaps (the Kalman reading)

- **The unknown** is a learnable `[d_model]` vector. It replaces the
  embedding at the masked positions, and autodiff trains it. Do not make it a
  vocab row. The head is tied and `pad_vocab_size_multiple` is 1, so a row
  becomes a softmax class.
- **The seam.** `MambaVocabNet::forward` takes ids, and the embedding is
  inside burn-stack's `VocabNetwork`. The unknown needs an entry that takes
  the embedded input, or an additive `[b, s, d]` input. That change is
  block-agnostic, so it goes to burn-stack.
- **What a gap means** (tally-drift). A gap is elapsed time: doubt with no
  evidence. After `k` gap tokens, the filter gives the past a weight of
  `1/(k·q)`, which never drops to 0. `Gain::KalmanProjectedNoise` separates
  evidence (`Δ`) from doubt (`q`). In text, the plot moves during skipped
  text, so `q_gap > 0` is correct. The data sets its value.
- **Use 1: the join control of the contrast.** Own `P` + `k` unknowns + `C`
  against other `P` + `k` unknowns + `C`. A sweep of `k` measures how much
  the prompt is still worth after `k` gap tokens. That is the quantity of
  tally-drift, for text.
- **Use 2: gapped LM rows.** On 10–25% of the rows, replace word- or
  sentence-aligned spans with unknowns. Mask the loss of the hidden chars.
  Keep the loss of the first char after the gap: it is pure far use.
- **Frequency** has two axes. The width chooses the timescale that must carry
  the prediction. The rate chooses the share of gapped signal, and so the
  mismatch with real text. Use a width of one word or more, at a low rate.
  Width 1 at a high rate trains a 2–3 char context only: skip it.
- **What carries over to text without gaps:** the writes before a gap and
  the reads after it. Low-evidence text ("they played and had fun") is a
  soft gap. A weak phrase must not overwrite the plot (the "dirty words"
  hijack).
- Later, a 2 × 2: {Projected, Kalman} × {no gaps, gaps}. Kalman costs 1.35×
  the step (`notes.md` §2).

## Parked

- **Distillation as a ceiling** (Hinton et al. 2015). Train a larger teacher
  once, then distill a 1M student, only as a diagnostic. If the student gets
  much more far use, the limit is the training signal. If not, the limit is
  the 1M shape. Training from scratch stays the method.
- **A word-clock layer** (SpaceByte, Slagle 2024). One layer runs only at the
  spaces. Its sequence is ~5× shorter, so each of its timescales covers ~5×
  more chars. This is structural work in the stack.
- **Training-only heads.** Multi-token prediction (Gloeckle et al. 2024 saw
  gains on byte-level models). A bag-of-words target after a gap (Zhao et al.
  2017, and TopicRNN by Dieng et al. 2017). Open question: does the 1M budget
  count a head that training drops?
- **Loss weights.** LongCE (Fang et al. 2024) weights each char by its gain
  from the long context. The weight depends on the target, so it biases the
  model to copy. Use mild weights only.

## Rejected

- **A shared prefix** (`P | C_correct` against `P | C_wrong`). Its `Δ` keeps
  the fluency difference `log p(C_correct) − log p(C_wrong)`. Its optimum is
  the PMI distribution `∝ p(C | P)/p(C)`, which penalizes generic text (the
  MMI objective of Li et al. 2016). DPO and CPO need a reference model to
  remove that term.
- **A detached cache at the cut.** It blocks the gradient that trains what
  the model writes. Mask the loss of the prompt instead.
- **Distractors** (a sentence of another story in the gap). That is false
  evidence, an outlier. A Kalman filter cannot reject outliers, so the model
  learns read-but-ignore. That generalizes as "distrust recent text", which
  also hits real new elements.
- **Engineered word lists** (names, objects, verbs) for edits or probes. The
  user prefers negatives and probes that come from the data alone.

## References

- Khandelwal et al. 2018, *Sharp Nearby, Fuzzy Far Away*: the far-use
  measure at word level. Far context acts as a bag of words.
- Fernandes et al. 2021, *Measuring and Increasing Context Usage in
  Context-Aware Machine Translation*. CXMI is the far use of MT. CoWord
  dropout is an input dropout, and it raised CXMI.
- Malkin et al. 2022, *Coherence Boosting*. Shi et al. 2023, *Context-Aware
  Decoding*. Both contrast the predictions with and without the context, at
  inference.
- Holtzman et al. 2021, *Surface Form Competition*: PMI scoring removes the
  fluency confound of answer choices.
- Li et al. 2016: MMI for dialogue, against generic answers.
- BERT (NSP), RoBERTa (NSP removed), ALBERT (sentence-order prediction),
  CONPONO (Iter et al. 2020, contrastive sentence objectives).
- Feng et al. 2025, CPO (arXiv 2502.16433): a shared prefix, batch
  negatives, a reference model, after pretraining (GPT2-XL, OpenLlama-3B).
- Fränken et al. 2024, SAMI (arXiv 2404.14313): a symmetric InfoNCE on LM
  log-probs. Its "which constitution wrote this response" direction is the
  shared suffix. Alignment fine-tuning at 7B–70B.
- van den Oord et al. 2018, CPC: InfoNCE against negatives from other
  sequences, on representations.
- Bowman et al. 2016: word dropout. Tay et al. 2022, UL2: short- and
  long-span corruption tasks.
