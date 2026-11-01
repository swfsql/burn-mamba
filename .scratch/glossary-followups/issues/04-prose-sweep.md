# 04: Sweep the prose and the identifiers for the glossary terms

Status: resolved

Blocked by: 02, 03

**What to build:** All prose and all identifiers in the crate use the terms of
`GLOSSARY.md` and `../burn-stack/GLOSSARY.md`. They use no term that the
glossaries list under _Avoid_. A reader finds one name for each concept, in
the docs and in the code.

## Scope

- Doc comments and comments in `src`, `benches` and `examples` (without the
  experiment directories `examples/*/tmp/`).
- Panic, assert and error messages. Before you change one, find the tests that
  match its text (`should_panic(expected = …)`), and keep the matched part or
  update the test.
- `info/`, README.md, examples/README.md, CLAUDE.md.
- Identifiers too. Rename each identifier that uses an _Avoid_ term. Do it in
  the same commit as the prose of that term, with one commit for each row of
  the term map. Examples: `check_each_slot_alone`, `assert_slot`, `all_slots_flat`,
  the local `rows` of `show_chunks` (`examples/hnet-stories/inference.rs`).
- A word can also have a meaning that the term map permits. So decide each
  identifier on its own. Example: "slot" for a row of a padded batch changes
  to "row", but the cache slots of the tap FIFO (`slots_buhp`, `prev_slots`)
  keep "slot". List each identifier that keeps an _Avoid_ word, with its
  reason, under `## Comments`.
- A renamed config field needs no compatibility (as in tickets 02, 03 and 05).

## Term map (decided in the grill)

| Avoid | Use |
|---|---|
| "Kalman gate", "gate" for the decay | **gain** (a positive system). "Gate" is only the SiLU output gate `z`. The file `info/kalman/gate-as-positive-system.md` keeps its name, and its prose follows the terms. |
| "the gate" for the `Reset` gating of the lag-1 tap | "the lag-1 tap closes at token boundaries" |
| "hidden state", "SSM state", "recurrent state", "latent state" | **state** (the `h` of the plant) |
| "state" for the single-SSD running sum | **accumulator** |
| bare "step" for one recurrence step | **micro-step**. "Step" alone is only the step call. Write "step size" for `Δ` and "training step" for the optimizer. |
| "folded axis", "sequence axis" | **write axis** (one entry for each micro-step). The token axis is the **read axis**. |
| "sequence length" for the write-axis length | "sequence length" counts tokens only |
| "slot" for one sequence of a batch (padding) | **row** |
| "row" for a time position ("padded row", "real row", "pad row", the *lead rows* of `examples/common/context.rs`) | **padded position**, **real token**, **position** |
| "restart" as a synonym of reset | **reset**. "Restart" stays only for a chunk that starts with a reset. |
| "initial cache", "initial state" for the cache that a segment starts from | **origin** (it is not the learnable initial state) |
| "mixer", "mixer core", "SSM core", "core block" | **block**, or **plant** for the state recurrence |
| "variant", "version", "generation" for Mamba-1/2/3, and "cross-variant" | **family**, "cross-family" |
| "algorithm", "variant", "mode" for `Minimal`/`Serial`/`SerialRecalculated` | **SSD path** |
| "algorithm", "variant", "mode", "path" for double-SSD/single-SSD | **pathway** |
| "weight set" | **real layer** |
| "special token", "CLS", "register token" | **class marker**, **class token**, **class latent** |
| "document", "story", "episode" for a sequence in a packed row | **segment** (in the tiny-stories dataset, "story" stays for the data) |
| "DeltaProduct" for MambaProduct | **MambaProduct** (DeltaProduct stays as the name of the other mechanism) |
| "main network", "inner network", "encoder network", "decoder network", "level" in H-Net | **main stack**, **inner H-Net**, **encoder**, **decoder**, **stage** |
| bare "boundary" for the position where an H-Net chunk starts | **chunk boundary** |

burn-stack ticket 03 swept the same H-Net terms
(`../burn-stack/.scratch/glossary-followups/issues/03-prose-sweep-hnet.md`).
Its section "Stale lines in the consumers" lists the H-Net prose of this
crate. Ticket 06 renamed the H-Net identifiers of this crate, and it changed
their prose: "start embedding" and the padded-rows test of
`src/unified/tests/hnet.rs`. The other H-Net terms of the table are here.

Tickets 02 and 03 change the prose of their own renames. This ticket only
checks that they left no "RoPE" for the rotation and no "interior tap" or
"far tap".

## Method

Use the Edit tool, site by site. Each site needs a judgment, so no scripted
replacement. For an identifier that you decided to rename, `replace_all` in
one file is correct. Review every hit with `rg` afterwards. Write in the style
of the `asd-ste100` skill.

## Acceptance criteria

- [x] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      table permits, in the prose and in the identifiers. Each other hit that
      you keep is listed with its reason under `## Comments`.
- [x] The acceptance `rg` of tickets 02 and 03 still finds only what those
      tickets permit.
- [x] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [x] `cargo doc --all --no-deps` has no warnings.

## Comments

- Each row of the term map has its own commit (rows 16 and 17 share one).
  The rows "segment/story/episode" and "chunk boundary" needed no change: no
  packed-row "document" or "episode" is left, the "story" hits are the data
  of tiny-stories, and ticket 06 already wrote "chunk boundary" in the H-Net
  code. One more commit each covers the "segment", "chunk" and "graph" sites
  of ticket 08 (channel group, window, captured step and recording).
- Renamed identifiers: `kalman::gate` → `kalman::gain` (+ `GainCase`,
  `gain_case`, `run_gain`, `gain_matches_the_covariance_form_filter`),
  `no_stock_decay_solves_the_task`, `handmade_gain_solves_every_family`,
  `ChF::flag`, `tap_open_mask` (+ the closure `open_mask`),
  `RESET_AND_TWO_TAP`, `tokens_since_reset`, `write_len` (the `folded`
  lengths), `fresh_fields`, `assert_field`,
  `the_fields_survive_a_no_grad_round_trip`, `all_caches_flat`,
  `layer_cache_flat`, `layer_cache_sum`, `read_positions`,
  `scatter_read_positions`, `repeat_positions`, `kept_reads`,
  `scatter_reads`, `c_reads`, `gamma_reads`, `y_head_reads`, `off_reads`,
  the `positions` of `show_chunks`, `scan::Resets` (+ the fields `resets`),
  `block_only_layer_is_unchanged`, `PREFILL_WINDOW`, and the locals
  `train_step`, `increments`, `ulp`, `field`, `latent`, `at`, `parts`.
- Same-step γ correction → diagonal γ correction, after `diag.rs` and
  `y_diag_correction` (the File Map line of CLAUDE.md follows). The Rotation
  entry of `GLOSSARY.md` said "at each step": it says "at each micro-step".
- Kept, with the reason:
  - `ssm_*` (`ssm_bhpr`, …): "SSM" alone is not an avoid term, and the field
    mirrors the Python `ssm_state`.
  - The SSD-kernel names `initial_state_bhpr`, `final_state_bhpr`, "chunk
    state" (K3) and "state passing" (K4): the SSD paths of both pathways share
    them. "Initial cache" and "initial state" also stay where they name the
    cache that a call starts from, not the origin of a segment.
  - "Step" for the step call, a procedure step ("Step 7", "the four steps"),
    compounds ("Euler step", "Newton step", "half-step", "optimizer step",
    "validation step", "gradient step"), the JSONL key `"step"`, the step
    counters and `total_steps` of the examples (the run logs and the LR
    schedules read them).
  - "Gate" for the output gate `z`, the β gates of MGR, `GatedMlp`, and the
    "frontier gate" of tiny-stories (a trainer-side gate, not the gain).
  - `sequence` as an identifier: the notation table defines it as the
    write-axis length (Python `seqlen`).
  - "Slot" for the entries of the tap FIFO, the class-latent slots of a packed
    row, the symbol slots of a reset-spinor-product token, and the
    `slot_count`/`into_slots`/`from_slots` of the burn-stack `CacheStack`.
  - "Row" for a row of the batch, a packed row, a matrix row (`row(i)` of the
    SSD masks, the score) and a dataset row (the parquet of tiny-stories).
  - "Restart" for a chunk that starts with a reset: `Restarts`, `restart_bn`,
    `restart_heads`, `restart_window`, `restart_factor`, `d_restart_*`, and
    "the restart state" of a chunk.
  - "Variant" for a Rust enum variant ("the cache variant selects the
    pathway"), and "the single-SSD form" for the math form of the merged
    recurrence.
  - "Graph" for the autodiff graph and the tap graph. The flag `--no-graph`
    and `AppArgs::graphs` (with the `graphs` locals that carry it) belong to
    burn-stack.
  - The CLI help copy in `examples/README.md` matches the pinned burn-stack
    rev `e40fcf5`. It changes when the pin moves (the newer burn-stack text
    already says "training step").
  - `examples/reset/quintic/HANDOFF.md` is a session record, so it does not
    change.
- Outside the scope list, the sweep also changed what follows the renamed
  prose: the comments and labels of the numpy checks
  `scripts/kalman/gate_as_positive_system.py` (`gate` → `gain`),
  `scripts/mamba-3/trapezoid_as_integration.py` and
  `scripts/mamba-3/mimo_as_batch.py` (all still pass), ADR 0005, and the
  Rotation entry of `GLOSSARY.md`.
- The review round renamed `run_chunked_parity` and
  `chunked_forward_matches_single_forward*` to `run_split_parity` and
  `split_forward_matches_single_forward*`, the chunk length of
  `optimal_chunk_len` back to `chunk_len`, and the test locals of
  `last_micro_takes_the_read_position` to `writes`/`reads`.
- The docs of `n_caches` and `from_options` say "layer cache", but the
  `CacheStack` methods of burn-stack that they implement are still
  `slot_count`/`into_slots`/`from_slots`. A rename there belongs to
  burn-stack.
- Judgement calls of the review, not acted on: `kalman::gain` is a noun next
  to the enum `Gain`; `scan::Resets` holds the segment starts like
  `packing::Segments`; `RESET_AND_TWO_TAP` reuses "Reset" from the member
  names; the test helpers `kept_reads`/`scatter_reads` are copied in both
  `ssd_path/tests.rs` files; the two `Resets { … }` literals of `mamba3.rs`
  repeat each other.
- The commit "Prose sweep: channel group and window" names a `fold_micro`
  panic message. The messages are those of `unfold_micro_bs` and
  `unfold_micro_b` (the dcg hook blocks an amend).
- `../burn-stack/GLOSSARY.md` also lists "segment" (for a sub-projection),
  "chunk" (for a window) and "graph" (for a recording) under _Avoid_. The
  Comments of ticket 08 name some sites of these words that ticket 08 did not
  change. Find the others with `rg`.
