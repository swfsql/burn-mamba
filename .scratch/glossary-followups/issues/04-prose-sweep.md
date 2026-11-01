# 04: Sweep the prose and the identifiers for the glossary terms

Status: ready-for-agent

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

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      table permits, in the prose and in the identifiers. Each other hit that
      you keep is listed with its reason under `## Comments`.
- [ ] The acceptance `rg` of tickets 02 and 03 still finds only what those
      tickets permit.
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments

- Progress (partial). Done, with one commit for each row: gain, the lag-1 tap
  closes, state, accumulator, micro-step. Still to do: the rows from "write
  axis" down to "chunk boundary", the "segment"/"chunk"/"graph" sites below,
  then the full test suite, `cargo doc` and the review.
- Kept in the done rows:
  - `ssm_*` (`ssm_bhpr`, …): "SSM" alone is not an avoid term, and the field
    mirrors the Python `ssm_state`.
  - The SSD-kernel names `initial_state_bhpr`, `final_state_bhpr`, "chunk
    state" (K3) and "state passing" (K4): the SSD paths of both pathways share
    them.
  - "Step" for a procedure step ("Step 7", "the four steps"), compounds
    ("Euler step", "Newton step", "half-step", "optimizer step", "validation
    step"), the JSONL key `"step"`, the step counters and `total_steps` of
    the examples (the run logs and the LR schedules read them).
  - "Frontier gate" (tiny-stories): a trainer-side gate, not the gain or a
    tap. The β gates of MGR.
  - The CLI help copy in `examples/README.md` matches the pinned burn-stack
    rev `e40fcf5`. It changes when the pin moves.
  - `examples/reset/quintic/HANDOFF.md` is a session record, so it does not
    change.
- `../burn-stack/GLOSSARY.md` also lists "segment" (for a sub-projection),
  "chunk" (for a window) and "graph" (for a recording) under _Avoid_. The
  Comments of ticket 08 name some sites of these words that ticket 08 did not
  change. Find the others with `rg`.
