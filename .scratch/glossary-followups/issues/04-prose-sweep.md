# 04: Prose sweep for the glossary terms

Status: ready-for-agent

Blocked by: 02, 03

**What to build:** All prose in the crate uses the terms of `GLOSSARY.md` and
`../burn-stack/GLOSSARY.md`, and no term that they list under _Avoid_. A reader
finds one name for each concept.

## Scope

- Doc comments and comments in `src`, `benches` and `examples` (without the
  experiment directories `examples/*/tmp/`).
- Panic, assert and error messages. Before you change one, find the tests that
  match its text (`should_panic(expected = …)`), and keep the matched part or
  update the test.
- `info/`, README.md, examples/README.md, CLAUDE.md.
- Identifiers stay. List each identifier that conflicts with a glossary term
  (for example `check_each_slot_alone`) under `## Comments` for a later
  decision.

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
| "padded row", "real row", "pad row" (for a time position) | **padded position**, **real token**, **position** |
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
| leftovers of tickets 02 and 03 ("RoPE" for the rotation, "interior/far tap") | **rotation**, **lag-1 tap**, **lag-`u` tap** |
| "main network", "inner network", "encoder network", "decoder network", "level" in H-Net | **main stack**, **inner H-Net**, **encoder**, **decoder**, **stage** |
| bare "boundary" for the position where an H-Net chunk starts | **chunk boundary** |
| "start rows", "start tokens" for the learnable vectors that open an H-Net sequence | **start embedding** |

burn-stack ticket 03 swept the same H-Net terms
(`../burn-stack/.scratch/glossary-followups/issues/03-prose-sweep-hnet.md`).
Its section "Stale lines in the consumers" lists the H-Net prose of this
crate. Ticket 06 renames the H-Net identifiers of this crate, so this ticket
only changes prose.

## Method

Use the Edit tool, site by site. Each site needs a judgment, so no scripted
replacement. Write in the style of the `asd-ste100` skill.

## Acceptance criteria

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      table permits. Each other hit that you keep is listed with its reason
      under `## Comments`.
- [ ] `cargo check`, `cargo test --lib -- --test-threads=1`,
      `cargo test --examples -j 1 -- --test-threads=1` pass (flex).
- [ ] `cargo doc --all --no-deps` has no warnings.

## Comments
