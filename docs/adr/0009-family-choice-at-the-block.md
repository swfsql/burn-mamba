# All three families stay selectable, and the choice sits at the block

Third-party projects select Mamba-1 and Mamba-2 at run time, although the
examples of this repo use only Mamba-3. So all three families stay equally
selectable, and low use in this repo is no reason to cut one. The run-time
choice sits at the block: one enum over the three blocks fills `Block`, and the
generic containers of burn-stack take it. Nothing about a network, a
bidirectional stack or an H-Net varies per family.

## Considered Options

- **One enum for each container, with one arm for each family**: each body only
  forwards, and each new container method needs a new pass-through.
- **A macro that writes those enums**: it keeps every pass-through.
- **The family types only**: a project must then select its family at compile
  time.

## Consequences

- A variant adds one segment to each parameter path, for example
  `layers.real_layers.0.block.Mamba3.in_proj.weight`. So a record of any other
  form does not load, and Burn reports the missing tensors. We give no
  migration.
- A project that knows its family at compile time can hold
  `LatentNetwork<Mamba3>`. The paths of its parameters have no variant segment.
- A stack holds one family. `init` takes one block config. A hand-built mixed
  stack panics at its first `forward`, because `Options` is one value. It also
  panics at a `step` with no cache, because the first block sizes each zero
  cache.
- The stages of an H-Net hold one family, and its main stack can hold another.
  `init` does not check this. `forward` panics on mixed stages, because `Options`
  is one value for all stages. `step` and `prime` take no `Options`, and each
  stage sizes its own zero caches. So they do not enforce the rule.
- `MambaCaches` is a flat list of slots, and each slot has its family tag. So
  `CacheStack::from_slots` guesses nothing.
