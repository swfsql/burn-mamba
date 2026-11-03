# 06: One traversal per cache, one fresh cache, one tap FIFO

Status: needs-triage

Blocked by: `../../block-family-choice/issues/04-block-enum.md`

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`. Not compiled.

**Files:** `src/unified/cache.rs:308-383`, `src/unified/capture.rs:37-168`,
`src/mamba3/double_ssd/cache.rs:108-165,249-297`, `src/mamba3/single_ssd/cache.rs:108-152,234-282`,
`src/mamba3/cache.rs:189-213`, `../burn-stack/src/modules/cache.rs:36-102,138-173`.

**Problem:** One more cached tensor costs about 20 edits that carry no knowledge.
The Mamba-3 fields have 5 lists, and 12 struct literals name all 6 fields. A fresh
cache has 2 sources, and 2 config structs hold the same 12 fields. `tropical_bh` has
37 non-test mentions in 8 files. A field missing from `cache_to_inner` shows only
when a cut crosses a config that holds this field.

The tap slots are two `Option`s, "present or absent together". 2 asserts and 10
`.expect` calls hold that rule.

**Direction, in steps:**

1. One fresh-cache constructor and one cache config. This step is the strongest.
2. The `CacheTensors` traversal becomes the one hand-written list. `cache_to_inner`,
   `cache_from_inner` and `cache_device` follow from it. Zero caches take the batch
   and the device.
3. One tap FIFO value, optional as a whole. It owns its slot order and its zeros.
4. One field group inside both Mamba-3 cache types.

Split step 2 by repo at triage.

**Holds:** ADR-0003: the list stays by hand, and both cache types stay.

**Cost:** `CacheTensors` becomes mandatory. A map through the visitor costs a
refcount per tensor per cut, and no tensor op. A nested `#[derive(Module)]` group is
not compiled. 39 test lines name the tap fields.
