# 22: An SSD path is a kind and a chunk length

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of commit
`7807d31b`.

**Files:** `src/mamba3/ssd_path.rs:30-155`, `src/mamba2/ssd/ssd_path.rs:19-30`,
`src/family/mod.rs:82-112`, `src/family/adapters.rs` (the two `reset_align`).

**Problem:** `Mamba2SsdPath` and `Mamba3SsdPath` each have three variants with the
same payload: `Minimal(Option<usize>)`, `Serial(Option<usize>)` and
`SerialRecalculated(Option<usize>)`. That is a product written as a sum. So
`chunk_len()` matches all three variants to read one field, and a test passes a
variant as `fn(Option<usize>) -> Mamba3SsdPath` to set the chunk length
(`src/family/tests/block_contract.rs`, `m3_path`).

**Direction:** A struct with two fields: the kind (`Minimal`, `Serial`,
`SerialRecalculated`) and the chunk length. Ticket 18 then adds the pathway as a third
field of the Mamba-3 struct.

**Cost:** A break of the public API at each site that names a variant: the examples
(`--ssd-path`), the tests and the benches. The change is mechanical. Do it with
ticket 18, or directly before it, so that the callers change one time.
