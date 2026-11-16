# 20: The tests of the tap pattern move to `tap_pattern/tests.rs`

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of commit
`7807d31b`.

**Files:** `src/mamba3/tap_pattern.rs:363-1436`.

**Problem:** `tap_pattern.rs` holds its tests inline: a `mod tests { … }` of about
1070 lines with 24 tests, in a file of 1436 lines. The File Map rule of `CLAUDE.md`
puts the tests of `x.rs` in `x/tests.rs`. A reader of the tap pattern loads the tests
with it.

**Direction:** Move the module to `src/mamba3/tap_pattern/tests.rs`, with no change to
a test. `tap_pattern.rs` keeps `#[cfg(all(test, feature = "_dev-test"))] mod tests;`.

**Cost:** None in production. The coverage gate of `docs/agents/testing.md` must show
no lost condition.
