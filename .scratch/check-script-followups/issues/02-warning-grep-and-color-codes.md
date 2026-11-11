# 02: `no_warnings` does not see a warning in CI

Status: needs-triage

Blocked by: None (can start immediately)

**What to build:** `no_warnings` of `scripts/check.sh` fails on each warning,
also when cargo writes color codes. `.github/workflows/check.yml` sets
`CARGO_TERM_COLOR: always`. Then a color code comes before the word of each
warning line, and `grep -q '^warning'` finds no match. So the check of CI
passes with warnings.

`../burn-mamba-example/scripts/check.sh` has a fix. Its `no_warnings` lets color
codes come before the word:

```bash
local esc=$'\e'
grep -qE "^(${esc}\[[0-9;]*m)*warning" <<<"$output"
```

Another fix is `CARGO_TERM_COLOR=never` for each cargo step of the check. Then
the CI log has no colors.

## Acceptance criteria

- [ ] `CARGO_TERM_COLOR=always ./scripts/check.sh` fails when a cargo step
      gives a warning.
- [ ] Without the variable, the check fails on a warning, and passes without
      one.

## Notes

- A throwaway test crate with one unused function showed the problem, with
  cargo 1.99.1 and `CARGO_TERM_COLOR=always`. The grep `^warning` found 0
  lines. The grep above found the 2 warning lines.
- `.github/workflows/docs.yml` sets the same variable, but it runs no warning
  grep.

## Comments
