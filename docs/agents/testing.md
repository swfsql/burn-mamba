# Test seams

`/to-spec` and `/tdd` agree the test seams of a change with the user. Propose seams from this table first. Propose a new seam only when no row fits the change.

| Seam | Tests | Catches | Misses |
|---|---|---|---|
| Form agreement (ADR-0007) | `ssd_path/tests.rs` of each SSD directory, `src/mamba3/single_ssd/single_ssd/tests.rs` | A fast form that differs from its simple form in outputs, final caches or gradients | A bug that all forms share |
| Forward/step parity | The `tests.rs` of `src/mamba1/mamba1/`, `src/mamba2/mamba2/`, `src/mamba3/double_ssd/double_ssd/`, `src/mamba3/single_ssd/single_ssd/` and `src/mamba3/product/` | A forward and a step that disagree, and an error in the cache | A bug that both calls share |
| Mechanism against its direct form | `src/mamba3/rotation/tests.rs`, `src/mamba3/single_ssd/token_band/tests.rs`, `src/mamba3/quat_scan/tests.rs`, `src/mamba3/helpers/tests.rs`, `src/mamba3/plant_frame/tests.rs` | A factored or fast mechanism that differs from its direct computation | A wrong definition |
| Containers with real blocks | `src/family/tests/` | A block that breaks a contract of a burn-stack container: forward/step parity, cache threading, gradient horizon, class markers, the Muon plan | An error in the math of a block that the contracts do not show |
| Example data and hand-set solutions | `tests.rs` of each example, run with `cargo test --examples` | An error in the data of an example, or a hand-set solution that fails | Whether training finds a solution |

## Expected values

An expected value comes from a source outside the code under test:

- A closed form or a definition. Examples: the half-angle closed form in `src/mamba3/rotation/tests.rs`, and `prefix_sum_matches_the_definition` in `src/mamba3/helpers/tests.rs`.
- The float64 `numpy` check of an `info/` note, in `scripts/`. The test cites the note, as `src/mamba3/ssd_path/tests.rs` does.

## Outside the seams

- A claim about training quality (for example, "this option learns faster") has no test seam. A run of an example answers it.
- The tests run on `flex`. A failure that only another backend shows needs a run on that backend.
