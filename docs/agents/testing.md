# Test seams

`/to-spec` and `/tdd` agree the test seams of a change with the user. Propose seams from this table first. Propose a new seam only when no row fits the change.

| Seam | Tests | Catches | Misses |
|---|---|---|---|
| Form agreement (ADR-0007) | `ssd_path/tests.rs` of each SSD directory, `src/mamba3/single_ssd/single_ssd/tests.rs` | A fast form that differs from its simple form in outputs, final caches or gradients | A bug that all forms share |
| Forward/step parity | The `tests.rs` of `src/mamba1/mamba1/`, `src/mamba2/mamba2/`, `src/mamba3/double_ssd/double_ssd/`, `src/mamba3/single_ssd/single_ssd/` and `src/mamba3/product/` | A forward and a step that disagree, and an error in the cache | A bug that both calls share |
| Mechanism against its direct form | `src/mamba3/rotation/tests.rs`, `src/mamba3/single_ssd/token_band/tests.rs`, `src/mamba3/quat_scan/tests.rs`, `src/mamba3/helpers/tests.rs`, `src/mamba3/plant_frame/tests.rs`, `src/row_end/tests.rs` | A factored or fast mechanism that differs from its direct computation | A wrong definition |
| Block contracts | `src/family/tests/block_contract.rs`: each block contract of burn-stack on each block setting. The other files of `src/family/tests/` test the containers with real blocks. | A block setting that breaks a contract, with a failure that names the contract and the setting | A container need that no contract states (the coverage report shows it), and a math error that the contracts do not show |
| Example data and hand-set solutions | `tests.rs` of each example, run with `cargo test --examples` | An error in the data of an example, or a hand-set solution that fails | Whether training finds a solution |

## Expected values

An expected value comes from a source outside the code under test:

- A closed form or a definition. Examples: the half-angle closed form in `src/mamba3/rotation/tests.rs`, and `prefix_sum_matches_the_definition` in `src/mamba3/helpers/tests.rs`.
- The float64 `numpy` check of an `info/` note, in `scripts/`. The test cites the note, as `src/mamba3/ssd_path/tests.rs` does.

## Outside the seams

- A claim about training quality (for example, "this option learns faster") has no test seam. A run of an example answers it.
- The tests run on `flex`. A failure that only another backend shows needs a run on that backend.

## Coverage gate

`scripts/check.sh` runs `../burn-stack/scripts/coverage.py` on this crate. The script gives the line, branch and condition coverage of `src/` by the lib tests. A box without that script, `cargo-llvm-cov` or the `llvm-tools-preview` of nightly skips it with a note.

The "Coverage gate" section of `../burn-stack/docs/agents/testing.md` gives the rules of a coverage gate. Run its compare from the root of this repo: `python3 ../burn-stack/scripts/coverage.py . --base BASE`. A commit has no `[patch]`, so a commit base builds against its pin.

Tickets 03–06 of `.scratch/container-tests-home/` move the container tests to burn-stack. Their gate is the record `tmp/coverage-record/record.md`. After each batch, the lib tests of the two repos together cover each item of the record, in the `src/` of each repo. `coverage.py` counts only the `src/` of the crate that it runs. So for the `src/` of burn-stack, the gate uses the throwaway script `tmp/coverage-record/dep_report.py`. The record gives the commands, and each batch ticket gives the place of its result.

## Bench gates

A ticket can gate on the reports of `./bench.sh` and `./kernels.sh`. Only the user runs them.

A ticket with a bench gate states two values:

- `BASE`: the commit that holds the reports before the change.
- The changed cells: the cells that the change reaches, as globs of `section/case`. An example is `'train/mamba[23]*'`. Quote each glob, because the shell can expand it.

After the user runs the benches, run `scripts/compare_reports.py BASE --changed 'GLOB'...`. Copy its numbers into the **Bench result** of the ticket. Do not write another comparison script. When a ticket needs a comparison that the script does not give, change the script.

- A launch gate reads the `cuda` column of `kernels.md`. A count of that column is exact and repeatable.
- The `cuda + fusion + autotune` column can change between runs, because the tuner selects by time. `kernels.sh` runs it three times, and a cell `a–b` gives the lowest and the highest count. Two ranges that overlap show no change.
- A time gate compares the changed cells with the other cells. The change does not touch the other cells, so their change shows the noise of the run.
- To find the kernels that caused a change of a launch count, run `scripts/kernels_report.py diff`. It compares the last two runs of `./kernels.sh`, per kernel.
