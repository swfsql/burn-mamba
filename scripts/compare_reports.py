#!/usr/bin/env python3
"""Compare `bench.md` and `kernels.md` with the reports of a revision.

Usage: scripts/compare_reports.py BASE [--new REV] --changed GLOB...

BASE and REV are git revisions. Without `--new`, the new side is the working
tree. A glob matches `section/case`, for example `train/mamba3/siso`, and a `*`
also matches a `/`. A changed cell is a cell that a glob matches. The change
does not touch the other cells, so the other cells show the noise of the run.

- Launches: each column gives the counts that are the same, and the counts that
  rise, fall or overlap. Two ranges `a–b` that overlap show no change.
- Time: each section and all the sections give the geometric mean and the range
  of new / old, for the changed cells and for the other cells.
- Noise: a changed cell is listed when it is outside the range of the other
  cells of its section. A section with few other cells uses its whole column.

The script reports and does not judge. So it exits with 0 when it can compare
the reports. A warning names a configuration that differs and the cells that are
on one side only.
"""
import argparse
import fnmatch
import math
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ["kernels.md", "bench.md"]
TO_MS = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1e3}

# The most cells that one warning names.
MAX_LISTED = 5

# A section with fewer other cells than this gives a weak range of the noise.
# Then the whole column gives the range.
MIN_SECTION_OTHERS = 3


def parse(text):
    """The cells of a report as `{(section, case, column): text}`, and its runs.

    The runs are the rows of the configuration table. A cell `—` has no value, so
    the cells do not hold it.
    """
    cells, runs, section, columns = {}, {}, None, None
    for line in text.splitlines():
        heading = re.match(r"## `(\w+)`", line)
        if heading:
            section, columns = heading[1], None
        elif line.startswith("|") and not line.startswith("|---"):
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if section is None:
                if parts[0] != "run":
                    runs[parts[0].strip("`")] = parts[1].strip("`")
            elif parts[0] == "case":
                columns = parts[1:]
            else:
                for column, cell in zip(columns, parts[1:]):
                    if cell != "—":
                        cells[(section, parts[0].strip("`"), column)] = cell
    return cells, runs


def mismatches(report, old, new):
    """The warnings about two reports that do not measure the same things."""
    (old_cells, old_runs), (new_cells, new_runs) = old, new
    out = []
    for run in sorted(old_runs.keys() & new_runs.keys()):
        if old_runs[run] != new_runs[run]:
            out += [
                f"WARNING: {report}: the configuration of `{run}` differs.",
                f"  old: {old_runs[run]}",
                f"  new: {new_runs[run]}",
            ]
    alone = sorted(old_cells.keys() ^ new_cells.keys())
    if alone:
        listed = ", ".join(f"{cell_name(key)} [{key[2]}]" for key in alone[:MAX_LISTED])
        more = f", and {len(alone) - MAX_LISTED} more" if len(alone) > MAX_LISTED else ""
        out.append(f"WARNING: {report}: {len(alone)} cells are on one side only: {listed}{more}.")
    return out


def bounds(cell):
    """The lowest and the highest count of a cell `a` or `a–b`."""
    parts = cell.split("–")
    return int(parts[0]), int(parts[-1])


def change_kind(old, new):
    """Compare two `(lowest, highest)` ranges. Two ranges that overlap show no change."""
    if new[0] > old[1]:
        return "rise"
    if new[1] < old[0]:
        return "fall"
    return "overlap"


def cell_name(key):
    return f"{key[0]}/{key[1]}"


def in_order(keys, index):
    """The distinct values of one field of the keys, in the order of the report."""
    return list(dict.fromkeys(key[index] for key in keys))


def launches(old, new, is_changed):
    out = ["## Launches"]
    for column in in_order(old, 2):
        keys = [key for key in old if key[2] == column and key in new]
        rows, same = {"rise": [], "fall": [], "overlap": []}, 0
        for key in keys:
            if old[key] == new[key]:
                same += 1
            else:
                rows[change_kind(bounds(old[key]), bounds(new[key]))].append(key)
        counts = ", ".join(f"{kind} {len(selected)}" for kind, selected in rows.items())
        out += ["", f"{column}: {len(keys)} counts, same {same}, {counts}"]
        for kind, selected in rows.items():
            for key in selected:
                mark = "*" if is_changed(key) else ""
                out.append(f"  {kind:8}{cell_name(key) + mark:34}{old[key]:>10} -> {new[key]}")
    return out


def millis(cell):
    value, unit = cell.split()
    return float(value) * TO_MS[unit]


def percent(ratio):
    return f"{(ratio - 1) * 100:+.1f}%"


def summary(ratios):
    """The geometric mean and the range of some ratios."""
    geomean = math.exp(sum(map(math.log, ratios)) / len(ratios))
    return f"{percent(geomean)} ({percent(min(ratios))} to {percent(max(ratios))})"


def times(old, new, is_changed):
    ratios = {key: millis(new[key]) / millis(old[key]) for key in old if key in new}
    columns = in_order(ratios, 2)
    head = " | ".join(columns) + " |"
    rule = "---|" * len(columns)

    def row(section, cells, select):
        """One table row, or nothing when the selection has no cell."""
        by_column = [
            [ratio for key, ratio in ratios.items() if key[2] == column and select(key)]
            for column in columns
        ]
        count = max(map(len, by_column), default=0)
        if count == 0:
            return []
        parts = " | ".join(summary(selected) if selected else "—" for selected in by_column)
        return [f"| {section} | {cells} n={count} | {parts} |"]

    out = ["## Time, new / old: geometric mean (lowest to highest)", ""]
    out += [f"| section | cells | {head}", f"|---|---|{rule}"]
    for section in in_order(ratios, 0):
        out += row(section, "changed", lambda key: key[0] == section and is_changed(key))
        out += row(section, "other", lambda key: key[0] == section and not is_changed(key))
    out += row("all", "changed", is_changed)
    out += row("all", "other", lambda key: not is_changed(key))

    out += ["", "## Time of each changed cell", "", f"| cell | {head}", f"|---|{rule}"]
    for section, case in dict.fromkeys(key[:2] for key in ratios if is_changed(key)):
        parts = []
        for column in columns:
            key = (section, case, column)
            parts.append(f"{percent(ratios[key])} ({old[key]} -> {new[key]})" if key in ratios else "—")
        out.append(f"| {section}/{case} | " + " | ".join(parts) + " |")

    out += ["", "## Changed cells outside the range of the other cells", ""]
    for key, ratio in ratios.items():
        if not is_changed(key):
            continue
        column = [r for k, r in ratios.items() if k[2] == key[2] and not is_changed(k)]
        section = [r for k, r in ratios.items() if k[0] == key[0] and k[2] == key[2] and not is_changed(k)]
        others, scope = (section, "section") if len(section) >= MIN_SECTION_OTHERS else (column, "column")
        if others and not min(others) <= ratio <= max(others):
            out.append(
                f"  [{key[2]}] {cell_name(key)}: {percent(ratio)}, {len(others)} other cells "
                f"of the {scope}: {percent(min(others))} to {percent(max(others))}"
            )
    return out


def compare(old, new, globs):
    """The text of the comparison. `old` and `new` map a report to its text."""
    parsed = {report: (parse(old[report]), parse(new[report])) for report in REPORTS}
    known = {key for sides in parsed.values() for side in sides for key in side[0]}
    for glob in globs:
        if not any(fnmatch.fnmatchcase(cell_name(key), glob) for key in known):
            sys.exit(f"error: no cell matches the glob `{glob}`.")

    def is_changed(key):
        return any(fnmatch.fnmatchcase(cell_name(key), glob) for glob in globs)

    out = [line for report, sides in parsed.items() for line in mismatches(report, *sides)]
    if out:
        out.append("")
    (old_kernels, _), (new_kernels, _) = parsed["kernels.md"]
    (old_bench, _), (new_bench, _) = parsed["bench.md"]
    out += launches(old_kernels, new_kernels, is_changed)
    out += [""] + times(old_bench, new_bench, is_changed)
    return "\n".join(out) + "\n"


def read(rev, path):
    """A report of a git revision, or of the working tree when `rev` is `None`."""
    if rev is None:
        return (REPO / path).read_text(encoding="utf-8")
    done = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
        capture_output=True,
        encoding="utf-8",
    )
    if done.returncode != 0:
        sys.exit(f"error: `git show {rev}:{path}` failed: {done.stderr.strip()}")
    return done.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Compare bench.md and kernels.md with the reports of a git revision."
    )
    parser.add_argument("base", nargs="?", metavar="BASE", help="the git revision of the old reports")
    parser.add_argument(
        "--new",
        metavar="REV",
        help="the git revision of the new reports. The default is the working tree",
    )
    parser.add_argument(
        "--changed",
        nargs="+",
        metavar="GLOB",
        help="the cells that the change reaches, as section/case. A `*` also matches a `/`",
    )
    parser.add_argument("--selftest", action="store_true", help="run the tests of this script")
    args = parser.parse_args()

    if args.selftest:
        run_selftests(SELFTESTS)
        return
    if args.base is None or args.changed is None:
        parser.error("give BASE and --changed")

    old = {report: read(args.base, report) for report in REPORTS}
    new = {report: read(args.new, report) for report in REPORTS}
    text = compare(old, new, args.changed)
    print(f"# {args.base} -> {args.new or 'working tree'}")
    print(f"\nChanged cells: {' '.join(args.changed)}. A `*` marks a changed cell.\n")
    print(text, end="")


# Self-test: inline reports go in, and the text of the comparison comes out.

OLD_KERNELS = """\
## `train` — forward + backward

| case | cuda | fused |
|---|---|---|
| `same` | 10 | 20 |
| `more` | 10 | 20 |
| `less` | 10 | 20 |
| `noisy` | 10 | 20–24 |
| `past` | 10 | 20–24 |
"""

NEW_KERNELS = """\
## `train` — forward + backward

| case | cuda | fused |
|---|---|---|
| `same` | 10 | 20 |
| `more` | 12 | 25 |
| `less` | 9 | 19 |
| `noisy` | 10 | 22–26 |
| `past` | 10 | 25 |
"""


def assert_lines(text, wanted):
    """Each wanted line is in the text. Runs of blanks count as one blank."""
    lines = {" ".join(line.split()) for line in text.splitlines()}
    missing = [line for line in wanted if line not in lines]
    assert not missing, f"missing {missing} in:\n{text}"


def launches_are_classified_by_their_ranges():
    text = compare(
        {"bench.md": "", "kernels.md": OLD_KERNELS},
        {"bench.md": "", "kernels.md": NEW_KERNELS},
        ["train/more"],
    )
    assert_lines(
        text,
        [
            "cuda: 5 counts, same 3, rise 1, fall 1, overlap 0",
            "fused: 5 counts, same 1, rise 2, fall 1, overlap 1",
            "rise train/more* 10 -> 12",
            "rise train/more* 20 -> 25",
            "rise train/past 20–24 -> 25",
            "fall train/less 10 -> 9",
            "fall train/less 20 -> 19",
            "overlap train/noisy 20–24 -> 22–26",
        ],
    )


OLD_BENCH = """\
## `train` — forward + backward

| case | flex | cuda |
|---|---|---|
| `a` | 1.00 ms | 500 µs |
| `b` | 1.00 ms | 2.00 s |
| `c` | 2.00 ms | 1.00 ms |
| `d` | 1.00 ms | 1.00 ms |

## `step` — one step

| case | flex | cuda |
|---|---|---|
| `e` | 1.00 ms | 1.00 ms |
"""

NEW_BENCH = """\
## `train` — forward + backward

| case | flex | cuda |
|---|---|---|
| `a` | 1.21 ms | 1.00 ms |
| `b` | 1.00 ms | 500.00 ms |
| `c` | 1.00 ms | 1.10 ms |
| `d` | 2.00 ms | 1.10 ms |

## `step` — one step

| case | flex | cuda |
|---|---|---|
| `e` | 1.00 ms | 1.00 ms |
"""


def times_give_the_mean_and_the_range_of_each_set_of_cells():
    text = compare(
        {"bench.md": OLD_BENCH, "kernels.md": ""},
        {"bench.md": NEW_BENCH, "kernels.md": ""},
        ["train/[ab]"],
    )
    assert_lines(
        text,
        [
            # flex: sqrt(1.21 * 1.00) = 1.10. cuda: sqrt(2.00 * 0.25) = 0.7071.
            "| train | changed n=2 | +10.0% (+0.0% to +21.0%) | -29.3% (-75.0% to +100.0%) |",
            "| train | other n=2 | +0.0% (-50.0% to +100.0%) | +10.0% (+10.0% to +10.0%) |",
            "| step | other n=1 | +0.0% (+0.0% to +0.0%) | +0.0% (+0.0% to +0.0%) |",
            "| all | changed n=2 | +10.0% (+0.0% to +21.0%) | -29.3% (-75.0% to +100.0%) |",
            # cuda: 1.21 ** (1 / 3) = 1.0656.
            "| all | other n=3 | +0.0% (-50.0% to +100.0%) | +6.6% (+0.0% to +10.0%) |",
            "| train/a | +21.0% (1.00 ms -> 1.21 ms) | +100.0% (500 µs -> 1.00 ms) |",
            "| train/b | +0.0% (1.00 ms -> 1.00 ms) | -75.0% (2.00 s -> 500.00 ms) |",
        ],
    )


OLD_NOISE = """\
## `forward` — forward

| case | cuda |
|---|---|
| `mamba3/siso` | 1.00 ms |
| `mamba2` | 1.00 ms |
| `one` | 1.00 ms |
| `two` | 1.00 ms |
| `three` | 1.00 ms |

## `train` — train

| case | cuda |
|---|---|
| `mamba3/siso` | 1.00 ms |
| `four` | 1.00 ms |
"""

NEW_NOISE = """\
## `forward` — forward

| case | cuda |
|---|---|
| `mamba3/siso` | 1.10 ms |
| `mamba2` | 1.02 ms |
| `one` | 1.01 ms |
| `two` | 1.02 ms |
| `three` | 1.03 ms |

## `train` — train

| case | cuda |
|---|---|
| `mamba3/siso` | 1.05 ms |
| `four` | 1.00 ms |
"""


def changed_cells_outside_the_noise_are_listed():
    text = compare(
        {"bench.md": OLD_NOISE, "kernels.md": ""},
        {"bench.md": NEW_NOISE, "kernels.md": ""},
        ["*/mamba[23]*"],
    )
    assert_lines(
        text,
        [
            "[cuda] forward/mamba3/siso: +10.0%, 3 other cells of the section: +1.0% to +3.0%",
            # `train` has one other cell, so the whole column gives the range.
            "[cuda] train/mamba3/siso: +5.0%, 4 other cells of the column: +0.0% to +3.0%",
        ],
    )
    assert "forward/mamba2:" not in text, text


OLD_PARTIAL = """\
| run | configuration |
|---|---|
| `cuda` | `sequence=256` |
| `flex` | `sequence=256` |

## `step` — one step

| case | cuda | flex |
|---|---|---|
| `a` | 1.00 ms | — |
| `b` | 1.00 ms | 1.00 ms |
| `gone` | 1.00 ms | 1.00 ms |
"""

NEW_PARTIAL = """\
| run | configuration |
|---|---|
| `cuda` | `sequence=1024` |
| `flex` | `sequence=256` |

## `step` — one step

| case | cuda | flex |
|---|---|---|
| `a` | 1.10 ms | 1.00 ms |
| `b` | 1.00 ms | 1.00 ms |
| `added` | 1.00 ms | 1.00 ms |
"""


def a_report_mismatch_gives_a_warning():
    text = compare(
        {"bench.md": OLD_PARTIAL, "kernels.md": ""},
        {"bench.md": NEW_PARTIAL, "kernels.md": ""},
        ["step/a"],
    )
    assert_lines(
        text,
        [
            "WARNING: bench.md: the configuration of `cuda` differs.",
            "old: sequence=256",
            "new: sequence=1024",
            "WARNING: bench.md: 5 cells are on one side only: step/a [flex], "
            "step/added [cuda], step/added [flex], step/gone [cuda], step/gone [flex].",
            # The `—` cell of `a` gives no ratio, and `b` is the other cell.
            "| step | changed n=1 | +10.0% (+10.0% to +10.0%) | — |",
            "| step/a | +10.0% (1.00 ms -> 1.10 ms) | — |",
        ],
    )
    assert "`flex` differs" not in text, text


def a_glob_with_no_cell_is_an_error():
    try:
        compare(
            {"bench.md": OLD_BENCH, "kernels.md": ""},
            {"bench.md": NEW_BENCH, "kernels.md": ""},
            ["train/[ab]", "train/typo"],
        )
    except SystemExit as error:
        assert str(error) == "error: no cell matches the glob `train/typo`.", error
        return
    raise AssertionError("no error for a glob with no cell")


def run_selftests(tests):
    """Run each test. A failed test sets the exit status."""
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except (Exception, SystemExit) as error:
            failed += 1
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
    if failed:
        sys.exit(f"{failed} of {len(tests)} tests failed")


SELFTESTS = [
    launches_are_classified_by_their_ranges,
    times_give_the_mean_and_the_range_of_each_set_of_cells,
    changed_cells_outside_the_noise_are_listed,
    a_report_mismatch_gives_a_warning,
    a_glob_with_no_cell_is_an_error,
]


if __name__ == "__main__":
    main()
