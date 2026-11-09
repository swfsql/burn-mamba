#!/usr/bin/env python3
"""Read the logs of `kernels.sh`: the report, and a diff per kernel of two runs.

Usage:
    scripts/kernels_report.py report OUT LOG_DIR RAN
    scripts/kernels_report.py diff [--old DIR] [--new DIR] [--case GLOB...]
    scripts/kernels_report.py --selftest

`kernels.sh` runs `report`. RAN holds `label:runs` items, separated by blanks.

`diff` finds which kernels caused a change of a launch count. It compares the
measured table of each case. Without `--old` and `--new`, it compares the
previous run of `kernels.sh` with the last run. When a label has several runs,
each kernel has a range. Two ranges that overlap show no change, so `diff` does
not list that kernel.
"""
import argparse
import collections
import datetime
import fnmatch
import pathlib
import re
import sys
import tempfile

from compare_reports import REPO, assert_lines, change_kind, run_selftests

# (label, column heading) in report order.
CONFIGS = [
    ("cuda", "cuda"),
    ("cuda-fusion", "cuda + fusion + autotune"),
]
GROUP_TITLES = {
    "forward": "`forward` — chunkwise prefill / inference",
    "train": "`train` — forward + backward (autodiff device)",
    "step": "`step` — one recurrent decode step",
}

# The syncs of a case without tuning: three warm-up iterations and the measured
# iteration.
SYNCS = 4

# The parser reads these kinds of line, in their order in the log:
#   - `Testing <case>`: the per-case banner of criterion under `--test`. The
#     tables after a banner belong to the case that it names. Attribution by
#     banner (not by position) keeps the surplus tables of a case away from
#     its neighbours.
#   - `| Name | ...`: the first line of a table.
#   - `| <kernel> | <duration> | <num computed> | <ratio> |`: one kernel. Kernel
#     names can contain `|`, so the fields are counted from the right.
#   - `| Total | <duration> | <num computed> | <ratio> |`: the last line of a
#     table.
BANNER = re.compile(r"^Testing (\S+)\s*$")
TOTAL = re.compile(r"^\| Total\s+\|.*?\|\s*(\d+)\s*\|\s*\d+ %\s*\|")
KERNEL = re.compile(r"^\| (.+?)\s+\| [0-9.]+\S*\s+\| (\d+)\s+\| \d+ %\s*\|$")

# The end of a kernel name identifies a variant, and a change of the code can
# change it. So the parser adds the counts of the variants of a kernel.
VARIANT = re.compile(r"_[0-9a-f]{8}$")

Table = collections.namedtuple("Table", "total kernels")


def parse_log(text, cases=None):
    """The tables of each case, in order: `{case: [Table, ...]}`.

    The tables before the first banner belong to no case, and the parser drops
    them. With `cases`, the parser also drops the tables of each other banner.
    """
    tables = {case: [] for case in cases or []}
    current, kernels = None, collections.Counter()
    for line in text.split("\n"):
        banner = BANNER.match(line)
        total = TOTAL.match(line)
        kernel = KERNEL.match(line)
        if banner:
            if cases is None:
                tables.setdefault(banner[1], [])
            current = banner[1] if banner[1] in tables else None
        elif line.startswith("| Name "):
            kernels = collections.Counter()
        elif total:
            if current is not None:
                tables[current].append(Table(int(total[1]), kernels))
            kernels = collections.Counter()
        elif kernel:
            kernels[VARIANT.sub("", kernel[1])] += int(kernel[2])
    return tables


def read_run(label, log, cases):
    """The text of one run, and the launch count of each case in that run."""
    text = log.read_text(errors="replace")
    tables = parse_log(text, cases)

    # The *shape* of the per-case counts (not their total) tells which failure
    # occurred. Every case runs the same code, so a changed sync point moves
    # all of them together. A killed run makes them uneven. Autotuning also
    # makes them uneven, but it is not a failure. A filter can select one case,
    # and one case has no shape to compare.
    counted = {c: len(t) for c, t in tables.items()}
    shape = set(counted.values())

    if shape == {0}:
        sys.exit(
            f"{label}: no summary table belongs to any of the {len(cases)} "
            f"cases. The `Testing <case>` banners of criterion do not match "
            f"`--list`, or profiling is off. See {log}"
        )
    short = sorted(c for c, n in counted.items() if n < SYNCS)
    if len(shape) == 1 and (short or (len(cases) > 1 and shape != {SYNCS})):
        n = shape.pop()
        sys.exit(
            f"{label}: every case emitted {n} summary table{'s'[:n != 1]}, "
            f"not {SYNCS}. The sync points in benches/layer.rs changed. See {log}"
        )
    if short:
        sys.exit(
            f"{label}: {len(short)} of {len(cases)} cases emitted fewer "
            f"than the {SYNCS} expected summary tables "
            f"({', '.join(short[:3])}). The run stopped early, or a table "
            f"came after the next banner. See {log}"
        )

    # The last two iterations run the same work. So different counts mean that
    # the last table cannot be trusted.
    unstable = sorted(c for c, t in tables.items() if t[-2].total != t[-1].total)
    if unstable:
        worst = ", ".join(
            f"{c} ({tables[c][-2].total} then {tables[c][-1].total})" for c in unstable[:3]
        )
        sys.exit(
            f"{label}: {len(unstable)} of {len(cases)} cases launched different "
            f"kernel counts in their last two iterations ({worst}). The two "
            "iterations run the same work, so the last table is not reliable. "
            f"Tuning or a change of block state can cause this. See {log}"
        )

    # The last table is the measured iteration, alone.
    return text, {case: tables[case][-1].total for case in cases}


def cell(counts):
    """One count, or the lowest and the highest count of runs that differ."""
    low, high = min(counts), max(counts)
    return str(low) if low == high else f"{low}–{high}"


def build_report(log_dir, ran):
    """The text of `kernels.md` and the number of its counts.

    `ran` gives the number of runs of each label.
    """
    # (group, case, label) -> the count of each run.
    results, config_lines, present = {}, {}, []
    for label, _ in CONFIGS:
        if label not in ran:
            continue
        logs = [log_dir / f"{label}.{run}.log" for run in range(1, ran[label] + 1)]
        cases_file = log_dir / f"{label}.cases"
        if not (cases_file.exists() and all(log.exists() for log in logs)):
            continue
        cases = cases_file.read_text().split()
        runs = [read_run(label, log, cases) for log in logs]

        present.append(label)
        for m in re.finditer(r"^bench-config: (.*)$", runs[0][0], re.M):
            config_lines[label] = m.group(1)
            break
        for case in cases:
            group, _, name = case.partition("/")
            results[(group, name, label)] = [counts[case] for _, counts in runs]

    cols = [(label, head) for label, head in CONFIGS if label in present]
    if not cols:
        sys.exit("no run logs found")

    notes = (
        "One iteration per case is measured rather than a criterion sample. A "
        "`cuda` count follows from the op graph, not from the machine, so it is "
        "exact and repeatable. The counts come from the per-sync profiling summary "
        "of cubecl, which serialises the queue. So the timings in "
        "`target/kernel-logs/` have no meaning."
    )
    if "cuda-fusion" in present:
        notes += (
            " A count of the autotune column also depends on the candidates that "
            "the tuner selects in each process, so it can change between runs. "
            f"That column comes from {ran['cuda-fusion']} run(s). A cell `a–b` "
            "gives the lowest and the highest count."
        )

    lines = [
        "# Kernel launch counts",
        "",
        f"One SSM block per case, generated by `./kernels.sh` on "
        f"{datetime.date.today().isoformat()}. Each cell is the number of GPU "
        "kernels launched by one iteration of that case. Lower is better.",
        "",
        notes,
        "",
    ]

    if config_lines:
        lines += ["| run | configuration |", "|---|---|"]
        for label, _ in cols:
            lines.append(f"| `{label}` | `{config_lines.get(label, 'n/a')}` |")
        lines.append("")

    for group in ["forward", "train", "step"]:
        cases = []
        for (g, case, label) in results:
            if g == group and case not in cases:
                cases.append(case)
        if not cases:
            continue
        lines += [
            f"## {GROUP_TITLES[group]}",
            "",
            "| case | " + " | ".join(head for _, head in cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for case in cases:
            row = [
                cell(results[(group, case, label)]) if (group, case, label) in results else "—"
                for label, _ in cols
            ]
            lines.append(f"| `{case}` | " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines), len(results)


def measured_tables(directory):
    """`{label: [{case: Table}, ...]}`: the measured table of each case, per run.

    A diagnosis can read a run that the report refuses, so this function does
    not apply the checks of `read_run`.
    """
    runs = collections.defaultdict(list)
    for path in sorted(directory.glob("*.log")):
        match = re.fullmatch(r"(.+)\.\d+\.log", path.name)
        if match:
            tables = parse_log(path.read_text(errors="replace"))
            runs[match[1]].append({case: t[-1] for case, t in tables.items() if t})
    if not runs:
        sys.exit(f"error: {directory} holds no log of kernels.sh.")
    return runs


def diff_case(case, old_tables, new_tables):
    """The lines of one case, or no line when no kernel rises or falls."""
    rows, overlaps = [], 0
    for name in sorted({name for table in old_tables + new_tables for name in table.kernels}):
        old = [table.kernels[name] for table in old_tables]
        new = [table.kernels[name] for table in new_tables]
        old_range, new_range = (min(old), max(old)), (min(new), max(new))
        if old_range == new_range:
            continue
        kind = change_kind(old_range, new_range)
        if kind == "overlap":
            overlaps += 1
        else:
            size = abs(sum(new) / len(new) - sum(old) / len(old))
            rows.append((-size, kind, cell(old), cell(new), name))
    if not rows:
        return []
    old_totals = cell([table.total for table in old_tables])
    new_totals = cell([table.total for table in new_tables])
    out = ["", f"{case}: {old_totals} -> {new_totals}"]
    for _, kind, old, new, name in sorted(rows):
        out.append(f"  {kind:6}{old:>10} -> {new:<10}{name}")
    if overlaps:
        kernels = "1 kernel" if overlaps == 1 else f"{overlaps} kernels"
        out.append(f"  Not listed: {kernels} with ranges that overlap.")
    return out


def diff_text(old_dir, new_dir, globs=None):
    """The text of the diff per kernel of two log directories."""
    old_label_runs, new_label_runs = measured_tables(old_dir), measured_tables(new_dir)
    known = {
        case
        for label_runs in (old_label_runs, new_label_runs)
        for runs in label_runs.values()
        for run in runs
        for case in run
    }
    for glob in globs or []:
        if not any(fnmatch.fnmatchcase(case, glob) for case in known):
            sys.exit(f"error: no case matches the glob `{glob}`.")

    out = []
    for label in sorted(old_label_runs.keys() & new_label_runs.keys()):
        old_runs, new_runs = old_label_runs[label], new_label_runs[label]
        cases = list(dict.fromkeys(case for run in new_runs + old_runs for case in run))
        if globs:
            cases = [c for c in cases if any(fnmatch.fnmatchcase(c, glob) for glob in globs)]
        body, compared, alone, differ = [], 0, [], 0
        for case in cases:
            old_tables = [run[case] for run in old_runs if case in run]
            new_tables = [run[case] for run in new_runs if case in run]
            if not (old_tables and new_tables):
                alone.append(case)
                continue
            compared += 1
            lines = diff_case(case, old_tables, new_tables)
            differ += bool(lines)
            body += lines
        old_count = f"{len(old_runs)} old run{'s'[:len(old_runs) != 1]}"
        new_count = f"{len(new_runs)} new run{'s'[:len(new_runs) != 1]}"
        out.append(f"## {label}: {old_count}, {new_count}, {differ} of {compared} cases differ")
        out += body
        if alone:
            out += ["", f"Only one side has: {', '.join(alone)}."]
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Read the logs of kernels.sh.")
    parser.add_argument("--selftest", action="store_true", help="run the tests of this script")
    commands = parser.add_subparsers(dest="command")

    report = commands.add_parser("report", help="write kernels.md from the logs of one run")
    report.add_argument("out", help="the path of the report")
    report.add_argument("log_dir", type=pathlib.Path, help="the directory of the logs")
    report.add_argument("ran", help="the `label:runs` items of this run, separated by blanks")

    diff = commands.add_parser("diff", help="compare the measured tables of two runs, per kernel")
    diff.add_argument(
        "--old",
        type=pathlib.Path,
        default=REPO / "target/kernel-logs.prev",
        help="the old logs. The default is the previous run of kernels.sh",
    )
    diff.add_argument(
        "--new",
        type=pathlib.Path,
        default=REPO / "target/kernel-logs",
        help="the new logs. The default is the last run of kernels.sh",
    )
    diff.add_argument(
        "--case",
        nargs="+",
        metavar="GLOB",
        help="only the cases that a glob matches, as section/case",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftests(SELFTESTS)
    elif args.command == "report":
        ran = {label: int(runs) for label, runs in (item.split(":") for item in args.ran.split())}
        text, count = build_report(args.log_dir, ran)
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({count} counts)")
    elif args.command == "diff":
        print(diff_text(args.old, args.new, args.case), end="")
    else:
        parser.error("give a command, or --selftest")


# Self-test: inline logs go in, and the report or the diff comes out.


def table(kernels):
    """One summary table of cubecl."""
    rows = [f"| {name:30} | 1.5µs | {count} | 10 % |" for name, count in kernels.items()]
    total = sum(kernels.values())
    head = "| Name | Duration | Num Computed | Ratio |"
    return [head, *rows, f"| Total | 1.5ms | {total} | 100 % |"]


def log(cases, before_the_first_banner=None):
    """The text of one run. `cases` maps a case to the kernels of each of its tables."""
    lines = ["bench-config: size=1"]
    if before_the_first_banner:
        lines += table(before_the_first_banner)
    for case, tables in cases.items():
        lines.append(f"Testing {case}")
        for kernels in tables:
            lines += table(kernels)
    return "\n".join(lines) + "\n"


def write_run(directory, label, runs, cases):
    """Write the logs of one label: the text of each run, and the list of cases."""
    (directory / f"{label}.cases").write_text("\n".join(cases) + "\n", encoding="utf-8")
    for run, text in enumerate(runs, start=1):
        (directory / f"{label}.{run}.log").write_text(text, encoding="utf-8")


def the_last_table_is_the_count_and_the_runs_give_the_range():
    measured = [{"x": 5}] * 4
    with tempfile.TemporaryDirectory() as name:
        directory = pathlib.Path(name)
        write_run(
            directory,
            "cuda",
            [log({"forward/a": measured, "train/b": [{"x": 6, "y": 4}] * 4}, {"x": 7919})],
            ["forward/a", "train/b"],
        )
        # The first run has one surplus table of the tuner.
        write_run(
            directory,
            "cuda-fusion",
            [log({"forward/a": [{"t": 77}] + [{"x": 3}] * 4}), log({"forward/a": [{"x": 4}] * 4})],
            ["forward/a"],
        )
        text, count = build_report(directory, {"cuda": 1, "cuda-fusion": 2})
    assert count == 3, count
    assert_lines(
        text,
        [
            "| `cuda` | `size=1` |",
            "## `forward` — chunkwise prefill / inference",
            "| case | cuda | cuda + fusion + autotune |",
            "| `a` | 5 | 3–4 |",
            "| `b` | 10 | — |",
        ],
    )
    assert "7919" not in text, text


def refusal(cases_of_the_log):
    """The message of a report that refuses a run with these tables."""
    with tempfile.TemporaryDirectory() as name:
        directory = pathlib.Path(name)
        write_run(directory, "cuda", [log(cases_of_the_log)], list(cases_of_the_log))
        try:
            build_report(directory, {"cuda": 1})
        except SystemExit as error:
            return str(error)
    raise AssertionError("the report accepted the run")


def different_counts_in_the_last_two_tables_are_refused():
    message = refusal({"forward/a": [{"x": 5}, {"x": 5}, {"x": 6}, {"x": 7}]})
    assert "forward/a (6 then 7)" in message, message


def a_killed_run_is_refused():
    message = refusal({"forward/a": [{"x": 5}] * 4, "forward/b": [{"x": 5}] * 2})
    assert "1 of 2 cases emitted fewer than the 4 expected" in message, message
    assert "The run stopped early" in message, message


def a_changed_number_of_syncs_is_refused():
    message = refusal({"forward/a": [{"x": 5}] * 3, "forward/b": [{"x": 5}] * 3})
    assert "every case emitted 3 summary tables, not 4" in message, message


def a_log_with_no_table_is_refused():
    message = refusal({"forward/a": [], "forward/b": []})
    assert "no summary table belongs to any of the 2 cases" in message, message


def surplus_tables_of_one_case_are_accepted():
    with tempfile.TemporaryDirectory() as name:
        directory = pathlib.Path(name)
        cases = {"forward/a": [{"x": 5}] * 4, "forward/b": [{"t": 1}] * 2 + [{"x": 8}] * 4}
        write_run(directory, "cuda", [log(cases)], list(cases))
        text, _ = build_report(directory, {"cuda": 1})
    assert_lines(text, ["| `a` | 5 |", "| `b` | 8 |"])


def the_diff_gives_each_kernel_that_changed():
    multiply = "KernelBinop | MulOp"
    old_runs = [
        {f"{multiply}_aaaaaaaa": 3, f"{multiply}_bbbbbbbb": 1, "CopyKernel_3a9881a9": 2, "Fuse_11111111": 10},
        {f"{multiply}_aaaaaaaa": 3, f"{multiply}_bbbbbbbb": 1, "CopyKernel_3a9881a9": 2, "Fuse_11111111": 12},
    ]
    new_run = {f"{multiply}_cccccccc": 6, "CopyKernel_3a9881a9": 1, "Fuse_11111111": 11}
    cases = ["train/a", "train/same"]
    with tempfile.TemporaryDirectory() as name:
        old_dir, new_dir = pathlib.Path(name, "old"), pathlib.Path(name, "new")
        old_dir.mkdir()
        new_dir.mkdir()
        write_run(
            old_dir,
            "cuda-fusion",
            [log({"train/a": [kernels] * 4, "train/same": [{"x": 5}] * 4}) for kernels in old_runs],
            cases,
        )
        write_run(
            new_dir,
            "cuda-fusion",
            [log({"train/a": [new_run] * 4, "train/same": [{"x": 5}] * 4})],
            cases,
        )
        text = diff_text(old_dir, new_dir)
        only_same = diff_text(old_dir, new_dir, ["train/same"])
    assert_lines(
        text,
        [
            "## cuda-fusion: 2 old runs, 1 new run, 1 of 2 cases differ",
            # Totals: 3 + 1 + 2 + 10 = 16, 3 + 1 + 2 + 12 = 18, 6 + 1 + 11 = 18.
            "train/a: 16–18 -> 18",
            "rise 4 -> 6 KernelBinop | MulOp",
            "fall 2 -> 1 CopyKernel",
            "Not listed: 1 kernel with ranges that overlap.",
        ],
    )
    assert "Fuse" not in text, text
    assert "train/same:" not in text, text
    assert_lines(only_same, ["## cuda-fusion: 2 old runs, 1 new run, 0 of 1 cases differ"])


def a_diff_of_a_directory_with_no_logs_is_an_error():
    with tempfile.TemporaryDirectory() as name:
        directory = pathlib.Path(name)
        try:
            diff_text(directory, directory)
        except SystemExit as error:
            assert str(error) == f"error: {directory} holds no log of kernels.sh.", error
            return
    raise AssertionError("no error for an empty directory")


SELFTESTS = [
    the_last_table_is_the_count_and_the_runs_give_the_range,
    different_counts_in_the_last_two_tables_are_refused,
    a_killed_run_is_refused,
    a_changed_number_of_syncs_is_refused,
    a_log_with_no_table_is_refused,
    surplus_tables_of_one_case_are_accepted,
    the_diff_gives_each_kernel_that_changed,
    a_diff_of_a_directory_with_no_logs_is_an_error,
]


if __name__ == "__main__":
    main()
