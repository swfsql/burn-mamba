#!/usr/bin/env bash
#
# kernels.sh: count the GPU kernel launches of each benchmark case, and collect
# the counts into a comparison report (kernels.md).
#
# What the script counts
# ----------------------
# Every dispatch on a cubecl backend goes through one function,
# `ComputeClient::launch_inner`, which reads the profiling logger. At level
# `basic`, each `client.sync()` flushes a per-kernel summary table
# (`Name | Duration | Num Computed | Ratio`) and resets it. So cubecl itself
# counts the launches between two syncs. This needs no external profiler, and
# it works on every cubecl backend, not only CUDA.
#
# Under `--test`, this script runs three warm-up iterations. So
# `benches/layer.rs` syncs four times per case:
#   1. after the first warm-up iteration (with the model construction),
#   2. after the second warm-up iteration,
#   3. after the third warm-up iteration,
#   4. at the end of `timed()` (the measured iteration, alone).
# The last table of a case is its per-iteration launch count. The table before
# it counts the same work, and the script compares the two.
#
# How many runs
# -------------
# A launch count on `cuda` is a property of the op graph, not of the machine.
# It is exact and repeatable, without the variance that makes criterion sample
# a benchmark hundreds of times. So one run is sufficient there. This script
# runs the same binary in the `--test` mode of criterion (one measured iteration
# per case), with `BENCH_WARMUP_ITERS=3`. The whole matrix takes about a minute
# per configuration, almost all of it kernel compilation, not measurement.
#
# A count on `cuda-fusion` also depends on the tuner (see below). So the script
# runs that configuration `KERNELS_FUSION_RUNS` times (3 by default), and the
# report gives the lowest and the highest count.
#
# The `basic` level has two consequences that are worth knowing:
#   - It times every launch with `submit_blocking`, which serialises the queue.
#     So trust the counts of this run, but never its wall-clock times.
#   - A sync flushes a table, so autotuning adds surplus tables. The first use
#     of a tuner key runs throughput probes and tuner candidates, and each of
#     them syncs. So a case can emit more than four tables.
#
# These builds do not have the `persistence` feature of cubecl. So the tuner
# results stay in memory, and each run tunes again from an empty cache. The
# tuner selects by time, so a different selection can change a `cuda-fusion`
# count between runs.
#
# Under autotune, the second iteration of a case can launch more kernels than
# the third. So the script runs three warm-up iterations, not fewer. It ignores
# the surplus tables and reads only the last two. Tuning in the last two
# iterations splits their tables or changes their counts. Then the two counts
# differ, and the script refuses the run.
#
# cubecl prints the tables from a separate task, and a sync does not wait for
# that task. So the last table of a case can appear after the banner of the
# next case. `BENCH_SETTLE_MS` makes the bench wait after the measured
# iteration, so that the task can print that table.
#
# Configurations
# --------------
#   cuda         + backend-cuda                          (no fusion, no autotune)
#   cuda-fusion  + backend-cuda,fusion,dev-autotune       (as deployed)
#
# `flex` is intentionally absent: it is not a cubecl backend, so it launches no
# kernels to count. Any other cubecl backend works. Change the array below, or
# set BURN_DEVICE/features to wgpu, vulkan, metal, rocm or cpu.
#
# The target directories are shared with `bench.sh`. So if you ran that script,
# this one rebuilds nothing.
#
# Usage
# -----
#   ./kernels.sh                    # both configurations, every case
#   ./kernels.sh step               # only cases matching the criterion filter
#   BENCH_SEQ=1024 ./kernels.sh     # any BENCH_* the bench understands
#   KERNELS_SKIP=cuda-fusion ./kernels.sh    # skip configurations by label
#   KERNELS_FUSION_RUNS=1 ./kernels.sh       # one fused run, so no range

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KERNELS_OUT:-kernels.md}"
LOG_DIR="${KERNELS_LOG_DIR:-target/kernel-logs}"
FILTER="${1:-}"
SKIP="${KERNELS_SKIP:-}"

mkdir -p "$LOG_DIR"

# cubecl loads the nearest cubecl.toml above the *current directory*. So the
# runs occur in a scratch dir with a cubecl.toml that enables profiling. The
# cubecl.toml of the repository does not change.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat >"$WORK/cubecl.toml" <<'EOF'
[profiling.logger]
level = "basic"
stdout = true
EOF

# label | extra cargo features (on top of the defaults) | BURN_DEVICE | target dir | runs
CONFIGS=(
    "cuda|backend-cuda|cuda|target/bench-cuda|1"
    "cuda-fusion|backend-cuda,fusion,dev-autotune|cuda|target/bench-cuda-fusion|${KERNELS_FUSION_RUNS:-3}"
)

# The report has only the configurations that this invocation ran. The log
# directory can still hold the log of a skipped label from an earlier run, at a
# different filter or size. To mix the two silently is worse than to leave the
# column out.
RAN=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label features device target runs <<<"$entry"

    if [[ ",$SKIP," == *",$label,"* ]]; then
        echo "==> skipping $label"
        continue
    fi

    echo "==> $label — BURN_DEVICE=$device, features: default${features:+,$features}"

    # Build, then find the binary. It must run from the scratch dir, and
    # `cargo bench` runs it from the package root.
    bin=$(CARGO_TARGET_DIR="$target" \
        cargo bench ${features:+--features "$features"} --bench layer \
        --no-run --message-format=json 2>/dev/null |
        python3 -c 'import json,sys
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("executable") and m.get("target", {}).get("name") == "layer":
        print(m["executable"])' | tail -1)

    if [[ -z "$bin" ]]; then
        echo "    could not locate the compiled bench binary" >&2
        exit 1
    fi

    # The case list, in the order of criterion. So the script can attribute the
    # tables, and does not have to hardcode the cases.
    ( cd "$WORK" && BURN_DEVICE="$device" "$bin" --list ${FILTER:+"$FILTER"} ) \
        2>/dev/null | sed -n 's/: benchmark$//p' >"$LOG_DIR/$label.cases"

    # Each process tunes from an empty cache.
    for run in $(seq 1 "$runs"); do
        ( cd "$WORK" && BURN_DEVICE="$device" BENCH_WARMUP_ITERS=3 BENCH_SETTLE_MS=200 \
            "$bin" --test ${FILTER:+"$FILTER"} ) >"$LOG_DIR/$label.$run.log" 2>&1
    done

    RAN+=("$label:$runs")
done

# --------------------------------------------------------------------------
# Report: read the last summary table of each case into one table per group.
# --------------------------------------------------------------------------
python3 - "$OUT" "$LOG_DIR" "${RAN[*]:-}" <<'PY'
import re, sys, datetime, pathlib

out_path, log_dir = sys.argv[1], pathlib.Path(sys.argv[2])
# label -> the number of its runs.
ran = {label: int(runs) for label, runs in (e.split(":") for e in sys.argv[3].split())}

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

# The parser reads two kinds of line, in their order in the log:
#   - `Testing <case>`: the per-case banner of criterion under `--test`. The
#     tables after a banner belong to the case that it names. Attribution by
#     banner (not by position) keeps the surplus tables of a case away from
#     its neighbours.
#   - `| Total | <duration> | <num computed> | <ratio> |`: the last line of a
#     table. Kernel names can contain `|`, so the fields are counted from the
#     right.
EVENT = re.compile(
    r"^Testing (?P<case>\S+)\s*$"
    r"|^\| Total\s+\|.*?\|\s*(?P<total>\d+)\s*\|\s*\d+ %\s*\|",
    re.M,
)

def read_run(label, log, cases):
    """The text of one run, and the launch count of each case in that run."""
    text = log.read_text(errors="replace")

    # case -> its summary tables, in order. The tables before the first banner
    # belong to no case, and the parser drops them.
    tables, current = {c: [] for c in cases}, None
    for m in EVENT.finditer(text):
        case = m.group("case")
        if case is not None:
            current = case if case in tables else None
        elif current is not None:
            tables[current].append(int(m.group("total")))

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
    unstable = sorted(c for c, t in tables.items() if t[-2] != t[-1])
    if unstable:
        worst = ", ".join(
            f"{c} ({tables[c][-2]} then {tables[c][-1]})" for c in unstable[:3]
        )
        sys.exit(
            f"{label}: {len(unstable)} of {len(cases)} cases launched different "
            f"kernel counts in their last two iterations ({worst}). The two "
            "iterations run the same work, so the last table is not reliable. "
            f"Tuning or a change of block state can cause this. See {log}"
        )

    # The last table is the measured iteration, alone.
    return text, {case: tables[case][-1] for case in cases}


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


def cell(counts):
    """One count, or the lowest and the highest count of runs that differ."""
    low, high = min(counts), max(counts)
    return str(low) if low == high else f"{low}–{high}"


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

pathlib.Path(out_path).write_text("\n".join(lines))
print(f"wrote {out_path} ({len(results)} counts)")
PY
