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
# Under `--test`, `benches/layer.rs` syncs exactly twice per case:
#   1. after the warm-up loop (model construction + one iteration),
#   2. at the end of `timed()` (the measured iteration, alone).
# So each case emits a *pair* of tables, and the second table of each pair is
# its per-iteration launch count.
#
# One run is sufficient
# ---------------------
# A launch count is a property of the op graph, not of the machine. It is exact
# and repeatable, without the variance that makes criterion sample a benchmark
# hundreds of times. So this script runs the same binary in the `--test` mode
# of criterion (one iteration per case), with `BENCH_WARMUP_ITERS=1`. The whole
# matrix takes about a minute per configuration, almost all of it kernel
# compilation, not measurement.
#
# The `basic` level has two consequences that are worth knowing:
#   - It times every launch with `submit_blocking`, which serialises the queue.
#     So trust the counts of this run, but never its wall-clock times.
#   - Autotuning measures each candidate behind a sync of its own. So a *cold*
#     tuner emits one table per candidate, in addition to the pair. cubecl
#     namespaces its cache by its own version (~/.cache/cubecl), so the first
#     run after a burn upgrade tunes everything again. The script detects that
#     surplus and refuses it: under fusion, a candidate is a whole fused
#     segment, and its tables look the same as real ones. To fix it, run the
#     script again after the tuner has written its results.
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

# label | extra cargo features (on top of the defaults) | BURN_DEVICE | target dir
CONFIGS=(
    "cuda|backend-cuda|cuda|target/bench-cuda"
    "cuda-fusion|backend-cuda,fusion,dev-autotune|cuda|target/bench-cuda-fusion"
)

# The report has only the configurations that this invocation ran. The log
# directory can still hold the log of a skipped label from an earlier run, at a
# different filter or size. To mix the two silently is worse than to leave the
# column out.
RAN=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label features device target <<<"$entry"

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
    # table pairs, and does not have to hardcode the cases.
    ( cd "$WORK" && BURN_DEVICE="$device" "$bin" --list ${FILTER:+"$FILTER"} ) \
        2>/dev/null | sed -n 's/: benchmark$//p' >"$LOG_DIR/$label.cases"

    ( cd "$WORK" && BURN_DEVICE="$device" BENCH_WARMUP_ITERS=1 \
        "$bin" --test ${FILTER:+"$FILTER"} ) >"$LOG_DIR/$label.log" 2>&1

    RAN+=("$label")
done

# --------------------------------------------------------------------------
# Report: pair up the summary tables of each run into one table per group.
# --------------------------------------------------------------------------
python3 - "$OUT" "$LOG_DIR" "${RAN[*]:-}" <<'PY'
import re, sys, datetime, pathlib

out_path, log_dir = sys.argv[1], pathlib.Path(sys.argv[2])
ran = set(sys.argv[3].split())

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

results, config_lines, present = {}, {}, []
for label, _ in CONFIGS:
    if label not in ran:
        continue
    log, cases_file = log_dir / f"{label}.log", log_dir / f"{label}.cases"
    if not (log.exists() and cases_file.exists()):
        continue
    text = log.read_text(errors="replace")
    cases = cases_file.read_text().split()

    # case -> its summary tables, in order. The tables before the first banner
    # (the device throughput probe) belong to no case, and the parser drops
    # them.
    tables, current = {c: [] for c in cases}, None
    for m in EVENT.finditer(text):
        case = m.group("case")
        if case is not None:
            current = case if case in tables else None
        elif current is not None:
            tables[current].append(int(m.group("total")))

    # The *shape* of the per-case counts (not their total) tells which of the
    # three failures occurred. Every case runs the same code, so a changed sync
    # point moves all of them together. A cold tuner or a killed run makes them
    # uneven.
    counted = {c: len(t) for c, t in tables.items()}
    shape = set(counted.values())

    if shape == {0}:
        sys.exit(
            f"{label}: no summary table belongs to any of the {len(cases)} "
            f"cases. The `Testing <case>` banners of criterion do not match "
            f"`--list`, or profiling is off. See {log}"
        )
    if len(shape) == 1 and shape != {2}:
        n = shape.pop()
        sys.exit(
            f"{label}: every case emitted {n} summary table{'s'[:n != 1]}, "
            f"not 2. The sync points in benches/layer.rs changed. See {log}"
        )
    if shape != {2}:
        short = sorted(c for c, n in counted.items() if n < 2)
        if short:
            sys.exit(
                f"{label}: {len(short)} of {len(cases)} cases emitted fewer "
                f"than the 2 expected summary tables "
                f"({', '.join(short[:3])}). The run stopped early. See {log}"
            )
        surplus = sorted((n, c) for c, n in counted.items() if n > 2)
        worst = ", ".join(f"{c} ({n})" for n, c in reversed(surplus[-3:]))
        sys.exit(
            f"{label}: {len(surplus)} of {len(cases)} cases emitted more than "
            f"the 2 expected summary tables ({worst}). The autotune cache was "
            "cold (this is not a sync-point change). cubecl namespaces that "
            "cache by its own version (~/.cache/cubecl). So the first run after "
            "a burn upgrade measures every candidate again, and the sync of "
            "each candidate flushes its own table. The cache now holds those "
            f"results: run this script again. See {log}"
        )

    present.append(label)
    for m in re.finditer(r"^bench-config: (.*)$", text, re.M):
        config_lines[label] = m.group(1)
        break
    # A pair is (model init + warm-up iteration, measured iteration). The
    # second table is the clean per-iteration count.
    for case in cases:
        group, _, name = case.partition("/")
        results[(group, name, label)] = tables[case][1]

cols = [(label, head) for label, head in CONFIGS if label in present]
if not cols:
    sys.exit("no run logs found")

lines = [
    "# Kernel launch counts",
    "",
    f"One SSM block per case, generated by `./kernels.sh` on "
    f"{datetime.date.today().isoformat()}. Each cell is the number of GPU "
    "kernels launched by one iteration of that case; lower is better.",
    "",
    "Counts are exact and repeatable — they follow from the op graph, not from "
    "the machine — so one iteration per case is measured rather than a "
    "criterion sample. They are read from cubecl's own per-sync profiling "
    "summary, which serialises the queue: the counts are meaningful, the timings "
    "in `target/kernel-logs/` are not.",
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
            str(results.get((group, case, label), "—")) for label, _ in cols
        ]
        lines.append(f"| `{case}` | " + " | ".join(row) + " |")
    lines.append("")

pathlib.Path(out_path).write_text("\n".join(lines))
print(f"wrote {out_path} ({len(results)} counts)")
PY
