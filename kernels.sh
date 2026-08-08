#!/usr/bin/env bash
#
# kernels.sh — count the GPU kernel launches each benchmark case costs, and
# collect the counts into a comparison report (kernels.md).
#
# What is being counted
# ---------------------
# Every dispatch on a cubecl backend funnels through one function,
# `ComputeClient::launch_inner`, which consults the profiling logger. At level
# `basic` each `client.sync()` flushes a per-kernel summary table
# (`Name | Duration | Num Computed | Ratio`) and resets it, so the launches
# between two syncs are counted for us — no external profiler, and it works on
# every cubecl backend, not just CUDA.
#
# `benches/layer.rs` syncs exactly twice per case under `--test`: once after the
# warm-up loop (model construction + one iteration) and once at the end of
# `timed()` (the measured iteration, alone). So each case emits a *pair* of
# tables and the second of each pair is its per-iteration launch count.
#
# One run is enough
# -----------------
# A launch count is a property of the op graph, not of the machine: it is exact
# and repeatable, with none of the variance that makes criterion sample a
# benchmark hundreds of times. So this drives the same binary with criterion's
# `--test` mode (one iteration per case) and `BENCH_WARMUP_ITERS=1`, and the
# whole matrix takes about a minute per configuration — nearly all of it kernel
# compilation, not measurement.
#
# Two consequences of the `basic` level are worth knowing: it times every launch
# with `submit_blocking`, which serialises the queue (trust the counts, never
# the wall-clock, from this run), and autotuning launches each candidate, which
# is why the *second* table of each pair is the one read — by then the tuner has
# settled.
#
# Configurations
# --------------
#   cuda         + backend-cuda                          (no fusion, no autotune)
#   cuda-fusion  + backend-cuda,fusion,dev-autotune       (as deployed)
#
# `flex` is absent on purpose: it is not a cubecl backend, so it launches no
# kernels to count. Any other cubecl backend works — override the array below,
# or point BURN_DEVICE/features at wgpu, vulkan, metal, rocm or cpu.
#
# The target directories are shared with `bench.sh`, so if you have run that,
# nothing is rebuilt here.
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

# cubecl loads the nearest cubecl.toml walking up from the *current directory*,
# so the runs happen in a scratch dir carrying a profiling-enabled one. That
# leaves the repository's own cubecl.toml untouched.
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

# Only the configurations this invocation actually ran are reported: the log
# directory may still hold a skipped label's log from an earlier run, taken at a
# different filter or size, and silently mixing the two would be worse than
# leaving the column out.
RAN=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label features device target <<<"$entry"

    if [[ ",$SKIP," == *",$label,"* ]]; then
        echo "==> skipping $label"
        continue
    fi

    echo "==> $label — BURN_DEVICE=$device, features: default${features:+,$features}"

    # Build, then locate the binary: it has to be run from the scratch dir, and
    # `cargo bench` would run it from the package root instead.
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

    # The case list in the order criterion will run them, so the table pairs can
    # be attributed without hardcoding the cases here.
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

# The `| Total | <duration> | <num computed> | <ratio> |` line closing a table.
# Kernel names contain `|` themselves, so fields are counted from the right.
TOTAL = re.compile(r"^\| Total\s+\|.*?\|\s*(\d+)\s*\|\s*\d+ %\s*\|", re.M)

results, config_lines, present = {}, {}, []
for label, _ in CONFIGS:
    if label not in ran:
        continue
    log, cases_file = log_dir / f"{label}.log", log_dir / f"{label}.cases"
    if not (log.exists() and cases_file.exists()):
        continue
    text = log.read_text(errors="replace")
    cases = cases_file.read_text().split()
    counts = [int(m.group(1)) for m in TOTAL.finditer(text)]

    if len(counts) != 2 * len(cases):
        sys.exit(
            f"{label}: expected {2 * len(cases)} summary tables for "
            f"{len(cases)} cases, found {len(counts)} — the sync points in "
            f"benches/layer.rs changed; see {log}"
        )

    present.append(label)
    for m in re.finditer(r"^bench-config: (.*)$", text, re.M):
        config_lines[label] = m.group(1)
        break
    # Pairs are (model init + warm-up iteration, measured iteration); the
    # second is the clean per-iteration count.
    for case, measured in zip(cases, counts[1::2]):
        group, _, name = case.partition("/")
        results[(group, name, label)] = measured

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
