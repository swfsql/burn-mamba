#!/usr/bin/env bash
#
# bench.sh: run the single-block benchmarks in every backend configuration, and
# collect the results into a comparison report (bench.md).
#
# Configurations
# --------------
#   flex         + backend-cuda, run with BURN_DEVICE=flex   (CPU)
#   cuda         + backend-cuda                              (GPU, no fusion, no autotune)
#   cuda-fusion  + backend-cuda,fusion,dev-autotune          (GPU, as deployed)
#
# Two builds, three configurations
# --------------------------------
# `flex` and `cuda` share one build. Several backends can be compiled in at the
# same time, and `BURN_DEVICE` selects one of them at runtime. This holds in
# every group, also in `train`, whose custom backward dispatches through
# `#[backend_extension]`.
#
# But fusion is compile-time. `burn_cuda::Cuda` is a *type alias*:
# `CubeBackend<CudaRuntime>` usually, `Fusion<CubeBackend<CudaRuntime>>` under
# the `fusion` feature. `DispatchDevice::Cuda` is bound to that alias (there is
# no fusion *device* variant, unlike autodiff). So the fused build must be a
# different binary. Autotune is also a compile-time cubecl feature. Its runtime
# knobs set the tuning *level* and the cache, but they cannot turn it off.
#
# Each build keeps its own `CARGO_TARGET_DIR`. So a second run of this script
# rebuilds nothing, and the criterion baseline histories stay separate.
#
# Usage
# -----
#   ./bench.sh                    # all three configurations
#   ./bench.sh step               # only cases matching the criterion filter
#   BENCH_SEQ=1024 ./bench.sh     # any BENCH_* override the bench understands
#   BENCH_SKIP=cuda,cuda-fusion ./bench.sh   # skip configurations by label
#
# The report leaves out a skipped configuration. It does not use the previous
# log of that configuration, which can come from another filter or size.

set -euo pipefail
cd "$(dirname "$0")"

OUT="${BENCH_OUT:-bench.md}"
LOG_DIR="${BENCH_LOG_DIR:-target/bench-logs}"
FILTER="${1:-}"
SKIP="${BENCH_SKIP:-}"

mkdir -p "$LOG_DIR"

# label | extra cargo features (on top of the defaults) | BURN_DEVICE | target dir
# The first two rows are the same build (same features, same target dir). So it
# compiles once and then runs on two devices.
CONFIGS=(
    "flex|backend-cuda|flex|target/bench-cuda"
    "cuda|backend-cuda|cuda|target/bench-cuda"
    "cuda-fusion|backend-cuda,fusion,dev-autotune|cuda|target/bench-cuda-fusion"
)

# If both GPU rows are skipped, flex shares its build with nothing, so it gets a
# build of its own. Then a machine without CUDA can still run
# `BENCH_SKIP=cuda,cuda-fusion ./bench.sh`.
if [[ ",$SKIP," == *",cuda,"* && ",$SKIP," == *",cuda-fusion,"* ]]; then
    CONFIGS[0]="flex||flex|target/bench-flex"
fi

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
    CARGO_TARGET_DIR="$target" BURN_DEVICE="$device" \
        cargo bench ${features:+--features "$features"} --bench layer -- \
        --save-baseline "$label" ${FILTER:+"$FILTER"} \
        2>&1 | tee "$LOG_DIR/$label.log"

    RAN+=("$label")
done

# --------------------------------------------------------------------------
# Report: parse the criterion output of each run into one table per group.
# --------------------------------------------------------------------------
python3 - "$OUT" "$LOG_DIR" "${RAN[*]:-}" <<'PY'
import re, sys, datetime, pathlib

out_path, log_dir = sys.argv[1], pathlib.Path(sys.argv[2])
ran = set(sys.argv[3].split())

# (label, column heading) in report order.
CONFIGS = [
    ("flex", "flex (CPU)"),
    ("cuda", "cuda"),
    ("cuda-fusion", "cuda + fusion + autotune"),
]
GROUPS = ["forward", "train", "step"]
GROUP_TITLES = {
    "forward": "`forward` — chunkwise prefill / inference",
    "train": "`train` — forward + backward (autodiff device)",
    "step": "`step` — one recurrent decode step",
}

# `name  time: [lo unit mid unit hi unit]`, where a long name sits on its own line.
RESULT = re.compile(
    r"^(?P<group>forward|train|step)/(?P<case>\S+)\s+"
    r"time:\s+\[[\d.]+ \S+ (?P<mid>[\d.]+) (?P<unit>\S+) [\d.]+ \S+\]",
    re.M,
)
TO_MS = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1000.0}


def fmt(ms):
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms * 1000:.0f} µs"


results, config_lines, present = {}, {}, []
for label, _ in CONFIGS:
    if label not in ran:
        continue
    log = log_dir / f"{label}.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    present.append(label)
    for m in re.finditer(r"^bench-config: (.*)$", text, re.M):
        config_lines[label] = m.group(1)
        break
    for m in RESULT.finditer(text):
        ms = float(m.group("mid")) * TO_MS[m.group("unit")]
        results[(m.group("group"), m.group("case"), label)] = ms

cols = [(label, head) for label, head in CONFIGS if label in present]
if not cols:
    sys.exit("no run logs found")

lines = [
    "# Benchmark results",
    "",
    f"One SSM block per case, generated by `./bench.sh` on "
    f"{datetime.date.today().isoformat()}. Each cell is criterion's median "
    "wall-clock time per iteration; lower is better.",
    "",
    "Every measured iteration ends in a device sync, and each case runs untimed "
    "warm-up iterations first so kernel compilation and autotuning stay out of "
    "the samples.",
    "",
]

if config_lines:
    lines += ["| run | configuration |", "|---|---|"]
    for label, _ in cols:
        lines.append(f"| `{label}` | `{config_lines.get(label, 'n/a')}` |")
    lines.append("")

for group in GROUPS:
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
        row = [fmt(results.get((group, case, label))) for label, _ in cols]
        lines.append(f"| `{case}` | " + " | ".join(row) + " |")
    lines.append("")

pathlib.Path(out_path).write_text("\n".join(lines))
print(f"wrote {out_path} ({sum(1 for k in results)} measurements)")
PY
