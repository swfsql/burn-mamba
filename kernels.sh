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
#
# Each configuration that runs moves its previous logs to `target/kernel-logs.prev`.
# A run with a filter also moves them. After a run, `scripts/kernels_report.py diff`
# gives the kernels that caused a change of a launch count.

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KERNELS_OUT:-kernels.md}"
LOG_DIR="${KERNELS_LOG_DIR:-target/kernel-logs}"
PREV_DIR="$LOG_DIR.prev"
FILTER="${1:-}"
SKIP="${KERNELS_SKIP:-}"

mkdir -p "$LOG_DIR" "$PREV_DIR"

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

    # Keep the logs of the previous run of this label. A failed build keeps
    # them in place.
    if compgen -G "$LOG_DIR/$label.*" >/dev/null; then
        rm -f "$PREV_DIR/$label."*
        mv "$LOG_DIR/$label."* "$PREV_DIR/"
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
python3 scripts/kernels_report.py report "$OUT" "$LOG_DIR" "${RAN[*]:-}"
