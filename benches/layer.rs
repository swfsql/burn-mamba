//! # Single-block benchmarks (`cargo bench`)
//!
//! One SSM block per case — no `Layer`/`Layers`/network wrapper — measured in
//! the three modes every block exposes:
//!
//! | Group | What it runs |
//! |-------|--------------|
//! | `forward` | `block_forward` on a plain device (chunkwise prefill / inference) |
//! | `train`   | `block_forward` + `loss.backward()` on an autodiff device |
//! | `step`    | one recurrent `block_step` from the previous step's cache (decode) |
//!
//! Cases: Mamba-1, Mamba-2, and six Mamba-3 configurations covering the axes
//! that have their own code paths — `mimo_rank`, the `siso_specialization`
//! branch choice at `mimo_rank == 1`, the rotation algebra, and the SSD pathway.
//!
//! ## Running
//!
//! ```bash
//! cargo bench                                   # default features (flex)
//!
//! # CUDA as it is actually deployed — kernel fusion and autotuning on:
//! BURN_DEVICE=cuda cargo bench --features "backend-cuda,fusion,dev-autotune"
//!
//! # regression tracking (criterion stores baselines under target/criterion):
//! cargo bench -- --save-baseline flex
//! cargo bench -- --baseline flex                # report % change vs. that run
//! ```
//!
//! `BURN_DEVICE` picks the backend when several are compiled in, for every group
//! — including `train`, whose custom backward dispatches through the
//! `#[backend_extension(…)]` traits. So one build benches both flex and CUDA;
//! only kernel fusion, being compile-time, needs a build of its own. [`bench.sh`]
//! drives all three configurations that way.
//!
//! [`bench.sh`]: https://github.com/swfsql/burn-mamba/blob/main/bench.sh
//! [`kernels.sh`]: https://github.com/swfsql/burn-mamba/blob/main/kernels.sh
//!
//! Every case runs [`warmup_iters`] untimed iterations first, so kernel
//! compilation and autotuning are finished before criterion measures anything.
//! Each measured *batch* then submits all its iterations and drains the device
//! once at the end ([`timed`]) — an async backend is measured at steady state,
//! not one submit-drain round trip at a time.
//!
//! The block, its input and that warm-up are built inside the closure criterion
//! only calls for cases that pass its filter, so `-- mamba2` really does touch
//! nothing else — which is also what lets [`kernels.sh`] attribute kernel
//! launches to a single case.
//!
//! ## Sizing
//!
//! Defaults are small enough to finish on the CPU backends and still large
//! enough to be GEMM-bound on a GPU. Override per run with the environment:
//! `BENCH_BATCH`, `BENCH_SEQ`, `BENCH_D_MODEL`, `BENCH_STATE_RANK`,
//! `BENCH_HEAD_DIM`, plus `BENCH_SAMPLES` / `BENCH_TIME_MS` for criterion's
//! sampling and `BENCH_WARMUP_ITERS` / `BENCH_SYNC_EVERY` for the warm-up and
//! drain policy.
//!
//! ```bash
//! BENCH_SEQ=2048 BENCH_D_MODEL=1024 cargo bench --features backend-cuda -- forward
//! ```
//!
//! `train/mamba1` dominates a CPU-backend run: Mamba-1 backpropagates through a
//! sequential per-token scan, and on the CPU backends that backward grows
//! **quadratically** with `BENCH_SEQ` (its forward is linear). Filter it out
//! while iterating — `cargo bench -- 'train/mamba[23]'` — or shorten the
//! sequence for that group.

use burn::prelude::*;
use burn_mamba::mamba1::prelude::*;
use burn_mamba::mamba2::prelude::*;
use burn_mamba::mamba3::double_ssd::prelude::*;
use burn_mamba::mamba3::prelude::*;
use burn_mamba::prelude::*;
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The problem size every case is measured at (shared, so the numbers are
/// comparable across families).
#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: usize,
    sequence: usize,
    d_model: usize,
    state_rank: usize,
    per_head_dim: usize,
}

impl Shape {
    fn from_env() -> Self {
        Self {
            batch: env_usize("BENCH_BATCH", 2),
            sequence: env_usize("BENCH_SEQ", 256),
            d_model: env_usize("BENCH_D_MODEL", 256),
            state_rank: env_usize("BENCH_STATE_RANK", 64),
            per_head_dim: env_usize("BENCH_HEAD_DIM", 64),
        }
    }

    /// Tokens per `forward` / `train` iteration (criterion reports elem/s).
    fn tokens(&self) -> u64 {
        (self.batch * self.sequence) as u64
    }

    /// Print the effective configuration once per process, so a bench log is
    /// self-describing (`bench.sh` reads this line back into its report).
    fn announce(&self, device: &Device) {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let Self {
                batch,
                sequence,
                d_model,
                state_rank,
                per_head_dim,
            } = self;
            // `Backend::name` nests the wrappers that are *compiled in*, e.g.
            // `dispatch<fusion<cubecl<cuda>>>` vs `dispatch<cubecl<cuda>>`, so
            // the log proves which flavour ran instead of trusting the feature
            // flags. (Fusion is a compile-time type alias in `burn_cuda`, not a
            // device property — unlike autodiff, which is a device wrapper.)
            let backend =
                <burn::backend::Dispatch as burn::backend::Backend>::name(device.as_dispatch());
            eprintln!(
                "bench-config: batch={batch} sequence={sequence} d_model={d_model} \
                 state_rank={state_rank} per_head_dim={per_head_dim} \
                 warmup_iters={} backend={backend}",
                warmup_iters(),
            );
        });
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{key}: expected an integer, got {v:?}"))
        })
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Devices and timing
// ---------------------------------------------------------------------------

/// Block until every queued operation has actually run.
///
/// The GPU backends are asynchronous: without this a measured iteration would
/// only time the op *submission*, and the real work would land in whichever
/// iteration happens to synchronise next.
fn sync(device: &Device) {
    device.sync().expect("device sync failed");
}

/// Time `iters` iterations of `work`, draining the device **once at the end**.
///
/// The cubecl backends are asynchronous. Syncing inside every iteration would
/// measure submit-then-drain latency and serialise the queue — the CPU would
/// wait for each kernel before submitting the next, which is not how a training
/// loop feeds the GPU. Submitting the whole batch and draining once measures
/// steady-state throughput instead; criterion divides the returned duration by
/// `iters`. The drain stays *inside* the timed region, so no work escapes the
/// measurement — it just amortises over the batch.
///
/// `BENCH_SYNC_EVERY=N` drains every `N` iterations instead (`0`, the default,
/// drains only at the end). Use it if a case's queued intermediates exhaust
/// device memory; `1` restores a sync per iteration.
fn timed<T>(device: &Device, iters: u64, mut work: impl FnMut() -> T) -> Duration {
    let sync_every = env_usize("BENCH_SYNC_EVERY", 0) as u64;
    let start = Instant::now();
    for i in 0..iters {
        black_box(work());
        if sync_every != 0 && (i + 1) % sync_every == 0 {
            sync(device);
        }
    }
    sync(device);
    start.elapsed()
}

/// How many untimed iterations to run before criterion starts measuring.
///
/// A cubecl backend compiles a kernel on its first execution for a given shape,
/// and with `dev-autotune` it also *tunes* it then — one-off costs that must not
/// land in a measured sample. Criterion's own warm-up normally absorbs them, but
/// it is time-bounded: on a slow case it may fit only a single iteration. This
/// is the explicit floor (`BENCH_WARMUP_ITERS`, default 2).
fn warmup_iters() -> usize {
    env_usize("BENCH_WARMUP_ITERS", 2)
}

fn configure(group: &mut BenchmarkGroup<'_, WallTime>, tokens: u64) {
    group.throughput(Throughput::Elements(tokens));
    group.sample_size(env_usize("BENCH_SAMPLES", 10));
    group.warm_up_time(Duration::from_millis(
        env_usize("BENCH_TIME_MS", 5000) as u64 / 5,
    ));
    group.measurement_time(Duration::from_millis(
        env_usize("BENCH_TIME_MS", 5000) as u64
    ));
}

// ---------------------------------------------------------------------------
// Case configurations
// ---------------------------------------------------------------------------

fn mamba1_config(shape: Shape) -> Mamba1Config {
    // Mamba-1's scan is sequential over the sequence and its state is small by
    // design; `state_rank` stays at the paper's 16 rather than the shared one.
    Mamba1Config::new(shape.d_model).with_state_rank(16)
}

fn mamba2_config(shape: Shape) -> Mamba2Config {
    Mamba2Config::new(shape.d_model)
        .with_state_rank(shape.state_rank)
        .with_per_head_dim(shape.per_head_dim)
}

fn mamba3_config(
    shape: Shape,
    mimo_rank: usize,
    rotation: RotationKind,
    siso_specialization: bool,
) -> Mamba3Config {
    Mamba3Config::new(shape.d_model)
        .with_state_rank(shape.state_rank)
        .with_per_head_dim(shape.per_head_dim)
        .with_mimo_rank(mimo_rank)
        .with_rotation(rotation)
        // The chunkwise and per-token flags are independent knobs (their
        // backend preferences differ); the bench pairs "all specialized"
        // against "all general", so it moves them together.
        .with_siso_specialization(siso_specialization)
        .with_siso_specialization_decode(siso_specialization)
}

/// The Mamba-3 cases, as `(name, config)`.
///
/// - `siso` / `mimo-rank1`: the same SISO block, with the specialized
///   `mimo_rank == 1` kernels on and off — the head-to-head that says whether
///   the specialization pays off on this backend.
/// - `mimo-rank4`: genuine MIMO, where only the general kernels exist.
///
/// The last three sweep the rotation ladder against `siso`'s `Complex2D`:
///
/// - `real1d`: no rotation at all — no in-projection columns, no cumulative
///   accumulator, no `B`/`C` application. The floor the other kinds are priced
///   against.
/// - `quaternion4d`: the non-abelian rotation (an associative scan over the
///   sequence instead of a `cumsum`).
/// - `rotor4d`: the full `SO(4)` rotation — the same scan over a doubled block
///   axis, plus one extra quaternion product per `B`/`C` application.
fn mamba3_cases(shape: Shape) -> Vec<(&'static str, Mamba3Config)> {
    use RotationKind::{Complex2D, Quaternion4D, Real1D, Rotor4D};
    vec![
        ("mamba3/siso", mamba3_config(shape, 1, Complex2D, true)),
        (
            "mamba3/mimo-rank1",
            mamba3_config(shape, 1, Complex2D, false),
        ),
        (
            "mamba3/mimo-rank4",
            mamba3_config(shape, 4, Complex2D, true),
        ),
        ("mamba3/real1d", mamba3_config(shape, 1, Real1D, true)),
        (
            "mamba3/quaternion4d",
            mamba3_config(shape, 1, Quaternion4D, true),
        ),
        ("mamba3/rotor4d", mamba3_config(shape, 1, Rotor4D, true)),
    ]
}

/// A zero double-SSD cache — passing one selects the double-SSD pathway, since
/// a missing cache defaults to single-SSD.
fn double_ssd_cache(config: &Mamba3Config, batch: usize, device: &Device) -> Mamba3Cache {
    Mamba3DoubleSsdCacheConfig::new_from_block_config(batch, config.clone())
        .init(device)
        .into()
}

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

/// `forward` on a plain device. `cache` is rebuilt per iteration (caches are
/// consumed by the call); for the single-SSD default that is just `None`.
///
/// The block, its input, and the warm-up are all built *inside* the closure,
/// which criterion only calls when the case passes its filter — so `-- mamba2`
/// neither allocates nor warms any other case, and one block is alive at a
/// time. Anything hoisted out here would run on every invocation of the binary
/// however narrow the filter, which is what makes a filtered run's kernel
/// launches attributable to the case named (see `kernels.sh`).
fn run_forward<M, B, C>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    shape: Shape,
    device: &Device,
    build: B,
    cache: C,
    path: M::Options,
) where
    M: Block,
    M::Options: Clone,
    B: Fn(&Device) -> M,
    C: Fn(&Device) -> Option<M::Cache>,
{
    group.bench_function(name, |b| {
        let block = build(device);
        let x = input_3d(shape, device);

        // Untimed: compile (and autotune) the kernels this case needs.
        for _ in 0..warmup_iters() {
            let (_y, _cache) = block.block_forward(x.clone(), cache(device), path.clone());
            sync(device);
        }
        b.iter_custom(|iters| {
            timed(device, iters, || {
                let (y, _cache) = block.block_forward(x.clone(), cache(device), path.clone());
                y
            })
        })
    });
}

/// `forward` + `backward` on an autodiff device: one training iteration minus
/// the optimizer update.
fn run_train<M, B, C>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    shape: Shape,
    device: &Device,
    build: B,
    cache: C,
    path: M::Options,
) where
    M: Block,
    M::Options: Clone,
    B: Fn(&Device) -> M,
    C: Fn(&Device) -> Option<M::Cache>,
{
    group.bench_function(name, |b| {
        let block = build(device);
        let x = input_3d(shape, device);

        // Untimed: the backward has kernels of its own to compile and tune.
        for _ in 0..warmup_iters() {
            let (y, _cache) = block.block_forward(x.clone(), cache(device), path.clone());
            let _grads = y.powf_scalar(2.0).mean().backward();
            sync(device);
        }
        b.iter_custom(|iters| {
            timed(device, iters, || {
                let (y, _cache) = block.block_forward(x.clone(), cache(device), path.clone());
                y.powf_scalar(2.0).mean().backward()
            })
        })
    });
}

/// One decode step, fed by the cache the previous iteration produced (so the
/// recurrence really advances, as it would while generating).
fn run_step<M, B, C>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    shape: Shape,
    device: &Device,
    build: B,
    cache: C,
) where
    M: Block,
    B: Fn(&Device) -> M,
    C: Fn(&Device) -> Option<M::Cache>,
{
    group.bench_function(name, |b| {
        let block = build(device);
        let x = input_2d(shape, device);
        let mut cache = cache(device);

        // Untimed: warm the decode kernels, advancing the cache as a real
        // decode would (the steady state is what the measured iterations then
        // see).
        for _ in 0..warmup_iters() {
            let (_y, next) = block.block_step(x.clone(), cache.take());
            cache = Some(next);
            sync(device);
        }
        b.iter_custom(|iters| {
            timed(device, iters, || {
                let (y, next) = block.block_step(x.clone(), cache.take());
                cache = Some(next);
                y
            })
        })
    });
}

fn input_3d(shape: Shape, device: &Device) -> Tensor<3> {
    Tensor::random(
        [shape.batch, shape.sequence, shape.d_model],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn input_2d(shape: Shape, device: &Device) -> Tensor<2> {
    Tensor::random(
        [shape.batch, shape.d_model],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

fn bench_forward(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = Device::default();
    shape.announce(&device);

    let mut group = c.benchmark_group("forward");
    configure(&mut group, shape.tokens());

    run_forward(
        &mut group,
        "mamba1",
        shape,
        &device,
        |d| mamba1_config(shape).init(d),
        |_| None,
        (),
    );

    run_forward(
        &mut group,
        "mamba2",
        shape,
        &device,
        |d| mamba2_config(shape).init(d),
        |_| None,
        Mamba2SsdPath::default(),
    );

    let path3 = Mamba3SsdPath::default();
    for (name, config) in mamba3_cases(shape) {
        run_forward(
            &mut group,
            name,
            shape,
            &device,
            |d| config.init(d),
            |_| None,
            path3.clone(),
        );
    }

    // The other SSD pathway: same block, selected by handing it a double-SSD
    // cache instead of letting it default to single-SSD.
    let config = mamba3_config(shape, 1, RotationKind::Complex2D, true);
    run_forward(
        &mut group,
        "mamba3/siso-double-ssd",
        shape,
        &device,
        |d| config.init(d),
        |d| Some(double_ssd_cache(&config, shape.batch, d)),
        path3,
    );

    group.finish();
}

fn bench_train(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = Device::default().autodiff();
    shape.announce(&device);

    let mut group = c.benchmark_group("train");
    configure(&mut group, shape.tokens());

    run_train(
        &mut group,
        "mamba1",
        shape,
        &device,
        |d| mamba1_config(shape).init(d),
        |_| None,
        (),
    );

    run_train(
        &mut group,
        "mamba2",
        shape,
        &device,
        |d| mamba2_config(shape).init(d),
        |_| None,
        Mamba2SsdPath::default(),
    );

    let path3 = Mamba3SsdPath::default();
    for (name, config) in mamba3_cases(shape) {
        run_train(
            &mut group,
            name,
            shape,
            &device,
            |d| config.init(d),
            |_| None,
            path3.clone(),
        );
    }

    let config = mamba3_config(shape, 1, RotationKind::Complex2D, true);
    run_train(
        &mut group,
        "mamba3/siso-double-ssd",
        shape,
        &device,
        |d| config.init(d),
        |d| Some(double_ssd_cache(&config, shape.batch, d)),
        path3,
    );

    group.finish();
}

fn bench_step(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = Device::default();
    shape.announce(&device);

    let mut group = c.benchmark_group("step");
    // One token per sequence in the batch, not `batch · sequence`.
    configure(&mut group, shape.batch as u64);

    run_step(
        &mut group,
        "mamba1",
        shape,
        &device,
        |d| mamba1_config(shape).init(d),
        |_| None,
    );

    run_step(
        &mut group,
        "mamba2",
        shape,
        &device,
        |d| mamba2_config(shape).init(d),
        |_| None,
    );

    for (name, config) in mamba3_cases(shape) {
        run_step(
            &mut group,
            name,
            shape,
            &device,
            |d| config.init(d),
            |_| None,
        );
    }

    // Decode on the double-SSD cache. (Single-SSD decoding round-trips through
    // this same recurrence, so the two differ only by the cache conversion.)
    let config = mamba3_config(shape, 1, RotationKind::Complex2D, true);
    run_step(
        &mut group,
        "mamba3/siso-double-ssd",
        shape,
        &device,
        |d| config.init(d),
        |d| Some(double_ssd_cache(&config, shape.batch, d)),
    );

    group.finish();
}

criterion_group!(benches, bench_forward, bench_train, bench_step);
criterion_main!(benches);
