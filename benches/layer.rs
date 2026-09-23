//! # Single-block benchmarks (`cargo bench`)
//!
//! Each case is one SSM block, with no `Layer`/`Layers`/network wrapper. It is
//! measured in the three modes of every block:
//!
//! | Group | What it runs |
//! |-------|--------------|
//! | `forward` | `block_forward` on a plain device (chunkwise prefill / inference) |
//! | `train`   | `block_forward` + `loss.backward()` on an autodiff device |
//! | `step`    | one recurrent `block_step` from the cache of the previous step (decode) |
//!
//! The cases are Mamba-1, Mamba-2, and seven Mamba-3 cases. The Mamba-3 cases
//! cover the axes that have their own code paths:
//!
//! - `mimo_rank`,
//! - the `siso_specialization` branch choice at `mimo_rank == 1`,
//! - the rotation algebra,
//! - the SSD pathway (`mamba3/siso-double-ssd`).
//!
//! ## Running
//!
//! ```bash
//! cargo bench                                   # default features (flex)
//!
//! # CUDA as deployed, with kernel fusion and autotuning:
//! BURN_DEVICE=cuda cargo bench --features "backend-cuda,fusion,dev-autotune"
//!
//! # regression tracking (criterion stores baselines under target/criterion):
//! cargo bench -- --save-baseline flex
//! cargo bench -- --baseline flex                # report % change vs. that run
//! ```
//!
//! When several backends are compiled in, `BURN_DEVICE` selects the backend
//! for every group. This includes `train`, whose custom backward dispatches
//! through the `#[backend_extension(…)]` traits. So one build benches both flex
//! and CUDA. Only kernel fusion is compile-time, so it needs a separate build.
//! [`bench.sh`] runs all three configurations in that way.
//!
//! [`bench.sh`]: https://github.com/swfsql/burn-mamba/blob/main/bench.sh
//! [`kernels.sh`]: https://github.com/swfsql/burn-mamba/blob/main/kernels.sh
//!
//! Every case first runs [`warmup_iters`] untimed iterations. So kernel
//! compilation and autotuning are complete before criterion measures anything.
//! Each measured *batch* then submits all its iterations, and drains the device
//! once at the end ([`timed`]). So the bench measures an async backend at
//! steady state, not one submit-drain round trip at a time.
//!
//! The bench builds the block, its input and that warm-up inside the closure.
//! Criterion calls the closure only for the cases that pass its filter. So
//! `-- mamba2` touches nothing else, and [`kernels.sh`] can attribute kernel
//! launches to a single case.
//!
//! ## Sizing
//!
//! The defaults are small enough to finish on the CPU backends, and large
//! enough to be GEMM-bound on a GPU. To change them for one run, set these
//! environment variables:
//!
//! - `BENCH_BATCH`, `BENCH_SEQ`, `BENCH_D_MODEL`, `BENCH_STATE_RANK`,
//!   `BENCH_HEAD_DIM`: the shape,
//! - `BENCH_SAMPLES` / `BENCH_TIME_MS`: the sampling of criterion,
//! - `BENCH_WARMUP_ITERS` / `BENCH_SYNC_EVERY`: the warm-up and drain policy.
//!
//! ```bash
//! BENCH_SEQ=2048 BENCH_D_MODEL=1024 cargo bench --features backend-cuda -- forward
//! ```
//!
//! `train/mamba1` dominates a CPU-backend run. Mamba-1 backpropagates through a
//! sequential per-token scan. On the CPU backends, that backward grows
//! **quadratically** with `BENCH_SEQ` (its forward is linear). While you
//! iterate, filter it out (`cargo bench -- 'train/mamba[23]'`), or use a
//! shorter sequence for that group.

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

/// The problem size of every case (shared, so that the numbers of different
/// families are comparable).
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

    /// Print the effective configuration once per process, so that a bench log
    /// describes itself (`bench.sh` reads this line back into its report).
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
            // `Backend::name` nests the wrappers that are *compiled in*, for
            // example `dispatch<fusion<cubecl<cuda>>>` vs `dispatch<cubecl<cuda>>`.
            // So the log proves which flavour ran, and does not trust the
            // feature flags. (Fusion is a compile-time type alias in
            // `burn_cuda`, not a device property. Autodiff is different: it is
            // a device wrapper.)
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

/// Block until every queued operation has run.
///
/// The GPU backends are asynchronous. Without this, a measured iteration times
/// only the *submission* of the ops. The real work then lands in the next
/// iteration that synchronises.
fn sync(device: &Device) {
    device.sync().expect("device sync failed");
}

/// Time `iters` iterations of `work`, draining the device **once at the end**.
///
/// The cubecl backends are asynchronous. A sync inside every iteration measures
/// the submit-then-drain latency and serialises the queue: the CPU waits for
/// each kernel before it submits the next. A training loop does not feed the
/// GPU in that way. So this submits the whole batch and drains once, which
/// measures the steady-state throughput. Criterion divides the returned
/// duration by `iters`. The drain stays *inside* the timed region, so no work
/// escapes the measurement: it only amortises over the batch.
///
/// `BENCH_SYNC_EVERY=N` drains every `N` iterations (`0`, the default, drains
/// only at the end). Use it if the queued intermediates of a case exhaust the
/// device memory. `1` gives a sync per iteration.
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
/// A cubecl backend compiles a kernel on its first execution for a given
/// shape. With `dev-autotune`, it also *tunes* the kernel then. These one-off
/// costs must not land in a measured sample. The warm-up of criterion usually
/// absorbs them, but it has a time limit: on a slow case, it can fit only one
/// iteration. This is the explicit minimum (`BENCH_WARMUP_ITERS`, default 2).
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
    // The scan of Mamba-1 is sequential over the sequence, and its state is
    // small by design. So `state_rank` stays at the 16 of the paper, not at the
    // shared value.
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
        // backend preferences are different). The bench compares "all
        // specialized" with "all general", so it moves them together.
        .with_siso_specialization(siso_specialization)
        .with_siso_specialization_decode(siso_specialization)
}

/// The Mamba-3 cases, as `(name, config)`.
///
/// - `siso` / `mimo-rank1`: the same SISO block, with the specialized
///   `mimo_rank == 1` kernels on and off. This comparison shows whether the
///   specialization is worth it on this backend.
/// - `mimo-rank4`: real MIMO, where only the general kernels exist.
///
/// The last three sweep the rotation ladder against the `Complex2D` of `siso`:
///
/// - `real1d`: no rotation at all. No in-projection columns, no cumulative
///   accumulator, no `B`/`C` application. The other kinds are priced against
///   this floor.
/// - `quaternion4d`: the non-abelian rotation (an associative scan over the
///   sequence, not a `cumsum`).
/// - `rotor4d`: the full `SO(4)` rotation. The same scan over a doubled block
///   axis, plus one more quaternion product per `B`/`C` application.
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

/// A zero double-SSD cache. It selects the double-SSD pathway, because a
/// missing cache selects single-SSD.
fn double_ssd_cache(config: &Mamba3Config, batch: usize, device: &Device) -> Mamba3Cache {
    Mamba3DoubleSsdCacheConfig::new_from_block_config(batch, config.clone())
        .init(device)
        .into()
}

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

/// `forward` on a plain device. The call consumes its cache, so each iteration
/// needs a fresh one. But the bench *builds* it once, before the timed region,
/// and clones it in. A clone is a refcount increment, and a build is an
/// allocation and a zero-fill. Only `mamba3/siso-double-ssd` gives a real
/// cache. For the single-SSD default the cache is `None`, and every case pays
/// the same.
///
/// The bench builds the block, its input and the warm-up *inside* the closure.
/// Criterion calls the closure only when the case passes its filter. So
/// `-- mamba2` does not allocate or warm any other case, and one block is alive
/// at a time. Code moved out of the closure runs on every invocation of the
/// binary, for any filter. So the closure is what lets `kernels.sh` attribute
/// the kernel launches of a filtered run to the named case.
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
    M::Cache: Clone,
    M::Options: Clone,
    B: Fn(&Device) -> M,
    C: Fn(&Device) -> Option<M::Cache>,
{
    group.bench_function(name, |b| {
        let block = build(device);
        let x = input_3d(shape, device);
        let seed = cache(device);

        // Untimed: compile (and autotune) the kernels this case needs.
        for _ in 0..warmup_iters() {
            let (_y, _cache) = block.block_forward(x.clone(), seed.clone(), path.clone(), None);
            sync(device);
        }
        b.iter_custom(|iters| {
            timed(device, iters, || {
                let (y, _cache) = block.block_forward(x.clone(), seed.clone(), path.clone(), None);
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
    M::Cache: Clone,
    M::Options: Clone,
    B: Fn(&Device) -> M,
    C: Fn(&Device) -> Option<M::Cache>,
{
    group.bench_function(name, |b| {
        let block = build(device);
        let x = input_3d(shape, device);
        let seed = cache(device);

        // Untimed: the backward has kernels of its own to compile and tune.
        for _ in 0..warmup_iters() {
            let (y, _cache) = block.block_forward(x.clone(), seed.clone(), path.clone(), None);
            let _grads = y.powf_scalar(2.0).mean().backward();
            sync(device);
        }
        b.iter_custom(|iters| {
            timed(device, iters, || {
                let (y, _cache) = block.block_forward(x.clone(), seed.clone(), path.clone(), None);
                y.powf_scalar(2.0).mean().backward()
            })
        })
    });
}

/// One decode step, fed by the cache of the previous iteration (so the
/// recurrence advances, as it does during generation).
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

        // Untimed: warm the decode kernels, and advance the cache as a real
        // decode does (so the measured iterations see the steady state).
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

    // The other SSD pathway: the same block, with a double-SSD cache that
    // selects it (without a cache, the block selects single-SSD).
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

    // Decode on the double-SSD cache. (A single-SSD decode goes through this
    // same recurrence, so the two differ only by the cache conversion.)
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
