use super::*;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// Random inputs for the single-SSD. `da` is drawn from a
/// negative-mean distribution so the implied per-token decay `exp(da)`
/// stays in `(0, 1]`. `gamma` and `scale` are non-negative (matching
/// `Δ·σ(λ)`-style outputs of `helpers::trapezoidal_coefficients`).
#[allow(clippy::too_many_arguments)]
fn random_input(
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    read_stride: usize,
    mimo_rank: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
    random_init: bool,
    device: &Device,
) -> (
    Tensor<6>, // v
    Tensor<6>, // b
    Tensor<6>, // c
    Tensor<4>, // da
    Tensor<4>, // gamma
    Tensor<4>, // scale
    Tensor<4>, // initial_state
) {
    let v = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let b = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    // `C` and `γ` live on the chunk's read axis (one row per token).
    let chunk_tokens = chunk_len / read_stride;
    let c = Tensor::<6>::random(
        [batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let da = Tensor::<4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Normal(-0.5, 0.1),
        device,
    );
    let gamma = Tensor::<4>::random(
        [batch, nchunks, chunk_tokens, nheads],
        Distribution::Uniform(0.05, 0.5),
        device,
    );
    let scale = Tensor::<4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Uniform(0.05, 0.5),
        device,
    );
    // Random (general case) or zero (fresh-start) initial merged-form state
    // per `random_init`, covering the whole {zero, random} dimension.
    let initial_state = if random_init {
        Tensor::<4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        )
    } else {
        Tensor::<4>::zeros([batch, nheads, per_head_dim, state_rank], device)
    };
    (v, b, c, da, gamma, scale, initial_state)
}

struct Inputs {
    v: Param<Tensor<6>>,
    b: Param<Tensor<6>>,
    c: Param<Tensor<6>>,
    da: Param<Tensor<4>>,
    gamma: Param<Tensor<4>>,
    scale: Param<Tensor<4>>,
    initial_state: Param<Tensor<4>>,
    read_stride: usize,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    fn from_inner(
        v: Tensor<6>,
        b: Tensor<6>,
        c: Tensor<6>,
        da: Tensor<4>,
        gamma: Tensor<4>,
        scale: Tensor<4>,
        initial_state: Tensor<4>,
        read_stride: usize,
    ) -> Self {
        Self {
            v: Param::from_tensor(Tensor::from_inner(v)),
            b: Param::from_tensor(Tensor::from_inner(b)),
            c: Param::from_tensor(Tensor::from_inner(c)),
            da: Param::from_tensor(Tensor::from_inner(da)),
            gamma: Param::from_tensor(Tensor::from_inner(gamma)),
            scale: Param::from_tensor(Tensor::from_inner(scale)),
            initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            read_stride,
        }
    }

    fn ssd_input(&self, siso_specialization: bool) -> Mamba3SingleSsdInput {
        Mamba3SingleSsdInput {
            v_bnlmhp: self.v.val(),
            b_bnlmhr: self.b.val(),
            c_bntmhr: self.c.val(),
            da_bnlh: self.da.val(),
            gamma_bnth: self.gamma.val(),
            scale_bnlh: self.scale.val(),
            initial_state_bhpr: self.initial_state.val(),
            // Serial asserts this is None — see single_ssd_serial.
            init_state_hpr: None,
            read_stride: self.read_stride,
            siso_specialization,
        }
    }
}

struct PathRun {
    y: Tensor<6>,
    state: Tensor<4>,
    d_v: Tensor<6>,
    d_b: Tensor<6>,
    d_c: Tensor<6>,
    d_da: Tensor<4>,
    d_gamma: Tensor<4>,
    d_scale: Tensor<4>,
    d_init_state: Tensor<4>,
}

// ---------------------------------------------------------------------------
// The read axis, spelled out row by row
//
// `helpers::read_rows` / `scatter_read_rows` are what the read-axis tests below
// check, so they are re-derived here one `narrow` at a time rather than reused —
// a test that calls the implementation it is verifying proves nothing about it.
// ---------------------------------------------------------------------------

/// The folded positions a read row sits at: `i·stride + (stride − 1)`.
fn kept_rows<const D: usize>(t: Tensor<D>, dim: usize, stride: usize) -> Tensor<D> {
    let rows = t.dims()[dim] / stride;
    Tensor::cat(
        (0..rows)
            .map(|i| t.clone().narrow(dim, i * stride + stride - 1, 1))
            .collect(),
        dim,
    )
}

/// [`kept_rows`]' transpose: put each row back where it came from, zero between.
fn scatter_rows<const D: usize>(t: Tensor<D>, dim: usize, stride: usize) -> Tensor<D> {
    let dims = t.dims();
    let device = t.device();
    let mut gap = dims;
    gap[dim] = stride - 1;
    let mut parts = Vec::with_capacity(dims[dim] * 2);
    for i in 0..dims[dim] {
        parts.push(Tensor::zeros(gap, &device));
        parts.push(t.clone().narrow(dim, i, 1));
    }
    Tensor::cat(parts, dim)
}

fn loss_from_outputs(
    y_bnlmhp: Tensor<6>,
    final_state_bhpr: Tensor<4>,
    y_head: Tensor<6>,
    s_head: Tensor<4>,
) -> Tensor<1> {
    let y_head = Tensor::from_inner(y_head);
    let s_head = Tensor::from_inner(s_head);
    (y_bnlmhp * y_head).sum() + (final_state_bhpr * s_head).sum()
}

fn run_path(
    path: Mamba3SsdPath,
    inputs: &Inputs,
    y_head: Tensor<6>,
    s_head: Tensor<4>,
    siso_specialization: bool,
) -> PathRun {
    let (y, state) = inputs.ssd_input(siso_specialization).run(&path);
    let y_inner = y.clone().inner();
    let state_inner = state.clone().inner();
    let loss = loss_from_outputs(y, state, y_head, s_head);
    let grads = loss.backward();
    PathRun {
        y: y_inner,
        state: state_inner,
        d_v: inputs.v.val().grad(&grads).expect("grad v"),
        d_b: inputs.b.val().grad(&grads).expect("grad b"),
        d_c: inputs.c.val().grad(&grads).expect("grad c"),
        d_da: inputs.da.val().grad(&grads).expect("grad da"),
        d_gamma: inputs.gamma.val().grad(&grads).expect("grad gamma"),
        d_scale: inputs.scale.val().grad(&grads).expect("grad scale"),
        d_init_state: inputs
            .initial_state
            .val()
            .grad(&grads)
            .expect("grad initial_state"),
    }
}

fn assert_path_runs_agree(label: &str, a: &PathRun, b: &PathRun, val_tol: f32, grad_tol: f32) {
    use burn_stack::utils::test_helpers::max_abs_diff;
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check_inner {
        ($field:ident, $name:expr, $tol:expr) => {{
            let d = max_abs_diff(a.$field.clone(), b.$field.clone());
            eprintln!("{:>22} {:>14} | max abs diff = {:>10.6}", label, $name, d);
            if d >= $tol {
                failures.push(format!(
                    "{}: {} max abs diff = {:.6} (tol {})",
                    label, $name, d, $tol
                ));
            }
        }};
    }
    check_inner!(y, "y", val_tol);
    check_inner!(state, "final_state", val_tol);
    check_inner!(d_v, "grad v", grad_tol);
    check_inner!(d_b, "grad b", grad_tol);
    check_inner!(d_c, "grad c", grad_tol);
    check_inner!(d_da, "grad da", grad_tol);
    check_inner!(d_gamma, "grad gamma", grad_tol);
    check_inner!(d_scale, "grad scale", grad_tol);
    check_inner!(d_init_state, "grad init_state", grad_tol);
    assert!(
        failures.is_empty(),
        "single-ssd path mismatches:\n  {}",
        failures.join("\n  ")
    );
}

#[allow(clippy::too_many_arguments)]
fn run_minimal_matches_serial(
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    read_stride: usize,
    mimo_rank: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
    random_init: bool,
) {
    let device: Device = Default::default();
    let (v, b, c, da, gamma, scale, init) = random_input(
        batch,
        nchunks,
        chunk_len,
        read_stride,
        mimo_rank,
        nheads,
        per_head_dim,
        state_rank,
        random_init,
        &device,
    );

    let y_head = Tensor::<6>::random(
        [
            batch,
            nchunks,
            chunk_len / read_stride,
            mimo_rank,
            nheads,
            per_head_dim,
        ],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let s_head = Tensor::<4>::random(
        [batch, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let inputs_min = Inputs::from_inner(
        v.clone(),
        b.clone(),
        c.clone(),
        da.clone(),
        gamma.clone(),
        scale.clone(),
        init.clone(),
        read_stride,
    );
    let inputs_ser = Inputs::from_inner(
        v.clone(),
        b.clone(),
        c.clone(),
        da.clone(),
        gamma.clone(),
        scale.clone(),
        init.clone(),
        read_stride,
    );
    let inputs_rec = Inputs::from_inner(v, b, c, da, gamma, scale, init, read_stride);

    // At `mimo_rank == 1` both γ-correction branches are reachable
    // (`siso_specialization`); above it only the general one is. Running the
    // whole agreement matrix under each also pins SISO ≡ MIMO end-to-end.
    let specializations: &[bool] = if mimo_rank == 1 {
        &[true, false]
    } else {
        &[true]
    };
    for &siso in specializations {
        let r_min = run_path(
            Mamba3SsdPath::Minimal(Some(chunk_len)),
            &inputs_min,
            y_head.clone(),
            s_head.clone(),
            siso,
        );
        let r_ser = run_path(
            Mamba3SsdPath::Serial(Some(chunk_len)),
            &inputs_ser,
            y_head.clone(),
            s_head.clone(),
            siso,
        );
        let r_rec = run_path(
            Mamba3SsdPath::SerialRecalculated(Some(chunk_len)),
            &inputs_rec,
            y_head.clone(),
            s_head.clone(),
            siso,
        );

        // Same algorithm, different schedule / backward: stricter on values
        // (1e-4), moderate on gradients (1e-3) — same tolerances as the
        // original-form SSD-path agreement tests.
        let label = format!("siso_specialization={siso}");
        assert_path_runs_agree(
            &format!("{label}: Minimal vs Serial"),
            &r_min,
            &r_ser,
            1e-4,
            1e-3,
        );
        assert_path_runs_agree(
            &format!("{label}: Minimal vs SerialRecalculated"),
            &r_min,
            &r_rec,
            1e-4,
            1e-3,
        );
    }
}

#[test]
fn single_ssd_paths_agree_siso() {
    run_minimal_matches_serial(2, 3, 4, 1, 1, 2, 8, 8, true);
}

#[test]
fn single_ssd_paths_agree_siso_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 1, 1, 2, 8, 8, false);
}

#[test]
fn single_ssd_paths_agree_mimo() {
    run_minimal_matches_serial(2, 3, 4, 1, 2, 2, 8, 8, true);
}

#[test]
fn single_ssd_paths_agree_mimo_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 1, 2, 2, 8, 8, false);
}

#[test]
fn single_ssd_paths_agree_single_chunk() {
    run_minimal_matches_serial(2, 1, 4, 1, 1, 2, 8, 8, true);
}

#[test]
fn single_ssd_paths_agree_single_chunk_zero_init() {
    run_minimal_matches_serial(2, 1, 4, 1, 1, 2, 8, 8, false);
}

/// The chunk's **read axis**: `micro_steps > 1` narrows `C`/`γ`/`y` to one row
/// per token while the writes stay on the folded axis. All three algorithms —
/// and, for `SerialRecalculated`, its hand-written backward's scatter back onto
/// the folded axis — have to agree there too.
#[test]
fn single_ssd_paths_agree_with_a_read_stride() {
    run_minimal_matches_serial(2, 3, 4, 2, 1, 2, 8, 8, true);
    run_minimal_matches_serial(2, 3, 6, 3, 1, 2, 8, 8, false);
}

#[test]
fn single_ssd_paths_agree_with_a_read_stride_mimo() {
    run_minimal_matches_serial(2, 2, 6, 3, 2, 2, 8, 8, true);
}

/// The read axis against the **undecimated** kernel it replaces.
///
/// `read_stride = 1` is that kernel: `read_rows` and `scatter_read_rows` are the
/// identity there and the rectangular mask reduces to the `triu` the kernel
/// always built, so running one set of inputs both ways compares the new
/// algorithm against the old one rather than against a sibling of itself.
///
/// The claim being pinned is the whole of the change: at the positions the
/// readout happens at, the decimated kernel computes *exactly* what the full one
/// did — output, final state and every input gradient — and at the positions in
/// between (which `u > 1` used to compute and then discard) the full kernel's own
/// gradient is identically zero. That second half is why the deleted work was
/// deletable.
#[test]
fn read_axis_matches_the_undecimated_kernel() {
    use burn_stack::utils::test_helpers::max_abs_diff;
    let device: Device = Default::default();
    let (batch, nchunks, nheads, per_head_dim, state_rank) = (2, 3, 2, 8, 8);

    for (chunk_len, stride, mimo_rank) in [(4, 2, 1), (6, 3, 1), (6, 3, 2), (4, 4, 2)] {
        // One draw: `v`/`b`/`da`/`scale` on the write axis, `c`/`gamma` on the read one.
        let (v, b, c_rows, da, gamma_rows, scale, init) = random_input(
            batch,
            nchunks,
            chunk_len,
            stride,
            mimo_rank,
            nheads,
            per_head_dim,
            state_rank,
            true,
            &device,
        );
        // The same `C`/`γ` the full kernel would have been handed. What sits at
        // the positions it never reads is arbitrary — zero says so loudest.
        let c_full = scatter_rows(c_rows.clone(), 2, stride);
        let gamma_full = scatter_rows(gamma_rows.clone(), 2, stride);

        let y_head_rows = Tensor::<6>::random(
            [
                batch,
                nchunks,
                chunk_len / stride,
                mimo_rank,
                nheads,
                per_head_dim,
            ],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        // Zero between the read rows, so the full run's loss sees exactly the
        // rows the decimated one produces.
        let y_head_full = scatter_rows(y_head_rows.clone(), 2, stride);
        let s_head = Tensor::<4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        for path in [
            Mamba3SsdPath::Minimal(Some(chunk_len)),
            Mamba3SsdPath::Serial(Some(chunk_len)),
            Mamba3SsdPath::SerialRecalculated(Some(chunk_len)),
        ] {
            for &siso in if mimo_rank == 1 {
                &[true, false][..]
            } else {
                &[true][..]
            } {
                let label = format!("{path:?} l={chunk_len} u={stride} m={mimo_rank} siso={siso}");
                let read = run_path(
                    path.clone(),
                    &Inputs::from_inner(
                        v.clone(),
                        b.clone(),
                        c_rows.clone(),
                        da.clone(),
                        gamma_rows.clone(),
                        scale.clone(),
                        init.clone(),
                        stride,
                    ),
                    y_head_rows.clone(),
                    s_head.clone(),
                    siso,
                );
                let full = run_path(
                    path.clone(),
                    &Inputs::from_inner(
                        v.clone(),
                        b.clone(),
                        c_full.clone(),
                        da.clone(),
                        gamma_full.clone(),
                        scale.clone(),
                        init.clone(),
                        1,
                    ),
                    y_head_full.clone(),
                    s_head.clone(),
                    siso,
                );

                // Same values at the read rows …
                let checks: [(&str, f32); 7] = [
                    ("y", max_abs_diff(kept_rows(full.y, 2, stride), read.y)),
                    ("final_state", max_abs_diff(full.state, read.state)),
                    ("grad v", max_abs_diff(full.d_v, read.d_v)),
                    ("grad b", max_abs_diff(full.d_b, read.d_b)),
                    ("grad da", max_abs_diff(full.d_da, read.d_da)),
                    ("grad scale", max_abs_diff(full.d_scale, read.d_scale)),
                    (
                        "grad init_state",
                        max_abs_diff(full.d_init_state, read.d_init_state),
                    ),
                ];
                for (what, d) in checks {
                    assert!(d < 1e-4, "{label}: {what} max abs diff = {d:.3e}");
                }
                let d_c_full = full.d_c;
                let d_gamma_full = full.d_gamma;
                for (what, d) in [
                    (
                        "grad c",
                        max_abs_diff(kept_rows(d_c_full.clone(), 2, stride), read.d_c),
                    ),
                    (
                        "grad gamma",
                        max_abs_diff(kept_rows(d_gamma_full.clone(), 2, stride), read.d_gamma),
                    ),
                ] {
                    assert!(d < 1e-4, "{label}: {what} max abs diff = {d:.3e}");
                }

                // … and, off the read rows, nothing to compute: the full
                // kernel's `C`/`γ` gradients there are identically zero.
                let off_rows = |g: Tensor<6>| {
                    let dropped = scatter_rows(kept_rows(g.clone(), 2, stride), 2, stride);
                    max_abs_diff(g - dropped, Tensor::<6>::zeros([1, 1, 1, 1, 1, 1], &device))
                };
                let off_rows4 = |g: Tensor<4>| {
                    let dropped = scatter_rows(kept_rows(g.clone(), 2, stride), 2, stride);
                    max_abs_diff(g - dropped, Tensor::<4>::zeros([1, 1, 1, 1], &device))
                };
                for (what, d) in [("c", off_rows(d_c_full)), ("gamma", off_rows4(d_gamma_full))] {
                    assert!(
                        d < 1e-6,
                        "{label}: grad {what} is non-zero between the read rows ({d:.3e})",
                    );
                }
            }
        }
    }
}
