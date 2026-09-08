use super::*;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// Build a randomised set of tensors on the inner backend.  `Param`s
/// wrapping these are built per-path so each path gets a fresh autodiff
/// graph.
///
/// `da` is drawn from a negative-mean distribution so that the implied
/// per-token decay `exp(da)` stays in `(0, 1]`, matching how the upstream
/// block produces `Δ · A` with `A < 0`.
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
) -> (Tensor<6>, Tensor<4>, Tensor<6>, Tensor<6>, Tensor<4>) {
    let v = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let da = Tensor::<4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Normal(-0.5, 0.1),
        device,
    );
    let b = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    // `C` lives on the chunk's read axis (one row per token).
    let c = Tensor::<6>::random(
        [
            batch,
            nchunks,
            chunk_len / read_stride,
            mimo_rank,
            nheads,
            state_rank,
        ],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    // Random (general case) or zero (fresh-start) initial SSM state per
    // `random_init`, so the path-agreement check spans the whole
    // {zero, random} initial-state dimension.
    let initial_state = if random_init {
        Tensor::<4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        )
    } else {
        Tensor::<4>::zeros([batch, nheads, per_head_dim, state_rank], device)
    };
    (v, da, b, c, initial_state)
}

/// Inputs wrapped as `Param`s so each tensor becomes an autodiff leaf
/// with `require_grad`.  A fresh `Inputs` is built per path so each path
/// runs with its own independent autodiff graph.
struct Inputs {
    v: Param<Tensor<6>>,
    da: Param<Tensor<4>>,
    b: Param<Tensor<6>>,
    c: Param<Tensor<6>>,
    initial_state: Param<Tensor<4>>,
    read_stride: usize,
}

impl Inputs {
    fn from_inner(
        v: Tensor<6>,
        da: Tensor<4>,
        b: Tensor<6>,
        c: Tensor<6>,
        initial_state: Tensor<4>,
        read_stride: usize,
    ) -> Self {
        Self {
            v: Param::from_tensor(Tensor::from_inner(v)),
            da: Param::from_tensor(Tensor::from_inner(da)),
            b: Param::from_tensor(Tensor::from_inner(b)),
            c: Param::from_tensor(Tensor::from_inner(c)),
            initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            read_stride,
        }
    }

    fn ssd_input(&self) -> Mamba3DoubleSsdInput {
        Mamba3DoubleSsdInput {
            v_bnlmhp: self.v.val(),
            da_bnlh: self.da.val(),
            b_bnlmhr: self.b.val(),
            c_bntmhr: self.c.val(),
            initial_state_bhpr: self.initial_state.val(),
            // Serial paths assert this is None — see ssd_serial / ssd_serial_recalculated.
            init_state_hpr: None,
            read_stride: self.read_stride,
        }
    }
}

/// Collected forward outputs and input gradients for a single SSD path run.
struct PathRun {
    y: Tensor<6>,
    state: Tensor<4>,
    d_v: Tensor<6>,
    d_da: Tensor<4>,
    d_b: Tensor<6>,
    d_c: Tensor<6>,
    d_init_state: Tensor<4>,
}

// ---------------------------------------------------------------------------
// The read axis, spelled out row by row
//
// Re-derived here one `narrow` at a time rather than reusing
// `helpers::read_rows` / `scatter_read_rows`: a test that calls the
// implementation it is verifying proves nothing about it.
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

/// Combine `y` and `final_state` into a single deterministic scalar loss
/// using fixed (non-tracked) random "head" tensors. Two distinct heads so
/// that gradients for the y-branch and the state-branch are independent.
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

/// Run a single SSD path and extract the gradients of all 5 inputs.
fn run_path(path: Mamba3SsdPath, inputs: &Inputs, y_head: Tensor<6>, s_head: Tensor<4>) -> PathRun {
    let (y, state) = inputs.ssd_input().run(&path);
    let y_inner = y.clone().inner();
    let state_inner = state.clone().inner();

    let loss = loss_from_outputs(y, state, y_head, s_head);
    let grads = loss.backward();

    PathRun {
        y: y_inner,
        state: state_inner,
        d_v: inputs.v.val().grad(&grads).expect("grad v"),
        d_da: inputs.da.val().grad(&grads).expect("grad da"),
        d_b: inputs.b.val().grad(&grads).expect("grad b"),
        d_c: inputs.c.val().grad(&grads).expect("grad c"),
        d_init_state: inputs
            .initial_state
            .val()
            .grad(&grads)
            .expect("grad initial_state"),
    }
}

/// Run the same input through `Minimal`, `Serial`, and `SerialRecalculated`
/// and assert that all three agree on:
///   1. the forward outputs (`y`, `final_state`)
///   2. the gradients of every input through a fixed scalar loss.
///
/// All three are chunkwise reformulations of the same MIMO-first SSD, so
/// both the values and their gradients must agree up to floating-point
/// noise.
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
    let (v, da, b, c, init) = random_input(
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

    // Fixed (non-tracked) "downstream heads" for the loss.
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

    // Each path gets its own fresh autodiff graph (Param leaves).
    let mk = |v: Tensor<6>, da: Tensor<4>, b: Tensor<6>, c: Tensor<6>, init: Tensor<4>| {
        Inputs::from_inner(v, da, b, c, init, read_stride)
    };
    let inputs_min = mk(v.clone(), da.clone(), b.clone(), c.clone(), init.clone());
    let inputs_ser = mk(v.clone(), da.clone(), b.clone(), c.clone(), init.clone());
    let inputs_rec = mk(v, da, b, c, init);

    let r_min = run_path(
        Mamba3SsdPath::Minimal(Some(chunk_len)),
        &inputs_min,
        y_head.clone(),
        s_head.clone(),
    );
    let r_ser = run_path(
        Mamba3SsdPath::Serial(Some(chunk_len)),
        &inputs_ser,
        y_head.clone(),
        s_head.clone(),
    );
    let r_rec = run_path(
        Mamba3SsdPath::SerialRecalculated(Some(chunk_len)),
        &inputs_rec,
        y_head,
        s_head,
    );

    // ── Forward agreement ────────────────────────────────────────────
    use burn_stack::utils::test_helpers::max_abs_diff;
    let tol = 1e-4f32;
    let dy_ser = max_abs_diff(r_min.y.clone(), r_ser.y.clone());
    let ds_ser = max_abs_diff(r_min.state.clone(), r_ser.state.clone());
    let dy_rec = max_abs_diff(r_min.y.clone(), r_rec.y.clone());
    let ds_rec = max_abs_diff(r_min.state.clone(), r_rec.state.clone());
    assert!(
        dy_ser < tol,
        "Minimal vs Serial: y max abs diff = {dy_ser:.6} (tol {tol})"
    );
    assert!(
        ds_ser < tol,
        "Minimal vs Serial: final_state max abs diff = {ds_ser:.6} (tol {tol})"
    );
    assert!(
        dy_rec < tol,
        "Minimal vs SerialRecalculated: y max abs diff = {dy_rec:.6} (tol {tol})"
    );
    assert!(
        ds_rec < tol,
        "Minimal vs SerialRecalculated: final_state max abs diff = {ds_rec:.6} (tol {tol})"
    );

    // ── Gradient agreement ───────────────────────────────────────────
    // Looser tolerance: every path computes the same mathematical
    // gradients, but the chunkwise reformulations accumulate sums in
    // different orders, so small drift is expected.
    burn_stack::check_grads_match_two_paths!(
        baseline: r_min,
        alt1: ("Serial", r_ser),
        alt2: ("SerialRecalculated", r_rec),
        tol: 1e-3,
        fields: [
            d_v => "v",
            d_da => "da",
            d_b => "b",
            d_c => "c",
            d_init_state => "initial_state",
        ],
    );
}

#[test]
fn paths_agree_siso() {
    // batch=2, nchunks=3, chunk_len=4, read_stride=1, mimo_rank=1, nheads=2,
    // per_head_dim=8, state_rank=8
    run_minimal_matches_serial(2, 3, 4, 1, 1, 2, 8, 8, true);
}

#[test]
fn paths_agree_siso_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 1, 1, 2, 8, 8, false);
}

#[test]
fn paths_agree_mimo() {
    // mimo_rank=2 exercises the fused-L (= chunk_len · R) reshape shared by all three paths.
    run_minimal_matches_serial(2, 3, 4, 1, 2, 2, 8, 8, true);
}

#[test]
fn paths_agree_mimo_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 1, 2, 2, 8, 8, false);
}

#[test]
fn paths_agree_single_chunk() {
    // nchunks=1 — no inter-chunk scan; checks the intra-chunk + state-passing
    // boundary case where K4 runs a single iteration.
    run_minimal_matches_serial(2, 1, 4, 1, 1, 2, 8, 8, true);
}

#[test]
fn paths_agree_single_chunk_zero_init() {
    run_minimal_matches_serial(2, 1, 4, 1, 1, 2, 8, 8, false);
}

/// The chunk's **read axis**: `micro_steps > 1` narrows `C` and `y` to one row
/// per token while the writes stay on the folded axis. All three algorithms —
/// and, for `SerialRecalculated`, its hand-written backward's scatter back onto
/// the folded axis — have to agree there too.
#[test]
fn paths_agree_with_a_read_stride() {
    run_minimal_matches_serial(2, 3, 4, 2, 1, 2, 8, 8, true);
    run_minimal_matches_serial(2, 3, 6, 3, 1, 2, 8, 8, false);
}

#[test]
fn paths_agree_with_a_read_stride_mimo() {
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
/// `C` gradient is identically zero. That second half is why the deleted work was
/// deletable. The single-SSD twin of this test carries `γ` as well.
#[test]
fn read_axis_matches_the_undecimated_kernel() {
    use burn_stack::utils::test_helpers::max_abs_diff;
    let device: Device = Default::default();
    let (batch, nchunks, nheads, per_head_dim, state_rank) = (2, 3, 2, 8, 8);

    for (chunk_len, stride, mimo_rank) in [(4, 2, 1), (6, 3, 1), (6, 3, 2), (4, 4, 2)] {
        // One draw: `v`/`b`/`da` on the write axis, `c` on the read one.
        let (v, da, b, c_rows, init) = random_input(
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
        // The same `C` the full kernel would have been handed. What sits at the
        // positions it never reads is arbitrary — zero says so loudest.
        let c_full = scatter_rows(c_rows.clone(), 2, stride);

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
            let label = format!("{path:?} l={chunk_len} u={stride} m={mimo_rank}");
            let read = run_path(
                path.clone(),
                &Inputs::from_inner(
                    v.clone(),
                    da.clone(),
                    b.clone(),
                    c_rows.clone(),
                    init.clone(),
                    stride,
                ),
                y_head_rows.clone(),
                s_head.clone(),
            );
            let full = run_path(
                path.clone(),
                &Inputs::from_inner(
                    v.clone(),
                    da.clone(),
                    b.clone(),
                    c_full.clone(),
                    init.clone(),
                    1,
                ),
                y_head_full.clone(),
                s_head.clone(),
            );

            // Same values at the read rows …
            let d_c_full = full.d_c;
            for (what, d) in [
                ("y", max_abs_diff(kept_rows(full.y, 2, stride), read.y)),
                ("final_state", max_abs_diff(full.state, read.state)),
                ("grad v", max_abs_diff(full.d_v, read.d_v)),
                ("grad b", max_abs_diff(full.d_b, read.d_b)),
                ("grad da", max_abs_diff(full.d_da, read.d_da)),
                (
                    "grad init_state",
                    max_abs_diff(full.d_init_state, read.d_init_state),
                ),
                (
                    "grad c",
                    max_abs_diff(kept_rows(d_c_full.clone(), 2, stride), read.d_c),
                ),
            ] {
                assert!(d < 1e-4, "{label}: {what} max abs diff = {d:.3e}");
            }

            // … and, off the read rows, nothing to compute.
            let dropped = scatter_rows(kept_rows(d_c_full.clone(), 2, stride), 2, stride);
            let d = max_abs_diff(
                d_c_full - dropped,
                Tensor::<6>::zeros([1, 1, 1, 1, 1, 1], &device),
            );
            assert!(
                d < 1e-6,
                "{label}: grad c is non-zero between the read rows ({d:.3e})",
            );
        }
    }
}
