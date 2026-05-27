use super::*;
use burn::backend::{Autodiff, Flex};
use burn::module::Param;
use burn::tensor::Distribution;

/// Inner (non-autodiff) backend used for materialising values and
/// extracted gradients.
type InnerB = Flex;
/// Autodiff-wrapped backend used to drive `.backward()`.
type B = Autodiff<InnerB>;

type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

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
    mimo_rank: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
    random_init: bool,
    device: &Device,
) -> (
    Tensor<6>,
    Tensor<4>,
    Tensor<6>,
    Tensor<6>,
    Tensor<4>,
) {
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
    let c = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
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
}

impl Inputs {
    fn from_inner(
        v: Tensor<6>,
        da: Tensor<4>,
        b: Tensor<6>,
        c: Tensor<6>,
        initial_state: Tensor<4>,
    ) -> Self {
        Self {
            v: Param::from_tensor(Tensor::from_inner(v)),
            da: Param::from_tensor(Tensor::from_inner(da)),
            b: Param::from_tensor(Tensor::from_inner(b)),
            c: Param::from_tensor(Tensor::from_inner(c)),
            initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
        }
    }

    fn ssd_input(&self) -> Mamba3DoubleSsdInput {
        Mamba3DoubleSsdInput {
            v_bnlmhp: self.v.val(),
            da_bnlh: self.da.val(),
            b_bnlmhr: self.b.val(),
            c_bnlmhr: self.c.val(),
            initial_state_bhpr: self.initial_state.val(),
            // Serial paths assert this is None — see ssd_serial / ssd_serial_recalculated.
            init_state_hpr: None,
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
fn run_path(
    path: Mamba3SsdPath,
    inputs: &Inputs,
    y_head: Tensor<6>,
    s_head: Tensor<4>,
) -> PathRun {
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
        mimo_rank,
        nheads,
        per_head_dim,
        state_rank,
        random_init,
        &device,
    );

    // Fixed (non-tracked) "downstream heads" for the loss.
    let y_head = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let s_head = Tensor::<4>::random(
        [batch, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Each path gets its own fresh autodiff graph (Param leaves).
    let inputs_min = Inputs::from_inner(v.clone(), da.clone(), b.clone(), c.clone(), init.clone());
    let inputs_ser = Inputs::from_inner(v.clone(), da.clone(), b.clone(), c.clone(), init.clone());
    let inputs_rec = Inputs::from_inner(v, da, b, c, init);

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
    use crate::utils::test_helpers::max_abs_diff;
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
    crate::check_grads_match_two_paths!(
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
    // batch=2, nchunks=3, chunk_len=4, mimo_rank=1, nheads=2, per_head_dim=8, state_rank=8
    run_minimal_matches_serial(2, 3, 4, 1, 2, 8, 8, true);
}

#[test]
fn paths_agree_siso_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 1, 2, 8, 8, false);
}

#[test]
fn paths_agree_mimo() {
    // mimo_rank=2 exercises the fused-L (= chunk_len · R) reshape shared by all three paths.
    run_minimal_matches_serial(2, 3, 4, 2, 2, 8, 8, true);
}

#[test]
fn paths_agree_mimo_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 2, 2, 8, 8, false);
}

#[test]
fn paths_agree_single_chunk() {
    // nchunks=1 — no inter-chunk scan; checks the intra-chunk + state-passing
    // boundary case where K4 runs a single iteration.
    run_minimal_matches_serial(2, 1, 4, 1, 2, 8, 8, true);
}

#[test]
fn paths_agree_single_chunk_zero_init() {
    run_minimal_matches_serial(2, 1, 4, 1, 2, 8, 8, false);
}
