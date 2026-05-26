use super::*;
use burn::backend::{Autodiff, Flex};
use burn::module::Param;
use burn::tensor::Distribution;

type InnerB = Flex;
type B = Autodiff<InnerB>;
type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

/// Random inputs for the single-SSD. `da` is drawn from a
/// negative-mean distribution so the implied per-token decay `exp(da)`
/// stays in `(0, 1]`. `gamma` and `scale` are non-negative (matching
/// `Δ·σ(λ)`-style outputs of `helpers::trapezoidal_coefficients`).
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
    Tensor<InnerB, 6>, // v
    Tensor<InnerB, 6>, // b
    Tensor<InnerB, 6>, // c
    Tensor<InnerB, 4>, // da
    Tensor<InnerB, 4>, // gamma
    Tensor<InnerB, 4>, // scale
    Tensor<InnerB, 4>, // initial_state
) {
    let v = Tensor::<InnerB, 6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let b = Tensor::<InnerB, 6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let c = Tensor::<InnerB, 6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let da = Tensor::<InnerB, 4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Normal(-0.5, 0.1),
        device,
    );
    let gamma = Tensor::<InnerB, 4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Uniform(0.05, 0.5),
        device,
    );
    let scale = Tensor::<InnerB, 4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Uniform(0.05, 0.5),
        device,
    );
    // Random (general case) or zero (fresh-start) initial merged-form state
    // per `random_init`, covering the whole {zero, random} dimension.
    let initial_state = if random_init {
        Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        )
    } else {
        Tensor::<InnerB, 4>::zeros([batch, nheads, per_head_dim, state_rank], device)
    };
    (v, b, c, da, gamma, scale, initial_state)
}

struct Inputs {
    v: Param<Tensor<B, 6>>,
    b: Param<Tensor<B, 6>>,
    c: Param<Tensor<B, 6>>,
    da: Param<Tensor<B, 4>>,
    gamma: Param<Tensor<B, 4>>,
    scale: Param<Tensor<B, 4>>,
    initial_state: Param<Tensor<B, 4>>,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    fn from_inner(
        v: Tensor<InnerB, 6>,
        b: Tensor<InnerB, 6>,
        c: Tensor<InnerB, 6>,
        da: Tensor<InnerB, 4>,
        gamma: Tensor<InnerB, 4>,
        scale: Tensor<InnerB, 4>,
        initial_state: Tensor<InnerB, 4>,
    ) -> Self {
        Self {
            v: Param::from_tensor(Tensor::from_inner(v)),
            b: Param::from_tensor(Tensor::from_inner(b)),
            c: Param::from_tensor(Tensor::from_inner(c)),
            da: Param::from_tensor(Tensor::from_inner(da)),
            gamma: Param::from_tensor(Tensor::from_inner(gamma)),
            scale: Param::from_tensor(Tensor::from_inner(scale)),
            initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
        }
    }

    fn ssd_input(&self) -> Mamba3SingleSsdInput<B> {
        Mamba3SingleSsdInput {
            v_bnlmhp: self.v.val(),
            b_bnlmhr: self.b.val(),
            c_bnlmhr: self.c.val(),
            da_bnlh: self.da.val(),
            gamma_bnlh: self.gamma.val(),
            scale_bnlh: self.scale.val(),
            initial_state_bhpr: self.initial_state.val(),
            // Serial asserts this is None — see single_ssd_serial.
            init_state_hpr: None,
        }
    }
}

struct PathRun {
    y: Tensor<InnerB, 6>,
    state: Tensor<InnerB, 4>,
    d_v: Tensor<InnerB, 6>,
    d_b: Tensor<InnerB, 6>,
    d_c: Tensor<InnerB, 6>,
    d_da: Tensor<InnerB, 4>,
    d_gamma: Tensor<InnerB, 4>,
    d_scale: Tensor<InnerB, 4>,
    d_init_state: Tensor<InnerB, 4>,
}

fn loss_from_outputs(
    y_bnlmhp: Tensor<B, 6>,
    final_state_bhpr: Tensor<B, 4>,
    y_head: Tensor<InnerB, 6>,
    s_head: Tensor<InnerB, 4>,
) -> Tensor<B, 1> {
    let y_head = Tensor::from_inner(y_head);
    let s_head = Tensor::from_inner(s_head);
    (y_bnlmhp * y_head).sum() + (final_state_bhpr * s_head).sum()
}

fn run_path(
    path: Mamba3SsdPath,
    inputs: &Inputs,
    y_head: Tensor<InnerB, 6>,
    s_head: Tensor<InnerB, 4>,
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
    use crate::utils::test_helpers::max_abs_diff;
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
        mimo_rank,
        nheads,
        per_head_dim,
        state_rank,
        random_init,
        &device,
    );

    let y_head = Tensor::<InnerB, 6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let s_head = Tensor::<InnerB, 4>::random(
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
    );
    let inputs_ser = Inputs::from_inner(
        v.clone(),
        b.clone(),
        c.clone(),
        da.clone(),
        gamma.clone(),
        scale.clone(),
        init.clone(),
    );
    let inputs_rec = Inputs::from_inner(v, b, c, da, gamma, scale, init);

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

    // Same algorithm, different schedule / backward: stricter on values
    // (1e-4), moderate on gradients (1e-3) — same tolerances as the
    // original-form SSD-path agreement tests.
    assert_path_runs_agree("Minimal vs Serial", &r_min, &r_ser, 1e-4, 1e-3);
    assert_path_runs_agree("Minimal vs SerialRecalculated", &r_min, &r_rec, 1e-4, 1e-3);
}

#[test]
fn single_ssd_paths_agree_siso() {
    run_minimal_matches_serial(2, 3, 4, 1, 2, 8, 8, true);
}

#[test]
fn single_ssd_paths_agree_siso_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 1, 2, 8, 8, false);
}

#[test]
fn single_ssd_paths_agree_mimo() {
    run_minimal_matches_serial(2, 3, 4, 2, 2, 8, 8, true);
}

#[test]
fn single_ssd_paths_agree_mimo_zero_init() {
    run_minimal_matches_serial(2, 3, 4, 2, 2, 8, 8, false);
}

#[test]
fn single_ssd_paths_agree_single_chunk() {
    run_minimal_matches_serial(2, 1, 4, 1, 2, 8, 8, true);
}

#[test]
fn single_ssd_paths_agree_single_chunk_zero_init() {
    run_minimal_matches_serial(2, 1, 4, 1, 2, 8, 8, false);
}
