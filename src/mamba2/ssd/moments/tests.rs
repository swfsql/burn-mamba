use super::*;
use crate::modules::StateMoments;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// Inputs wrapped as `Param`s (autodiff leaves), mirroring the `ssd_path`
/// tests: one `MomentInputs` per run so the closed-form and brute-force
/// computations each own a fresh graph over the same underlying values.
/// `C`/`D` are not read by the moments, so they stay untracked.
struct MomentInputs {
    x: Param<Tensor<5>>,
    dt: Param<Tensor<4>>,
    a_decay: Param<Tensor<1>>,
    b: Param<Tensor<5>>,
    initial_state: Param<Tensor<4>>,
    c_untracked: Tensor<5>,
    d_untracked: Tensor<1>,
    init_state_hpr: Option<Tensor<3>>,
}

impl MomentInputs {
    #[allow(clippy::too_many_arguments)]
    fn random(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        nheads: usize,
        per_head_dim: usize,
        state_rank: usize,
        random_init: bool,
        learnable_init: bool,
        device: &Device,
    ) -> Self {
        let x = Tensor::<5>::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let dt = Tensor::<4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Uniform(0.05, 0.3),
            device,
        );
        let a_decay = Tensor::<1>::random([nheads], Distribution::Uniform(-1.0, -0.5), device);
        let b = Tensor::<5>::random(
            [batch, nchunks, chunk_len, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let initial_state = if random_init {
            Tensor::<4>::random(
                [batch, nheads, per_head_dim, state_rank],
                Distribution::Normal(0.0, 0.1),
                device,
            )
        } else {
            Tensor::<4>::zeros([batch, nheads, per_head_dim, state_rank], device)
        };
        let init_state_hpr = learnable_init.then(|| {
            Tensor::from_inner(Tensor::<3>::random(
                [nheads, per_head_dim, state_rank],
                Distribution::Normal(0.0, 0.1),
                device,
            ))
        });
        Self {
            x: Param::from_tensor(Tensor::from_inner(x)),
            dt: Param::from_tensor(Tensor::from_inner(dt)),
            a_decay: Param::from_tensor(Tensor::from_inner(a_decay)),
            b: Param::from_tensor(Tensor::from_inner(b)),
            initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            c_untracked: Tensor::from_inner(Tensor::<5>::zeros(
                [batch, nchunks, chunk_len, nheads, state_rank],
                device,
            )),
            d_untracked: Tensor::from_inner(Tensor::<1>::zeros([nheads], device)),
            init_state_hpr,
        }
    }

    fn ssd_input(&self) -> Mamba2SsdInput {
        Mamba2SsdInput {
            x_bnlhp: self.x.val(),
            dt_bnlh: self.dt.val(),
            a_decay_h: self.a_decay.val(),
            b_bnlhr: self.b.val(),
            c_bnlhr: self.c_untracked.clone(),
            d_h: self.d_untracked.clone(),
            initial_state_bhpr: self.initial_state.val(),
            init_state_hpr: self.init_state_hpr.clone(),
        }
    }
}

/// Reference: run the per-token recurrence `hₜ = Āₜhₜ₋₁ + xₜ ⊗ B̄ₜ` for
/// `valid_len` tokens, returning every post-step state (what a `step` loop
/// reading the cache observes).
fn brute_force_states(input: &Mamba2SsdInput, valid_len: usize) -> Vec<Tensor<4>> {
    let [batch, _nchunks, chunk_len, nheads, per_head_dim] = input.x_bnlhp.dims();
    let [.., state_rank] = input.b_bnlhr.dims();

    let mut h_bhpr = input.initial_state_bhpr.clone();
    if let Some(init_hpr) = &input.init_state_hpr {
        h_bhpr = h_bhpr
            + init_hpr.clone().unsqueeze_dim::<4>(0).expand([
                batch,
                nheads,
                per_head_dim,
                state_rank,
            ]);
    }
    let mut states = Vec::with_capacity(valid_len);
    for g in 0..valid_len {
        let (n, j) = (g / chunk_len, g % chunk_len);
        let x_bhp = input
            .x_bnlhp
            .clone()
            .slice(s![.., n..n + 1, j..j + 1, .., ..])
            .reshape([batch, nheads, per_head_dim]);
        let dt_bh = input
            .dt_bnlh
            .clone()
            .slice(s![.., n..n + 1, j..j + 1, ..])
            .reshape([batch, nheads]);
        let b_bhr = input
            .b_bnlhr
            .clone()
            .slice(s![.., n..n + 1, j..j + 1, .., ..])
            .reshape([batch, nheads, state_rank]);
        let abar_bh = (dt_bh.clone() * input.a_decay_h.clone().unsqueeze::<2>()).exp();
        let bbar_bhr = dt_bh.unsqueeze_dim::<3>(2) * b_bhr;
        h_bhpr = abar_bh.unsqueeze_dims::<4>(&[2, 3]) * h_bhpr
            + x_bhp.unsqueeze_dim::<4>(3) * bbar_bhr.unsqueeze_dim::<4>(2);
        states.push(h_bhpr.clone());
    }
    states
}

/// Fold [`brute_force_states`] into the moment sums the closed form must match.
fn brute_force_moments(input: &Mamba2SsdInput, valid_len: usize) -> StateMoments {
    let [batch, .., nheads, per_head_dim] = input.x_bnlhp.dims();
    let [.., state_rank] = input.b_bnlhr.dims();
    let device = input.x_bnlhp.device();

    let mut m2_bhrr = Tensor::zeros([batch, nheads, state_rank, state_rank], &device);
    let mut m1_bhr = Tensor::zeros([batch, nheads, state_rank], &device);
    for h_bhpr in brute_force_states(input, valid_len) {
        m2_bhrr = m2_bhrr + h_bhpr.clone().permute([0, 1, 3, 2]).matmul(h_bhpr.clone());
        m1_bhr = m1_bhr + h_bhpr.clone().sum_dim(2).reshape([batch, nheads, state_rank]);
    }
    StateMoments {
        m2_bhrr,
        m1_bhr,
        count: valid_len * per_head_dim,
    }
}

/// `max |a - b| < tol_rel · max(1, max |b|)` — the moments are sums of many
/// O(1) terms, so the comparison is scaled by the reference magnitude.
fn assert_close<const D: usize>(label: &str, a: Tensor<D>, b: Tensor<D>, tol_rel: f32) {
    let scale = max_abs_diff(b.clone(), b.zeros_like()).max(1.0);
    let d = max_abs_diff(a, b);
    assert!(
        d < tol_rel * scale,
        "{label}: max abs diff = {d:.6} (tol {tol_rel} × scale {scale:.3})"
    );
}

#[allow(clippy::too_many_arguments)]
fn run_values_match(
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
    valid_len: usize,
    random_init: bool,
    learnable_init: bool,
) {
    let device: Device = Default::default();
    let inputs = MomentInputs::random(
        batch,
        nchunks,
        chunk_len,
        nheads,
        per_head_dim,
        state_rank,
        random_init,
        learnable_init,
        &device,
    );
    let closed = inputs.ssd_input().state_moments(valid_len);
    let brute = brute_force_moments(&inputs.ssd_input(), valid_len);
    assert_eq!(closed.count, brute.count);
    assert_close("m2", closed.m2_bhrr, brute.m2_bhrr, 1e-4);
    assert_close("m1", closed.m1_bhr, brute.m1_bhr, 1e-4);
}

#[test]
fn moments_match_brute_force() {
    run_values_match(2, 3, 4, 2, 8, 8, 12, true, false);
}

#[test]
fn moments_match_brute_force_zero_init() {
    run_values_match(2, 3, 4, 2, 8, 8, 12, false, false);
}

/// `valid_len` inside the last chunk: positions past it must not contribute
/// (their content is random here, not the zero-pad the block would feed —
/// stronger than the padded case, the mask must fully exclude them).
#[test]
fn moments_match_brute_force_padded() {
    run_values_match(2, 3, 4, 2, 8, 8, 9, true, false);
}

#[test]
fn moments_match_brute_force_single_chunk() {
    run_values_match(2, 1, 4, 2, 8, 8, 3, true, false);
}

#[test]
fn moments_match_brute_force_learnable_init() {
    run_values_match(2, 3, 4, 2, 8, 8, 12, true, true);
}

/// Brute-force participation ratio straight from a pooled sample matrix
/// `[samples, state_rank]` — covariance and traces taken directly (the form
/// the grokking example's stepwise diagnostic uses), independent of the
/// [`StateMoments::pr`] code path.
fn brute_force_pr(h_sn: Tensor<2>, center: bool) -> f32 {
    let [samples, _n] = h_sn.dims();
    let h_sn = if center {
        h_sn.clone() - h_sn.mean_dim(0)
    } else {
        h_sn
    };
    let tr = h_sn.clone().powf_scalar(2.0).sum().into_scalar::<f32>() / samples as f32;
    let sigma_nn = h_sn.clone().transpose().matmul(h_sn) / samples as f32;
    let tr2 = sigma_nn.powf_scalar(2.0).sum().into_scalar::<f32>();
    (tr * tr) / tr2
}

/// End-to-end PR: per head, `pool_batch().pr(center)` from the closed-form
/// moments must equal the brute-force PR over the explicitly collected
/// per-token states, pooled the same way (`(token, batch, p)` rows in
/// `ℝ^{state_rank}`) — the equivalence the grokking example's
/// `state_pr_forward` ↔ `state_pr` swap rests on. Catches sample-count and
/// batch-pooling mistakes that identical raw sums cannot surface.
#[test]
fn pr_matches_brute_force_states() {
    let device: Device = Default::default();
    let (batch, nchunks, chunk_len, nheads, per_head_dim, state_rank) = (2, 3, 4, 2, 8, 8);
    let valid_len = 9; // pooled count must reflect only the unpadded tokens
    let inputs = MomentInputs::random(
        batch,
        nchunks,
        chunk_len,
        nheads,
        per_head_dim,
        state_rank,
        true,
        false,
        &device,
    );
    let input = inputs.ssd_input();

    let pooled = input.state_moments(valid_len).pool_batch();
    assert_eq!(pooled.count, valid_len * per_head_dim * batch);
    let states = brute_force_states(&input, valid_len);

    for center in [false, true] {
        let closed_h = pooled.pr(center).into_data().to_vec::<f32>().unwrap();
        for head in 0..nheads {
            let samples: Vec<Tensor<2>> = states
                .iter()
                .map(|h_bhpr| {
                    h_bhpr
                        .clone()
                        .narrow(1, head, 1)
                        .reshape([batch * per_head_dim, state_rank])
                })
                .collect();
            let brute = brute_force_pr(Tensor::cat(samples, 0), center);
            let closed = closed_h[head];
            let d = (closed - brute).abs();
            assert!(
                d < 1e-3 * brute.abs().max(1.0),
                "head {head} (center {center}): closed PR {closed:.5} vs brute {brute:.5}"
            );
        }
    }
}

/// The closed form is the future penalty primitive, so its gradients must
/// match the brute-force recurrence too (same scalar loss over both moments).
#[test]
fn moments_grads_match_brute_force() {
    let device: Device = Default::default();
    let (batch, nchunks, chunk_len, nheads, per_head_dim, state_rank) = (2, 3, 4, 2, 8, 8);
    let valid_len = 9; // exercise the validity mask's gradient path too

    let m2_head = Tensor::<4>::random(
        [batch, nheads, state_rank, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let m1_head = Tensor::<3>::random(
        [batch, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let loss_of = |m: &StateMoments| {
        (m.m2_bhrr.clone() * Tensor::from_inner(m2_head.clone())).sum()
            + (m.m1_bhr.clone() * Tensor::from_inner(m1_head.clone())).sum()
    };

    // Two identically-valued input sets, each with its own autodiff graph.
    let seed_inputs = MomentInputs::random(
        batch,
        nchunks,
        chunk_len,
        nheads,
        per_head_dim,
        state_rank,
        true,
        false,
        &device,
    );
    let clone_inputs = || MomentInputs {
        x: Param::from_tensor(Tensor::from_inner(seed_inputs.x.val().inner())),
        dt: Param::from_tensor(Tensor::from_inner(seed_inputs.dt.val().inner())),
        a_decay: Param::from_tensor(Tensor::from_inner(seed_inputs.a_decay.val().inner())),
        b: Param::from_tensor(Tensor::from_inner(seed_inputs.b.val().inner())),
        initial_state: Param::from_tensor(Tensor::from_inner(
            seed_inputs.initial_state.val().inner(),
        )),
        c_untracked: seed_inputs.c_untracked.clone(),
        d_untracked: seed_inputs.d_untracked.clone(),
        init_state_hpr: None,
    };

    let closed_inputs = clone_inputs();
    let closed_grads = loss_of(&closed_inputs.ssd_input().state_moments(valid_len)).backward();
    let brute_inputs = clone_inputs();
    let brute_grads =
        loss_of(&brute_force_moments(&brute_inputs.ssd_input(), valid_len)).backward();

    macro_rules! check {
        ($field:ident) => {
            assert_close(
                concat!("d_", stringify!($field)),
                closed_inputs.$field.val().grad(&closed_grads).expect("grad"),
                brute_inputs.$field.val().grad(&brute_grads).expect("grad"),
                1e-3,
            );
        };
    }
    check!(x);
    check!(dt);
    check!(a_decay);
    check!(b);
    check!(initial_state);
}
