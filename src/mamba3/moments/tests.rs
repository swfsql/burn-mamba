use super::*;
use crate::mamba3::rotation::quat_normalize;
use crate::modules::StatePairing;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

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

/// The rotation leaf of a [`MomentInputs`] set (a `Param` so gradients w.r.t.
/// the cumulative angles/quaternions are checkable).
enum RotParam {
    Angle {
        cum: Param<Tensor<4>>,
        rope_dim: usize,
        rotate_pairwise: bool,
    },
    Quaternion {
        cum: Param<Tensor<5>>,
    },
}

impl RotParam {
    fn seq(&self) -> RotationSeq {
        match self {
            RotParam::Angle {
                cum,
                rope_dim,
                rotate_pairwise,
            } => RotationSeq::Angle {
                cum_bsha: cum.val(),
                rope_dim: *rope_dim,
                rotate_pairwise: *rotate_pairwise,
            },
            RotParam::Quaternion { cum } => RotationSeq::Quaternion { cum_bshj4: cum.val() },
        }
    }
}

/// Which rotation the input set carries (test axis).
#[derive(Clone, Copy)]
enum RotCase {
    /// Complex, `rope_dim` entries rotated, interleaved or half-and-half.
    Complex { rope_dim: usize, rotate_pairwise: bool },
    /// Quaternion, `blocks·4` entries rotated.
    Quaternion { blocks: usize },
}

/// Inputs wrapped as `Param`s (autodiff leaves), mirroring the Mamba-2 moments
/// tests: one set per run so the chunked and brute-force computations each own
/// a fresh graph over the same underlying values. Every position — including
/// the ones past `valid_len` — is random (stronger than the zero pads the
/// block seam would feed: the validity mask alone must exclude them, for the
/// rotation too).
struct MomentInputs {
    xhat: Param<Tensor<6>>,
    bhat: Param<Tensor<6>>,
    da: Param<Tensor<4>>,
    rot: RotParam,
    initial_state: Param<Tensor<4>>,
    init_state_hpr: Option<Tensor<3>>,
}

#[derive(Clone, Copy)]
struct Dims {
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    chan: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
}

const DIMS: Dims = Dims {
    batch: 2,
    nchunks: 3,
    chunk_len: 4,
    chan: 2,
    nheads: 2,
    per_head_dim: 3,
    state_rank: 8,
};

impl MomentInputs {
    fn random(
        d: Dims,
        rot_case: RotCase,
        random_init: bool,
        learnable_init: bool,
        device: &Device,
    ) -> Self {
        let seq = d.nchunks * d.chunk_len;
        let xhat = Tensor::<6>::random(
            [d.batch, d.nchunks, d.chunk_len, d.chan, d.nheads, d.per_head_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let bhat = Tensor::<6>::random(
            [d.batch, d.nchunks, d.chunk_len, d.chan, d.nheads, d.state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let da = Tensor::<4>::random(
            [d.batch, d.nchunks, d.chunk_len, d.nheads],
            Distribution::Uniform(-0.5, -0.05),
            device,
        );
        let rot = match rot_case {
            RotCase::Complex {
                rope_dim,
                rotate_pairwise,
            } => {
                let num_angles = (rope_dim / 2).max(1);
                let cum = Tensor::<4>::random(
                    [d.batch, seq, d.nheads, num_angles],
                    Distribution::Uniform(-2.5, 2.5),
                    device,
                );
                RotParam::Angle {
                    cum: Param::from_tensor(Tensor::from_inner(cum)),
                    rope_dim,
                    rotate_pairwise,
                }
            }
            RotCase::Quaternion { blocks } => {
                let raw = Tensor::<5>::random(
                    [d.batch, seq, d.nheads, blocks, 4],
                    Distribution::Normal(0.0, 1.0),
                    device,
                );
                RotParam::Quaternion {
                    cum: Param::from_tensor(Tensor::from_inner(quat_normalize(raw))),
                }
            }
        };
        let initial_state = if random_init {
            Tensor::<4>::random(
                [d.batch, d.nheads, d.per_head_dim, d.state_rank],
                Distribution::Normal(0.0, 0.3),
                device,
            )
        } else {
            Tensor::<4>::zeros([d.batch, d.nheads, d.per_head_dim, d.state_rank], device)
        };
        let init_state_hpr = learnable_init.then(|| {
            Tensor::from_inner(Tensor::<3>::random(
                [d.nheads, d.per_head_dim, d.state_rank],
                Distribution::Normal(0.0, 0.3),
                device,
            ))
        });
        Self {
            xhat: Param::from_tensor(Tensor::from_inner(xhat)),
            bhat: Param::from_tensor(Tensor::from_inner(bhat)),
            da: Param::from_tensor(Tensor::from_inner(da)),
            rot,
            initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            init_state_hpr,
        }
    }

    fn input(&self) -> Mamba3MomentsInput {
        Mamba3MomentsInput {
            xhat_bnlMhp: self.xhat.val(),
            bhat_bnlMhr: self.bhat.val(),
            da_bnlh: self.da.val(),
            rotation: self.rot.seq(),
            initial_state_bhpr: self.initial_state.val(),
            init_state_hpr: self.init_state_hpr.clone(),
        }
    }
}

/// Reference: run the combined-injection recurrence
/// `h̃ₜ = exp(daₜ)·h̃ₜ₋₁ + Σₘ x̂ₜ[m] ⊗ b̂ₜ[m]` per token, returning every
/// post-step **cache-frame** state.
fn brute_force_cache_states(input: &Mamba3MomentsInput, valid_len: usize) -> Vec<Tensor<4>> {
    let [batch, _n, chunk_len, chan, nheads, per_head_dim] = input.xhat_bnlMhp.dims();
    let [.., state_rank] = input.bhat_bnlMhr.dims();

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
        let da_bh = input
            .da_bnlh
            .clone()
            .slice(s![.., n..n + 1, j..j + 1, ..])
            .reshape([batch, nheads]);
        let xhat_bmhp = input
            .xhat_bnlMhp
            .clone()
            .slice(s![.., n..n + 1, j..j + 1, .., .., ..])
            .reshape([batch, chan, nheads, per_head_dim]);
        let bhat_bmhr = input
            .bhat_bnlMhr
            .clone()
            .slice(s![.., n..n + 1, j..j + 1, .., .., ..])
            .reshape([batch, chan, nheads, state_rank]);
        // write[b,h,p,r] = Σₘ x̂[b,m,h,p]·b̂[b,m,h,r]
        let write_bhpr = xhat_bmhp
            .permute([0, 2, 3, 1])
            .matmul(bhat_bmhr.permute([0, 2, 1, 3]));
        h_bhpr = da_bh.exp().unsqueeze_dims::<4>(&[2, 3]) * h_bhpr + write_bhpr;
        states.push(h_bhpr.clone());
    }
    states
}

/// Fold per-token states into moment sums (with or without the physical-frame
/// de-rotation).
fn fold_moments(
    input: &Mamba3MomentsInput,
    states: &[Tensor<4>],
    derotate: bool,
) -> StateMoments {
    let [batch, .., nheads, per_head_dim] = input.xhat_bnlMhp.dims();
    let [.., state_rank] = input.bhat_bnlMhr.dims();
    let device = input.xhat_bnlMhp.device();

    let mut m2_bhrr = Tensor::zeros([batch, nheads, state_rank, state_rank], &device);
    let mut m1_bhr = Tensor::zeros([batch, nheads, state_rank], &device);
    for (g, h_bhpr) in states.iter().enumerate() {
        let h_bhpr = if derotate {
            input
                .rotation
                .derotate_states(h_bhpr.clone().unsqueeze_dim::<5>(1), g)
                .squeeze_dim::<4>(1)
        } else {
            h_bhpr.clone()
        };
        m2_bhrr = m2_bhrr + h_bhpr.clone().permute([0, 1, 3, 2]).matmul(h_bhpr.clone());
        m1_bhr = m1_bhr + h_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
    }
    StateMoments {
        m2_bhrr,
        m1_bhr,
        count: states.len() * per_head_dim,
    }
}

fn brute_force_phys_moments(input: &Mamba3MomentsInput, valid_len: usize) -> StateMoments {
    let states = brute_force_cache_states(input, valid_len);
    fold_moments(input, &states, true)
}

/// Cache-frame moments `M̃` (no de-rotation) — the test-gated verifier used by
/// the trace identity and the θ≡0 degeneracy.
fn brute_force_cache_moments(input: &Mamba3MomentsInput, valid_len: usize) -> StateMoments {
    let states = brute_force_cache_states(input, valid_len);
    fold_moments(input, &states, false)
}

fn run_values_match(rot_case: RotCase, valid_len: usize, random_init: bool, learnable_init: bool) {
    let device: Device = Default::default();
    let inputs = MomentInputs::random(DIMS, rot_case, random_init, learnable_init, &device);
    let chunked = inputs.input().state_moments_phys(valid_len);
    let brute = brute_force_phys_moments(&inputs.input(), valid_len);
    assert_eq!(chunked.count, brute.count);
    assert_close("m2", chunked.m2_bhrr, brute.m2_bhrr, 1e-4);
    assert_close("m1", chunked.m1_bhr, brute.m1_bhr, 1e-4);
}

#[test]
fn phys_moments_match_brute_force_interleaved_full() {
    run_values_match(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        12,
        true,
        false,
    );
}

#[test]
fn phys_moments_match_brute_force_interleaved_partial() {
    run_values_match(
        RotCase::Complex { rope_dim: 4, rotate_pairwise: true },
        12,
        true,
        false,
    );
}

#[test]
fn phys_moments_match_brute_force_half_half() {
    run_values_match(
        RotCase::Complex { rope_dim: 4, rotate_pairwise: false },
        12,
        true,
        false,
    );
}

#[test]
fn phys_moments_match_brute_force_quaternion() {
    run_values_match(RotCase::Quaternion { blocks: 1 }, 12, true, false);
}

#[test]
fn phys_moments_match_brute_force_quaternion_full() {
    run_values_match(RotCase::Quaternion { blocks: 2 }, 12, true, false);
}

/// `valid_len` inside the last chunk: positions past it (random content AND
/// random rotation here) must not contribute.
#[test]
fn phys_moments_match_brute_force_padded() {
    run_values_match(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        9,
        true,
        false,
    );
    run_values_match(RotCase::Quaternion { blocks: 2 }, 9, true, false);
}

#[test]
fn phys_moments_match_brute_force_zero_init() {
    run_values_match(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        12,
        false,
        false,
    );
}

#[test]
fn phys_moments_match_brute_force_learnable_init() {
    run_values_match(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        12,
        true,
        true,
    );
}

/// Gradients of the chunked computation must match the brute-force recurrence
/// — including w.r.t. the **cumulative rotation** (the de-rotation is
/// θ-differentiable; this is what lets the penalty shape the rotation).
#[test]
fn phys_moments_grads_match_brute_force() {
    let device: Device = Default::default();
    let d = DIMS;
    let valid_len = 9; // exercise the validity mask's gradient path too
    let rot_case = RotCase::Complex { rope_dim: 8, rotate_pairwise: true };

    let m2_head = Tensor::<4>::random(
        [d.batch, d.nheads, d.state_rank, d.state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let m1_head = Tensor::<3>::random(
        [d.batch, d.nheads, d.state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let loss_of = |m: &StateMoments| {
        (m.m2_bhrr.clone() * Tensor::from_inner(m2_head.clone())).sum()
            + (m.m1_bhr.clone() * Tensor::from_inner(m1_head.clone())).sum()
    };

    // Two identically-valued input sets, each with its own autodiff graph.
    let seed = MomentInputs::random(d, rot_case, true, false, &device);
    let clone_inputs = || MomentInputs {
        xhat: Param::from_tensor(Tensor::from_inner(seed.xhat.val().inner())),
        bhat: Param::from_tensor(Tensor::from_inner(seed.bhat.val().inner())),
        da: Param::from_tensor(Tensor::from_inner(seed.da.val().inner())),
        rot: match &seed.rot {
            RotParam::Angle { cum, rope_dim, rotate_pairwise } => RotParam::Angle {
                cum: Param::from_tensor(Tensor::from_inner(cum.val().inner())),
                rope_dim: *rope_dim,
                rotate_pairwise: *rotate_pairwise,
            },
            RotParam::Quaternion { cum } => RotParam::Quaternion {
                cum: Param::from_tensor(Tensor::from_inner(cum.val().inner())),
            },
        },
        initial_state: Param::from_tensor(Tensor::from_inner(seed.initial_state.val().inner())),
        init_state_hpr: None,
    };

    let chunked_inputs = clone_inputs();
    let chunked_grads = loss_of(&chunked_inputs.input().state_moments_phys(valid_len)).backward();
    let brute_inputs = clone_inputs();
    let brute_grads =
        loss_of(&brute_force_phys_moments(&brute_inputs.input(), valid_len)).backward();

    macro_rules! check {
        ($field:ident) => {
            assert_close(
                concat!("d_", stringify!($field)),
                chunked_inputs.$field.val().grad(&chunked_grads).expect("grad"),
                brute_inputs.$field.val().grad(&brute_grads).expect("grad"),
                1e-3,
            );
        };
    }
    check!(xhat);
    check!(bhat);
    check!(da);
    check!(initial_state);
    let (RotParam::Angle { cum: c1, .. }, RotParam::Angle { cum: c2, .. }) =
        (&chunked_inputs.rot, &brute_inputs.rot)
    else {
        unreachable!()
    };
    assert_close(
        "d_angles",
        c1.val().grad(&chunked_grads).expect("grad"),
        c2.val().grad(&brute_grads).expect("grad"),
        1e-3,
    );
}

/// The de-rotation is orthogonal, so the trace is frame-invariant:
/// `tr M_phys = tr M̃` exactly — the cheap cross-check between the physical
/// and cache frames.
#[test]
fn trace_matches_cache_frame() {
    let device: Device = Default::default();
    for rot_case in [
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        RotCase::Complex { rope_dim: 4, rotate_pairwise: false },
        RotCase::Quaternion { blocks: 2 },
    ] {
        let inputs = MomentInputs::random(DIMS, rot_case, true, false, &device);
        let phys = inputs.input().state_moments_phys(10);
        let cache = brute_force_cache_moments(&inputs.input(), 10);
        assert_close("trace", phys.trace(), cache.trace(), 1e-4);
    }
}

/// `θ ≡ 0` (identity rotation) ⇒ the physical and cache frames coincide:
/// `M_phys ≡ M̃` on both moments — pins the de-rotation as exactly inverse.
#[test]
fn theta_zero_degenerates_to_cache_frame() {
    let device: Device = Default::default();
    let d = DIMS;
    let mut inputs = MomentInputs::random(
        d,
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        true,
        false,
        &device,
    );
    inputs.rot = RotParam::Angle {
        cum: Param::from_tensor(Tensor::from_inner(Tensor::zeros(
            [d.batch, d.nchunks * d.chunk_len, d.nheads, 4],
            &device,
        ))),
        rope_dim: 8,
        rotate_pairwise: true,
    };
    let phys = inputs.input().state_moments_phys(12);
    let cache = brute_force_cache_moments(&inputs.input(), 12);
    assert_close("m2", phys.m2_bhrr, cache.m2_bhrr, 1e-5);
    assert_close("m1", phys.m1_bhr, cache.m1_bhr, 1e-5);
}

/// Sign/convention pin, independent of the shared de-rotation helper: a single
/// write of a **rotated** key `b̂ = R(θ)·b_raw` (rotated exactly as the block
/// rotates B) must appear in the physical frame as the **raw** key — a write
/// is un-rotated in the physical frame at its own write time
/// (`c = R(θ)ᵀ h̃ = R(θ)ᵀ x̂ ⊗ (R(θ)b_raw) = x̂ ⊗ b_raw`).
#[test]
fn write_appears_raw_in_physical_frame() {
    let device: Device = Default::default();
    let (state_rank, num_angles) = (4, 2);
    let b_raw_1r = Tensor::<2>::random([1, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let theta_1a = Tensor::<2>::random([1, num_angles], Distribution::Uniform(-2.5, 2.5), &device);
    // Rotate the key exactly as the block rotates B (interleaved SISO).
    let b_rot_1r = crate::modules::apply_rope::<2>(b_raw_1r.clone(), theta_1a.clone(), true);

    let input = Mamba3MomentsInput {
        xhat_bnlMhp: Tensor::ones([1, 1, 1, 1, 1, 1], &device),
        bhat_bnlMhr: b_rot_1r.reshape([1, 1, 1, 1, 1, state_rank]),
        da_bnlh: Tensor::zeros([1, 1, 1, 1], &device),
        rotation: RotationSeq::Angle {
            cum_bsha: theta_1a.reshape([1, 1, 1, num_angles]),
            rope_dim: state_rank,
            rotate_pairwise: true,
        },
        initial_state_bhpr: Tensor::zeros([1, 1, 1, state_rank], &device),
        init_state_hpr: None,
    };
    let moments = input.state_moments_phys(1);
    let expected_m2 = b_raw_1r
        .clone()
        .transpose()
        .matmul(b_raw_1r.clone())
        .reshape([1, 1, state_rank, state_rank]);
    assert_close("m2", moments.m2_bhrr, expected_m2, 1e-5);
    assert_close("m1", moments.m1_bhr, b_raw_1r.reshape([1, 1, state_rank]), 1e-5);
}

/// Angle ramps `[seq, num_angles]` with one rate per angle channel.
fn angle_ramps(rates: &[f32], seq: usize, device: &Device) -> Tensor<2> {
    let t_s1 = Tensor::<1, Int>::arange(0..seq as i64, device)
        .float()
        .reshape([seq, 1]);
    let rates_1a = Tensor::<1>::from_floats(rates, device).reshape([1, rates.len()]);
    t_s1 * rates_1a
}

/// `R(θ)·c₀` realified interleaved: pair `a` of the output is
/// `(cosθₐ·x₀ₐ − sinθₐ·y₀ₐ, sinθₐ·x₀ₐ + cosθₐ·y₀ₐ)`.
fn rope_of_const(theta_sa: Tensor<2>, x0_1a: Tensor<2>, y0_1a: Tensor<2>) -> Tensor<2> {
    let [seq, np] = theta_sa.dims();
    let (cos, sin) = (theta_sa.clone().cos(), theta_sa.sin());
    let x = cos.clone() * x0_1a.clone() - sin.clone() * y0_1a.clone();
    let y = sin * x0_1a + cos * y0_1a;
    Tensor::cat(vec![x.unsqueeze_dim::<3>(2), y.unsqueeze_dim::<3>(2)], 2)
        .reshape([seq, 2 * np])
}

/// Rank-honesty (a): a rotating physical conveyor — a static cache state under
/// a **common-rate** rotation, i.e. `cₜ = e^{−iθₜ}·c₀` — reads `PR_ℂ ≡ 1`
/// while the realified real PR reads ≈ 2 (within-plane rotation is free).
#[test]
fn rank_honesty_rotating_conveyor() {
    let device: Device = Default::default();
    let (seq, state_rank, np) = (12, 4, 2);
    // Static cache state spanning both planes; equal rotation rate per plane.
    let h0 = Tensor::<4>::random([1, 1, 1, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let input = Mamba3MomentsInput {
        xhat_bnlMhp: Tensor::zeros([1, 1, seq, 1, 1, 1], &device),
        bhat_bnlMhr: Tensor::zeros([1, 1, seq, 1, 1, state_rank], &device),
        da_bnlh: Tensor::zeros([1, 1, seq, 1], &device),
        rotation: RotationSeq::Angle {
            cum_bsha: angle_ramps(&[0.7, 0.7], seq, &device).reshape([1, seq, 1, np]),
            rope_dim: state_rank,
            rotate_pairwise: true,
        },
        initial_state_bhpr: h0,
        init_state_hpr: None,
    };
    let m = input.state_moments_phys(seq).pool_batch();
    let pairing = StatePairing::ComplexInterleaved { num_pairs: np };
    let pr_c = m.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!((pr_c - 1.0).abs() < 1e-3, "conveyor: PR_ℂ(M_phys) should be 1, got {pr_c}");
    let pr_r = m.pr(false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(pr_r > 1.5, "realified PR should read the ×2 artifact, got {pr_r}");
}

/// Rank-honesty (b): a **static physical state** under ongoing multi-rate
/// rotation spins in the cache frame — `M̃` decoheres toward the number of
/// active planes while `M_phys` reads the true rank 1. (This is why the
/// cache-frame moment is only a test verifier, never the shipped metric.)
#[test]
fn rank_honesty_static_physical_state() {
    let device: Device = Default::default();
    let (seq, state_rank, np) = (24, 4, 2);
    let theta_sa = angle_ramps(&[0.3, 1.1], seq, &device);
    // Equal modulus per plane (deterministic): a random c₀ can be badly
    // power-imbalanced across planes, which alone drags any PR toward 1.
    let x0 = Tensor::<1>::from_floats([1.0, 1.0], &device).reshape([1, np]);
    let y0 = Tensor::<1>::from_floats([0.5, -0.5], &device).reshape([1, np]);
    // h̃ₜ = R(θₜ)c₀ (so cₜ = R(θₜ)ᵀh̃ₜ = c₀, static). With da ≡ 0 the writes
    // are the increments b̂ₜ = R(θₜ)c₀ − R(θₜ₋₁)c₀ (b̂₀ = R(θ₀)c₀), x̂ ≡ 1.
    let rot_sr = rope_of_const(theta_sa.clone(), x0, y0);
    let prev_sr = Tensor::cat(
        vec![
            Tensor::zeros([1, state_rank], &device),
            rot_sr.clone().narrow(0, 0, seq - 1),
        ],
        0,
    );
    let writes_sr = rot_sr - prev_sr;
    let input = Mamba3MomentsInput {
        xhat_bnlMhp: Tensor::ones([1, 1, seq, 1, 1, 1], &device),
        bhat_bnlMhr: writes_sr.reshape([1, 1, seq, 1, 1, state_rank]),
        da_bnlh: Tensor::zeros([1, 1, seq, 1], &device),
        rotation: RotationSeq::Angle {
            cum_bsha: theta_sa.reshape([1, seq, 1, np]),
            rope_dim: state_rank,
            rotate_pairwise: true,
        },
        initial_state_bhpr: Tensor::zeros([1, 1, 1, state_rank], &device),
        init_state_hpr: None,
    };
    let pairing = StatePairing::ComplexInterleaved { num_pairs: np };

    let phys = input.state_moments_phys(seq).pool_batch();
    let pr_phys = phys.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        (pr_phys - 1.0).abs() < 1e-3,
        "static physical state: PR_ℂ(M_phys) should be 1, got {pr_phys}"
    );

    let cache = brute_force_cache_moments(&input, seq).pool_batch();
    let pr_cache = cache.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        pr_cache > 1.5,
        "cache-frame M̃ should decohere toward the 2 active planes, got {pr_cache}"
    );
}

/// Rank-honesty (c): a constant **raw** writer spread over 2 planes rotated at
/// different rates genuinely occupies 2 complex dims of retained memory —
/// `PR_ℂ(M_phys) → 2` — while the rotation-stripped content moments `M°`
/// (the same recurrence fed the raw keys) are blind to it (`PR ≈ 1`). Charged
/// rank here is real: no frame can collapse retained writes rotated apart.
#[test]
fn rank_honesty_k_plane_writer() {
    let device: Device = Default::default();
    let (nchunks, chunk_len, state_rank, np) = (2, 12, 4, 2);
    let seq = nchunks * chunk_len;
    let rates = [0.3f32, 1.1];
    let theta_sa = angle_ramps(&rates, seq, &device);
    // With no decay, plane `a`'s retained sum is the geometric series
    // `Σⱼ e^{i rₐ (j−t)}`, whose power scales as `1/|1 − e^{−i rₐ}|²` — so a
    // fair 2-plane writer needs its per-plane amplitude ∝ `|1 − e^{−i rₐ}|`,
    // otherwise the slow plane swamps the PR.
    let amps: Vec<f32> = rates
        .iter()
        .map(|r| ((1.0 - r.cos()).powi(2) + r.sin().powi(2)).sqrt())
        .collect();
    let b0x = Tensor::<1>::from_floats(amps.as_slice(), &device).reshape([1, np]);
    let b0y = Tensor::<2>::zeros([1, np], &device);
    // Rotated keys b̂ₜ = R(θₜ)b₀ (what the block writes); raw keys b₀ constant.
    let bhat_sr = rope_of_const(theta_sa.clone(), b0x.clone(), b0y.clone());
    let braw_sr = rope_of_const(theta_sa.zeros_like(), b0x, b0y);
    let make_input = |b_sr: Tensor<2>| Mamba3MomentsInput {
        xhat_bnlMhp: Tensor::ones([1, nchunks, chunk_len, 1, 1, 1], &device),
        bhat_bnlMhr: b_sr.reshape([1, nchunks, chunk_len, 1, 1, state_rank]),
        da_bnlh: Tensor::zeros([1, nchunks, chunk_len, 1], &device),
        rotation: RotationSeq::Angle {
            cum_bsha: theta_sa.clone().reshape([1, seq, 1, np]),
            rope_dim: state_rank,
            rotate_pairwise: true,
        },
        initial_state_bhpr: Tensor::zeros([1, 1, 1, state_rank], &device),
        init_state_hpr: None,
    };
    let pairing = StatePairing::ComplexInterleaved { num_pairs: np };

    let phys = make_input(bhat_sr).state_moments_phys(seq).pool_batch();
    let pr_phys = phys.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        pr_phys > 1.5,
        "2-plane writer: PR_ℂ(M_phys) should approach 2, got {pr_phys}"
    );

    // M°: the same recurrence over the raw (source-frame) keys.
    let content = brute_force_cache_moments(&make_input(braw_sr), seq).pool_batch();
    let pr_content = content.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        (pr_content - 1.0).abs() < 1e-3,
        "content M° is rotation-blind: PR should be 1, got {pr_content}"
    );
}

// ---------------------------------------------------------------------------
// Custom recompute backward ≡ plain autodiff (values and every input gradient)
// ---------------------------------------------------------------------------

/// Rebuild an identically-valued input set with fresh autodiff leaves.
fn clone_moment_inputs(seed: &MomentInputs, learnable_init: bool) -> MomentInputs {
    MomentInputs {
        xhat: Param::from_tensor(Tensor::from_inner(seed.xhat.val().inner())),
        bhat: Param::from_tensor(Tensor::from_inner(seed.bhat.val().inner())),
        da: Param::from_tensor(Tensor::from_inner(seed.da.val().inner())),
        rot: match &seed.rot {
            RotParam::Angle {
                cum,
                rope_dim,
                rotate_pairwise,
            } => RotParam::Angle {
                cum: Param::from_tensor(Tensor::from_inner(cum.val().inner())),
                rope_dim: *rope_dim,
                rotate_pairwise: *rotate_pairwise,
            },
            RotParam::Quaternion { cum } => RotParam::Quaternion {
                cum: Param::from_tensor(Tensor::from_inner(cum.val().inner())),
            },
        },
        initial_state: Param::from_tensor(Tensor::from_inner(seed.initial_state.val().inner())),
        init_state_hpr: learnable_init
            .then(|| seed.init_state_hpr.clone())
            .flatten(),
    }
}

/// `state_moments_phys_recalculated` must equal `state_moments_phys` on values
/// **and** every input gradient — the two are the same math with different
/// backward memory profiles.
fn run_recalculated_matches_autodiff(rot_case: RotCase, valid_len: usize, learnable_init: bool) {
    let device: Device = Default::default();
    let d = DIMS;
    let seed = MomentInputs::random(d, rot_case, true, learnable_init, &device);

    // ── Values ────────────────────────────────────────────────────────────
    let plain = seed.input().state_moments_phys(valid_len);
    let recal = seed.input().state_moments_phys_recalculated(valid_len);
    assert_eq!(plain.count, recal.count);
    assert_close("m2", recal.m2_bhrr.clone(), plain.m2_bhrr.clone(), 1e-5);
    assert_close("m1", recal.m1_bhr.clone(), plain.m1_bhr.clone(), 1e-5);

    // ── Gradients through a fixed random loss over (m2, m1) ───────────────
    let m2_head = Tensor::<4>::random(
        [d.batch, d.nheads, d.state_rank, d.state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let m1_head = Tensor::<3>::random(
        [d.batch, d.nheads, d.state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let loss_of = |m: &StateMoments| {
        (m.m2_bhrr.clone() * Tensor::from_inner(m2_head.clone())).sum()
            + (m.m1_bhr.clone() * Tensor::from_inner(m1_head.clone())).sum()
    };

    let plain_inputs = clone_moment_inputs(&seed, learnable_init);
    let plain_grads = loss_of(&plain_inputs.input().state_moments_phys(valid_len)).backward();
    let recal_inputs = clone_moment_inputs(&seed, learnable_init);
    let recal_grads =
        loss_of(&recal_inputs.input().state_moments_phys_recalculated(valid_len)).backward();

    macro_rules! check {
        ($field:ident) => {
            assert_close(
                concat!("d_", stringify!($field)),
                recal_inputs.$field.val().grad(&recal_grads).expect("grad"),
                plain_inputs.$field.val().grad(&plain_grads).expect("grad"),
                1e-4,
            );
        };
    }
    check!(xhat);
    check!(bhat);
    check!(da);
    check!(initial_state);
    match (&recal_inputs.rot, &plain_inputs.rot) {
        (RotParam::Angle { cum: r, .. }, RotParam::Angle { cum: p, .. }) => {
            assert_close(
                "d_angles",
                r.val().grad(&recal_grads).expect("grad"),
                p.val().grad(&plain_grads).expect("grad"),
                1e-4,
            );
        }
        (RotParam::Quaternion { cum: r }, RotParam::Quaternion { cum: p }) => {
            assert_close(
                "d_quats",
                r.val().grad(&recal_grads).expect("grad"),
                p.val().grad(&plain_grads).expect("grad"),
                1e-4,
            );
        }
        _ => unreachable!(),
    }
}

#[test]
fn recalculated_matches_autodiff_interleaved_full() {
    run_recalculated_matches_autodiff(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        12,
        false,
    );
}

#[test]
fn recalculated_matches_autodiff_interleaved_partial_padded() {
    run_recalculated_matches_autodiff(
        RotCase::Complex { rope_dim: 4, rotate_pairwise: true },
        9,
        false,
    );
}

#[test]
fn recalculated_matches_autodiff_half_half() {
    run_recalculated_matches_autodiff(
        RotCase::Complex { rope_dim: 4, rotate_pairwise: false },
        12,
        false,
    );
}

#[test]
fn recalculated_matches_autodiff_quaternion() {
    run_recalculated_matches_autodiff(RotCase::Quaternion { blocks: 1 }, 12, false);
}

#[test]
fn recalculated_matches_autodiff_quaternion_full_padded() {
    run_recalculated_matches_autodiff(RotCase::Quaternion { blocks: 2 }, 9, false);
}

#[test]
fn recalculated_matches_autodiff_learnable_init() {
    run_recalculated_matches_autodiff(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        12,
        true,
    );
}

/// `valid_len` early in the first chunk: every later chunk is skipped by both
/// the forward `break` and the backward's zero-fill.
#[test]
fn recalculated_matches_autodiff_first_chunk_only() {
    run_recalculated_matches_autodiff(
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        3,
        false,
    );
}

/// Channel-fold bookkeeping: zeroing the β channels equals dropping them
/// (the `λ ≡ 1` degeneracy at the injections level).
#[test]
fn zeroed_channels_are_inert() {
    let device: Device = Default::default();
    let d = DIMS;
    let inputs = MomentInputs::random(
        d,
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        true,
        false,
        &device,
    );
    let full = inputs.input();
    let gamma_only = Mamba3MomentsInput {
        xhat_bnlMhp: full.xhat_bnlMhp.clone().narrow(3, 0, 1),
        bhat_bnlMhr: full.bhat_bnlMhr.clone().narrow(3, 0, 1),
        da_bnlh: full.da_bnlh.clone(),
        rotation: full.rotation.clone(),
        initial_state_bhpr: full.initial_state_bhpr.clone(),
        init_state_hpr: None,
    };
    let zero_l = full.xhat_bnlMhp.clone().narrow(3, 1, 1).zeros_like();
    let zeroed_beta = Mamba3MomentsInput {
        xhat_bnlMhp: Tensor::cat(vec![full.xhat_bnlMhp.clone().narrow(3, 0, 1), zero_l], 3),
        bhat_bnlMhr: full.bhat_bnlMhr.clone(),
        da_bnlh: full.da_bnlh.clone(),
        rotation: full.rotation.clone(),
        initial_state_bhpr: full.initial_state_bhpr.clone(),
        init_state_hpr: None,
    };
    let a = gamma_only.state_moments_phys(12);
    let b = zeroed_beta.state_moments_phys(12);
    assert_close("m2", a.m2_bhrr, b.m2_bhrr, 1e-5);
    assert_close("m1", a.m1_bhr, b.m1_bhr, 1e-5);
}

// ---------------------------------------------------------------------------
// Block level: forward moments ≡ a step() loop reading the physical state
// ---------------------------------------------------------------------------

use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Config, Mamba3SsdPath, RotationKind};

fn small_cfg() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
}

/// The cache's SSM state in the **physical frame** (what the forward moments
/// pool) — the step-side read the parity rests on.
fn physical_state(model: &Mamba3, cache: &Mamba3Cache) -> Tensor<4> {
    let (ssm_bhpr, rotation) = match cache {
        Mamba3Cache::SingleSsd(c) => (c.ssm_bhpr.clone(), &c.rotation),
        Mamba3Cache::DoubleSsd(c) => (c.ssm_bhpr.clone(), &c.rotation),
    };
    rotation.derotate_state(ssm_bhpr, model.rope_dim, model.mimo_rank == 1)
}

/// `forward_with_state_moments` vs a literal `step()` loop accumulating the
/// physical cache state per token, both continued from an identically warmed
/// cache (non-zero SSM state, previous-token k/v entries, and cumulative
/// rotation). `double_cache` selects the SSD pathway.
fn run_block_moments_match_step(label: &str, cfg: Mamba3Config, seq: usize, double_cache: bool) {
    let device: Device = Default::default();
    let model = cfg.init(&device);
    let path = || Mamba3SsdPath::SerialRecalculated(Some(4));
    let batch = 2;
    let normal = Distribution::Normal(0.0, 1.0);
    let warm_bsd = Tensor::<3>::random([batch, 3, cfg.d_model], normal, &device);
    let input_bsd = Tensor::<3>::random([batch, seq, cfg.d_model], normal, &device);

    // Deterministic warm cache (rebuilt per use — caches are not Clone).
    let make_warm = || {
        let (_o, c) = model.forward(warm_bsd.clone(), None, path());
        if double_cache {
            Mamba3Cache::DoubleSsd(c.single_ssd().expect("fresh cache is single-ssd").into())
        } else {
            c
        }
    };

    let (_out, _cache, moments) =
        model.forward_with_state_moments(input_bsd.clone(), Some(make_warm()), path());

    // Reference: step token-by-token, pooling the physical cache state.
    let (nheads, state_rank, per_head_dim) = (cfg.nheads(), cfg.state_rank, cfg.per_head_dim);
    let mut m2 = Tensor::<4>::zeros([batch, nheads, state_rank, state_rank], &device);
    let mut m1 = Tensor::<3>::zeros([batch, nheads, state_rank], &device);
    let mut cache = Some(make_warm());
    for t in 0..seq {
        let x_bd = input_bsd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (_o, c) = model.step(x_bd, cache);
        let phys_bhpr = physical_state(&model, &c);
        m2 = m2 + phys_bhpr.clone().permute([0, 1, 3, 2]).matmul(phys_bhpr.clone());
        m1 = m1 + phys_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
        cache = Some(c);
    }

    assert_eq!(moments.count, seq * per_head_dim, "{label}: count");
    assert_close(&format!("{label}: m2"), moments.m2_bhrr, m2, 1e-4);
    assert_close(&format!("{label}: m1"), moments.m1_bhr, m1, 1e-4);
}

#[test]
fn block_moments_match_step_complex_siso() {
    // seq 7, chunk 4 ⇒ multi-chunk + padding in the last chunk.
    run_block_moments_match_step("complex siso", small_cfg(), 7, false);
}

#[test]
fn block_moments_match_step_complex_double_cache() {
    run_block_moments_match_step("complex double-ssd", small_cfg(), 7, true);
}

#[test]
fn block_moments_match_step_complex_rope_full() {
    run_block_moments_match_step(
        "complex rope=1.0",
        small_cfg().with_rope_fraction(1.0),
        7,
        false,
    );
}

#[test]
fn block_moments_match_step_complex_rope_zero() {
    run_block_moments_match_step(
        "complex rope=0.0",
        small_cfg().with_rope_fraction(0.0),
        7,
        false,
    );
}

#[test]
fn block_moments_match_step_complex_mimo() {
    run_block_moments_match_step("complex mimo", small_cfg().with_mimo_rank(4), 7, false);
}

#[test]
fn block_moments_match_step_quat_siso() {
    run_block_moments_match_step(
        "quat siso",
        small_cfg().with_rotation(RotationKind::Quaternion4D),
        7,
        false,
    );
}

#[test]
fn block_moments_match_step_quat_rope_full_double_cache() {
    run_block_moments_match_step(
        "quat rope=1.0 double-ssd",
        small_cfg()
            .with_rotation(RotationKind::Quaternion4D)
            .with_rope_fraction(1.0),
        7,
        true,
    );
}

#[test]
fn block_moments_match_step_quat_mimo() {
    run_block_moments_match_step(
        "quat mimo",
        small_cfg()
            .with_rotation(RotationKind::Quaternion4D)
            .with_mimo_rank(2),
        7,
        false,
    );
}

/// Two streamed `forward_with_state_moments` calls threaded through the cache,
/// their moments `merge`d, must equal one full-sequence call.
#[test]
fn block_moments_streamed_merge_matches_full() {
    let device: Device = Default::default();
    let cfg = small_cfg();
    let model = cfg.init(&device);
    let path = || Mamba3SsdPath::SerialRecalculated(Some(4));
    let (batch, seq, split) = (2, 9, 5);
    let input_bsd = Tensor::<3>::random(
        [batch, seq, cfg.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (_o, _c, full) = model.forward_with_state_moments(input_bsd.clone(), None, path());

    let (_o1, c1, part_a) =
        model.forward_with_state_moments(input_bsd.clone().narrow(1, 0, split), None, path());
    let (_o2, _c2, part_b) = model.forward_with_state_moments(
        input_bsd.narrow(1, split, seq - split),
        Some(c1),
        path(),
    );
    let merged = part_a.merge(part_b);

    assert_eq!(merged.count, full.count);
    assert_close("m2", merged.m2_bhrr, full.m2_bhrr, 1e-4);
    assert_close("m1", merged.m1_bhr, full.m1_bhr, 1e-4);
}

/// Block-level gradients: a moments loss through
/// `forward_with_state_moments_grad` (the custom recompute node behind the
/// single-ssd seam) must match the same loss accumulated through a literal
/// `step()` chain reading the physical state — the rotation-scan gradient
/// through the de-rotation is the genuinely new coverage.
#[test]
fn block_moments_grads_match_step_chain() {
    let device: Device = Default::default();
    let cfg = small_cfg();
    let model = cfg.init(&device.clone().autodiff());
    let (batch, seq) = (2, 6);
    let normal = Distribution::Normal(0.0, 1.0);
    let input_bsd = Tensor::<3>::random([batch, seq, cfg.d_model], normal, &device);
    let (nheads, state_rank) = (cfg.nheads(), cfg.state_rank);
    let m2_head = Tensor::<4>::random([batch, nheads, state_rank, state_rank], normal, &device);
    let m1_head = Tensor::<3>::random([batch, nheads, state_rank], normal, &device);
    let loss_of = |m2: Tensor<4>, m1: Tensor<3>| {
        (m2 * Tensor::from_inner(m2_head.clone())).sum()
            + (m1 * Tensor::from_inner(m1_head.clone())).sum()
    };

    // Forward path (single-ssd seam, custom recompute backward).
    let (_o, _c, moments) = model.forward_with_state_moments_grad(
        Tensor::from_inner(input_bsd.clone()),
        None,
        Mamba3SsdPath::SerialRecalculated(Some(4)),
    );
    let grads_f = loss_of(moments.m2_bhrr, moments.m1_bhr).backward();

    // Step chain (plain autodiff through the recurrence + rotation).
    let ad_device = device.clone().autodiff();
    let mut m2 = Tensor::<4>::zeros([batch, nheads, state_rank, state_rank], &ad_device);
    let mut m1 = Tensor::<3>::zeros([batch, nheads, state_rank], &ad_device);
    let mut cache = None;
    for t in 0..seq {
        let x_bd = Tensor::from_inner(input_bsd.clone().narrow(1, t, 1).squeeze_dim::<2>(1));
        let (_o, c) = model.step(x_bd, cache);
        let phys_bhpr = physical_state(&model, &c);
        m2 = m2 + phys_bhpr.clone().permute([0, 1, 3, 2]).matmul(phys_bhpr.clone());
        m1 = m1 + phys_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
        cache = Some(c);
    }
    let grads_s = loss_of(m2, m1).backward();

    macro_rules! check_param {
        ($param:expr, $name:expr) => {
            assert_close(
                $name,
                $param.val().grad(&grads_f).expect("forward grad"),
                $param.val().grad(&grads_s).expect("step grad"),
                1e-3,
            );
        };
    }
    check_param!(model.in_proj.weight, "d_in_proj.weight");
    check_param!(model.dt_bias_h, "d_dt_bias_h");
    check_param!(model.b_bias_hmr, "d_b_bias_hmr");
}

/// Multi-consumer smoke test: the moments branch is a *second* consumer of the
/// rotation and key tensors (the SSD consumes them too). Gradients through a
/// combined loss must exist and stay finite for both consumers.
#[test]
fn second_consumer_grads_compose() {
    let device: Device = Default::default();
    let d = DIMS;
    let inputs = MomentInputs::random(
        d,
        RotCase::Complex { rope_dim: 8, rotate_pairwise: true },
        true,
        false,
        &device,
    );
    let input = inputs.input();
    // First consumer: an SSD-like use of the same leaves (keys × values read
    // through the rotation tensor).
    let RotationSeq::Angle { cum_bsha, .. } = &input.rotation else {
        unreachable!()
    };
    let first = (input.bhat_bnlMhr.clone().sum() + cum_bsha.clone().sin().sum())
        + input.xhat_bnlMhp.clone().sum();
    // Second consumer: the moments PR.
    let pairing = StatePairing::ComplexInterleaved { num_pairs: 4 };
    let second = input
        .state_moments_phys(12)
        .pool_batch()
        .pr_complex(&pairing, false)
        .sum();
    let grads = (first + second).backward();
    for (name, g) in [
        ("bhat", inputs.bhat.val().grad(&grads)),
        ("xhat", inputs.xhat.val().grad(&grads)),
    ] {
        let g = g.expect("grad exists");
        assert!(
            g.into_data().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
            "{name} gradient must be finite under two consumers"
        );
    }
    let g = inputs.da.val().grad(&grads).expect("da grad exists");
    assert!(
        g.into_data().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
        "da gradient must be finite under two consumers"
    );
    let RotParam::Angle { cum, .. } = &inputs.rot else { unreachable!() };
    let g = cum.val().grad(&grads).expect("angle grad exists");
    assert!(
        g.into_data().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
        "angle gradient must be finite under two consumers"
    );
}
