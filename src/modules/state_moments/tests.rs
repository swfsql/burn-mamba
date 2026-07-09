use super::*;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// `pr()`'s gradient must equal the gradient of the plain `tr(Σ)²/tr(Σ²)`
/// formula (the true PR gradient). Guards against a normalisation rewrite that
/// silently changes the gradient — e.g. collapsing `tr(Σ̂)²/tr(Σ̂²)` to
/// `1/tr(Σ̂²)` with a *detached* normaliser drops the numerator's (nonzero)
/// gradient, leaving only the radial (magnitude) direction — orthogonal to the
/// real ∇PR, i.e. a penalty that no longer reduces rank.
#[test]
fn pr_gradient_matches_direct_formula() {
    let device: Device = Default::default();
    let (samples, state_rank) = (16, 6);
    let base = Tensor::<2>::random([samples, state_rank], Distribution::Normal(0.0, 1.0), &device);

    // via StateMoments::pr
    let h1 = Param::from_tensor(Tensor::from_inner(base.clone()));
    let v1 = h1.val();
    let m2 = v1.clone().transpose().matmul(v1.clone()).reshape([1, 1, state_rank, state_rank]);
    let m1 = v1.clone().sum_dim(0).reshape([1, 1, state_rank]);
    let moments = StateMoments { m2_bhrr: m2, m1_bhr: m1, count: samples };
    let g1 = h1.val().grad(&moments.pr(false).sum().backward()).unwrap();

    // direct `tr(Σ)² / tr(Σ²)` with `Σ = HᵀH / S`
    let h2 = Param::from_tensor(Tensor::from_inner(base.clone()));
    let v2 = h2.val();
    let sigma = v2.clone().transpose().matmul(v2.clone()) / samples as f32;
    let tr1 = (sigma.clone() * Tensor::<2>::eye(state_rank, &sigma.device())).sum();
    let tr2 = sigma.powf_scalar(2.0).sum();
    let pr_direct = tr1.clone() * tr1 / tr2;
    let g2 = h2.val().grad(&pr_direct.backward()).unwrap();

    let scale = max_abs_diff(g2.clone(), g2.zeros_like()).max(1e-6);
    let d = max_abs_diff(g1, g2);
    assert!(d < 1e-3 * scale, "pr() gradient must match tr1²/tr2 (off by {d}, scale {scale})");
}

/// The PR *gradient* must stay finite as the state magnitude is driven toward
/// zero (what weight decay does to the recurrent state). PR is homogeneous of
/// degree 0, so its gradient grows as 1/‖Σ‖ — but it must remain *finite and
/// representable*, not NaN. Regression guard for the fp underflow that a
/// through-the-trace normaliser produced (`-Σ/tr(Σ)²`, with `tr(Σ)²`
/// underflowing to 0) and that detaching the normaliser removes. The value
/// stays scale-invariant throughout.
#[test]
fn pr_gradient_finite_as_magnitude_shrinks() {
    let device: Device = Default::default();
    let (samples, state_rank) = (16, 8);
    let base = Tensor::<2>::random([samples, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let full = base.clone(); // reference PR at unit magnitude
    let pr_ref = moments_from_samples(full).pr(false).into_data().to_vec::<f32>().unwrap()[0];
    // Down to 1e-16 the second moment h⊗h is still representable in fp32
    // (below ~1e-18 it underflows — a forward floor, not a gradient bug).
    for exp in [-2i32, -4, -6, -8, -10, -12, -14, -16] {
        let scaled = base.clone().mul_scalar(10f32.powi(exp));
        let h = Param::from_tensor(Tensor::from_inner(scaled));
        let hv = h.val();
        let m2 = hv.clone().transpose().matmul(hv.clone()).reshape([1, 1, state_rank, state_rank]);
        let m1 = hv.clone().sum_dim(0).reshape([1, 1, state_rank]);
        let moments = StateMoments { m2_bhrr: m2, m1_bhr: m1, count: samples };
        let pr_val = moments.pr(false).into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            (pr_val - pr_ref).abs() < 1e-2,
            "PR must be scale-invariant at 1e{exp}: {pr_val} vs {pr_ref}"
        );
        let grads = moments.pr(false).sum().backward();
        let g = h.val().grad(&grads).expect("grad exists");
        let gvec = g.into_data().to_vec::<f32>().unwrap();
        assert!(
            gvec.iter().all(|v| v.is_finite()),
            "PR gradient must stay finite at magnitude 1e{exp}"
        );
    }
}

/// Build moments directly from an explicit sample matrix `h_sr` (`[samples,
/// state_rank]`, one `(batch, head)` slice) — the brute-force definition the
/// closed forms must reproduce.
fn moments_from_samples(h_sr: Tensor<2>) -> StateMoments {
    let [samples, state_rank] = h_sr.dims();
    let m2_rr = h_sr.clone().transpose().matmul(h_sr.clone());
    StateMoments {
        m2_bhrr: m2_rr.reshape([1, 1, state_rank, state_rank]),
        m1_bhr: h_sr.sum_dim(0).reshape([1, 1, state_rank]),
        count: samples,
    }
}

/// Isotropic samples: `Σ = I`, so `PR = (tr I)²/tr(I²) = r` exactly.
#[test]
fn pr_of_identity_covariance_is_full_rank() {
    let device: Device = Default::default();
    let (state_rank, samples) = (4, 10);
    let moments = StateMoments {
        m2_bhrr: Tensor::<2>::eye(state_rank, &device).unsqueeze::<4>() * samples as f32,
        m1_bhr: Tensor::zeros([1, 1, state_rank], &device),
        count: samples,
    };
    for center in [false, true] {
        let pr = moments.pr(center);
        let expected = Tensor::<2>::full([1, 1], state_rank as f32, &device);
        let d = max_abs_diff(pr, expected);
        assert!(d < 1e-4, "identity covariance: PR should be {state_rank}, off by {d}");
    }
}

/// `trace()` is the raw uncentered magnitude `⟨‖h‖²⟩ = trace(m2)/count` — the
/// mean squared state magnitude, independent of any eigen-structure.
#[test]
fn trace_is_mean_squared_magnitude() {
    let device: Device = Default::default();
    let (samples, state_rank) = (12, 5);
    let h_sr = Tensor::<2>::random([samples, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let moments = moments_from_samples(h_sr.clone());
    let expected = (h_sr.powf_scalar(2.0).sum() / samples as f32).reshape([1, 1]);
    let d = max_abs_diff(moments.trace(), expected);
    assert!(d < 1e-5, "trace should equal mean squared magnitude, off by {d}");
}

/// PR is scale-invariant: shrinking every state by a large factor (into the
/// magnitude regime that an absolute `tr(Σ²)` floor would have dragged below
/// PR's true lower bound of 1) leaves the participation ratio unchanged.
#[test]
fn pr_is_scale_invariant() {
    let device: Device = Default::default();
    let (samples, state_rank) = (16, 6);
    let h_sr = Tensor::<2>::random([samples, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let big = moments_from_samples(h_sr.clone());
    // ×1e-4 ⇒ tr(Σ²) ~ 1e-16, well past the former 1e-12 clamp.
    let tiny = moments_from_samples(h_sr.mul_scalar(1e-4));
    for center in [false, true] {
        let d = max_abs_diff(big.pr(center), tiny.pr(center));
        assert!(d < 1e-3, "PR must be scale-invariant (center={center}), off by {d}");
    }
}

/// All samples equal to one vector: uncentered `PR = 1` (a single direction).
/// (The *centered* covariance of identical samples is a pure fp cancellation
/// — numerically undefined — so only the uncentered ratio is asserted.)
#[test]
fn pr_of_repeated_sample_is_rank_one() {
    let device: Device = Default::default();
    let (state_rank, samples) = (6, 7);
    let v_1r = Tensor::<2>::random([1, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let h_sr = v_1r.expand([samples, state_rank]);
    let moments = moments_from_samples(h_sr);

    let pr_raw = moments.pr(false);
    let d = max_abs_diff(pr_raw, Tensor::<2>::full([1, 1], 1.0, &device));
    assert!(d < 1e-4, "repeated sample: uncentered PR should be 1, off by {d}");
}

/// `pr(center: true)` from raw moments must equal the *uncentered* PR of the
/// explicitly mean-subtracted samples — the centering algebra
/// (`Σ = M₂/S − μμᵀ`) against its by-hand counterpart.
#[test]
fn centered_pr_matches_explicitly_centered_samples() {
    let device: Device = Default::default();
    let (state_rank, samples) = (5, 16);
    let h_sr = Tensor::<2>::random(
        [samples, state_rank],
        Distribution::Normal(1.5, 1.0), // strong mean so centering matters
        &device,
    );
    let from_raw = moments_from_samples(h_sr.clone()).pr(true);
    let centered_sr = h_sr.clone() - h_sr.mean_dim(0);
    let from_centered = moments_from_samples(centered_sr).pr(false);
    let d = max_abs_diff(from_raw, from_centered);
    assert!(d < 1e-3, "centered PR must match explicitly centered samples: {d}");
}

/// `merge` of two halves equals the moments of the concatenated samples, and
/// their PRs agree (both centered and uncentered).
#[test]
fn merge_equals_pooled_samples() {
    let device: Device = Default::default();
    let (state_rank, samples) = (5, 12);
    let h_sr = Tensor::<2>::random(
        [samples, state_rank],
        Distribution::Normal(0.5, 1.0), // non-zero mean so `center` matters
        &device,
    );
    let half = samples / 2;
    let merged = moments_from_samples(h_sr.clone().narrow(0, 0, half))
        .merge(moments_from_samples(h_sr.clone().narrow(0, half, samples - half)));
    let full = moments_from_samples(h_sr);

    assert_eq!(merged.count, full.count);
    let d2 = max_abs_diff(merged.m2_bhrr.clone(), full.m2_bhrr.clone());
    let d1 = max_abs_diff(merged.m1_bhr.clone(), full.m1_bhr.clone());
    assert!(d2 < 1e-4 && d1 < 1e-4, "merge must equal pooled sums (m2 {d2}, m1 {d1})");
    for center in [false, true] {
        let d = max_abs_diff(merged.pr(center), full.pr(center));
        assert!(d < 1e-4, "merged PR must equal pooled PR (center {center}): {d}");
    }
}

/// `pool_batch` folds the batch axis into the samples: same totals, `batch=1`,
/// count scaled by the folded batch.
#[test]
fn pool_batch_folds_batch_into_samples() {
    let device: Device = Default::default();
    let (batch, nheads, state_rank, count) = (3, 2, 4, 10);
    let moments = StateMoments {
        m2_bhrr: Tensor::<4>::random(
            [batch, nheads, state_rank, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        ),
        m1_bhr: Tensor::<3>::random(
            [batch, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        ),
        count,
    };
    let expected_m2 = moments.m2_bhrr.clone().sum_dim(0);
    let pooled = moments.pool_batch();
    assert_eq!(pooled.count, count * batch);
    assert_eq!(pooled.m2_bhrr.dims(), [1, nheads, state_rank, state_rank]);
    assert_eq!(pooled.m1_bhr.dims(), [1, nheads, state_rank]);
    let d = max_abs_diff(pooled.m2_bhrr, expected_m2);
    assert!(d < 1e-4, "pool_batch must sum over the batch axis: {d}");
}
