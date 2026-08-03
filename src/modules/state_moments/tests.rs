use super::*;
use crate::utils::test_helpers::max_abs_diff;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

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
