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

/// Weight-PR twin of [`pr_gradient_finite_as_magnitude_shrinks`], probing the
/// grokking example's differentiable weight participation-ratio penalty
/// `pr_tensor(W) = (tr WᵀW)² / tr((WᵀW)²)` (denominator `clamp_min(1e-12)`).
/// That penalty (`--pr-lambda`) is the prime suspect for the combined-run NaN:
/// under `--wd 1.0` a penalised matrix's norm is driven toward zero, and the
/// `tr²/tr(·²)` form is the same fp-underflow class the state PR had. The
/// gradient must stay finite, never NaN. `pr_tensor` lives in the example
/// (unreachable from the lib), so its exact formula is replicated here to run
/// under `cargo test --lib`. Exercised at full rank and — the regime the block
/// weights were actually in before the NaN (`z/x/B/C ≈ 1.0`) — rank 1.
fn example_pr_tensor(w: Tensor<2>) -> Tensor<1> {
    let [rows, cols] = w.dims();
    let g = if rows <= cols {
        w.clone().matmul(w.clone().transpose())
    } else {
        w.clone().transpose().matmul(w.clone())
    };
    let tr = w.powf_scalar(2.0).sum();
    let tr2 = g.powf_scalar(2.0).sum().clamp_min(1e-12);
    tr.powf_scalar(2.0) / tr2
}

fn assert_weight_pr_grad_finite_as_shrinks(base: Tensor<2>) {
    for exp in [-2i32, -4, -6, -8, -10, -12, -14, -16] {
        let scaled = base.clone().mul_scalar(10f32.powi(exp));
        let w = Param::from_tensor(Tensor::from_inner(scaled));
        let grads = example_pr_tensor(w.val()).sum().backward();
        let g = w.val().grad(&grads).expect("grad exists");
        let gvec = g.into_data().to_vec::<f32>().unwrap();
        assert!(
            gvec.iter().all(|v| v.is_finite()),
            "weight-PR gradient must stay finite at magnitude 1e{exp} \
             (first few: {:?})",
            &gvec[..gvec.len().min(4)]
        );
    }
}

#[test]
fn weight_pr_gradient_finite_full_rank() {
    let device: Device = Default::default();
    let base = Tensor::<2>::random([16, 12], Distribution::Normal(0.0, 1.0), &device);
    assert_weight_pr_grad_finite_as_shrinks(base);
}

#[test]
fn weight_pr_gradient_finite_rank_one() {
    let device: Device = Default::default();
    let u = Tensor::<2>::random([16, 1], Distribution::Normal(0.0, 1.0), &device);
    let v = Tensor::<2>::random([1, 12], Distribution::Normal(0.0, 1.0), &device);
    assert_weight_pr_grad_finite_as_shrinks(u.matmul(v));
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

// ===========================================================================
// pr_complex — Hermitian PR over the pairing's complex/quaternionic view
// ===========================================================================

/// Trace of a `[n, n]` matrix.
fn tr_nn(m: Tensor<2>) -> Tensor<1> {
    let [n, _] = m.dims();
    (m * Tensor::<2>::eye(n, &Default::default())).sum()
}

/// Brute-force Hermitian PR straight from the split real samples: `x`/`y` the
/// realified pair components `[samples, num_pairs]`, `u` the un-rotated real
/// coordinates `[samples, tail]` (zero-width not supported — pass `None`).
/// Computed entirely from the complex definition (`M = A + iS`, cross block
/// `Σ c̄ u`, real block `Σ uuᵀ`), independent of the sub-block extraction in
/// [`StateMoments::pr_complex`]. PR is scale-invariant, so the `1/samples`
/// normalisation cancels and is skipped.
fn brute_force_pr_complex(x_sa: Tensor<2>, y_sa: Tensor<2>, u_sk: Option<Tensor<2>>) -> f32 {
    let a = x_sa.clone().transpose().matmul(x_sa.clone())
        + y_sa.clone().transpose().matmul(y_sa.clone());
    let s = x_sa.clone().transpose().matmul(y_sa.clone())
        - y_sa.clone().transpose().matmul(x_sa.clone());
    let mut tr = tr_nn(a.clone());
    let mut tr2 = a.powf_scalar(2.0).sum() + s.powf_scalar(2.0).sum();
    if let Some(u_sk) = u_sk {
        let cross_re = x_sa.transpose().matmul(u_sk.clone());
        let cross_im = y_sa.transpose().matmul(u_sk.clone());
        let ublk = u_sk.clone().transpose().matmul(u_sk);
        tr = tr + tr_nn(ublk.clone());
        tr2 = tr2
            + (cross_re.powf_scalar(2.0).sum() + cross_im.powf_scalar(2.0).sum()) * 2.0
            + ublk.powf_scalar(2.0).sum();
    }
    let tr = tr.into_scalar::<f32>();
    let tr2 = tr2.into_scalar::<f32>();
    tr * tr / tr2
}

/// Realify `(x, y)` pair components into the interleaved (NeoX) layout
/// `[x₀ y₀ x₁ y₁ … | tail]`.
fn realify_interleaved(x_sa: Tensor<2>, y_sa: Tensor<2>, u_sk: Option<Tensor<2>>) -> Tensor<2> {
    let [samples, np] = x_sa.dims();
    let pairs = Tensor::cat(
        vec![x_sa.unsqueeze_dim::<3>(2), y_sa.unsqueeze_dim::<3>(2)],
        2,
    )
    .reshape([samples, 2 * np]);
    match u_sk {
        Some(u) => Tensor::cat(vec![pairs, u], 1),
        None => pairs,
    }
}

fn assert_pr_close(label: &str, got: Tensor<2>, want: f32, tol: f32) {
    let got = got.into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        (got - want).abs() < tol * want.abs().max(1.0),
        "{label}: pr_complex {got:.5} vs brute force {want:.5}"
    );
}

/// `Real` pairing is exactly [`StateMoments::pr`].
#[test]
fn pr_complex_real_pairing_equals_pr() {
    let device: Device = Default::default();
    let h_sr = Tensor::<2>::random([16, 6], Distribution::Normal(0.5, 1.0), &device);
    let m = moments_from_samples(h_sr);
    for center in [false, true] {
        let d = max_abs_diff(m.pr_complex(&StatePairing::Real, center), m.pr(center));
        assert!(d < 1e-6, "Real pairing must delegate to pr() (center {center}): {d}");
    }
}

/// Interleaved complex pairing (with a real tail) vs the brute-force Hermitian
/// PR, uncentered and centered (centered = uncentered of mean-subtracted
/// samples, the same identity `pr` obeys).
#[test]
fn pr_complex_matches_brute_force_interleaved() {
    let device: Device = Default::default();
    let (samples, np, tail) = (24, 3, 2);
    let x = Tensor::<2>::random([samples, np], Distribution::Normal(0.7, 1.0), &device);
    let y = Tensor::<2>::random([samples, np], Distribution::Normal(-0.3, 1.0), &device);
    let u = Tensor::<2>::random([samples, tail], Distribution::Normal(0.2, 1.0), &device);
    let v_sr = realify_interleaved(x.clone(), y.clone(), Some(u.clone()));
    let pairing = StatePairing::ComplexInterleaved { num_pairs: np };

    let m = moments_from_samples(v_sr.clone());
    let brute = brute_force_pr_complex(x.clone(), y.clone(), Some(u.clone()));
    assert_pr_close("interleaved", m.pr_complex(&pairing, false), brute, 1e-3);

    // Centered: subtract each component's mean, recompute both sides.
    let xc = x.clone() - x.clone().mean_dim(0);
    let yc = y.clone() - y.clone().mean_dim(0);
    let uc = u.clone() - u.mean_dim(0);
    let brute_c = brute_force_pr_complex(xc, yc, Some(uc));
    assert_pr_close("interleaved centered", m.pr_complex(&pairing, true), brute_c, 1e-3);

    // Fully-rotated variant (no tail).
    let v_full = realify_interleaved(x.clone(), y.clone(), None);
    let m_full = moments_from_samples(v_full);
    let brute_full = brute_force_pr_complex(x, y, None);
    let pairing_full = StatePairing::ComplexInterleaved { num_pairs: np };
    assert_pr_close("interleaved full", m_full.pr_complex(&pairing_full, false), brute_full, 1e-3);
}

/// Half-and-half complex pairing with un-rotated leftovers in **both** halves
/// (`num_pairs < state_rank/2` — the layout partial-RoPE MIMO produces).
#[test]
fn pr_complex_matches_brute_force_half_half() {
    let device: Device = Default::default();
    let (samples, np, leftover) = (24, 2, 2); // state_rank = 2·(np + leftover) = 8
    let x = Tensor::<2>::random([samples, np], Distribution::Normal(0.4, 1.0), &device);
    let y = Tensor::<2>::random([samples, np], Distribution::Normal(-0.6, 1.0), &device);
    let u1 = Tensor::<2>::random([samples, leftover], Distribution::Normal(0.0, 1.0), &device);
    let u2 = Tensor::<2>::random([samples, leftover], Distribution::Normal(0.3, 1.0), &device);
    // Layout: [ x | u1 | y | u2 ] — pairs at (a, half + a), leftovers real.
    let v_sr = Tensor::cat(vec![x.clone(), u1.clone(), y.clone(), u2.clone()], 1);
    let pairing = StatePairing::ComplexHalfHalf { num_pairs: np };

    let m = moments_from_samples(v_sr);
    let brute = brute_force_pr_complex(x, y, Some(Tensor::cat(vec![u1, u2], 1)));
    assert_pr_close("half-and-half", m.pr_complex(&pairing, false), brute, 1e-3);
}

/// Hamilton product on the trailing `(w, x, y, z)` axis — local test copy so
/// this suite stays independent of the mamba3 rotation module.
fn quat_mul_t<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
    let n = D - 1;
    let comp = |t: &Tensor<D>, i: usize| t.clone().narrow(n, i, 1);
    let (aw, ax, ay, az) = (comp(&a, 0), comp(&a, 1), comp(&a, 2), comp(&a, 3));
    let (bw, bx, by, bz) = (comp(&b, 0), comp(&b, 1), comp(&b, 2), comp(&b, 3));
    let w = aw.clone() * bw.clone()
        - ax.clone() * bx.clone()
        - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x = aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
        - az.clone() * by.clone();
    let y = aw.clone() * by.clone() - ax.clone() * bz.clone()
        + ay.clone() * bw.clone()
        + az.clone() * bx.clone();
    let z = aw * bz + ax * by - ay * bx + az * bw;
    Tensor::cat(vec![w, x, y, z], n)
}

/// Quaternion pairing (with a real tail) vs the brute-force quaternionic
/// Hermitian PR computed from the component matrices (`M_jk = Σ q̄ⱼqₖ`).
#[test]
fn pr_complex_matches_brute_force_quaternion() {
    let device: Device = Default::default();
    let (samples, j, tail) = (24, 2, 3); // state_rank = 4·2 + 3 = 11
    let q_sj4 = Tensor::<3>::random([samples, j, 4], Distribution::Normal(0.2, 1.0), &device);
    let u_sk = Tensor::<2>::random([samples, tail], Distribution::Normal(0.1, 1.0), &device);
    let v_sr = Tensor::cat(vec![q_sj4.clone().reshape([samples, 4 * j]), u_sk.clone()], 1);
    let pairing = StatePairing::QuaternionBlocks { num_blocks: j };

    // Component matrices [samples, j].
    let comp = |i: usize| q_sj4.clone().narrow(2, i, 1).reshape([samples, j]);
    let (w, x, y, z) = (comp(0), comp(1), comp(2), comp(3));
    let g = |a: &Tensor<2>, b: &Tensor<2>| a.clone().transpose().matmul(b.clone());
    // M_jk = Σ q̄ⱼqₖ, components per the Hamilton product with conjugate left.
    let mw = g(&w, &w) + g(&x, &x) + g(&y, &y) + g(&z, &z);
    let mx = g(&w, &x) - g(&x, &w) - g(&y, &z) + g(&z, &y);
    let my = g(&w, &y) + g(&x, &z) - g(&y, &w) - g(&z, &x);
    let mz = g(&w, &z) - g(&x, &y) + g(&y, &x) - g(&z, &w);
    let ublk = u_sk.clone().transpose().matmul(u_sk.clone());
    let cross2 = g(&w, &u_sk).powf_scalar(2.0).sum()
        + g(&x, &u_sk).powf_scalar(2.0).sum()
        + g(&y, &u_sk).powf_scalar(2.0).sum()
        + g(&z, &u_sk).powf_scalar(2.0).sum();
    let tr = (tr_nn(mw.clone()) + tr_nn(ublk.clone())).into_scalar::<f32>();
    let tr2 = (mw.powf_scalar(2.0).sum()
        + mx.powf_scalar(2.0).sum()
        + my.powf_scalar(2.0).sum()
        + mz.powf_scalar(2.0).sum()
        + cross2 * 2.0
        + ublk.powf_scalar(2.0).sum())
    .into_scalar::<f32>();
    let brute = tr * tr / tr2;

    let m = moments_from_samples(v_sr);
    assert_pr_close("quaternion", m.pr_complex(&pairing, false), brute, 1e-3);
}

/// The motivating rank-honesty case: a rotating **single complex direction**
/// (`c_s = e^{iφ_s}·c₀`, the conveyor a within-plane rotation produces) must
/// read `PR_ℂ ≡ 1`, while the realified real PR reads ≈ 2 — the ×2 the
/// Hermitian recombination exists to remove.
#[test]
fn pr_complex_rotating_conveyor_is_rank_one() {
    let device: Device = Default::default();
    let (samples, np) = (32, 3);
    let phi_s1 = Tensor::<2>::random([samples, 1], Distribution::Uniform(-3.0, 3.0), &device);
    let (cos, sin) = (phi_s1.clone().cos(), phi_s1.sin());
    let r0 = Tensor::<2>::random([1, np], Distribution::Normal(0.0, 1.0), &device);
    let i0 = Tensor::<2>::random([1, np], Distribution::Normal(0.0, 1.0), &device);
    // c_s = e^{iφ_s} (r0 + i·i0):
    let x = cos.clone() * r0.clone() - sin.clone() * i0.clone();
    let y = sin * r0 + cos * i0;
    let v_sr = realify_interleaved(x, y, None);
    let m = moments_from_samples(v_sr);

    let pairing = StatePairing::ComplexInterleaved { num_pairs: np };
    let pr_c = m.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!((pr_c - 1.0).abs() < 1e-3, "rotating conveyor: PR_ℂ should be 1, got {pr_c}");
    let pr_r = m.pr(false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(pr_r > 1.5, "realified PR should read the ×2 artifact, got {pr_r}");
}

/// Quaternion twin: samples `q_s ⊗ v₀` (a common **left** unit-quaternion
/// rotation — exactly how `rotate_state_rank_blocks` acts) span one
/// quaternionic direction: `PR_ℍ ≡ 1` while the realified PR reads up to 4.
#[test]
fn pr_complex_quaternion_conveyor_is_rank_one() {
    let device: Device = Default::default();
    let (samples, j) = (32, 2);
    // Random unit quaternions per sample (shared across blocks: a global frame
    // rotation), applied to a fixed block vector v0.
    let g = Tensor::<3>::random([samples, 1, 4], Distribution::Normal(0.0, 1.0), &device);
    let norm = (g.clone() * g.clone()).sum_dim(2).sqrt();
    let q_s14 = g / norm;
    let v0_1j4 = Tensor::<3>::random([1, j, 4], Distribution::Normal(0.0, 1.0), &device);
    let v_sj4 = quat_mul_t(
        q_s14.expand([samples, j, 4]),
        v0_1j4.expand([samples, j, 4]),
    );
    let m = moments_from_samples(v_sj4.reshape([samples, 4 * j]));

    let pairing = StatePairing::QuaternionBlocks { num_blocks: j };
    let pr_q = m.pr_complex(&pairing, false).into_data().to_vec::<f32>().unwrap()[0];
    assert!((pr_q - 1.0).abs() < 1e-3, "quaternion conveyor: PR_ℍ should be 1, got {pr_q}");
    let pr_r = m.pr(false).into_data().to_vec::<f32>().unwrap()[0];
    assert!(pr_r > 2.0, "realified PR should read the ×4-class artifact, got {pr_r}");
}

/// `pr_complex` shares `pr`'s scale invariance and gradient finiteness as the
/// state magnitude is driven toward zero (the weight-decay regime).
#[test]
fn pr_complex_scale_invariant_and_grad_finite() {
    let device: Device = Default::default();
    let (samples, state_rank) = (16, 8);
    let pairing = StatePairing::ComplexInterleaved { num_pairs: 3 }; // tail of 2
    let base = Tensor::<2>::random([samples, state_rank], Distribution::Normal(0.0, 1.0), &device);
    let pr_ref = moments_from_samples(base.clone())
        .pr_complex(&pairing, false)
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    for exp in [-2i32, -6, -10, -14, -16] {
        let scaled = base.clone().mul_scalar(10f32.powi(exp));
        let h = Param::from_tensor(Tensor::from_inner(scaled));
        let hv = h.val();
        let m2 = hv.clone().transpose().matmul(hv.clone()).reshape([1, 1, state_rank, state_rank]);
        let m1 = hv.clone().sum_dim(0).reshape([1, 1, state_rank]);
        let moments = StateMoments { m2_bhrr: m2, m1_bhr: m1, count: samples };
        let pr_val = moments
            .pr_complex(&pairing, false)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            (pr_val - pr_ref).abs() < 1e-2,
            "PR_ℂ must be scale-invariant at 1e{exp}: {pr_val} vs {pr_ref}"
        );
        let grads = moments.pr_complex(&pairing, false).sum().backward();
        let g = h.val().grad(&grads).expect("grad exists");
        assert!(
            g.into_data().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
            "PR_ℂ gradient must stay finite at magnitude 1e{exp}"
        );
    }
}

/// `Mamba3::state_pairing()` mirrors the rotation the block applies: kind,
/// SISO/MIMO layout, and rotated width.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_state_pairing_mirrors_rotation() {
    use crate::mamba3::prelude::{Mamba3Config, RotationKind};
    let device: Device = Default::default();
    let cfg = |mimo: usize, frac: f64, rot: RotationKind| {
        Mamba3Config::new(16)
            .with_state_rank(8)
            .with_per_head_dim(8)
            .with_mimo_rank(mimo)
            .with_rope_fraction(frac)
            .with_rotation(rot)
            .init(&device)
            .state_pairing()
    };
    assert_eq!(
        cfg(1, 1.0, RotationKind::Complex2D),
        StatePairing::ComplexInterleaved { num_pairs: 4 }
    );
    assert_eq!(
        cfg(1, 0.5, RotationKind::Complex2D),
        StatePairing::ComplexInterleaved { num_pairs: 2 }
    );
    assert_eq!(
        cfg(2, 0.5, RotationKind::Complex2D),
        StatePairing::ComplexHalfHalf { num_pairs: 2 }
    );
    assert_eq!(cfg(1, 0.0, RotationKind::Complex2D), StatePairing::Real);
    assert_eq!(
        cfg(1, 1.0, RotationKind::Quaternion4D),
        StatePairing::QuaternionBlocks { num_blocks: 2 }
    );
    assert_eq!(
        cfg(1, 0.5, RotationKind::Quaternion4D),
        StatePairing::QuaternionBlocks { num_blocks: 1 }
    );
}
