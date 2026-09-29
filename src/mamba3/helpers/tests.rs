//! The Mamba-3 decode state update (both β and γ terms) and the single-SSD
//! boundary-β seed share [`mimo_outer_sum`]. These tests check the shared form
//! against the einsum that it stands for.
//!
//! [`prefix_sum`] is the blocked scan that the cumulative rotation angle uses
//! instead of `Tensor::cumsum`. Its tests compare it against the
//! **definition**, computed on the host, so they stay valid if `cumsum`
//! changes. One cross-check against `cumsum` exists, and only that test
//! depends on `cumsum`.

use super::*;
use burn::module::Param;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::{dtype_tol, test_device};

/// `Σₘ v[m] ⊗ k[m]` computed elementwise from host data — the definition the
/// tensor form has to reproduce.
fn reference(v: &[f32], k: &[f32], dims: [usize; 4]) -> Vec<f32> {
    let [batch, mimo_rank, nheads, per_head_dim] = dims;
    let state_rank = k.len() / (batch * mimo_rank * nheads);
    let mut out = vec![0.0f32; batch * nheads * per_head_dim * state_rank];
    for b in 0..batch {
        for h in 0..nheads {
            for p in 0..per_head_dim {
                for r in 0..state_rank {
                    let mut acc = 0.0;
                    for m in 0..mimo_rank {
                        let vi = ((b * mimo_rank + m) * nheads + h) * per_head_dim + p;
                        let ki = ((b * mimo_rank + m) * nheads + h) * state_rank + r;
                        acc += v[vi] * k[ki];
                    }
                    out[((b * nheads + h) * per_head_dim + p) * state_rank + r] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn mimo_outer_sum_matches_einsum() {
    let device = test_device();
    let (batch, nheads, per_head_dim, state_rank) = (2, 3, 4, 5);

    // `1` is the SISO shape the block actually runs at; `3` exercises the sum.
    for mimo_rank in [1, 3] {
        let v = Tensor::<4>::random(
            [batch, mimo_rank, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [batch, mimo_rank, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v_host: Vec<f32> = v.to_data().try_into_vec_as().unwrap();
        let k_host: Vec<f32> = k.to_data().try_into_vec_as().unwrap();

        let want = reference(&v_host, &k_host, [batch, mimo_rank, nheads, per_head_dim]);

        // Both branches must reproduce the einsum: the broadcast SISO form is
        // only reachable at `mimo_rank == 1`, the matmul form at any rank.
        for siso in [true, false] {
            let got = mimo_outer_sum(v.clone(), k.clone(), siso);
            assert_eq!([batch, nheads, per_head_dim, state_rank], got.dims());

            let got_host: Vec<f32> = got.to_data().try_into_vec_as().unwrap();
            for (i, (g, w)) in got_host.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < dtype_tol(1e-5),
                    "mimo_rank={mimo_rank} siso_specialization={siso} idx={i}: {g} vs {w}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// prefix_sum — the log-depth inclusive scan
// ---------------------------------------------------------------------------

/// `out[i] = Σ_{j ≤ i} data[j]` along `dim`, summed **sequentially** on the
/// host. This is the definition [`prefix_sum`] has to reproduce, and it owes
/// nothing to `Tensor::cumsum` — so it keeps holding if the op `prefix_sum`
/// replaced is ever dropped, deprecated, or changed.
///
/// A row-major buffer collapses to `[outer, len, inner]` around the scanned
/// axis, which is all the indexing needs.
fn reference_prefix_sum(data: &[f32], dims: &[usize], dim: usize) -> Vec<f32> {
    let outer: usize = dims[..dim].iter().product();
    let len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = data.to_vec();
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
            for k in 0..len {
                let idx = (o * len + k) * inner + i;
                acc += data[idx];
                out[idx] = acc;
            }
        }
    }
    out
}

/// How far a prefix of magnitude `expected` may drift between two summation
/// orders.
///
/// It has to **scale with the value**: partial sums of standard normals
/// random-walk to `~√len`, and f32 carries ~1e-7 relative, so any constant bound
/// is a length away from flaking (`len = 2048` reaches ±60, where the observed
/// drift is ~1e-4). A real scan defect is off by whole summands — `O(1)`, orders
/// above either term here — so nothing is given up by scaling.
fn prefix_tol(expected: f32) -> f32 {
    dtype_tol(1e-4) + dtype_tol(1e-5) * expected.abs()
}

/// One `(shape, dim)` case, run both without a carry-in and with one — the
/// carry only shifts every output by a constant, so the same host reference
/// serves for both once it is added back.
fn check_prefix_sum<const D: usize, const DP1: usize>(dims: [usize; D], dim: usize) {
    let device = test_device();
    let t = Tensor::<D>::random(dims, Distribution::Normal(0.0, 1.0), &device);
    let host: Vec<f32> = t.to_data().try_into_vec_as().unwrap();
    let want = reference_prefix_sum(&host, &dims, dim);

    // `init` is `t`'s shape with a single position on the scanned axis.
    let mut init_dims = dims;
    init_dims[dim] = 1;
    let init = Tensor::<D>::random(init_dims, Distribution::Normal(0.0, 1.0), &device);
    let init_host: Vec<f32> = init.to_data().try_into_vec_as().unwrap();
    let inner: usize = dims[dim + 1..].iter().product();

    for carried in [false, true] {
        let got = prefix_sum::<D, DP1>(t.clone(), dim, carried.then(|| init.clone()), None);
        assert_eq!(dims, got.dims(), "prefix_sum must preserve the shape");
        let got: Vec<f32> = got.to_data().try_into_vec_as().unwrap();

        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            // `init` repeats over the scanned axis: drop that axis from `i`.
            let carry = if carried {
                let (outer, within) = (i / (dims[dim] * inner), i % inner);
                init_host[outer * inner + within]
            } else {
                0.0
            };
            let want = w + carry;
            assert!(
                (g - want).abs() < prefix_tol(want),
                "dims={dims:?} dim={dim} carried={carried} idx={i}: {g} vs {want}"
            );
        }
    }
}

/// [`prefix_sum`] **is** the inclusive prefix sum, at every axis position, with
/// and without a carry-in.
///
/// The lengths straddle the two regimes: `≤ 16` takes the direct scan, above it
/// the blocked one, and `17`/`100`/`999` are the cases whose last block is
/// **partial** — the zero padding and the final `narrow` are where an off-by-one
/// would live. `1` is the degenerate no-op; `2048` is the order the folded
/// sequence reaches at `micro_steps = 8`, the length this function exists for.
#[test]
fn prefix_sum_matches_the_definition() {
    for len in [1, 2, 3, 5, 16, 17, 32, 100, 999] {
        // Scanned axis in the middle, as in the cumulative rotation angle.
        check_prefix_sum::<4, 5>([2, len, 3, 5], 1);
        // …first, and last (contiguous).
        check_prefix_sum::<2, 3>([len, 4], 0);
        check_prefix_sum::<3, 4>([3, 2, len], 2);
    }
    check_prefix_sum::<3, 4>([2, 2048, 3], 1);
}

/// Cross-check against the op [`prefix_sum`] replaced.
///
/// The two associate their additions differently, so they agree to rounding,
/// not bit-for-bit. **This is the only test here that depends on `cumsum`; it is
/// the one to delete if that op goes away.**
#[test]
fn prefix_sum_matches_cumsum() {
    let device = test_device();
    for len in [1, 5, 64, 257] {
        let t = Tensor::<4>::random([2, len, 3, 5], Distribution::Normal(0.0, 1.0), &device);
        let want = t.clone().cumsum(1);
        let got = prefix_sum::<4, 5>(t, 1, None, None);
        let scale = want.clone().abs().max().into_scalar::<f32>();
        let d = burn_stack::utils::test_helpers::max_abs_diff(got, want);
        assert!(
            d < prefix_tol(scale),
            "prefix_sum disagrees with cumsum at len={len}: {d:.3e}"
        );
    }
}

/// The backward, against the definition: `∂/∂t[j] Σᵢ w[i]·out[i] = Σ_{i ≥ j} w[i]`.
///
/// The cumulative rotation angle is trained through this scan, so its gradient
/// is as load-bearing as its value — and the blocked form's backward runs
/// through a *different* graph from a single scan's (a reshape, two short
/// scans and a broadcast), not merely a re-association. `len = 100` is a
/// partial last block. The scan on the circle has the same gradient, because
/// its reductions subtract detached multiples of the period.
#[test]
fn prefix_sum_gradient_matches_the_definition() {
    let device = test_device();
    let (batch, len, channels) = (2, 100, 3);

    let raw = Tensor::<3>::random(
        [batch, len, channels],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let w = Tensor::<3>::random(
        [batch, len, channels],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let w_host: Vec<f32> = w.to_data().try_into_vec_as().unwrap();

    for period in [None, Some(std::f32::consts::TAU)] {
        let p = Param::from_tensor(Tensor::from_inner(raw.clone()));
        let loss =
            (prefix_sum::<3, 4>(p.val(), 1, None, period) * Tensor::from_inner(w.clone())).sum();
        let grads = loss.backward();
        let got: Vec<f32> = p
            .val()
            .grad(&grads)
            .expect("grad through prefix_sum")
            .to_data()
            .try_into_vec_as()
            .unwrap();

        // Reverse-inclusive sum of `w` along the scanned axis.
        for b in 0..batch {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for j in (0..len).rev() {
                    let idx = (b * len + j) * channels + c;
                    acc += w_host[idx];
                    assert!(
                        (got[idx] - acc).abs() < prefix_tol(acc),
                        "period={period:?}: grad at (b={b}, j={j}, c={c}): {} vs {acc}",
                        got[idx]
                    );
                }
            }
        }
    }
}

/// `x − k·p` with `k` the nearest integer to `x/p`: the representative in
/// `[−p/2, p/2]`.
fn reduce_f64(x: f64, p: f64) -> f64 {
    x - (x / p).round() * p
}

/// On the circle, [`prefix_sum`] is the prefix sum **modulo the period**, and
/// its rounding does not grow with the length.
///
/// The steps drift (mean 2), so the plain prefix grows linearly and passes
/// `2.6·10⁵` at `len = 131072`, where one f32 step is `0.03`. The reference
/// sums in f64 on the host. The period is the f32 `2π` that the scan
/// subtracts, so the only difference left is rounding. The lengths use 0, 1, 2
/// and 3 levels of the recursive carry.
///
/// The SSD reads the angle **between** two positions: the rotation of the
/// state from one to the other. So the test compares differences over fixed
/// lags. The absolute angle also collects the rounding of every earlier block,
/// but both positions of a difference share that part, and it cancels. At a
/// fixed lag, one tolerance holds at every length: the worst is about `2·10⁻⁵`
/// at lag 1 and `3·10⁻⁴` at lag 1024. The plain scan (`period = None`) fails
/// the same check at `len = 5000` already (`10⁻³` at lag 1), because it rounds
/// each output at the size of the whole prefix.
///
/// Each output is also at most `π` plus one in-block prefix, whatever `len` is.
/// That bound is why the rounding stays small.
#[test]
fn prefix_sum_on_the_circle_matches_the_definition() {
    let device = test_device();
    let tau = std::f32::consts::TAU;
    let (batch, channels) = (2, 3);
    for len in [5, 100, 5000, 131072] {
        let t = Tensor::<3>::random(
            [batch, len, channels],
            Distribution::Normal(2.0, 3.0),
            &device,
        );
        let init = Tensor::<3>::random(
            [batch, 1, channels],
            Distribution::Uniform(-3.1, 3.1),
            &device,
        );
        let t_host: Vec<f32> = t.to_data().try_into_vec_as().unwrap();
        let init_host: Vec<f32> = init.to_data().try_into_vec_as().unwrap();
        let got: Vec<f32> = prefix_sum::<3, 4>(t, 1, Some(init), Some(tau))
            .to_data()
            .try_into_vec_as()
            .unwrap();

        let step_max = t_host.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let bound = std::f32::consts::PI + scan_block(len) as f32 * step_max + 1e-3;
        let mut worst = [0.0f64; 3];
        for b in 0..batch {
            for c in 0..channels {
                let idx = |i: usize| (b * len + i) * channels + c;
                let mut want = vec![0.0f64; len];
                let mut acc = init_host[b * channels + c] as f64;
                for i in 0..len {
                    acc += t_host[idx(i)] as f64;
                    want[i] = acc;
                    assert!(
                        got[idx(i)].abs() <= bound,
                        "len={len}: |out| = {} passes the bound {bound}",
                        got[idx(i)].abs()
                    );
                }
                for (worst, lag) in worst.iter_mut().zip([1, 64, 1024]) {
                    for i in lag..len {
                        let got_d = got[idx(i)] as f64 - got[idx(i - lag)] as f64;
                        let want_d = want[i] - want[i - lag];
                        *worst = worst.max(reduce_f64(got_d - want_d, tau as f64).abs());
                    }
                }
            }
        }
        assert!(
            worst.iter().all(|w| *w < f64::from(dtype_tol(5e-4))),
            "len={len}: the angle over lags 1, 64, 1024 is off by {worst:?}"
        );
    }
}
