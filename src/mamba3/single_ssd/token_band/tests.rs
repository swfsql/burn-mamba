//! The correction band against a direct, index-by-index sum of its own
//! definition — no block, no kernel, just the formula in the module header.

use super::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

/// `Σ_{j<u−1} νᵗᵃᵖ·dcy · (C[u−1]·B[j]) · V[j]`, written out one `(token, tap)`
/// pair at a time.
#[allow(non_snake_case)]
fn reference(
    v_bsmhp: Tensor<5>,
    b_bsmhr: Tensor<5>,
    c_bsmhr: Tensor<5>,
    excess_bsh: Tensor<3>,
    da_bsh: Tensor<3>,
    u: usize,
) -> Tensor<5> {
    let [batch, sequence, mimo_rank, nheads, per_head_dim] = v_bsmhp.dims();
    let [.., state_rank] = b_bsmhr.dims();
    let tokens = sequence / u;
    let device = v_bsmhp.device();
    let mut per_token = Vec::new();
    for token in 0..tokens {
        let read = token * u + (u - 1);
        let mut acc = Tensor::zeros([batch, mimo_rank, nheads, per_head_dim], &device);
        // C at the read, [batch, mimo_rank, nheads, state_rank].
        let c_bmhr: Tensor<4> = c_bsmhr.clone().narrow(1, read, 1).squeeze_dim(1);
        for j in 0..u - 1 {
            let tap = token * u + j;
            let b_bmhr: Tensor<4> = b_bsmhr.clone().narrow(1, tap, 1).squeeze_dim(1);
            let v_bmhp: Tensor<4> = v_bsmhp.clone().narrow(1, tap, 1).squeeze_dim(1);
            // The decay from just after the tap up to (and including) the read.
            let decay_bh = da_bsh
                .clone()
                .narrow(1, tap + 1, read - tap)
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .exp();
            let weight_bh = excess_bsh.clone().narrow(1, tap, 1).squeeze_dim::<2>(1) * decay_bh;

            // Σ_r C[m_out]·B[m_in]  →  [batch, nheads, m_out, m_in]
            let c_bhmr = c_bmhr.clone().swap_dims(1, 2);
            let b_bhrm = b_bmhr.permute([0, 2, 3, 1]);
            let qk_bhmM = c_bhmr.matmul(b_bhrm);
            let v_bhmp = v_bmhp.swap_dims(1, 2);
            let term_bhmp = qk_bhmM.matmul(v_bhmp);
            let term_bmhp = term_bhmp.swap_dims(1, 2)
                * weight_bh.unsqueeze_dims::<4>(&[1, 3]).expand([
                    batch,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
            acc = acc + term_bmhp;
        }
        per_token.push(acc.unsqueeze_dim::<5>(1));
    }
    let _ = state_rank;
    Tensor::cat(per_token, 1)
}

#[test]
fn matches_the_direct_sum() {
    let device: Device = Default::default();
    let dist = Distribution::Normal(0.0, 1.0);
    let (batch, tokens, nheads, per_head_dim, state_rank) = (2, 3, 2, 4, 6);
    for (u, mimo_rank) in [(2, 1), (3, 1), (3, 2), (4, 3)] {
        let sequence = tokens * u;
        let v = Tensor::<5>::random([batch, sequence, mimo_rank, nheads, per_head_dim], dist, &device);
        let b = Tensor::<5>::random([batch, sequence, mimo_rank, nheads, state_rank], dist, &device);
        let c = Tensor::<5>::random([batch, sequence, mimo_rank, nheads, state_rank], dist, &device);
        let excess = Tensor::<3>::random([batch, sequence, nheads], dist, &device);
        // `da = Δ·A ≤ 0`, so the decay it exponentiates is in (0, 1].
        let da = -Tensor::<3>::random([batch, sequence, nheads], dist, &device).abs();

        let got = token_band_correction(
            v.clone(),
            b.clone(),
            c.clone(),
            excess.clone(),
            da.clone(),
            u,
        )
        .expect("u > 1");
        let want = reference(v, b, c, excess, da, u);
        assert_eq!(got.dims(), [batch, tokens, mimo_rank, nheads, per_head_dim]);
        let d = max_abs_diff(got, want);
        assert!(d < 1e-4, "u={u} m={mimo_rank}: {d:.3e}");
    }
}

/// At `u = 1` the band *is* the diagonal, which the kernel already owns.
#[test]
fn no_band_at_one_micro_step() {
    let device: Device = Default::default();
    let dist = Distribution::Normal(0.0, 1.0);
    let (batch, sequence, nheads, per_head_dim, state_rank) = (1, 4, 2, 4, 6);
    assert!(
        token_band_correction(
            Tensor::<5>::random([batch, sequence, 1, nheads, per_head_dim], dist, &device),
            Tensor::<5>::random([batch, sequence, 1, nheads, state_rank], dist, &device),
            Tensor::<5>::random([batch, sequence, 1, nheads, state_rank], dist, &device),
            Tensor::<3>::zeros([batch, sequence, nheads], &device),
            Tensor::<3>::zeros([batch, sequence, nheads], &device),
            1,
        )
        .is_none()
    );
}
