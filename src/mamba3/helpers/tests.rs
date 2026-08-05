//! [`mimo_outer_sum`] is shared by the Mamba-3 decode state update (both β and
//! γ terms) and the single-SSD boundary-β seed, which previously each spelled
//! the same permute+matmul out. These guard the shared form against the
//! einsum it stands for.

use super::*;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

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
    let device = Device::default();
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
        let v_host: Vec<f32> = v.to_data().to_vec().unwrap();
        let k_host: Vec<f32> = k.to_data().to_vec().unwrap();

        let got = mimo_outer_sum(v, k);
        assert_eq!([batch, nheads, per_head_dim, state_rank], got.dims());

        let want = reference(&v_host, &k_host, [batch, mimo_rank, nheads, per_head_dim]);
        let got_host: Vec<f32> = got.to_data().to_vec().unwrap();
        for (i, (g, w)) in got_host.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "mimo_rank={mimo_rank} idx={i}: {g} vs {w}"
            );
        }
    }
}
