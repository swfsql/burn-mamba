//! ## Exact per-token state moments from the chunkwise SSD (no state materialisation)
//!
//! A token-by-token `step` loop exposes every SSM state
//! `hₜ ∈ ℝ^{per_head_dim × state_rank}`; the chunkwise `forward` only
//! materialises states at **chunk boundaries**. This module computes the
//! pooled first/second moments of *all* per-token states — everything a
//! participation-ratio diagnostic (or penalty) needs — **in closed form**,
//! from tensors the SSD decomposition already builds, without ever
//! materialising a `[batch, sequence, nheads, per_head_dim, state_rank]`
//! state tensor.
//!
//! ### Derivation
//!
//! Within chunk `n` (per `(batch, head)`, positions `t ∈ [0, chunk_len)`),
//! unrolling the recurrence `hₜ = Āₜhₜ₋₁ + xₜ ⊗ B̄ₜ` from the state `h₋`
//! entering the chunk gives
//!
//! ```text
//!   hₜ = dₜ·h₋ + sₜ ,   dₜ = exp(a_cum[t]) ,   sₜ = Σ_{j≤t} L[t,j] · xⱼ ⊗ B̄ⱼ
//! ```
//!
//! with `L[t,j] = exp(a_cum[t] − a_cum[j])` — exactly the 1-semiseparable
//! mask of SSD Step 1. Summing `hₜᵀhₜ` over `t` (the `ᵀ` contraction pools
//! `per_head_dim`) splits into three **chunk-level** contractions:
//!
//! ```text
//!   Σₜ hₜᵀhₜ = (Σₜ dₜ²)·h₋ᵀh₋                      (carried-state term)
//!            + h₋ᵀP + Pᵀh₋ ,  P = Xᵀ·diag(w)·B̄ ,  wⱼ = Σₜ dₜ·L[t,j]
//!            + B̄ᵀ·(K ∘ XXᵀ)·B̄ ,                    K = LᵀL
//! ```
//!
//! and the first moment likewise:
//! `Σₜ hₜ = (Σₜ dₜ)·h₋ + Xᵀ·diag(u)·B̄`, `uⱼ = Σₜ L[t,j]`. Everything is a
//! batched GEMM in the same size class as SSD Step 1 (`[chunk_len,
//! chunk_len]` / `[state_rank, state_rank]` intermediates). The boundary
//! states `h₋` are recomputed with SSD Steps 2–3, so this function is
//! **pathway-agnostic** (independent of which SSD variant produced `y`) and
//! fully differentiable through plain autodiff.
//!
//! Zero-pad positions (`Δ=0 ⇒ Ā=1, B̄=0`) carry the state unchanged, so
//! counting them would replicate the final state into the moments; a
//! validity mask excludes them from every `Σₜ`.

use crate::mamba2::prelude::*;
use crate::modules::{StateMoments, sanity as san, segsum};
use burn::prelude::*;

impl Mamba2SsdInput {
    /// A value-identical copy detached from any autodiff graph (used by the
    /// diagnostic wrapper so the moments branch records no backward nodes).
    pub fn detached(&self) -> Self {
        Self {
            x_bnlhp: self.x_bnlhp.clone().detach(),
            dt_bnlh: self.dt_bnlh.clone().detach(),
            a_decay_h: self.a_decay_h.clone().detach(),
            b_bnlhr: self.b_bnlhr.clone().detach(),
            c_bnlhr: self.c_bnlhr.clone().detach(),
            d_h: self.d_h.clone().detach(),
            initial_state_bhpr: self.initial_state_bhpr.clone().detach(),
            init_state_hpr: self.init_state_hpr.clone().map(Tensor::detach),
        }
    }

    /// Exact pooled moments of every per-token SSM state (see the module
    /// header). `valid_len` is the unpadded token count — states at zero-pad
    /// positions are excluded. `C`/`D` are not read.
    ///
    /// Returns raw sums; `count = valid_len · per_head_dim` samples per
    /// `(batch, head)` slice.
    pub fn state_moments(&self, valid_len: usize) -> StateMoments {
        let [batch, nchunks, chunk_len, nheads, per_head_dim] = self.x_bnlhp.dims();
        let [.., state_rank] = self.b_bnlhr.dims();
        let device = &self.x_bnlhp.device();
        assert!(
            (1..=nchunks * chunk_len).contains(&valid_len),
            "valid_len must be within the (padded) sequence"
        );

        // ── Discretised parameters (as in `ssd_minimal`) ──────────────────────
        // B̄ₜ = Δₜ·Bₜ ; log-decay a = Δₜ·A (negative).
        let delta_b_bnlhr = self.dt_bnlh.clone().unsqueeze_dim(4) * self.b_bnlhr.clone();
        let a_bnlh = self.dt_bnlh.clone()
            * self
                .a_decay_h
                .clone()
                .unsqueeze_dims::<4>(&[0, 1, 2])
                .expand([batch, nchunks, chunk_len, nheads]);
        let a_bhnl = a_bnlh.permute([0, 3, 1, 2]);
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);
        san(&a_cumsum_bhnl);

        // ── Boundary states h₋ entering each chunk (SSD Steps 2–3) ────────────
        let state_in_bnhpr = {
            // Step 2: per-chunk end state from a zero start.
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            let decay_state_bhnl = (a_cumsum_last_bhn1.clone() - a_cumsum_bhnl.clone()).exp();
            let decay_state_bnlh1 = decay_state_bhnl.permute([0, 2, 3, 1]).unsqueeze_dim(4);
            let decayed_x_bnhpl = (decay_state_bnlh1 * self.x_bnlhp.clone())
                .permute([0, 1, 3, 4, 2]);
            let state_bnhpr = decayed_x_bnhpl
                .matmul(delta_b_bnlhr.clone().permute([0, 1, 3, 2, 4]));

            // Step 3: inter-chunk state passing (segsum over chunks).
            let initial_state_b1hpr = self.initial_state_bhpr.clone().unsqueeze_dim(1);
            let initial_state_b1hpr = if let Some(init_hpr) = &self.init_state_hpr {
                let init_b1hpr = init_hpr.clone().unsqueeze_dim::<4>(0).expand([
                    batch,
                    1,
                    nheads,
                    per_head_dim,
                    state_rank,
                ]);
                initial_state_b1hpr + init_b1hpr
            } else {
                initial_state_b1hpr
            };
            let state_bNhpr = Tensor::cat(vec![initial_state_b1hpr, state_bnhpr], 1);
            let a_chunk_pad_bhN = Tensor::cat(
                vec![
                    Tensor::<3>::zeros(Shape::new([batch, nheads, 1]), device),
                    a_cumsum_last_bhn1.squeeze_dim::<3>(3),
                ],
                2,
            );
            let decay_chunk_bhNN = segsum::<3, 4>(a_chunk_pad_bhN).exp();
            let flat_state_dim = per_head_dim * state_rank;
            let state_bhNf = state_bNhpr
                .permute([0, 2, 1, 3, 4])
                .reshape([batch, nheads, 1 + nchunks, flat_state_dim]);
            let new_state_bhNf = decay_chunk_bhNN.matmul(state_bhNf);
            let new_state_bhNpr =
                new_state_bhNf.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);
            // Keep the state *entering* each chunk (drop the final state).
            new_state_bhNpr
                .slice(s![.., .., 0..nchunks, .., ..])
                .permute([0, 2, 1, 3, 4]) // state_in_bnhpr
        };
        san(&state_in_bnhpr);

        // ── Validity mask over the `t` axis (zero-pad positions excluded) ─────
        let mask_11nl = Tensor::<1, Int>::arange(0..(nchunks * chunk_len) as i64, device)
            .lower_elem(valid_len as i64)
            .float()
            .reshape([1, 1, nchunks, chunk_len]);

        // dₜ = exp(a_cum[t]) and the Step-1 mask L, both masked over `t`.
        let d_bhnl = a_cumsum_bhnl.exp() * mask_11nl.clone();
        let l_bhnll = segsum::<4, 5>(a_bhnl).exp() * mask_11nl.unsqueeze_dim::<5>(4);
        san(&d_bhnl);
        san(&l_bhnll);

        // ── Per-chunk `Σₜ` reductions (all over the masked `t` axis) ──────────
        let sd1_bhn = d_bhnl.clone().sum_dim(3).squeeze_dim::<3>(3); // Σₜ dₜ
        let sd2_bhn = d_bhnl
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .squeeze_dim::<3>(3); // Σₜ dₜ²
        // wⱼ = Σₜ dₜ·L[t,j] (a row-vector × matrix product).
        let w_bhnl = d_bhnl
            .unsqueeze_dim::<5>(3)
            .matmul(l_bhnll.clone())
            .squeeze_dim::<4>(3);
        // uⱼ = Σₜ L[t,j].
        let u_bhnl = l_bhnll.clone().sum_dim(3).squeeze_dim::<4>(3);
        // K[j,j'] = Σₜ L[t,j]·L[t,j'] = LᵀL.
        let k_bhnll = l_bhnll.clone().permute([0, 1, 2, 4, 3]).matmul(l_bhnll);

        // ── Assemble the three second-moment terms per chunk ──────────────────
        let x_bnhpl = self.x_bnlhp.clone().permute([0, 1, 3, 4, 2]);
        let bbar_bnhlr = delta_b_bnlhr.permute([0, 1, 3, 2, 4]);
        let state_in_bnhrp = state_in_bnhpr.clone().permute([0, 1, 2, 4, 3]);

        // Input² term: B̄ᵀ·(K ∘ XXᵀ)·B̄.
        let gram_x_bnhll = x_bnhpl.clone().permute([0, 1, 2, 4, 3]).matmul(x_bnhpl.clone());
        let kg_bnhll = k_bhnll.permute([0, 2, 1, 3, 4]) * gram_x_bnhll;
        let term_input_bnhrr = bbar_bnhlr
            .clone()
            .permute([0, 1, 2, 4, 3])
            .matmul(kg_bnhll)
            .matmul(bbar_bnhlr.clone());

        // Cross term: h₋ᵀP + Pᵀh₋, P = Xᵀ·diag(w)·B̄.
        let w_bnhl1 = w_bhnl.permute([0, 2, 1, 3]).unsqueeze_dim::<5>(4);
        let p_bnhpr = x_bnhpl.clone().matmul(w_bnhl1 * bbar_bnhlr.clone());
        let cross_bnhrr = state_in_bnhrp.clone().matmul(p_bnhpr);
        let term_cross_bnhrr = cross_bnhrr.clone() + cross_bnhrr.permute([0, 1, 2, 4, 3]);

        // Carried-state term: (Σₜ dₜ²)·h₋ᵀh₋.
        let sd2_bnh11 = sd2_bhn.permute([0, 2, 1]).unsqueeze_dims::<5>(&[3, 4]);
        let term_carry_bnhrr = sd2_bnh11 * state_in_bnhrp.matmul(state_in_bnhpr.clone());

        let m2_bhrr = (term_carry_bnhrr + term_cross_bnhrr + term_input_bnhrr)
            .sum_dim(1)
            .squeeze_dim::<4>(1);
        san(&m2_bhrr);

        // ── First moment: Σₜ hₜ = (Σₜ dₜ)·h₋ + Xᵀ·diag(u)·B̄, then pool `p` ────
        let sd1_bnh11 = sd1_bhn.permute([0, 2, 1]).unsqueeze_dims::<5>(&[3, 4]);
        let u_bnhl1 = u_bhnl.permute([0, 2, 1, 3]).unsqueeze_dim::<5>(4);
        let sum_h_bnhpr = sd1_bnh11 * state_in_bnhpr + x_bnhpl.matmul(u_bnhl1 * bbar_bnhlr);
        let m1_bhr = sum_h_bnhpr
            .sum_dim(1)
            .sum_dim(3)
            .reshape([batch, nheads, state_rank]);
        san(&m1_bhr);

        StateMoments {
            m2_bhrr,
            m1_bhr,
            count: valid_len * per_head_dim,
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
