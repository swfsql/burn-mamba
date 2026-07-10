//! # Custom autodiff node for the physical-frame state moments
//!
//! Implements [`Mamba3MomentsBackendExt`] for `Autodiff<B>` via a single Burn
//! [`Backward`] node. The forward saves only its five leaf inputs; backprop
//! re-materialises **one chunk of states at a time** (the `SerialRecalculated`
//! discipline) and evaluates the analytic VJPs, so the full per-token state
//! trajectory plain autodiff would retain is never kept alive.
//!
//! ## Gradient math (per chunk, reverse order, cotangent carry `d_h₊`)
//!
//! Forward: `h̃[t] = dₜ·h₋ + Σⱼ L[t,j]·Wⱼ` with writes `Wⱼ = Σ_c x̂ⱼ_c ⊗ b̂ⱼ_c`,
//! `cₜ = Rₜ†·h̃[t]` (de-rotation), `m2 += Σₜ maskₜ·cₜᵀcₜ`, `m1 += Σₜ maskₜ·cₜ`.
//!
//! - `d_c[t] = maskₜ·(cₜ·(d_m2 + d_m2ᵀ) + d_m1)` — the moments' VJP.
//! - `d_h̃[t] = Rₜ·d_c[t]` (the de-rotation's transpose), plus the **rotation's
//!   own gradient**: per complex pair, `∂cₜ/∂θ = (c_y, −c_x)`, so
//!   `d_θ = Σ_p (d_c_x·c_y − d_c_y·c_x)`; per quaternion block (`c = Q ⊗ h̃`),
//!   `d_Q = Σ_p d_c ⊗ conj(h̃)` (bilinear-product VJP, exact for all
//!   quaternions). This is what lets a PR penalty shape the rotation itself.
//! - `G[t] = d_h̃[t] + δ_{t,l−1}·d_h₊` (the next chunk consumed `h̃[l−1]` as
//!   its `h₋`).
//! - `d_h₋ = Σₜ dₜ·G[t]` (the new carry), `d_Wⱼ = Σₜ L[t,j]·G[t]`, then
//!   `d_x̂ⱼ_c = d_Wⱼ·b̂ⱼ_c`, `d_b̂ⱼ_c = d_Wⱼᵀ·x̂ⱼ_c`.
//! - `d_da`: through `dₜ = exp(Σ_{s≤t} da_s)` and `L[t,j] = exp(Σ_{j<s≤t} da_s)`,
//!   with `scalₜ = ⟨G[t], h₋⟩·dₜ` and `A[t,j] = ⟨G[t], Wⱼ⟩·L[t,j]`:
//!   `d_da_s = Σ_{t≥s} scalₜ + Σ_{j<s≤t} A[t,j]`
//!   `      = rev_cumsum(scal + Σⱼ A[·,j])_s − rev_cumsum(Σₜ A[t,·])_s`
//!   (the pair `(t, j)` contributes to exactly the positions `j < s ≤ t`, and
//!   `1[s≤t] − 1[s≤j]` is that indicator; `A` is lower-triangular).
//! - The boundary carries `h₋(n)` are recomputed by a cheap chunk-level pass
//!   (decay-to-chunk-end GEMM, `l×` cheaper than the states GEMM).
//!
//! Chunks entirely past `valid_len` were never entered by the forward
//! (`break`), so their gradients are zero and the reverse loop skips them.
//!
//! The two outputs `(m2, m1)` are flattened into one tracked tensor via
//! [`crate::utils::combined_grad`] so a single `Backward<B, 5>` node covers
//! both.

#![allow(non_snake_case)]

use super::recalculated::{
    Mamba3MomentsBackendExt, RotPrim, chunk_decay, chunk_mask, chunk_states, chunk_writes,
    quat_conj_prim, quat_mul_prim, rotate_chunk,
};
use crate::utils::fprim::F;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes};

/// All input gradients of the moments node.
struct MomentsGrads<B: Backend> {
    d_xhat_bnlchp: F<B, 6>,
    d_bhat_bnlchr: F<B, 6>,
    d_da_bnlh: F<B, 4>,
    d_rot: <B as BackendTypes>::FloatTensorPrimitive,
    d_initial_bhpr: F<B, 4>,
}

/// The per-pair angle gradient of the de-rotation (see the module header).
fn d_theta_chunk<B: Backend>(
    phys_bhlpr: &F<B, 5>,
    d_phys_bhlpr: &F<B, 5>,
    num_angles_total: usize,
    rope_dim: usize,
    rotate_pairwise: bool,
) -> F<B, 4> {
    let [batch, nheads, len, _p, state_rank] = phys_bhlpr.dims();
    if rope_dim == 0 {
        let device = phys_bhlpr.device();
        let dtype = phys_bhlpr.dtype();
        return F::<B, 4>::zeros([batch, nheads, len, num_angles_total], &device, dtype);
    }
    let num_angles = rope_dim / 2;
    let (px, py, dx, dy) = if rotate_pairwise {
        let split = |t: &F<B, 5>| {
            let a = t.clone().narrow(4, 0, rope_dim).reshape([
                batch,
                nheads,
                len,
                phys_bhlpr.dims()[3],
                num_angles,
                2,
            ]);
            (
                a.clone().narrow(5, 0, 1).squeeze_dim::<5>(5),
                a.narrow(5, 1, 1).squeeze_dim::<5>(5),
            )
        };
        let (px, py) = split(phys_bhlpr);
        let (dx, dy) = split(d_phys_bhlpr);
        (px, py, dx, dy)
    } else {
        let half = state_rank / 2;
        (
            phys_bhlpr.clone().narrow(4, 0, num_angles),
            phys_bhlpr.clone().narrow(4, half, num_angles),
            d_phys_bhlpr.clone().narrow(4, 0, num_angles),
            d_phys_bhlpr.clone().narrow(4, half, num_angles),
        )
    };
    // ∂c/∂θ = (c_y, −c_x) ⇒ d_θ = Σ_p (d_c_x·c_y − d_c_y·c_x).
    (dx * py - dy * px).sum_dim(3).squeeze_dim::<4>(3)
}

/// The per-block quaternion gradient of the de-rotation (`c = Q ⊗ h̃` ⇒
/// `d_Q = Σ_p d_c ⊗ conj(h̃)`).
fn d_quat_chunk<B: Backend>(
    states_bhlpr: &F<B, 5>,
    d_phys_bhlpr: &F<B, 5>,
    blocks: usize,
) -> F<B, 5> {
    let [batch, nheads, len, per_head_dim, _r] = states_bhlpr.dims();
    let rope_width = 4 * blocks;
    let as_blocks = |t: &F<B, 5>| {
        t.clone().narrow(4, 0, rope_width).reshape([
            batch,
            nheads,
            len,
            per_head_dim,
            blocks,
            4,
        ])
    };
    quat_mul_prim(as_blocks(d_phys_bhlpr), quat_conj_prim(as_blocks(states_bhlpr)))
        .sum_dim(3)
        .squeeze_dim::<5>(3)
}

/// Cotangent scatter: `[b,h,p,r]` placed at the last position of a
/// `[b,h,l,p,r]` chunk (zeros elsewhere).
fn scatter_last<B: Backend>(d_carry_bhpr: F<B, 4>, chunk_len: usize) -> F<B, 5> {
    let [batch, nheads, per_head_dim, state_rank] = d_carry_bhpr.dims();
    let device = d_carry_bhpr.device();
    let dtype = d_carry_bhpr.dtype();
    let dc = d_carry_bhpr.unsqueeze_dim::<5>(2);
    if chunk_len == 1 {
        dc
    } else {
        F::cat(
            vec![
                F::<B, 5>::zeros(
                    [batch, nheads, chunk_len - 1, per_head_dim, state_rank],
                    &device,
                    dtype,
                ),
                dc,
            ],
            2,
        )
    }
}

/// Suffix sum along `dim` (`out[s] = Σ_{t≥s} in[t]`).
fn rev_cumsum<B: Backend, const D: usize>(t: F<B, D>, dim: usize) -> F<B, D> {
    t.flip(&[dim]).cumsum(dim).flip(&[dim])
}

/// Recompute the boundary carries `h₋(n)` entering each of the `processed`
/// chunks (a chunk-level pass: decay-to-chunk-end GEMM + serial accumulation).
fn boundary_carries<B: Backend>(
    xhat_bnlchp: &F<B, 6>,
    bhat_bnlchr: &F<B, 6>,
    da_bnlh: &F<B, 4>,
    initial_state_bhpr: &F<B, 4>,
    processed: usize,
) -> Vec<F<B, 4>> {
    let [batch, _n, chunk_len, chan, nheads, per_head_dim] = xhat_bnlchp.dims();
    let [.., state_rank] = bhat_bnlchr.dims();

    let mut h_bhpr = initial_state_bhpr.clone();
    let mut carries = Vec::with_capacity(processed);
    for n in 0..processed {
        carries.push(h_bhpr.clone());
        if n + 1 == processed {
            break;
        }
        let a_bhl = da_bnlh
            .clone()
            .narrow(1, n, 1)
            .squeeze_dim::<3>(1)
            .permute([0, 2, 1])
            .cumsum(2);
        let a_last_bh1 = a_bhl.clone().narrow(2, chunk_len - 1, 1);
        let scale_bhj = (a_last_bh1
            .clone()
            .expand([batch, nheads, chunk_len])
            - a_bhl)
            .exp();
        let xh_bhjcp = xhat_bnlchp
            .clone()
            .narrow(1, n, 1)
            .squeeze_dim::<5>(1)
            .permute([0, 3, 1, 2, 4]);
        let bh_bhJr = bhat_bnlchr
            .clone()
            .narrow(1, n, 1)
            .squeeze_dim::<5>(1)
            .permute([0, 3, 1, 2, 4])
            .reshape([batch, nheads, chunk_len * chan, state_rank]);
        let xw_bhJp = (xh_bhjcp
            * scale_bhj
                .unsqueeze_dims::<5>(&[3, 4])
                .expand([batch, nheads, chunk_len, chan, per_head_dim]))
        .reshape([batch, nheads, chunk_len * chan, per_head_dim]);
        let contribution_bhpr = xw_bhJp.permute([0, 1, 3, 2]).matmul(bh_bhJr);
        let d_last_bhpr = a_last_bh1
            .exp()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, per_head_dim, state_rank]);
        h_bhpr = d_last_bhpr * h_bhpr + contribution_bhpr;
    }
    carries
}

/// The full recompute backward (see the module header).
#[allow(clippy::too_many_arguments)]
fn moments_phys_bwd<B: Backend>(
    d_m2_bhrr: F<B, 4>,
    d_m1_bhr: F<B, 3>,
    xhat_bnlchp: F<B, 6>,
    bhat_bnlchr: F<B, 6>,
    da_bnlh: F<B, 4>,
    rot: RotPrim<B>,
    initial_state_bhpr: F<B, 4>,
    valid_len: usize,
) -> MomentsGrads<B> {
    let [batch, nchunks, chunk_len, chan, nheads, per_head_dim] = xhat_bnlchp.dims();
    let [.., state_rank] = bhat_bnlchr.dims();
    let device = xhat_bnlchp.device();
    let dtype = xhat_bnlchp.dtype();
    let processed = valid_len.div_ceil(chunk_len).min(nchunks);

    let h_ins = boundary_carries::<B>(
        &xhat_bnlchp,
        &bhat_bnlchr,
        &da_bnlh,
        &initial_state_bhpr,
        processed,
    );
    let d_m2sym_bhrr = d_m2_bhrr.clone() + d_m2_bhrr.permute([0, 1, 3, 2]);

    let zero_dx = || F::<B, 5>::zeros([batch, chunk_len, chan, nheads, per_head_dim], &device, dtype);
    let zero_db = || F::<B, 5>::zeros([batch, chunk_len, chan, nheads, state_rank], &device, dtype);
    let zero_dda = || F::<B, 3>::zeros([batch, chunk_len, nheads], &device, dtype);
    let mut d_xhat_chunks: Vec<F<B, 5>> = (processed..nchunks).map(|_| zero_dx()).collect();
    let mut d_bhat_chunks: Vec<F<B, 5>> = (processed..nchunks).map(|_| zero_db()).collect();
    let mut d_da_chunks: Vec<F<B, 3>> = (processed..nchunks).map(|_| zero_dda()).collect();
    // d_rot chunks, kept rank-erased (angles [b,l,h,a] | quats [b,l,h,J,4]).
    let mut d_rot_chunks_rev: Vec<<B as BackendTypes>::FloatTensorPrimitive> = Vec::new();

    let mut d_carry_bhpr = F::<B, 4>::zeros([batch, nheads, per_head_dim, state_rank], &device, dtype);

    for n in (0..processed).rev() {
        let start = n * chunk_len;
        // ── Recompute this chunk's states and physical states ────────────────
        let da_blh = da_bnlh.clone().narrow(1, n, 1).squeeze_dim::<3>(1);
        let (d_bhl, l_bhll) = chunk_decay::<B>(da_blh);
        let xhat_blchp = xhat_bnlchp.clone().narrow(1, n, 1).squeeze_dim::<5>(1);
        let bhat_blchr = bhat_bnlchr.clone().narrow(1, n, 1).squeeze_dim::<5>(1);
        let states_bhlpr =
            chunk_states::<B>(xhat_blchp.clone(), bhat_blchr.clone(), &d_bhl, &l_bhll, &h_ins[n]);
        let phys_bhlpr = rotate_chunk::<B>(states_bhlpr.clone(), &rot, start, false);

        // ── Moments VJP: d_c = mask·(c·(d_m2 + d_m2ᵀ) + d_m1) ────────────────
        let mask_bhlpr = chunk_mask::<B>(chunk_len, start, valid_len, &device, dtype).expand([
            batch,
            nheads,
            chunk_len,
            per_head_dim,
            state_rank,
        ]);
        let d_phys_bhlpr = (phys_bhlpr.clone().matmul(
            d_m2sym_bhrr
                .clone()
                .unsqueeze_dim::<5>(2)
                .expand([batch, nheads, chunk_len, state_rank, state_rank]),
        ) + d_m1_bhr
            .clone()
            .unsqueeze_dims::<5>(&[2, 3])
            .expand([batch, nheads, chunk_len, per_head_dim, state_rank]))
            * mask_bhlpr;

        // ── Rotation VJP: cotangent back to the cache frame + d_rot ──────────
        let d_states_bhlpr = rotate_chunk::<B>(d_phys_bhlpr.clone(), &rot, start, true);
        let d_rot_chunk = match &rot {
            RotPrim::Complex {
                cum_bsha,
                rope_dim,
                rotate_pairwise,
            } => {
                let num_angles_total = cum_bsha.dims()[3];
                d_theta_chunk::<B>(
                    &phys_bhlpr,
                    &d_phys_bhlpr,
                    num_angles_total,
                    *rope_dim,
                    *rotate_pairwise,
                )
                .permute([0, 2, 1, 3]) // [b, l, h, a]
                .inner()
            }
            RotPrim::Quaternion { cum_bshj4 } => {
                let blocks = cum_bshj4.dims()[3];
                d_quat_chunk::<B>(&states_bhlpr, &d_phys_bhlpr, blocks)
                    .permute([0, 2, 1, 3, 4]) // [b, l, h, J, 4]
                    .inner()
            }
        };
        d_rot_chunks_rev.push(d_rot_chunk);

        // ── Chunk-state cotangent (moments + the next chunk's h₋ use) ────────
        let g_bhlpr = d_states_bhlpr + scatter_last::<B>(d_carry_bhpr.clone(), chunk_len);

        // ── d_h₋ (the new carry) ──────────────────────────────────────────────
        d_carry_bhpr = (g_bhlpr.clone()
            * d_bhl
                .clone()
                .unsqueeze_dims::<5>(&[3, 4])
                .expand([batch, nheads, chunk_len, per_head_dim, state_rank]))
        .sum_dim(2)
        .squeeze_dim::<4>(2);

        // ── d_W, then d_x̂ / d_b̂ ─────────────────────────────────────────────
        let g_bhtPR = g_bhlpr
            .clone()
            .reshape([batch, nheads, chunk_len, per_head_dim * state_rank]);
        let dw_bhjpr = l_bhll
            .clone()
            .permute([0, 1, 3, 2])
            .matmul(g_bhtPR.clone())
            .reshape([batch, nheads, chunk_len, per_head_dim, state_rank]);
        let xh_bhjcp = xhat_blchp.clone().permute([0, 3, 1, 2, 4]);
        let bh_bhjcr = bhat_blchr.clone().permute([0, 3, 1, 2, 4]);
        let d_xh_bhjcp = bh_bhjcr.matmul(dw_bhjpr.clone().permute([0, 1, 2, 4, 3]));
        let d_bh_bhjcr = xh_bhjcp.matmul(dw_bhjpr);
        d_xhat_chunks.push(d_xh_bhjcp.permute([0, 2, 3, 1, 4])); // [b, l, c, h, p]
        d_bhat_chunks.push(d_bh_bhjcr.permute([0, 2, 3, 1, 4])); // [b, l, c, h, r]

        // ── d_da ──────────────────────────────────────────────────────────────
        let w_bhjpr = chunk_writes::<B>(xhat_blchp, bhat_blchr);
        let scal_bhl = (g_bhlpr
            * h_ins[n]
                .clone()
                .unsqueeze_dim::<5>(2)
                .expand([batch, nheads, chunk_len, per_head_dim, state_rank]))
        .sum_dim(4)
        .sum_dim(3)
        .reshape([batch, nheads, chunk_len])
            * d_bhl;
        let gw_bhtj = g_bhtPR.matmul(
            w_bhjpr
                .reshape([batch, nheads, chunk_len, per_head_dim * state_rank])
                .permute([0, 1, 3, 2]),
        );
        let a_bhtj = gw_bhtj * l_bhll;
        let row_bhl = a_bhtj.clone().sum_dim(3).squeeze_dim::<3>(3);
        let col_bhl = a_bhtj.sum_dim(2).squeeze_dim::<3>(2);
        let d_da_bhl = rev_cumsum::<B, 3>(scal_bhl + row_bhl, 2) - rev_cumsum::<B, 3>(col_bhl, 2);
        d_da_chunks.push(d_da_bhl.permute([0, 2, 1])); // [b, l, h]
    }

    // The reverse loop pushed processed chunks newest-first after the zero pads
    // for the skipped tail — restore forward chunk order.
    fn reorder<T>(mut v: Vec<T>, nchunks: usize, processed: usize) -> Vec<T> {
        let tail = v.split_off(nchunks - processed); // the processed chunks, reversed
        let mut ordered: Vec<T> = tail.into_iter().rev().collect();
        ordered.extend(v); // the zero chunks for the skipped tail
        ordered
    }

    let d_xhat_bnlchp = F::stack::<6>(reorder(d_xhat_chunks, nchunks, processed), 1);
    let d_bhat_bnlchr = F::stack::<6>(reorder(d_bhat_chunks, nchunks, processed), 1);
    let d_da_bnlh = F::stack::<4>(reorder(d_da_chunks, nchunks, processed), 1);

    // d_rot: processed chunks (reversed) followed by zero pads to the full
    // (padded) sequence length, concatenated along the seq axis.
    let d_rot = {
        let pad = (nchunks - processed) * chunk_len;
        match &rot {
            RotPrim::Complex { cum_bsha, .. } => {
                let num_angles = cum_bsha.dims()[3];
                let mut chunks: Vec<F<B, 4>> = d_rot_chunks_rev
                    .into_iter()
                    .rev()
                    .map(F::<B, 4>::new)
                    .collect();
                if pad > 0 {
                    chunks.push(F::<B, 4>::zeros(
                        [batch, pad, nheads, num_angles],
                        &device,
                        dtype,
                    ));
                }
                F::cat(chunks, 1).inner()
            }
            RotPrim::Quaternion { cum_bshj4 } => {
                let blocks = cum_bshj4.dims()[3];
                let mut chunks: Vec<F<B, 5>> = d_rot_chunks_rev
                    .into_iter()
                    .rev()
                    .map(F::<B, 5>::new)
                    .collect();
                if pad > 0 {
                    chunks.push(F::<B, 5>::zeros(
                        [batch, pad, nheads, blocks, 4],
                        &device,
                        dtype,
                    ));
                }
                F::cat(chunks, 1).inner()
            }
        }
    };

    MomentsGrads {
        d_xhat_bnlchp,
        d_bhat_bnlchr,
        d_da_bnlh,
        d_rot,
        d_initial_bhpr: d_carry_bhpr,
    }
}

impl<B: Backend + Mamba3MomentsBackendExt, C: CheckpointStrategy> Mamba3MomentsBackendExt
    for Autodiff<B, C>
{
    fn mamba3_state_moments_phys(
        xhat_bnlchp: FloatTensor<Self>,
        bhat_bnlchr: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        rot: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        valid_len: usize,
        quaternion: bool,
        rope_dim: usize,
        rotate_pairwise: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // ── Backward node ────────────────────────────────────────────────────
        #[derive(Debug)]
        struct MomentsBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            xhat: <B as BackendTypes>::FloatTensorPrimitive,
            bhat: <B as BackendTypes>::FloatTensorPrimitive,
            da: <B as BackendTypes>::FloatTensorPrimitive,
            rot: <B as BackendTypes>::FloatTensorPrimitive,
            initial: <B as BackendTypes>::FloatTensorPrimitive,
            flat_len_m2: usize,
            flat_len_m1: usize,
            shape_m2: [usize; 4],
            shape_m1: [usize; 3],
            valid_len: usize,
            quaternion: bool,
            rope_dim: usize,
            rotate_pairwise: bool,
        }

        impl<B: Backend + Mamba3MomentsBackendExt> Backward<B, 5> for MomentsBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_xhat, node_bhat, node_da, node_rot, node_initial] = ops.parents;
                let d_combined: <B as BackendTypes>::FloatTensorPrimitive =
                    grads.consume::<B>(&ops.node);

                let State {
                    xhat,
                    bhat,
                    da,
                    rot,
                    initial,
                    flat_len_m2,
                    flat_len_m1,
                    shape_m2,
                    shape_m1,
                    valid_len,
                    quaternion,
                    rope_dim,
                    rotate_pairwise,
                } = ops.state;

                let (d_m2, d_m1) = crate::utils::combined_grad::unflatten_pair::<B, 4, 3>(
                    d_combined,
                    flat_len_m2,
                    flat_len_m1,
                    shape_m2,
                    shape_m1,
                );

                let grads_all = moments_phys_bwd::<B>(
                    F::<B, 4>::new(d_m2),
                    F::<B, 3>::new(d_m1),
                    F::<B, 6>::new(xhat),
                    F::<B, 6>::new(bhat),
                    F::<B, 4>::new(da),
                    RotPrim::<B>::wrap(rot, quaternion, rope_dim, rotate_pairwise),
                    F::<B, 4>::new(initial),
                    valid_len,
                );

                if let Some(n) = node_xhat {
                    grads.register::<B>(n.id, grads_all.d_xhat_bnlchp.inner());
                }
                if let Some(n) = node_bhat {
                    grads.register::<B>(n.id, grads_all.d_bhat_bnlchr.inner());
                }
                if let Some(n) = node_da {
                    grads.register::<B>(n.id, grads_all.d_da_bnlh.inner());
                }
                if let Some(n) = node_rot {
                    grads.register::<B>(n.id, grads_all.d_rot);
                }
                if let Some(n) = node_initial {
                    grads.register::<B>(n.id, grads_all.d_initial_bhpr.inner());
                }
            }
        }

        // ── Shape extraction ─────────────────────────────────────────────────
        use burn::backend::TensorMetadata;
        let [batch, _n, _l, _c, nheads, _p] = xhat_bnlchp.primitive.shape().dims();
        let [.., state_rank] = bhat_bnlchr.primitive.shape().dims::<6>();
        let shape_m2: [usize; 4] = [batch, nheads, state_rank, state_rank];
        let shape_m1: [usize; 3] = [batch, nheads, state_rank];
        let flat_len_m2 = batch * nheads * state_rank * state_rank;
        let flat_len_m1 = batch * nheads * state_rank;

        // ── Register backward / run forward ──────────────────────────────────
        match MomentsBackward
            .prepare::<C>([
                xhat_bnlchp.node.clone(),
                bhat_bnlchr.node.clone(),
                da_bnlh.node.clone(),
                rot.node.clone(),
                initial_state_bhpr.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (prim_m2, prim_m1) = B::mamba3_state_moments_phys(
                    xhat_bnlchp.primitive.clone(),
                    bhat_bnlchr.primitive.clone(),
                    da_bnlh.primitive.clone(),
                    rot.primitive.clone(),
                    initial_state_bhpr.primitive.clone(),
                    valid_len,
                    quaternion,
                    rope_dim,
                    rotate_pairwise,
                );
                let (prim_combined, _, _) =
                    crate::utils::combined_grad::flatten_pair::<B>(prim_m2, prim_m1);
                let state = State {
                    xhat: xhat_bnlchp.primitive.clone(),
                    bhat: bhat_bnlchr.primitive.clone(),
                    da: da_bnlh.primitive.clone(),
                    rot: rot.primitive.clone(),
                    initial: initial_state_bhpr.primitive.clone(),
                    flat_len_m2,
                    flat_len_m1,
                    shape_m2,
                    shape_m1,
                    valid_len,
                    quaternion,
                    rope_dim,
                    rotate_pairwise,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, prim_combined);
                crate::utils::combined_grad::autodiff_unflatten_pair::<B, C, 4, 3>(
                    tracked_combined,
                    flat_len_m2,
                    flat_len_m1,
                    shape_m2,
                    shape_m1,
                )
            }
            OpsKind::UnTracked(prep) => {
                let (prim_m2, prim_m1) = B::mamba3_state_moments_phys(
                    xhat_bnlchp.primitive,
                    bhat_bnlchr.primitive,
                    da_bnlh.primitive,
                    rot.primitive,
                    initial_state_bhpr.primitive,
                    valid_len,
                    quaternion,
                    rope_dim,
                    rotate_pairwise,
                );
                let (prim_combined, _, _) =
                    crate::utils::combined_grad::flatten_pair::<B>(prim_m2, prim_m1);
                let tracked_combined: FloatTensor<Autodiff<B, C>> = prep.finish(prim_combined);
                crate::utils::combined_grad::autodiff_unflatten_pair::<B, C, 4, 3>(
                    tracked_combined,
                    flat_len_m2,
                    flat_len_m1,
                    shape_m2,
                    shape_m1,
                )
            }
        }
    }
}
