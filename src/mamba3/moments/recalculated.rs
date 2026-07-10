//! # Physical-frame moments with a custom, memory-efficient backward
//!
//! The forward is the same serial chunkwise computation as
//! [`Mamba3MomentsInput::state_moments_phys`](super::Mamba3MomentsInput::state_moments_phys),
//! but routed through the [`Mamba3MomentsBackendExt`] trait so `Autodiff`
//! backends substitute a **custom recompute backward** (see
//! [`super::backward`]): the node saves only its five leaf inputs and the tiny
//! `(m2, m1)` outputs; backprop re-materialises one chunk of states at a time
//! — the full per-token state trajectory that plain autodiff would retain is
//! never kept alive. This is the at-scale execution model.
//!
//! The default body (every plain backend) runs the identical math on `B`'s
//! primitives via the rank-tagged [`F`] wrapper. The rotation is passed as a
//! rank-erased primitive plus a `quaternion` flag (`[b, s, h, a]` cumulative
//! angles, or `[b, s, h, J, 4]` cumulative unit quaternions), so a single
//! trait method covers both algebras.

#![allow(non_snake_case)]

use crate::utils::fprim::{F, Mask, san};
use burn::backend::tensor::{Device, FloatTensor};
use burn::backend::*;
use burn::backend::{Backend, backend_extension};

// ---------------------------------------------------------------------------
// Rotation primitive (rank-erased seam form)
// ---------------------------------------------------------------------------

/// The per-token cumulative rotation on `B`'s primitives — the rank-tagged
/// counterpart of [`RotationSeq`](crate::mamba3::rotation::RotationSeq).
pub(crate) enum RotPrim<B: Backend> {
    /// Cumulative angles `[batch, sequence, nheads, num_rope_angles]`.
    Complex {
        cum_bsha: F<B, 4>,
        rope_dim: usize,
        rotate_pairwise: bool,
    },
    /// Cumulative unit quaternions `[batch, sequence, nheads, blocks, 4]`.
    Quaternion { cum_bshj4: F<B, 5> },
}

impl<B: Backend> RotPrim<B> {
    /// Wrap the rank-erased seam arguments.
    pub fn wrap(
        rot: FloatTensor<B>,
        quaternion: bool,
        rope_dim: usize,
        rotate_pairwise: bool,
    ) -> Self {
        if quaternion {
            RotPrim::Quaternion {
                cum_bshj4: F::<B, 5>::new(rot),
            }
        } else {
            RotPrim::Complex {
                cum_bsha: F::<B, 4>::new(rot),
                rope_dim,
                rotate_pairwise,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive helpers (shared by the forward default body and the backward)
// ---------------------------------------------------------------------------

/// Chunk-local decay: `dₜ = exp(a_cum[t])` (`[b, h, l]`) and the
/// 1-semiseparable mask `L[t, j] = exp(a_cum[t] − a_cum[j])` (`[b, h, l, l]`,
/// zero strictly above the diagonal).
pub(crate) fn chunk_decay<B: Backend>(da_blh: F<B, 3>) -> (F<B, 3>, F<B, 4>) {
    let [batch, chunk_len, nheads] = da_blh.dims();
    let device = da_blh.device();
    let a_bhl = da_blh.permute([0, 2, 1]).cumsum(2);
    let d_bhl = a_bhl.clone().exp();
    let target_bhll = a_bhl
        .clone()
        .unsqueeze_dim::<4>(3)
        .expand([batch, nheads, chunk_len, chunk_len]);
    let source_bhll = a_bhl
        .unsqueeze_dim::<4>(2)
        .expand([batch, nheads, chunk_len, chunk_len]);
    let above_diag = Mask::tril_mask(chunk_len, chunk_len, 0, &device)
        .reshape([1, 1, chunk_len, chunk_len])
        .expand([batch, nheads, chunk_len, chunk_len]);
    let l_bhll = (target_bhll - source_bhll)
        .mask_fill(above_diag, f32::NEG_INFINITY)
        .exp();
    (d_bhl, l_bhll)
}

/// One chunk of cache-frame states
/// `h̃[t] = dₜ·h₋ + Σ_{j≤t,c} L[t,j]·x̂[j,c] ⊗ b̂[j,c]` — the channel axis is
/// folded into the write index for one batched GEMM.
///
/// # Shapes
/// - `xhat_blchp` / `bhat_blchr`: one chunk of the combined injections.
/// - returns `states_bhlpr`.
pub(crate) fn chunk_states<B: Backend>(
    xhat_blchp: F<B, 5>,
    bhat_blchr: F<B, 5>,
    d_bhl: &F<B, 3>,
    l_bhll: &F<B, 4>,
    h_in_bhpr: &F<B, 4>,
) -> F<B, 5> {
    let [batch, chunk_len, chan, nheads, per_head_dim] = xhat_blchp.dims();
    let [.., state_rank] = bhat_blchr.dims();

    let xhat_bh1jcp = xhat_blchp
        .permute([0, 3, 1, 2, 4]) // [b, h, j, c, p]
        .unsqueeze_dim::<6>(2); // [b, h, 1, j, c, p]
    let l_bhtj11 = l_bhll.clone().unsqueeze_dims::<6>(&[4, 5]);
    let xw_bhtJp = (l_bhtj11
        * xhat_bh1jcp.expand([batch, nheads, chunk_len, chunk_len, chan, per_head_dim]))
    .reshape([batch, nheads, chunk_len, chunk_len * chan, per_head_dim]);

    let bhat_bh1Jr = bhat_blchr
        .permute([0, 3, 1, 2, 4]) // [b, h, j, c, r]
        .reshape([batch, nheads, 1, chunk_len * chan, state_rank])
        .expand([batch, nheads, chunk_len, chunk_len * chan, state_rank]);

    let intra_bhtpr = xw_bhtJp.permute([0, 1, 2, 4, 3]).matmul(bhat_bh1Jr);

    let carried_bhtpr = d_bhl
        .clone()
        .unsqueeze_dims::<5>(&[3, 4])
        .expand([batch, nheads, chunk_len, per_head_dim, state_rank])
        * h_in_bhpr
            .clone()
            .unsqueeze_dim::<5>(2)
            .expand([batch, nheads, chunk_len, per_head_dim, state_rank]);
    carried_bhtpr + intra_bhtpr
}

/// The per-token write `W_j = Σ_c x̂[j,c] ⊗ b̂[j,c]` (`[b, h, j, p, r]`) — used
/// by the backward's `da` gradient.
pub(crate) fn chunk_writes<B: Backend>(xhat_blchp: F<B, 5>, bhat_blchr: F<B, 5>) -> F<B, 5> {
    let xh_bhjpc = xhat_blchp.permute([0, 3, 1, 4, 2]); // [b, h, j, p, c]
    let bh_bhjcr = bhat_blchr.permute([0, 3, 1, 2, 4]); // [b, h, j, c, r]
    xh_bhjpc.matmul(bh_bhjcr)
}

/// Quaternion Hamilton product on the packed trailing `(w, x, y, z)` axis
/// (primitive port of `rotation::quat_mul`).
pub(crate) fn quat_mul_prim<B: Backend, const D: usize>(a: F<B, D>, b: F<B, D>) -> F<B, D> {
    let n = D - 1;
    let comp = |t: &F<B, D>, i: usize| t.clone().narrow(n, i, 1);
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
    F::cat(vec![w, x, y, z], n)
}

/// Quaternion conjugate on the packed trailing axis.
pub(crate) fn quat_conj_prim<B: Backend, const D: usize>(q: F<B, D>) -> F<B, D> {
    let n = D - 1;
    let w = q.clone().narrow(n, 0, 1);
    let xyz = q.narrow(n, 1, 3);
    F::cat(vec![w, -xyz], n)
}

/// The rotation's per-pair `cos`/`sin` for one chunk, expanded over
/// `per_head_dim`: `[b, h, l, p, a]` each. `transpose` negates the angle —
/// `false` applies the **de-rotation** `R(−θ)` (cache → physical), `true` its
/// transpose `R(+θ)` (used on cotangents).
fn chunk_cos_sin<B: Backend>(
    cum_bsha: &F<B, 4>,
    start: usize,
    len: usize,
    per_head_dim: usize,
    transpose: bool,
) -> (F<B, 5>, F<B, 5>) {
    let [batch, _seq, nheads, num_angles] = cum_bsha.dims();
    let theta_bhlpa = cum_bsha
        .clone()
        .narrow(1, start, len)
        .permute([0, 2, 1, 3]) // [b, h, l, a]
        .unsqueeze_dim::<5>(3)
        .expand([batch, nheads, len, per_head_dim, num_angles]);
    let theta = if transpose { theta_bhlpa } else { -theta_bhlpa };
    (theta.clone().cos(), theta.sin())
}

/// Rotate the `state_rank` axis of one chunk of states by the per-token
/// cumulative rotation. `transpose = false` is the **de-rotation** to the
/// physical frame (`R(−θ)` / left-multiply by `Q`); `transpose = true` is its
/// transpose (`R(+θ)` / left-multiply by `conj(Q)`), the map cotangents take
/// back to the cache frame.
///
/// # Shapes
/// - `v_bhlpr`: `[batch, nheads, len, per_head_dim, state_rank]`
pub(crate) fn rotate_chunk<B: Backend>(
    v_bhlpr: F<B, 5>,
    rot: &RotPrim<B>,
    start: usize,
    transpose: bool,
) -> F<B, 5> {
    let [batch, nheads, len, per_head_dim, state_rank] = v_bhlpr.dims();
    match rot {
        RotPrim::Complex {
            cum_bsha,
            rope_dim,
            rotate_pairwise,
        } => {
            let rope_dim = *rope_dim;
            if rope_dim == 0 {
                return v_bhlpr;
            }
            let (cos, sin) = chunk_cos_sin::<B>(cum_bsha, start, len, per_head_dim, transpose);
            // R(φ) on a pair (x, y): x' = cos·x − sin·y ; y' = sin·x + cos·y.
            if *rotate_pairwise {
                // Interleaved: adjacent pairs within the rotated prefix.
                let num_angles = rope_dim / 2;
                let active = v_bhlpr.clone().narrow(4, 0, rope_dim).reshape([
                    batch,
                    nheads,
                    len,
                    per_head_dim,
                    num_angles,
                    2,
                ]);
                let x0 = active.clone().narrow(5, 0, 1).squeeze_dim::<5>(5);
                let x1 = active.narrow(5, 1, 1).squeeze_dim::<5>(5);
                let x0r = cos.clone() * x0.clone() - sin.clone() * x1.clone();
                let x1r = sin * x0 + cos * x1;
                let rotated = F::stack::<6>(vec![x0r, x1r], 5).reshape([
                    batch,
                    nheads,
                    len,
                    per_head_dim,
                    rope_dim,
                ]);
                if rope_dim == state_rank {
                    rotated
                } else {
                    let tail = v_bhlpr.narrow(4, rope_dim, state_rank - rope_dim);
                    F::cat(vec![rotated, tail], 4)
                }
            } else {
                // Half-and-half: pair distance is state_rank/2.
                let half = state_rank / 2;
                let num_angles = rope_dim / 2;
                let h1 = v_bhlpr.clone().narrow(4, 0, num_angles);
                let h2 = v_bhlpr.clone().narrow(4, half, num_angles);
                let h1r = cos.clone() * h1.clone() - sin.clone() * h2.clone();
                let h2r = sin * h1 + cos * h2;
                if num_angles == half {
                    F::cat(vec![h1r, h2r], 4)
                } else {
                    let h1_pass = v_bhlpr.clone().narrow(4, num_angles, half - num_angles);
                    let h2_pass = v_bhlpr.narrow(4, half + num_angles, half - num_angles);
                    F::cat(vec![h1r, h1_pass, h2r, h2_pass], 4)
                }
            }
        }
        RotPrim::Quaternion { cum_bshj4 } => {
            let blocks = cum_bshj4.dims()[3];
            let rope_width = 4 * blocks;
            let q_bhlpj4 = cum_bshj4
                .clone()
                .narrow(1, start, len)
                .permute([0, 2, 1, 3, 4]) // [b, h, l, J, 4]
                .unsqueeze_dim::<6>(3)
                .expand([batch, nheads, len, per_head_dim, blocks, 4]);
            // De-rotation left-multiplies by Q (B/C absorbed conj(Q));
            // the transpose left-multiplies by conj(Q).
            let q = if transpose {
                quat_conj_prim(q_bhlpj4)
            } else {
                q_bhlpj4
            };
            let active = v_bhlpr.clone().narrow(4, 0, rope_width).reshape([
                batch,
                nheads,
                len,
                per_head_dim,
                blocks,
                4,
            ]);
            let rotated = quat_mul_prim(q, active).reshape([
                batch,
                nheads,
                len,
                per_head_dim,
                rope_width,
            ]);
            if rope_width == state_rank {
                rotated
            } else {
                let tail = v_bhlpr.narrow(4, rope_width, state_rank - rope_width);
                F::cat(vec![rotated, tail], 4)
            }
        }
    }
}

/// Float validity mask `[1, 1, l, 1, 1]` for a chunk starting at `start`
/// (`1` for global positions `< valid_len`, `0` for pads).
pub(crate) fn chunk_mask<B: Backend>(
    chunk_len: usize,
    start: usize,
    valid_len: usize,
    device: &Device<B>,
    dtype: FloatDType,
) -> F<B, 5> {
    let valid = valid_len.saturating_sub(start).min(chunk_len);
    let ones = F::<B, 1>::full([valid], 1.0, device, dtype);
    let mask = if valid < chunk_len {
        F::cat(
            vec![ones, F::<B, 1>::zeros([chunk_len - valid], device, dtype)],
            0,
        )
    } else {
        ones
    };
    mask.reshape([1, 1, chunk_len, 1, 1])
}

/// The serial chunkwise forward on primitives — the exact math of
/// [`Mamba3MomentsInput::state_moments_phys`](super::Mamba3MomentsInput::state_moments_phys).
///
/// Returns `(m2_bhrr, m1_bhr)`.
pub(crate) fn moments_phys_fwd<B: Backend>(
    xhat_bnlchp: F<B, 6>,
    bhat_bnlchr: F<B, 6>,
    da_bnlh: F<B, 4>,
    rot: &RotPrim<B>,
    initial_state_bhpr: F<B, 4>,
    valid_len: usize,
) -> (F<B, 4>, F<B, 3>) {
    let [batch, nchunks, chunk_len, _chan, nheads, per_head_dim] = xhat_bnlchp.dims();
    let [.., state_rank] = bhat_bnlchr.dims();
    let device = xhat_bnlchp.device();
    let dtype = xhat_bnlchp.dtype();

    let mut h_bhpr = initial_state_bhpr;
    let mut m2_bhrr = F::<B, 4>::zeros([batch, nheads, state_rank, state_rank], &device, dtype);
    let mut m1_bhr = F::<B, 3>::zeros([batch, nheads, state_rank], &device, dtype);

    for n in 0..nchunks {
        let start = n * chunk_len;
        if start >= valid_len {
            break;
        }
        let da_blh = da_bnlh.clone().narrow(1, n, 1).squeeze_dim::<3>(1);
        let (d_bhl, l_bhll) = chunk_decay::<B>(da_blh);
        let xhat_blchp = xhat_bnlchp.clone().narrow(1, n, 1).squeeze_dim::<5>(1);
        let bhat_blchr = bhat_bnlchr.clone().narrow(1, n, 1).squeeze_dim::<5>(1);

        let states_bhlpr = chunk_states::<B>(xhat_blchp, bhat_blchr, &d_bhl, &l_bhll, &h_bhpr);
        h_bhpr = states_bhlpr
            .clone()
            .narrow(2, chunk_len - 1, 1)
            .squeeze_dim::<4>(2);

        let phys_bhlpr = rotate_chunk::<B>(states_bhlpr, rot, start, false);
        let masked_bhlpr =
            phys_bhlpr * chunk_mask::<B>(chunk_len, start, valid_len, &device, dtype).expand([
                batch,
                nheads,
                chunk_len,
                per_head_dim,
                state_rank,
            ]);

        let masked_bhLr =
            masked_bhlpr.reshape([batch, nheads, chunk_len * per_head_dim, state_rank]);
        m2_bhrr = m2_bhrr
            + masked_bhLr
                .clone()
                .permute([0, 1, 3, 2])
                .matmul(masked_bhLr.clone());
        m1_bhr = m1_bhr + masked_bhLr.sum_dim(2).squeeze_dim::<3>(2);
    }
    san(&m2_bhrr);
    san(&m1_bhr);
    (m2_bhrr, m1_bhr)
}

// ---------------------------------------------------------------------------
// Backend extension trait
// ---------------------------------------------------------------------------

/// Extends the backend with the physical-frame state-moments computation.
///
/// The default body runs the serial chunkwise forward on primitive tensors.
/// The `Autodiff` wrapper overrides it with the memory-efficient recompute
/// backward (in [`super::backward`]) that saves only the leaf inputs.
#[backend_extension(
    Cpu:  cfg(feature = "backend-cpu"),
    Cuda: cfg(feature = "backend-cuda"),
    Rocm:  cfg(feature = "backend-rocm"),
    Metal:  cfg(feature = "backend-metal"),
    Vulkan:  cfg(feature = "backend-vulkan"),
    Wgpu:  cfg(feature = "backend-wgpu"),
    WebGpu:  cfg(feature = "backend-webgpu"),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait Mamba3MomentsBackendExt: Backend {
    /// Pooled physical-frame state moments of the combined injections (see
    /// `mamba3/moments.rs` for the math). Returns `(m2_bhrr, m1_bhr)` raw
    /// sums, masked to `valid_len`.
    ///
    /// `rot` is rank-erased: cumulative angles `[b, s, h, a]` when
    /// `quaternion = false` (with `rope_dim` / `rotate_pairwise` selecting the
    /// pairing), cumulative unit quaternions `[b, s, h, J, 4]` otherwise
    /// (`rope_dim` / `rotate_pairwise` ignored).
    #[allow(clippy::too_many_arguments)]
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
        let rot = RotPrim::<Self>::wrap(rot, quaternion, rope_dim, rotate_pairwise);
        let (m2, m1) = moments_phys_fwd::<Self>(
            F::<Self, 6>::new(xhat_bnlchp),
            F::<Self, 6>::new(bhat_bnlchr),
            F::<Self, 4>::new(da_bnlh),
            &rot,
            F::<Self, 4>::new(initial_state_bhpr),
            valid_len,
        );
        (m2.inner(), m1.inner())
    }
}

// Per-backend impls delegate to the trait's default body; the custom autodiff
// backward lives in `super::backward` as a separate `Autodiff<B>` impl.
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3MomentsBackendExt);
