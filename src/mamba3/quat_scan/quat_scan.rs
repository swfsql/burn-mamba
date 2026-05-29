//! # Quaternion cumulative-product scan with a custom, memory-efficient backward
//!
//! The forward is the same Hillis–Steele parallel scan as
//! [`crate::mamba3::rotation::quat_cumprod`], but routed through the
//! [`Mamba3QuatScanBackendExt`] trait so that `Autodiff` backends substitute a
//! custom backward that recomputes the scan instead of retaining its
//! intermediates (see [`super::backward`]). Plain backends use the trait's
//! default body, which runs the scan on `B`'s primitives.
//!
//! The default body runs under a generic backend `B`, where the high-level
//! [`Tensor`] (pinned to `Dispatch`) is unavailable, so the quaternion algebra
//! goes through the rank-tagged [`F`] primitive wrapper. The same primitive
//! helpers ([`fquat_mul`], [`fquat_conj`], [`fquat_prefix_product`]) are reused
//! by the recompute backward.

#![allow(non_snake_case)]

use crate::utils::fprim::F;
use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::Tensor;

// ---------------------------------------------------------------------------
// Primitive quaternion algebra (rank-5: [batch, sequence, nheads, blocks, 4])
// ---------------------------------------------------------------------------

/// Hamilton product `a ⊗ b` on the trailing `(w, x, y, z)` axis (axis 4), the
/// primitive-`F` analogue of [`crate::mamba3::rotation::quat_mul`]. Broadcasts
/// over the leading dims, so a `[b,1,h,j,4]` carry multiplies a `[b,s,h,j,4]`
/// sequence.
pub(crate) fn fquat_mul<B: Backend>(a: F<B, 5>, b: F<B, 5>) -> F<B, 5> {
    let aw = a.clone().narrow(4, 0, 1);
    let ax = a.clone().narrow(4, 1, 1);
    let ay = a.clone().narrow(4, 2, 1);
    let az = a.narrow(4, 3, 1);
    let bw = b.clone().narrow(4, 0, 1);
    let bx = b.clone().narrow(4, 1, 1);
    let by = b.clone().narrow(4, 2, 1);
    let bz = b.narrow(4, 3, 1);

    let w = aw.clone() * bw.clone() - ax.clone() * bx.clone() - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x = aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
        - az.clone() * by.clone();
    let y = aw.clone() * by.clone() - ax.clone() * bz.clone() + ay.clone() * bw.clone()
        + az.clone() * bx.clone();
    let z = aw * bz + ax * by - ay * bx + az * bw;

    F::cat(vec![w, x, y, z], 4)
}

/// Quaternion conjugate `q* = (w, −x, −y, −z)` on axis 4 (primitive-`F` analogue
/// of [`crate::mamba3::rotation::quat_conj`]). For a unit quaternion `q* = q⁻¹`.
pub(crate) fn fquat_conj<B: Backend>(q: F<B, 5>) -> F<B, 5> {
    let w = q.clone().narrow(4, 0, 1);
    let xyz = q.narrow(4, 1, 3);
    F::cat(vec![w, -xyz], 4)
}

/// Pure prefix product `P[t] = qₜ ⊗ qₜ₋₁ ⊗ ⋯ ⊗ q₀` (no carry) along the sequence
/// axis (axis 1), via the same Hillis–Steele doubling as
/// [`crate::mamba3::rotation::quat_cumprod`] — `O(log seq)` dependency depth.
///
/// The carry is folded in by the caller (one extra [`fquat_mul`]); keeping `P`
/// separate is what the recompute backward needs (`G[t] = P[t] ⊗ S[t]`).
pub(crate) fn fquat_prefix_product<B: Backend>(q_bshj4: F<B, 5>) -> F<B, 5> {
    let [batch, sequence, nheads, blocks, _four] = q_bshj4.dims();
    let device = q_bshj4.device();
    let dtype = q_bshj4.dtype();

    let mut a = q_bshj4;
    let mut offset = 1usize;
    while offset < sequence {
        // identity quaternion (1,0,0,0) for the first `offset` (shifted-past-start) slots.
        let w = F::<B, 5>::full([batch, offset, nheads, blocks, 1], 1.0, &device, dtype);
        let xyz = F::<B, 5>::zeros([batch, offset, nheads, blocks, 3], &device, dtype);
        let ident = F::cat(vec![w, xyz], 4);
        let shifted = F::cat(vec![ident, a.clone()], 1).narrow(1, 0, sequence);
        a = fquat_mul::<B>(a, shifted); // recent (a) ⊗ older (shifted)
        offset *= 2;
    }
    a
}

// ---------------------------------------------------------------------------
// Backend extension trait (default body = the primitive scan)
// ---------------------------------------------------------------------------

/// Extends the backend with the quaternion cumulative-product scan.
///
/// The default body runs the Hillis–Steele scan on primitive tensors. The
/// `Autodiff` wrapper overrides it with a memory-efficient custom backward (in
/// [`super::backward`]) that recomputes the scan instead of saving its
/// intermediates.
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
pub trait Mamba3QuatScanBackendExt: Backend {
    /// Cumulative quaternion product `cum[t] = qₜ ⊗ ⋯ ⊗ q₀ ⊗ init` along the
    /// sequence axis (newest on the left), returning only `cum`.
    ///
    /// The caller derives `final_carry = cum[:, −1]` (a thin autodiff slice) in
    /// high-level land — see [`quat_cumprod_recalculated`].
    ///
    /// # Shapes
    /// - `q_bshj4`   : `[batch, sequence, nheads, blocks, 4]` — per-step **unit**
    ///   quaternions.
    /// - `init_bhj4` : `[batch, nheads, blocks, 4]` — the cross-chunk carry
    ///   (identity `(1,0,0,0)` for a fresh start).
    /// - returns `cum` : `[batch, sequence, nheads, blocks, 4]`.
    fn quat_cumprod(
        q_bshj4: FloatTensor<Self>,
        init_bhj4: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let q = F::<Self, 5>::new(q_bshj4);
        let init = F::<Self, 4>::new(init_bhj4);

        let p_bshj4 = fquat_prefix_product::<Self>(q);
        let init5_b1hj4 = init.unsqueeze_dim::<5>(1); // [batch, 1, nheads, blocks, 4]
        let cum_bshj4 = fquat_mul::<Self>(p_bshj4, init5_b1hj4); // broadcasts over seq
        cum_bshj4.inner()
    }
}

// Per-backend impls delegate to the trait's default body; the custom autodiff
// backward lives in `super::backward` as a separate `Autodiff<B>` impl.
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3QuatScanBackendExt);

// ---------------------------------------------------------------------------
// High-level wrapper (Dispatch-pinned `Tensor`)
// ---------------------------------------------------------------------------

/// Cumulative quaternion product with a custom, memory-efficient backward — the
/// drop-in for [`crate::mamba3::rotation::quat_cumprod`] used by the
/// Quaternion4D `forward`.
///
/// Routes the scan through [`Mamba3QuatScanBackendExt`] (so `Autodiff` gets the
/// recompute backward), then takes `final_carry = cum[:, −1]` as a thin autodiff
/// slice — its gradient folds back into `cum`'s gradient before the custom node
/// runs, so the node needs only the single `cum` output.
///
/// Mathematically identical to `quat_cumprod` (asserted on values and gradients
/// by the tests); only the backward's memory profile differs.
///
/// # Shapes
/// - `q_bshj4` : `[batch, sequence, nheads, blocks, 4]` — per-step **unit**
///   quaternions (the recompute backward's gradient identities assume unit norm,
///   which holds for [`quat_from_scaled_axis`](crate::mamba3::rotation::quat_from_scaled_axis)
///   outputs and products thereof).
/// - `init`    : optional carry `[batch, nheads, blocks, 4]` (identity if `None`).
/// - returns `(cum, final_carry)` — `[batch, sequence, nheads, blocks, 4]` and
///   `[batch, nheads, blocks, 4]`.
pub fn quat_cumprod_recalculated(
    q_bshj4: Tensor<5>,
    init: Option<Tensor<4>>,
) -> (Tensor<5>, Tensor<4>) {
    let [batch, sequence, nheads, blocks, _four] = q_bshj4.dims();
    let device = q_bshj4.device();

    let init_bhj4 = init.unwrap_or_else(|| {
        let w = Tensor::<4>::ones([batch, nheads, blocks, 1], &device);
        let xyz = Tensor::<4>::zeros([batch, nheads, blocks, 3], &device);
        Tensor::cat(vec![w, xyz], 3)
    });

    let cum_bshj4 = Tensor::<5>::from_primitive(<Dispatch as Mamba3QuatScanBackendExt>::quat_cumprod(
        q_bshj4.into_primitive(),
        init_bhj4.into_primitive(),
    ));

    let final_carry_bhj4 = cum_bshj4
        .clone()
        .narrow(1, sequence - 1, 1)
        .squeeze_dim::<4>(1);
    (cum_bshj4, final_carry_bhj4)
}
