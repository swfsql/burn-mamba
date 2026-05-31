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
//! goes through the rank-tagged [`F`] primitive wrapper, held in a
//! struct-of-arrays [`Quat`] (the four components as separate tensors) so the
//! Hamilton product is narrow/cat-free on the hot path. The same [`Quat`] helper
//! and [`quat_prefix_product_soa`] are reused by the recompute backward.

#![allow(non_snake_case)]

use crate::utils::fprim::F;
use burn::backend::tensor::Device;
use burn::backend::*;
use burn::backend::{Backend, Dispatch, FloatDType, backend_extension};
use burn::tensor::Tensor;

// ---------------------------------------------------------------------------
// Primitive quaternion algebra — struct-of-arrays (SoA)
// ---------------------------------------------------------------------------
//
// The scan's workhorse is the Hamilton product, run ~10× in the forward prefix
// product and again in the backward. Holding a quaternion as one packed
// `[…, 4]` tensor forces every product to `narrow` out the four components and
// `cat` them back — fusion-breaking ops (strided reads + a concat kernel) on the
// hottest path. Instead, [`Quat`] keeps the four components `(w, x, y, z)` as
// separate `[batch, sequence, nheads, blocks]` tensors threaded through the
// whole scan; the product is then pure (fusible) element-wise arithmetic with no
// narrow/cat. Packing to/from the `[…, 4]` layout happens once, at the
// boundaries ([`Quat::from_rank5`] / [`Quat::pack`]). This is also the layout a
// future custom kernel would use (w,x,y,z in registers).

/// A quaternion field in struct-of-arrays form: the four components
/// `(w, x, y, z)` as separate rank-4 `[batch, sequence, nheads, blocks]` tensors
/// (a leading seq-length of `1` broadcasts, e.g. the carry).
pub(crate) struct Quat<B: Backend> {
    /// Real part `w`.
    pub w: F<B, 4>,
    /// Imaginary `x`.
    pub x: F<B, 4>,
    /// Imaginary `y`.
    pub y: F<B, 4>,
    /// Imaginary `z`.
    pub z: F<B, 4>,
}

impl<B: Backend> Clone for Quat<B> {
    fn clone(&self) -> Self {
        Quat {
            w: self.w.clone(),
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<B: Backend> Quat<B> {
    /// Unpack a packed `[batch, sequence, nheads, blocks, 4]` tensor into SoA
    /// components (the only `narrow`s in the scan — done once at entry).
    pub fn from_rank5(q_bshj4: F<B, 5>) -> Self {
        let w = q_bshj4.clone().narrow(4, 0, 1).squeeze_dim::<4>(4);
        let x = q_bshj4.clone().narrow(4, 1, 1).squeeze_dim::<4>(4);
        let y = q_bshj4.clone().narrow(4, 2, 1).squeeze_dim::<4>(4);
        let z = q_bshj4.narrow(4, 3, 1).squeeze_dim::<4>(4);
        Quat { w, x, y, z }
    }

    /// Pack SoA components back into a `[batch, sequence, nheads, blocks, 4]`
    /// tensor (the only `cat` in the scan — done once at exit).
    pub fn pack(self) -> F<B, 5> {
        F::cat(
            vec![
                self.w.unsqueeze_dim::<5>(4),
                self.x.unsqueeze_dim::<5>(4),
                self.y.unsqueeze_dim::<5>(4),
                self.z.unsqueeze_dim::<5>(4),
            ],
            4,
        )
    }

    /// Identity quaternion `(1, 0, 0, 0)` of the given component shape.
    pub fn identity(shape: [usize; 4], device: &Device<B>, dtype: FloatDType) -> Self {
        let zero = F::<B, 4>::zeros(shape, device, dtype);
        Quat {
            w: F::<B, 4>::full(shape, 1.0, device, dtype),
            x: zero.clone(),
            y: zero.clone(),
            z: zero,
        }
    }

    /// Hamilton product `self ⊗ other` (pure element-wise; no narrow/cat).
    /// Broadcasts over the leading dims, so a `[b,1,h,j]` carry multiplies a
    /// `[b,s,h,j]` sequence.
    pub fn mul(self, other: Quat<B>) -> Quat<B> {
        let Quat {
            w: aw,
            x: ax,
            y: ay,
            z: az,
        } = self;
        let Quat {
            w: bw,
            x: bx,
            y: by,
            z: bz,
        } = other;
        Quat {
            w: aw.clone() * bw.clone()
                - ax.clone() * bx.clone()
                - ay.clone() * by.clone()
                - az.clone() * bz.clone(),
            x: aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
                - az.clone() * by.clone(),
            y: aw.clone() * by.clone() - ax.clone() * bz.clone()
                + ay.clone() * bw.clone()
                + az.clone() * bx.clone(),
            z: aw * bz + ax * by - ay * bx + az * bw,
        }
    }

    /// Quaternion conjugate `(w, −x, −y, −z)`. For a unit quaternion `q* = q⁻¹`.
    pub fn conj(self) -> Quat<B> {
        Quat {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Prepend `ident` along the sequence axis and re-narrow to `sequence` — the
    /// Hillis–Steile shift `shifted[t] = self[t-offset]` (identity before the
    /// start), with `offset = ident`'s sequence length.
    pub fn shift_prepend(self, ident: Quat<B>, sequence: usize) -> Quat<B> {
        let shift =
            |head: F<B, 4>, tail: F<B, 4>| F::cat(vec![head, tail], 1).narrow(1, 0, sequence);
        Quat {
            w: shift(ident.w, self.w),
            x: shift(ident.x, self.x),
            y: shift(ident.y, self.y),
            z: shift(ident.z, self.z),
        }
    }

    /// Inclusive **suffix**-sum along the sequence axis (`out[t] = Σ_{s≥t} in[s]`),
    /// per component — `flip → cumsum → flip`. Used by the backward's `S[t]`.
    pub fn reverse_cumsum_seq(self) -> Quat<B> {
        let rc = |t: F<B, 4>| t.flip(&[1]).cumsum(1).flip(&[1]);
        Quat {
            w: rc(self.w),
            x: rc(self.x),
            y: rc(self.y),
            z: rc(self.z),
        }
    }
}

/// Pure prefix product `P[t] = qₜ ⊗ qₜ₋₁ ⊗ ⋯ ⊗ q₀` (no carry) along the sequence
/// axis, via Hillis–Steele doubling — `O(log seq)` dependency depth, all on SoA
/// [`Quat`] components (no per-step narrow/cat).
///
/// The carry is folded in by the caller (one extra [`Quat::mul`]); keeping `P`
/// separate is what the recompute backward needs (`G[t] = P[t] ⊗ S[t]`).
pub(crate) fn quat_prefix_product_soa<B: Backend>(q: Quat<B>) -> Quat<B> {
    let [batch, sequence, nheads, blocks] = q.w.dims();
    let device = q.w.device();
    let dtype = q.w.dtype();

    let mut a = q;
    let mut offset = 1usize;
    while offset < sequence {
        let ident = Quat::<B>::identity([batch, offset, nheads, blocks], &device, dtype);
        let shifted = a.clone().shift_prepend(ident, sequence);
        a = a.mul(shifted); // recent (a) ⊗ older (shifted)
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
    fn quat_cumprod(q_bshj4: FloatTensor<Self>, init_bhj4: FloatTensor<Self>) -> FloatTensor<Self> {
        let q = Quat::from_rank5(F::<Self, 5>::new(q_bshj4));
        // Carry as a 1-long sequence [batch, 1, nheads, blocks, 4] → SoA, broadcasts over seq.
        let init = Quat::from_rank5(F::<Self, 4>::new(init_bhj4).unsqueeze_dim::<5>(1));

        let p = quat_prefix_product_soa::<Self>(q);
        let cum = p.mul(init); // cum[t] = Pₜ ⊗ init
        cum.pack().inner()
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

    let cum_bshj4 =
        Tensor::<5>::from_primitive(<Dispatch as Mamba3QuatScanBackendExt>::quat_cumprod(
            q_bshj4.into_primitive(),
            init_bhj4.into_primitive(),
        ));

    let final_carry_bhj4 = cum_bshj4
        .clone()
        .narrow(1, sequence - 1, 1)
        .squeeze_dim::<4>(1);
    (cum_bshj4, final_carry_bhj4)
}
