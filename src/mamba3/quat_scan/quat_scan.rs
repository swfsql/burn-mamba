//! # Quaternion cumulative-product scan with a custom, memory-efficient backward
//!
//! The forward is the same Hillis–Steele parallel scan as
//! [`crate::mamba3::rotation::quat_cumprod`], but it goes through the
//! [`Mamba3QuatScanBackendExt`] trait. So an `Autodiff` backend can use a
//! custom backward that recomputes the scan instead of keeping its
//! intermediates (see [`super::backward`](crate::mamba3::quat_scan::backward)).
//! Plain backends use the default body of the trait, which runs the scan on
//! the primitives of `B`.
//!
//! The default body runs under a generic backend `B`, where the high-level
//! [`Tensor`](burn::tensor::Tensor) (pinned to `Dispatch`) is not available.
//! So the quaternion algebra uses the rank-tagged `F` primitive wrapper, held
//! in a struct-of-arrays `Quat` (the four components as separate tensors).
//! Thus the Hamilton product has no narrow/cat on the hot path. The recompute
//! backward uses the same `Quat` helper and `quat_prefix_product_soa`.

#![allow(non_snake_case)]

use burn_stack::utils::fprim::F;
use burn::backend::tensor::{Device, FloatTensor};
use burn::backend::{Backend, Dispatch, FloatDType, backend_extension};
use burn::tensor::Tensor;

// ---------------------------------------------------------------------------
// Primitive quaternion algebra — struct-of-arrays (SoA)
// ---------------------------------------------------------------------------
//
// The main op of the scan is the Hamilton product: ~10× in the forward prefix
// product, and again in the backward. If a quaternion is one packed `[…, 4]`
// tensor, every product must `narrow` out the four components and `cat` them
// back. Those ops break fusion (strided reads + a concat kernel) on the
// hottest path. Instead, [`Quat`] keeps the four components `(w, x, y, z)` as
// separate `[batch, sequence, nheads, blocks]` tensors through the whole scan.
// The product is then pure (fusible) element-wise arithmetic with no
// narrow/cat. The packing to/from the `[…, 4]` layout happens once, at the
// boundaries ([`Quat::from_rank5`] / [`Quat::pack`]).

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

    /// Prepend `ident` along the sequence axis and narrow again to `sequence`:
    /// the Hillis–Steele shift `shifted[t] = self[t-offset]` (identity before
    /// the start), where `offset` is the sequence length of `ident`.
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

/// Pure prefix product `P[t] = qₜ ⊗ qₜ₋₁ ⊗ ⋯ ⊗ q₀` (no carry) along the
/// sequence axis, by Hillis–Steele doubling: `O(log seq)` dependency depth, all
/// on SoA [`Quat`] components (no per-step narrow/cat).
///
/// The caller folds in the carry (one extra [`Quat::mul`]). The recompute
/// backward needs `P` alone (`G[t] = P[t] ⊗ S[t]`).
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
    // Every cubecl runtime (CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu, CPU) is
    // this one backend. The device of a tensor tells which runtime it uses.
    // The cfg mirrors burn's own `cube_backend`.
    Cube: cfg(any(
        feature = "backend-cpu",
        feature = "backend-cuda",
        feature = "backend-rocm",
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-wgpu",
        feature = "backend-webgpu"
    )),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait Mamba3QuatScanBackendExt: Backend {
    /// Cumulative quaternion product `cum[t] = qₜ ⊗ ⋯ ⊗ q₀ ⊗ init` along the
    /// sequence axis (newest on the left), returning only `cum`.
    ///
    /// The caller takes `final_carry = cum[:, −1]` (a thin autodiff slice) on
    /// the high-level `Tensor` (see [`quat_cumprod_recalculated`]).
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

// Per-backend impls use the default body of the trait. The custom autodiff
// backward is a separate `Autodiff<B>` impl in `super::backward`.
burn_stack::impl_backend_ext_for_burn_backends!(Mamba3QuatScanBackendExt);

// ---------------------------------------------------------------------------
// High-level wrapper (Dispatch-pinned `Tensor`)
// ---------------------------------------------------------------------------

/// Cumulative quaternion product with a custom, memory-efficient backward. It
/// replaces [`crate::mamba3::rotation::quat_cumprod`] for both quaternion
/// kinds.
///
/// It sends the scan through [`Mamba3QuatScanBackendExt`] (so `Autodiff` gets
/// the recompute backward), then takes `final_carry = cum[:, −1]` as a thin
/// autodiff slice. The gradient of that slice folds back into the gradient of
/// `cum` before the custom node runs, so the node needs only the one `cum`
/// output.
///
/// Mathematically identical to `quat_cumprod` (the tests assert it on values
/// and gradients). Only the memory profile of the backward is different.
///
/// # Shapes
/// - `q_bshj4` : `[batch, sequence, nheads, blocks, 4]`, per-step **unit**
///   quaternions. The gradient identities of the recompute backward assume unit
///   norm, which holds for the outputs of
///   [`quat_from_scaled_axis`](crate::mamba3::rotation::quat_from_scaled_axis)
///   and their products.
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
        Tensor::<5>::from_dispatch(<Dispatch as Mamba3QuatScanBackendExt>::quat_cumprod(
            q_bshj4.into_dispatch(),
            init_bhj4.into_dispatch(),
        ));

    let final_carry_bhj4 = cum_bshj4
        .clone()
        .narrow(1, sequence - 1, 1)
        .squeeze_dim::<4>(1);
    (cum_bshj4, final_carry_bhj4)
}
