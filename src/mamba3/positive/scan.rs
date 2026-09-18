//! The log-semiring prefix product every positive member runs on.
//!
//! An element is a 2×2 nonnegative matrix in log coordinates, one per
//! `[batch, position, head]`. Two shapes of it exist:
//!
//! - [`Mobius`](scan::Mobius) — all four entries, acting **projectively** on `(ℓ, 0)`: the
//!   read is `σ₀ − σ₁`, so the product may be shifted by any constant and is,
//!   after every combine, to keep its entries bounded;
//! - [`Affine`](scan::Affine) — the lower row fixed at `(−∞, 0)`, so only `(a, b)` is carried
//!   and the read `lse(a + c₀, b)` is absolute: no shift is allowed, and none
//!   is needed.
//!
//! [`prefix`](scan::prefix) composes them by Hillis–Steele doubling, the same schedule as the
//! quaternion scan (`rotation::quat_cumprod`): `⌈log₂ len⌉` full-width
//! combines. [`fold`](scan::fold) is the sequential left fold, kept as the reference the
//! tests hold [`prefix`](scan::prefix) to.

use super::LOG_ZERO;
use burn::prelude::*;

/// `ln(eᵃ + eᵇ)`, elementwise.
///
/// Written as `hi + ln(1 + e^(lo − hi))` with `hi`/`lo` *selected* rather than
/// `max`/`abs`ed: the result is exact to the larger operand when the other is
/// [`LOG_ZERO`] (an `a + softplus(b − a)` form would round `b` through
/// `b − LOG_ZERO`), and its gradient is `σ(a − b)` on `a` even at a tie, where
/// a `max`-based form hands the whole of it to one side.
pub fn lse(a: Tensor<3>, b: Tensor<3>) -> Tensor<3> {
    let a_wins = a.clone().greater_equal(b.clone());
    let hi = b.clone().mask_where(a_wins.clone(), a.clone());
    let lo = a.mask_where(a_wins, b);
    hi.clone() + (lo - hi).exp().log1p()
}

/// An associative element over `[batch, len, nheads]` tensors.
pub trait Element: Sized + Clone {
    /// `later ∘ earlier` — apply `earlier` first.
    fn compose(later: Self, earlier: Self) -> Self;
    /// The identity, shaped `[batch, len, nheads]`.
    fn identity(batch: usize, len: usize, nheads: usize, device: &Device) -> Self;
    /// `[batch, len, nheads]`.
    fn dims(&self) -> [usize; 3];
    /// The device the element's tensors live on.
    fn device(&self) -> Device;
    /// A run of positions along axis 1.
    fn narrow(self, start: usize, len: usize) -> Self;
    /// Concatenate along axis 1.
    fn cat(parts: Vec<Self>) -> Self;
}

/// A projective 2×2 element: `σ ↦ M ⊗ σ` read as `σ₀ − σ₁`.
#[derive(Clone, Debug)]
pub struct Mobius {
    /// `ln M₀₀`, `[batch, len, nheads]`.
    pub m00: Tensor<3>,
    /// `ln M₀₁`.
    pub m01: Tensor<3>,
    /// `ln M₁₀`.
    pub m10: Tensor<3>,
    /// `ln M₁₁`.
    pub m11: Tensor<3>,
}

impl Mobius {
    /// The log-ratio after this (prefix) product, from `ℓ₀` per head:
    /// `lse(M₀₀ + ℓ₀, M₀₁) − lse(M₁₀ + ℓ₀, M₁₁)`.
    ///
    /// # Shapes
    /// - `carry_bh` : `[batch, nheads]`
    /// - out        : `[batch, len, nheads]`
    pub fn apply(self, carry_bh: Tensor<2>) -> Tensor<3> {
        let carry_b1h = carry_bh.unsqueeze_dim::<3>(1);
        lse(self.m00 + carry_b1h.clone(), self.m01) - lse(self.m10 + carry_b1h, self.m11)
    }
}

impl Element for Mobius {
    fn compose(later: Self, earlier: Self) -> Self {
        let (x, y) = (later, earlier);
        let m00 = lse(x.m00.clone() + y.m00.clone(), x.m01.clone() + y.m10.clone());
        let m01 = lse(x.m00.clone() + y.m01.clone(), x.m01 + y.m11.clone());
        let m10 = lse(x.m10.clone() + y.m00, x.m11.clone() + y.m10);
        let m11 = lse(x.m10 + y.m01, x.m11 + y.m11);
        // Projective: the read cancels any common shift, in value and in
        // gradient, so the largest entry is taken out (detached — its gradient
        // would cancel anyway) and the structural zeros re-floored.
        let shift = m00
            .clone()
            .max_pair(m01.clone())
            .max_pair(m10.clone())
            .max_pair(m11.clone())
            .detach();
        let floor = |t: Tensor<3>| (t - shift.clone()).clamp_min(LOG_ZERO);
        Mobius {
            m00: floor(m00),
            m01: floor(m01),
            m10: floor(m10),
            m11: floor(m11),
        }
    }

    fn identity(batch: usize, len: usize, nheads: usize, device: &Device) -> Self {
        let zero = Tensor::zeros([batch, len, nheads], device);
        let none = Tensor::full([batch, len, nheads], LOG_ZERO, device);
        Mobius {
            m00: zero.clone(),
            m01: none.clone(),
            m10: none,
            m11: zero,
        }
    }

    fn dims(&self) -> [usize; 3] {
        self.m00.dims()
    }

    fn device(&self) -> Device {
        self.m00.device()
    }

    fn narrow(self, start: usize, len: usize) -> Self {
        Mobius {
            m00: self.m00.narrow(1, start, len),
            m01: self.m01.narrow(1, start, len),
            m10: self.m10.narrow(1, start, len),
            m11: self.m11.narrow(1, start, len),
        }
    }

    fn cat(parts: Vec<Self>) -> Self {
        let mut m = [vec![], vec![], vec![], vec![]];
        for p in parts {
            m[0].push(p.m00);
            m[1].push(p.m01);
            m[2].push(p.m10);
            m[3].push(p.m11);
        }
        let [m00, m01, m10, m11] = m.map(|v| Tensor::cat(v, 1));
        Mobius { m00, m01, m10, m11 }
    }
}

/// An affine element `c ↦ lse(c + a, b)` — the matrix `[[a, b], [−∞, 0]]`.
#[derive(Clone, Debug)]
pub struct Affine {
    /// The log-gain, `[batch, len, nheads]` (any sign).
    pub a: Tensor<3>,
    /// The log-offset.
    pub b: Tensor<3>,
}

impl Affine {
    /// The register after this (prefix) map, from `c₀` per head:
    /// `lse(a + c₀, b)`.
    ///
    /// # Shapes
    /// - `carry_bh` : `[batch, nheads]`
    /// - out        : `[batch, len, nheads]`
    pub fn apply(self, carry_bh: Tensor<2>) -> Tensor<3> {
        lse(self.a + carry_bh.unsqueeze_dim::<3>(1), self.b)
    }
}

impl Element for Affine {
    fn compose(later: Self, earlier: Self) -> Self {
        // lse(lse(c + a₁, b₁) + a₂, b₂) = lse(c + (a₁ + a₂), lse(b₁ + a₂, b₂)).
        Affine {
            b: lse(earlier.b + later.a.clone(), later.b).clamp_min(LOG_ZERO),
            a: (earlier.a + later.a).clamp_min(LOG_ZERO),
        }
    }

    fn identity(batch: usize, len: usize, nheads: usize, device: &Device) -> Self {
        Affine {
            a: Tensor::zeros([batch, len, nheads], device),
            b: Tensor::full([batch, len, nheads], LOG_ZERO, device),
        }
    }

    fn dims(&self) -> [usize; 3] {
        self.a.dims()
    }

    fn device(&self) -> Device {
        self.a.device()
    }

    fn narrow(self, start: usize, len: usize) -> Self {
        Affine {
            a: self.a.narrow(1, start, len),
            b: self.b.narrow(1, start, len),
        }
    }

    fn cat(parts: Vec<Self>) -> Self {
        let (a, b) = parts.into_iter().map(|p| (p.a, p.b)).unzip();
        Affine {
            a: Tensor::cat(a, 1),
            b: Tensor::cat(b, 1),
        }
    }
}

/// Inclusive prefix products `Pₜ = eₜ ∘ ⋯ ∘ e₀` along axis 1, by Hillis–Steele
/// doubling.
///
/// Invariant after the round at `offset`: position `t` holds the product of
/// the window `[max(t − 2·offset + 1, 0), t]`, the positions before the axis
/// being the identity. `⌈log₂ len⌉` rounds cover every window.
pub fn prefix<E: Element>(elements: E) -> E {
    let [batch, len, nheads] = elements.dims();
    let mut acc = elements;
    let device = acc.device();
    let mut offset = 1usize;
    while offset < len {
        let shifted = E::cat(vec![
            E::identity(batch, offset, nheads, &device),
            acc.clone().narrow(0, len - offset),
        ]);
        acc = E::compose(acc, shifted);
        offset *= 2;
    }
    acc
}

/// The same prefix products by a sequential left fold — one combine per
/// position. The reference [`prefix`] is tested against.
pub fn fold<E: Element>(elements: E) -> E {
    let [_batch, len, _nheads] = elements.dims();
    let mut out = vec![elements.clone().narrow(0, 1)];
    for t in 1..len {
        let prev = out[t - 1].clone();
        out.push(E::compose(elements.clone().narrow(t, 1), prev));
    }
    E::cat(out)
}

