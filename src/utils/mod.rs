//! # Shared utilities
//!
//! Building blocks reused across the Mamba families: custom activations and
//! norms (often fp16-stable variants Burn lacks), loss functions, the
//! `segsum` / `gqa` tensor helpers, the custom-backward plumbing
//! (`backend_macros` / `combined_grad` / `primitive`), runtime `sanity` guards,
//! LR `scheduler`s, and the per-dtype numerical constants below.

use ElementConversion;
use burn::prelude::*;
use burn::tensor::{DType, Element};
use burn::backend::Backend;

/// Macros emitting per-backend `BackendExt` impls + autodiff marker traits.
#[macro_use]
pub mod backend_macros;
/// Flatten/unflatten `(y, final_state)` into one tracked tensor for the custom
/// backward.
pub mod combined_grad;
/// Group→head expansion of B/C (GQA-style sharing).
pub mod gqa;
/// Numerically-stable log-sigmoid (fp16-aware).
pub mod log_sigmoid;
/// Loss functions (binary cross-entropy, cross-entropy, mean squared error).
pub mod loss;
/// `FloatTensor` primitive ↔ `Tensor<D>` conversion helper.
pub(crate) mod primitive;
/// Root-mean-square normalisation (last-dim, fp16-safe); also the Mamba-3
/// QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (Mamba-2 output norm).
pub mod rms_norm_gated;
/// Optional `NaN`/`Inf` guards gated by [`crate::DENY_NAN`] / [`crate::DENY_INF`].
pub mod sanity;
/// Learning-rate schedulers (cosine-annealing + warmup, constant).
pub mod scheduler;
/// Stable segment-sum → 1-semiseparable mask (log-space prefix-sum differences).
pub mod segsum;
/// SiLU activation (fp16-aware).
pub mod silu;
/// Softplus activation (fp16-aware).
pub mod softplus;
/// Typed-array variant of `split_with_sizes` for clean destructuring.
pub mod split;
/// `max_abs_diff` + gradient-comparison macros used across the test suites.
#[cfg(test)]
pub mod test_helpers;

/// The largest finite value representable by the backend's float element type.
///
/// Used as a saturating upper bound (e.g. clamping) that works uniformly across
/// f64/f32/f16/bf16 without overflowing the narrower formats.  Panics on
/// non-float element types.
pub fn stable_max() -> B::FloatElem {
    match <B::FloatElem as Element>::dtype() {
        DType::F64 => f64::MAX.elem(),
        DType::F32 | DType::Flex32 => f32::MAX.elem(),
        DType::F16 => burn::tensor::f16::MAX.elem(),
        DType::BF16 => burn::tensor::bf16::MAX.elem(),
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8 => {
            unreachable!()
        }
        DType::Bool(_) => {
            unreachable!()
        }
        DType::QFloat(_) => {
            unimplemented!()
        }
    }
}

/// A small per-dtype epsilon for safe division (`x / (y + eps)`), returned as
/// `f32`.
///
/// The value is chosen per float format as the geometric mean (average in
/// log10 space) of two reference magnitudes: a scaled function of the format's
/// minimum exponent and the format's machine epsilon.  This places `eps`
/// comfortably above the denormal/underflow floor while staying negligible
/// relative to typical activations, for each of f64/f32/f16/bf16.  The
/// resulting constants are noted inline.  Panics on non-float element types.
pub fn div_eps_f32() -> f32 {
    match <B::FloatElem as Element>::dtype() {
        // 4.0693917e-16
        DType::F64 => {
            let raw_exp = -(-f64::MIN_EXP as f32 * 2.3f32).powf(0.35f32);
            let eps_exp = (f64::EPSILON as f32).log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 8.1584695e-8
        DType::F32 | DType::Flex32 => {
            let raw_exp = -(-f32::MIN_EXP as f32 * 2.3f32).powf(0.35f32);
            let eps_exp = f32::EPSILON.log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 7.1209995e-4
        DType::F16 => {
            let raw_exp = -(-burn::tensor::f16::MIN_EXP.to_f32() * 2.3f32).powf(0.35f32);
            let eps_exp = burn::tensor::f16::EPSILON.to_f32().log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 2.0885676e-5
        DType::BF16 => {
            let raw_exp = -(-burn::tensor::bf16::MIN_EXP.to_f32() * 2.3f32).powf(0.35f32);
            let eps_exp = burn::tensor::bf16::EPSILON.to_f32().log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8
        | DType::Bool(_) => {
            unreachable!()
        }
        DType::QFloat(_) => {
            unimplemented!()
        }
    }
}

/// [`div_eps_f32`] converted to the backend's native float element type.
pub fn div_eps() -> B::FloatElem {
    div_eps_f32().elem()
}
