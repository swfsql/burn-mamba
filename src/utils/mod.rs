//! # Shared utilities
//!
//! Building blocks reused across the Mamba families: custom activations and
//! norms (often fp16-stable variants Burn lacks), loss functions, the
//! `segsum` / `gqa` tensor helpers, the custom-backward plumbing
//! (`backend_macros` / `combined_grad` / `primitive`), runtime `sanity` guards,
//! LR `scheduler`s, and the per-dtype numerical constants below.

use burn::prelude::ToElement;
use burn::tensor::DType;

/// Macros emitting per-backend `BackendExt` impls + autodiff marker traits.
#[macro_use]
pub mod backend_macros;
/// Learnable `[CLS]`-style class tokens/latents spliced into the sequence.
pub mod class;
/// Flatten/unflatten `(y, final_state)` into one tracked tensor for the custom
/// backward.
pub mod combined_grad;
/// Rank-tagged `FloatTensor` primitive wrapper mirroring the `Tensor` method
/// API, used by the custom-backward gradient math.
pub(crate) mod fprim;
/// Virtual-layer → real-weight index scheduling shared by all families.
pub mod schedule;
/// Learning-rate schedulers (cosine-annealing + warmup, constant).
pub mod scheduler;
/// `max_abs_diff` + gradient-comparison macros used across the test suites.
#[cfg(test)]
pub mod test_helpers;

pub use class::{ClassLatent, ClassToken};
pub use schedule::{BidiSchedule, Schedule};
pub use scheduler::{ConstantLr, CosineAnnealingLr, Lr};

/// A small `dtype`-specific epsilon for safe division (`x / (y + eps)`),
/// returned as `f32`.
///
/// The value is chosen per float format as the geometric mean (average in
/// log10 space) of two reference magnitudes: a scaled function of the format's
/// minimum exponent and the format's machine epsilon.  This places `eps`
/// comfortably above the denormal/underflow floor while staying negligible
/// relative to typical activations, for each of f64/f32/f16/bf16.  The
/// resulting constants are noted inline.  `dtype` is the runtime float dtype of
/// the tensor being divided (e.g. `x.dtype()`).  Panics on non-float dtypes.
pub fn div_eps(dtype: DType) -> f32 {
    match dtype {
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
