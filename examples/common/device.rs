//! Runtime dtype selection + record format for the examples.
//!
//! With the Dispatch architecture the backend is chosen at runtime by the
//! [`Device`], so the examples just use [`Device::default`]: it resolves to the
//! enabled `backend-*` feature (each enables the matching `burn/<backend>`),
//! honouring the `BURN_DEVICE` env override and a built-in priority list when
//! several are compiled in. [`configure_dtype`] optionally installs a
//! non-default dtype (used by `dev-f16` to switch the device to fp16/i32) —
//! backend defaults are otherwise left untouched.
//!
//! The on-disk record format is [`RecorderTy`]. Its `PrecisionSettings` selects
//! the precision the recorder stores tensors at (fp32 by default; fp16 when
//! `dev-f16` is enabled). Tensors are cast on save/load as needed, independent
//! of the runtime dtype installed on the device.

use burn::prelude::*;
use burn::record::NamedMpkFileRecorder;

#[cfg(feature = "dev-f16")]
pub use burn::record::HalfPrecisionSettings as RecorderPrecision;
#[cfg(not(feature = "dev-f16"))]
pub use burn::record::FullPrecisionSettings as RecorderPrecision;

/// On-disk record format for model and optimizer state.
pub type RecorderTy = NamedMpkFileRecorder<RecorderPrecision>;

/// The host-side scalar type matching the device's default float dtype.
///
/// Used when reading tensor values back to the host (`to_vec`/`into_data`) so
/// the element type matches the runtime dtype — fp16 under `dev-f16`, fp32
/// otherwise.
#[cfg(feature = "dev-f16")]
pub type FloatElement = burn::tensor::f16;
/// The host-side scalar type matching the device's default float dtype.
#[cfg(not(feature = "dev-f16"))]
pub type FloatElement = f32;

/// When `dev-f16` is enabled, install fp16 (and i32) as the device defaults.
///
/// Must be called before any tensor is created on `device`. No-op when the
/// feature is off — the backend's own dtype defaults apply.
pub fn configure_dtype(device: &mut Device) {
    #[cfg(feature = "dev-f16")]
    {
        use burn::tensor::{FloatDType, IntDType};
        device
            .configure((FloatDType::F16, IntDType::I32))
            .expect("Failed to install fp16/i32 device defaults");
    }
    #[cfg(not(feature = "dev-f16"))]
    {
        let _ = device;
    }
}
