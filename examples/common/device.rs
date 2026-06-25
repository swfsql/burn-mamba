//! Runtime dtype selection for the examples.
//!
//! With the Dispatch architecture the backend is chosen at runtime by the
//! [`Device`], so the examples just use [`Device::default`]: it resolves to the
//! enabled `backend-*` feature (each enables the matching `burn/<backend>`),
//! honouring the `BURN_DEVICE` env override and a built-in priority list when
//! several are compiled in. [`configure_dtype`] optionally installs a
//! non-default dtype (used by `dev-f16` to switch the device to fp16/i32) —
//! backend defaults are otherwise left untouched.
//!
//! Model and optimizer state is persisted with the burnpack
//! [`store`](burn::store) format. The on-disk dtype follows whatever dtype the
//! module currently holds (fp16 under `dev-f16`, fp32 otherwise), so there is no
//! separate recorder precision to configure.

use burn::prelude::*;

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
