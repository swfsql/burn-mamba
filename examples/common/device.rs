//! Runtime device + dtype selection for the examples.
//!
//! With the new Dispatch architecture, the backend is selected at runtime by
//! constructing the appropriate [`Device`]. [`select_device`] picks one based on
//! which `backend-*` cargo feature is enabled. [`configure_dtype`] optionally
//! installs a non-default dtype (used by `dev-f16` to switch the device to
//! fp16/i32) — backend defaults are otherwise left untouched.
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

/// Pick a [`Device`] from the enabled `backend-*` feature.
///
/// Exactly one backend feature is expected to be enabled. With the Dispatch
/// mechanism, more than one *can* be compiled in safely; this helper just picks
/// the first matching one in cfg order.
#[allow(unreachable_code)]
pub fn select_device() -> Device {
    #[cfg(feature = "backend-cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);
    
    #[cfg(feature = "backend-vulkan")]
    return Device::vulkan(burn::tensor::DeviceKind::DefaultDevice);

    #[cfg(feature = "backend-wgpu")]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
    
    #[cfg(feature = "backend-rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);

    #[cfg(feature = "backend-metal")]
    return Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    
    #[cfg(all(feature = "backend-tch-gpu", not(target_os = "macos")))]
    return Device::libtorch_cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(all(feature = "backend-tch-gpu", target_os = "macos"))]
    return Device::libtorch_mps();

    #[cfg(feature = "backend-tch-cpu")]
    return Device::libtorch();

    #[cfg(feature = "backend-cpu")]
    return Device::cpu();

    #[cfg(feature = "backend-flex")]
    return Device::flex();

    #[cfg(feature = "backend-ndarray")]
    return Device::ndarray();

    panic!(
        "No backend feature enabled. Enable one of: \
         backend-flex, backend-cpu, backend-cuda, backend-rocm, \
         backend-metal, backend-vulkan, backend-wgpu, \
         backend-tch-cpu, backend-tch-gpu, backend-ndarray."
    );
}

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
