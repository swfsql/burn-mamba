use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, BackendTypes};

#[cfg(feature = "dev-f16")]
mod ty {
    use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};
    use burn::tensor::DType;
    pub type FloatElement = burn::tensor::f16;
    pub const FLOAT_DTYPE: DType = DType::F16;
    pub type IntElement = i32; // used mostly for indexing
    pub const INT_DTYPE: DType = DType::I32; // used mostly for indexing
    pub type RecorderTy = NamedMpkFileRecorder<HalfPrecisionSettings>;
}
#[cfg(not(feature = "dev-f16"))]
mod ty {
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
    use burn::tensor::DType;
    pub type FloatElement = f32;
    pub const FLOAT_DTYPE: DType = burn::tensor::DType::F32;
    pub type IntElement = i32; // used mostly for indexing
    pub const INT_DTYPE: DType = burn::tensor::DType::I32; // used mostly for indexing
    pub type RecorderTy = NamedMpkFileRecorder<FullPrecisionSettings>;
}
pub use ty::*;

#[cfg(feature = "backend-ndarray")]
pub type MainBackend = burn::backend::NdArray<FloatElement, IntElement>;
#[cfg(feature = "backend-flex")]
pub type MainBackend = burn::backend::Flex;
#[cfg(feature = "backend-cpu")]
pub type MainBackend = burn::backend::Cpu<FloatElement, IntElement>;
#[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
pub type MainBackend = burn::backend::libtorch::LibTorch<FloatElement, IntElement>;
#[cfg(any(
    feature = "backend-wgpu",
    feature = "backend-metal",
    feature = "backend-vulkan"
))]
pub type MainBackend = burn::backend::wgpu::Wgpu<FloatElement, IntElement>;
#[cfg(feature = "backend-cuda")]
pub type MainBackend = burn::backend::Cuda<FloatElement, IntElement>;
#[cfg(feature = "backend-rocm")]
pub type MainBackend = burn::backend::Rocm<FloatElement, IntElement>;
#[cfg(feature = "backend-remote")]
pub type MainBackend = burn::backend::RemoteBackend<FloatElement, IntElement>;

pub trait MainDevice: Backend {
    fn main_device() -> <Self as BackendTypes>::Device {
        Default::default()
    }
    fn set_dtype(device: &<Self as BackendTypes>::Device) {
        burn::tensor::set_default_dtypes::<Self>(
            &device,
            FLOAT_DTYPE, // default float
            INT_DTYPE,   // default int
        )
        .unwrap();
    }
}

#[cfg(any(
    feature = "backend-ndarray",
    feature = "backend-flex",
    feature = "backend-cpu",
    feature = "backend-tch-cpu",
    feature = "backend-wgpu",
    feature = "backend-metal",
    feature = "backend-vulkan",
    feature = "backend-cuda",
    feature = "backend-rocm",
    feature = "backend-remote"
))]
impl MainDevice for MainBackend {}
#[cfg(all(feature = "backend-tch-gpu", not(target_os = "macos")))]
impl MainDevice for MainBackend {
    fn main_device() -> <Self as BackendTypes>::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    }
}
#[cfg(all(feature = "backend-tch-gpu", target_os = "macos"))]
impl MainDevice for MainBackend {
    fn main_device() -> <Self as BackendTypes>::Device {
        burn::backend::libtorch::LibTorchDevice::Mps
    }
}

pub type MainAutoBackend = burn::backend::Autodiff<MainBackend>;
impl MainDevice for MainAutoBackend {
    fn main_device() -> <Self as BackendTypes>::Device {
        <<Self as AutodiffBackend>::InnerBackend as MainDevice>::main_device()
    }
}

#[cfg(not(feature = "_dev-has-backend"))]
mod err {
    use super::*;
    std::compile_error!(
        "No dev backend selected. Please check burn-mamba/Cargo.toml for more info."
    );

    // pretend to fallback to ndarray (to avoid too many other unrelated errors)
    pub type MainBackend = burn::backend::NdArray<FloatElement, IntElement>;
    impl MainDevice for MainBackend {}
}
#[cfg(not(feature = "_dev-has-backend"))]
pub use err::*;
