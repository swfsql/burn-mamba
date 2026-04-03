use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

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

#[cfg(feature = "dev-ndarray")]
pub type MainBackend = burn::backend::NdArray<FloatElement, IntElement>;
#[cfg(feature = "dev-flex")]
pub type MainBackend = burn_flex::Flex;
#[cfg(feature = "dev-cpu")]
pub type MainBackend = burn::backend::Cpu<FloatElement, IntElement>;
#[cfg(any(feature = "dev-tch-cpu", feature = "dev-tch-gpu"))]
pub type MainBackend = burn::backend::libtorch::LibTorch<FloatElement, IntElement>;
#[cfg(any(feature = "dev-wgpu", feature = "dev-metal", feature = "dev-vulkan"))]
pub type MainBackend = burn::backend::wgpu::Wgpu<FloatElement, IntElement>;
#[cfg(feature = "dev-cuda")]
pub type MainBackend = burn::backend::Cuda<FloatElement, IntElement>;
#[cfg(feature = "dev-rocm")]
pub type MainBackend = burn::backend::Rocm<FloatElement, IntElement>;
#[cfg(feature = "dev-remote")]
pub type MainBackend = burn::backend::RemoteBackend<FloatElement, IntElement>;

pub trait MainDevice: Backend {
    fn main_device() -> <Self as Backend>::Device {
        Default::default()
    }
    fn set_dtype(device: &<Self as Backend>::Device) {
        burn::tensor::set_default_dtypes::<Self>(
            &device,
            FLOAT_DTYPE, // default float
            INT_DTYPE,   // default int
        )
        .unwrap();
    }
}

#[cfg(any(
    feature = "dev-ndarray",
    feature = "dev-flex",
    feature = "dev-cpu",
    feature = "dev-tch-cpu",
    feature = "dev-wgpu",
    feature = "dev-metal",
    feature = "dev-vulkan",
    feature = "dev-cuda",
    feature = "dev-rocm",
    feature = "dev-remote"
))]
impl MainDevice for MainBackend {}
#[cfg(all(feature = "dev-tch-gpu", not(target_os = "macos")))]
impl MainDevice for MainBackend {
    fn main_device() -> <Self as Backend>::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    }
}
#[cfg(all(feature = "dev-tch-gpu", target_os = "macos"))]
impl MainDevice for MainBackend {
    fn main_device() -> <Self as Backend>::Device {
        burn::backend::libtorch::LibTorchDevice::Mps
    }
}

pub type MainAutoBackend = burn::backend::Autodiff<MainBackend>;
impl MainDevice for MainAutoBackend {
    fn main_device() -> <Self as Backend>::Device {
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
