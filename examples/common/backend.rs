use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "dev-f16")]
pub type Element = burn::tensor::f16;
#[cfg(not(feature = "dev-f16"))]
pub type Element = f32;

#[cfg(feature = "dev-ndarray")]
pub type MainBackend = burn::backend::NdArray<Element, i32>;
#[cfg(any(feature = "dev-tch-cpu", feature = "dev-tch-gpu"))]
pub type MainBackend = burn::backend::libtorch::LibTorch<Element, i32>;
#[cfg(any(feature = "dev-wgpu", feature = "dev-metal", feature = "dev-vulkan"))]
pub type MainBackend = burn::backend::wgpu::Wgpu<Element, i32>;
#[cfg(feature = "dev-cuda")]
pub type MainBackend = burn::backend::Cuda<Element, i32>;
#[cfg(feature = "dev-rocm")]
pub type MainBackend = burn::backend::Rocm<Element, i32>;
#[cfg(feature = "dev-remote")]
pub type MainBackend = burn::backend::RemoteBackend<Element, i32>;

pub trait MainDevice: Backend {
    fn main_device() -> <Self as Backend>::Device {
        Default::default()
    }
}

#[cfg(any(
    feature = "dev-ndarray",
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
    pub type MainBackend = burn::backend::NdArray<Element, i32>;
    impl MainDevice for MainBackend {}
}
#[cfg(not(feature = "_dev-has-backend"))]
pub use err::*;
