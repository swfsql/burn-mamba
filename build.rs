//! Sets the `cube_backend` cfg.
//!
//! Every cubecl runtime (CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu, CPU) is one
//! backend in Burn. The device of a tensor tells which runtime it uses. So
//! each `#[backend_extension]` site of this crate gates its `Cube` arm with
//! one cfg: `cube_backend`. This script sets it when a `backend-*` feature of
//! such a runtime is on, as burn's own `cube_backend` does. A new cubecl
//! backend feature goes into this list.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(cube_backend)");

    let cube = cfg!(feature = "backend-cpu")
        || cfg!(feature = "backend-cuda")
        || cfg!(feature = "backend-rocm")
        || cfg!(feature = "backend-metal")
        || cfg!(feature = "backend-vulkan")
        || cfg!(feature = "backend-wgpu")
        || cfg!(feature = "backend-webgpu");
    if cube {
        println!("cargo::rustc-cfg=cube_backend");
    }
}
