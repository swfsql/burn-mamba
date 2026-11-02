# Portable Burn tensor ops only, no custom kernels

burn-mamba ports the official CUDA/Triton/Tilelang kernels to the standard
tensor ops of Burn. So one code path runs on every Burn backend (CPU, WGPU,
CUDA, Metal, LibTorch, …). We accept that it is slower than the fused official
kernels.

## Consequences

We permit a custom *backward*: it recomputes the forward with the same
tensor ops, so it is not a kernel.
