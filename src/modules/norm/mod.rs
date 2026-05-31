/// Root-mean-square normalisation (last-dim, fp16-safe); also the Mamba-3
/// QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (Mamba-2 output norm).
pub mod rms_norm_gated;
