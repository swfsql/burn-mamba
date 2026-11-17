//! # The SSD path (Mamba-2 and Mamba-3)
//!
//! [`SsdPath`](crate::ssd_path::SsdPath) selects the method that computes the
//! chunkwise SSD scan. All SSD paths give the same values and gradients. The
//! scan options of a block hold an SSD path and a chunk length:
//!
//! - [`Mamba2ScanOptions`](crate::mamba2::scan_options::Mamba2ScanOptions) for
//!   Mamba-2.
//! - [`Mamba3ScanOptions`](crate::mamba3::scan_options::Mamba3ScanOptions) for
//!   Mamba-3, on each pathway.
//!
//! The `run` of each input bundle reads only the SSD path. The chunk length is
//! already in the shapes of its tensors.

/// The method that computes the chunkwise SSD scan.
///
/// The default is [`Self::SerialRecalculated`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SsdPath {
    /// Minimal SSD: mostly batched matmuls and a `segsum` mask, with the
    /// autodiff backward.
    ///
    /// For training, prefer [`Self::SerialRecalculated`]. This is the only path
    /// that supports a learnable initial state.
    ///
    /// See these functions:
    ///
    /// - Mamba-2:
    ///   [`Mamba2SsdInput::ssd_minimal`](crate::mamba2::ssd::Mamba2SsdInput::ssd_minimal).
    ///   It is based on `/mamba_ssm/modules/ssd_minimal.py` in the
    ///   `state-spaces/mamba` GitHub reference.
    /// - Mamba-3:
    ///   [`Mamba3DoubleSsdInput::double_ssd_minimal`](crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_minimal)
    ///   and
    ///   [`Mamba3SingleSsdInput::single_ssd_minimal`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_minimal).
    Minimal,

    /// (Hybrid) serial SSD: a serial loop over the chunks plus batched
    /// matmuls, with the autodiff backward.
    ///
    /// For a memory-saving custom backward, see [`Self::SerialRecalculated`].
    ///
    /// See these functions:
    ///
    /// - Mamba-2:
    ///   [`Mamba2SsdInput::ssd_serial`](crate::mamba2::ssd::Mamba2SsdInput::ssd_serial).
    ///   It is based on 5 kernels under `/mamba_ssm/ops/triton/` in the
    ///   `state-spaces/mamba` GitHub reference:
    ///   - `ssd_chunk_state.py` (K1, K3).
    ///   - `ssd_bmm.py` (K2).
    ///   - `ssd_state_passing.py` (K4).
    ///   - `ssd_chunk_scan.py` (K5).
    /// - Mamba-3:
    ///   [`Mamba3DoubleSsdInput::double_ssd_serial`](crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial)
    ///   and
    ///   [`Mamba3SingleSsdInput::single_ssd_serial`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial).
    Serial,

    /// (Hybrid) serial SSD with a custom, memory-efficient backward. The
    /// backward recomputes the forward intermediates instead of storing them.
    ///
    /// For a plain autodiff backward, see [`Self::Serial`].
    ///
    /// See these functions:
    ///
    /// - Mamba-2:
    ///   [`Mamba2SsdInput::ssd_serial_recalculated`](crate::mamba2::ssd::Mamba2SsdInput::ssd_serial_recalculated).
    ///   It calls the [standard SSD scan](crate::standard_ssd). The gradients
    ///   are those of the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py`
    ///   in the `state-spaces/mamba` GitHub reference. The order of the math is
    ///   that of the scan, and not that of the kernel (ADR-0010).
    /// - Mamba-3:
    ///   [`Mamba3DoubleSsdInput::double_ssd_serial_recalculated`](crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial_recalculated)
    ///   and
    ///   [`Mamba3SingleSsdInput::single_ssd_serial_recalculated`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial_recalculated).
    #[default]
    SerialRecalculated,
}
