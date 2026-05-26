use crate::mamba3::Mamba3DoubleSsdPath;
use crate::mamba3::Mamba3SingleSsdPath;

/// Algorithm selection for the double/single-pass SSD.
///
/// This selects the chunkwise SSD algorithm style.
///
/// The cache selection infers whether Double-SSD or Single-SSD is used.  
/// If none is specified, the cache defaults to [`crate::mamba3::cache::Mamba3Caches::SingleSsd`],
/// [`Self::SerialRecalculated`] ssd-path with the `chunk_len` unset
/// (which fallbacks to [`crate::mamba3::single_ssd::ssd::ssd_path::Mamba3SingleSsdPath::optimal_default`]).
#[derive(Debug, Clone)]
pub enum Mamba3SsdPath {
    /// Minimal/segsum SSD.
    ///
    /// This algorithm mostly uses batched matmuls. For the backward operation, this relies on autodiff.  
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_minimal`]/[`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_minimal`] for more info.
    ///
    /// For training, you may prefer using [`Self::SerialRecalculated`] instead.
    Minimal(Option<usize>),

    /// (Hybrid) Serial SSD.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// See See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial`]/[`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial`] for more info.  
    /// For the backward operation, this relies on autodiff.  
    /// For a custom backwards that saves memory, see [`Self::SerialRecalculated`].
    Serial(Option<usize>),

    /// (Hybrid) Serial SSD that triggers recalculations for the backward pass.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// See See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial_recalculated`]/[`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial_recalculated`] for more info.  
    /// Contains a custom backward operation that saves memory.  
    /// For an autodiff backwards, see [`Self::Serial`].
    SerialRecalculated(Option<usize>),
}

impl Default for Mamba3SsdPath {
    fn default() -> Mamba3SsdPath {
        // The SSD Path defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba3SsdPath::SerialRecalculated(None)
    }
}

impl From<Mamba3DoubleSsdPath> for Mamba3SsdPath {
    fn from(path: Mamba3DoubleSsdPath) -> Self {
        match path {
            Mamba3DoubleSsdPath::Minimal(chunk_len) => Mamba3SsdPath::Minimal(chunk_len),
            Mamba3DoubleSsdPath::Serial(chunk_len) => Mamba3SsdPath::Serial(chunk_len),
            Mamba3DoubleSsdPath::SerialRecalculated(chunk_len) => {
                Mamba3SsdPath::SerialRecalculated(chunk_len)
            }
        }
    }
}

impl From<Mamba3SingleSsdPath> for Mamba3SsdPath {
    fn from(path: Mamba3SingleSsdPath) -> Self {
        match path {
            Mamba3SingleSsdPath::Minimal(chunk_len) => Mamba3SsdPath::Minimal(chunk_len),
            Mamba3SingleSsdPath::Serial(chunk_len) => Mamba3SsdPath::Serial(chunk_len),
            Mamba3SingleSsdPath::SerialRecalculated(chunk_len) => {
                Mamba3SsdPath::SerialRecalculated(chunk_len)
            }
        }
    }
}
