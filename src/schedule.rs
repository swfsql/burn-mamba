use burn::prelude::*;

#[derive(Module, Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Schedule {
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion.
    ///
    /// Example: virtual len = 8, real len = 3:  
    /// (0→0, 1→1, 2→2), (3→0, 4→1, 5→2), (6→0, 7→1, ...)
    #[default]
    Cyclic,
    /// Fills virtual positions by stretching the real schedule.
    ///
    /// Example: virtual len = 8, real len = 3:
    /// (0→0, 1→0, 2→0), (3→1, 4→1, 5→1), (6→2, 7→2, ...)
    Stretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// Example: virtual len = 8, real len = 3, custom = [0, 1, 2, 2, 1, 0, 0, 0]:
    /// (0→0, 1→1, 2→2, 3→2, 4→1, 5→0, 6→0, 7→0, ...)
    Custom(Vec<usize>),
}

impl Schedule {
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        match self {
            Schedule::Cyclic => virtual_idx % real_len,
            Schedule::Stretched => (virtual_idx * real_len) / virtual_len,
            Schedule::Custom(map) => *map.get(virtual_idx).unwrap(),
        }
    }
}
