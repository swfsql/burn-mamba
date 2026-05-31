//! # Virtual-layer → real-weight scheduling
//!
//! A `{Model}Layers` stack can run `n_virtual_layers` logical passes over only
//! `n_real_layers` weight sets (e.g. 48 logical from 12 real); each virtual
//! layer keeps its own cache but shares parameters.  A [`Schedule`] maps a
//! virtual layer index to the real weight index to use.
//!
//! For **bidirectional** stacks, [`BidiSchedule`] additionally interleaves the
//! two directions: even virtual indices run the straight (→) pass and odd
//! indices run the reverse (←) pass.
//!
//! Each variant is documented with a worked virtual→real mapping example.

/// How a unidirectional layer stack maps virtual layer indices to real
/// (weight-bearing) layer indices.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Schedule {
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:  
    ///   `  →    →    →      →    →    →      →    →       `  
    ///   `(0⇒0, 1⇒1, 2⇒2), (3⇒0, 4⇒1, 5⇒2), (6⇒0, 7⇒1, ...)`
    #[default]
    Cyclic,
    /// Fills virtual positions by stretching the real schedule.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:  
    ///   `  →    →    →      →    →    →      →    →       `  
    ///   `(0⇒0, 1⇒0, 2⇒0), (3⇒1, 4⇒1, 5⇒1), (6⇒2, 7⇒2, ...)`
    Stretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3, custom = `[0, 1, 2, 2, 1, 0, 0, 0]`:  
    ///   `  →    →    →    →    →    →    →    →       `  
    ///   `(0⇒0, 1⇒1, 2⇒2, 3⇒2, 4⇒1, 5⇒0, 6⇒0, 7⇒0, ...)`
    Custom(Vec<usize>),
}

impl Schedule {
    /// Map `virtual_idx` (in `0..virtual_len`) to a real layer index in
    /// `0..real_len` according to this schedule.
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        match self {
            Schedule::Cyclic => virtual_idx % real_len,
            Schedule::Stretched => (virtual_idx * real_len) / virtual_len,
            Schedule::Custom(map) => *map.get(virtual_idx).unwrap(),
        }
    }
}

/// How a bidirectional layer stack maps virtual layer indices to real layer
/// indices, interleaving the straight (→, even indices) and reverse (←, odd
/// indices) directions.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BidiSchedule {
    /// Use even virtual positions for straight-direction (→), and odd virtual positions for
    /// reverse-direction (←), wrapping around for each schedule.
    //
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←        →    ←      →    ←        →    ←          `  
    ///   `[(0⇒0, 1⇒1), (2⇒2, 3⇒3)], [(4⇒0, 5⇒1), (6⇒2, 7⇒3)], [(8⇒0, 9⇒1), (...)]`
    #[default]
    StridedCyclic,
    /// Use even virtual positions for straight-direction (→), and odd virtual positions for
    /// reverse-direction (←), stretching for each schedule.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←      →    ←        →    ←      →    ←          `  
    ///   `[(0⇒0, 1⇒1), (2⇒0, 3⇒1), (4⇒0, 5⇒1)], [(6⇒2, 7⇒3), (8⇒2, 9⇒3), (...)]`
    StridedStretched,
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion,
    /// replicating between the straight (→) and reverse (←) directions.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←      →    ←      →    ←        →    ←          `  
    ///   `[(0⇒0, 1⇒0), (2⇒1, 3⇒1), (4⇒2, 5⇒2), (6⇒3, 7⇒3)], [(8⇒0, 9⇒0), (...)]`
    SymmetricCyclic,
    /// Fills virtual positions by stretching the real schedule, replicating between
    /// the straight (→) and reverse (←) directions.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←       →    ←               →    ←        →    ←   `  
    ///   `[(0⇒0, 1⇒0), (2⇒0, 3⇒0)],[(4⇒1, 5⇒1), (...)], [(6⇒2, 7⇒2)], [(8⇒3, 9⇒3)]`
    SymmetricStretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4, custom = `[0, 1, 2, 2, 1, 0, 0, 0, 3, 2]`:  
    ///   `   →    ←        →    ←        →    ←        →    ←        →    ←            `  
    ///   `[(0⇒0, 1⇒1)], [(2⇒2, 3⇒2)], [(4⇒1, 5⇒0)], [(6⇒0, 7⇒0)], [(8⇒3, 9⇒2)], [(...)]`
    Custom(Vec<usize>),
}

impl BidiSchedule {
    /// Map `virtual_idx` (in `0..virtual_len`) to a real layer index in
    /// `0..real_len`.  Even/odd `virtual_idx` selects the straight/reverse
    /// direction; the outer index `virtual_idx / 2` is what the schedule cycles
    /// or stretches over.
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        let virtual_outer_idx = virtual_idx / 2;
        let virtual_outer_len = virtual_len / 2;
        match self {
            BidiSchedule::StridedCyclic => {
                let odd_len = real_len / 2;
                let even_len = odd_len + real_len % 2;
                let is_even = virtual_idx.is_multiple_of(2);
                if is_even {
                    (virtual_outer_idx % even_len) * 2
                } else {
                    (virtual_outer_idx % odd_len) * 2 + 1
                }
            }
            BidiSchedule::StridedStretched => {
                let odd_len = real_len / 2;
                let even_len = odd_len + real_len % 2;
                let is_even = virtual_idx.is_multiple_of(2);
                if is_even {
                    ((virtual_outer_idx * even_len) / virtual_outer_len) * 2
                } else {
                    ((virtual_outer_idx * odd_len) / virtual_outer_len) * 2 + 1
                }
            }
            BidiSchedule::SymmetricCyclic => virtual_outer_idx % real_len,
            BidiSchedule::SymmetricStretched => (virtual_outer_idx * real_len) / virtual_outer_len,
            BidiSchedule::Custom(map) => *map.get(virtual_idx).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests;
