#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Schedule {
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:  
    /// `  →    →    →      →    →    →      →    →       `
    /// `(0⇒0, 1⇒1, 2⇒2), (3⇒0, 4⇒1, 5⇒2), (6⇒0, 7⇒1, ...)`
    #[default]
    Cyclic,
    /// Fills virtual positions by stretching the real schedule.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:  
    /// `  →    →    →      →    →    →      →    →       `
    /// `(0⇒0, 1⇒0, 2⇒0), (3⇒1, 4⇒1, 5⇒1), (6⇒2, 7⇒2, ...)`
    Stretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3, custom = [0, 1, 2, 2, 1, 0, 0, 0]:  
    /// `  →    →    →    →    →    →    →    →       `
    /// `(0⇒0, 1⇒1, 2⇒2, 3⇒2, 4⇒1, 5⇒0, 6⇒0, 7⇒0, ...)`
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

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BidiSchedule {
    /// Use even virtual positions for straight-direction (→), and odd virtual positions for
    /// reverse-direction (←), wrapping around for each schedule.
    //
    /// # Example
    /// - virtual len = 10, real len = 4:  
    /// `   →    ←      →    ←        →    ←      →    ←        →    ←          `
    /// `[(0⇒0, 1⇒1), (2⇒2, 3⇒3)], [(4⇒0, 5⇒1), (6⇒2, 7⇒3)], [(8⇒0, 9⇒1), (...)]`
    #[default]
    StridedCyclic,
    /// Use even virtual positions for straight-direction (→), and odd virtual positions for
    /// reverse-direction (←), stretching for each schedule.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    /// `   →    ←      →    ←      →    ←        →    ←      →    ←          `
    /// `[(0⇒0, 1⇒1), (2⇒0, 3⇒1), (4⇒0, 5⇒1)], [(6⇒2, 7⇒3), (8⇒2, 9⇒3), (...)]`
    StridedStretched,
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion,
    /// replicating between the straight (→) and reverse (←) directions.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    /// `   →    ←      →    ←      →    ←      →    ←        →    ←          `
    /// `[(0⇒0, 1⇒0), (2⇒1, 3⇒1), (4⇒2, 5⇒2), (6⇒3, 7⇒3)], [(8⇒0, 9⇒0), (...)]`
    SymmetricCyclic,
    /// Fills virtual positions by stretching the real schedule, replicating between
    /// the straight (→) and reverse (←) directions.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    /// `   →    ←      →    ←       →    ←               →    ←        →    ←   `
    /// `[(0⇒0, 1⇒0), (2⇒0, 3⇒0)],[(4⇒1, 5⇒1), (...)], [(6⇒2, 7⇒2)], [(8⇒3, 9⇒3)]`
    SymmetricStretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4, custom = [0, 1, 2, 2, 1, 0, 0, 0, 3, 2]:  
    /// `   →    ←        →    ←        →    ←        →    ←        →    ←            `
    /// `[(0⇒0, 1⇒1)], [(2⇒2, 3⇒2)], [(4⇒1, 5⇒0)], [(6⇒0, 7⇒0)], [(8⇒3, 9⇒2)], [(...)]`
    Custom(Vec<usize>),
}

impl BidiSchedule {
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        let virtual_outer_idx = virtual_idx / 2;
        let virtual_outer_len = virtual_len / 2;
        match self {
            BidiSchedule::StridedCyclic => {
                let odd_len = real_len / 2;
                let even_len = odd_len + real_len % 2;
                let is_even = virtual_idx % 2 == 0;
                if is_even {
                    (virtual_outer_idx % even_len) * 2
                } else {
                    (virtual_outer_idx % odd_len) * 2 + 1
                }
            }
            BidiSchedule::StridedStretched => {
                let odd_len = real_len / 2;
                let even_len = odd_len + real_len % 2;
                let is_even = virtual_idx % 2 == 0;
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
mod test {
    use super::*;

    #[test]
    fn schedule() {
        use Schedule::*;
        assert_eq!(
            (0..8)
                .map(|i| Schedule::real_idx(&Cyclic, i, 8, 3))
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 0, 1, 2, 0, 1]
        );
        assert_eq!(
            (0..8)
                .map(|i| Schedule::real_idx(&Stretched, i, 8, 3))
                .collect::<Vec<_>>(),
            vec![0, 0, 0, 1, 1, 1, 2, 2]
        );
        let custom = vec![0, 1, 2, 2, 1, 0, 0, 0];
        assert_eq!(
            (0..8)
                .map(|i| Schedule::real_idx(&Custom(custom.clone()), i, 8, 3))
                .collect::<Vec<_>>(),
            custom
        );
    }

    #[test]
    fn bidi_schedule() {
        use BidiSchedule::*;
        assert_eq!(
            (0..10)
                .map(|i| BidiSchedule::real_idx(&StridedCyclic, i, 10, 4))
                .collect::<Vec<_>>(),
            vec![
                0, 1, /**/ 2, 3, /**/ 0, 1, /**/ 2, 3, /**/ 0, 1
            ]
        );
        assert_eq!(
            (0..10)
                .map(|i| BidiSchedule::real_idx(&StridedStretched, i, 10, 4))
                .collect::<Vec<_>>(),
            vec![
                0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 2, 3, /**/ 2, 3
            ]
        );
        assert_eq!(
            (0..10)
                .map(|i| BidiSchedule::real_idx(&SymmetricCyclic, i, 10, 4))
                .collect::<Vec<_>>(),
            vec![
                0, 0, /**/ 1, 1, /**/ 2, 2, /**/ 3, 3, /**/ 0, 0
            ]
        );
        assert_eq!(
            (0..10)
                .map(|i| BidiSchedule::real_idx(&SymmetricStretched, i, 10, 4))
                .collect::<Vec<_>>(),
            vec![
                0, 0, /**/ 0, 0, /**/ 1, 1, /**/ 2, 2, /**/ 3, 3
            ]
        );
        let custom = vec![
            0, 1, /**/ 2, 2, /**/ 1, 0, /**/ 0, 0, /**/ 3, 2,
        ];
        assert_eq!(
            (0..10)
                .map(|i| BidiSchedule::real_idx(&Custom(custom.clone()), i, 10, 4))
                .collect::<Vec<_>>(),
            custom
        );
    }
}
