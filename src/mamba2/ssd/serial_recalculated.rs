#![allow(unused_variables)]

use crate::mamba2::prelude::*;
use burn::prelude::*;

impl<B: Backend> Mamba2<B> {
    /// Forward pass for the Mamba-2 SSD module.
    ///
    /// # Tensor shapes flowing through the function
    ///
    /// ```text
    /// x_bshp:                  [batch, sequence, nheads, per_head_dim]
    /// dt_bsh:                  [batch, sequence, nheads]
    /// a_decay_h:               [nheads]
    /// b_bsgr:                  [batch, sequence, ngroups, state_rank]
    /// c_bsgr:                  [batch, sequence, ngroups, state_rank]
    /// d_h:                     [nheads]
    /// ssm_initial_state_bhpr   [batch, nheads, per_head_dim, state_rank]
    /// _init_states_hpr         [nheads, per_head_dim, state_rank]
    /// ```
    #[allow(non_snake_case)]
    pub fn ssd_serial_recalculated(
        x_bshp: Tensor<B, 4>,
        dt_bsh: Tensor<B, 3>,
        a_decay_h: Tensor<B, 1>,
        b_bsgr: Tensor<B, 4>,
        c_bsgr: Tensor<B, 4>,
        d_h: Tensor<B, 1>,
        ssm_initial_state_bhpr: Tensor<B, 4>,
        _init_states_hpr: Option<Tensor<B, 3>>,
        ngroups: usize,
        state_rank: usize,
        // currently must be set by caller
        chunk_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // temporarily fallsback into the autodiff-serial
        Self::ssd_serial(
            x_bshp,
            dt_bsh,
            a_decay_h,
            b_bsgr,
            c_bsgr,
            d_h,
            ssm_initial_state_bhpr,
            _init_states_hpr,
            ngroups,
            state_rank,
            chunk_len,
        )
    }
}
