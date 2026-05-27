//! # Serial-over-chunks SSD (Mamba-2)
//!
//! The chunkwise SSD scan expressed as a serial loop over chunks, mirroring the
//! five Triton kernels of the reference `ssd_combined.py` (`ssd_chunk_state.py`,
//! `ssd_bmm.py`, `ssd_state_passing.py`, `ssd_chunk_scan.py`):
//!
//! - **K1** [`k1_ssd_chunk_cumsum`] — per-chunk cumulative `Δ·A` decays.
//! - **K2** [`k2_ssd_bmm`] — the intra-chunk `C·Bᵀ` block matmul.
//! - **K3** [`k3_ssd_chunk_state`] — each chunk's contribution to its end state
//!   (assuming a zero state at the chunk's start).
//! - **K4** `k4_ssd_state_passing` — the serial inter-chunk scan that carries the
//!   running state across chunk boundaries.
//! - **K5** [`k5_ssd_chunk_scan`] — combines the intra-chunk (attention-like) and
//!   inter-chunk (state-carried) contributions into the output `y`.
//!
//! This produces identical values and gradients to [`super::minimal`]; the
//! serial form keeps per-chunk tensors small (lower peak memory) and is the
//! basis of the recompute backward in [`super::serial_recalculated`].  Gradients
//! here still flow through plain autodiff.

#![allow(unused_variables)]

use crate::mamba2::prelude::*;
use crate::utils::sanity::sanity as san;
use burn::prelude::*;
use burn::backend::Backend;

impl Mamba2SsdInput {
    /// Forward pass for the Mamba-2 SSD module (serial-over-chunks form).
    ///
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    #[allow(non_snake_case)]
    pub fn ssd_serial(self) -> (Tensor<5>, Tensor<4>) {
        let input = self;
        let [batch, nchunks, chunk_len, nheads, per_head_dim] = input.x_bnlhp.dims();
        let [.., state_rank] = input.b_bnlhr.dims();
        assert!(nchunks > 0, "sequence length must be at least 1");

        san(&input.x_bnlhp);
        san(&input.dt_bnlh);
        san(&input.a_decay_h);
        san(&input.b_bnlhr);
        san(&input.c_bnlhr);
        san(&input.d_h);
        san(&input.initial_state_bhpr);

        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented"
        );

        // ── Permutes ──────────────────────────────────────────────────────────────────
        // Note: dt_bnlh calculation (originally in Kernel 1) moved to Step 4 (before padding).
        let dt_discretized_bhnl = input.dt_bnlh.permute([0, 3, 1, 2]);
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            dt_discretized_bhnl.dims()
        );
        san(&dt_discretized_bhnl);

        // ── Kernel 1 ──────────────────────────────────────────────────────────────────
        // IO: (..) -> (da_cumsum_bhnl [used in K3+K5][*], da_chunk_end_bhn [used in K4][omitted][*])
        let (da_cumsum_bhnl, da_chunk_end_bhn): (Tensor<4>, Tensor<3>) =
            k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), input.a_decay_h.clone());
        assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());
        assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());
        san(&da_cumsum_bhnl);
        san(&da_chunk_end_bhn);

        // ── Kernel 2 ──────────────────────────────────────────────────────────────────
        // IO: (..) -> (cb_bnhll [used in K5][!])
        let cb_bnhll: Tensor<5> = k2_ssd_bmm(input.c_bnlhr.clone(), input.b_bnlhr.clone());
        assert_eq!(
            [batch, nchunks, nheads, chunk_len, chunk_len],
            cb_bnhll.dims()
        );
        san(&cb_bnhll);

        // ── Kernel 3 ──────────────────────────────────────────────────────────────────
        // IO: (..) -> (intra_chunk_state_bnhpr [used in K4][!])
        let intra_chunk_state_bnhpr: Tensor<5> = k3_ssd_chunk_state(
            input.x_bnlhp.clone(),
            input.b_bnlhr.clone(),
            da_cumsum_bhnl.clone(),
            dt_discretized_bhnl.clone(),
        );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            intra_chunk_state_bnhpr.dims()
        );
        san(&intra_chunk_state_bnhpr);

        // ── Kernel 4 ──────────────────────────────────────────────────────────────────
        // IO: (..) -> (chunk_input_state_bnhpr [used in K5][!], final_state_bhpr [final output])
        let (chunk_input_state_bnhpr, final_state_bhpr): (Tensor<5>, Tensor<4>) =
            k4_ssd_state_passing(
                intra_chunk_state_bnhpr.clone(),
                da_chunk_end_bhn.clone(),
                input.initial_state_bhpr,
            );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_state_bnhpr.dims()
        );
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            final_state_bhpr.dims()
        );
        san(&chunk_input_state_bnhpr);
        san(&final_state_bhpr);

        // ── Kernel 5 ──────────────────────────────────────────────────────────────────
        let y_bnlhp: Tensor<5> = k5_ssd_chunk_scan(
            da_cumsum_bhnl,
            dt_discretized_bhnl,
            input.x_bnlhp,
            input.c_bnlhr,
            cb_bnhll,
            chunk_input_state_bnhpr,
            input.d_h,
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_bnlhp.dims()
        );
        san(&y_bnlhp);

        (y_bnlhp, final_state_bhpr)
    }
}

/// Based on the Kernel 1 Triton reference `_chunk_cumsum_fwd_kernel` (`ssd_chunk_state.py`).
///
/// Returns:
/// - da_cumsum_bhnl `[used in K3+K5][*]` - intra-chunk cumsum.
/// - da_chunk_end_bhn `[used in K4][omitted][*]` - last da_cumsum per chunk.
pub fn k1_ssd_chunk_cumsum(
    dt_discretized_bhnl: Tensor<4>,
    a_decay_h: Tensor<1>,
) -> (Tensor<4>, Tensor<3>) {
    let [batch, nheads, nchunks, chunk_len] = dt_discretized_bhnl.dims();
    let da_cumsum_bhnl: Tensor<4> = {
        let a_decay_bhnl = a_decay_h
            // - 1/6: unsqueeze-dims: (a_decay_h [*]) -> (a_decay_1h11)
            .unsqueeze_dims::<4>(&[0, 2, 3]) // a_decay_1h11
            // - 2: expand: (a_decay_1h11) -> (a_decay_bhnl)
            .expand([batch, nheads, nchunks, chunk_len]);
        // - 3: mul: (dt_discretized_bhnl [*], a_decay_bhnl) -> (da_bhnl)
        // - 4: cumsum: (da_bhnl) -> (da_cumsum_bhnl [out][*])
        (dt_discretized_bhnl * a_decay_bhnl).cumsum(3)
    };
    assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());

    let da_chunk_end_bhn = da_cumsum_bhnl
        .clone()
        // - 5: slice: (da_cumsum_bhnl [*]) -> (da_cumsum_bhn1)
        .slice(s![.., .., .., -1]) // da_cumsum_bhn1
        // - 6/6: squeeze: (da_cumsum_bhn1) -> (da_chunk_end_bhn [out])
        .squeeze_dim::<3>(3);
    assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

    (da_cumsum_bhnl, da_chunk_end_bhn)
}

/// Based on the Kernel 2 Triton reference `_bmm_chunk_fwd_kernel` (`ssd_bmm.py`).
///
/// Returns:
/// - cb_bnhll `[used in K5][!]`.
pub fn k2_ssd_bmm(c_bnlhr: Tensor<5>, b_bnlhr: Tensor<5>) -> Tensor<5> {
    let [batch, nchunks, chunk_len, nheads, _state_rank] = c_bnlhr.dims();

    // - 1/3: permute: (c_bnlhr [in][*]) -> (c_bnhlr)
    let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]);
    // - 2: permute: (b_bnlhr [in][*]) -> (b_bnhrl)
    let b_bnhrl = b_bnlhr.permute([0, 1, 3, 4, 2]);
    // - 3/3: matmul: (c_bnhlr, b_bnhrl) -> (cb_bnhll [out][!])
    let cb_bnhll: Tensor<5> = c_bnhlr.matmul(b_bnhrl);
    assert_eq!(
        [batch, nchunks, nheads, chunk_len, chunk_len],
        cb_bnhll.dims()
    );
    // Note: cb_bnhll is then only used by Kernel 5.
    cb_bnhll
}

/// Based on the Kernel 3 Triton reference `_chunk_state_fwd_kernel` (`ssd_chunk_state.py`).
///
/// Returns:
/// - cb_bngll `[used in K5][!]` - state assuming zero initial state at each chunk boundary.
/// - b_bar_scale_bhnl `[*]` - intermediary
pub fn k3_ssd_chunk_state(
    x_bnlhp: Tensor<5>,
    b_bnlhr: Tensor<5>,
    da_cumsum_bhnl: Tensor<4>,
    dt_discretized_bhnl: Tensor<4>,
) -> Tensor<5> {
    use burn::tensor::s;

    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
    let [.., state_rank] = b_bnlhr.dims();

    // permute b and x to prepare them for the matmul
    // - 1/15: permute: (x_bnlhp [in][*]) -> (x_bnhpl)
    let x_bnhpl = x_bnlhp.clone().permute([0, 1, 3, 4, 2]);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, chunk_len],
        x_bnhpl.dims()
    );
    // - 2: permute: (b_bnlhr [in][*]) -> (b_bnhlr)
    let b_bnhlr = b_bnlhr.permute([0, 1, 3, 2, 4]);
    assert_eq!(
        [batch, nchunks, nheads, chunk_len, state_rank],
        b_bnhlr.dims()
    );

    // scale b
    let b_scaled_bnhlr = {
        let b_bar_scale_bhnl = {
            let da_cumsum_last_in_chunk_bhn1 =
                // - 6: slice: (da_cumsum_bhnl [in][*]) -> (da_cumsum_last_in_chunk_bhn1)
                da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            assert_eq!(
                [batch, nheads, nchunks, 1],
                da_cumsum_last_in_chunk_bhn1.dims()
            );

            // - 7: expand: (da_cumsum_last_in_chunk_bhn1) -> (da_cumsum_last_bhnl)
            let da_cumsum_last_bhnl =
                da_cumsum_last_in_chunk_bhn1.expand([batch, nheads, nchunks, chunk_len]);
            // - 8: sub: (da_cumsum_last_bhnl, da_cumsum_bhnl [from K1][*]) -> (da_delta_bhnl)
            let da_delta_bhnl = da_cumsum_last_bhnl - da_cumsum_bhnl.clone();
            // - 9: exp: (da_delta_bhnl) -> (forward_decay_to_chunk_end_bhnl [+])
            let forward_decay_to_chunk_end_bhnl = da_delta_bhnl.exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len],
                forward_decay_to_chunk_end_bhnl.dims()
            );

            // - 10: mul: (forward_decay_to_chunk_end_bhnl [+], dt_discretized_bhnl [in][*]) -> (b_bar_scale_bhnl [+])
            forward_decay_to_chunk_end_bhnl * dt_discretized_bhnl.clone()
        };
        assert_eq!([batch, nheads, nchunks, chunk_len], b_bar_scale_bhnl.dims());

        // - 11: permute: (b_bar_scale_bhnl [+]) -> (b_bar_scale_bnhl)
        let b_bar_scale_bnhl = b_bar_scale_bhnl.permute([0, 2, 1, 3]);
        assert_eq!([batch, nchunks, nheads, chunk_len], b_bar_scale_bnhl.dims());
        let b_bar_scale_bnhlr = b_bar_scale_bnhl
            // - 12: unsqueeze: (b_bar_scale_bnhl) -> (b_bar_scale_bnhl1)
            .unsqueeze_dim::<5>(4) // b_bar_scale_bnhl1
            // - 13: expand: (b_bar_scale_bnhl1) -> (b_bar_scale_bnhlr)
            .expand([batch, nchunks, nheads, chunk_len, state_rank]);
        // - 14: mul: (b_bnhlr, b_bar_scale_bnhlr) -> (b_scaled_bnhlr [+])
        b_bnhlr * b_bar_scale_bnhlr
    };
    assert_eq!(
        [batch, nchunks, nheads, chunk_len, state_rank],
        b_scaled_bnhlr.dims()
    );

    // - 15/15: matmul: (x_bnhpl, b_scaled_bnhlr [+]) -> (intra_chunk_state_bnhpr [out][!])
    let intra_chunk_state_bnhpr: Tensor<5> = x_bnhpl.matmul(b_scaled_bnhlr);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    intra_chunk_state_bnhpr
}

/// Based on the Kernel 4 Triton reference `_state_passing_fwd_kernel` (`ssd_state_passing.py`).
///
/// Returns:
/// - chunk_input_state_bnhpr `[used in K5][!]`.
/// - final_state_bhpr `[final output]`.
pub fn k4_ssd_state_passing(
    intra_chunk_state_bnhpr: Tensor<5>,
    da_chunk_end_bhn: Tensor<3>,
    initial_state_bhpr: Tensor<4>,
) -> (Tensor<5>, Tensor<4>) {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_chunk_state_bnhpr.dims();
    let flat_state_dim = per_head_dim * state_rank;

    // - 1/5: init-mut: (initial_state_bhpr [in][*]) -> (running_state_bhpr)
    let mut running_state_bhpr = initial_state_bhpr;
    assert_eq!(
        [batch, nheads, per_head_dim, state_rank],
        running_state_bhpr.dims()
    );

    let mut chunk_input_state_vec_bhpr = Vec::with_capacity(nchunks + 1);
    // - 2: vec-push: (running_state_bhpr [elem]) -> (chunk_input_state_vec_bhpr [vec][!])
    chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());

    // - 3: serial-loop: (0..nchunks)
    for i_chunk in 0..nchunks {
        let intra_state_bhpr = intra_chunk_state_bnhpr
            .clone()
            //   - 3.1/3.9: slice: (intra_chunk_state_bnhpr [in][!]) -> (intra_chunk_state_b1hpr)
            .slice(s![.., i_chunk, .., .., ..]) // intra_chunk_state_b1hpr
            //   - 3.2: squeeze: (intra_chunk_state_b1hpr) -> (intra_state_bhpr)
            .squeeze_dim::<4>(1);
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            intra_state_bhpr.dims()
        );

        let decay_bhpr = da_chunk_end_bhn
            .clone()
            //   - 3.3: slice: (da_chunk_end_bhn [in][*]) -> (da_chunk_end_bh1)
            .slice(s![.., .., i_chunk]) // da_chunk_end_bh1
            //   - 3.4: exp: (da_chunk_end_bh1) -> (exp_da_chunk_end_bh1)
            .exp() // exp_da_chunk_end_bh1
            //   - 3.5: unsqueeze: (exp_da_chunk_end_bh1) -> (exp_da_chunk_end_bh11)
            .unsqueeze_dim::<4>(3) // exp_da_chunk_end_bh11
            //   - 3.6: expand: (exp_da_chunk_end_bh11) -> (decay_bhpr)
            .expand([batch, nheads, per_head_dim, state_rank]);

        // SSM recurrence: running_state = decay * running_state + intra_state
        running_state_bhpr =
        //   - 3.7: mul: (decay_bhpr, running_state_bhpr) -> (running_state_bhpr)
            (decay_bhpr * running_state_bhpr) // running_state_bhpr
        //   - 3.8: add: (running_state_bhpr, intra_state_bhpr) -> (running_state_bhpr)
            + intra_state_bhpr;
        //   - 3.9/3.9: vec-push: (running_state_bhpr [elem]) -> (chunk_input_state_vec_bhpr [vec][!])
        chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());
    }

    // - 4: vec-pop: (chunk_input_state_vec_bhpr [vec][!]) -> (final_state_bhpr [elem][out][!])
    let final_state_bhpr = chunk_input_state_vec_bhpr.pop().unwrap();
    assert_eq!(
        [batch, nheads, per_head_dim, state_rank],
        final_state_bhpr.dims()
    );

    // - 5/5: stack: (chunk_input_state_vec_bhpr [!]) -> (chunk_input_state_bnhpr [out][!])
    let chunk_input_state_bnhpr = Tensor::stack(chunk_input_state_vec_bhpr, 1);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        chunk_input_state_bnhpr.dims()
    );

    (chunk_input_state_bnhpr, final_state_bhpr)
}

/// Based on the Kernel 5 Triton reference `_chunk_scan_fwd_kernel` (`ssd_chunk_scan.py`).
///
/// Returns:
/// - y_bnlhp `[final output]`
pub fn k5_ssd_chunk_scan(
    da_cumsum_bhnl: Tensor<4>,
    dt_discretized_bhnl: Tensor<4>,
    x_bnlhp: Tensor<5>,
    c_bnlhr: Tensor<5>,
    cb_bnhll: Tensor<5>,
    chunk_input_state_bnhpr: Tensor<5>,
    d_h: Tensor<1>,
) -> Tensor<5> {
    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
    let device = x_bnlhp.device();

    // Rearrange inputs to the common [batch, nchunks, nheads, ...] ordering used below.
    // - 1/36: permute: (da_cumsum_bhnl [*]) -> (da_cumsum_bnhl)
    let da_cumsum_bnhl = da_cumsum_bhnl.permute([0, 2, 1, 3]);
    san(&da_cumsum_bnhl);
    // - 2: permute: (dt_discretized_bhnl [*]) -> (dt_bnhl)
    let dt_bnhl = dt_discretized_bhnl.permute([0, 2, 1, 3]);
    san(&dt_bnhl);
    // - 3: permute: (x_bnlhp [*]) -> (x_bnhlp)
    let x_bnhlp = x_bnlhp.clone().permute([0, 1, 3, 2, 4]);
    san(&x_bnhlp);

    // B/C are already per-head — only a permute is needed.
    let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]);
    san(&c_bnhlr);
    san(&cb_bnhll);

    // ── BLUE: exp(dA[l]) · C[l,:] @ state_in^T ─────────────────────────────
    //
    //   blue[b,n,h,l,p] = exp(da[b,n,h,l]) · Σ_r  c[b,n,h,l,r] · state[b,n,h,p,r]
    //
    //   [b,n,h,l,r] @ [b,n,h,r,p]  →  [b,n,h,l,p]
    let exp_da_cumsum_bnhlp = da_cumsum_bnhl
        .clone()
        // - 11: exp: (da_cumsum_bnhl) -> (exp_da_cumsum_bnhl)
        .exp()
        // - 12: unsqueeze: (exp_da_cumsum_bnhl) -> (exp_da_cumsum_bnhl1)
        .unsqueeze_dim::<5>(4) // exp_da_cumsum_bnhl1
        // - 13: expand: (exp_da_cumsum_bnhl1) -> (exp_da_cumsum_bnhlp)
        .expand([batch, nchunks, nheads, chunk_len, per_head_dim]);
    san(&exp_da_cumsum_bnhlp);
    // - 14: permute: (chunk_input_state_bnhpr [!]) -> (chunk_input_state_bnhrp)
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]);
    // - 15: matmul: (c_bnhlr, chunk_input_state_bnhrp) -> (blue_bnhlp)
    let blue_scaled_bnhlp = c_bnhlr
        .matmul(chunk_input_state_bnhrp)  // blue_bnhlp
        // - 16: mul: (blue_bnhlp, exp_da_cumsum_bnhlp) -> (blue_scaled_bnhlp)
        * exp_da_cumsum_bnhlp;
    san(&blue_scaled_bnhlp);

    // ── ORANGE: causal CB_weighted @ X ──────────────────────────────────────
    //
    //   orange[b,n,h,l,p] = Σ_{s≤l} CB[l,s] · exp(da[l]-da[s]) · dt[s] · x[s,p]
    //
    // Precompute the full lower-triangular weight matrix, then do a single matmul.
    //
    let da_cumsum_target_bnhll = da_cumsum_bnhl
        .clone()
        // - 17: unsqueeze: (da_cumsum_bnhl) -> (da_cumsum_bnhl1)
        .unsqueeze_dim::<5>(4) // da_cumsum_bnhl1
        // - 18: expand: (da_cumsum_bnhl1) -> (da_cumsum_target_bnhll)
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    // println!("{}", da_cumsum_target_bnhll);
    san(&da_cumsum_target_bnhll);
    let da_cumsum_source_bnhll = da_cumsum_bnhl
        // - 19: unsqueeze: (da_cumsum_bnhl) -> (da_cumsum_bnh1l)
        .unsqueeze_dim::<5>(3) // da_cumsum_bnh1l
        // - 20: expand: (da_cumsum_bnh1l) -> (da_cumsum_source_bnhll)
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    // println!("{}", da_cumsum_source_bnhll);
    san(&da_cumsum_source_bnhll);
    // - 21: sub: (da_cumsum_target_bnhll, da_cumsum_source_bnhll) -> (da_cumsum_diff_bnhll)
    let da_cumsum_diff_bnhll = da_cumsum_target_bnhll - da_cumsum_source_bnhll;
    san(&da_cumsum_diff_bnhll);

    // note: overflow instability at step 22, a `minimal::segsum`-like upper triangle protection is necessary.
    // - 21.1: tril-mask: (0) -> (causal_mask_ll), expanded as a view to causal_mask_bnhll.
    // true above the main diagonal, false at diagonal and below.
    // Built at [L,L] and broadcast — the mask values do not depend on (b,n,h).
    let causal_mask_bnhll: Tensor<5, burn::prelude::Bool> =
        Tensor::<2, burn::prelude::Bool>::tril_mask([chunk_len, chunk_len], 0, &device)
            .reshape([1, 1, 1, chunk_len, chunk_len])
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    // - 21.2: mask-fill: (da_cumsum_diff_bnhll, causal_mask_bnhll) -> (da_cumsum_diff_masked_bnhll)
    // Causal mask and exp stabilizer: above upper diagonal set to -inf.
    let da_cumsum_diff_masked_bnhll =
        da_cumsum_diff_bnhll.mask_fill(causal_mask_bnhll, f32::NEG_INFINITY);

    // - 22: exp: (da_cumsum_diff_masked_bnhll) -> (da_cumsum_diff_exp_bnhll)
    let da_cumsum_diff_exp_bnhll = da_cumsum_diff_masked_bnhll.exp();
    san(&da_cumsum_diff_exp_bnhll);
    let dt_source_bnhll = dt_bnhl
        // - 23: unsqueeze: (dt_bnhl) -> (dt_bnh1l)
        .unsqueeze_dim::<5>(3) // dt_bnh1l
        // - 24: expand: (dt_bnh1l) -> (dt_source_bnhll)
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    san(&dt_source_bnhll);

    // note: steps 25, 26 and 29 are no longer necessary.
    // // Causal mask (0 above the main diagonal, 1 elsewhere).
    // let causal_mask_bnhll =
    //     // - 25: ones: (1) -> (ones_bnhll)
    //     Tensor::ones([batch, nchunks, nheads, chunk_len, chunk_len], &device)
    //     // - 26: tril: (ones_bnhll, 0) -> (causal_mask_bnhll)
    //     .tril(0);

    //   [b,n,h,l,l] @ [b,n,h,l,p]  →  [b,n,h,l,p]
    // - 27: mul: (cb_bnhll, da_cumsum_diff_exp_bnhll) -> (orange_lhs_partial1_bnhll)
    let orange_lhs_partial1_bnhll = cb_bnhll * da_cumsum_diff_exp_bnhll;
    san(&orange_lhs_partial1_bnhll);
    // - 28: mul: (orange_lhs_partial1_bnhll, dt_source_bnhll) -> (orange_lhs_partial2_bnhll)
    let orange_lhs_partial2_bnhll = orange_lhs_partial1_bnhll * dt_source_bnhll;
    san(&orange_lhs_partial2_bnhll);
    // // - 29: mul: (orange_lhs_partial2_bnhll, causal_mask_bnhll) -> (orange_lhs_partial3_bnhll)
    // let orange_lhs_partial3_bnhll = orange_lhs_partial2_bnhll * causal_mask_bnhll;
    // san(&orange_lhs_partial3_bnhll);
    // - 30: matmul: (orange_lhs_partial3_bnhll, x_bnhlp) -> (orange_bnhlp)
    // - 30: matmul: (orange_lhs_partial2_bnhll, x_bnhlp) -> (orange_bnhlp)
    let orange_bnhlp = orange_lhs_partial2_bnhll.matmul(x_bnhlp);
    san(&orange_bnhlp);

    // ── SKIP: D[h] · x[l,p] ─────────────────────────────────────────────────
    //
    //   D_HAS_HDIM = False: D is a scalar per head, shape [nheads].
    //   Triton: `acc += x_residual * D`
    let skip_bnlhp = d_h
        // - 31: unsqueeze-dims: (d_h [*]) -> (d_111h1)
        .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // d_111h1
        // - 32: expand: (d_111h1) -> (d_bnlhp)
        .expand([
            batch,
            nchunks,
            chunk_len,
            nheads,
            per_head_dim,
        ]) // d_bnlhp
    // - 33: mul: (d_bnlhp, x_bnlhp[*]) -> (skip_bnlhp)
    * x_bnlhp;
    san(&skip_bnlhp);

    // Permute BLUE + ORANGE from [b,n,h,l,p] back to [b,n,l,h,p], then add SKIP.
    // - 34: add: (blue_scaled_bnhlp, orange_bnhlp) -> (y_partial_bnhlp)
    let y_partial_bnhlp = blue_scaled_bnhlp + orange_bnhlp;
    san(&y_partial_bnhlp);
    // - 35: permute: (y_partial_bnhlp) -> (y_partial_bnlhp)
    let y_partial_bnlhp = y_partial_bnhlp.permute([0, 1, 3, 2, 4]);
    san(&y_partial_bnlhp);
    // - 36/36: add: (y_partial_bnlhp, skip_bnlhp) -> (y_bnlhp [out])
    let y_bnlhp: Tensor<5> = y_partial_bnlhp + skip_bnlhp;
    san(&y_bnlhp);

    assert_eq!(
        [batch, nchunks, chunk_len, nheads, per_head_dim],
        y_bnlhp.dims()
    );
    y_bnlhp
}
