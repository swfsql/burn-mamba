//! # Custom autodiff node for the quaternion cumulative-product scan
//!
//! Implements [`Mamba3QuatScanBackendExt`](super::quat_scan::Mamba3QuatScanBackendExt)
//! for `Autodiff<B>` via a single Burn [`Backward`] node. The forward keeps only
//! its two leaf inputs (the per-step quaternions `q` and the carry `init`);
//! backprop recomputes the prefix product and evaluates the exact quaternion VJP
//! of the cumulative product, so the `O(log seq)` scan intermediates are never
//! retained.
//!
//! ## Gradient math
//!
//! The scan is `cum[t] = qₜ ⊗ cum[t-1]`, `cum[-1] = init` (so `cum[t] = Pₜ ⊗ init`
//! with the prefix product `Pₜ = qₜ ⊗ ⋯ ⊗ q₀`). Quaternion multiplication is
//! bilinear, and for the real inner product the per-factor VJPs are exact for
//! **all** quaternions:
//!
//! ```text
//!   o = a ⊗ b ,  cotangent ḡ   ⟹   grad_a = ḡ ⊗ conj(b) ,  grad_b = conj(a) ⊗ ḡ .
//! ```
//!
//! Let `G[t]` be the total cotangent reaching `cum[t]` (the upstream `d_cum[t]`
//! plus what flows back from `cum[t+1]`). The reverse recurrence
//! `G[t] = d_cum[t] + conj(qₜ₊₁) ⊗ G[t+1]` telescopes, and **for unit
//! quaternions** (which the rotation always produces) collapses to a
//! parallel-friendly closed form:
//!
//! ```text
//!   S[t]   = Σ_{s≥t} conj(Pₛ) ⊗ d_cum[s]        (suffix-sum = reverse cumsum)
//!   G[t]   = Pₜ ⊗ S[t]
//!   d_q[t] = G[t] ⊗ conj(cum[t-1])              (cum[-1] = init)
//!   d_init = S[0]
//! ```
//!
//! `final_carry = cum[:, −1]` is a high-level autodiff slice (in
//! [`quat_cumprod_recalculated`](super::quat_scan::quat_cumprod_recalculated)),
//! so its gradient is already folded into `d_cum` at the last position before
//! this node runs — the node has a single output.

#![allow(non_snake_case)]

use super::quat_scan::{Mamba3QuatScanBackendExt, fquat_conj, fquat_mul, fquat_prefix_product};
use crate::utils::fprim::F;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes};
use burn::prelude::*;

/// Recompute-based gradient of the quaternion cumulative product.
///
/// Given the saved inputs (`q`, `init`) and the upstream cotangent `d_cum`,
/// recomputes the prefix product `P` and returns `(d_q, d_init)` via the
/// closed-form unit-quaternion VJP documented in the module header. All ops are
/// parallel (a prefix product + a reverse-cumsum); there is no token loop.
fn quat_scan_backward<B: Backend>(
    q_bshj4: F<B, 5>,
    init_bhj4: F<B, 4>,
    d_cum_bshj4: F<B, 5>,
) -> (F<B, 5>, F<B, 4>) {
    let [_batch, sequence, _nheads, _blocks, _four] = q_bshj4.dims();

    // Pₜ = qₜ ⊗ … ⊗ q₀ (prefix product, no carry) — recomputed, not stored.
    let p_bshj4 = fquat_prefix_product::<B>(q_bshj4);

    // cum[t] = Pₜ ⊗ init (broadcast over the sequence axis).
    let init5_b1hj4 = init_bhj4.unsqueeze_dim::<5>(1);
    let cum_bshj4 = fquat_mul::<B>(p_bshj4.clone(), init5_b1hj4.clone());

    // S[t] = Σ_{s≥t} conj(Pₛ) ⊗ d_cum[s] — suffix-sum via flip → cumsum → flip.
    let term_bshj4 = fquat_mul::<B>(fquat_conj::<B>(p_bshj4.clone()), d_cum_bshj4);
    let s_bshj4 = term_bshj4.flip(&[1]).cumsum(1).flip(&[1]);

    // G[t] = Pₜ ⊗ S[t].
    let g_bshj4 = fquat_mul::<B>(p_bshj4, s_bshj4.clone());

    // cum_prev[t] = cum[t-1]; cum_prev[0] = init.
    let cum_prev_bshj4 = if sequence == 1 {
        init5_b1hj4
    } else {
        F::cat(
            vec![init5_b1hj4, cum_bshj4.narrow(1, 0, sequence - 1)],
            1,
        )
    };

    // d_q[t] = G[t] ⊗ conj(cum_prev[t]);  d_init = S[0].
    let d_q_bshj4 = fquat_mul::<B>(g_bshj4, fquat_conj::<B>(cum_prev_bshj4));
    let d_init_bhj4 = s_bshj4.narrow(1, 0, 1).squeeze_dim::<4>(1);

    (d_q_bshj4, d_init_bhj4)
}

impl<B: Backend + Mamba3QuatScanBackendExt, C: CheckpointStrategy> Mamba3QuatScanBackendExt
    for Autodiff<B, C>
{
    fn quat_cumprod(
        q_bshj4: FloatTensor<Self>,
        init_bhj4: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // ── Backward node definition ─────────────────────────────────────────
        #[derive(Debug)]
        struct QuatScanBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            // Saved forward inputs (only the leaves — no scan intermediates).
            q_bshj4: <B as BackendTypes>::FloatTensorPrimitive,
            init_bhj4: <B as BackendTypes>::FloatTensorPrimitive,
            shape_q_bshj4: [usize; 5],
            shape_init_bhj4: [usize; 4],
        }

        impl<B: Backend + Mamba3QuatScanBackendExt> Backward<B, 2> for QuatScanBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_q, node_init] = ops.parents;

                // Upstream gradient of `cum` (already includes the final_carry
                // slice's contribution, scattered in at the last position).
                let d_cum: <B as BackendTypes>::FloatTensorPrimitive = grads.consume::<B>(&ops.node);

                let State {
                    q_bshj4,
                    init_bhj4,
                    shape_q_bshj4,
                    shape_init_bhj4,
                } = ops.state;

                let q = F::<B, 5>::new(q_bshj4).reshape(shape_q_bshj4);
                let init = F::<B, 4>::new(init_bhj4).reshape(shape_init_bhj4);
                // `cum` has the same shape as `q`, hence so does its gradient.
                let d_cum = F::<B, 5>::new(d_cum).reshape(shape_q_bshj4);

                let (d_q, d_init) = quat_scan_backward::<B>(q, init, d_cum);

                if let Some(n) = node_q {
                    grads.register::<B>(n.id, d_q.inner());
                }
                if let Some(n) = node_init {
                    grads.register::<B>(n.id, d_init.inner());
                }
            }
        }

        // ── Shape extraction ───────────────────────────────────────────────
        use burn::backend::TensorMetadata;
        let shape_q_bshj4: [usize; 5] = q_bshj4.primitive.shape().dims();
        let shape_init_bhj4: [usize; 4] = init_bhj4.primitive.shape().dims();

        // ── Register backward / run forward ─────────────────────────────────
        match QuatScanBackward
            .prepare::<C>([q_bshj4.node.clone(), init_bhj4.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let prim_cum = B::quat_cumprod(
                    q_bshj4.primitive.clone(),
                    init_bhj4.primitive.clone(),
                );
                let state = State {
                    q_bshj4: q_bshj4.primitive.clone(),
                    init_bhj4: init_bhj4.primitive.clone(),
                    shape_q_bshj4,
                    shape_init_bhj4,
                };
                prep.finish(state, prim_cum)
            }
            OpsKind::UnTracked(prep) => {
                let prim_cum = B::quat_cumprod(q_bshj4.primitive, init_bhj4.primitive);
                prep.finish(prim_cum)
            }
        }
    }
}
