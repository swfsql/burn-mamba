use crate::mamba2::prelude::*;
use burn::prelude::*;

/// Ssd algorithm selection.
///
/// Each variant carries the chunk length Q for the SSD algorithm.  
/// Larger values increase the intra-chunk GEMM work and reduce the
/// inter-chunk scan length.  
/// Optimal value is approximately `√(state_rank · per_head_dim)`.
#[derive(Debug, Clone)]
pub enum Mamba2SsdPath {
    /// Minimal SSD.
    ///
    /// This algorithm mostly uses batched matmuls. For the backward operation, this relies on autodiff.  
    /// See [`chunked_selective_scan`] for more info.
    ///
    /// For training, you may prefer using [SerialRecalculated](Self::SerialRecalculated) instead.
    ///
    /// Based on `/mamba_ssm/modules/ssd_minimal.py` from the `state-spaces/mamba` github reference.
    Minimal(Option<usize>),
    /// (Hybrid) Serial SSD.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// For the backward operation, this relies on autodiff.  
    /// For a custom backwards that saves memory, see [SerialRecalculated](Self::SerialRecalculated).
    ///
    /// Based on 5 kernels on `/mamba_ssm/ops/triton/` from the `state-spaces/mamba` github reference:
    /// - `ssd_chunk_state.py` (K1, K3).
    /// - `ssd_bmm.py` (K2).
    /// - `ssd_state_passing.py` (K4).
    /// - `ssd_chunk_scan.py` (K5).
    Serial(Option<usize>),
    /// (Hybrid) Serial SSD that triggers recalculations for the backward pass.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// Contains a custom backward operation that saves memory.  
    /// For an autodiff backwards, see [Serial](Self::Serial).
    ///
    /// Based on the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py` from the `state-spaces/mamba`
    /// github reference.
    SerialRecalculated(Option<usize>),
}

pub struct Mamba2SsdInput<B: Backend> {
    /// # Shape
    /// - [batch, nchunks, chunk_len, nheads, per_head_dim]
    pub x_bnlhp: Tensor<B, 5>,
    /// # Shape
    /// - [batch, nchunks, chunk_len, nheads]
    pub dt_bnlh: Tensor<B, 4>,
    /// # Shape
    /// - [nheads]
    pub a_decay_h: Tensor<B, 1>,
    /// # Shape
    /// - [batch, nchunks, chunk_len, ngroups, state_rank]
    pub b_bnlgr: Tensor<B, 5>,
    /// # Shape
    /// - [batch, nchunks, chunk_len, ngroups, state_rank]
    pub c_bnlgr: Tensor<B, 5>,
    /// # Shape
    /// - [nheads]
    pub d_h: Tensor<B, 1>,
    /// # Shape
    /// - [batch, nheads, per_head_dim, state_rank]
    pub initial_state_bhpr: Tensor<B, 4>,
    /// # Shape
    /// - [nheads, per_head_dim, state_rank]
    pub init_state_hpr: Option<Tensor<B, 3>>,
}

impl<B: Backend> Mamba2SsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.x_bnlhp);
        san(&self.dt_bnlh);
        san(&self.a_decay_h);
        san(&self.b_bnlgr);
        san(&self.c_bnlgr);
        san(&self.d_h);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl Mamba2SsdPath {
    /// Optimal chunk length is approximately `√(state_rank · per_head_dim)`.
    pub fn optimal_default(state_rank: usize, per_head_dim: usize) -> usize {
        (state_rank * per_head_dim)
            .isqrt()
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .min(512) // rule-of-thumb: ceiling at 512.
    }

    /// Optimal Minimal variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn core_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Minimal(Some(optim))
    }

    /// Optimal Minimal variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::core_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Serial(Some(optim))
    }

    /// Optimal Serial variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_recalculated_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::SerialRecalculated(Some(optim))
    }

    /// Optimal Serial Recalculated variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_recalculated_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_recalculated_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Mamba2SsdPath::Minimal(chunk_len) => *chunk_len,
            Mamba2SsdPath::Serial(chunk_len) => *chunk_len,
            Mamba2SsdPath::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        match self {
            Mamba2SsdPath::Minimal(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            Mamba2SsdPath::Serial(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            Mamba2SsdPath::SerialRecalculated(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
        }
    }

    /// Run the SSD algorithm on the given input.
    ///
    /// Dispatches to `ssd_minimal`, `ssd_serial`, or `ssd_serial_recalculated` based on the variant.
    ///
    /// # Returns
    /// - `y_bnlhp`: `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run<B: Backend + Mamba2BackendExt>(
        &self,
        input: Mamba2SsdInput<B>,
    ) -> (Tensor<B, 5>, Tensor<B, 4>) {
        match self {
            Mamba2SsdPath::Minimal(_) => Mamba2::<B>::ssd_minimal(input),
            Mamba2SsdPath::Serial(_) => Mamba2::<B>::ssd_serial(input),
            Mamba2SsdPath::SerialRecalculated(_) => Mamba2::<B>::ssd_serial_recalculated(input),
        }
    }
}

impl Default for Mamba2SsdPath {
    fn default() -> Mamba2SsdPath {
        // Mamba2SsdPath defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba2SsdPath::SerialRecalculated(None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::Distribution;

    type B = Flex;

    /// Build a randomised set of tensors suitable for cross-path comparison.
    ///
    /// `dt` is drawn from a positive distribution (softplus-like) and `a_decay`
    /// from a negative range so that the implied per-token decay `exp(dt·a)`
    /// stays in `(0, 1]`, matching how the upstream block produces them.
    fn random_input(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        nheads: usize,
        per_head_dim: usize,
        ngroups: usize,
        state_rank: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (
        Tensor<B, 5>,
        Tensor<B, 4>,
        Tensor<B, 1>,
        Tensor<B, 5>,
        Tensor<B, 5>,
        Tensor<B, 1>,
        Tensor<B, 4>,
    ) {
        let x = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let dt = Tensor::<B, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Uniform(0.05, 0.3),
            device,
        );
        let a_decay = Tensor::<B, 1>::random([nheads], Distribution::Uniform(-1.0, -0.5), device);
        let b = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, ngroups, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let c = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, ngroups, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let d = Tensor::<B, 1>::random([nheads], Distribution::Normal(0.0, 0.1), device);
        let initial_state = Tensor::<B, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        );
        (x, dt, a_decay, b, c, d, initial_state)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_input(
        x: Tensor<B, 5>,
        dt: Tensor<B, 4>,
        a_decay: Tensor<B, 1>,
        b: Tensor<B, 5>,
        c: Tensor<B, 5>,
        d: Tensor<B, 1>,
        initial_state: Tensor<B, 4>,
    ) -> Mamba2SsdInput<B> {
        Mamba2SsdInput {
            x_bnlhp: x,
            dt_bnlh: dt,
            a_decay_h: a_decay,
            b_bnlgr: b,
            c_bnlgr: c,
            d_h: d,
            initial_state_bhpr: initial_state,
            // Serial paths assert this is None — see ssd_serial / ssd_serial_recalculated.
            init_state_hpr: None,
        }
    }

    /// Run the same `Mamba2SsdInput` through `Minimal`, `Serial`, and
    /// `SerialRecalculated` and assert that all three yield the same output
    /// and final state. They are all chunkwise reformulations of the same
    /// SSD, so the results must agree up to floating-point noise.
    fn run_minimal_matches_serial(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        nheads: usize,
        per_head_dim: usize,
        ngroups: usize,
        state_rank: usize,
    ) {
        let device = Default::default();
        let (x, dt, a_decay, b, c, d, init) = random_input(
            batch,
            nchunks,
            chunk_len,
            nheads,
            per_head_dim,
            ngroups,
            state_rank,
            &device,
        );

        let (y_min, s_min) = Mamba2SsdPath::Minimal(Some(chunk_len)).run(make_input(
            x.clone(),
            dt.clone(),
            a_decay.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            init.clone(),
        ));
        let (y_ser, s_ser) = Mamba2SsdPath::Serial(Some(chunk_len)).run(make_input(
            x.clone(),
            dt.clone(),
            a_decay.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            init.clone(),
        ));
        let (y_rec, s_rec) = Mamba2SsdPath::SerialRecalculated(Some(chunk_len))
            .run(make_input(x, dt, a_decay, b, c, d, init));

        let tol = 1e-4;

        let dy_ser = (y_min.clone() - y_ser).abs().max().into_scalar();
        let ds_ser = (s_min.clone() - s_ser).abs().max().into_scalar();
        let dy_rec = (y_min - y_rec).abs().max().into_scalar();
        let ds_rec = (s_min - s_rec).abs().max().into_scalar();

        assert!(
            dy_ser < tol,
            "Minimal vs Serial: y max abs diff = {dy_ser:.6} (tol {tol})"
        );
        assert!(
            ds_ser < tol,
            "Minimal vs Serial: final_state max abs diff = {ds_ser:.6} (tol {tol})"
        );
        assert!(
            dy_rec < tol,
            "Minimal vs SerialRecalculated: y max abs diff = {dy_rec:.6} (tol {tol})"
        );
        assert!(
            ds_rec < tol,
            "Minimal vs SerialRecalculated: final_state max abs diff = {ds_rec:.6} (tol {tol})"
        );
    }

    #[test]
    fn paths_agree_no_gqa() {
        // ngroups == nheads (no GQA expansion): B/C are per-head.
        run_minimal_matches_serial(2, 3, 4, 2, 8, 2, 8);
    }

    #[test]
    fn paths_agree_gqa() {
        // ngroups < nheads: B/C are shared across `heads_per_group` heads.
        run_minimal_matches_serial(2, 3, 4, 4, 8, 1, 8);
    }

    #[test]
    fn paths_agree_single_chunk() {
        // nchunks=1 — no inter-chunk scan; checks the intra-chunk + state-passing
        // boundary case where K4 runs a single iteration.
        run_minimal_matches_serial(2, 1, 4, 2, 8, 2, 8);
    }
}
