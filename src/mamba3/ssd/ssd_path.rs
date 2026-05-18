use crate::mamba3::prelude::*;
use burn::prelude::*;

/// Ssd algorithm selection.
///
/// Each variant carries the chunk length Q for the SSD algorithm.
/// Larger values increase the intra-chunk GEMM work and reduce the
/// inter-chunk scan length.
/// Optimal value is approximately `√(state_rank · per_head_dim)`.
#[derive(Debug, Clone)]
pub enum SsdPath {
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

/// MIMO-first SSD input.
///
/// All tensors are pre-processed: B/C are already QK-normed, RoPE-applied, bias-added, and
/// expanded to per-head (not per-group). V is already scaled by the trapezoidal coefficient
/// (γ or β). The combined log-decay `da = Δ·A` is pre-computed. D skip is handled by the caller.
pub struct SsdInput<B: Backend> {
    /// Value tensor, already scaled by trapezoidal coefficient (γ or β).
    ///
    /// # Shape
    /// - [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]
    pub v_bnlrhp: Tensor<B, 6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - [batch, nchunks, chunk_len, nheads]
    pub da_bnlh: Tensor<B, 4>,

    /// Key/B tensor: QK-normed, RoPE-applied, bias-added, expanded to per-head, per-rank.
    ///
    /// # Shape
    /// - [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]
    pub b_bnlrhn: Tensor<B, 6>,

    /// Query/C tensor: same processing as B.
    ///
    /// # Shape
    /// - [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]
    pub c_bnlrhn: Tensor<B, 6>,

    /// Initial SSM hidden state.
    ///
    /// # Shape
    /// - [batch, nheads, per_head_dim, state_rank]
    pub initial_state_bhpr: Tensor<B, 4>,

    /// Optional learnable initial state (broadcast over batch).
    ///
    /// # Shape
    /// - [nheads, per_head_dim, state_rank]
    pub init_state_hpr: Option<Tensor<B, 3>>,
}

impl<B: Backend> SsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.v_bnlrhp);
        san(&self.da_bnlh);
        san(&self.b_bnlrhn);
        san(&self.c_bnlrhn);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl SsdPath {
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
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
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
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
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
    pub fn chunked_recalculated_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
        Self::chunked_recalculated_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            SsdPath::Minimal(chunk_len) => *chunk_len,
            SsdPath::Serial(chunk_len) => *chunk_len,
            SsdPath::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        match self {
            SsdPath::Minimal(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            SsdPath::Serial(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            SsdPath::SerialRecalculated(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
        }
    }

    /// Run the SSD algorithm on the given MIMO-first input.
    ///
    /// Dispatches to `ssd_minimal`, `ssd_serial`, or `ssd_serial_recalculated` based on the variant.
    ///
    /// # Returns
    /// - `y_bnlrhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run<B: Backend + Mamba3BackendExt>(
        &self,
        input: SsdInput<B>,
    ) -> (Tensor<B, 6>, Tensor<B, 4>) {
        match self {
            SsdPath::Minimal(_) => Mamba3::<B>::ssd_minimal(input),
            SsdPath::Serial(_) => Mamba3::<B>::ssd_serial(input),
            SsdPath::SerialRecalculated(_) => Mamba3::<B>::ssd_serial_recalculated(input),
        }
    }
}

impl Default for SsdPath {
    fn default() -> SsdPath {
        // SsdPath defaults to the SerialRecalculated algorithm with the optimal chunk length.
        SsdPath::SerialRecalculated(None)
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

    /// Build a randomised [`SsdInput`] suitable for cross-path comparison.
    ///
    /// `da` is drawn from a negative-mean distribution so that the implied
    /// per-token decay `exp(da)` stays in `(0, 1]`, matching how the upstream
    /// block produces `Δ · A` with `A < 0`.
    fn random_input(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        mimo_rank: usize,
        nheads: usize,
        per_head_dim: usize,
        state_rank: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (
        Tensor<B, 6>,
        Tensor<B, 4>,
        Tensor<B, 6>,
        Tensor<B, 6>,
        Tensor<B, 4>,
    ) {
        let v = Tensor::<B, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let da = Tensor::<B, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Normal(-0.5, 0.1),
            device,
        );
        let b = Tensor::<B, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let c = Tensor::<B, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let initial_state = Tensor::<B, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        );
        (v, da, b, c, initial_state)
    }

    fn make_input(
        v: Tensor<B, 6>,
        da: Tensor<B, 4>,
        b: Tensor<B, 6>,
        c: Tensor<B, 6>,
        initial_state: Tensor<B, 4>,
    ) -> SsdInput<B> {
        SsdInput {
            v_bnlrhp: v,
            da_bnlh: da,
            b_bnlrhn: b,
            c_bnlrhn: c,
            initial_state_bhpr: initial_state,
            // Serial paths assert this is None — see ssd_serial / ssd_serial_recalculated.
            init_state_hpr: None,
        }
    }

    /// Run the same `SsdInput` through `Minimal`, `Serial`, and
    /// `SerialRecalculated` and assert that all three yield the same output
    /// and final state. They are all chunkwise reformulations of the same
    /// MIMO-first SSD, so the results must agree up to floating-point noise.
    fn run_minimal_matches_serial(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        mimo_rank: usize,
        nheads: usize,
        per_head_dim: usize,
        state_rank: usize,
    ) {
        let device = Default::default();
        let (v, da, b, c, init) = random_input(
            batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim, state_rank, &device,
        );

        let (y_min, s_min) = SsdPath::Minimal(Some(chunk_len)).run(make_input(
            v.clone(), da.clone(), b.clone(), c.clone(), init.clone(),
        ));
        let (y_ser, s_ser) = SsdPath::Serial(Some(chunk_len)).run(make_input(
            v.clone(), da.clone(), b.clone(), c.clone(), init.clone(),
        ));
        let (y_rec, s_rec) = SsdPath::SerialRecalculated(Some(chunk_len))
            .run(make_input(v, da, b, c, init));

        let tol = 1e-4;

        let dy_ser = (y_min.clone() - y_ser).abs().max().into_scalar();
        let ds_ser = (s_min.clone() - s_ser).abs().max().into_scalar();
        let dy_rec = (y_min - y_rec).abs().max().into_scalar();
        let ds_rec = (s_min - s_rec).abs().max().into_scalar();

        assert!(dy_ser < tol, "Minimal vs Serial: y max abs diff = {dy_ser:.6} (tol {tol})");
        assert!(ds_ser < tol, "Minimal vs Serial: final_state max abs diff = {ds_ser:.6} (tol {tol})");
        assert!(dy_rec < tol, "Minimal vs SerialRecalculated: y max abs diff = {dy_rec:.6} (tol {tol})");
        assert!(ds_rec < tol, "Minimal vs SerialRecalculated: final_state max abs diff = {ds_rec:.6} (tol {tol})");
    }

    #[test]
    fn paths_agree_siso() {
        // batch=2, nchunks=3, chunk_len=4, mimo_rank=1, nheads=2, per_head_dim=8, state_rank=8
        run_minimal_matches_serial(2, 3, 4, 1, 2, 8, 8);
    }

    #[test]
    fn paths_agree_mimo() {
        // mimo_rank=2 exercises the fused-L (= chunk_len · R) reshape shared by all three paths.
        run_minimal_matches_serial(2, 3, 4, 2, 2, 8, 8);
    }

    #[test]
    fn paths_agree_single_chunk() {
        // nchunks=1 — no inter-chunk scan; checks the intra-chunk + state-passing
        // boundary case where K4 runs a single iteration.
        run_minimal_matches_serial(2, 1, 4, 1, 2, 8, 8);
    }
}
