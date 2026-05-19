use crate::mamba3::prelude::*;
use burn::prelude::*;

/// Ssd algorithm selection.
///
/// Each variant carries the chunk length Q for the SSD algorithm.
/// Larger values increase the intra-chunk GEMM work and reduce the
/// inter-chunk scan length.
/// Optimal value is approximately `√(state_rank · per_head_dim)`.
#[derive(Debug, Clone)]
pub enum Mamba3SsdPath {
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
pub struct Mamba3SsdInput<B: Backend> {
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

impl<B: Backend> Mamba3SsdInput<B> {
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

impl Mamba3SsdPath {
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
            Mamba3SsdPath::Minimal(chunk_len) => *chunk_len,
            Mamba3SsdPath::Serial(chunk_len) => *chunk_len,
            Mamba3SsdPath::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        match self {
            Mamba3SsdPath::Minimal(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            Mamba3SsdPath::Serial(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            Mamba3SsdPath::SerialRecalculated(chunk_len) => {
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
        input: Mamba3SsdInput<B>,
    ) -> (Tensor<B, 6>, Tensor<B, 4>) {
        match self {
            Mamba3SsdPath::Minimal(_) => Mamba3::<B>::ssd_minimal(input),
            Mamba3SsdPath::Serial(_) => Mamba3::<B>::ssd_serial(input),
            Mamba3SsdPath::SerialRecalculated(_) => Mamba3::<B>::ssd_serial_recalculated(input),
        }
    }
}

impl Default for Mamba3SsdPath {
    fn default() -> Mamba3SsdPath {
        // Mamba3SsdPath defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba3SsdPath::SerialRecalculated(None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::module::Param;
    use burn::tensor::Distribution;

    /// Inner (non-autodiff) backend used for materialising values and
    /// extracted gradients.
    type InnerB = Flex;
    /// Autodiff-wrapped backend used to drive `.backward()`.
    type B = Autodiff<InnerB>;

    type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

    /// Build a randomised set of tensors on the inner backend.  `Param`s
    /// wrapping these are built per-path so each path gets a fresh autodiff
    /// graph.
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
        device: &Device,
    ) -> (
        Tensor<InnerB, 6>,
        Tensor<InnerB, 4>,
        Tensor<InnerB, 6>,
        Tensor<InnerB, 6>,
        Tensor<InnerB, 4>,
    ) {
        let v = Tensor::<InnerB, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let da = Tensor::<InnerB, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Normal(-0.5, 0.1),
            device,
        );
        let b = Tensor::<InnerB, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let c = Tensor::<InnerB, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let initial_state = Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        );
        (v, da, b, c, initial_state)
    }

    /// Inputs wrapped as `Param`s so each tensor becomes an autodiff leaf
    /// with `require_grad`.  A fresh `Inputs` is built per path so each path
    /// runs with its own independent autodiff graph.
    struct Inputs {
        v: Param<Tensor<B, 6>>,
        da: Param<Tensor<B, 4>>,
        b: Param<Tensor<B, 6>>,
        c: Param<Tensor<B, 6>>,
        initial_state: Param<Tensor<B, 4>>,
    }

    impl Inputs {
        fn from_inner(
            v: Tensor<InnerB, 6>,
            da: Tensor<InnerB, 4>,
            b: Tensor<InnerB, 6>,
            c: Tensor<InnerB, 6>,
            initial_state: Tensor<InnerB, 4>,
        ) -> Self {
            Self {
                v: Param::from_tensor(Tensor::from_inner(v)),
                da: Param::from_tensor(Tensor::from_inner(da)),
                b: Param::from_tensor(Tensor::from_inner(b)),
                c: Param::from_tensor(Tensor::from_inner(c)),
                initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            }
        }

        fn ssd_input(&self) -> Mamba3SsdInput<B> {
            Mamba3SsdInput {
                v_bnlrhp: self.v.val(),
                da_bnlh: self.da.val(),
                b_bnlrhn: self.b.val(),
                c_bnlrhn: self.c.val(),
                initial_state_bhpr: self.initial_state.val(),
                // Serial paths assert this is None — see ssd_serial / ssd_serial_recalculated.
                init_state_hpr: None,
            }
        }
    }

    /// Collected forward outputs and input gradients for a single SSD path run.
    struct PathRun {
        y: Tensor<InnerB, 6>,
        state: Tensor<InnerB, 4>,
        d_v: Tensor<InnerB, 6>,
        d_da: Tensor<InnerB, 4>,
        d_b: Tensor<InnerB, 6>,
        d_c: Tensor<InnerB, 6>,
        d_init_state: Tensor<InnerB, 4>,
    }

    /// Combine `y` and `final_state` into a single deterministic scalar loss
    /// using fixed (non-tracked) random "head" tensors. Two distinct heads so
    /// that gradients for the y-branch and the state-branch are independent.
    fn loss_from_outputs(
        y_bnlrhp: Tensor<B, 6>,
        final_state_bhpr: Tensor<B, 4>,
        y_head: Tensor<InnerB, 6>,
        s_head: Tensor<InnerB, 4>,
    ) -> Tensor<B, 1> {
        let y_head = Tensor::from_inner(y_head);
        let s_head = Tensor::from_inner(s_head);
        (y_bnlrhp * y_head).sum() + (final_state_bhpr * s_head).sum()
    }

    /// Run a single SSD path and extract the gradients of all 5 inputs.
    fn run_path(
        path: Mamba3SsdPath,
        inputs: &Inputs,
        y_head: Tensor<InnerB, 6>,
        s_head: Tensor<InnerB, 4>,
    ) -> PathRun {
        let (y, state) = path.run(inputs.ssd_input());
        let y_inner = y.clone().inner();
        let state_inner = state.clone().inner();

        let loss = loss_from_outputs(y, state, y_head, s_head);
        let grads = loss.backward();

        PathRun {
            y: y_inner,
            state: state_inner,
            d_v: inputs.v.val().grad(&grads).expect("grad v"),
            d_da: inputs.da.val().grad(&grads).expect("grad da"),
            d_b: inputs.b.val().grad(&grads).expect("grad b"),
            d_c: inputs.c.val().grad(&grads).expect("grad c"),
            d_init_state: inputs
                .initial_state
                .val()
                .grad(&grads)
                .expect("grad initial_state"),
        }
    }

    /// Run the same input through `Minimal`, `Serial`, and `SerialRecalculated`
    /// and assert that all three agree on:
    ///   1. the forward outputs (`y`, `final_state`)
    ///   2. the gradients of every input through a fixed scalar loss.
    ///
    /// All three are chunkwise reformulations of the same MIMO-first SSD, so
    /// both the values and their gradients must agree up to floating-point
    /// noise.
    fn run_minimal_matches_serial(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        mimo_rank: usize,
        nheads: usize,
        per_head_dim: usize,
        state_rank: usize,
    ) {
        let device: Device = Default::default();
        let (v, da, b, c, init) = random_input(
            batch,
            nchunks,
            chunk_len,
            mimo_rank,
            nheads,
            per_head_dim,
            state_rank,
            &device,
        );

        // Fixed (non-tracked) "downstream heads" for the loss.
        let y_head = Tensor::<InnerB, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s_head = Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Each path gets its own fresh autodiff graph (Param leaves).
        let inputs_min =
            Inputs::from_inner(v.clone(), da.clone(), b.clone(), c.clone(), init.clone());
        let inputs_ser =
            Inputs::from_inner(v.clone(), da.clone(), b.clone(), c.clone(), init.clone());
        let inputs_rec = Inputs::from_inner(v, da, b, c, init);

        let r_min = run_path(
            Mamba3SsdPath::Minimal(Some(chunk_len)),
            &inputs_min,
            y_head.clone(),
            s_head.clone(),
        );
        let r_ser = run_path(
            Mamba3SsdPath::Serial(Some(chunk_len)),
            &inputs_ser,
            y_head.clone(),
            s_head.clone(),
        );
        let r_rec = run_path(
            Mamba3SsdPath::SerialRecalculated(Some(chunk_len)),
            &inputs_rec,
            y_head,
            s_head,
        );

        // ── Forward agreement ────────────────────────────────────────────
        let tol = 1e-4;
        let dy_ser = (r_min.y.clone() - r_ser.y.clone())
            .abs()
            .max()
            .into_scalar();
        let ds_ser = (r_min.state.clone() - r_ser.state.clone())
            .abs()
            .max()
            .into_scalar();
        let dy_rec = (r_min.y.clone() - r_rec.y.clone())
            .abs()
            .max()
            .into_scalar();
        let ds_rec = (r_min.state.clone() - r_rec.state.clone())
            .abs()
            .max()
            .into_scalar();
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

        // ── Gradient agreement ───────────────────────────────────────────
        // Looser tolerance: every path computes the same mathematical
        // gradients, but the chunkwise reformulations accumulate sums in
        // different orders, so small drift is expected.
        let grad_tol = 1e-3;

        let mut failures: Vec<String> = Vec::new();
        macro_rules! diff {
            ($a:expr, $b:expr) => {
                ($a.clone() - $b.clone()).abs().max().into_scalar()
            };
        }
        macro_rules! check_grad {
            ($field:ident, $name:expr) => {{
                let d_ser = diff!(r_min.$field, r_ser.$field);
                let d_rec = diff!(r_min.$field, r_rec.$field);
                eprintln!(
                    "grad {:>14} | min↔ser = {:>10.6} | min↔rec = {:>10.6}",
                    $name, d_ser, d_rec
                );
                if d_ser >= grad_tol {
                    failures.push(format!(
                        "Minimal vs Serial: grad of {} max abs diff = {:.6} (tol {})",
                        $name, d_ser, grad_tol
                    ));
                }
                if d_rec >= grad_tol {
                    failures.push(format!(
                        "Minimal vs SerialRecalculated: grad of {} max abs diff = {:.6} (tol {})",
                        $name, d_rec, grad_tol
                    ));
                }
            }};
        }
        check_grad!(d_v, "v");
        check_grad!(d_da, "da");
        check_grad!(d_b, "b");
        check_grad!(d_c, "c");
        check_grad!(d_init_state, "initial_state");

        assert!(
            failures.is_empty(),
            "gradient mismatches:\n  {}",
            failures.join("\n  ")
        );
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
