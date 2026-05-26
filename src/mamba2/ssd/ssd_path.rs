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
    /// See [`Mamba2SsdInput::ssd_minimal`] for more info.
    ///
    /// For training, you may prefer using [`Self::SerialRecalculated`] instead.
    ///
    /// Based on `/mamba_ssm/modules/ssd_minimal.py` from the `state-spaces/mamba` github reference.
    Minimal(Option<usize>),
    /// (Hybrid) Serial SSD.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// See [`Mamba2SsdInput::ssd_serial`] for more info.  
    /// For the backward operation, this relies on autodiff.  
    /// For a custom backwards that saves memory, see [`Self::SerialRecalculated`].
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
    /// See [`Mamba2SsdInput::ssd_serial_recalculated`] for more info.  
    /// Contains a custom backward operation that saves memory.  
    /// For an autodiff backwards, see [`Self::Serial`].
    ///
    /// Based on the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py` from the `state-spaces/mamba`
    /// github reference.
    SerialRecalculated(Option<usize>),
}

/// SSD input.
///
/// All tensors are pre-processed: B/C are already GQA-expanded to per-head.
pub struct Mamba2SsdInput<B: Backend> {
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    pub x_bnlhp: Tensor<B, 5>,
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub dt_bnlh: Tensor<B, 4>,
    /// # Shape
    /// - `[nheads]`
    pub a_decay_h: Tensor<B, 1>,
    /// B tensor, expanded to per-head.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, state_rank]`
    pub b_bnlhr: Tensor<B, 5>,
    /// C tensor, expanded to per-head.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, state_rank]`
    pub c_bnlhr: Tensor<B, 5>,
    /// # Shape
    /// - `[nheads]`
    pub d_h: Tensor<B, 1>,
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<B, 4>,
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<B, 3>>,
}

impl<B: Backend> Mamba2SsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.x_bnlhp);
        san(&self.dt_bnlh);
        san(&self.a_decay_h);
        san(&self.b_bnlhr);
        san(&self.c_bnlhr);
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
    /// See [`Self::optimal_default`] for more info.
    pub fn core_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Minimal(Some(optim))
    }

    /// Optimal Minimal variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::core_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Serial(Some(optim))
    }

    /// Optimal Serial variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_recalculated_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::SerialRecalculated(Some(optim))
    }

    /// Optimal Serial Recalculated variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_recalculated_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_recalculated_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Mamba2SsdPath::Minimal(chunk_len)
            | Mamba2SsdPath::Serial(chunk_len)
            | Mamba2SsdPath::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len()
            .unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
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
            Mamba2SsdPath::Minimal(_) => input.ssd_minimal(),
            Mamba2SsdPath::Serial(_) => input.ssd_serial(),
            Mamba2SsdPath::SerialRecalculated(_) => input.ssd_serial_recalculated(),
        }
    }
}

impl Default for Mamba2SsdPath {
    fn default() -> Mamba2SsdPath {
        // SSD Path defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba2SsdPath::SerialRecalculated(None)
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

    /// Build a randomised set of tensors on the inner backend (no grad
    /// tracking yet — `Param::from_tensor` is applied per-path below to
    /// give each path its own fresh autodiff graph).
    ///
    /// `dt` is drawn from a positive distribution (softplus-like) and `a_decay`
    /// from a negative range so that the implied per-token decay `exp(dt·a)`
    /// stays in `(0, 1]`, matching how the upstream block produces them.
    #[allow(clippy::too_many_arguments)]
    fn random_input(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        nheads: usize,
        per_head_dim: usize,
        state_rank: usize,
        random_init: bool,
        device: &Device,
    ) -> (
        Tensor<InnerB, 5>,
        Tensor<InnerB, 4>,
        Tensor<InnerB, 1>,
        Tensor<InnerB, 5>,
        Tensor<InnerB, 5>,
        Tensor<InnerB, 1>,
        Tensor<InnerB, 4>,
    ) {
        let x = Tensor::<InnerB, 5>::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let dt = Tensor::<InnerB, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Uniform(0.05, 0.3),
            device,
        );
        let a_decay =
            Tensor::<InnerB, 1>::random([nheads], Distribution::Uniform(-1.0, -0.5), device);
        let b = Tensor::<InnerB, 5>::random(
            [batch, nchunks, chunk_len, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let c = Tensor::<InnerB, 5>::random(
            [batch, nchunks, chunk_len, nheads, state_rank],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let d = Tensor::<InnerB, 1>::random([nheads], Distribution::Normal(0.0, 0.1), device);
        // The initial SSM state is random (the general case) or zero (the
        // standard fresh-start case) per `random_init`. All paths must agree on
        // both, so the comparison spans the whole {zero, random} dimension.
        let initial_state = if random_init {
            Tensor::<InnerB, 4>::random(
                [batch, nheads, per_head_dim, state_rank],
                Distribution::Normal(0.0, 0.1),
                device,
            )
        } else {
            Tensor::<InnerB, 4>::zeros([batch, nheads, per_head_dim, state_rank], device)
        };
        (x, dt, a_decay, b, c, d, initial_state)
    }

    /// Inputs wrapped as `Param`s so each tensor becomes an autodiff leaf
    /// with `require_grad`. One `Inputs` is built per path, sharing the same
    /// underlying inner values but its own autodiff graph.
    struct Inputs {
        x: Param<Tensor<B, 5>>,
        dt: Param<Tensor<B, 4>>,
        a_decay: Param<Tensor<B, 1>>,
        b: Param<Tensor<B, 5>>,
        c: Param<Tensor<B, 5>>,
        d: Param<Tensor<B, 1>>,
        initial_state: Param<Tensor<B, 4>>,
    }

    impl Inputs {
        #[allow(clippy::too_many_arguments)]
        fn from_inner(
            x: Tensor<InnerB, 5>,
            dt: Tensor<InnerB, 4>,
            a_decay: Tensor<InnerB, 1>,
            b: Tensor<InnerB, 5>,
            c: Tensor<InnerB, 5>,
            d: Tensor<InnerB, 1>,
            initial_state: Tensor<InnerB, 4>,
        ) -> Self {
            Self {
                x: Param::from_tensor(Tensor::from_inner(x)),
                dt: Param::from_tensor(Tensor::from_inner(dt)),
                a_decay: Param::from_tensor(Tensor::from_inner(a_decay)),
                b: Param::from_tensor(Tensor::from_inner(b)),
                c: Param::from_tensor(Tensor::from_inner(c)),
                d: Param::from_tensor(Tensor::from_inner(d)),
                initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            }
        }

        fn ssd_input(&self) -> Mamba2SsdInput<B> {
            Mamba2SsdInput {
                x_bnlhp: self.x.val(),
                dt_bnlh: self.dt.val(),
                a_decay_h: self.a_decay.val(),
                b_bnlhr: self.b.val(),
                c_bnlhr: self.c.val(),
                d_h: self.d.val(),
                initial_state_bhpr: self.initial_state.val(),
                // Serial paths assert this is None — see ssd_serial / ssd_serial_recalculated.
                init_state_hpr: None,
            }
        }
    }

    /// Collected forward outputs and input gradients for a single SSD path run.
    struct PathRun {
        y: Tensor<InnerB, 5>,
        state: Tensor<InnerB, 4>,
        d_x: Tensor<InnerB, 5>,
        d_dt: Tensor<InnerB, 4>,
        d_a_decay: Tensor<InnerB, 1>,
        d_b: Tensor<InnerB, 5>,
        d_c: Tensor<InnerB, 5>,
        d_d: Tensor<InnerB, 1>,
        d_init_state: Tensor<InnerB, 4>,
    }

    /// Combine `y` and `final_state` into a single deterministic scalar loss
    /// using fixed (non-tracked) random "head" tensors. The two heads differ so
    /// that gradients for the y-branch and the state-branch are independent
    /// (a mistake in either path shows up in the parameter grads).
    fn loss_from_outputs(
        y_bnlhp: Tensor<B, 5>,
        final_state_bhpr: Tensor<B, 4>,
        y_head: Tensor<InnerB, 5>,
        s_head: Tensor<InnerB, 4>,
    ) -> Tensor<B, 1> {
        let y_head = Tensor::from_inner(y_head);
        let s_head = Tensor::from_inner(s_head);
        (y_bnlhp * y_head).sum() + (final_state_bhpr * s_head).sum()
    }

    /// Run a single SSD path and extract the gradients of all 7 inputs.
    fn run_path(
        path: Mamba2SsdPath,
        inputs: &Inputs,
        y_head: Tensor<InnerB, 5>,
        s_head: Tensor<InnerB, 4>,
    ) -> PathRun {
        let (y, state) = path.run(inputs.ssd_input());
        let y_inner = y.clone().inner();
        let state_inner = state.clone().inner();

        let loss = loss_from_outputs(y, state, y_head, s_head);
        let grads = loss.backward();

        // Inline grad extraction (a closure cannot be reused here since the
        // gradient tensor rank varies per call).
        PathRun {
            y: y_inner,
            state: state_inner,
            d_x: inputs.x.val().grad(&grads).expect("grad x"),
            d_dt: inputs.dt.val().grad(&grads).expect("grad dt"),
            d_a_decay: inputs.a_decay.val().grad(&grads).expect("grad a_decay"),
            d_b: inputs.b.val().grad(&grads).expect("grad b"),
            d_c: inputs.c.val().grad(&grads).expect("grad c"),
            d_d: inputs.d.val().grad(&grads).expect("grad d"),
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
    /// All three are chunkwise reformulations of the same SSD, so both the
    /// values and their gradients must agree up to floating-point noise.
    #[allow(clippy::too_many_arguments)]
    fn run_minimal_matches_serial(
        batch: usize,
        nchunks: usize,
        chunk_len: usize,
        nheads: usize,
        per_head_dim: usize,
        state_rank: usize,
        random_init: bool,
    ) {
        let device: Device = Default::default();
        let (x, dt, a_decay, b, c, d, init) = random_input(
            batch,
            nchunks,
            chunk_len,
            nheads,
            per_head_dim,
            state_rank,
            random_init,
            &device,
        );

        // Fixed (non-tracked) "downstream heads" for the loss. Two distinct
        // random tensors so y- and state-gradient paths are exercised
        // independently.
        let y_head = Tensor::<InnerB, 5>::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s_head = Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Each path gets its own fresh autodiff graph (Param leaves).
        let inputs_min = Inputs::from_inner(
            x.clone(),
            dt.clone(),
            a_decay.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            init.clone(),
        );
        let inputs_ser = Inputs::from_inner(
            x.clone(),
            dt.clone(),
            a_decay.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            init.clone(),
        );
        let inputs_rec = Inputs::from_inner(x, dt, a_decay, b, c, d, init);

        let r_min = run_path(
            Mamba2SsdPath::Minimal(Some(chunk_len)),
            &inputs_min,
            y_head.clone(),
            s_head.clone(),
        );
        let r_ser = run_path(
            Mamba2SsdPath::Serial(Some(chunk_len)),
            &inputs_ser,
            y_head.clone(),
            s_head.clone(),
        );
        let r_rec = run_path(
            Mamba2SsdPath::SerialRecalculated(Some(chunk_len)),
            &inputs_rec,
            y_head,
            s_head,
        );

        // ── Forward agreement ────────────────────────────────────────────
        use crate::utils::test_helpers::max_abs_diff;
        let tol = 1e-4f32;
        let dy_ser = max_abs_diff(r_min.y.clone(), r_ser.y.clone());
        let ds_ser = max_abs_diff(r_min.state.clone(), r_ser.state.clone());
        let dy_rec = max_abs_diff(r_min.y.clone(), r_rec.y.clone());
        let ds_rec = max_abs_diff(r_min.state.clone(), r_rec.state.clone());
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
        // gradients, but the chunkwise reformulations accumulate the
        // summations in different orders, so small drift is expected.
        crate::check_grads_match_two_paths!(
            baseline: r_min,
            alt1: ("Serial", r_ser),
            alt2: ("SerialRecalculated", r_rec),
            tol: 1e-3,
            fields: [
                d_x => "x",
                d_dt => "dt",
                d_a_decay => "a_decay",
                d_b => "b",
                d_c => "c",
                d_d => "d",
                d_init_state => "initial_state",
            ],
        );
    }

    #[test]
    fn paths_agree() {
        // B/C are already per-head (GQA expansion happens before constructing the SsdInput).
        run_minimal_matches_serial(2, 3, 4, 2, 8, 8, true);
    }

    #[test]
    fn paths_agree_zero_init() {
        run_minimal_matches_serial(2, 3, 4, 2, 8, 8, false);
    }

    #[test]
    fn paths_agree_more_heads() {
        // Slightly larger nheads to vary the shape.
        run_minimal_matches_serial(2, 3, 4, 4, 8, 8, true);
    }

    #[test]
    fn paths_agree_more_heads_zero_init() {
        run_minimal_matches_serial(2, 3, 4, 4, 8, 8, false);
    }

    #[test]
    fn paths_agree_single_chunk() {
        // nchunks=1 — no inter-chunk scan; checks the intra-chunk + state-passing
        // boundary case where K4 runs a single iteration.
        run_minimal_matches_serial(2, 1, 4, 2, 8, 8, true);
    }

    #[test]
    fn paths_agree_single_chunk_zero_init() {
        run_minimal_matches_serial(2, 1, 4, 2, 8, 8, false);
    }
}
