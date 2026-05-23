//! # Trapezoidal-Merged (Single-Pass) SSD — Path Dispatcher
//!
//! Sibling to [`crate::mamba3::ssd::ssd_path::Mamba3SsdPath`]. Where the existing
//! [`Mamba3SsdPath`] runs the *standard* SSD twice (γ-term and β-term), this
//! module's [`Mamba3TrapSsdPath`] runs **one** merged SSD pass that absorbs both
//! contributions by scaling `K` with `scaleₜ = γₜ + (1−λₜ₊₁) Δₜ₊₁`. The same-step
//! diagonal contribution differs (it must use `γₜ`, not `scaleₜ`) and is patched
//! via an explicit correction term inside each variant.
//!
//! Reference kernels:
//! - `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`
//!
//! The interface is MIMO-first (matches the existing burn-mamba SSD inputs),
//! with `R = 1` collapsing to the SISO case.

use crate::mamba3::prelude::*;
use burn::prelude::*;

/// Algorithm selection for the trapezoidal-merged (single-pass) SSD.
///
/// Mirrors [`Mamba3SsdPath`] but each variant computes the trapezoidal
/// recurrence with **one** chunkwise pass.
#[derive(Debug, Clone)]
pub enum Mamba3TrapSsdPath {
    /// Minimal/segsum variant.
    ///
    /// Mostly batched matmuls; the backward pass relies on autodiff. The
    /// algorithm is the merged-form analogue of [`Mamba3SsdPath::Minimal`]:
    /// strict lower-triangular intra-chunk (excludes same-step block),
    /// a separate γ-scaled diagonal correction, and a state recurrence using
    /// the `scaleₜ`-scaled K. See [`Mamba3TrapSsdInput::ssd_trap_minimal`].
    Minimal(Option<usize>),

    /// (Hybrid) Serial variant — chunk-serial K1–K5 reformulation.
    ///
    /// Reuses K1–K4 from [`crate::mamba3::ssd::serial`] and supplies a new K5
    /// that does strict-lower intra-chunk + per-column `scaleₜ` + γ-weighted
    /// same-step diagonal correction. The state passing loop is sequential,
    /// matching [`Mamba3SsdPath::Serial`]. See
    /// [`Mamba3TrapSsdInput::ssd_trap_serial`].
    Serial(Option<usize>),
}

/// MIMO-first input bundle for the merged-form SSD.
///
/// All tensors are pre-processed by the caller (`Mamba3::forward2`): B/C are
/// already QK-normed, RoPE-applied, bias-added, and expanded to per-head; V is
/// the raw, *unscaled* MIMO-expanded value. The combined log-decay `da = Δ·A`
/// is pre-computed. The two trapezoidal coefficients `gammaₜ` and `scaleₜ` are
/// supplied separately because the SSD itself does the K-scaling and γ-weighted
/// diagonal correction internally. D-skip and Z-gating are handled by the
/// caller.
pub struct Mamba3TrapSsdInput<B: Backend> {
    /// Value tensor, MIMO-expanded but **not** trapezoidally scaled.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<B, 6>,

    /// K/B tensor: QK-normed, RoPE-applied, bias-added, expanded to per-head.
    /// Not pre-scaled — the SSD multiplies by `scaleₜ` internally for the
    /// lower-triangular and state-recurrence paths, while the diagonal
    /// correction reuses the unscaled tensor.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<B, 6>,

    /// Q/C tensor: same processing as `b_bnlmhr`.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub c_bnlmhr: Tensor<B, 6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<B, 4>,

    /// `γₜ = λₜ · Δₜ` — used as the per-token diagonal multiplier.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub gamma_bnlh: Tensor<B, 4>,

    /// `scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁` — K is multiplied by this for the
    /// lower-triangular and state recurrence paths. The shifted term is zero
    /// at the very last sequence position (no future token exists).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub scale_bnlh: Tensor<B, 4>,

    /// Initial SSM hidden state (merged-form accumulator).
    ///
    /// When continuing from a prior call, this should already include the
    /// boundary β contribution `(1 − λ₀) · Δ₀ · Σₘ Kₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_xₘ)`
    /// (which the previous call could not yet add because it did not know
    /// `λ₀, Δ₀`).
    ///
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<B, 4>,

    /// Optional learnable initial state (broadcast over batch).
    ///
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<B, 3>>,
}

impl<B: Backend> Mamba3TrapSsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.v_bnlmhp);
        san(&self.b_bnlmhr);
        san(&self.c_bnlmhr);
        san(&self.da_bnlh);
        san(&self.gamma_bnlh);
        san(&self.scale_bnlh);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl Mamba3TrapSsdPath {
    /// Optimal chunk length — same heuristic as [`Mamba3SsdPath::optimal_default`].
    pub fn optimal_default(state_rank: usize, per_head_dim: usize) -> usize {
        Mamba3SsdPath::optimal_default(state_rank, per_head_dim)
    }

    /// Optimal Minimal variant.
    pub fn core_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Minimal(Some(optim))
    }

    /// Optimal Minimal variant from a block.
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
        Self::core_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    pub fn chunked_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Serial(Some(optim))
    }

    /// Optimal Serial variant from a block.
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
        Self::chunked_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Mamba3TrapSsdPath::Minimal(chunk_len) | Mamba3TrapSsdPath::Serial(chunk_len) => {
                *chunk_len
            }
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len()
            .unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
    }

    /// Run the merged-form SSD on the given MIMO-first input.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]` —
    ///   the merged-form accumulator at the last token (to be stored in the
    ///   cache for streaming).
    pub fn run<B: Backend>(&self, input: Mamba3TrapSsdInput<B>) -> (Tensor<B, 6>, Tensor<B, 4>) {
        match self {
            Mamba3TrapSsdPath::Minimal(_) => input.ssd_trap_minimal(),
            Mamba3TrapSsdPath::Serial(_) => input.ssd_trap_serial(),
        }
    }
}

impl Default for Mamba3TrapSsdPath {
    fn default() -> Mamba3TrapSsdPath {
        Mamba3TrapSsdPath::Minimal(None)
    }
}

// ---------------------------------------------------------------------------
// Tests — Minimal ≡ Serial (forward outputs + input gradients)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::module::Param;
    use burn::tensor::Distribution;

    type InnerB = Flex;
    type B = Autodiff<InnerB>;
    type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

    /// Random inputs for the trapezoidal-merged SSD. `da` is drawn from a
    /// negative-mean distribution so the implied per-token decay `exp(da)`
    /// stays in `(0, 1]`. `gamma` and `scale` are non-negative (matching
    /// `Δ·σ(λ)`-style outputs of `helpers::trapezoidal_coefficients`).
    #[allow(clippy::too_many_arguments)]
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
        Tensor<InnerB, 6>, // v
        Tensor<InnerB, 6>, // b
        Tensor<InnerB, 6>, // c
        Tensor<InnerB, 4>, // da
        Tensor<InnerB, 4>, // gamma
        Tensor<InnerB, 4>, // scale
        Tensor<InnerB, 4>, // initial_state
    ) {
        let v = Tensor::<InnerB, 6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
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
        let da = Tensor::<InnerB, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Normal(-0.5, 0.1),
            device,
        );
        let gamma = Tensor::<InnerB, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Uniform(0.05, 0.5),
            device,
        );
        let scale = Tensor::<InnerB, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Uniform(0.05, 0.5),
            device,
        );
        let initial_state = Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        );
        (v, b, c, da, gamma, scale, initial_state)
    }

    struct Inputs {
        v: Param<Tensor<B, 6>>,
        b: Param<Tensor<B, 6>>,
        c: Param<Tensor<B, 6>>,
        da: Param<Tensor<B, 4>>,
        gamma: Param<Tensor<B, 4>>,
        scale: Param<Tensor<B, 4>>,
        initial_state: Param<Tensor<B, 4>>,
    }

    impl Inputs {
        #[allow(clippy::too_many_arguments)]
        fn from_inner(
            v: Tensor<InnerB, 6>,
            b: Tensor<InnerB, 6>,
            c: Tensor<InnerB, 6>,
            da: Tensor<InnerB, 4>,
            gamma: Tensor<InnerB, 4>,
            scale: Tensor<InnerB, 4>,
            initial_state: Tensor<InnerB, 4>,
        ) -> Self {
            Self {
                v: Param::from_tensor(Tensor::from_inner(v)),
                b: Param::from_tensor(Tensor::from_inner(b)),
                c: Param::from_tensor(Tensor::from_inner(c)),
                da: Param::from_tensor(Tensor::from_inner(da)),
                gamma: Param::from_tensor(Tensor::from_inner(gamma)),
                scale: Param::from_tensor(Tensor::from_inner(scale)),
                initial_state: Param::from_tensor(Tensor::from_inner(initial_state)),
            }
        }

        fn ssd_input(&self) -> Mamba3TrapSsdInput<B> {
            Mamba3TrapSsdInput {
                v_bnlmhp: self.v.val(),
                b_bnlmhr: self.b.val(),
                c_bnlmhr: self.c.val(),
                da_bnlh: self.da.val(),
                gamma_bnlh: self.gamma.val(),
                scale_bnlh: self.scale.val(),
                initial_state_bhpr: self.initial_state.val(),
                // Serial asserts this is None — see ssd_trap_serial.
                init_state_hpr: None,
            }
        }
    }

    struct PathRun {
        y: Tensor<InnerB, 6>,
        state: Tensor<InnerB, 4>,
        d_v: Tensor<InnerB, 6>,
        d_b: Tensor<InnerB, 6>,
        d_c: Tensor<InnerB, 6>,
        d_da: Tensor<InnerB, 4>,
        d_gamma: Tensor<InnerB, 4>,
        d_scale: Tensor<InnerB, 4>,
        d_init_state: Tensor<InnerB, 4>,
    }

    fn loss_from_outputs(
        y_bnlmhp: Tensor<B, 6>,
        final_state_bhpr: Tensor<B, 4>,
        y_head: Tensor<InnerB, 6>,
        s_head: Tensor<InnerB, 4>,
    ) -> Tensor<B, 1> {
        let y_head = Tensor::from_inner(y_head);
        let s_head = Tensor::from_inner(s_head);
        (y_bnlmhp * y_head).sum() + (final_state_bhpr * s_head).sum()
    }

    fn run_path(
        path: Mamba3TrapSsdPath,
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
            d_b: inputs.b.val().grad(&grads).expect("grad b"),
            d_c: inputs.c.val().grad(&grads).expect("grad c"),
            d_da: inputs.da.val().grad(&grads).expect("grad da"),
            d_gamma: inputs.gamma.val().grad(&grads).expect("grad gamma"),
            d_scale: inputs.scale.val().grad(&grads).expect("grad scale"),
            d_init_state: inputs
                .initial_state
                .val()
                .grad(&grads)
                .expect("grad initial_state"),
        }
    }

    fn assert_path_runs_agree(label: &str, a: &PathRun, b: &PathRun, val_tol: f32, grad_tol: f32) {
        use crate::utils::test_helpers::max_abs_diff;
        let mut failures: Vec<String> = Vec::new();
        macro_rules! check_inner {
            ($field:ident, $name:expr, $tol:expr) => {{
                let d = max_abs_diff(a.$field.clone(), b.$field.clone());
                eprintln!("{:>22} {:>14} | max abs diff = {:>10.6}", label, $name, d);
                if d >= $tol {
                    failures.push(format!(
                        "{}: {} max abs diff = {:.6} (tol {})",
                        label, $name, d, $tol
                    ));
                }
            }};
        }
        check_inner!(y, "y", val_tol);
        check_inner!(state, "final_state", val_tol);
        check_inner!(d_v, "grad v", grad_tol);
        check_inner!(d_b, "grad b", grad_tol);
        check_inner!(d_c, "grad c", grad_tol);
        check_inner!(d_da, "grad da", grad_tol);
        check_inner!(d_gamma, "grad gamma", grad_tol);
        check_inner!(d_scale, "grad scale", grad_tol);
        check_inner!(d_init_state, "grad init_state", grad_tol);
        assert!(
            failures.is_empty(),
            "trap-path mismatches:\n  {}",
            failures.join("\n  ")
        );
    }

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
        let (v, b, c, da, gamma, scale, init) = random_input(
            batch,
            nchunks,
            chunk_len,
            mimo_rank,
            nheads,
            per_head_dim,
            state_rank,
            &device,
        );

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

        let inputs_min = Inputs::from_inner(
            v.clone(),
            b.clone(),
            c.clone(),
            da.clone(),
            gamma.clone(),
            scale.clone(),
            init.clone(),
        );
        let inputs_ser = Inputs::from_inner(v, b, c, da, gamma, scale, init);

        let r_min = run_path(
            Mamba3TrapSsdPath::Minimal(Some(chunk_len)),
            &inputs_min,
            y_head.clone(),
            s_head.clone(),
        );
        let r_ser = run_path(
            Mamba3TrapSsdPath::Serial(Some(chunk_len)),
            &inputs_ser,
            y_head,
            s_head,
        );

        // Same algorithm, different schedule: stricter on values (1e-4),
        // moderate on gradients (1e-3) — same tolerances as the original-form
        // SSD-path agreement tests.
        assert_path_runs_agree("Minimal vs Serial", &r_min, &r_ser, 1e-4, 1e-3);
    }

    #[test]
    fn trap_paths_agree_siso() {
        run_minimal_matches_serial(2, 3, 4, 1, 2, 8, 8);
    }

    #[test]
    fn trap_paths_agree_mimo() {
        run_minimal_matches_serial(2, 3, 4, 2, 2, 8, 8);
    }

    #[test]
    fn trap_paths_agree_single_chunk() {
        run_minimal_matches_serial(2, 1, 4, 1, 2, 8, 8);
    }
}
