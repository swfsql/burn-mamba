//! State participation-ratio (PR) diagnostics — the point of the experiment.
//!
//! `PR(Σ) = (tr Σ)² / tr(Σ²)`, range `1…N`: the effective rank of a sample
//! covariance, computed from the two traces only (no eigendecomposition, no
//! degenerate-eigenvalue gradient issues, rotation-invariant, and invariant to
//! uniform rescaling — weight decay shrinking all norms cannot move it).
//!
//! The primary measure is the **N-side state PR**: per layer and head, the
//! recurrent states `ssm_bhpr` collected over (batch, step, channel `p`) are
//! treated as samples of `state_rank`-vectors. Within a head every channel's
//! state lies in `span{B_τ}`, so this reads as "how many distinct write
//! directions does the model use" — the Fourier-circuit-sized quantity. The
//! hypothesis: memorization keeps it near its ceiling, generalization
//! collapses it to ≈ 2×(#frequencies).
//!
//! Secondary weight-side measures (embedding / LM-head spectral PR, and the
//! embedding's *exact* `p`-periodic Fourier-energy PR — `rfft` is unusable
//! here as it needs power-of-two lengths) make a null state-PR result
//! interpretable: a lookup table can hide in projections without touching
//! state rank.

use burn::prelude::*;
use burn_mamba::prelude::*;

/// N-side PR of one layer/head's collected states, in all four read-outs.
/// Centered (the mean state subtracted) is primary — a large shared mean
/// direction drags uncentered PR toward 1; pooled (all steps) vs final-step
/// answer subtly different questions (accumulation vs read-out state).
pub struct StatePr {
    /// (Virtual) layer index.
    pub layer: usize,
    /// Head index.
    pub head: usize,
    /// All steps pooled, centered.
    pub pooled_centered: f64,
    /// All steps pooled, uncentered.
    pub pooled_uncentered: f64,
    /// Final step only, centered.
    pub final_centered: f64,
    /// Final step only, uncentered.
    pub final_uncentered: f64,
    /// Raw pooled state magnitude `tr Σ = ⟨‖h‖²⟩` (all steps, uncentered) —
    /// PR's numerator scale. When this collapses toward 0 the clamped pooled
    /// PR reads below 1 (magnitude collapse, not a clean rank-1 state); under
    /// strong weight decay it falls regardless of the state penalty.
    pub pooled_trace: f64,
    /// Raw final-step state magnitude `tr Σ = ⟨‖h‖²⟩` (uncentered).
    pub final_trace: f64,
}

/// Weight-side effective ranks (same PR formula on `WᵀW`'s spectrum — the
/// `1/samples` factor cancels in the ratio).
pub struct WeightPr {
    /// Spectral PR of the embedding table.
    pub emb: f64,
    /// Spectral PR of the (untied) LM head; `NaN` when the head is tied.
    pub lm_head: f64,
    /// PR over the embedding's `p`-periodic Fourier energies (DC excluded):
    /// the effective number of active frequencies, ~5–6 for the known
    /// transformer circuit.
    pub emb_freq: f64,
    /// Per-layer block-weight PRs.
    pub layers: Vec<LayerWeightPr>,
}

/// Spectral PRs of one block's weights: each `in_proj` slice (both families
/// lead with `[z | x | B | C | …]`; the per-head tail columns are skipped),
/// `out_proj`, and the token-centered B-alphabet.
pub struct LayerWeightPr {
    /// Real-layer index.
    pub layer: usize,
    /// `in_proj` gate slice `z`.
    pub z: f64,
    /// `in_proj` value slice `x`.
    pub x: f64,
    /// `in_proj` write-key slice `B`.
    pub b: f64,
    /// `in_proj` read-key slice `C`.
    pub c: f64,
    /// `out_proj`.
    pub out: f64,
    /// PR of the rows of `emb·W_B` **centered across tokens** — the
    /// write-alphabet differentiation with the shared (token-independent) DC
    /// component removed. This is the confound-free companion to the state
    /// PR, whose samples carry the DC (it is multiplied by the sign-varying
    /// per-channel scalar and cannot be centered away there).
    pub b_alphabet: f64,
}

/// How the network's `state_rank` axis pairs into complex/quaternionic
/// coordinates: [`StatePairing::Real`] for Mamba-1/2, the first block's
/// [`Mamba3::state_pairing`] for Mamba-3 (every layer shares the config).
/// Selects the Hermitian PR (`PR_ℂ(M_phys)`) for the Mamba-3 diagnostics and
/// penalty.
pub fn state_pairing_of(model: &MambaVocabNet) -> StatePairing {
    match model {
        MambaVocabNet::Mamba3(net) => net.layers.real_layers[0].mamba_block.state_pairing(),
        _ => StatePairing::Real,
    }
}

/// Pairing-aware [`pr`] of a sample cloud: real PR for [`StatePairing::Real`]
/// (byte-identical to the Mamba-2 read-out), the Hermitian `PR_ℂ` otherwise
/// (via the library's [`StateMoments::pr_complex`] recombination).
fn pr_state(h_sn: Tensor<2>, pairing: &StatePairing, center: bool) -> f64 {
    if matches!(pairing, StatePairing::Real) {
        return pr(h_sn, center);
    }
    let [samples, state_rank] = h_sn.dims();
    let moments = StateMoments {
        m2_bhrr: h_sn
            .clone()
            .transpose()
            .matmul(h_sn.clone())
            .reshape([1, 1, state_rank, state_rank]),
        m1_bhr: h_sn.sum_dim(0).reshape([1, 1, state_rank]),
        count: samples,
    };
    scalar_f64(moments.pr_complex(pairing, center).reshape([1]))
}

/// The **physical-frame** per-layer states of a Mamba-3 cache bundle (the
/// cache state de-rotated by the cumulative rotation — what raw C reads and
/// what the forward moments pool).
fn mamba3_physical_states(
    caches: &Mamba3Caches,
    rope_dim: usize,
    rotate_pairwise: bool,
) -> Vec<Tensor<4>> {
    let derotate = |ssm: &Tensor<4>, rotation: &RotationState| {
        rotation.derotate_state(ssm.clone(), rope_dim, rotate_pairwise)
    };
    match caches {
        Mamba3Caches::SingleSsd(cs) => cs
            .caches
            .iter()
            .map(|l| derotate(&l.ssm_bhpr, &l.rotation))
            .collect(),
        Mamba3Caches::DoubleSsd(cs) => cs
            .caches
            .iter()
            .map(|l| derotate(&l.ssm_bhpr, &l.rotation))
            .collect(),
    }
}

/// Run `inputs_bs` `[n, s]` through the model token-by-token on the plain
/// backend, reading every layer's per-step state — `ssm_bhpr` for Mamba-2,
/// its **physical-frame** de-rotation for Mamba-3 — and return the N-side PR
/// per (layer, head) (`PR_ℂ` for Mamba-3, per [`state_pairing_of`]).
pub fn state_pr(model: &MambaVocabNet, inputs_bs: &Tensor<2, Int>) -> Vec<StatePr> {
    let [_n, s] = inputs_bs.dims();
    let pairing = state_pairing_of(model);
    let mamba3_meta = match model {
        MambaVocabNet::Mamba3(net) => {
            let block = &net.layers.real_layers[0].mamba_block;
            Some((block.rope_dim, block.mimo_rank == 1))
        }
        _ => None,
    };
    let mut caches = None;
    // per_step[t][layer]: `[batch, nheads, per_head_dim, state_rank]`
    let mut per_step: Vec<Vec<Tensor<4>>> = Vec::with_capacity(s);
    for t in 0..s {
        let x_b = inputs_bs.clone().narrow(1, t, 1).squeeze_dim::<1>(1);
        let (_logits, new_caches) = model.step(x_b, caches, None, None);
        let states = match &new_caches {
            MambaCaches::Mamba2(c) => c.caches.iter().map(|l| l.ssm_bhpr.clone()).collect(),
            MambaCaches::Mamba3(c) => {
                let (rope_dim, pairwise) = mamba3_meta.expect("Mamba-3 model");
                mamba3_physical_states(c, rope_dim, pairwise)
            }
            _ => panic!("the state-PR diagnostic expects a Mamba-2 or Mamba-3 network"),
        };
        per_step.push(states);
        caches = Some(new_caches);
    }

    let n_layers = per_step[0].len();
    let [_b, nheads, _p, _r] = per_step[0][0].dims();
    let mut out = Vec::with_capacity(n_layers * nheads);
    for layer in 0..n_layers {
        for head in 0..nheads {
            // Each step's samples: channels stacked over the batch, `[b·p, r]`.
            let step_samples: Vec<Tensor<2>> = per_step
                .iter()
                .map(|states| {
                    let bhpr = states[layer].clone();
                    let [b, _h, p, r] = bhpr.dims();
                    bhpr.narrow(1, head, 1).reshape([b * p, r])
                })
                .collect();
            let final_sn = step_samples.last().expect("at least one step").clone();
            let pooled_sn = Tensor::cat(step_samples, 0);
            out.push(StatePr {
                layer,
                head,
                pooled_centered: pr_state(pooled_sn.clone(), &pairing, true),
                pooled_uncentered: pr_state(pooled_sn.clone(), &pairing, false),
                final_centered: pr_state(final_sn.clone(), &pairing, true),
                final_uncentered: pr_state(final_sn.clone(), &pairing, false),
                pooled_trace: trace(pooled_sn),
                final_trace: trace(final_sn),
            });
        }
    }
    out
}

/// Forward-path counterpart of [`state_pr`]: one chunkwise
/// `forward_with_state_moments` call instead of `s` `step`s, so it stays
/// cheap when stepwise evaluation is memory-heavy. Pooled PRs come from the
/// library's closed-form per-layer state moments (exact — every per-token
/// state is counted, none materialised); final-step PRs from the returned
/// caches. Matches [`state_pr`] by the moments parity tests.
pub fn state_pr_forward(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    ssd_path: MambaSsdPath,
) -> Vec<StatePr> {
    let pairing = state_pairing_of(model);
    let (_logits, caches, moments) =
        model.forward_with_state_moments(inputs_bs.clone(), None, ssd_path);
    let final_states: Vec<Tensor<4>> = match (&caches, model) {
        (MambaCaches::Mamba2(c), _) => c.caches.iter().map(|l| l.ssm_bhpr.clone()).collect(),
        (MambaCaches::Mamba3(c), MambaVocabNet::Mamba3(net)) => {
            let block = &net.layers.real_layers[0].mamba_block;
            mamba3_physical_states(c, block.rope_dim, block.mimo_rank == 1)
        }
        _ => panic!("the state-PR diagnostic expects a Mamba-2 or Mamba-3 network"),
    };

    let mut out = Vec::new();
    for (layer, (m, final_bhpr)) in moments.into_iter().zip(final_states).enumerate() {
        let pooled = m.pool_batch();
        let per_head = |t: Tensor<2>| t.into_data().to_vec::<f32>().unwrap();
        // `pr_complex(Real)` delegates to `pr`, so this is family-uniform.
        let centered_h = per_head(pooled.pr_complex(&pairing, true));
        let uncentered_h = per_head(pooled.pr_complex(&pairing, false));
        let trace_h = per_head(pooled.trace());
        let [b, nheads, p, r] = final_bhpr.dims();
        for head in 0..nheads {
            let final_sn = final_bhpr.clone().narrow(1, head, 1).reshape([b * p, r]);
            out.push(StatePr {
                layer,
                head,
                pooled_centered: centered_h[head] as f64,
                pooled_uncentered: uncentered_h[head] as f64,
                final_centered: pr_state(final_sn.clone(), &pairing, true),
                final_uncentered: pr_state(final_sn.clone(), &pairing, false),
                pooled_trace: trace_h[head] as f64,
                final_trace: trace(final_sn),
            });
        }
    }
    out
}

/// Weight-side PRs: embedding, LM head, the embedding's exact `p`-point
/// Fourier-energy PR (only the first `p` rows — the vocab may be padded),
/// and each block's per-slice weight PRs.
pub fn weight_pr(model: &MambaVocabNet, p: usize) -> WeightPr {
    let (emb_vd, lm_head_w) = family_emb_head(model);
    let emb = pr(emb_vd.clone(), false);
    let lm_head = match lm_head_w {
        Some(w) => pr(w, false),
        None => f64::NAN,
    };
    let emb_pd = emb_vd.narrow(0, 0, p);
    let emb_freq = pr_of_energies(&dft_energy(emb_pd.clone()));

    let layers = family_block_weights(model)
        .into_iter()
        .enumerate()
        .map(|(layer, bw)| LayerWeightPr {
            layer,
            z: pr(bw.z, false),
            x: pr(bw.x, false),
            b: pr(bw.b.clone(), false),
            c: pr(bw.c, false),
            out: pr(bw.out, false),
            b_alphabet: pr(emb_pd.clone().matmul(bw.b), true),
        })
        .collect();

    WeightPr { emb, lm_head, emb_freq, layers }
}

/// One block's `in_proj` `[z|x|B|C]` column slices and `out_proj` — the
/// weight-side observables/penalty targets, family-uniform.
struct BlockWeights {
    z: Tensor<2>,
    x: Tensor<2>,
    b: Tensor<2>,
    c: Tensor<2>,
    out: Tensor<2>,
}

/// Embedding + optional (untied) LM-head weights, per family.
fn family_emb_head(model: &MambaVocabNet) -> (Tensor<2>, Option<Tensor<2>>) {
    match model {
        MambaVocabNet::Mamba2(net) => (
            net.embedding.weight.val(),
            net.lm_head.as_ref().map(|l| l.weight.val()),
        ),
        MambaVocabNet::Mamba3(net) => (
            net.embedding.weight.val(),
            net.lm_head.as_ref().map(|l| l.weight.val()),
        ),
        _ => panic!("the weight diagnostics expect a Mamba-2 or Mamba-3 network"),
    }
}

/// Per-layer [`BlockWeights`]. Both families lay `in_proj` out as
/// `[z | x | B | C | …tail]` (Mamba-2 tail: `dt`; Mamba-3 tail:
/// `dd_dt | dd_A | λ | θ`); the B/C column width is `ngroups·state_rank`,
/// times `mimo_rank` for Mamba-3.
fn family_block_weights(model: &MambaVocabNet) -> Vec<BlockWeights> {
    match model {
        MambaVocabNet::Mamba2(net) => net
            .layers
            .real_layers
            .iter()
            .map(|l| {
                let block = &l.mamba_block;
                let d_inner = block.d_inner();
                let gn = block.ngroups * block.state_rank;
                let w = block.in_proj.weight.val();
                BlockWeights {
                    z: w.clone().narrow(1, 0, d_inner),
                    x: w.clone().narrow(1, d_inner, d_inner),
                    b: w.clone().narrow(1, 2 * d_inner, gn),
                    c: w.narrow(1, 2 * d_inner + gn, gn),
                    out: block.out_proj.weight.val(),
                }
            })
            .collect(),
        MambaVocabNet::Mamba3(net) => net
            .layers
            .real_layers
            .iter()
            .map(|l| {
                let block = &l.mamba_block;
                let d_inner = block.d_inner();
                let gnm = block.ngroups * block.state_rank * block.mimo_rank;
                let w = block.in_proj.weight.val();
                BlockWeights {
                    z: w.clone().narrow(1, 0, d_inner),
                    x: w.clone().narrow(1, d_inner, d_inner),
                    b: w.clone().narrow(1, 2 * d_inner, gnm),
                    c: w.narrow(1, 2 * d_inner + gnm, gnm),
                    out: block.out_proj.weight.val(),
                }
            })
            .collect(),
        _ => panic!("the weight diagnostics expect a Mamba-2 or Mamba-3 network"),
    }
}

/// Which weight matrices the differentiable spectral-PR penalty
/// ([`weight_pr_penalty`]) applies to.
#[derive(Config, Debug, Copy, PartialEq)]
pub enum PrPenaltyTarget {
    /// The embedding table only.
    Emb,
    /// The embedding table and the (untied) LM head.
    EmbHead,
    /// Every layer's `in_proj` B and C slices (the write/read keys).
    Bc,
    /// All 2-D weights: embedding, LM head, and each layer's `z`/`x`/`B`/`C`
    /// slices and `out_proj`.
    All,
}

/// The differentiable **state**-PR penalty: batch-pooled *uncentered* PR
/// summed over every (virtual layer, head) — the state-side counterpart of
/// [`weight_pr_penalty`], fed by the training forward's attached moments
/// (`final_logits_with_moments`). Like the weight version it is
/// scale-invariant: pure rank pressure on the states' write directions, no
/// norm shrinkage.
///
/// `pairing` (from [`state_pairing_of`]) selects the read-out: the real PR
/// for Mamba-2, the Hermitian `PR_ℂ(M_phys)` for Mamba-3 — the single shipped
/// observable of the physical-frame complex state (within-plane rotation
/// free, rotation-*created* rank charged).
pub fn state_pr_penalty(moments: &[StateMoments], pairing: &StatePairing) -> Tensor<1> {
    moments
        .iter()
        .map(|m| m.clone().pool_batch().pr_complex(pairing, false).sum())
        .reduce(|a, b| a + b)
        .expect("at least one layer of moments")
}

/// The differentiable weight-PR penalty: Σ [`pr_tensor`] over the weights
/// selected by `target`. Added to the loss as `pr_lambda · penalty` this is
/// the causal test of the Step-1 correlation: spectral compression applied as
/// *pressure* (in place of weight decay, which only correlates with it)
/// rather than observed as a side effect. Being scale-invariant, it exerts
/// pure rank pressure — no norm shrinkage at all.
pub fn weight_pr_penalty(model: &MambaVocabNet, target: PrPenaltyTarget) -> Tensor<1> {
    penalty_weights(model, target)
        .into_iter()
        .map(pr_tensor)
        .reduce(|a, b| a + b)
        .expect("at least one penalty target")
}

/// The rank-specificity control for [`weight_pr_penalty`]: a plain L2
/// (Frobenius²) penalty `Σ ‖W‖²_F` over the *same* target matrices, through
/// the same loss pathway. Pure norm pressure, no rank preference.
pub fn weight_l2_penalty(model: &MambaVocabNet, target: PrPenaltyTarget) -> Tensor<1> {
    penalty_weights(model, target)
        .into_iter()
        .map(|w| w.powf_scalar(2.0).sum())
        .reduce(|a, b| a + b)
        .expect("at least one penalty target")
}

/// The weight-independent-gradient control: `Σ ⟨W, ε⟩` with `ε ~ N(0,1)`
/// resampled every call and detached — the gradient w.r.t. `W` is pure noise
/// (unit RMS per element, scaled by the caller's coefficient), through the
/// same loss/Adam pathway as the PR and L2 terms but carrying no information
/// about `W`. Discriminates "any live auxiliary gradient catalyzes" from
/// "the gradient must be a persistent function of the weights".
pub fn weight_noise_penalty(model: &MambaVocabNet, target: PrPenaltyTarget) -> Tensor<1> {
    penalty_weights(model, target)
        .into_iter()
        .map(|w| {
            let noise = w.random_like(burn::tensor::Distribution::Normal(0.0, 1.0));
            (w * noise.detach()).sum()
        })
        .reduce(|a, b| a + b)
        .expect("at least one penalty target")
}

/// The weight matrices selected by `target` (shared by the PR and L2
/// penalties) — the same slices [`weight_pr`] logs, via
/// [`family_block_weights`].
fn penalty_weights(model: &MambaVocabNet, target: PrPenaltyTarget) -> Vec<Tensor<2>> {
    use PrPenaltyTarget::*;
    let (emb, lm_head) = family_emb_head(model);
    let mut weights: Vec<Tensor<2>> = Vec::new();
    if matches!(target, Emb | EmbHead | All) {
        weights.push(emb);
    }
    if matches!(target, EmbHead | All) {
        if let Some(w) = lm_head {
            weights.push(w);
        }
    }
    if matches!(target, Bc | All) {
        for bw in family_block_weights(model) {
            weights.push(bw.b);
            weights.push(bw.c);
            if matches!(target, All) {
                weights.push(bw.z);
                weights.push(bw.x);
                weights.push(bw.out);
            }
        }
    }
    weights
}

/// Differentiable twin of [`pr`] for weight matrices: the spectral PR
/// `(tr WᵀW)² / tr((WᵀW)²)` as a graph-connected scalar tensor, via the two
/// traces (the Gram matrix is taken on the smaller side — same non-zero
/// spectrum). Equals `pr(w, false)` up to the trace read-out.
pub fn pr_tensor(w: Tensor<2>) -> Tensor<1> {
    let [rows, cols] = w.dims();
    let g = if rows <= cols {
        w.clone().matmul(w.clone().transpose())
    } else {
        w.clone().transpose().matmul(w.clone())
    };
    let tr = w.powf_scalar(2.0).sum();
    let tr2 = g.powf_scalar(2.0).sum().clamp_min(1e-12);
    tr.powf_scalar(2.0) / tr2
}

/// Participation ratio of the sample covariance of `h_sn` (rows = samples):
/// `(tr Σ)² / tr(Σ²)` with `Σ = HᵀH/S`, via the two traces only.
pub fn pr(h_sn: Tensor<2>, center: bool) -> f64 {
    let [samples, _n] = h_sn.dims();
    let h_sn = if center {
        h_sn.clone() - h_sn.mean_dim(0)
    } else {
        h_sn
    };
    // tr Σ = Σ_s ‖h_s‖² / S ; tr Σ² = ‖Σ‖²_F (Σ symmetric) — no diagonal op needed.
    let tr = scalar_f64(h_sn.clone().powf_scalar(2.0).sum()) / samples as f64;
    let sigma_nn = h_sn.clone().transpose().matmul(h_sn) / samples as f32;
    let tr2 = scalar_f64(sigma_nn.powf_scalar(2.0).sum());
    (tr * tr) / tr2.max(f64::MIN_POSITIVE)
}

/// Raw uncentered magnitude `tr Σ = Σ_s ‖h_s‖² / S` of a sample cloud (rows =
/// samples): the mean squared state magnitude, [`pr`]'s numerator scale.
pub fn trace(h_sn: Tensor<2>) -> f64 {
    let [samples, _n] = h_sn.dims();
    scalar_f64(h_sn.powf_scalar(2.0).sum()) / samples as f64
}

/// Energy per non-DC frequency of the exact `p`-point DFT of `w_pd` along the
/// token axis, summed over feature columns: `e_k = Σ_d |F(k, d)|²`,
/// `k = 1 … p/2`.
fn dft_energy(w_pd: Tensor<2>) -> Vec<f64> {
    let [p, _d] = w_pd.dims();
    let device = w_pd.device();
    let k_max = p / 2; // non-DC bins 1..=p/2
    let mut cos_flat = vec![0.0f32; p * k_max];
    let mut sin_flat = vec![0.0f32; p * k_max];
    for t in 0..p {
        for k in 0..k_max {
            let angle = 2.0 * std::f64::consts::PI * ((k + 1) * t) as f64 / p as f64;
            cos_flat[t * k_max + k] = angle.cos() as f32;
            sin_flat[t * k_max + k] = angle.sin() as f32;
        }
    }
    let cos_pk = Tensor::<1>::from_floats(cos_flat.as_slice(), &device).reshape([p, k_max]);
    let sin_pk = Tensor::<1>::from_floats(sin_flat.as_slice(), &device).reshape([p, k_max]);
    let re_kd = cos_pk.transpose().matmul(w_pd.clone());
    let im_kd = sin_pk.transpose().matmul(w_pd);
    let energy_k1 = (re_kd.powf_scalar(2.0) + im_kd.powf_scalar(2.0)).sum_dim(1);
    energy_k1
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|e| e as f64)
        .collect()
}

/// PR over a non-negative energy vector: `(Σe)² / Σe²` — the effective number
/// of active entries.
fn pr_of_energies(energies: &[f64]) -> f64 {
    let sum: f64 = energies.iter().sum();
    let sum2: f64 = energies.iter().map(|e| e * e).sum();
    (sum * sum) / sum2.max(f64::MIN_POSITIVE)
}

/// Read a single-element float tensor back to the host.
fn scalar_f64(t: Tensor<1>) -> f64 {
    t.into_data().to_vec::<f32>().unwrap()[0] as f64
}
