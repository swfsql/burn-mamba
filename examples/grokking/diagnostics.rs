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

/// Spectral PRs of one Mamba-2 block's weights: each `in_proj` slice
/// (`[z | x | B | C | dt]`; `dt` skipped — it is one column per head),
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

/// Run `inputs_bs` `[n, s]` through the model token-by-token on the plain
/// backend, reading `ssm_bhpr` from every layer's cache after each step, and
/// return the N-side PR per (layer, head).
pub fn state_pr(model: &MambaVocabNet, inputs_bs: &Tensor<2, Int>) -> Vec<StatePr> {
    let [_n, s] = inputs_bs.dims();
    let mut caches = None;
    // per_step[t][layer]: `[batch, nheads, per_head_dim, state_rank]`
    let mut per_step: Vec<Vec<Tensor<4>>> = Vec::with_capacity(s);
    for t in 0..s {
        let x_b = inputs_bs.clone().narrow(1, t, 1).squeeze_dim::<1>(1);
        let (_logits, new_caches) = model.step(x_b, caches, None);
        let states = match &new_caches {
            MambaCaches::Mamba2(c) => c.caches.iter().map(|l| l.ssm_bhpr.clone()).collect(),
            _ => panic!("the state-PR diagnostic expects a Mamba-2 network"),
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
                pooled_centered: pr(pooled_sn.clone(), true),
                pooled_uncentered: pr(pooled_sn, false),
                final_centered: pr(final_sn.clone(), true),
                final_uncentered: pr(final_sn, false),
            });
        }
    }
    out
}

/// Weight-side PRs: embedding, LM head, the embedding's exact `p`-point
/// Fourier-energy PR (only the first `p` rows — the vocab may be padded),
/// and each block's per-slice weight PRs.
pub fn weight_pr(model: &MambaVocabNet, p: usize) -> WeightPr {
    let net = match model {
        MambaVocabNet::Mamba2(net) => net,
        _ => panic!("the weight-PR diagnostic expects a Mamba-2 network"),
    };
    let emb_vd = net.embedding.weight.val();
    let emb = pr(emb_vd.clone(), false);
    let lm_head = match &net.lm_head {
        Some(linear) => pr(linear.weight.val(), false),
        None => f64::NAN,
    };
    let emb_pd = emb_vd.narrow(0, 0, p);
    let emb_freq = pr_of_energies(&dft_energy(emb_pd.clone()));

    let layers = net
        .layers
        .real_layers
        .iter()
        .enumerate()
        .map(|(layer, l)| {
            let block = &l.mamba_block;
            let d_inner = block.d_inner();
            let gn = block.ngroups * block.state_rank;
            // in_proj weight `[d_model, d_inner + conv_dim + nheads]`,
            // columns laid out `[z | x | B | C | dt]`.
            let w = block.in_proj.weight.val();
            let w_z = w.clone().narrow(1, 0, d_inner);
            let w_x = w.clone().narrow(1, d_inner, d_inner);
            let w_b = w.clone().narrow(1, 2 * d_inner, gn);
            let w_c = w.narrow(1, 2 * d_inner + gn, gn);
            LayerWeightPr {
                layer,
                z: pr(w_z, false),
                x: pr(w_x, false),
                b: pr(w_b.clone(), false),
                c: pr(w_c, false),
                out: pr(block.out_proj.weight.val(), false),
                b_alphabet: pr(emb_pd.clone().matmul(w_b), true),
            }
        })
        .collect();

    WeightPr {
        emb,
        lm_head,
        emb_freq,
        layers,
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
/// penalties).
fn penalty_weights(model: &MambaVocabNet, target: PrPenaltyTarget) -> Vec<Tensor<2>> {
    use PrPenaltyTarget::*;
    let net = match model {
        MambaVocabNet::Mamba2(net) => net,
        _ => panic!("the weight penalties expect a Mamba-2 network"),
    };
    let mut weights: Vec<Tensor<2>> = Vec::new();
    if matches!(target, Emb | EmbHead | All) {
        weights.push(net.embedding.weight.val());
    }
    if matches!(target, EmbHead | All)
        && let Some(linear) = &net.lm_head
    {
        weights.push(linear.weight.val());
    }
    if matches!(target, Bc | All) {
        for l in &net.layers.real_layers {
            let block = &l.mamba_block;
            let d_inner = block.d_inner();
            let gn = block.ngroups * block.state_rank;
            // Same `[z | x | B | C | dt]` column layout as in [`weight_pr`].
            let w = block.in_proj.weight.val();
            weights.push(w.clone().narrow(1, 2 * d_inner, gn));
            weights.push(w.clone().narrow(1, 2 * d_inner + gn, gn));
            if matches!(target, All) {
                weights.push(w.clone().narrow(1, 0, d_inner));
                weights.push(w.narrow(1, d_inner, d_inner));
                weights.push(block.out_proj.weight.val());
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
        .try_to_vec::<f32>()
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
    t.into_data().try_to_vec::<f32>().unwrap()[0] as f64
}
