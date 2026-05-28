//! # State-tracking example — abelian RoPE vs. quaternion rotation
//!
//! A self-contained demo of the capability gap motivating the quaternion
//! (`RotationKind::Quaternion4D`) rotation: tracking composition in the
//! **non-solvable** group `A₅` (the alternating group on 5 letters, the rotation
//! group of the icosahedron).
//!
//! The model reads a sequence of `A₅` generators (one-hot) and must output, at
//! **every position**, the cumulative product so far (a 60-way classification).
//! By Barrington's theorem this word problem is `NC¹`-complete; a single-layer
//! linear SSM with **abelian** (`SO(2)`/complex RoPE) state transitions is
//! confined to the solvable/`TC⁰` regime and cannot track it, whereas the
//! **non-abelian** `SU(2)` quaternion rotation can represent the icosahedral
//! group `2I = SL(2,5)` (a double cover of `A₅`).
//!
//! This is a *minimal* manual training loop (no `burn::train::Learner`,
//! no dataloader) so the whole demo is one file and easy to read.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example state-tracking --features backend-flex -- --rotation complex
//! cargo run --release --example state-tracking --features backend-flex -- --rotation quaternion
//! ```
//!
//! Compare the two final per-token accuracies: chance is `1/60 ≈ 1.7%`. The
//! quaternion run climbs above the complex run, which tends to plateau.
//!
//! The defaults are deliberately **tiny** (fibonacci-scale: `d_model=32`,
//! one layer, `seq=12`, 512 sequences) so the demo runs quickly on CPU and
//! already shows the quaternion learning the non-abelian composition. For a
//! *wider, cleaner* gap, run on GPU (`--features backend-cuda`) and scale up the
//! model / sequence length / `--epochs` — this is a demonstration of the
//! capability, not a tuned benchmark.

use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Int;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// A₅ group: enumerate the 60 even permutations of {0,..,4} and compose them.
// ---------------------------------------------------------------------------

/// `n!` permutations of `[0,1,2,3,4]`, keeping the even ones (sign `+1`).
fn even_permutations() -> Vec<[usize; 5]> {
    let mut perms = Vec::new();
    let mut p = [0usize, 1, 2, 3, 4];
    permute(&mut p, 0, &mut perms);
    perms.retain(|p| parity_even(p));
    perms.sort();
    perms
}

fn permute(p: &mut [usize; 5], k: usize, out: &mut Vec<[usize; 5]>) {
    if k == 5 {
        out.push(*p);
        return;
    }
    for i in k..5 {
        p.swap(k, i);
        permute(p, k + 1, out);
        p.swap(k, i);
    }
}

/// A permutation is even when its number of inversions is even.
fn parity_even(p: &[usize; 5]) -> bool {
    let mut inv = 0;
    for i in 0..5 {
        for j in (i + 1)..5 {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    inv % 2 == 0
}

/// `(a ∘ b)[i] = a[b[i]]` — apply `b`, then `a`.
fn compose(a: &[usize; 5], b: &[usize; 5]) -> [usize; 5] {
    let mut r = [0usize; 5];
    for i in 0..5 {
        r[i] = a[b[i]];
    }
    r
}

// ---------------------------------------------------------------------------
// Tiny deterministic RNG (no external deps), and the dataset generator.
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// One generated batch: one-hot generator inputs and the per-position target
/// class (index of the cumulative product in the `A₅` enumeration).
struct Data {
    /// `[n, seq, num_gen]` one-hot generator at each step.
    inputs: Tensor<3>,
    /// `[n * seq]` flattened target class indices (for the loss).
    targets_flat: Tensor<1, Int>,
    /// Host copy of the targets for accuracy, `[n * seq]`.
    targets_host: Vec<i64>,
}

/// Generate `n` random generator sequences and their running-product targets.
fn make_data(
    n: usize,
    seq: usize,
    generators: &[[usize; 5]],
    index_of: &std::collections::HashMap<[usize; 5], usize>,
    rng: &mut Lcg,
    device: &Device,
) -> Data {
    let num_gen = generators.len();
    let mut input_flat = vec![0.0f32; n * seq * num_gen];
    let mut targets = vec![0i64; n * seq];

    for b in 0..n {
        let mut state = [0usize, 1, 2, 3, 4]; // identity
        for t in 0..seq {
            let g = rng.below(num_gen);
            input_flat[(b * seq + t) * num_gen + g] = 1.0;
            state = compose(&generators[g], &state); // P_t = g ∘ P_{t-1}
            targets[b * seq + t] = index_of[&state] as i64;
        }
    }

    let inputs = Tensor::<1>::from_floats(input_flat.as_slice(), device).reshape([n, seq, num_gen]);
    let targets_flat = Tensor::<1, Int>::from_ints(targets.as_slice(), device);
    Data {
        inputs,
        targets_flat,
        targets_host: targets,
    }
}

// ---------------------------------------------------------------------------
// Model: in_proj → Mamba3Layers → out_proj (per-position 60-way head).
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Net {
    in_proj: Linear,
    layers: Mamba3Layers,
    out_proj: Linear,
}

impl Net {
    /// `[batch, seq, num_gen]` → `[batch, seq, num_classes]` logits.
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.in_proj.forward(x);
        let (x, _caches) = self.layers.forward(x, None, Mamba3SsdPath::Minimal(None));
        self.out_proj.forward(x)
    }
}

fn build_model(
    num_gen: usize,
    num_classes: usize,
    rotation: RotationKind,
    device: &Device,
) -> Net {
    // Intentionally tiny (fibonacci-scale) so the demo runs quickly on CPU.
    // Scale these up (and use `--features backend-cuda`) for a wider gap.
    let d_model = 32;
    let expand = 2;
    let per_head_dim = 8; // d_inner = 64, nheads = 8
    let state_rank = 16; // multiple of 4 (required by Quaternion4D)
    let n_real_layers = 1;

    let block = Mamba3Config::new(d_model)
        .with_state_rank(state_rank)
        .with_expand(expand)
        .with_per_head_dim(per_head_dim)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rotation(rotation);
    let layers = Mamba3LayersConfig::new(n_real_layers, block).init(device);

    Net {
        in_proj: LinearConfig::new(num_gen, d_model).init(device),
        layers,
        out_proj: LinearConfig::new(d_model, num_classes).init(device),
    }
}

// ---------------------------------------------------------------------------
// Training / evaluation
// ---------------------------------------------------------------------------

fn accuracy(model: &Net, data: &Data, num_classes: usize) -> f32 {
    let [n, seq, _] = data.inputs.dims();
    let logits = model.forward(data.inputs.clone()); // [n, seq, C]
    let pred = logits
        .reshape([n * seq, num_classes])
        .argmax(1) // [n*seq, 1]
        .reshape([n * seq])
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    let correct = pred
        .iter()
        .zip(&data.targets_host)
        .filter(|(a, b)| **a as i64 == **b)
        .count();
    correct as f32 / (n * seq) as f32
}

fn main() {
    // ── Args ──────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let rotation = match arg_value(&args, "--rotation").as_deref() {
        Some("quaternion") | Some("quat") => RotationKind::Quaternion4D,
        Some("complex") | None => RotationKind::Complex2D,
        Some(other) => panic!("--rotation must be 'complex' or 'quaternion', got {other:?}"),
    };
    let epochs: usize = arg_value(&args, "--epochs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);

    // ── Setup ───────────────────────────────────────────────────────────────
    let device = Device::default();
    let train_device = device.clone().autodiff();

    let perms = even_permutations();
    assert_eq!(perms.len(), 60, "A₅ has 60 elements");
    let index_of: std::collections::HashMap<[usize; 5], usize> =
        perms.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    // Two generators of A₅: a 5-cycle and a 3-cycle.
    let generators = [[1usize, 2, 3, 4, 0], [1usize, 2, 0, 3, 4]];
    let num_gen = generators.len();
    let num_classes = perms.len();

    // Tiny by default (quick CPU demo); raise on backend-cuda for a wider gap.
    let seq = 12;
    let n_train = 512;
    let n_eval = 128;
    let batch_size = 64;

    let mut rng = Lcg(0xC0FFEE);
    let train = make_data(n_train, seq, &generators, &index_of, &mut rng, &train_device);
    let eval = make_data(n_eval, seq, &generators, &index_of, &mut rng, &train_device);

    println!(
        "State-tracking on A₅ (non-solvable): rotation={rotation:?}, seq={seq}, \
         classes={num_classes}, chance={:.1}%",
        100.0 / num_classes as f32
    );

    let mut model = build_model(num_gen, num_classes, rotation, &train_device);
    println!("parameters: {}", model.num_params());

    let mut optim = AdamWConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&train_device);
    let lr = 3e-3;

    let n_batches = n_train / batch_size;
    for epoch in 1..=epochs {
        let mut running = 0.0f32;
        for bi in 0..n_batches {
            let lo = bi * batch_size;
            let x = train.inputs.clone().narrow(0, lo, batch_size); // [b, seq, g]
            let logits = model.forward(x); // [b, seq, C]
            let logits = logits.reshape([batch_size * seq, num_classes]);
            // The matching target slice (targets are flattened as [n*seq]).
            let t = train
                .targets_flat
                .clone()
                .narrow(0, lo * seq, batch_size * seq);
            let loss = loss_fn.forward(logits, t);
            running += loss.clone().into_scalar::<f32>();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }

        if epoch % 5 == 0 || epoch == 1 || epoch == epochs {
            // Evaluate on the non-autodiff (valid) model.
            let acc = accuracy(&model, &eval, num_classes);
            println!(
                "epoch {epoch:>3}/{epochs}  avg_loss {:.4}  eval_acc {:.1}%",
                running / n_batches as f32,
                acc * 100.0,
            );
        }
    }

    let final_acc = accuracy(&model, &eval, num_classes);
    println!("\nFinal eval accuracy ({rotation:?}): {:.1}%", final_acc * 100.0);
    let _ = model.valid(); // demonstrates the inference (non-autodiff) module exists
}

/// Minimal `--flag value` lookup.
fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
