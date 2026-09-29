//! A packed forward of a vocab network against each of its sequences run
//! alone.
//!
//! Each row holds several sequences, each at a multiple of
//! [`MambaVocabNet::pack_align`], each with its opening slots (one per class
//! latent of the stack). Alone, a sequence runs through the plain `forward`,
//! which splices the latents in front of it. The test compares:
//!
//! - the logits of every opening slot and every token of each sequence,
//! - the gradients of every parameter, of one loss over those logits.
//!
//! The lengths make sequences shorter and longer than a chunk, and a gap
//! after each one.

use crate::prelude::*;
use crate::unified::network::*;
use burn::module::{ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients};
use burn_stack::modules::ResidualsConfig;
use burn_stack::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff};
use burn_stack::utils::{ClassLatent, Packed};
use burn_stack::utils::test_helpers::test_device;

const VOCAB: usize = 11;
const VAL_TOL: f32 = 1e-4;
/// Relative to the largest gradient of a parameter (at least 1).
const GRAD_TOL: f32 = 1e-5;
/// The tokens of each sequence of each row (the chunk is 4 or 8 tokens).
const ROWS: [&[usize]; 2] = [&[5, 9, 1], &[12, 3]];

/// The first row of each sequence, and the width of the rows.
fn starts(align: usize, lead: usize) -> (Vec<Vec<usize>>, usize) {
    let mut width = 0;
    let starts = ROWS
        .iter()
        .map(|lens| {
            let mut at = 0;
            let starts = lens
                .iter()
                .map(|&len| {
                    let start = at;
                    at = (start + lead + len).next_multiple_of(align);
                    start
                })
                .collect();
            // A free tail after the last sequence.
            width = width.max(at + align);
            starts
        })
        .collect();
    (starts, width)
}

struct Collect<'a> {
    grads: &'a Gradients,
    out: Vec<Option<Tensor<1>>>,
}

impl ModuleVisitor for Collect<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let n = param.val().shape().num_elements();
        self.out.push(param.val().grad(self.grads).map(|g| g.reshape([n])));
    }
}

fn param_grads<M: Module>(module: &M, grads: &Gradients) -> Vec<Option<Tensor<1>>> {
    let mut collect = Collect {
        grads,
        out: Vec::new(),
    };
    module.visit(&mut collect);
    collect.out
}

fn check(net: MambaVocabNet, path: MambaSsdPath) {
    let device = test_device().autodiff();
    let align = net.pack_align(&path).expect("a family with packed rows");
    let lead = net.n_class_latents();
    let (starts, width) = starts(align, lead);
    let batch = ROWS.len();

    // The tokens of each sequence, and the packed rows (token 0 in the slots
    // and the gaps).
    let tokens: Vec<Vec<Vec<i64>>> = ROWS
        .iter()
        .map(|lens| {
            lens.iter()
                .map(|&len| {
                    Tensor::<1>::random([len], Distribution::Uniform(0.0, VOCAB as f64), &device)
                        .int()
                        .into_data()
                        .convert::<i64>()
                        .try_to_vec::<i64>()
                        .unwrap()
                })
                .collect()
        })
        .collect();
    let mut packed_ids = vec![0i64; batch * width];
    for (b, row) in tokens.iter().enumerate() {
        for (seq, &start) in row.iter().zip(&starts[b]) {
            for (j, &t) in seq.iter().enumerate() {
                packed_ids[b * width + start + lead + j] = t;
            }
        }
    }
    let packed_ids = Tensor::<1, Int>::from_ints(packed_ids.as_slice(), &device).reshape([batch, width]);
    let layout = Packed::from_starts(&starts, lead, width, &device);

    // A fixed weight per logit, so the loss reaches every one of them.
    let weight = Tensor::<3>::random([batch, width, VOCAB], Distribution::Normal(0.0, 1.0), &device);

    let (logits, _) = net.forward_packed(packed_ids, None, path.clone(), &layout);
    let mut loss_packed = Tensor::<1>::zeros([1], &device);
    let mut loss_alone = Tensor::<1>::zeros([1], &device);
    for (b, row) in tokens.iter().enumerate() {
        for (seq, &start) in row.iter().zip(&starts[b]) {
            let n = lead + seq.len();
            let ids = Tensor::<1, Int>::from_ints(seq.as_slice(), &device).reshape([1, seq.len()]);
            let (alone, _) = net.forward(ids, None, path.clone(), None, None);
            assert_eq!([1, n, VOCAB], alone.dims());
            let packed = logits.clone().narrow(0, b, 1).narrow(1, start, n);
            let diff = max_rel_diff(packed.clone(), alone.clone());
            assert!(diff < dtype_tol(VAL_TOL), "row {b}, sequence at {start}: logits differ by {diff}");
            let w = weight.clone().narrow(0, b, 1).narrow(1, start, n);
            loss_packed = loss_packed + (packed * w.clone()).sum();
            loss_alone = loss_alone + (alone * w).sum();
        }
    }

    let grads_packed = param_grads(&net, &loss_packed.backward());
    let grads_alone = param_grads(&net, &loss_alone.backward());
    assert_eq!(grads_packed.len(), grads_alone.len());
    for (i, (p, a)) in grads_packed.into_iter().zip(grads_alone).enumerate() {
        match (p, a) {
            (Some(p), Some(a)) => {
                // The loss sums every logit, so a gradient can be in the
                // thousands (the tied embedding). Compare against its scale.
                let scale = a.clone().abs().max().into_scalar::<f32>().max(1.0);
                let diff = max_abs_diff(p, a);
                assert!(
                    diff < dtype_tol(GRAD_TOL) * scale,
                    "the gradient of parameter {i} differs by {diff} (max |grad| {scale})"
                );
            }
            (None, None) => {}
            (p, a) => panic!("parameter {i}: gradient in one run only ({} vs {})", p.is_some(), a.is_some()),
        }
    }
}

#[cfg(feature = "mamba3")]
fn mamba3_net(micro_steps: usize, trapezoid: Trapezoid, residuals: ResidualsConfig) -> MambaVocabNet {
    let block = Mamba3Config::new(8)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_micro_steps(micro_steps)
        .with_trapezoid(trapezoid);
    MambaVocabNetConfig::Mamba3 {
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        vocab_size: VOCAB,
        pad_vocab_size_multiple: 1,
        mamba_block: block,
        missing_lm_head: true,
        class_latents: vec![ClassLatent::Start; 2],
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals,
        mlp: None,
        untied: Vec::new(),
    }
    .init(&test_device().autodiff())
}

#[cfg(feature = "mamba3")]
#[test]
fn packed_mamba3_u1_matches_alone() {
    let net = mamba3_net(1, Trapezoid::default(), ResidualsConfig::Standard);
    check(net, MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(4))));
}

#[cfg(feature = "mamba3")]
#[test]
fn packed_mamba3_u2_vertical_multigate_matches_alone() {
    let residuals = ResidualsConfig::MultiGate {
        n_stream: 2,
        init_bias: 0.0,
        init_bias_step: 0.0,
        per_virtual_layer: false,
    };
    let net = mamba3_net(2, Trapezoid::Vertical, residuals);
    check(net, MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(8))));
}

#[cfg(feature = "mamba3")]
#[test]
fn packed_mamba3_serial_matches_alone() {
    let net = mamba3_net(1, Trapezoid::default(), ResidualsConfig::Standard);
    check(net, MambaSsdPath::Mamba3(Mamba3SsdPath::Serial(Some(4))));
}

#[cfg(feature = "mamba2")]
#[test]
fn packed_mamba2_matches_alone() {
    let block = Mamba2Config::new(8)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = MambaVocabNetConfig::Mamba2 {
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        vocab_size: VOCAB,
        pad_vocab_size_multiple: 1,
        mamba_block: block,
        missing_lm_head: true,
        class_latents: vec![ClassLatent::Start; 2],
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: ResidualsConfig::Standard,
        mlp: None,
        untied: Vec::new(),
    }
    .init(&test_device().autodiff());
    check(net, MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(Some(4))));
}
