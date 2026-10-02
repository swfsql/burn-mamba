//! The cut kinds, the bounds of a cut, and [`loss`] against a reference. The
//! reference runs each pair as one `forward` over `concat(P, C)`, with no
//! cache, no padding and no roll. It builds the loss with other ops (`cumsum`,
//! a softplus). The values and the gradient of the embedding must agree. So
//! the gradient of the split passes reaches the prompts through the caches
//! (the check fails with detached caches).

use super::{ContrastBatch, ContrastConfig, Cut, cut, loss};
use crate::dataset::VOCAB;
use crate::test_net::tiny_net;
use burn::module::{ModuleMapper, Param};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn_mamba::prelude::*;
use burn_stack::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn cut_kinds_allow_their_boundaries() {
    let text = "tom ran. \"hi!\" he said\nthe end";
    let tokens = VOCAB.encode(text);
    let allowed = |cut: Cut| -> Vec<usize> { (1..tokens.len()).filter(|&at| cut.allows(&tokens, at)).collect() };
    // The first character of each word.
    let word: Vec<usize> = (1..text.len())
        .filter(|&at| matches!(text.as_bytes()[at - 1], b' ' | b'\n'))
        .collect();
    assert_eq!(allowed(Cut::Word), word);
    // After "ran. ", after "!\" " and after the newline.
    assert_eq!(allowed(Cut::Sentence), vec![9, 15, 23]);
    assert_eq!(allowed(Cut::Char), (1..tokens.len()).collect::<Vec<_>>());
}

#[test]
fn cut_stays_in_its_bounds() {
    let tokens = VOCAB.encode(&"a b. c d! e f? ".repeat(10));
    for kind in [Cut::Char, Cut::Word, Cut::Sentence] {
        let config = ContrastConfig::new()
            .with_prompt_min(20)
            .with_prompt_max(60)
            .with_continuation(50)
            .with_cut(kind);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..200 {
            let at = cut(&config, &mut rng, &tokens).expect("each 5 characters hold a boundary");
            assert!((20..=60).contains(&at), "{kind:?}: cut {at}");
            assert!(at + 50 <= tokens.len(), "{kind:?}: no room for the continuation");
            assert!(kind.allows(&tokens, at), "{kind:?}: cut {at}");
        }
    }
}

#[test]
fn weight_is_zero_then_ramps() {
    let config = ContrastConfig::new().with_weight(0.8).with_start(100).with_ramp(40);
    let at = |step| config.weight_at(step);
    assert_eq!([at(0), at(99), at(100)], [0.0, 0.0, 0.0]);
    assert!((at(110) - 0.2).abs() < 1e-12 && (at(130) - 0.6).abs() < 1e-12);
    assert_eq!([at(140), at(10_000)], [0.8, 0.8]);
    let step = ContrastConfig::new().with_weight(0.8).with_start(100).with_ramp(0);
    assert_eq!([step.weight_at(99), step.weight_at(100)], [0.0, 0.8]);
}

/// Makes every float parameter a leaf of the autodiff graph again. The noise
/// of `tiny_net` makes each one the output of an op, and the gradients go only
/// to the leaves.
struct Leaf;

impl ModuleMapper for Leaf {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| t.detach().require_grad())
    }
}

#[test]
fn loss_matches_a_forward_over_each_pair() {
    let device = test_device().autodiff();
    device.seed(0);
    let net = tiny_net(&device).map(&mut Leaf);
    let path = MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None));
    // Three rows. The prompts have three lengths, so the pass over them is
    // padded.
    let ids = |len: usize, salt: usize| -> Vec<u8> {
        (0..len).map(|k| ((k * 7 + salt * 13 + 3) % crate::dataset::VOCAB_SIZE) as u8).collect()
    };
    let prompts = [ids(5, 1), ids(8, 2), ids(3, 3)];
    let continuations = [ids(6, 4), ids(6, 5), ids(6, 6)];
    let (k, n, prompt_max, lm_positions) = (3, 6, 8, 40);
    let config = ContrastConfig::new()
        .with_stories(k)
        .with_prompt_min(2)
        .with_prompt_max(prompt_max)
        .with_continuation(n)
        .with_skip(2)
        .with_lm(true);
    let p: Vec<&[u8]> = prompts.iter().map(Vec::as_slice).collect();
    let c: Vec<&[u8]> = continuations.iter().map(Vec::as_slice).collect();
    let batch = ContrastBatch::new(&p, &c, prompt_max, &device);
    let (got, readouts) = loss(&net, &path, batch, &config, 0.5, lm_positions);

    // The log-probabilities `[n]` of `continuation` after `prompt`, from one
    // `forward` over both (the class latents in front).
    let log_p = |prompt: &[u8], continuation: &[u8]| -> Tensor<1> {
        let tokens: Vec<i32> = prompt.iter().chain(&continuation[..n - 1]).map(|&t| t as i32).collect();
        let x = Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).reshape([1, tokens.len()]);
        let (logits, _) = net.forward(x, None, path.clone(), None, None);
        let lead = logits.dims()[1] - tokens.len();
        let targets: Vec<i32> = continuation.iter().map(|&t| t as i32).collect();
        let targets = Tensor::<1, Int>::from_ints(targets.as_slice(), &device).reshape([1, n, 1]);
        log_softmax(logits.narrow(1, lead + prompt.len() - 1, n), 2)
            .gather(2, targets)
            .reshape([n])
    };
    let own: Vec<Tensor<1>> = (0..k).map(|i| log_p(&prompts[i], &continuations[i])).collect();
    let other: Vec<Tensor<1>> = (0..k).map(|i| log_p(&prompts[(i + k - 1) % k], &continuations[i])).collect();
    let own_kn = Tensor::stack::<2>(own, 0);
    let other_kn = Tensor::stack::<2>(other, 0);
    let prefix = (own_kn.clone() - other_kn.clone()).narrow(1, 2, n - 2).cumsum(1);
    // −log σ(x) = log(1 + e^{−x}).
    let contrast = prefix.neg().exp().add_scalar(1.0).log().mean();
    let lm = own_kn.clone().neg().sum().div_scalar(lm_positions as f64);
    // λ = 0.5, times the scale 4/(m + 1) of m = n − J = 4 scored characters.
    let want = contrast.clone().mul_scalar(0.5 * 4.0 / 5.0) + lm;
    let want_delta = (own_kn - other_kn).narrow(1, 2, n - 2).mean();

    let tol = dtype_tol(1e-4);
    let scalar = |t: Tensor<1>| t.into_data().try_into_vec_as::<f32>().unwrap()[0];
    let read: Vec<f32> = readouts.into_data().try_into_vec_as().unwrap();
    assert!((scalar(got.clone()) - scalar(want.clone())).abs() < tol, "loss");
    assert!((read[0] - scalar(contrast)).abs() < tol, "contrast readout");
    assert!((read[1] - scalar(want_delta)).abs() < tol, "delta readout");

    let embedding_grad = |loss: Tensor<1>| -> Tensor<2> {
        let grads = loss.backward();
        let MambaVocabNet::Mamba3(net) = &net else { panic!("a Mamba-3 network") };
        net.embedding.weight.val().grad(&grads).expect("embedding grad")
    };
    let (got_grad, want_grad) = (embedding_grad(got), embedding_grad(want));
    let diff = max_rel_diff(got_grad, want_grad);
    assert!(diff < dtype_tol(1e-3), "embedding grad: max rel diff {diff}");
}
