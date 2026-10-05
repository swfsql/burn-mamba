//! The cut kinds, the bounds of a cut, and [`loss`] against a reference. The
//! reference runs each pair as one `forward` over `concat(P, C)`, with no
//! cache, no padding and no roll. It builds the loss with other ops (`cumsum`,
//! a softplus). The values and the gradient of the embedding must agree. So
//! the gradient of the split passes reaches the prompts through the caches
//! (the check fails with detached caches).

use super::{
    ContrastBatch, ContrastConfig, Cut, WordIndex, cut, loss, pair_log_probs, scored, shuffle_sentences,
};
use crate::dataset::VOCAB;
use crate::test_net::tiny_net;
use burn::module::{ModuleMapper, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn_stack::optim::SgdConfig;
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

/// The log-probabilities `[n]` of `continuation` after `prompt`, from one
/// `forward` over both (the class latents in front).
fn reference_log_p(net: &MambaVocabNet, path: &MambaSsdPath, prompt: &[u8], continuation: &[u8]) -> Tensor<1> {
    let n = continuation.len();
    let device = net.devices()[0].clone();
    let tokens: Vec<i32> = prompt.iter().chain(&continuation[..n - 1]).map(|&t| t as i32).collect();
    let x = Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).reshape([1, tokens.len()]);
    let (logits, _) = net.forward(x, None, path.clone(), None, None);
    let lead = logits.dims()[1] - tokens.len();
    let targets: Vec<i32> = continuation.iter().map(|&t| t as i32).collect();
    let targets = Tensor::<1, Int>::from_ints(targets.as_slice(), &device).reshape([1, n, 1]);
    log_softmax(logits.narrow(1, lead + prompt.len() - 1, n), 2)
        .gather(2, targets)
        .reshape([n])
}

/// The direction of the loss, apart from its formula and from its readouts:
/// plain gradient descent on the contrast alone makes the own prompt win.
/// The reference measures `Δ = log p(C_i | P_i) − log p(C_i | P_{i−1})`, and
/// it must go up.
#[test]
fn descent_on_the_contrast_raises_delta() {
    let device = test_device().autodiff();
    device.seed(0);
    let mut net = tiny_net(&device).map(&mut Leaf);
    let path = MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None));
    let ids = |len: usize, salt: usize| -> Vec<u8> {
        (0..len).map(|j| ((j * (salt + 2) + salt * 5) % crate::dataset::VOCAB_SIZE) as u8).collect()
    };
    let (k, n) = (4, 16);
    let prompts: Vec<Vec<u8>> = (0..k).map(|i| ids(12, i)).collect();
    let continuations: Vec<Vec<u8>> = (0..k).map(|i| ids(n, i + k)).collect();
    let p: Vec<&[u8]> = prompts.iter().map(Vec::as_slice).collect();
    let c: Vec<&[u8]> = continuations.iter().map(Vec::as_slice).collect();
    let batch = ContrastBatch::new(&p, &c, 12, &device);
    let config = ContrastConfig::new()
        .with_stories(k)
        .with_prompt_min(2)
        .with_prompt_max(12)
        .with_continuation(n);
    // The mean of Δ per character, from the reference, and the contrast
    // readout of `loss`.
    let delta = |net: &MambaVocabNet| -> f32 {
        let sum: f32 = (0..k)
            .map(|i| {
                let own = reference_log_p(net, &path, &prompts[i], &continuations[i]);
                let other = reference_log_p(net, &path, &prompts[(i + k - 1) % k], &continuations[i]);
                (own - other).sum().into_scalar::<f32>()
            })
            .sum();
        sum / (k * n) as f32
    };
    let contrast = |net: &MambaVocabNet| -> f32 {
        let (_, read) = loss(net, &path, batch.clone(), &config, 1.0, 1);
        read.into_data().try_into_vec_as::<f32>().unwrap()[0]
    };
    let (delta_before, contrast_before) = (delta(&net), contrast(&net));
    let mut optim = SgdConfig::new().init();
    for _ in 0..10 {
        let (total, _) = loss(&net, &path, batch.clone(), &config, 1.0, 1);
        let grads = GradientsParams::from_grads(total.backward(), &net);
        net = optim.step(0.05, net, grads);
    }
    let (delta_after, contrast_after) = (delta(&net), contrast(&net));
    assert!(delta_after > delta_before + 0.01, "Δ (own − other) must rise: {delta_before} → {delta_after}");
    assert!(contrast_after < contrast_before, "the loss must fall: {contrast_before} → {contrast_after}");
}

/// At chance (the same log-probabilities after both prompts), the loss is
/// `ln 2`, `Δ` is 0 and the push is 1, at any skip.
#[test]
fn push_is_one_at_chance() {
    let device = test_device();
    let log_p = Tensor::<2>::random([3, 10], burn::tensor::Distribution::Uniform(-3.0, 0.0), &device);
    for skip in [0, 4] {
        let (_, readouts) = scored(log_p.clone(), log_p.clone(), skip);
        let read: Vec<f32> = readouts.into_data().try_into_vec_as().unwrap();
        let tol = dtype_tol(1e-5);
        assert!((read[0] - std::f32::consts::LN_2).abs() < tol, "loss {}", read[0]);
        assert!(read[1].abs() < tol, "delta {}", read[1]);
        assert!((read[2] - 1.0).abs() < tol, "push {}", read[2]);
    }
}

#[test]
fn shuffle_keeps_the_characters_and_the_last_sentence() {
    let text = "tom ran. sue sat. a cat came! it was big. and then the";
    let tokens = VOCAB.encode(text);
    let (head, tail) = (VOCAB.encode("tom ran. "), VOCAB.encode("and then the"));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut first_moved = false;
    for keep_first in [true, false] {
        for _ in 0..20 {
            let shuffled = shuffle_sentences(&tokens, keep_first, &mut rng).expect("sentences to move");
            assert_ne!(shuffled, tokens, "a new order");
            let mut a = shuffled.clone();
            let mut b = tokens.clone();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "the same characters");
            assert!(shuffled.ends_with(&tail), "the last sentence stays at the cut");
            if keep_first {
                assert!(shuffled.starts_with(&head), "the first sentence stays");
            } else {
                first_moved |= !shuffled.starts_with(&head);
            }
        }
    }
    assert!(first_moved, "without keep_first, the first sentence moves");
    // Too few sentences to move: one for keep_first, after the first one.
    assert!(shuffle_sentences(&VOCAB.encode("tom ran. sue sat. and then"), true, &mut rng).is_none());
    assert!(shuffle_sentences(&VOCAB.encode("tom ran. and then"), false, &mut rng).is_none());
}

#[test]
fn word_index_weights_rare_words() {
    let stories: Vec<Vec<u8>> = [
        "once upon a time a dragon named zork ate a pie",
        "once upon a time a dragon named zork sang a song",
        "once upon a time a girl named lily ate a pie",
        "once upon a time a boy named tim had a ball",
    ]
    .iter()
    .map(|s| VOCAB.encode(s))
    .collect();
    let index = WordIndex::new(&stories, 1000);
    let words: Vec<Vec<u32>> = stories.iter().map(|s| index.words(s)).collect();
    let sim = |a: usize, b: usize| index.similarity(&words[a], &words[b]);
    assert!((sim(0, 0) - 1.0).abs() < 1e-6);
    // The shared rare words ("dragon zork", "ate pie") weigh more than the
    // words that every story has.
    assert!(sim(0, 1) > sim(0, 3));
    assert!(sim(0, 2) > sim(0, 3));
    assert!(sim(0, 3) < 0.05, "only the common words: {}", sim(0, 3));
    // A prompt compares by its own words only: the cut before "ate a pie"
    // leaves story 0 nothing rare in common with story 2.
    let prompt = VOCAB.encode("once upon a time a dragon named zork ");
    assert!(index.similarity(&index.words(&prompt), &words[2]) < 0.05);
}

#[test]
fn explicit_negatives_match_a_forward_over_each_pair() {
    let device = test_device();
    device.seed(0);
    let net = tiny_net(&device);
    let path = MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None));
    let ids = |len: usize, salt: usize| -> Vec<u8> {
        (0..len).map(|j| ((j * 5 + salt * 11 + 1) % crate::dataset::VOCAB_SIZE) as u8).collect()
    };
    let prompts = [ids(6, 1), ids(4, 2), ids(7, 3)];
    let negatives = [ids(3, 7), ids(7, 8), ids(5, 9)];
    let continuations = [ids(5, 4), ids(5, 5), ids(5, 6)];
    let p: Vec<&[u8]> = prompts.iter().map(Vec::as_slice).collect();
    let neg: Vec<&[u8]> = negatives.iter().map(Vec::as_slice).collect();
    let c: Vec<&[u8]> = continuations.iter().map(Vec::as_slice).collect();
    let batch = ContrastBatch::with_negatives(&p, &neg, &c, 8, &device);
    let (own_kn, other_kn) = pair_log_probs(&net, &path, batch);
    let tol = dtype_tol(1e-4);
    for i in 0..3 {
        let own = reference_log_p(&net, &path, &prompts[i], &continuations[i]);
        let other = reference_log_p(&net, &path, &negatives[i], &continuations[i]);
        let row = |t: &Tensor<2>| t.clone().narrow(0, i, 1).reshape([5]);
        assert!(max_rel_diff(row(&own_kn), own) < tol, "own row {i}");
        assert!(max_rel_diff(row(&other_kn), other) < tol, "other row {i}");
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

    let log_p = |prompt: &[u8], continuation: &[u8]| reference_log_p(&net, &path, prompt, continuation);
    let own: Vec<Tensor<1>> = (0..k).map(|i| log_p(&prompts[i], &continuations[i])).collect();
    let other: Vec<Tensor<1>> = (0..k).map(|i| log_p(&prompts[(i + k - 1) % k], &continuations[i])).collect();
    let own_kn = Tensor::stack::<2>(own, 0);
    let other_kn = Tensor::stack::<2>(other, 0);
    let prefix = (own_kn.clone() - other_kn.clone()).narrow(1, 2, n - 2).cumsum(1);
    // The push on the host: `4/(m + 1) · mean_{i,t} (t + 1)·σ(−Δ_≤t)`.
    let m = n - 2;
    let prefix_host: Vec<f64> = prefix.clone().into_data().try_into_vec_as().unwrap();
    let push = prefix_host
        .iter()
        .enumerate()
        .map(|(it, &x)| (it % m + 1) as f64 / (1.0 + x.exp()))
        .sum::<f64>()
        / (k * m) as f64
        * 4.0
        / (m + 1) as f64;
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
    assert!((f64::from(read[2]) - push).abs() < f64::from(tol), "push readout: {} vs {push}", read[2]);

    let embedding_grad = |loss: Tensor<1>| -> Tensor<2> {
        let grads = loss.backward();
        let MambaVocabNet::Mamba3(net) = &net else { panic!("a Mamba-3 network") };
        net.embedding.weight.val().grad(&grads).expect("embedding grad")
    };
    let (got_grad, want_grad) = (embedding_grad(got), embedding_grad(want));
    let diff = max_rel_diff(got_grad, want_grad);
    assert!(diff < dtype_tol(1e-3), "embedding grad: max rel diff {diff}");
}
