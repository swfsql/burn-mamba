//! The opening of a prompted story ([`open`]) runs the class latents before
//! the prompt, and the decode continues it as the model was scored. For one
//! prompt and one continuation, [`check_decode`] asserts:
//!
//! - The chunked prefill path equals `prime`, then a `forward` of the prompt
//!   from the cache that `prime` leaves (the latents first).
//! - The one-`forward` path (no prefill) equals it too: there, the cursors
//!   splice the latents in front of the prompt, as in training.
//! - Other latent values give other logits, so the opening reads the latents.
//!   (`forward` with no cursors cannot serve as the contrast: it covers the
//!   whole sequence in one call, so it also splices the latents.)
//! - After each opening, teacher-forced `step`s over the continuation give the
//!   logits of one `forward` over the prompt and the continuation (the path of
//!   the training and the validation loss).
//!
//! [`the_latents_run_before_the_prompt`] runs it on the tiny network of
//! [`test_net`](crate::test_net). [`a_checkpoint_decodes_as_it_scores`] runs
//! it on a trained checkpoint.

use super::{PREFILL_CHUNK, open, prefill};
use crate::common::cli::{load_model, load_model_config};
use crate::dataset::{VOCAB, VOCAB_SIZE};
use crate::test_net::{N_LATENTS, tiny_net};
use crate::training::Run;
use burn::prelude::*;
use burn_mamba::prelude::*;
use burn_stack::examples::tiny_stories::sample::Prefill;
use burn_stack::utils::ClassCursors;
use burn_stack::utils::test_helpers::test_device;

/// Characters decoded after each opening of the tiny network.
const CONTINUATION: usize = 64;

fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    (a - b).abs().max().into_scalar::<f32>()
}

/// `tol` is relative to the scale of `want`.
fn assert_close(got: Tensor<2>, want: Tensor<2>, tol: f32, what: &str) {
    let scale = 1.0 + want.clone().abs().max().into_scalar::<f32>();
    let d = max_abs_diff(got, want);
    assert!(d <= tol * scale, "{what}: differs by {d} (scale {scale})");
}

/// `net` with other values in its class latents.
fn shifted_latents(net: &MambaVocabNet) -> MambaVocabNet {
    let mut shifted = net.clone();
    let MambaVocabNet::Mamba3(inner) = &mut shifted else {
        panic!("a Mamba-3 network")
    };
    let latents = inner.layers.class_latents_emb.take().expect("class latents");
    inner.layers.class_latents_emb = Some(latents.map(|t| t + 1.0));
    shifted
}

fn eager_run(chunk_len: Option<usize>) -> Run {
    Run {
        ssd_path: MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(chunk_len)),
        graphs: false,
        max_vram_mib: None,
    }
}

fn ids(tokens: &[u8], device: &Device) -> Tensor<2, Int> {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, tokens.len()])
}

/// The logits of the last position.
fn last(logits: Tensor<3>) -> Tensor<2> {
    let rows = logits.dims()[1];
    logits.narrow(1, rows - 1, 1).squeeze_dim::<2>(1)
}

/// The mean bits per character of `targets` under `logits` (one row each).
fn bits(logits: &[Tensor<2>], targets: &[u8]) -> f64 {
    let total: f64 = logits
        .iter()
        .zip(targets)
        .map(|(l, &t)| {
            let log_p = burn::tensor::activation::log_softmax(l.clone(), 1);
            -log_p.narrow(1, t as usize, 1).into_scalar::<f32>() as f64
        })
        .sum();
    total / targets.len() as f64 / std::f64::consts::LN_2
}

/// The checks of the header for one prompt and one continuation. `chunked` is
/// held across calls, as `infer` holds it. Returns the bits per character of
/// the continuation, through the prefill and the decode, and through one
/// `forward`.
#[allow(clippy::too_many_arguments)]
fn check_decode(
    net: &MambaVocabNet,
    shifted: &MambaVocabNet,
    run: &Run,
    device: &Device,
    chunked: &mut Prefill<'_, MambaCaches>,
    tokens: &[u8],
    continuation: &[u8],
    tol: f32,
) -> (f64, f64) {
    let len = tokens.len();
    let (logits, caches, class) = open(net, run, device, Some(tokens), Some(chunked));

    // The definition: `prime` (the latents), then the prompt from its cache.
    let mut primed_class = ClassCursors::stream();
    let (_, primed) = net.prime(1, None, Some(&mut primed_class));
    assert!(primed.is_some(), "the Start latents leave a cache");
    let (want, _) =
        net.forward(ids(tokens, device), primed, run.ssd_path.clone(), Some(&mut primed_class), None);
    let want = last(want);
    assert_close(logits.clone(), want.clone(), tol, &format!("length {len}: prefill vs prime + forward"));

    // The training splice: one forward with fresh cursors.
    let (one_pass, one_pass_caches, one_pass_class) = open(net, run, device, Some(tokens), None);
    assert_close(one_pass, want.clone(), tol, &format!("length {len}: one pass vs prime + forward"));

    // Other latent values, other logits: the opening reads the latents.
    let (moved, _, _) = open(shifted, run, device, Some(tokens), None);
    let d = max_abs_diff(moved, want);
    assert!(d > 1e-2, "length {len}: shifted latents change the logits by {d} only");

    // The decode: teacher-forced steps after each opening, against one
    // `forward` over the prompt and the continuation.
    let whole = [tokens, continuation].concat();
    let (all, _) = net.forward(ids(&whole, device), None, run.ssd_path.clone(), None, None);
    // The rows of the latents come first, then one row per token.
    let lead = all.dims()[1] - whole.len();
    assert_eq!(lead, N_LATENTS, "one row per latent, then one per token");
    let row = |i: usize| all.clone().narrow(1, lead + i, 1).squeeze_dim::<2>(1);
    let mut decoded = vec![logits];
    let mut scored = vec![row(len - 1)];
    let mut openings = [(caches, class), (one_pass_caches, one_pass_class)];
    for (k, &token) in continuation.iter().enumerate().take(continuation.len() - 1) {
        let x = Tensor::<1, Int>::from_ints([token as i32], device);
        let want = row(len + k);
        for (path, (caches, class)) in ["prefill", "one pass"].iter().zip(&mut openings) {
            let (got, next) = net.step(x.clone(), Some(caches.clone()), Some(class));
            *caches = next;
            assert_close(got.clone(), want.clone(), tol, &format!("length {len}, {path}: step {k}"));
            if *path == "prefill" {
                decoded.push(got);
            }
        }
        scored.push(want);
    }
    (bits(&decoded, continuation), bits(&scored, continuation))
}

#[test]
fn the_latents_run_before_the_prompt() {
    let device = test_device();
    device.seed(0);
    let net = tiny_net(&device);
    let shifted = shifted_latents(&net);
    let run = eager_run(None);
    assert!(net.only_start_latents(), "the prefill path needs Start latents only");
    // One prefill across the prompts, as `infer` holds it.
    let mut chunked = prefill(&net, &run, &device);
    for len in [1, PREFILL_CHUNK - 1, PREFILL_CHUNK, PREFILL_CHUNK + 1] {
        // Different ids for every length.
        let tokens: Vec<u8> = (0..len).map(|i| ((i * 5 + len * 3) % VOCAB_SIZE) as u8).collect();
        let continuation: Vec<u8> =
            (0..CONTINUATION).map(|i| ((i * 7 + len) % VOCAB_SIZE) as u8).collect();
        check_decode(&net, &shifted, &run, &device, &mut chunked, &tokens, &continuation, 1e-4);
    }
}

/// The two prompts of the user, each with a continuation that stays on topic.
const CHECKPOINT_PROMPTS: [(&str, &str); 2] = [
    (
        "lily likes cats and dogs. she asked her mom for a dog and her mom said no, so instead she asked",
        " her mom for a cat. her mom said yes, and lily was very happy. she named the cat tom.",
    ),
    (
        "alice and jack walked up the street and met a girl in a red dress. the girl said to them, \"hi, i'm jane. what are your names?\"",
        "\n\"i'm alice, and this is jack,\" said alice. jane smiled and said, \"do you want to play with me?\"",
    ),
];

/// [`check_decode`] on the checkpoint in `$TS_CHECKPOINT` (a run directory:
/// `model_config.json` and `model.bpk`), with the prompts of
/// [`CHECKPOINT_PROMPTS`]. Run it with
/// `TS_CHECKPOINT=<run dir> cargo test --example tiny-stories -- --ignored --nocapture`.
#[test]
#[ignore = "needs a trained checkpoint in $TS_CHECKPOINT"]
fn a_checkpoint_decodes_as_it_scores() {
    let dir = std::path::PathBuf::from(
        std::env::var("TS_CHECKPOINT").expect("TS_CHECKPOINT names a run directory"),
    );
    let device = test_device();
    let config: MambaVocabNetConfig =
        load_model_config(&dir.join("model_config.json")).expect("a model config in the run");
    let net = load_model(&dir, &config, &device).expect("a model in the run");
    let shifted = shifted_latents(&net);
    let run = eager_run(Some(128));
    let mut chunked = prefill(&net, &run, &device);
    for (prompt, continuation) in CHECKPOINT_PROMPTS {
        let (tokens, continuation) = (VOCAB.encode(prompt), VOCAB.encode(continuation));
        let (decoded, scored) =
            check_decode(&net, &shifted, &run, &device, &mut chunked, &tokens, &continuation, 1e-3);
        println!("continuation bits/char: decode {decoded:.4}, forward {scored:.4} | {prompt}");
    }
}
