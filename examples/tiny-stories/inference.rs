//! Sampling from the trained character LM.
//!
//! [`generate`] shows off the library's three execution modes back to back: the
//! class latents a story opens with are replayed by one
//! [`prime`](MambaVocabNet::prime) — no input token, and it already answers with
//! the first character's distribution — a prompt, when there is one, is consumed
//! by one chunkwise [`forward`](MambaVocabNet::forward) (prefill), and every
//! generated character then costs one [`step`](MambaVocabNet::step) against the
//! same cache — O(state) per token, with no growing KV cache. It is
//! `burn_stack`'s
//! [`generate`](burn_stack::examples::tiny_stories::sample::generate) written
//! over this crate's family enum, which the block-generic one cannot dispatch;
//! the decode loop itself is shared
//! ([`decode`](burn_stack::examples::tiny_stories::sample::decode)): it draws
//! every character on the device, and past its first few steps it replays one
//! captured graph of the step and its draw instead of launching it anew
//! (`TS_GRAPH=0` turns that off; the text is the same either way). [`infer`] loads the checkpoint and prints a few stories at different
//! temperatures.
//!
//! One call is one story. A second one starts from a **zero** cache and primes
//! again, which is the only place these examples genuinely reset a cache: the
//! model never saw a story follow another, so continuing into a second one would
//! be as out-of-distribution as the `"\n\n"` seed the latents replaced.

use crate::AppArgs;
use crate::dataset::VOCAB;
use crate::training::ssd_path;
use burn::prelude::*;
use burn_mamba::prelude::{MambaVocabNet, MambaVocabNetConfig};
use burn_stack::examples::tiny_stories::sample::decode;
use burn_stack::utils::ClassCursors;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Temperatures sampled by [`infer`], from near-greedy to loose.
const TEMPERATURES: &[f64] = &[0.5, 0.8, 1.0];

/// Characters generated per sample by [`infer`].
const SAMPLE_CHARS: usize = 800;

/// Load the trained LM and print one story per temperature, plus one
/// continuation of a fixed prompt.
pub fn infer(model_config: MambaVocabNetConfig, infer_device: Device, app_args: &AppArgs) {
    let model: MambaVocabNet = app_args
        .load_model(&model_config, &infer_device)
        .expect("no trained model in the artifacts directory; run with --training first");

    let out_dir = app_args.artifacts_path.join("inference");
    std::fs::create_dir_all(&out_dir).expect("failed to create the inference directory");

    // Per-sample wall time, opening included.
    let per_char = |t: Instant| t.elapsed().as_secs_f64() * 1e3 / SAMPLE_CHARS as f64;

    for (i, &temperature) in TEMPERATURES.iter().enumerate() {
        // Nothing is fed in: the class latents are the model's "a story starts
        // here", and `prime` replays them.
        let t = Instant::now();
        let text = generate(
            &model,
            &infer_device,
            None,
            SAMPLE_CHARS,
            temperature,
            i as u64,
        );
        let ms = per_char(t);
        println!("\n--- unprompted, temperature {temperature} ({ms:.2} ms/char) ---\n{text}");
        let path = out_dir.join(format!("sample-t{temperature}.txt"));
        std::fs::write(&path, &text).expect("failed to write the sample");
    }

    let prompt = "once upon a time, there was a little girl named lily. she";
    let t = Instant::now();
    let text = generate(
        &model,
        &infer_device,
        Some(prompt),
        SAMPLE_CHARS,
        0.8,
        TEMPERATURES.len() as u64,
    );
    let ms = per_char(t);
    println!("\n--- prompted, temperature 0.8 ({ms:.2} ms/char) ---\n{prompt}{text}");
    let path = out_dir.join("sample-prompted.txt");
    std::fs::write(&path, format!("{prompt}{text}")).expect("failed to write the sample");

    println!("\nsaved {} samples to {out_dir:?}", TEMPERATURES.len() + 1);
}

/// Sample `n_chars` characters of one story, continuing `prompt` when there is
/// one.
///
/// With `prompt: None` the model writes from its own opening: `prime` replays the
/// class latents a story starts with and hands back the distribution of its first
/// character, so nothing has to be fed in.
///
/// A prompt is case-folded and filtered through the alphabet (see
/// [`VOCAB`](crate::dataset::VOCAB)) and must not come out empty; the latents are
/// spliced in front of it by the same cursors, exactly as in training.
/// `temperature` scales the logits before the softmax; `<= 0` samples greedily
/// (argmax). Returns only the generated characters, not the prompt.
pub fn generate(
    model: &MambaVocabNet,
    device: &Device,
    prompt: Option<&str>,
    n_chars: usize,
    temperature: f64,
    seed: u64,
) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // One story: the cursors open the sequence here and are threaded through
    // every call below, so the latents are emitted once.
    let mut class = ClassCursors::stream();

    let (logits, caches) = match prompt {
        // Prefill: one chunkwise pass over the latents and the whole prompt,
        // keeping its cache and the logits of its last character (what the next
        // character is drawn from).
        Some(prompt) => {
            let tokens = VOCAB.encode(prompt);
            assert!(
                !tokens.is_empty(),
                "the prompt has no character inside the alphabet: {prompt:?}"
            );
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let input = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, ids.len()]);
            let (logits, caches) = model.forward(input, None, ssd_path(), Some(&mut class), None);
            let last = logits.dims()[1] - 1;
            (logits.narrow(1, last, 1).squeeze_dim::<2>(1), Some(caches))
        }
        // Seedless: the latents alone, which already predict the first character.
        None => {
            let (logits, caches) = model.prime(1, None, Some(&mut class));
            (
                logits
                    .expect("the model has no class latents to prime from; pass a prompt instead"),
                caches,
            )
        }
    };

    // Decode: one `step` per character, against that same cache — replayed from
    // one captured graph after the first few, unless `TS_GRAPH=0` (or a class
    // latent is still to land).
    let caches = caches.expect("the opening leaves a cache");
    let capture = model.only_start_latents() && !matches!(std::env::var("TS_GRAPH").as_deref(), Ok("0"));
    // Safety: the step reads nothing but its arguments and `model`, which it
    // borrows for the whole call.
    unsafe {
        decode(
            device,
            logits,
            caches,
            class,
            n_chars,
            temperature,
            &mut rng,
            capture,
            |x, caches, class| model.step(x, Some(caches), class),
        )
    }
}
