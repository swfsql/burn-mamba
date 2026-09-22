//! Sampling from the trained character LM.
//!
//! [`generate`] shows off the library's three execution modes back to back: the
//! class latents a story opens with are replayed by one
//! [`prime`](MambaVocabNet::prime) — no input token, and it already answers with
//! the first character's distribution — a prompt, when there is one, is consumed
//! by chunkwise [`forward`](MambaVocabNet::forward)s (prefill), and every
//! generated character then costs one [`step`](MambaVocabNet::step) against the
//! same cache — O(state) per token, with no growing KV cache. It is
//! `burn_stack`'s
//! [`generate`](burn_stack::examples::tiny_stories::sample::generate) written
//! over this crate's family enum, which the block-generic one cannot dispatch;
//! the decode loop itself is shared
//! ([`decode`](burn_stack::examples::tiny_stories::sample::decode)): it draws
//! every character on the device, and past its first few steps it replays one
//! captured graph of the step and its draw instead of launching it anew. So is
//! the prefill ([`prefill`]): the prompt follows the latents in right-padded
//! fixed-shape chunks, one captured graph for every chunk of every prompt it is
//! held across (`--no-graph` turns both captures off; the text is the same
//! either way). [`infer`] loads the checkpoint and prints a few stories at
//! different temperatures, and a few continuations of fixed prompts.
//!
//! One call is one story. A second one starts from a **zero** cache and primes
//! again, which is the only place these examples genuinely reset a cache: the
//! model never saw a story follow another, so continuing into a second one would
//! be as out-of-distribution as the `"\n\n"` seed the latents replaced.

use crate::AppArgs;
use crate::dataset::VOCAB;
use crate::training::Run;
use burn::prelude::*;
use burn_mamba::prelude::{MambaCaches, MambaVocabNet, MambaVocabNetConfig};
use burn_stack::examples::tiny_stories::sample::{Prefill, decode};
use burn_stack::utils::ClassCursors;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Temperatures sampled by [`infer`], from near-greedy to loose.
const TEMPERATURES: &[f64] = &[0.5, 0.8, 1.0];

/// Characters generated per sample by [`infer`].
const SAMPLE_CHARS: usize = 800;

/// Prompts [`infer`] continues, at temperature 0.8.
const PROMPTS: &[&str] = &[
    "once upon a time, there was a little girl named lily. she",
    "tom and his dog went to the park. they",
    "the sun was hot, so the kids",
];

/// Load the trained LM and print one story per temperature, plus one
/// continuation of each of [`PROMPTS`].
pub fn infer(
    model_config: MambaVocabNetConfig,
    infer_device: Device,
    app_args: &AppArgs,
    run: &Run,
) {
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
            run,
            &infer_device,
            None,
            SAMPLE_CHARS,
            temperature,
            i as u64,
            None,
        );
        let ms = per_char(t);
        println!("\n--- unprompted, temperature {temperature} ({ms:.2} ms/char) ---\n{text}");
        let path = out_dir.join(format!("sample-t{temperature}.txt"));
        std::fs::write(&path, &text).expect("failed to write the sample");
    }

    // One prefill for every prompt: they share the latents' opening and, unless
    // `--no-graph`, one captured chunk.
    let mut prefill = prefill(&model, run, &infer_device);
    for (i, prompt) in PROMPTS.iter().enumerate() {
        let t = Instant::now();
        let text = generate(
            &model,
            run,
            &infer_device,
            Some(prompt),
            SAMPLE_CHARS,
            0.8,
            (TEMPERATURES.len() + i) as u64,
            Some(&mut prefill),
        );
        let ms = per_char(t);
        println!("\n--- prompted, temperature 0.8 ({ms:.2} ms/char) ---\n{prompt}{text}");
        let path = out_dir.join(format!("sample-prompted-{i}.txt"));
        std::fs::write(&path, format!("{prompt}{text}")).expect("failed to write the sample");
    }
    println!("\nsaved {} samples to {out_dir:?}", TEMPERATURES.len() + PROMPTS.len());
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
/// (argmax). With a `prefill` (see [`prefill`]) the prompt follows the primed
/// latents in fixed-shape chunks; with none, one `forward` takes the latents
/// and the prompt. Returns only the generated characters, not the prompt.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &MambaVocabNet,
    run: &Run,
    device: &Device,
    prompt: Option<&str>,
    n_chars: usize,
    temperature: f64,
    seed: u64,
    prefill: Option<&mut Prefill<'_, MambaCaches>>,
) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // One story: the cursors open the sequence here and are threaded through
    // every call below, so the latents are emitted once.
    let mut class = ClassCursors::stream();

    let (logits, caches) = match prompt {
        // Prefill, keeping the cache and the logits of the prompt's last
        // character (what the next character is drawn from).
        Some(prompt) => {
            let tokens = VOCAB.encode(prompt);
            assert!(
                !tokens.is_empty(),
                "the prompt has no character inside the alphabet: {prompt:?}"
            );
            // In chunks after the latents, when they are all `Start`s.
            let prefilled = match prefill {
                Some(prefill) if model.only_start_latents() => prefill.run(&tokens, || {
                    let mut class = ClassCursors::stream();
                    let (_, caches) = model.prime(1, None, Some(&mut class));
                    caches.map(|caches| (caches, class))
                }),
                _ => None,
            };
            match prefilled {
                Some((logits, caches, opened)) => {
                    class = opened;
                    (logits, Some(caches))
                }
                // One chunkwise pass over the latents and the whole prompt.
                None => {
                    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                    let input =
                        Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, ids.len()]);
                    let (logits, caches) =
                        model.forward(input, None, run.ssd_path.clone(), Some(&mut class), None);
                    let last = logits.dims()[1] - 1;
                    (logits.narrow(1, last, 1).squeeze_dim::<2>(1), Some(caches))
                }
            }
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
    // one captured graph after the first few, unless `--no-graph` (or a class
    // latent is still to land).
    let caches = caches.expect("the opening leaves a cache");
    let capture = model.only_start_latents() && run.graphs;
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

/// Prompt characters per [`prefill`] chunk: a training window. A replayed chunk
/// costs about the same at any width up to it, so a wide one serves most prompts
/// in one replay.
const PREFILL_CHUNK: usize = 256;

/// A [`Prefill`] of `model` in chunks of [`PREFILL_CHUNK`], captured unless
/// `--no-graph`. Hold it across prompts: they share its opening and its graph.
pub fn prefill<'a>(model: &'a MambaVocabNet, run: &Run, device: &Device) -> Prefill<'a, MambaCaches> {
    let ssd_path = run.ssd_path.clone();
    // Safety: the chunk reads nothing but its arguments, `model`, borrowed for
    // as long as the prefill lives, and `ssd_path`, which it owns.
    unsafe {
        Prefill::new(device, PREFILL_CHUNK, run.graphs, move |x, caches, pad, class| {
            model.forward(x, Some(caches), ssd_path.clone(), Some(class), Some(pad))
        })
    }
}
