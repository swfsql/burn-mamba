//! Sampling from the trained H-Net, and the chunk view.
//!
//! [`generate`] opens a story in one of two ways:
//!
//! 1. With no prompt, one [`prime`](MambaHNetVocabNet::prime) steps the start
//!    rows of the network. Their last output already gives the distribution of
//!    the first character.
//! 2. With a prompt, one [`forward_opened`](MambaHNetVocabNet::forward_opened)
//!    runs the start rows and the prompt chunkwise (the prefill).
//!
//! Each generated character then costs one [`step`](MambaHNetVocabNet::step):
//! O(state) per character. The decode loop is the shared
//! [`decode`](burn_stack::examples::tiny_stories::sample::decode). A masked
//! step runs the same launches at every call, so the loop can replay it from a
//! captured graph. A gathered step reads the decisions on the host, so it runs
//! eagerly.
//!
//! **The chunk view** (`-- --show-chunks`). `forward` from a cache equals
//! `step` unrolled, so the decisions of the decode are those of one `forward`
//! over the finished story. [`show_chunks`] runs that `forward` and marks each
//! character where a stage starts a chunk.

use crate::AppArgs;
use crate::dataset::{VOCAB, pair};
use crate::model::HNetStoriesConfig;
use crate::training::Run;
use burn::prelude::*;
use burn_mamba::prelude::MambaHNetVocabNet;
use burn_stack::examples::tiny_stories::sample::decode;
use burn_stack::modules::hnet::StepMode;
use burn_stack::utils::ClassCursors;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Temperatures sampled by [`infer`], from near-greedy to loose.
const TEMPERATURES: &[f64] = &[0.5, 0.8, 1.0];

/// Characters generated per sample by [`infer`].
const SAMPLE_CHARS: usize = 600;

/// Prompts [`infer`] continues when no `--prompt` is given.
const PROMPTS: &[&str] = &[
    "Once upon a time, there was a little girl named Lily. She",
    "Tom and his dog went to the park. They",
];

/// The temperature of the prompted continuations when no `--temperature` is
/// given.
pub const PROMPT_TEMPERATURE: f64 = 0.8;

/// The prompted continuations of [`infer`] (`--prompt`, `--temperature`).
pub struct Prompting {
    /// The prompts. Empty selects [`PROMPTS`].
    pub prompts: Vec<String>,
    /// The sampling temperature of the continuations.
    pub temperature: f64,
}

/// Load the trained H-Net and print one story per temperature, plus one
/// continuation of each prompt.
pub fn infer(model_config: HNetStoriesConfig, device: Device, app_args: &AppArgs, run: &Run, prompting: &Prompting) {
    let model: MambaHNetVocabNet = app_args
        .load_model(&model_config, &device)
        .expect("no trained model in the artifacts directory: train first (--training)");
    for (i, &temperature) in TEMPERATURES.iter().enumerate() {
        let story = generate(&model, run, &device, None, SAMPLE_CHARS, temperature, i as u64);
        println!("--- story (temperature {temperature}) ---\n{story}\n");
    }
    let prompts: Vec<String> = match prompting.prompts.is_empty() {
        true => PROMPTS.iter().map(|p| p.to_string()).collect(),
        false => prompting.prompts.clone(),
    };
    for (i, prompt) in prompts.iter().enumerate() {
        let text = generate(&model, run, &device, Some(prompt), SAMPLE_CHARS, prompting.temperature, 100 + i as u64);
        println!("--- continuation (temperature {}) ---\n{prompt}{text}\n", prompting.temperature);
    }
}

/// Sample `n_chars` characters of one story, after `prompt` when there is one.
/// Returns only the generated characters (with the chunk marks of
/// [`show_chunks`] under `--show-chunks`).
pub fn generate(
    model: &MambaHNetVocabNet,
    run: &Run,
    device: &Device,
    prompt: Option<&str>,
    n_chars: usize,
    temperature: f64,
    seed: u64,
) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let prompt_tokens = prompt.map(|p| VOCAB.encode_cased(p)).unwrap_or_default();
    let (logits, caches) = match prompt {
        None => model.prime(1, run.step_mode).expect("the network has no start rows. Pass a prompt."),
        Some(prompt) => {
            assert!(!prompt_tokens.is_empty(), "the prompt has no character inside the alphabet: {prompt:?}");
            let (logits, caches, _) = model.forward_opened(tokens(&prompt_tokens, device), run.paths.clone(), None);
            let last = logits.dims()[1] - 1;
            (logits.narrow(1, last, 1).squeeze_dim::<2>(1), caches)
        }
    };
    let capture = run.graphs && run.step_mode == StepMode::Masked;
    // SAFETY: `step` reads only `model` (borrowed for the whole call) and its
    // arguments, so every buffer that a captured step reads stays the same.
    let text = unsafe {
        decode(
            device,
            logits,
            caches,
            ClassCursors::stream(),
            n_chars,
            temperature,
            &mut rng,
            capture,
            |token, caches, _class| {
                let (logits, caches, _) = model.step(token, Some(caches), run.step_mode);
                (logits, caches)
            },
        )
    };
    match run.show_chunks {
        true => show_chunks(model, run, device, &prompt_tokens, &text),
        false => text,
    }
}

/// The pairs `[1, len, 2]` of `tokens`.
fn tokens(tokens: &[u8], device: &Device) -> Tensor<3, Int> {
    let pairs: Vec<i32> = tokens.iter().flat_map(|&t| pair(t)).collect();
    Tensor::<1, Int>::from_ints(pairs.as_slice(), device).reshape([1, tokens.len(), 2])
}

/// `text` (generated after `prompt_tokens`), with a mark before each
/// character where a stage starts a chunk. The mark is `|` for stage 0, `‖`
/// where stage 1 also starts one, and `⦀` deeper.
///
/// It runs one `forward_opened` over the start rows, the prompt and the text.
/// Stage `s + 1` routes the rows where stage `s` started a chunk. So the
/// depth of a character is the number of stages that start a chunk at it.
pub fn show_chunks(model: &MambaHNetVocabNet, run: &Run, device: &Device, prompt_tokens: &[u8], text: &str) -> String {
    let text_tokens = VOCAB.encode_cased(text);
    let all: Vec<u8> = prompt_tokens.iter().chain(&text_tokens).copied().collect();
    if all.is_empty() {
        return String::new();
    }
    let (_, _, routing) = model.forward_opened(tokens(&all, device), run.paths.clone(), None);
    let n_start = model.n_start();
    // The rows (of the whole opened sequence) that each stage routes.
    let mut rows: Vec<usize> = (0..n_start + all.len()).collect();
    let mut depth = vec![0usize; rows.len()];
    for r in &routing {
        let boundary: Vec<bool> = r.boundary_bs.clone().int().into_data().iter::<i64>().map(|v| v != 0).collect();
        rows = rows.iter().zip(&boundary).filter(|(_, b)| **b).map(|(&row, _)| row).collect();
        for &row in &rows {
            depth[row] += 1;
        }
    }
    let mut out = String::new();
    for (i, &token) in text_tokens.iter().enumerate() {
        match depth[n_start + prompt_tokens.len() + i] {
            0 => {}
            1 => out.push('|'),
            2 => out.push('‖'),
            _ => out.push('⦀'),
        }
        out.push_str(&VOCAB.decode(&[token]));
    }
    out
}
