//! Sampling from the trained character LM.
//!
//! [`generate`] is the reusable sampler and shows off the library's two
//! execution modes back to back: the prompt is consumed by one chunkwise
//! [`forward`](MambaVocabNet::forward) (prefill), and every generated character
//! then costs one [`step`](MambaVocabNet::step) against the same cache — O(state)
//! per token, with no growing KV cache. [`infer`] loads the checkpoint and
//! prints a few stories at different temperatures.

use crate::AppArgs;
use crate::common::device::FloatElement;
use crate::dataset::{STORY_SEPARATOR, VOCAB, VOCAB_SIZE};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::activation::softmax;
use burn_mamba::prelude::{Mamba3SsdPath, MambaSsdPath, MambaVocabNet, MambaVocabNetConfig};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

    for (i, &temperature) in TEMPERATURES.iter().enumerate() {
        // The document boundary is the model's "start of story" prompt.
        let text = generate(
            &model,
            &infer_device,
            STORY_SEPARATOR,
            SAMPLE_CHARS,
            temperature,
            i as u64,
        );
        println!("\n--- unprompted, temperature {temperature} ---\n{text}");
        let path = out_dir.join(format!("sample-t{temperature}.txt"));
        std::fs::write(&path, &text).expect("failed to write the sample");
    }

    let prompt = "once upon a time, there was a little girl named lily. she";
    let text = generate(
        &model,
        &infer_device,
        prompt,
        SAMPLE_CHARS,
        0.8,
        TEMPERATURES.len() as u64,
    );
    println!("\n--- prompted, temperature 0.8 ---\n{prompt}{text}");
    let path = out_dir.join("sample-prompted.txt");
    std::fs::write(&path, format!("{prompt}{text}")).expect("failed to write the sample");

    println!("\nsaved {} samples to {out_dir:?}", TEMPERATURES.len() + 1);
}

/// Continue `prompt` with `n_chars` sampled characters.
///
/// The prompt is case-folded and filtered through the alphabet (see
/// [`VOCAB`](crate::dataset::VOCAB)) and must not come out empty. `temperature`
/// scales the logits before the softmax; `<= 0` samples greedily (argmax).
/// Returns only the generated continuation, not the prompt.
pub fn generate(
    model: &MambaVocabNet,
    device: &Device,
    prompt: &str,
    n_chars: usize,
    temperature: f64,
    seed: u64,
) -> String {
    let tokens = VOCAB.encode(prompt);
    assert!(
        !tokens.is_empty(),
        "the prompt has no character inside the alphabet: {prompt:?}"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Prefill: one chunkwise pass over the whole prompt, keeping its cache and
    // the logits of its last character (what the next character is drawn from).
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let prompt_len = ids.len();
    let input = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, prompt_len]);
    let ssd_path = MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None));
    let (logits, caches) = model.forward(input, None, ssd_path, None);
    let mut logits = logits.narrow(1, prompt_len - 1, 1).squeeze_dim::<2>(1); // [1, VOCAB_SIZE]
    let mut caches = Some(caches);

    // Decode: one `step` per character, against that same cache.
    let mut out = String::with_capacity(n_chars);
    for _ in 0..n_chars {
        let token = sample(logits, temperature, &mut rng);
        out.push(VOCAB.character(token));
        let next = Tensor::<1, Int>::from_ints([token as i32], device);
        let (next_logits, next_caches) = model.step(next, caches.take(), None);
        logits = next_logits;
        caches = Some(next_caches);
    }
    out
}

/// Draw one token from `logits` (`[1, VOCAB_SIZE]`): temperature-scaled
/// multinomial sampling, or argmax when `temperature <= 0`.
fn sample(logits: Tensor<2>, temperature: f64, rng: &mut ChaCha8Rng) -> u8 {
    assert_eq!([1, VOCAB_SIZE], logits.dims());
    if temperature <= 0.0 {
        let best = logits.argmax(1).into_data().try_to_vec::<i32>().unwrap();
        return best[0] as u8;
    }
    let probs = to_host(softmax(logits / temperature, 1));
    let threshold: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (token, p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return token as u8;
        }
    }
    // Only reachable when the probabilities sum to slightly under 1 (rounding).
    (VOCAB_SIZE - 1) as u8
}

/// Read a float tensor back to a host `Vec<f32>` (dtype-agnostic).
fn to_host<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .try_to_vec::<FloatElement>()
        .unwrap()
        .into_iter()
        .map(|x| x.elem::<f32>())
        .collect()
}
