//! How much a trained model uses the earlier part of a story when it predicts
//! the later part (`--context-use`).
//!
//! The validation loss mixes two kinds of prediction. The local one
//! (spelling, grammar, stock phrases) is most of it. The use of the earlier
//! story (who is in it, what was asked, what happened) is a small part, and it
//! is what keeps a continuation coherent. This measure isolates the second.
//!
//! Each validation story with at least [`PROMPT`] + [`CONTINUATION`]
//! characters gives a prompt `P` (its first [`PROMPT`] characters) and a
//! continuation `C` (the next [`CONTINUATION`] characters). [`score`] scores
//! `C` as the validation scores it (one `forward`, the class latents in
//! front): after its own `P`, and after the `P` of another story (story `i`
//! reads the prompt of story `i + 1`, and in a second pairing of story
//! `i + 2`, whose spread is the noise). The use is the difference, in bits
//! per character of `C`:
//!
//! `use = bits(C | other P) − bits(C | own P)`
//!
//! It is 0 when the model ignores the prompt. `C` has two segments, reported
//! apart:
//! - The join, the first [`JOIN`] characters: mostly the local syntax across
//!   the cut (the rest of a word or a sentence).
//! - The far part, the rest: there, only the information that the model
//!   carries over from the prompt (names, objects, events) makes `C` cheaper
//!   after its own prompt.

use crate::AppArgs;
use crate::dataset::{Split, VOCAB, stories};
use crate::training::Run;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn_mamba::prelude::{MambaVocabNet, MambaVocabNetConfig};

/// Prompt characters.
pub const PROMPT: usize = 256;

/// Continuation characters.
pub const CONTINUATION: usize = 256;

/// The characters of the join segment, at the start of the continuation.
pub const JOIN: usize = 64;

/// Sequences per `forward` when no `--context-batch` is given. Small, because
/// the measure often runs on the host (flex) next to a training.
pub const BATCH: usize = 4;

/// The mean bits per character of the two segments of the continuations.
#[derive(Clone, Copy, Debug)]
pub struct Bits {
    /// The join segment.
    pub join: f64,
    /// The far part.
    pub far: f64,
}

/// Score each continuation after the prompt that `pair` gives it:
/// `continuations[i]` follows `prompts[pair(i)]`, with the class latents in
/// front, as the validation scores it. All prompts have one length, and all
/// continuations another. One `forward` takes `batch` sequences. Returns the
/// mean bits per character of the first `join` characters of the
/// continuations, and of the rest.
#[allow(clippy::too_many_arguments)]
pub fn score(
    model: &MambaVocabNet,
    run: &Run,
    device: &Device,
    prompts: &[Vec<u8>],
    continuations: &[Vec<u8>],
    join: usize,
    batch: usize,
    pair: impl Fn(usize) -> usize,
) -> Bits {
    let (n, p, c) = (continuations.len(), prompts[0].len(), continuations[0].len());
    assert!(
        prompts.iter().all(|t| t.len() == p) && continuations.iter().all(|t| t.len() == c),
        "all prompts have one length, and all continuations another"
    );
    assert!(p >= 1 && 0 < join && join < c, "a prompt, and a join shorter than the continuation");
    let (mut near, mut far) = (0.0, 0.0);
    let order: Vec<usize> = (0..n).collect();
    for rows in order.chunks(batch) {
        let b = rows.len();
        let ids: Vec<i32> = rows
            .iter()
            .flat_map(|&i| prompts[pair(i)].iter().chain(&continuations[i]))
            .map(|&t| t as i32)
            .collect();
        let x = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([b, p + c]);
        let (logits, _) = model.forward(x, None, run.ssd_path.clone(), None, None);
        // The rows of the latents come first, then one row per token. The row
        // of token `p + k − 1` predicts `continuations[i][k]`.
        let lead = logits.dims()[1] - (p + c);
        let log_p = log_softmax(logits.narrow(1, lead + p - 1, c), 2);
        let targets: Vec<i32> = rows
            .iter()
            .flat_map(|&i| continuations[i].iter().map(|&t| t as i32))
            .collect();
        let targets = Tensor::<1, Int>::from_ints(targets.as_slice(), device).reshape([b, c, 1]);
        let nll = log_p.gather(2, targets).reshape([b, c]).neg();
        near += nll.clone().narrow(1, 0, join).sum().into_scalar::<f32>() as f64;
        far += nll.narrow(1, join, c - join).sum().into_scalar::<f32>() as f64;
    }
    let ln2 = std::f64::consts::LN_2;
    Bits {
        join: near / (n * join) as f64 / ln2,
        far: far / (n * (c - join)) as f64 / ln2,
    }
}

/// Load the trained model and measure its context use over the first
/// `n_stories` validation stories (the stories of the validation loss), with
/// `batch` sequences per `forward` (`--context-batch`). Print it, and save it
/// as `context-use.json` in the artifacts directory.
pub fn context_use(
    model_config: MambaVocabNetConfig,
    device: Device,
    app_args: &AppArgs,
    run: &Run,
    n_stories: usize,
    batch: usize,
) {
    let model: MambaVocabNet = app_args
        .load_model(&model_config, &device)
        .expect("no trained model in the artifacts directory; run with --training first");
    let (prompts, continuations): (Vec<Vec<u8>>, Vec<Vec<u8>>) = stories(Split::Valid, n_stories)
        .iter()
        .map(|story| VOCAB.encode(story))
        .filter(|tokens| tokens.len() >= PROMPT + CONTINUATION)
        .map(|tokens| (tokens[..PROMPT].to_vec(), tokens[PROMPT..PROMPT + CONTINUATION].to_vec()))
        .unzip();
    let n = prompts.len();
    assert!(
        n >= 2,
        "context use needs two stories of {} characters or more",
        PROMPT + CONTINUATION
    );
    let score = |pair: fn(usize, usize) -> usize| {
        score(&model, run, &device, &prompts, &continuations, JOIN, batch, |i| pair(i, n))
    };
    let own = score(|i, _| i);
    // Two pairings (story `i` reads the prompt of story `i + 1`, then of
    // `i + 2`): their spread is the noise of the use.
    let other = score(|i, n| (i + 1) % n);
    let other2 = score(|i, n| (i + 2) % n);

    println!(
        "context use: {n} of {n_stories} validation stories, prompt {PROMPT} chars, \
         continuation {CONTINUATION} chars"
    );
    println!("  bits/char            own prompt  other (i+1)  other (i+2)   use (i+1)  use (i+2)");
    let segments = [
        (format!("join (0..{JOIN})"), own.join, other.join, other2.join),
        (format!("far ({JOIN}..{CONTINUATION})"), own.far, other.far, other2.far),
    ];
    for (name, own, other, other2) in segments {
        println!(
            "  {name:<18} {own:>12.4} {other:>12.4} {other2:>12.4}   {:>+9.4} {:>+10.4}",
            other - own,
            other2 - own
        );
    }
    let json = format!(
        "{{\"stories\":{n},\"prompt\":{PROMPT},\"continuation\":{CONTINUATION},\"join\":{JOIN},\
         \"own_join\":{},\"own_far\":{},\"other_join\":{},\"other_far\":{},\
         \"other2_join\":{},\"other2_far\":{}}}\n",
        own.join, own.far, other.join, other.far, other2.join, other2.far
    );
    let path = app_args.artifacts_path.join("context-use.json");
    std::fs::write(&path, json).expect("failed to write context-use.json");
    println!("saved to {path:?}");
}

#[cfg(test)]
mod tests;
