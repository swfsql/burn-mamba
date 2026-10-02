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
//!
//! **Coherence boosting** (Malkin et al. 2022) shows whether the readout gives
//! the prompt too little weight. [`score`] also scores `C` with no prompt (the
//! class latents, then `C`). Then it scores `C` after each prompt with the
//! boosted log-probabilities, for each α of [`ALPHAS`]:
//!
//! `log_softmax((1 + α)·log p(· | P, C<k) − α·log p(· | C<k))`
//!
//! α = 0 is the plain score. If an α > 0 makes `bits(C | own P)` lower, the
//! state holds information about the prompt that the readout does not use in
//! full. If no α > 0 does, the limit is what the state holds.

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

/// The coherence boosting strengths α of `--context-use`. α = 0 is the plain
/// score, and it comes first.
pub const ALPHAS: [f64; 7] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0];
const _: () = assert!(ALPHAS[0] == 0.0, "the first α is the plain score");

/// The mean bits per character of the two segments of a continuation (or of
/// many continuations).
#[derive(Clone, Copy, Debug)]
pub struct Bits {
    /// The join segment.
    pub join: f64,
    /// The far part.
    pub far: f64,
}

impl Bits {
    /// The mean over the stories. All continuations have one length, so this
    /// is also the mean per character of all the stories.
    pub fn mean(stories: &[Bits]) -> Bits {
        let n = stories.len() as f64;
        Bits {
            join: stories.iter().map(|s| s.join).sum::<f64>() / n,
            far: stories.iter().map(|s| s.far).sum::<f64>() / n,
        }
    }
}

/// The mean of `xs` and its standard error (the sample deviation / `√n`).
pub fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// The result of [`score`]: the bits of each story, in the order of the
/// continuations.
#[derive(Clone, Debug)]
pub struct Scores {
    /// With no prompt: the class latents, then the continuation.
    pub none: Vec<Bits>,
    /// `pairings[j][a]`: after the prompts of pairing `j`, boosted by
    /// `alphas[a]`.
    pub pairings: Vec<Vec<Vec<Bits>>>,
}

/// The log-probabilities `[rows, c, vocab]` of the continuations of `rows`.
/// Each continuation follows the first `p` characters of the prompt that
/// `pair` gives it, with the class latents in front (as the validation scores
/// it). With `p = 0`, the readout of the last latent predicts the first
/// character.
#[allow(clippy::too_many_arguments)]
fn log_probs(
    model: &MambaVocabNet,
    run: &Run,
    device: &Device,
    prompts: &[Vec<u8>],
    continuations: &[Vec<u8>],
    rows: &[usize],
    p: usize,
    pair: &dyn Fn(usize) -> usize,
) -> Tensor<3> {
    let (b, c) = (rows.len(), continuations[0].len());
    let ids: Vec<i32> = rows
        .iter()
        .flat_map(|&i| prompts[pair(i)][..p].iter().chain(&continuations[i]))
        .map(|&t| t as i32)
        .collect();
    let x = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([b, p + c]);
    let (logits, _) = model.forward(x, None, run.ssd_path.clone(), None, None);
    // The rows of the latents come first, then one row per token. The row of
    // token `p + k − 1` predicts `continuations[i][k]`.
    let lead = logits.dims()[1] - (p + c);
    assert!(lead + p >= 1, "no row predicts the first character: no prompt and no class latents");
    log_softmax(logits.narrow(1, lead + p - 1, c), 2)
}

/// Score each continuation with no prompt, and after the prompt that each
/// pairing gives it: `continuations[i]` follows `prompts[pairings[j](i)]`.
/// After a prompt, score it at each boost strength of `alphas` (see the module
/// header). The class latents are in front, as the validation scores it. All
/// prompts have one length, and all continuations another. One `forward`
/// takes `batch` sequences. Each [`Bits`] holds the mean bits per character of
/// the first `join` characters of one continuation, and of the rest.
#[allow(clippy::too_many_arguments)]
pub fn score(
    model: &MambaVocabNet,
    run: &Run,
    device: &Device,
    prompts: &[Vec<u8>],
    continuations: &[Vec<u8>],
    join: usize,
    batch: usize,
    pairings: &[&dyn Fn(usize) -> usize],
    alphas: &[f64],
) -> Scores {
    let (n, p, c) = (continuations.len(), prompts[0].len(), continuations[0].len());
    assert!(
        prompts.iter().all(|t| t.len() == p) && continuations.iter().all(|t| t.len() == c),
        "all prompts have one length, and all continuations another"
    );
    assert!(p >= 1 && 0 < join && join < c, "a prompt, and a join shorter than the continuation");
    // The summed negative log-likelihood of the join and of the far part of
    // each story, per score. Score 0 is the one with no prompt. Score
    // `1 + j·alphas.len() + a` is pairing `j` at `alphas[a]`.
    let n_scores = 1 + pairings.len() * alphas.len();
    let mut nll_kn2 = vec![[0.0f64; 2]; n_scores * n];
    let order: Vec<usize> = (0..n).collect();
    for rows in order.chunks(batch) {
        let b = rows.len();
        let targets: Vec<i32> = rows
            .iter()
            .flat_map(|&i| continuations[i].iter().map(|&t| t as i32))
            .collect();
        let targets = Tensor::<1, Int>::from_ints(targets.as_slice(), device).reshape([b, c, 1]);
        // The summed negative log-likelihood of the join and of the far part
        // of each row: `[b, 2]`.
        let nll_b2 = |log_p: Tensor<3>| {
            let nll_bc = log_p.gather(2, targets.clone()).reshape([b, c]).neg();
            let near_b1 = nll_bc.clone().narrow(1, 0, join).sum_dim(1);
            let far_b1 = nll_bc.narrow(1, join, c - join).sum_dim(1);
            Tensor::cat(vec![near_b1, far_b1], 1)
        };
        let log_p = |p: usize, pair: &dyn Fn(usize) -> usize| {
            log_probs(model, run, device, prompts, continuations, rows, p, pair)
        };
        let none = log_p(0, &|i| i);
        let mut nll = vec![nll_b2(none.clone())];
        for &pair in pairings {
            let full = log_p(p, pair);
            for &alpha in alphas {
                let boosted = if alpha == 0.0 {
                    full.clone()
                } else {
                    log_softmax(full.clone().mul_scalar(1.0 + alpha) - none.clone().mul_scalar(alpha), 2)
                };
                nll.push(nll_b2(boosted));
            }
        }
        // One read-back per batch.
        let nll: Vec<f64> = Tensor::stack::<3>(nll, 0).into_data().try_into_vec_as().unwrap();
        for (k, nll_b2) in nll.chunks(b * 2).enumerate() {
            for (&i, sums) in rows.iter().zip(nll_b2.chunks(2)) {
                nll_kn2[k * n + i] = [sums[0], sums[1]];
            }
        }
    }
    let ln2 = std::f64::consts::LN_2;
    let bits = |k: usize| -> Vec<Bits> {
        nll_kn2[k * n..(k + 1) * n]
            .iter()
            .map(|&[near, far]| Bits {
                join: near / join as f64 / ln2,
                far: far / (c - join) as f64 / ln2,
            })
            .collect()
    };
    Scores {
        none: bits(0),
        pairings: (0..pairings.len())
            .map(|j| (0..alphas.len()).map(|a| bits(1 + j * alphas.len() + a)).collect())
            .collect(),
    }
}

/// Load the trained model and measure its context use over the first
/// `n_stories` validation stories (the stories of the validation loss). One
/// `forward` takes `batch` sequences (`--context-batch`). Also score with no
/// prompt, and with coherence boosting at each α of [`ALPHAS`]. Print it, and
/// save it as `context-use.json` in the artifacts directory, with the scores
/// of each story at α = 0 in `context-stories.jsonl`.
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
    // `indices`: the index of each scored story in the validation split.
    let (indices, (prompts, continuations)): (Vec<usize>, (Vec<Vec<u8>>, Vec<Vec<u8>>)) =
        stories(Split::Valid, n_stories)
            .iter()
            .map(|story| VOCAB.encode(story))
            .enumerate()
            .filter(|(_, tokens)| tokens.len() >= PROMPT + CONTINUATION)
            .map(|(index, tokens)| {
                let prompt = tokens[..PROMPT].to_vec();
                (index, (prompt, tokens[PROMPT..PROMPT + CONTINUATION].to_vec()))
            })
            .unzip();
    let n = prompts.len();
    assert!(
        n >= 2,
        "context use needs two stories of {} characters or more",
        PROMPT + CONTINUATION
    );
    // The own prompt, then two other pairings: story `i` reads the prompt of
    // story `i + 1`, then of `i + 2`. Their spread is the noise of the use.
    let own = |i: usize| i;
    let other = |i: usize| (i + 1) % n;
    let other2 = |i: usize| (i + 2) % n;
    let scores = score(
        &model,
        run,
        &device,
        &prompts,
        &continuations,
        JOIN,
        batch,
        &[&own, &other, &other2],
        &ALPHAS,
    );
    // The means over the stories, per α.
    let [own, other, other2] =
        [0, 1, 2].map(|j| scores.pairings[j].iter().map(|s| Bits::mean(s)).collect::<Vec<_>>());
    let none = Bits::mean(&scores.none);
    // The far part of each story at α = 0 (pairing i + 1): the gain of the own
    // prompt over no prompt, the penalty of the other prompt, and the use
    // (their sum).
    let story = |j: usize, i: usize| scores.pairings[j][0][i];
    let far = |f: &dyn Fn(usize) -> f64| mean_se(&(0..n).map(f).collect::<Vec<_>>());
    let gain = far(&|i| scores.none[i].far - story(0, i).far);
    let penalty = far(&|i| story(1, i).far - scores.none[i].far);
    let use_ = far(&|i| story(1, i).far - story(0, i).far);

    println!(
        "context use: {n} of {n_stories} validation stories, prompt {PROMPT} chars, \
         continuation {CONTINUATION} chars"
    );
    println!("  bits/char            own prompt  other (i+1)  other (i+2)   use (i+1)  use (i+2)");
    let segments = [
        (format!("join (0..{JOIN})"), own[0].join, other[0].join, other2[0].join),
        (format!("far ({JOIN}..{CONTINUATION})"), own[0].far, other[0].far, other2[0].far),
    ];
    for (name, own, other, other2) in segments {
        println!(
            "  {name:<18} {own:>12.4} {other:>12.4} {other2:>12.4}   {:>+9.4} {:>+10.4}",
            other - own,
            other2 - own
        );
    }
    println!("  no prompt: join {:.4}, far {:.4}", none.join, none.far);
    println!(
        "  gain {:.4} ± {:.4}, penalty {:.4} ± {:.4}, use {:.4} ± {:.4} \
         (far part, i+1, mean ± s.e. over the stories)",
        gain.0, gain.1, penalty.0, penalty.1, use_.0, use_.1
    );
    println!("coherence boosting: (1 + α)·log p(C | P) − α·log p(C | no prompt)");
    println!("       α   own join    own far   use join (i+1)  use far (i+1)  use far (i+2)");
    for (a, alpha) in ALPHAS.iter().enumerate() {
        println!(
            "  {alpha:>6.2} {:>10.4} {:>10.4} {:>16.4} {:>14.4} {:>14.4}",
            own[a].join,
            own[a].far,
            other[a].join - own[a].join,
            other[a].far - own[a].far,
            other2[a].far - own[a].far
        );
    }
    let boost: Vec<String> = ALPHAS
        .iter()
        .enumerate()
        .map(|(a, alpha)| {
            format!(
                "{{\"alpha\":{alpha},\"own_join\":{},\"own_far\":{},\"other_join\":{},\
                 \"other_far\":{},\"other2_join\":{},\"other2_far\":{}}}",
                own[a].join, own[a].far, other[a].join, other[a].far, other2[a].join, other2[a].far
            )
        })
        .collect();
    let json = format!(
        "{{\"stories\":{n},\"valid_stories\":{n_stories},\"prompt\":{PROMPT},\
         \"continuation\":{CONTINUATION},\"join\":{JOIN},\
         \"own_join\":{},\"own_far\":{},\"other_join\":{},\"other_far\":{},\
         \"other2_join\":{},\"other2_far\":{},\"none_join\":{},\"none_far\":{},\
         \"gain_far\":[{},{}],\"penalty_far\":[{},{}],\"use_far\":[{},{}],\"boost\":[{}]}}\n",
        own[0].join,
        own[0].far,
        other[0].join,
        other[0].far,
        other2[0].join,
        other2[0].far,
        none.join,
        none.far,
        gain.0,
        gain.1,
        penalty.0,
        penalty.1,
        use_.0,
        use_.1,
        boost.join(",")
    );
    let path = app_args.artifacts_path.join("context-use.json");
    std::fs::write(&path, json).expect("failed to write context-use.json");
    println!("saved to {path:?}");

    // One line per story, at α = 0. `story` is its index in the validation
    // split, so two models pair up on it.
    let lines: String = (0..n)
        .map(|i| {
            let (own, other, other2, none) = (story(0, i), story(1, i), story(2, i), scores.none[i]);
            format!(
                "{{\"story\":{},\"own_join\":{},\"own_far\":{},\"none_join\":{},\"none_far\":{},\
                 \"other_join\":{},\"other_far\":{},\"other2_join\":{},\"other2_far\":{}}}\n",
                indices[i],
                own.join,
                own.far,
                none.join,
                none.far,
                other.join,
                other.far,
                other2.join,
                other2.far
            )
        })
        .collect();
    let path = app_args.artifacts_path.join("context-stories.jsonl");
    std::fs::write(&path, lines).expect("failed to write context-stories.jsonl");
    println!("saved to {path:?}");
}

#[cfg(test)]
mod tests;
