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
//!
//! **Re-mentions** ([`remention_marks`]) isolate the recall of names and
//! objects. They are the words of the far part that also occur in the own
//! prompt, and that only few stories hold (so not the function words). Their
//! bits after the own prompt, with no prompt, and after the other prompts show
//! how well the model recalls an entity of the prompt when the story names it
//! again. The overall far use hides this part.

use crate::AppArgs;
use crate::dataset::{Split, VOCAB, stories};
use crate::training::Run;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn_mamba::prelude::{MambaVocabNet, MambaVocabNetConfig};
use std::collections::{HashMap, HashSet};

/// Prompt characters.
pub const PROMPT: usize = 256;

/// Continuation characters.
pub const CONTINUATION: usize = 256;

/// The characters of the join segment, at the start of the continuation.
pub const JOIN: usize = 64;

/// Sequences per `forward` when no `--context-batch` is given. Small, so the
/// measure can run on the GPU beside a training. Run it on the GPU: on flex,
/// 2048 stories take ~36 min (h16), too slow to be useful. On the GPU beside
/// the training of h16 × g2, a batch of 8 kept the card total at 4.7 GiB or
/// less (at most ~0.9 GiB above the training alone), at 85–200 s per
/// checkpoint. The training then ran ~1.8× slower.
pub const BATCH: usize = 4;

/// The coherence boosting strengths α of `--context-use`. α = 0 is the plain
/// score, and it comes first.
pub const ALPHAS: [f64; 7] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0];
const _: () = assert!(ALPHAS[0] == 0.0, "the first α is the plain score");

/// The shortest word that counts as a re-mention ([`remention_marks`]).
pub const REMENTION_LEN: usize = 3;

/// A word counts as a re-mention only if fewer than this share of the scored
/// stories hold it (in the prompt or the continuation). On the validation
/// stories, the names and the objects are below it (lily 0.15, tom 0.11, ball
/// 0.09, dog 0.16), and the function words and the common words are above it
/// (the 0.99, she 0.48, mom 0.28).
pub const REMENTION_DF: f64 = 0.2;

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
    /// `marked[k][m][i]`: the summed bits (not the mean) of continuation `i`
    /// over the characters of `marks[m][i]`, at `alphas[0]`. Score `k = 0` has
    /// no prompt, and `k = 1 + j` follows the prompts of pairing `j`.
    pub marked: Vec<Vec<Vec<f64>>>,
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
/// the first `join` characters of one continuation, and of the rest. For each
/// mark set of `marks` (`marks[m][i][k]`: character `k` of continuation `i` is
/// in set `m`), the scores also hold the summed bits over the marked
/// characters ([`Scores::marked`]).
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
    marks: &[Vec<Vec<bool>>],
) -> Scores {
    let (n, p, c) = (continuations.len(), prompts[0].len(), continuations[0].len());
    assert!(
        prompts.iter().all(|t| t.len() == p) && continuations.iter().all(|t| t.len() == c),
        "all prompts have one length, and all continuations another"
    );
    assert!(p >= 1 && 0 < join && join < c, "a prompt, and a join shorter than the continuation");
    assert!(
        marks.iter().all(|set| set.len() == n && set.iter().all(|row| row.len() == c)),
        "one mark per character of each continuation"
    );
    // Per score and story: the summed negative log-likelihood of the join, of
    // the far part, and of each mark set. Score 0 is the one with no prompt.
    // Score `1 + j·alphas.len() + a` is pairing `j` at `alphas[a]`.
    let m = marks.len();
    let width = 2 + m;
    let n_scores = 1 + pairings.len() * alphas.len();
    let mut nll_kns = vec![0.0f64; n_scores * n * width];
    let order: Vec<usize> = (0..n).collect();
    for rows in order.chunks(batch) {
        let b = rows.len();
        let targets: Vec<i32> = rows
            .iter()
            .flat_map(|&i| continuations[i].iter().map(|&t| t as i32))
            .collect();
        let targets = Tensor::<1, Int>::from_ints(targets.as_slice(), device).reshape([b, c, 1]);
        // The marks of the rows as 0/1: `[b, m, c]`.
        let mask_bmc = (m > 0).then(|| {
            let mask: Vec<f32> = rows
                .iter()
                .flat_map(|&i| marks.iter().flat_map(move |set| set[i].iter().map(|&x| f32::from(u8::from(x)))))
                .collect();
            Tensor::<1>::from_floats(mask.as_slice(), device).reshape([b, m, c])
        });
        // The summed negative log-likelihood of the join, of the far part and
        // of each mark set, per row: `[b, 2 + m]`.
        let nll_bs = |log_p: Tensor<3>| {
            let nll_bc = log_p.gather(2, targets.clone()).reshape([b, c]).neg();
            let near_b1 = nll_bc.clone().narrow(1, 0, join).sum_dim(1);
            let far_b1 = nll_bc.clone().narrow(1, join, c - join).sum_dim(1);
            let mut sums = vec![near_b1, far_b1];
            if let Some(mask_bmc) = &mask_bmc {
                sums.push((nll_bc.reshape([b, 1, c]) * mask_bmc.clone()).sum_dim(2).reshape([b, m]));
            }
            Tensor::cat(sums, 1)
        };
        let log_p = |p: usize, pair: &dyn Fn(usize) -> usize| {
            log_probs(model, run, device, prompts, continuations, rows, p, pair)
        };
        let none = log_p(0, &|i| i);
        let mut nll = vec![nll_bs(none.clone())];
        for &pair in pairings {
            let full = log_p(p, pair);
            for &alpha in alphas {
                let boosted = if alpha == 0.0 {
                    full.clone()
                } else {
                    log_softmax(full.clone().mul_scalar(1.0 + alpha) - none.clone().mul_scalar(alpha), 2)
                };
                nll.push(nll_bs(boosted));
            }
        }
        // One read-back per batch.
        let nll: Vec<f64> = Tensor::stack::<3>(nll, 0).into_data().try_into_vec_as().unwrap();
        for (k, nll_bs) in nll.chunks(b * width).enumerate() {
            for (&i, sums) in rows.iter().zip(nll_bs.chunks(width)) {
                nll_kns[(k * n + i) * width..(k * n + i + 1) * width].copy_from_slice(sums);
            }
        }
    }
    let ln2 = std::f64::consts::LN_2;
    let sums = |k: usize, i: usize| &nll_kns[(k * n + i) * width..(k * n + i + 1) * width];
    let bits = |k: usize| -> Vec<Bits> {
        (0..n)
            .map(|i| Bits {
                join: sums(k, i)[0] / join as f64 / ln2,
                far: sums(k, i)[1] / (c - join) as f64 / ln2,
            })
            .collect()
    };
    let marked = |k: usize| -> Vec<Vec<f64>> {
        (0..m).map(|s| (0..n).map(|i| sums(k, i)[2 + s] / ln2).collect()).collect()
    };
    Scores {
        none: bits(0),
        pairings: (0..pairings.len())
            .map(|j| (0..alphas.len()).map(|a| bits(1 + j * alphas.len() + a)).collect())
            .collect(),
        marked: std::iter::once(marked(0))
            .chain((0..pairings.len()).map(|j| marked(1 + j * alphas.len())))
            .collect(),
    }
}

/// The ratio `Σ xs / Σ ns` and its standard error, with the stories as the
/// clusters: `√(Σ (x − r·n)²) / Σ n`. `xs[i]` sums over the `ns[i]`
/// characters of story `i`.
pub fn ratio_se(xs: &[f64], ns: &[usize]) -> (f64, f64) {
    let total: f64 = ns.iter().map(|&k| k as f64).sum();
    let r = xs.iter().sum::<f64>() / total;
    let var: f64 = xs.iter().zip(ns).map(|(x, &k)| (x - r * k as f64).powi(2)).sum();
    (r, var.sqrt() / total)
}

/// The words of `tokens`: `(start, word)` for each run of at least
/// [`REMENTION_LEN`] letters.
fn words(tokens: &[u8]) -> Vec<(usize, String)> {
    let text: Vec<char> = tokens.iter().map(|&t| VOCAB.character(t)).collect();
    let mut out = Vec::new();
    let mut k = 0;
    while k < text.len() {
        let start = k;
        while k < text.len() && text[k].is_ascii_lowercase() {
            k += 1;
        }
        if k - start >= REMENTION_LEN {
            out.push((start, text[start..k].iter().collect()));
        }
        k = k.max(start + 1);
    }
    out
}

/// The re-mention marks of the continuations. Mark set 0 holds every letter,
/// and mark set 1 the first letter, of each word of the far part (from `join`
/// on) that also occurs in the own prompt. A word has [`REMENTION_LEN`]
/// letters or more, and fewer than [`REMENTION_DF`] of the stories hold it
/// (in the prompt or the continuation). So the data selects the names and the
/// objects, with no word list. A word cut by the start of the continuation
/// starts before `join`, so it does not count.
pub fn remention_marks(prompts: &[Vec<u8>], continuations: &[Vec<u8>], join: usize) -> [Vec<Vec<bool>>; 2] {
    let n = prompts.len();
    let prompt_words: Vec<HashSet<String>> =
        prompts.iter().map(|p| words(p).into_iter().map(|(_, w)| w).collect()).collect();
    let mut df: HashMap<String, usize> = HashMap::new();
    for (p, c) in prompts.iter().zip(continuations) {
        let seen: HashSet<String> = words(p).into_iter().chain(words(c)).map(|(_, w)| w).collect();
        for w in seen {
            *df.entry(w).or_default() += 1;
        }
    }
    let rare = |w: &str| (df[w] as f64) < REMENTION_DF * n as f64;
    let mut sets = [Vec::with_capacity(n), Vec::with_capacity(n)];
    for (c, prompt_words) in continuations.iter().zip(&prompt_words) {
        let (mut letters, mut first) = (vec![false; c.len()], vec![false; c.len()]);
        for (start, w) in words(c) {
            if start >= join && prompt_words.contains(&w) && rare(&w) {
                letters[start..start + w.len()].fill(true);
                first[start] = true;
            }
        }
        sets[0].push(letters);
        sets[1].push(first);
    }
    sets
}

/// Load the trained model and measure its context use over the first
/// `n_stories` validation stories (the stories of the validation loss). One
/// `forward` takes `batch` sequences (`--context-batch`). Also score with no
/// prompt, with coherence boosting at each α of [`ALPHAS`], and the
/// re-mentions ([`remention_marks`]). Print it, and save it as
/// `context-use.json` in the artifacts directory, with the scores of each
/// story at α = 0 in `context-stories.jsonl`.
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
    let marks = remention_marks(&prompts, &continuations, JOIN);
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
        &marks,
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
    // The re-mentions, pooled per character. `marked[k]`: no prompt (0), own
    // (1), other i+1 (2), other i+2 (3).
    let counts = marks.each_ref().map(|set| set.iter().map(|row| row.iter().filter(|&&x| x).count()).collect::<Vec<_>>());
    let rem_stories = counts[0].iter().filter(|&&k| k > 0).count();
    println!(
        "re-mention: far-part words of {REMENTION_LEN}+ letters of the own prompt, held by < {REMENTION_DF} of the \
         stories: {} letters ({} words) in {rem_stories} stories",
        counts[0].iter().sum::<usize>(),
        counts[1].iter().sum::<usize>()
    );
    println!("  bits/char        own     none  other (i+1)  other (i+2)   gain (none − own)    use (i+1)");
    let rem: Vec<String> = ["letters", "first"]
        .iter()
        .enumerate()
        .map(|(s, name)| {
            let k_bits = |k: usize| ratio_se(&scores.marked[k][s], &counts[s]).0;
            let diff = |a: usize, b: usize| {
                let xs: Vec<f64> = (0..n).map(|i| scores.marked[a][s][i] - scores.marked[b][s][i]).collect();
                ratio_se(&xs, &counts[s])
            };
            let (gain, use_) = (diff(0, 1), diff(2, 1));
            println!(
                "  {name:<9} {:>8.4} {:>8.4} {:>12.4} {:>12.4}   {:>+8.4} ± {:.4} {:>+8.4} ± {:.4}",
                k_bits(1),
                k_bits(0),
                k_bits(2),
                k_bits(3),
                gain.0,
                gain.1,
                use_.0,
                use_.1
            );
            format!(
                "\"{name}\":{{\"n\":{},\"own\":{},\"none\":{},\"other\":{},\"other2\":{},\"gain\":[{},{}],\"use\":[{},{}]}}",
                counts[s].iter().sum::<usize>(),
                k_bits(1),
                k_bits(0),
                k_bits(2),
                k_bits(3),
                gain.0,
                gain.1,
                use_.0,
                use_.1
            )
        })
        .collect();
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
         \"gain_far\":[{},{}],\"penalty_far\":[{},{}],\"use_far\":[{},{}],\"boost\":[{}],\
         \"remention\":{{{}}}}}\n",
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
        boost.join(","),
        rem.join(",")
    );
    let path = app_args.artifacts_path.join("context-use.json");
    std::fs::write(&path, json).expect("failed to write context-use.json");
    println!("saved to {path:?}");

    // One line per story, at α = 0. `story` is its index in the validation
    // split, so two models pair up on it. `rem*`: the re-mentions (`rem` the
    // letters, `rem1` the first letters), their count and their summed bits.
    let lines: String = (0..n)
        .map(|i| {
            let (own, other, other2, none) = (story(0, i), story(1, i), story(2, i), scores.none[i]);
            let rem_fields = |s: usize, name: &str| {
                let k = |k: usize| scores.marked[k][s][i];
                format!(
                    ",\"{name}_n\":{},\"{name}_own\":{},\"{name}_none\":{},\"{name}_other\":{},\"{name}_other2\":{}",
                    counts[s][i],
                    k(1),
                    k(0),
                    k(2),
                    k(3)
                )
            };
            format!(
                "{{\"story\":{},\"own_join\":{},\"own_far\":{},\"none_join\":{},\"none_far\":{},\
                 \"other_join\":{},\"other_far\":{},\"other2_join\":{},\"other2_far\":{}{}{}}}\n",
                indices[i],
                own.join,
                own.far,
                none.join,
                none.far,
                other.join,
                other.far,
                other2.join,
                other2.far,
                rem_fields(0, "rem"),
                rem_fields(1, "rem1")
            )
        })
        .collect();
    let path = app_args.artifacts_path.join("context-stories.jsonl");
    std::fs::write(&path, lines).expect("failed to write context-stories.jsonl");
    println!("saved to {path:?}");
}

#[cfg(test)]
mod tests;
