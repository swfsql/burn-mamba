//! The **shared-suffix contrast** (`--contrast K`): an auxiliary loss that
//! trains the model to use the earlier part of a story.
//!
//! At each training step, beside the packed LM batch, the [`Feed`] cuts `K`
//! train stories into a prompt `P` and a continuation `C`. [`loss`] runs one
//! forward over the prompts, which gives the cache of each row at its cut.
//! From these caches, the continuations run twice:
//!
//! - as they are: row `i` reads `C_i` after `P_i` (the own term),
//! - rolled by one row: row `i − 1` reads `C_i` after `P_{i−1}` (the other
//!   term). The roll is cyclic.
//!
//! So the two terms score the same characters of `C_i` after two prompts (a
//! shared suffix):
//!
//! `Δ_i,s = log p(C_i,s | P_i, C_i,<s) − log p(C_i,s | P_{i−1}, C_i,<s)`
//!
//! `L = λ · 4/(m + 1) · mean_{i,t} −log σ(Δ_i,≤t)`, with
//! `Δ_i,≤t = Σ_{J ≤ s ≤ t} Δ_i,s` over the `m = n − J` scored characters.
//!
//! - **The scale of λ.** The push on `Δ_i,s` is the sum of `σ(−Δ_i,≤t)` over
//!   the prefixes that contain it. At chance (`σ = ½`), the mean pushes all
//!   the own log-probs together by `(m + 1)/4`. The factor `4/(m + 1)` makes
//!   this 1, as the LM loss pushes all its log-probs together. So at λ = 1,
//!   the contrast pushes as hard as the LM loss until the pairs separate. The
//!   other log-probs get the same push, with the other sign.
//! - **One-sided** (`--contrast-one-sided`): the other pass is detached. The
//!   push on the own log-probs does not change, and the other log-probs get
//!   none. So the contrast cannot win by a worse prediction after the
//!   negative (the penalty). It is then a weighted LM loss on `C | P`, with
//!   the weight `σ(−Δ_≤t)` on each prefix.
//! - **The schedule of λ** ([`ContrastConfig::weight_at`]): 0 up to the step
//!   `start`, then a linear ramp to λ over `ramp` steps. An untrained model
//!   has no grammar yet, so it has nothing to gain from the prompt. While the
//!   weight is 0 (and `--contrast-lm` is off), the pass does not run.
//!
//! - `log p(C | P) = log p(C) + PMI(C; P)`. So the fluency of `C` cancels in
//!   `Δ`, and only the part that depends on the prompt stays.
//! - At a scale of 1 on `Δ` (in nats), the true model is the best classifier
//!   of "which prompt wrote `C`". The weight `σ(−Δ)` of a pair goes to 0 when
//!   the model separates it, so the loss is bounded.
//! - The prefix sums give a gradient in the first characters after the start
//!   `J`, where the classifier is not sure yet. A single sum over all of `C`
//!   saturates.
//! - The prompts and the first `J` characters of `C` get no contrast loss.
//!   The gradient still goes back into every step of both prompts, through
//!   the caches (they are not detached). So the loss trains what the model
//!   writes into its state while it reads a prompt.
//!
//! The two passes over `C` are the forward over `concat(P, C)`, split at the
//! cut so that they share the pass over `P`. The last character of `P` goes
//! into the passes over `C`. Then the row that predicts each character of `C`
//! is in them, and the shapes do not depend on the cut. The prompts are
//! right-padded to the longest prompt that the configuration allows, so no
//! shape changes from step to step (burn#5751).
//!
//! The training runs the contrast after the backward of the LM loss, with a
//! backward of its own. The two gradients add up into one optimizer step. So
//! the peak memory is that of the larger pass, not of their sum.

use crate::AppArgs;
use crate::dataset::{Split, VOCAB, stories};
use burn::config::Config;
use burn::prelude::*;
use burn::tensor::activation::{log_sigmoid, log_softmax, sigmoid};
use burn_mamba::prelude::{MambaSsdPath, MambaVocabNet, MambaVocabNetConfig};
use burn_stack::utils::ClassCursors;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Where a cut can fall: the first character of the continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Cut {
    /// At any character.
    Char,
    /// At the start of a word: after a space or a newline.
    Word,
    /// At the start of a sentence: after a newline, or after a space that
    /// follows `.`, `!` or `?` (a closing `"` can come between).
    Sentence,
}

impl Cut {
    /// Whether the continuation can start at `tokens[at]` (`at ≥ 1`).
    pub fn allows(self, tokens: &[u8], at: usize) -> bool {
        let char_at = |i: usize| VOCAB.character(tokens[i]);
        let end = |c: char| matches!(c, '.' | '!' | '?');
        match self {
            Cut::Char => true,
            Cut::Word => matches!(char_at(at - 1), ' ' | '\n'),
            Cut::Sentence => match char_at(at - 1) {
                '\n' => true,
                ' ' => {
                    (at >= 2 && end(char_at(at - 2)))
                        || (at >= 3 && char_at(at - 2) == '"' && end(char_at(at - 3)))
                }
                _ => false,
            },
        }
    }

    /// `char`, `word` or `sentence`.
    pub fn parse(arg: &str) -> Result<Cut, String> {
        match arg {
            "char" => Ok(Cut::Char),
            "word" => Ok(Cut::Word),
            "sentence" => Ok(Cut::Sentence),
            other => Err(format!("unknown cut {other:?} (char | word | sentence)")),
        }
    }
}

/// The negative prompt of a row: the prompt that does not belong to its
/// continuation. All of them come from the data alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Negative {
    /// The prompt of another row: a random other story.
    Random,
    /// The own prompt with its middle sentences in a random new order. The
    /// first sentence (the opening, "once upon a time …") and the last one
    /// (it ends at the cut) stay in place. The words are the same, so only
    /// the order of the events tells the two apart. A moved opening would be
    /// a cheap sign of a negative.
    Shuffled,
    /// As [`Shuffled`](Self::Shuffled), but the first sentence moves too.
    ShuffledAll,
    /// The prompt of the story of a pool of `pool` random stories whose words
    /// overlap most with the own prompt ([`WordIndex`]). Only the prompts
    /// compare, each up to its own cut.
    Nearest {
        /// The stories drawn per row.
        pool: usize,
    },
}

impl Negative {
    /// `random`, `shuffled`, `shuffled-all`, or `nearest[:POOL]` (the pool
    /// defaults to 64).
    pub fn parse(arg: &str) -> Result<Negative, String> {
        match arg.split_once(':') {
            None if arg == "random" => Ok(Negative::Random),
            None if arg == "shuffled" => Ok(Negative::Shuffled),
            None if arg == "shuffled-all" => Ok(Negative::ShuffledAll),
            None if arg == "nearest" => Ok(Negative::Nearest { pool: 64 }),
            Some(("nearest", pool)) => match pool.parse::<usize>() {
                Ok(pool) if pool >= 1 => Ok(Negative::Nearest { pool }),
                _ => Err(format!("pool {pool:?} is not a positive integer")),
            },
            _ => Err(format!(
                "unknown negative {arg:?} (random | shuffled | shuffled-all | nearest[:POOL])"
            )),
        }
    }

    /// The name of the flag value.
    pub fn name(self) -> String {
        match self {
            Negative::Random => "random".into(),
            Negative::Shuffled => "shuffled".into(),
            Negative::ShuffledAll => "shuffled-all".into(),
            Negative::Nearest { pool } => format!("nearest:{pool}"),
        }
    }
}

/// `tokens` with its sentences in a random new order. The last sentence stays
/// in place, and the first one too with `keep_first`. A sentence ends where
/// [`Cut::Sentence`] allows a cut. `None` with fewer than two sentences to
/// move.
pub fn shuffle_sentences(tokens: &[u8], keep_first: bool, rng: &mut impl Rng) -> Option<Vec<u8>> {
    let starts: Vec<usize> = std::iter::once(0)
        .chain((1..tokens.len()).filter(|&at| Cut::Sentence.allows(tokens, at)))
        .collect();
    // The sentences that move: from the first (or the second) to the one
    // before the last.
    let (first, last) = (usize::from(keep_first), starts.len() - 1);
    if last < first + 2 {
        return None;
    }
    let mut order: Vec<usize> = (first..last).collect();
    while order.iter().zip(first..).all(|(&s, i)| i == s) {
        rand::seq::SliceRandom::shuffle(order.as_mut_slice(), rng);
    }
    let mut shuffled = Vec::with_capacity(tokens.len());
    shuffled.extend_from_slice(&tokens[..starts[first]]);
    for s in order {
        shuffled.extend_from_slice(&tokens[starts[s]..starts[s + 1]]);
    }
    shuffled.extend_from_slice(&tokens[starts[last]..]);
    Some(shuffled)
}

/// The word weights of [`Negative::Nearest`]. A word is a run of letters.
/// Two prompts compare by the cosine of their word sets, each word weighted
/// by its `idf = ln(N / df)` over the starts of the stories. So a word that
/// every story has ("once", "upon") counts for almost nothing. Only the
/// prompts compare, never their continuations: a negative that shares the
/// words of the continuation would predict it for the wrong reason.
pub struct WordIndex {
    /// The id of each word.
    ids: std::collections::HashMap<Vec<u8>, u32>,
    /// The `idf²` of each word id.
    weight: Vec<f32>,
}

impl WordIndex {
    /// The weights of the words of the first `region` characters of each
    /// story. A prompt is never longer than `region`, so the index knows
    /// each of its whole words.
    pub fn new(stories: &[Vec<u8>], region: usize) -> Self {
        let mut ids: std::collections::HashMap<Vec<u8>, u32> = Default::default();
        let mut df: Vec<u32> = Vec::new();
        for tokens in stories {
            let mut set: Vec<u32> = split_words(&tokens[..region.min(tokens.len())])
                .map(|word| {
                    let next = ids.len() as u32;
                    *ids.entry(word.to_vec()).or_insert(next)
                })
                .collect();
            set.sort_unstable();
            set.dedup();
            df.resize(ids.len(), 0);
            set.iter().for_each(|&w| df[w as usize] += 1);
        }
        let n = stories.len() as f32;
        let weight = df.iter().map(|&d| (n / d as f32).ln().powi(2)).collect();
        Self { ids, weight }
    }

    /// The sorted, unique ids of the known words of `tokens` (a prompt).
    pub fn words(&self, tokens: &[u8]) -> Vec<u32> {
        let mut set: Vec<u32> = split_words(tokens).filter_map(|word| self.ids.get(word).copied()).collect();
        set.sort_unstable();
        set.dedup();
        set
    }

    /// The weighted cosine of two word sets of [`words`](Self::words).
    pub fn similarity(&self, a: &[u32], b: &[u32]) -> f32 {
        let (mut i, mut j, mut dot) = (0, 0, 0.0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    dot += self.weight[a[i] as usize];
                    i += 1;
                    j += 1;
                }
            }
        }
        let norm = |set: &[u32]| set.iter().map(|&w| self.weight[w as usize]).sum::<f32>().sqrt();
        let norm = norm(a) * norm(b);
        if norm > 0.0 { dot / norm } else { 0.0 }
    }
}

/// The words of `tokens`: the runs of letters.
fn split_words(tokens: &[u8]) -> impl Iterator<Item = &[u8]> {
    tokens
        .split(|&t| !VOCAB.character(t).is_ascii_lowercase())
        .filter(|word| !word.is_empty())
}

/// The settings of the contrast. The training saves them as `contrast.json`
/// in the artifacts directory, so a resumed run keeps them.
#[derive(Config, Debug)]
pub struct ContrastConfig {
    /// Stories per step (`--contrast K`). 0 disables the contrast. 1 is not
    /// allowed: a story needs the prompt of another story.
    #[config(default = 0)]
    pub stories: usize,
    /// The shortest prompt, in characters (`--contrast-prompt MIN:MAX`).
    #[config(default = 192)]
    pub prompt_min: usize,
    /// The longest prompt, in characters.
    #[config(default = 320)]
    pub prompt_max: usize,
    /// Where a cut can fall (`--contrast-cut`).
    #[config(default = "Cut::Char")]
    pub cut: Cut,
    /// The negative prompt of each row (`--contrast-negative`).
    #[config(default = "Negative::Random")]
    pub negative: Negative,
    /// Continuation characters (`--contrast-continuation`).
    #[config(default = 256)]
    pub continuation: usize,
    /// The first characters of each continuation that the contrast does not
    /// score (`--contrast-skip`, `J`). They are still context.
    #[config(default = 0)]
    pub skip: usize,
    /// The weight `λ` of the contrast loss (`--contrast-weight`), after the
    /// ramp. At λ = 1, the contrast pushes as hard as the LM loss at chance
    /// (see the module header).
    #[config(default = 0.1)]
    pub weight: f64,
    /// The schedule step where the ramp of `λ` starts (`--contrast-start`).
    /// Before it, the weight is 0.
    #[config(default = 1000)]
    pub start: usize,
    /// The steps of the linear ramp from 0 to `λ` (`--contrast-ramp`). 0: the
    /// full weight from `start` on.
    #[config(default = 1000)]
    pub ramp: usize,
    /// Also train the next-character loss on the continuations of the own
    /// pass (`--contrast-lm`). Each of their characters weighs about as much
    /// as one character of the packed batch.
    #[config(default = false)]
    pub lm: bool,
    /// Detach the other pass (`--contrast-one-sided`). Then the contrast only
    /// raises the own log-probs. It cannot win by a worse prediction after
    /// the negative (the penalty).
    #[config(default = false)]
    pub one_sided: bool,
}

impl ContrastConfig {
    /// Whether the contrast runs.
    pub fn enabled(&self) -> bool {
        self.stories > 0
    }

    /// The weight of the contrast loss at the schedule step `step`: 0 before
    /// `start`, then a linear ramp to `weight` over `ramp` steps.
    pub fn weight_at(&self, step: usize) -> f64 {
        match (step.checked_sub(self.start), self.ramp) {
            (None, _) => 0.0,
            (Some(_), 0) => self.weight,
            (Some(done), ramp) => self.weight * (done as f64 / ramp as f64).min(1.0),
        }
    }

    /// Panic on settings that cannot run.
    pub fn validate(&self) {
        assert!(self.stories != 1, "--contrast needs 2 stories or more (or 0 for none)");
        assert!(self.prompt_min >= 2, "a contrast prompt has at least 2 characters");
        assert!(self.prompt_min <= self.prompt_max, "--contrast-prompt MIN:MAX needs MIN ≤ MAX");
        assert!(
            self.skip < self.continuation,
            "--contrast-skip must be shorter than the continuation"
        );
        assert!(self.weight >= 0.0, "--contrast-weight must not be negative");
    }

    /// The saved settings of the artifacts directory `dir` (the defaults when
    /// there are none), with the flags of `overrides`. Saves the result.
    pub fn resolve(dir: &Path, overrides: &ContrastOverrides) -> Self {
        let path = dir.join("contrast.json");
        let saved = match path.exists() {
            true => ContrastConfig::load(&path).expect("contrast.json does not parse"),
            false => ContrastConfig::new(),
        };
        let config = overrides.apply(saved);
        config.validate();
        config.save(&path).expect("failed to write contrast.json");
        config
    }
}

/// The `--contrast*` flags. Each one that is given replaces its saved value.
#[derive(Debug, Default)]
pub struct ContrastOverrides {
    /// `--contrast K`.
    pub stories: Option<usize>,
    /// `--contrast-prompt MIN:MAX`.
    pub prompt: Option<(usize, usize)>,
    /// `--contrast-cut char|word|sentence`.
    pub cut: Option<Cut>,
    /// `--contrast-negative random|shuffled|nearest[:POOL]`.
    pub negative: Option<Negative>,
    /// `--contrast-continuation N`.
    pub continuation: Option<usize>,
    /// `--contrast-skip J`.
    pub skip: Option<usize>,
    /// `--contrast-weight λ`.
    pub weight: Option<f64>,
    /// `--contrast-start S`.
    pub start: Option<usize>,
    /// `--contrast-ramp R`.
    pub ramp: Option<usize>,
    /// `--contrast-lm`.
    pub lm: bool,
    /// `--contrast-one-sided`.
    pub one_sided: bool,
}

impl ContrastOverrides {
    /// The lines of `-- --help`.
    pub const HELP: &str = concat!(
        "    --contrast <K>         Train the shared-suffix contrast on K stories per step (0: none; saved with the run)\n",
        "    --contrast-prompt <MIN:MAX>  The contrast prompt length, in characters (default 192:320)\n",
        "    --contrast-cut <CUT>   Where a contrast cut can fall: char (default), word or sentence\n",
        "    --contrast-negative <NEG>  The negative prompt: random (another story, default), shuffled (the own middle sentences in a new order), shuffled-all (the first sentence moves too) or nearest[:POOL] (the most similar of POOL stories, default 64)\n",
        "    --contrast-continuation <N>  The contrast continuation length, in characters (default 256)\n",
        "    --contrast-skip <J>    The first J characters of each continuation get no contrast loss (default 0)\n",
        "    --contrast-weight <λ>  The weight of the contrast loss after the ramp; 1 pushes as hard as the LM loss at chance (default 0.1)\n",
        "    --contrast-start <S>   The step where the ramp of the contrast weight starts; 0 before it (default 1000)\n",
        "    --contrast-ramp <R>    The steps of the linear ramp of the contrast weight (default 1000)\n",
        "    --contrast-lm          Also train the next-character loss on the own continuations of the contrast\n",
        "    --contrast-one-sided   Detach the pass after the negative: the contrast only raises the own log-probabilities\n",
    );

    /// Parse the flags.
    pub fn parse(pargs: &mut pico_args::Arguments) -> Self {
        let prompt = pargs
            .opt_value_from_fn("--contrast-prompt", |arg: &str| {
                let (min, max) = arg.split_once(':').ok_or("expected MIN:MAX")?;
                let parse = |s: &str| s.parse::<usize>().map_err(|e| e.to_string());
                Ok::<_, String>((parse(min)?, parse(max)?))
            })
            .unwrap();
        Self {
            stories: pargs.opt_value_from_str("--contrast").unwrap(),
            prompt,
            cut: pargs.opt_value_from_fn("--contrast-cut", Cut::parse).unwrap(),
            negative: pargs.opt_value_from_fn("--contrast-negative", Negative::parse).unwrap(),
            continuation: pargs.opt_value_from_str("--contrast-continuation").unwrap(),
            skip: pargs.opt_value_from_str("--contrast-skip").unwrap(),
            weight: pargs.opt_value_from_str("--contrast-weight").unwrap(),
            start: pargs.opt_value_from_str("--contrast-start").unwrap(),
            ramp: pargs.opt_value_from_str("--contrast-ramp").unwrap(),
            lm: pargs.contains("--contrast-lm"),
            one_sided: pargs.contains("--contrast-one-sided"),
        }
    }

    /// `config` with the given flags.
    pub fn apply(&self, mut config: ContrastConfig) -> ContrastConfig {
        if let Some(stories) = self.stories {
            config.stories = stories;
        }
        if let Some((min, max)) = self.prompt {
            config.prompt_min = min;
            config.prompt_max = max;
        }
        if let Some(cut) = self.cut {
            config.cut = cut;
        }
        if let Some(negative) = self.negative {
            config.negative = negative;
        }
        if let Some(continuation) = self.continuation {
            config.continuation = continuation;
        }
        if let Some(skip) = self.skip {
            config.skip = skip;
        }
        if let Some(weight) = self.weight {
            config.weight = weight;
        }
        if let Some(start) = self.start {
            config.start = start;
        }
        if let Some(ramp) = self.ramp {
            config.ramp = ramp;
        }
        if self.lm {
            config.lm = true;
        }
        if self.one_sided {
            config.one_sided = true;
        }
        config
    }
}

/// One step of the contrast: `K` rows, row `i` cut from story `i`.
#[derive(Clone, Debug)]
pub struct ContrastBatch {
    /// Each prompt but its last character, right-padded:
    /// `[K, prompt_max − 1]`, and `true` at the padding.
    pub prompt: (Tensor<2, Int>, Tensor<2, Bool>),
    /// The inputs of the own pass: the last character of `P_i`, then `C_i`
    /// but its last character: `[K, n]`.
    pub own: Tensor<2, Int>,
    /// The targets of the own pass: `C_i`, `[K, n]`.
    pub own_targets: Tensor<2, Int>,
    /// The negative prompts `N_i`, as `prompt`. `None`: the negative of row
    /// `i` is the prompt of row `i − 1` (a random other story).
    pub negative: Option<(Tensor<2, Int>, Tensor<2, Bool>)>,
    /// The inputs of the other pass, `[K, n]`. With no `negative`: the last
    /// character of `P_i`, then `C_{i+1}` (row `i − 1` of the roll reads
    /// `C_i` after `P_{i−1}`). With a `negative`: the last character of
    /// `N_i`, then `C_i`.
    pub other: Tensor<2, Int>,
    /// The targets of the other pass: `C_{i+1}` with no `negative`, else
    /// `C_i`.
    pub other_targets: Tensor<2, Int>,
}

impl ContrastBatch {
    /// The batch of the prompts and the continuations (at least 2 rows), with
    /// the prompt of the previous row as the negative. The prompts are 2 to
    /// `prompt_max` characters long, and the continuations all have one
    /// length.
    pub fn new(prompts: &[&[u8]], continuations: &[&[u8]], prompt_max: usize, device: &Device) -> Self {
        Self::build(prompts, None, continuations, prompt_max, device)
    }

    /// As [`new`](Self::new), with the negative prompt of each row given:
    /// row `i` reads `C_i` after `P_i` and after `negatives[i]`.
    pub fn with_negatives(
        prompts: &[&[u8]],
        negatives: &[&[u8]],
        continuations: &[&[u8]],
        prompt_max: usize,
        device: &Device,
    ) -> Self {
        Self::build(prompts, Some(negatives), continuations, prompt_max, device)
    }

    fn build(
        prompts: &[&[u8]],
        negatives: Option<&[&[u8]]>,
        continuations: &[&[u8]],
        prompt_max: usize,
        device: &Device,
    ) -> Self {
        let k = prompts.len();
        let n = continuations[0].len();
        assert!(k >= 2 && continuations.len() == k, "one continuation per prompt, 2 rows or more");
        assert!(continuations.iter().all(|c| c.len() == n), "the continuations have one length");
        let fits = |p: &&[u8]| (2..=prompt_max).contains(&p.len());
        assert!(prompts.iter().all(fits), "a prompt has 2 to {prompt_max} characters");
        assert!(
            negatives.is_none_or(|neg| neg.len() == k && neg.iter().all(fits)),
            "one negative of 2 to {prompt_max} characters per prompt"
        );
        let ints = |v: &[i32], cols: usize| Tensor::<1, Int>::from_ints(v, device).reshape([k, cols]);
        // Each prompt but its last character, right-padded.
        let padded = |prompts: &[&[u8]]| {
            let width = prompt_max - 1;
            let mut ids = vec![0i32; k * width];
            let mut pad = vec![true; k * width];
            for (i, p) in prompts.iter().enumerate() {
                for (j, &t) in p[..p.len() - 1].iter().enumerate() {
                    ids[i * width + j] = t as i32;
                    pad[i * width + j] = false;
                }
            }
            let pad = Tensor::<1, Bool>::from_bool(burn::tensor::TensorData::new(pad, [k * width]), device);
            (ints(&ids, width), pad.reshape([k, width]))
        };
        // The inputs: the last character of `prefix`, then `continuation` but
        // its last character. The targets: `continuation`.
        let pass = |rows: &[(&[u8], &[u8])]| {
            let (mut inputs, mut targets) = (Vec::with_capacity(k * n), Vec::with_capacity(k * n));
            for (prefix, c) in rows {
                inputs.push(prefix[prefix.len() - 1] as i32);
                inputs.extend(c[..n - 1].iter().map(|&t| t as i32));
                targets.extend(c.iter().map(|&t| t as i32));
            }
            (ints(&inputs, n), ints(&targets, n))
        };
        let own_rows: Vec<(&[u8], &[u8])> = prompts.iter().copied().zip(continuations.iter().copied()).collect();
        let other_rows: Vec<(&[u8], &[u8])> = match negatives {
            None => (0..k).map(|i| (prompts[i], continuations[(i + 1) % k])).collect(),
            Some(neg) => neg.iter().copied().zip(continuations.iter().copied()).collect(),
        };
        let (own, own_targets) = pass(&own_rows);
        let (other, other_targets) = pass(&other_rows);
        Self {
            prompt: padded(prompts),
            own,
            own_targets,
            negative: negatives.map(padded),
            other,
            other_targets,
        }
    }
}

/// The contrast loss of `batch` (see the module header) at the weight
/// `weight` (`λ` at this step). Under `--contrast-lm`, plus the LM loss of
/// the own continuations. The LM term divides by `lm_positions`, the
/// positions of the packed batch. Also returns the readouts of [`scored`].
pub fn loss(
    model: &MambaVocabNet,
    ssd_path: &MambaSsdPath,
    batch: ContrastBatch,
    config: &ContrastConfig,
    weight: f64,
    lm_positions: usize,
) -> (Tensor<1>, Tensor<1>) {
    let device = batch.own.device();
    let (own_kn, other_kn) = pair_log_probs(model, ssd_path, batch);
    let other_kn = if config.one_sided { other_kn.detach() } else { other_kn };
    let (contrast, readouts) = scored(own_kn.clone(), other_kn, config.skip);
    let m = own_kn.dims()[1] - config.skip;
    // At chance, the mean pushes the own log-probs by `(m + 1)/4` together.
    // The weight changes during the ramp, so it is data, not a kernel scalar.
    // A new scalar is a new key of the cubecl cache (burn#5751).
    let weight_1 = Tensor::<1>::from_floats([(weight * 4.0 / (m + 1) as f64) as f32], &device);
    let mut total = contrast * weight_1;
    if config.lm {
        total = total + own_kn.neg().sum().div_scalar(lm_positions as f64);
    }
    (total, readouts)
}

/// The log-probabilities `[K, n]` of the continuations of `batch`: row `i`
/// of both scores `C_i`, after `P_i` (own) and after its negative (other:
/// `P_{i−1}`, or `N_i`).
pub fn pair_log_probs(model: &MambaVocabNet, ssd_path: &MambaSsdPath, batch: ContrastBatch) -> (Tensor<2>, Tensor<2>) {
    let rolled = batch.negative.is_none();
    let [own, other] =
        pair_passes(model, ssd_path, batch).map(|(log_p_knv, targets_kn)| target_log_p(log_p_knv, targets_kn));
    (own, if rolled { roll_back(other) } else { other })
}

/// The log-softmax `[K, n, vocab]` of the two passes over the continuations
/// of `batch` (own, then other), each with its targets `[K, n]`. With no
/// `negative`, row `i` of the other pass scores `C_{i+1}` (see
/// [`roll_back`]).
fn pair_passes(
    model: &MambaVocabNet,
    ssd_path: &MambaSsdPath,
    batch: ContrastBatch,
) -> [(Tensor<3>, Tensor<2, Int>); 2] {
    let n = batch.own.dims()[1];
    // A pass over prompts opens each row with the class latents, and leaves
    // the cursor after them. So the passes over `C` emit none.
    let prompt_pass = |(ids, pad): (Tensor<2, Int>, Tensor<2, Bool>)| {
        let mut class = ClassCursors::stream();
        let (_, caches) = model.forward(ids, None, ssd_path.clone(), Some(&mut class), Some(pad));
        (caches, class)
    };
    let (caches, mut class) = prompt_pass(batch.prompt);
    let (other_caches, mut class_other) = match batch.negative.clone() {
        None => (caches.clone(), class.clone()),
        Some(negative) => prompt_pass(negative),
    };
    let (own_logits, _) = model.forward(batch.own, Some(caches), ssd_path.clone(), Some(&mut class), None);
    let (other_logits, _) =
        model.forward(batch.other, Some(other_caches), ssd_path.clone(), Some(&mut class_other), None);
    assert_eq!(own_logits.dims()[1], n, "the passes over the continuations emit no latents");
    [
        (log_softmax(own_logits, 2), batch.own_targets),
        (log_softmax(other_logits, 2), batch.other_targets),
    ]
}

/// The log-probability `[K, n]` of each target under a log-softmax
/// `[K, n, vocab]`.
fn target_log_p(log_p_knv: Tensor<3>, targets_kn: Tensor<2, Int>) -> Tensor<2> {
    let [k, n, _] = log_p_knv.dims();
    log_p_knv.gather(2, targets_kn.reshape([k, n, 1])).reshape([k, n])
}

/// Row `i` of a score of the other pass moved to row `i + 1` (cyclic). With
/// no `negative`, row `i` of the other pass scores `C_{i+1}`. After the roll,
/// row `i` of both terms scores `C_i`.
fn roll_back(other_kn: Tensor<2>) -> Tensor<2> {
    let k = other_kn.dims()[0];
    Tensor::cat(vec![other_kn.clone().narrow(0, k - 1, 1), other_kn.narrow(0, 0, k - 1)], 0)
}

/// The entropy `[K, n]` (nats) of each prediction of a log-softmax
/// `[K, n, vocab]`.
fn entropy(log_p_knv: Tensor<3>) -> Tensor<2> {
    let [k, n, _] = log_p_knv.dims();
    (log_p_knv.clone().exp() * log_p_knv).sum_dim(2).neg().reshape([k, n])
}

/// The contrast loss of the log-probabilities of [`pair_log_probs`], scored
/// from the character `skip` (`J`) on: `mean −log σ(Δ_≤t)` (`[1]`, before
/// the weight and the scale). Also returns the readouts `[3]`, detached:
///
/// - that loss,
/// - the mean of `Δ` per scored character (nats),
/// - the **push**: the gradient mass of that loss on the own log-probs. It
///   is relative to its value at chance: 1 at chance, 0 when every pair is
///   separated. The push of the training is the weight times this.
pub fn scored(own_kn: Tensor<2>, other_kn: Tensor<2>, skip: usize) -> (Tensor<1>, Tensor<1>) {
    let [_, n] = own_kn.dims();
    let device = own_kn.device();
    let m = n - skip;
    let delta_km = (own_kn - other_kn).narrow(1, skip, m);
    // `Δ_≤t = Σ_{s ≤ t} Δ_s`: a product with the upper triangle of ones.
    let upper_mm = Tensor::<2>::ones([m, m], &device).triu(0);
    let prefix_km = delta_km.clone().matmul(upper_mm);
    let contrast = log_sigmoid(prefix_km.clone()).neg().mean();
    // The push on `Δ_s` sums `σ(−Δ_≤t)` over the `t ≥ s`. So the prefix of
    // index `t` (from 0) counts `t + 1` times. At chance, this mean is
    // `(m + 1)/4`.
    let counts_1m = Tensor::<1, Int>::arange(1..m as i64 + 1, &device).float().reshape([1, m]);
    let push = (sigmoid(prefix_km.neg()) * counts_1m).mean().mul_scalar(4.0 / (m + 1) as f64);
    let readouts = Tensor::cat(vec![contrast.clone(), delta_km.mean(), push], 0).detach();
    (contrast, readouts)
}

/// The cut of `tokens`, which hold at least `prompt_min + continuation`
/// characters. It is a random length in `prompt_min..=prompt_max` that leaves
/// a continuation, moved back to where the cut kind of `config` allows it.
/// `None` when no position from `prompt_min` up to that length allows it.
pub fn cut(config: &ContrastConfig, rng: &mut impl Rng, tokens: &[u8]) -> Option<usize> {
    let (min, max) = (config.prompt_min, config.prompt_max);
    let hi = max.min(tokens.len() - config.continuation);
    let drawn = rng.random_range(min..=hi);
    (min..=drawn).rev().find(|&at| config.cut.allows(tokens, at))
}

/// The train stories of the contrast and the order in which it draws them.
pub struct Feed {
    config: ContrastConfig,
    /// The token ids of each story with `prompt_min + continuation`
    /// characters or more, truncated to `prompt_max + continuation`.
    stories: Vec<Vec<u8>>,
    /// A shuffle of `stories`. A new one starts when it runs out.
    order: Vec<usize>,
    /// The next position in `order`.
    next: usize,
    rng: ChaCha8Rng,
    /// The word index of [`Negative::Nearest`], built at its first use.
    index: Option<WordIndex>,
    /// The schedule step of the next batch.
    step: usize,
    /// `contrast.jsonl` in the artifacts directory: one line per step.
    log: Option<std::fs::File>,
    /// The sums of the readouts since the last printed line, and their count.
    sums: [f64; 3],
    count: usize,
}

/// Steps per printed line of the contrast readouts.
const PRINT_EVERY: usize = 100;

impl Feed {
    /// The stories of the first `train_stories` of the train split (those of
    /// the LM batches), in the order of a shuffle with `seed`. `step` is the
    /// schedule step of the first batch (that of a resumed run). With a
    /// `dir`, appends the readouts to `contrast.jsonl` in it.
    pub fn new(config: ContrastConfig, train_stories: usize, seed: u64, step: usize, dir: Option<&Path>) -> Self {
        assert!(config.enabled(), "the contrast is disabled");
        let keep = config.prompt_max + config.continuation;
        let need = config.prompt_min + config.continuation;
        let stories: Vec<Vec<u8>> = stories(Split::Train, train_stories)
            .iter()
            .map(|story| {
                let mut tokens = VOCAB.encode(story);
                tokens.truncate(keep);
                tokens
            })
            .filter(|tokens| tokens.len() >= need)
            .collect();
        assert!(
            stories.len() >= config.stories,
            "only {} train stories have the {need} characters of a contrast row",
            stories.len()
        );
        println!(
            "contrast: {} of {train_stories} train stories, {} per step, prompt {}..={}, {:?} cut, \
             {:?} negative, continuation {}, skip {}, weight {} (0 before step {}, then a ramp of {} \
             steps), lm {}, one-sided {}",
            stories.len(),
            config.stories,
            config.prompt_min,
            config.prompt_max,
            config.cut,
            config.negative,
            config.continuation,
            config.skip,
            config.weight,
            config.start,
            config.ramp,
            config.lm,
            config.one_sided
        );
        let log = dir.map(|dir| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("contrast.jsonl"))
                .expect("failed to open contrast.jsonl")
        });
        Self {
            config,
            stories,
            order: Vec::new(),
            next: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            index: None,
            step,
            log,
            sums: [0.0; 3],
            count: 0,
        }
    }

    /// Draw again from the start of a shuffle with `seed`, with these cut and
    /// negative kinds.
    pub fn restart(&mut self, cut: Cut, negative: Negative, seed: u64) {
        self.config.cut = cut;
        self.config.negative = negative;
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.order.clear();
        self.next = 0;
    }

    /// The settings.
    pub fn config(&self) -> &ContrastConfig {
        &self.config
    }

    /// The weight of the contrast loss at the current step.
    pub fn weight(&self) -> f64 {
        self.config.weight_at(self.step)
    }

    /// Whether the pass of the contrast runs at the current step: a weight
    /// above 0, or `--contrast-lm`.
    pub fn runs(&self) -> bool {
        self.weight() > 0.0 || self.config.lm
    }

    /// The next story of the shuffle that has a cut, and its cut.
    fn draw(&mut self) -> (usize, usize) {
        let mut misses = 0;
        loop {
            if self.next == self.order.len() {
                self.order = (0..self.stories.len()).collect();
                rand::seq::SliceRandom::shuffle(self.order.as_mut_slice(), &mut self.rng);
                self.next = 0;
            }
            let story = self.order[self.next];
            self.next += 1;
            match cut(&self.config, &mut self.rng, &self.stories[story]) {
                Some(at) => return (story, at),
                None => {
                    misses += 1;
                    assert!(misses < 1000, "the contrast finds no {:?} cut in the stories", self.config.cut);
                }
            }
        }
    }

    /// The next `K` stories, cut, with their negatives.
    pub fn batch(&mut self, device: &Device) -> ContrastBatch {
        let (k, n, max) = (self.config.stories, self.config.continuation, self.config.prompt_max);
        let mut rows: Vec<(usize, usize)> = Vec::with_capacity(k);
        // The negative prompts, when they are not those of the other rows.
        let mut negatives: Vec<Vec<u8>> = Vec::new();
        match self.config.negative {
            Negative::Random => rows.extend((0..k).map(|_| self.draw())),
            Negative::Shuffled | Negative::ShuffledAll => {
                let keep_first = self.config.negative == Negative::Shuffled;
                let mut misses = 0;
                while rows.len() < k {
                    let (story, at) = self.draw();
                    match shuffle_sentences(&self.stories[story][..at], keep_first, &mut self.rng) {
                        Some(negative) => {
                            rows.push((story, at));
                            negatives.push(negative);
                        }
                        None => {
                            misses += 1;
                            assert!(misses < 1000, "no contrast prompt has two sentences to move");
                        }
                    }
                }
            }
            Negative::Nearest { pool } => {
                rows.extend((0..k).map(|_| self.draw()));
                let candidates: Vec<(usize, usize)> = (0..pool).map(|_| self.draw()).collect();
                let index = self
                    .index
                    .get_or_insert_with(|| WordIndex::new(&self.stories, self.config.prompt_max));
                // The prompts compare, each up to its own cut.
                let candidate_words: Vec<Vec<u32>> =
                    candidates.iter().map(|&(c, at)| index.words(&self.stories[c][..at])).collect();
                for (i, &(story, at)) in rows.iter().enumerate() {
                    let own = index.words(&self.stories[story][..at]);
                    let best = candidates
                        .iter()
                        .zip(&candidate_words)
                        .filter(|((c, _), _)| *c != story)
                        .max_by(|(_, a), (_, b)| index.similarity(&own, a).total_cmp(&index.similarity(&own, b)))
                        .map(|(candidate, _)| candidate);
                    // Only when the pool holds no other story: the next row.
                    let (c, at) = *best.unwrap_or(&rows[(i + 1) % k]);
                    negatives.push(self.stories[c][..at].to_vec());
                }
            }
        }
        let prompts: Vec<&[u8]> = rows.iter().map(|&(s, at)| &self.stories[s][..at]).collect();
        let continuations: Vec<&[u8]> = rows.iter().map(|&(s, at)| &self.stories[s][at..at + n]).collect();
        match self.config.negative {
            Negative::Random => ContrastBatch::new(&prompts, &continuations, max, device),
            _ => {
                let negatives: Vec<&[u8]> = negatives.iter().map(Vec::as_slice).collect();
                ContrastBatch::with_negatives(&prompts, &negatives, &continuations, max, device)
            }
        }
    }

    /// End the current step: log the readouts of [`loss`] if the pass ran
    /// (this reads them from the device), and move to the next step.
    pub fn end_step(&mut self, readouts: Option<Tensor<1>>) {
        let weight = self.weight();
        let step = self.step;
        self.step += 1;
        let Some(readouts) = readouts else {
            return;
        };
        let values: Vec<f64> = readouts.into_data().try_into_vec_as().unwrap();
        let [contrast, delta, push] = [values[0], values[1], values[2]];
        let bits = delta / std::f64::consts::LN_2;
        if let Some(log) = &mut self.log {
            writeln!(
                log,
                "{{\"step\":{step},\"weight\":{weight},\"contrast\":{contrast},\"delta_bits\":{bits},\
                 \"push\":{push}}}"
            )
            .expect("failed to write contrast.jsonl");
        }
        self.sums[0] += contrast;
        self.sums[1] += bits;
        self.sums[2] += push;
        self.count += 1;
        if self.count == PRINT_EVERY {
            let n = self.count as f64;
            println!(
                "contrast step={step} weight {weight:.4} loss {:.4} delta {:+.4} bits/char push {:.2e} \
                 (mean of {} steps)",
                self.sums[0] / n,
                self.sums[1] / n,
                self.sums[2] / n,
                self.count
            );
            self.sums = [0.0; 3];
            self.count = 0;
        }
    }
}

/// The contrast of the training process: set once by [`init`].
pub static FEED: OnceLock<Mutex<Feed>> = OnceLock::new();

/// Start the contrast of this process.
pub fn init(feed: Feed) {
    assert!(FEED.set(Mutex::new(feed)).is_ok(), "the contrast starts once");
}

/// The skips `J` of `--contrast-eval` (those below the continuation).
pub const EVAL_SKIPS: [usize; 5] = [0, 16, 32, 64, 128];

/// The seed of the stories of `--contrast-eval`. It is fixed, so every
/// checkpoint gets the same stories.
const EVAL_SEED: u64 = 0;

/// The pool of [`Negative::Nearest`] in `--contrast-eval`.
const EVAL_POOL: usize = 64;

/// `--contrast-eval N`: how strongly the contrast pushes the trained model,
/// with no backward and no change to the checkpoint. For each cut kind and
/// each kind of negative, it scores `batches` batches of train stories (the
/// same ones for every checkpoint). From one pair of forwards per batch, it gives the readouts
/// of [`scored`] at each `J` of [`EVAL_SKIPS`]. The other settings come from
/// the `--contrast*` flags (`K` defaults to 4). Prints a table, and saves it
/// as `contrast-eval.json` in the artifacts directory.
///
/// With `profile` (`--contrast-profile`), only the cut and the negative of
/// the flags run, and [`profiled_pair_log_probs`] also writes the scores of
/// each character into `contrast-profile.jsonl`.
#[allow(clippy::too_many_arguments)]
pub fn contrast_eval(
    model_config: MambaVocabNetConfig,
    device: Device,
    app_args: &AppArgs,
    ssd_path: &MambaSsdPath,
    overrides: &ContrastOverrides,
    train_stories: usize,
    batches: usize,
    profile: bool,
) {
    let model: MambaVocabNet = app_args
        .load_model(&model_config, &device)
        .expect("no trained model in the artifacts directory");
    let config = overrides.apply(ContrastConfig::new().with_stories(4));
    config.validate();
    assert!(config.enabled(), "--contrast-eval needs 2 stories or more per batch");
    let n = config.continuation;
    let skips: Vec<usize> = EVAL_SKIPS.into_iter().filter(|&j| j < n).collect();
    let negatives = [
        Negative::Random,
        Negative::Shuffled,
        Negative::ShuffledAll,
        Negative::Nearest { pool: EVAL_POOL },
    ];
    let kinds: Vec<(Cut, Negative)> = match profile {
        true => vec![(config.cut, config.negative)],
        false => [Cut::Char, Cut::Word, Cut::Sentence]
            .into_iter()
            .flat_map(|cut| negatives.map(|negative| (cut, negative)))
            .collect(),
    };
    let profile_path = app_args.artifacts_path.join("contrast-profile.jsonl");
    let mut profile_log =
        profile.then(|| std::fs::File::create(&profile_path).expect("failed to create contrast-profile.jsonl"));
    let mut feed = Feed::new(config, train_stories, EVAL_SEED, 0, None);
    println!(
        "contrast eval: {batches} batches of {} stories per cut. push = the gradient mass relative to chance",
        feed.config().stories
    );
    println!("  cut        negative        J   loss       delta bits/char   push");
    let mut json = Vec::new();
    for (cut, negative) in kinds {
        feed.restart(cut, negative, EVAL_SEED);
        let mut sums = vec![[0.0f64; 3]; skips.len()];
        for _ in 0..batches {
            let batch = feed.batch(&device);
            let (own_kn, other_kn) = match profile_log.as_mut() {
                Some(log) => profiled_pair_log_probs(&model, ssd_path, batch, log),
                None => pair_log_probs(&model, ssd_path, batch),
            };
            let readouts: Vec<Tensor<1>> = skips
                .iter()
                .map(|&j| scored(own_kn.clone(), other_kn.clone(), j).1)
                .collect();
            // One read-back per batch.
            let values: Vec<f64> = Tensor::cat(readouts, 0).into_data().try_into_vec_as().unwrap();
            for (sum, v) in sums.iter_mut().zip(values.chunks(3)) {
                sum.iter_mut().zip(v).for_each(|(s, x)| *s += x);
            }
        }
        let name = negative.name();
        for (&j, sum) in skips.iter().zip(&sums) {
            let [contrast, delta, push] = sum.map(|s| s / batches as f64);
            let bits = delta / std::f64::consts::LN_2;
            println!(
                "  {:<10} {name:<12} {j:>4}   {contrast:.4}     {bits:+.4}           {push:.3e}",
                format!("{cut:?}")
            );
            json.push(format!(
                "{{\"cut\":\"{cut:?}\",\"negative\":\"{name}\",\"skip\":{j},\"contrast\":{contrast},\
                 \"delta_bits\":{bits},\"push\":{push}}}"
            ));
        }
    }
    let path = app_args.artifacts_path.join("contrast-eval.json");
    std::fs::write(&path, format!("[{}]\n", json.join(","))).expect("failed to write contrast-eval.json");
    println!("saved to {path:?}");
    if profile {
        println!("saved to {profile_path:?}");
    }
}

/// [`pair_log_probs`] of `batch`. Also writes one line per row of `batch`
/// into `log` (`contrast-profile.jsonl`):
///
/// - `text`: the last character of the prompt, then the `n` characters of
///   the continuation,
/// - `own`, `other`: the log-probability (nats) of each character of the
///   continuation, after the own prompt and after the negative,
/// - `h_own`, `h_other`: the entropy (nats) of each of these predictions.
pub fn profiled_pair_log_probs(
    model: &MambaVocabNet,
    ssd_path: &MambaSsdPath,
    batch: ContrastBatch,
    log: &mut impl Write,
) -> (Tensor<2>, Tensor<2>) {
    let [k, n] = batch.own.dims();
    let rolled = batch.negative.is_none();
    // The first input of the own pass is the last character of the prompt.
    let text_ids = Tensor::cat(vec![batch.own.clone().narrow(1, 0, 1), batch.own_targets.clone()], 1);
    let [(own_knv, own_targets), (other_knv, other_targets)] = pair_passes(model, ssd_path, batch);
    let align = |other_kn: Tensor<2>| if rolled { roll_back(other_kn) } else { other_kn };
    let own_kn = target_log_p(own_knv.clone(), own_targets);
    let other_kn = align(target_log_p(other_knv.clone(), other_targets));
    let columns = vec![own_kn.clone(), other_kn.clone(), entropy(own_knv), align(entropy(other_knv))];
    let ids: Vec<i64> = text_ids.into_data().try_into_vec_as().unwrap();
    let values: Vec<f32> = Tensor::cat(columns, 0).into_data().try_into_vec_as().unwrap();
    for (i, row_ids) in ids.chunks(n + 1).enumerate() {
        let text: String = row_ids.iter().map(|&t| VOCAB.character(t as u8)).collect();
        // Column `c` of row `i` is row `c·K + i` of the concatenation.
        let column = |c: usize| {
            let row = &values[(c * k + i) * n..(c * k + i + 1) * n];
            row.iter().map(|v| format!("{v:.5}")).collect::<Vec<_>>().join(",")
        };
        // The alphabet is printable ASCII and `\n`, so the debug form of
        // `text` is a JSON string.
        writeln!(
            log,
            "{{\"text\":{text:?},\"own\":[{}],\"other\":[{}],\"h_own\":[{}],\"h_other\":[{}]}}",
            column(0),
            column(1),
            column(2),
            column(3)
        )
        .expect("failed to write contrast-profile.jsonl");
    }
    (own_kn, other_kn)
}

#[cfg(test)]
mod tests;
