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

use crate::dataset::{Split, VOCAB, stories};
use burn::config::Config;
use burn::prelude::*;
use burn::tensor::activation::{log_sigmoid, log_softmax};
use burn_mamba::prelude::{MambaSsdPath, MambaVocabNet};
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
}

impl ContrastOverrides {
    /// The lines of `-- --help`.
    pub const HELP: &str = concat!(
        "    --contrast <K>         Train the shared-suffix contrast on K stories per step (0: none; saved with the run)\n",
        "    --contrast-prompt <MIN:MAX>  The contrast prompt length, in characters (default 192:320)\n",
        "    --contrast-cut <CUT>   Where a contrast cut can fall: char (default), word or sentence\n",
        "    --contrast-continuation <N>  The contrast continuation length, in characters (default 256)\n",
        "    --contrast-skip <J>    The first J characters of each continuation get no contrast loss (default 0)\n",
        "    --contrast-weight <λ>  The weight of the contrast loss after the ramp; 1 pushes as hard as the LM loss at chance (default 0.1)\n",
        "    --contrast-start <S>   The step where the ramp of the contrast weight starts; 0 before it (default 1000)\n",
        "    --contrast-ramp <R>    The steps of the linear ramp of the contrast weight (default 1000)\n",
        "    --contrast-lm          Also train the next-character loss on the own continuations of the contrast\n",
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
            continuation: pargs.opt_value_from_str("--contrast-continuation").unwrap(),
            skip: pargs.opt_value_from_str("--contrast-skip").unwrap(),
            weight: pargs.opt_value_from_str("--contrast-weight").unwrap(),
            start: pargs.opt_value_from_str("--contrast-start").unwrap(),
            ramp: pargs.opt_value_from_str("--contrast-ramp").unwrap(),
            lm: pargs.contains("--contrast-lm"),
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
        config
    }
}

/// One step of the contrast: `K` rows, row `i` cut from story `i`.
#[derive(Clone, Debug)]
pub struct ContrastBatch {
    /// Each prompt but its last character, right-padded:
    /// `[K, prompt_max − 1]`.
    pub prompt: Tensor<2, Int>,
    /// `true` at the padding of `prompt`.
    pub pad: Tensor<2, Bool>,
    /// The inputs of the own pass: the last character of `P_i`, then `C_i`
    /// but its last character: `[K, n]`.
    pub own: Tensor<2, Int>,
    /// The inputs of the other pass: the last character of `P_i`, then
    /// `C_{i+1}` but its last character: `[K, n]`.
    pub other: Tensor<2, Int>,
    /// The targets of the own pass: `C_i`, `[K, n]`.
    pub own_targets: Tensor<2, Int>,
    /// The targets of the other pass: `C_{i+1}`, `[K, n]`.
    pub other_targets: Tensor<2, Int>,
}

impl ContrastBatch {
    /// The batch of the prompts and the continuations (at least 2 rows). The
    /// prompts are 2 to `prompt_max` characters long, and the continuations
    /// all have one length.
    pub fn new(prompts: &[&[u8]], continuations: &[&[u8]], prompt_max: usize, device: &Device) -> Self {
        let k = prompts.len();
        let n = continuations[0].len();
        assert!(k >= 2 && continuations.len() == k, "one continuation per prompt, 2 rows or more");
        assert!(continuations.iter().all(|c| c.len() == n), "the continuations have one length");
        assert!(
            prompts.iter().all(|p| (2..=prompt_max).contains(&p.len())),
            "a prompt has 2 to {prompt_max} characters"
        );
        let width = prompt_max - 1;
        let mut prompt = vec![0i32; k * width];
        let mut pad = vec![true; k * width];
        let (mut own, mut other) = (Vec::with_capacity(k * n), Vec::with_capacity(k * n));
        let (mut own_targets, mut other_targets) = (Vec::with_capacity(k * n), Vec::with_capacity(k * n));
        for (i, p) in prompts.iter().enumerate() {
            let (head, last) = (&p[..p.len() - 1], p[p.len() - 1]);
            for (j, &t) in head.iter().enumerate() {
                prompt[i * width + j] = t as i32;
                pad[i * width + j] = false;
            }
            let next = continuations[(i + 1) % k];
            for (inputs, targets, c) in [(&mut own, &mut own_targets, continuations[i]), (&mut other, &mut other_targets, next)] {
                inputs.push(last as i32);
                inputs.extend(c[..n - 1].iter().map(|&t| t as i32));
                targets.extend(c.iter().map(|&t| t as i32));
            }
        }
        let ints = |v: &[i32], cols: usize| Tensor::<1, Int>::from_ints(v, device).reshape([k, cols]);
        Self {
            prompt: ints(&prompt, width),
            pad: Tensor::<1, Bool>::from_bool(burn::tensor::TensorData::new(pad, [k * width]), device)
                .reshape([k, width]),
            own: ints(&own, n),
            other: ints(&other, n),
            own_targets: ints(&own_targets, n),
            other_targets: ints(&other_targets, n),
        }
    }
}

/// The contrast loss of `batch` (see the module header) at the weight
/// `weight` (`λ` at this step). Under `--contrast-lm`, plus the LM loss of
/// the own continuations. The LM term divides by `lm_positions`, the
/// positions of the packed batch. Also returns the readouts `[2]`: the mean
/// of `−log σ(Δ_≤t)` (before the weight and the scale), and the mean of `Δ`
/// per scored character (nats).
pub fn loss(
    model: &MambaVocabNet,
    ssd_path: &MambaSsdPath,
    batch: ContrastBatch,
    config: &ContrastConfig,
    weight: f64,
    lm_positions: usize,
) -> (Tensor<1>, Tensor<1>) {
    let [k, n] = batch.own.dims();
    let device = batch.own.device();
    // The pass over the prompts opens each row with the class latents, and
    // leaves the cursor after them. So the passes over `C` emit none.
    let mut class = ClassCursors::stream();
    let (_, caches) = model.forward(batch.prompt, None, ssd_path.clone(), Some(&mut class), Some(batch.pad));
    let mut class_other = class.clone();
    let (own_logits, _) = model.forward(batch.own, Some(caches.clone()), ssd_path.clone(), Some(&mut class), None);
    let (other_logits, _) = model.forward(batch.other, Some(caches), ssd_path.clone(), Some(&mut class_other), None);
    assert_eq!(own_logits.dims()[1], n, "the passes over the continuations emit no latents");
    let log_p = |logits: Tensor<3>, targets: Tensor<2, Int>| {
        log_softmax(logits, 2).gather(2, targets.reshape([k, n, 1])).reshape([k, n])
    };
    let own_kn = log_p(own_logits, batch.own_targets);
    // Row `i` of the other pass scores `C_{i+1}`. Roll it back by one row, so
    // that row `i` of both terms scores `C_i`.
    let other_kn = log_p(other_logits, batch.other_targets);
    let other_kn = Tensor::cat(vec![other_kn.clone().narrow(0, k - 1, 1), other_kn.narrow(0, 0, k - 1)], 0);
    let m = n - config.skip;
    let delta_km = (own_kn.clone() - other_kn).narrow(1, config.skip, m);
    // `Δ_≤t = Σ_{s ≤ t} Δ_s`: a product with the upper triangle of ones.
    let upper_mm = Tensor::<2>::ones([m, m], &device).triu(0);
    let prefix_km = delta_km.clone().matmul(upper_mm);
    let contrast = log_sigmoid(prefix_km).neg().mean();
    // At chance, the mean pushes the own log-probs by `(m + 1)/4` together.
    // The weight changes during the ramp, so it is data, not a kernel scalar.
    // A new scalar is a new key of the cubecl cache (burn#5751).
    let weight_1 = Tensor::<1>::from_floats([(weight * 4.0 / (m + 1) as f64) as f32], &device);
    let mut total = contrast.clone() * weight_1;
    if config.lm {
        total = total + own_kn.neg().sum().div_scalar(lm_positions as f64);
    }
    let readouts = Tensor::cat(vec![contrast, delta_km.mean()], 0).detach();
    (total, readouts)
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
    /// The schedule step of the next batch.
    step: usize,
    /// `contrast.jsonl` in the artifacts directory: one line per step.
    log: std::fs::File,
    /// The sums of the readouts since the last printed line, and their count.
    sums: [f64; 2],
    count: usize,
}

/// Steps per printed line of the contrast readouts.
const PRINT_EVERY: usize = 100;

impl Feed {
    /// The stories of the first `train_stories` of the train split (those of
    /// the LM batches), in the order of a shuffle with `seed`. `step` is the
    /// schedule step of the first batch (that of a resumed run). Appends the
    /// readouts to `contrast.jsonl` in `dir`.
    pub fn new(config: ContrastConfig, train_stories: usize, seed: u64, step: usize, dir: &Path) -> Self {
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
             continuation {}, skip {}, weight {} (0 before step {}, then a ramp of {} steps), lm {}",
            stories.len(),
            config.stories,
            config.prompt_min,
            config.prompt_max,
            config.cut,
            config.continuation,
            config.skip,
            config.weight,
            config.start,
            config.ramp,
            config.lm
        );
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("contrast.jsonl"))
            .expect("failed to open contrast.jsonl");
        Self {
            config,
            stories,
            order: Vec::new(),
            next: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            step,
            log,
            sums: [0.0; 2],
            count: 0,
        }
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

    /// The next `K` stories, cut.
    pub fn batch(&mut self, device: &Device) -> ContrastBatch {
        let (k, n) = (self.config.stories, self.config.continuation);
        let mut rows: Vec<(usize, usize)> = Vec::with_capacity(k);
        let mut misses = 0;
        while rows.len() < k {
            if self.next == self.order.len() {
                self.order = (0..self.stories.len()).collect();
                rand::seq::SliceRandom::shuffle(self.order.as_mut_slice(), &mut self.rng);
                self.next = 0;
            }
            let story = self.order[self.next];
            self.next += 1;
            match cut(&self.config, &mut self.rng, &self.stories[story]) {
                Some(at) => rows.push((story, at)),
                None => {
                    misses += 1;
                    assert!(misses < 100 * k, "the contrast finds no {:?} cut in the stories", self.config.cut);
                }
            }
        }
        let prompts: Vec<&[u8]> = rows.iter().map(|&(s, at)| &self.stories[s][..at]).collect();
        let continuations: Vec<&[u8]> = rows.iter().map(|&(s, at)| &self.stories[s][at..at + n]).collect();
        ContrastBatch::new(&prompts, &continuations, self.config.prompt_max, device)
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
        let [contrast, delta] = [values[0], values[1]];
        let bits = delta / std::f64::consts::LN_2;
        writeln!(
            self.log,
            "{{\"step\":{step},\"weight\":{weight},\"contrast\":{contrast},\"delta_bits\":{bits}}}"
        )
        .expect("failed to write contrast.jsonl");
        self.sums[0] += contrast;
        self.sums[1] += bits;
        self.count += 1;
        if self.count == PRINT_EVERY {
            let n = self.count as f64;
            println!(
                "contrast step={step} weight {weight:.4} loss {:.4} delta {:+.4} bits/char (mean of {} steps)",
                self.sums[0] / n,
                self.sums[1] / n,
                self.count
            );
            self.sums = [0.0; 2];
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

#[cfg(test)]
mod tests;
