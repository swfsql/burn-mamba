//! `--context-use` and `--long-stories` of this example: the measures of
//! [`crate::common::context`] and [`crate::common::long_stories`] (their
//! headers hold the definitions), over the trained [`MambaVocabNet`]. The lead
//! rows are the class latents.

pub use crate::common::context::*;

use crate::AppArgs;
use crate::training::Run;
use burn::prelude::*;
use burn_mamba::prelude::{MambaVocabNet, MambaVocabNetConfig};

/// Load the trained model and measure its context use over the first
/// `n_stories` validation stories, `batch` sequences per `forward` (see
/// [`crate::common::context::context_use`]).
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
    let logits_of = |x: Tensor<3, Int>| model.forward(x, None, run.ssd_path.clone(), None, None).0;
    crate::common::context::context_use(&logits_of, &device, &app_args.artifacts_path, n_stories, batch);
}

/// Load the trained model and score the long validation stories and the first
/// `valid_stories` validation stories over their whole length, `batch`
/// stories per `forward` (see [`crate::common::long_stories::long_stories`]).
pub fn long_stories(
    model_config: MambaVocabNetConfig,
    device: Device,
    app_args: &AppArgs,
    run: &Run,
    valid_stories: usize,
    batch: usize,
) {
    let model: MambaVocabNet = app_args
        .load_model(&model_config, &device)
        .expect("no trained model in the artifacts directory; run with --training first");
    let logits_of = |x: Tensor<3, Int>| model.forward(x, None, run.ssd_path.clone(), None, None).0;
    let artifacts = &app_args.artifacts_path;
    crate::common::long_stories::long_stories(&logits_of, &device, artifacts, valid_stories, batch);
}

#[cfg(test)]
mod tests;
