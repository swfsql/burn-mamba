use crate::common::{backend::RecorderTy, model::ModelConfigExt, optim::OptimConfigExt};
use burn::module::AutodiffModule;
use burn::record::{FileRecorder, Recorder};
use burn::{optim::Optimizer, prelude::*, tensor::backend::AutodiffBackend};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::common::cli::{MODEL_CONFIG_NAME, parse_path};

pub const HELP: &str = "\
Burn Mamba Distillation

Extra arguments processing.

USAGE:
    -- [OPTIONS]

BEHAVIOR OVERVIEW
- The teacher artifacts directory (--teacher-artifacts-path) is used to read the teacher's model weights and configurations.

FLAGS:
    -h, --help                  Show this help message and exit

OPTIONS:
    --teacher-artifacts-path <PATH>
                                Directory where the teacher's configurations and model weights are loaded.
                                Defaults to the mnist-class/tmp/f32/0/ directory.
";

/// For field descriptions, see [HELP](HELP).
#[derive(Debug)]
pub struct TeacherArgs {
    pub teacher_artifacts_path: PathBuf,
}

impl TeacherArgs {
    pub fn parse(extra_args: Vec<OsString>) -> Result<Self, pico_args::Error> {
        let mut pargs = pico_args::Arguments::from_vec(extra_args);

        // Help has a higher priority and should be handled separately.
        if pargs.contains(["-h", "--help"]) {
            println!("{}", HELP);
            std::process::exit(0);
        }

        let mut args = TeacherArgs {
            teacher_artifacts_path: pargs
                .opt_value_from_os_str("--teacher-artifacts-path", parse_path)?
                .unwrap_or_else(|| {
                    // $CARGO_MANIFEST_DIR/examples/mnist-class/tmp/f32/0/
                    PathBuf::from(std::env!("CARGO_MANIFEST_DIR"))
                        .join("examples/mnist-class/tmp/f32/0")
                }),
        };

        let remaining = pargs.finish();
        if !remaining.is_empty() {
            panic!("unused extra arguments: {remaining:?}");
        }

        Ok(args)
    }

    pub fn load_model_config<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
    ) -> Option<ModelConfig> {
        {
            let path = self
                .teacher_artifacts_path
                .join(MODEL_CONFIG_NAME)
                .with_added_extension("json");
            crate::common::cli::load_model_config::<B, ModelConfig>(&path)
        }
    }

    pub fn load_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
        model_config: &ModelConfig,
        device: &B::Device,
    ) -> Option<ModelConfig::Model> {
        crate::common::cli::load_model(&self.teacher_artifacts_path, model_config, device)
    }
}
