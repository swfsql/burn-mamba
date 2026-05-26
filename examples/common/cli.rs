//! CLI plumbing shared by the examples: argument parsing into [`AppArgs`],
//! artifact-directory management, and load/save of the training config, model
//! config, model weights, and optimizer state.  See [`HELP`] for the full
//! command-line behaviour.

use crate::common::{backend::RecorderTy, model::ModelConfigExt, optim::OptimConfigExt};
use burn::module::AutodiffModule;
use burn::record::{FileRecorder, Recorder};
use burn::{optim::Optimizer, prelude::*, tensor::backend::AutodiffBackend};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The `--help` text describing every flag and the train/infer/config flow.
pub const HELP: &str = "\
Burn Mamba Example

A command-line tool for training and/or running inference with machine learning models.
Models, optimizers, and configurations are persisted in an artifacts directory.

USAGE:
    example-name [OPTIONS] [-- <EXTRA_ARGS>...]

When no --training or --inference flag is provided, the program exits after handling configuration logic.

BEHAVIOR OVERVIEW
- The program manages two configurations: training config and model config.
- If --training-config or --model-config is given, the corresponding config is loaded from the specified file and saved to the artifacts directory (overwriting any existing file).
- If no explicit config file is provided for a component, the program attempts to load it from the artifacts directory; if absent, a default configuration is created and saved.
- The artifacts directory (--artifacts-path) is used to read/write model weights, optimizer state, and configurations. If not specified, a new temporary directory is created and its path is printed.
- With --remove-artifacts, any existing model and optimizer files in the artifacts directory are deleted before training (if --training is active).
- Model and optimizer weights are loaded from the artifacts directory if present; otherwise new ones are created and saved.
- If both --training and --inference are specified, training executes first, followed by inference using the trained model.
- Any arguments following -- are captured as-is and forwarded to downstream processing.

FLAGS:
    -h, --help                  Show this help message and exit

OPTIONS:
    -t, --training              Run training (creates or updates model / optimizer)
    -i, --inference             Run inference after training (if both flags are used) or immediately (if only inference is requested)
    -r, --remove-artifacts      Delete existing model and optimizer files from the artifacts directory before training
                                (has no effect if --training is not used)
    -c, --training-config <PATH>
                                Load training configuration from this file (overrides any config in artifacts directory)
    -m, --model-config <PATH>   Load model configuration from this file (overrides any config in artifacts directory)
    -a, --artifacts-path <PATH>
                                Directory where configurations, model weights, and optimizer state are saved and loaded.
                                If the directory does not exist, it will be created.
                                Defaults to a newly created temporary directory (path will be printed).

ARGS:
    -- <EXTRA_ARGS>             All arguments after -- are forwarded verbatim to further processing stages.
                                If further processing is available, passing -h or --help will display its help information.
";

/// Parsed command-line arguments. For field descriptions, see [`HELP`].
#[derive(Debug)]
pub struct AppArgs {
    /// Whether to run training.
    pub training: bool,
    /// Whether to run inference.
    pub inference: bool,
    /// Whether to delete existing model/optim artifacts before training.
    pub remove_artifacts: bool,
    /// Optional path to load the training config from.
    pub training_config: Option<PathBuf>,
    /// Optional path to load the model config from.
    pub model_config: Option<PathBuf>,
    /// Directory for configs, model weights, and optimizer state.
    pub artifacts_path: PathBuf,
    /// Arguments after `--`, forwarded verbatim to downstream processing.
    pub extra_args: Vec<OsString>,
}

impl AppArgs {
    /// Parse [`AppArgs`] from `std::env::args_os` (handles `--`, `-h/--help`).
    pub fn parse() -> Result<Self, pico_args::Error> {
        let mut args: Vec<_> = std::env::args_os().collect();
        args.remove(0); // remove the executable path.

        // Find and process `--`.
        let extra_args = if let Some(dash_dash) = args.iter().position(|arg| arg == "--") {
            // Store all arguments following ...
            let later_args = args.drain(dash_dash + 1..).collect();
            // .. then remove the `--`
            args.pop();
            later_args
        } else {
            Vec::new()
        };

        let mut pargs = pico_args::Arguments::from_vec(args);

        // Help has a higher priority and should be handled separately.
        if pargs.contains(["-h", "--help"]) {
            println!("{}", HELP);
            std::process::exit(0);
        }

        let args = AppArgs {
            training_config: pargs
                .opt_value_from_os_str(["-c", "--training-config"], parse_path)?,
            model_config: pargs.opt_value_from_os_str(["-m", "--model-config"], parse_path)?,
            artifacts_path: pargs
                .opt_value_from_os_str(["-a", "--artifacts-path"], parse_path)?
                .unwrap_or_else(|| {
                    // e.g. /tmp/burn-mamba-fibonacci-abcd-0
                    let name = format!(
                        "{}-{}-",
                        std::env!("CARGO_PKG_NAME"), // burn-mamba
                        std::env!("CARGO_BIN_NAME")  // e.g. fibonacci
                    );
                    let tmp = temp_dir::TempDir::with_prefix(name)
                        .expect("Failed to create the temporary directory")
                        .dont_delete_on_drop();
                    let path = tmp.path();
                    println!("new artifacts directory: {path:?}");
                    path.into()
                }),
            // must parse flags after values
            training: pargs.contains(["-t", "--training"]),
            inference: pargs.contains(["-i", "--inference"]),
            remove_artifacts: pargs.contains(["-r", "--remove-artifacts"]),
            extra_args,
        };

        let remaining = pargs.finish();
        if !remaining.is_empty() {
            panic!("unused arguments: {remaining:?}");
        }

        Ok(args)
    }

    /// Create the artifacts directory (removing model/optim first if requested).
    pub fn create_artifact_dir(&self) {
        create_artifact_dir(&self.artifacts_path, self.remove_artifacts && self.training)
    }

    /// Save the training config into the artifacts directory.
    pub fn save_training_config(&self, training_config: &impl Config) {
        let path = self
            .artifacts_path
            .join(TRAINING_CONFIG_NAME)
            .with_added_extension("json");
        save_training_config(&path, training_config)
    }

    /// Load the training config (from `--training-config` or the artifacts dir).
    pub fn load_training_config<TrainingConfig: Config>(&self) -> Option<TrainingConfig> {
        self.training_config
            .as_ref()
            .map(|path| {
                load_training_config(path)
                    .expect("Failed to find the training config file {path:?}")
            })
            .or({
                let path = self
                    .artifacts_path
                    .join(TRAINING_CONFIG_NAME)
                    .with_added_extension("json");
                load_training_config(&path)
            })
    }

    /// Save the model config into the artifacts directory.
    pub fn save_model_config(&self, model_config: &impl Config) {
        let path = self
            .artifacts_path
            .join(MODEL_CONFIG_NAME)
            .with_added_extension("json");
        save_model_config(&path, model_config)
    }

    /// Load the model config (from `--model-config` or the artifacts dir).
    pub fn load_model_config<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
    ) -> Option<ModelConfig> {
        self.model_config
            .as_ref()
            .map(|path| {
                load_model_config::<B, ModelConfig>(path)
                    .expect("Failed to find the model config file {path:?}")
            })
            .or({
                let path = self
                    .artifacts_path
                    .join(MODEL_CONFIG_NAME)
                    .with_added_extension("json");
                load_model_config::<B, ModelConfig>(&path)
            })
    }

    /// Save the model weights into the artifacts directory.
    pub fn save_model<B: Backend>(&self, model: &impl Module<B>) {
        save_model(&self.artifacts_path, model)
    }

    /// Load model weights from the artifacts directory, if present.
    pub fn load_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
        model_config: &ModelConfig,
        device: &B::Device,
    ) -> Option<ModelConfig::Model> {
        load_model(&self.artifacts_path, model_config, device)
    }

    /// Load the model if saved, otherwise initialise a new one and save it.
    pub fn load_or_save_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
        model_config: &ModelConfig,
        device: &B::Device,
    ) -> ModelConfig::Model {
        self.load_model(model_config, device).unwrap_or_else(|| {
            println!("Initializing new model");
            let model_init = init_model(model_config, device);
            self.save_model(&model_init);
            model_init
        })
    }

    /// Save the optimizer state into the artifacts directory.
    pub fn save_optim<AutoB, AutoM>(&self, optim: &impl Optimizer<AutoM, AutoB>)
    where
        AutoB: AutodiffBackend,
        AutoM: AutodiffModule<AutoB>,
    {
        save_optim(&self.artifacts_path, optim)
    }

    /// Load optimizer state from the artifacts directory, if present.
    pub fn load_optim<AutoB, AutoM, OptimConfig>(
        &self,
        optim_config: &OptimConfig,
        device: &AutoB::Device,
    ) -> Option<OptimConfig::Adaptor>
    where
        AutoB: AutodiffBackend,
        AutoM: AutodiffModule<AutoB>,
        OptimConfig: OptimConfigExt<AutoB, AutoM>,
    {
        load_optim(&self.artifacts_path, optim_config, device)
    }

    /// Load the optimizer if saved, otherwise initialise a new one and save it.
    pub fn load_or_save_optim<AutoB, AutoM, OptimConfig>(
        &self,
        optim_config: &OptimConfig,
        device: &AutoB::Device,
    ) -> OptimConfig::Adaptor
    where
        AutoB: AutodiffBackend,
        AutoM: AutodiffModule<AutoB>,
        OptimConfig: OptimConfigExt<AutoB, AutoM>,
    {
        self.load_optim(optim_config, device).unwrap_or_else(|| {
            println!("Initializing new optim");
            let optim_init = init_optim(optim_config, device);
            self.save_optim(&optim_init);
            optim_init
        })
    }
}

/// `pico-args` value parser turning an `OsStr` into a `PathBuf`.
pub fn parse_path(s: &std::ffi::OsStr) -> Result<std::path::PathBuf, &'static str> {
    Ok(s.into())
}

/// Create the artifacts directory; when `delete` is set, remove any existing
/// `model`/`optim` files first.
pub fn create_artifact_dir(artifact_dir: &Path, delete: bool) {
    if delete {
        // enforce that the removal should not have errors,
        // including for when files didn't exist
        println!("removing {artifact_dir:?}/{{model,optim}}");
        std::fs::remove_file(artifact_dir.join("model")).expect("failed to remove the model");
        std::fs::remove_file(artifact_dir.join("optim")).expect("failed to remove the optim");
    }
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Base filename (without extension) for the persisted training config.
pub const TRAINING_CONFIG_NAME: &str = "training_config";
/// Save a training config as JSON to `path`.
pub fn save_training_config(path: &Path, training_config: &impl Config) {
    println!("Saving training config into {path:?}");
    training_config
        .save(path)
        .expect("Failed to save the training config");
}

/// Load a training config from `path`, or `None` if the file is absent.
pub fn load_training_config<TrainingConfig: Config>(path: &Path) -> Option<TrainingConfig> {
    let exists = std::fs::exists(path).expect("failed to check {path:?}");
    if exists {
        println!("Loading training config from {path:?}");
        let training_config =
            TrainingConfig::load(path).expect("Failed to load the training config");
        Some(training_config)
    } else {
        None
    }
}

/// Base filename (without extension) for the persisted model config.
pub const MODEL_CONFIG_NAME: &str = "model_config";
/// Save a model config as JSON to `path`.
pub fn save_model_config(path: &Path, model_config: &impl Config) {
    println!("Saving model config into {path:?}");
    model_config
        .save(path)
        .expect("Failed to save the model config");
}

/// Load a model config from `path`, or `None` if the file is absent.
pub fn load_model_config<B: Backend, ModelConfig: Config>(path: &Path) -> Option<ModelConfig> {
    let exists = std::fs::exists(path).expect("failed to check {path:?}");
    if exists {
        println!("Loading model config from {path:?}");
        let model_config = ModelConfig::load(path).expect("Failed to load the model config");
        Some(model_config)
    } else {
        None
    }
}

/// Base filename (without extension) for the persisted model weights.
pub const MODEL_NAME: &str = "model";
/// Save model weights into `artifact_dir` using the configured recorder.
pub fn save_model<B: Backend>(artifact_dir: &Path, model: &impl Module<B>) {
    let path = artifact_dir.join(MODEL_NAME);
    let file_ext = <RecorderTy as FileRecorder<B>>::file_extension();
    let path_ext = path.with_added_extension(file_ext);
    println!("Saving model to {path_ext:?}");
    model
        .clone()
        .save_file(path, &RecorderTy::new()) // ext added automatically
        .expect("Failed to save the model");
}

/// Load model weights from `artifact_dir`, or `None` if absent.
pub fn load_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
    artifact_dir: &Path,
    model_config: &ModelConfig,
    device: &B::Device,
) -> Option<ModelConfig::Model> {
    let path = artifact_dir.join(MODEL_NAME);
    let file_ext = <RecorderTy as FileRecorder<B>>::file_extension();
    let path_ext = path.with_added_extension(file_ext);
    let exists = std::fs::exists(&path_ext).expect("failed to check {path:?}");
    if exists {
        println!("Loading model from {path_ext:?}");
        let model_init = init_model(model_config, device);
        let model = model_init
            .load_file(path, &RecorderTy::new(), device) // ext added automatically
            .expect("Failed to load the initial model");
        Some(model)
    } else {
        None
    }
}

/// Initialise a fresh model from its config on `device`.
pub fn init_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
    model_config: &ModelConfig,
    device: &B::Device,
) -> ModelConfig::Model {
    model_config.init(device)
}

/// Base filename (without extension) for the persisted optimizer state.
pub const OPTIM_NAME: &str = "optim";
/// Save optimizer state into `artifact_dir` using the configured recorder.
pub fn save_optim<AutoB, AutoM, Optim>(artifact_dir: &Path, optim: &Optim)
where
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
    Optim: Optimizer<AutoM, AutoB>,
{
    let path = artifact_dir.join(OPTIM_NAME);
    let file_ext = <RecorderTy as FileRecorder<AutoB>>::file_extension();
    let path_ext = path.with_added_extension(file_ext);
    println!("Saving optim to {path_ext:?}");
    let record = optim.to_record();
    RecorderTy::new()
        .record(record, path) // ext added automatically
        .expect("Failed to save the optim");
}

/// Load optimizer state from `artifact_dir`, or `None` if absent.
pub fn load_optim<AutoB, AutoM, OptimConfig>(
    artifact_dir: &Path,
    optim_config: &OptimConfig,
    device: &AutoB::Device,
) -> Option<OptimConfig::Adaptor>
where
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
    OptimConfig: OptimConfigExt<AutoB, AutoM>,
{
    let path = artifact_dir.join(OPTIM_NAME);
    let file_ext = <RecorderTy as FileRecorder<AutoB>>::file_extension();
    let path_ext = path.with_added_extension(file_ext);
    let exists = std::fs::exists(&path_ext).expect("failed to check {path:?}");
    if exists {
        println!("Loading initial optim from {path_ext:?}");
        let optim_init = init_optim(optim_config, device);
        let record = RecorderTy::new()
            .load(path, device) // ext added automatically
            .expect("Failed to load the initial optim");
        let optim = optim_init.load_record(record);
        Some(optim)
    } else {
        None
    }
}

/// Initialise a fresh optimizer from its config.
pub fn init_optim<AutoB, AutoM, OptimConfig>(
    optim_config: &OptimConfig,
    _device: &AutoB::Device,
) -> OptimConfig::Adaptor
where
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
    OptimConfig: OptimConfigExt<AutoB, AutoM>,
{
    optim_config.init()
}
