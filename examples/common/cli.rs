use crate::common::{backend::RecorderTy, model::ModelConfigExt, optim::OptimConfigExt};
use burn::module::AutodiffModule;
use burn::record::{FileRecorder, Recorder};
use burn::{optim::Optimizer, prelude::*, tensor::backend::AutodiffBackend};
use std::path::{Path, PathBuf};

pub const HELP: &str = "\
Burn Mamba Example

A command-line tool for training and/or running inference with machine learning models.
Models, optimizers, and configurations are persisted in an artifacts directory.

USAGE:
    example-name [OPTIONS]

When no --training or --inference flag is provided, the program exits after handling configuration logic.

BEHAVIOR OVERVIEW
- The program manages two configurations: training config and model config.
- If --training-config or --model-config is given, the corresponding config is loaded from the specified file and saved to the artifacts directory (overwriting any existing file).
- If no explicit config file is provided for a component, the program attempts to load it from the artifacts directory; if absent, a default configuration is created and saved.
- The artifacts directory (--artifacts-path) is used to read/write model weights, optimizer state, and configurations. If not specified, a new temporary directory is created and its path is printed.
- With --remove-artifacts, any existing model and optimizer files in the artifacts directory are deleted before training (if --training is active).
- Model and optimizer weights are loaded from the artifacts directory if present; otherwise new ones are created and saved.
- If both --training and --inference are specified, training executes first, followed by inference using the trained model.

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
";

#[derive(Debug)]
pub struct AppArgs {
    pub training: bool,
    pub inference: bool,
    pub remove_artifacts: bool,
    pub training_config: Option<PathBuf>,
    pub model_config: Option<PathBuf>,
    pub artifacts_path: PathBuf,
}

impl AppArgs {
    pub fn parse() -> Result<Self, pico_args::Error> {
        let mut pargs = pico_args::Arguments::from_env();

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
        };

        // It's up to the caller what to do with the remaining arguments.
        let remaining = pargs.finish();
        if !remaining.is_empty() {
            panic!("unused arguments: {remaining:?}");
        }

        Ok(args)
    }

    pub fn create_artifact_dir(&self) {
        create_artifact_dir(&self.artifacts_path, self.remove_artifacts && self.training)
    }

    pub fn save_training_config(&self, training_config: &impl Config) {
        let path = self
            .artifacts_path
            .join(TRAINING_CONFIG_NAME)
            .with_added_extension("json");
        save_training_config(&path, training_config)
    }

    pub fn load_training_config<TrainingConfig: Config>(&self) -> Option<TrainingConfig> {
        self.training_config
            .as_ref()
            .map(|path| {
                load_training_config(&path)
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

    pub fn save_model_config(&self, model_config: &impl Config) {
        let path = self
            .artifacts_path
            .join(MODEL_CONFIG_NAME)
            .with_added_extension("json");
        save_model_config(&path, model_config)
    }

    pub fn load_model_config<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
    ) -> Option<ModelConfig> {
        self.model_config
            .as_ref()
            .map(|path| {
                load_model_config::<B, ModelConfig>(&path)
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

    pub fn save_model<B: Backend>(&self, model: &impl Module<B>) {
        save_model(&self.artifacts_path, model)
    }

    pub fn load_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
        &self,
        model_config: &ModelConfig,
        device: &B::Device,
    ) -> Option<ModelConfig::Model> {
        load_model(&self.artifacts_path, model_config, device)
    }

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

    pub fn save_optim<AutoB, AutoM>(&self, optim: &impl Optimizer<AutoM, AutoB>)
    where
        AutoB: AutodiffBackend,
        AutoM: AutodiffModule<AutoB>,
    {
        save_optim(&self.artifacts_path, optim)
    }

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

fn parse_path(s: &std::ffi::OsStr) -> Result<std::path::PathBuf, &'static str> {
    Ok(s.into())
}

// Create the directory to save the model and model config
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

pub const TRAINING_CONFIG_NAME: &'static str = "training_config";
pub fn save_training_config(path: &Path, training_config: &impl Config) {
    println!("Saving training config into {path:?}");
    training_config
        .save(path)
        .expect("Failed to save the training config");
}

pub fn load_training_config<TrainingConfig: Config>(path: &Path) -> Option<TrainingConfig> {
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
    if exists {
        println!("Loading training config from {path:?}");
        let training_config =
            TrainingConfig::load(path).expect("Failed to load the training config");
        Some(training_config)
    } else {
        None
    }
}

pub const MODEL_CONFIG_NAME: &'static str = "model_config";
pub fn save_model_config(path: &Path, model_config: &impl Config) {
    println!("Saving model config into {path:?}");
    model_config
        .save(path)
        .expect("Failed to save the model config");
}

pub fn load_model_config<B: Backend, ModelConfig: Config>(path: &Path) -> Option<ModelConfig> {
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
    if exists {
        println!("Loading model config from {path:?}");
        let model_config = ModelConfig::load(path).expect("Failed to load the model config");
        Some(model_config)
    } else {
        None
    }
}

pub const MODEL_NAME: &'static str = "model";
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

pub fn init_model<B: Backend, ModelConfig: ModelConfigExt<B>>(
    model_config: &ModelConfig,
    device: &B::Device,
) -> ModelConfig::Model {
    model_config.init(device)
}

pub const OPTIM_NAME: &'static str = "optim";
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
