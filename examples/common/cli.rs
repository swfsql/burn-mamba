//! CLI plumbing shared by the examples: argument parsing into [`AppArgs`],
//! artifact-directory management, and load/save of the training config, model
//! config, model weights, and optimizer state.  See [`HELP`] for the full
//! command-line behaviour.

use crate::common::model::ModelConfigExt;
use burn::module::{ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamWConfig, ModuleOptimizer};
use burn::store::ModuleRecord;
use burn::prelude::*;
use std::collections::{HashMap, HashSet};
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
    pub fn load_model_config<ModelConfig: ModelConfigExt>(&self) -> Option<ModelConfig> {
        self.model_config
            .as_ref()
            .map(|path| {
                load_model_config::<ModelConfig>(path)
                    .expect("Failed to find the model config file {path:?}")
            })
            .or({
                let path = self
                    .artifacts_path
                    .join(MODEL_CONFIG_NAME)
                    .with_added_extension("json");
                load_model_config::<ModelConfig>(&path)
            })
    }

    /// Save the model weights into the artifacts directory.
    pub fn save_model(&self, model: &impl Module) {
        save_model(&self.artifacts_path, model)
    }

    /// Load model weights from the artifacts directory, if present.
    pub fn load_model<ModelConfig: ModelConfigExt>(
        &self,
        model_config: &ModelConfig,
        device: &Device,
    ) -> Option<ModelConfig::Model> {
        load_model(&self.artifacts_path, model_config, device)
    }

    /// Load the model if saved, otherwise initialise a new one and save it.
    pub fn load_or_save_model<ModelConfig: ModelConfigExt>(
        &self,
        model_config: &ModelConfig,
        device: &Device,
    ) -> ModelConfig::Model {
        self.load_model(model_config, device).unwrap_or_else(|| {
            println!("Initializing new model");
            let model_init = model_config.init(device);
            self.save_model(&model_init);
            model_init
        })
    }

    /// Save the optimizer state into the artifacts directory.
    pub fn save_optim(&self, optim: &ModuleOptimizer) {
        save_optim(&self.artifacts_path, optim)
    }

    /// Load optimizer state from the artifacts directory, if present. `model`
    /// is the (already loaded) module the optimizer drives — its live
    /// `ParamId`s prune orphaned state entries from the record (see the free
    /// [`load_optim`]).
    pub fn load_optim(
        &self,
        optim_config: &AdamWConfig,
        model: &impl Module,
    ) -> Option<ModuleOptimizer> {
        load_optim(&self.artifacts_path, optim_config, model)
    }

    /// Load the optimizer if saved, otherwise initialise a new one and save it.
    pub fn load_or_save_optim(
        &self,
        optim_config: &AdamWConfig,
        model: &impl Module,
    ) -> ModuleOptimizer {
        self.load_optim(optim_config, model).unwrap_or_else(|| {
            println!("Initializing new optim");
            let optim_init = optim_config.init();
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
        println!("removing {artifact_dir:?}/{{model,optim}}.{RECORD_EXT}");
        std::fs::remove_file(artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT))
            .expect("failed to remove the model");
        std::fs::remove_file(artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT))
            .expect("failed to remove the optim");
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
pub fn load_model_config<ModelConfig: Config>(path: &Path) -> Option<ModelConfig> {
    let exists = std::fs::exists(path).expect("failed to check {path:?}");
    if exists {
        println!("Loading model config from {path:?}");
        let model_config = ModelConfig::load(path).expect("Failed to load the model config");
        Some(model_config)
    } else {
        None
    }
}

/// Canonical burnpack file extension appended to the model/optim records.
///
/// `ModuleRecord`/`ModuleOptimizer` save/load auto-append this when the path
/// carries no extension, so spell it out here for the existence checks and the
/// `--remove-artifacts` cleanup to match the files actually written.
pub const RECORD_EXT: &str = "bpk";

/// Base filename (without extension) for the persisted model weights.
pub const MODEL_NAME: &str = "model";
/// Save model weights into `artifact_dir` as a burnpack record.
pub fn save_model(artifact_dir: &Path, model: &impl Module) {
    let path = artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT);
    println!("Saving model to {path:?}");
    model
        .clone()
        .into_record()
        .save(path)
        .expect("Failed to save the model");
}

/// Load model weights from `artifact_dir`, or `None` if absent.
///
/// After applying the record, the persisted `ParamId`s are restored onto the
/// model (see [`restore_param_ids`]) so the ParamId-keyed optimizer state
/// stays associated across process relaunches.
pub fn load_model<ModelConfig: ModelConfigExt>(
    artifact_dir: &Path,
    model_config: &ModelConfig,
    device: &Device,
) -> Option<ModelConfig::Model> {
    let path = artifact_dir.join(MODEL_NAME).with_extension(RECORD_EXT);
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
    if exists {
        println!("Loading model from {path:?}");
        let record = ModuleRecord::load(&path).expect("Failed to load the model record");
        let model = model_config.init(device).load_record(record);
        let model = restore_param_ids(model, &path);
        Some(model)
    } else {
        None
    }
}

/// Restore each parameter's persisted `ParamId` onto a freshly-loaded model.
///
/// A burnpack record stores every tensor's originating `ParamId`
/// ("training-state identity"), but burn's `load_record` discards them: the
/// loaded module keeps the ids freshly minted by `init()`, re-keying the whole
/// model on every process launch. Optimizer state, however, is keyed BY
/// `ParamId` — so without this restore, every resume orphans the entire loaded
/// optimizer state (silently resetting the Adam moments) and each checkpoint
/// re-saves the dead entries alongside the new ones: the optim record grows by
/// one full-model AdamW cohort (~2× model size) per relaunch. Stamping the
/// saved ids back keeps the (model, optim) key space stable; [`load_optim`]
/// then drops any entries that remain orphaned (cohorts from pre-fix resumes).
fn restore_param_ids<M: Module>(model: M, record_path: &Path) -> M {
    let ids = read_param_ids(record_path);
    let mut stamper = ParamIdStamper {
        path: Vec::new(),
        ids,
        missing: 0,
    };
    let model = model.map(&mut stamper);
    if stamper.missing > 0 {
        eprintln!(
            "warning: restore_param_ids: {} params have no persisted id in {record_path:?} \
             (their optimizer state starts fresh)",
            stamper.missing
        );
    }
    model
}

/// Read the `module path → persisted ParamId` map from a burnpack record file.
/// Header/metadata only — no tensor data is read.
fn read_param_ids(path: &Path) -> HashMap<String, ParamId> {
    let reader = burn_pack::Reader::from_file(path)
        .unwrap_or_else(|e| panic!("Failed to re-read the record {path:?} for ParamIds: {e:?}"));
    reader
        .into_tensors()
        .unwrap_or_else(|e| panic!("Failed to list the record tensors of {path:?}: {e:?}"))
        .into_iter()
        .filter_map(|t| t.param_id.map(|id| (t.name, ParamId::from(id))))
        .collect()
}

/// [`ModuleMapper`] that replaces each parameter's fresh (`init()`-minted)
/// `ParamId` with the id persisted in the record, matched by module path.
/// Values, param mappers, and `require_grad` are untouched. The traversal
/// mirrors burn-core's record collector/mapper: every submodule name is pushed
/// (no enum-variant skipping) and paths join with `.` — the map keys come from
/// the same collector-written record, so the two stay symmetric.
struct ParamIdStamper {
    path: Vec<String>,
    ids: HashMap<String, ParamId>,
    missing: usize,
}

macro_rules! stamp_kind {
    ($method:ident, $kind:ty) => {
        fn $method<const D: usize>(
            &mut self,
            param: Param<Tensor<D, $kind>>,
        ) -> Param<Tensor<D, $kind>> {
            match self.ids.get(&self.path.join(".")) {
                Some(&id) => {
                    let (_fresh_id, value, mapper) = param.consume();
                    Param::from_mapped_value(id, value, mapper)
                }
                None => {
                    self.missing += 1;
                    param
                }
            }
        }
    };
}

impl ModuleMapper for ParamIdStamper {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    stamp_kind!(map_float, Float);
    stamp_kind!(map_int, Int);
    stamp_kind!(map_bool, Bool);
}

/// Collect the `ParamId` of every parameter in a module (the "live" id set).
fn collect_param_ids(module: &impl Module) -> HashSet<ParamId> {
    struct ParamIdCollector(HashSet<ParamId>);
    impl ModuleVisitor for ParamIdCollector {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.0.insert(param.id);
        }
        fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
            self.0.insert(param.id);
        }
        fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
            self.0.insert(param.id);
        }
    }
    let mut collector = ParamIdCollector(HashSet::new());
    module.visit(&mut collector);
    collector.0
}

/// Base filename for the persisted optimizer state.
pub const OPTIM_NAME: &str = "optim";
/// Save optimizer state into `artifact_dir` as a burnpack record.
pub fn save_optim(artifact_dir: &Path, optim: &ModuleOptimizer) {
    let path = artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT);
    println!("Saving optim to {path:?}");
    optim.save(path).expect("Failed to save the optim");
}

/// Load optimizer state from `artifact_dir`, or `None` if absent.
///
/// The record is filtered against `model`'s live `ParamId`s before loading:
/// optimizer state is keyed by `ParamId` and burn never prunes entries whose
/// parameter no longer exists — before [`restore_param_ids`], every relaunch
/// re-keyed the whole model, so resumed runs accreted one dead full-model
/// AdamW cohort (~2× model size) per restart. Dropping the orphans here heals
/// those bloated records on the next load→checkpoint cycle and keeps the dead
/// state from occupying device memory for the whole run.
pub fn load_optim<M: Module>(
    artifact_dir: &Path,
    optim_config: &AdamWConfig,
    model: &M,
) -> Option<ModuleOptimizer> {
    let path = artifact_dir.join(OPTIM_NAME).with_extension(RECORD_EXT);
    let exists = std::fs::exists(&path).expect("failed to check {path:?}");
    if !exists {
        return None;
    }
    println!("Loading initial optim from {path:?}");
    let live = collect_param_ids(model);
    let reader = burn_pack::Reader::from_file(&path)
        .unwrap_or_else(|e| panic!("Failed to read the optim record {path:?}: {e:?}"));
    // Optimizer-record scalar keys are `"{param_id}.{field}"` (`__rank`, Adam `time`).
    let scalar_is_live = |key: &str| {
        key.split_once('.')
            .and_then(|(id, _)| id.parse::<u64>().ok())
            .is_some_and(|id| live.contains(&ParamId::from(id)))
    };
    let scalars: Vec<(String, burn_pack::Scalar)> = reader
        .scalars()
        .iter()
        .filter(|(key, _)| scalar_is_live(key))
        .map(|(key, value)| (key.clone(), *value))
        .collect();
    let tensors = reader
        .into_tensors()
        .unwrap_or_else(|e| panic!("Failed to read the optim tensors of {path:?}: {e:?}"));
    let total = tensors.len();
    let tensors: Vec<_> = tensors
        .into_iter()
        .filter(|t| t.param_id.is_some_and(|id| live.contains(&ParamId::from(id))))
        .collect();
    if tensors.len() < total {
        eprintln!(
            "warning: load_optim: dropping {} of {total} optimizer-state tensors as orphaned \
             (ParamIds absent from the model — dead cohorts from pre-fix relaunches); \
             the next checkpoint re-saves the record pruned",
            total - tensors.len()
        );
    }
    let mut writer = burn_pack::Writer::new(tensors);
    for (key, value) in &scalars {
        writer = writer.with_scalar(key, *value);
    }
    let bytes = writer
        .into_bytes()
        .unwrap_or_else(|e| panic!("Failed to repack the optim record {path:?}: {e:?}"));
    let optim = optim_config
        .init()
        .from_bytes(bytes)
        .expect("Failed to load the initial optim");
    Some(optim)
}
