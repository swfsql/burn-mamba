//! A small wrapper trait letting an optimizer config build its optimizer for a
//! given model/backend, so the generic training loop can stay optimizer-agnostic
//! (implemented here for `AdamW`).

use burn::prelude::*;
use burn::{
    module::AutodiffModule,
    optim::{self, Optimizer, SimpleOptimizer, adaptor::OptimizerAdaptor},
    tensor::backend::AutodiffBackend,
};

/// An optimizer config that can initialise its optimizer for model `AutoM` on
/// autodiff backend `AutoB`.
pub trait OptimConfigExt<AutoB, AutoM>
where
    Self: Config,
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
{
    /// The underlying simple optimizer (e.g. `AdamW`).
    type Optim: SimpleOptimizer<AutoB::InnerBackend>;
    /// The model-bound optimizer adaptor returned by [`Self::init`].
    type Adaptor: Optimizer<AutoM, AutoB>;
    /// Build the optimizer adaptor for the model.
    fn init(&self) -> Self::Adaptor;
}

impl<AutoB, AutoM> OptimConfigExt<AutoB, AutoM> for optim::AdamWConfig
where
    Self: Config,
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
{
    type Optim = burn::optim::AdamW;
    type Adaptor = OptimizerAdaptor<Self::Optim, AutoM, AutoB>;
    fn init(&self) -> Self::Adaptor {
        optim::AdamWConfig::init::<AutoB, AutoM>(self)
    }
}
