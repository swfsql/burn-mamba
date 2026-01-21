use burn::prelude::*;
use burn::{
    module::AutodiffModule,
    optim::{self, Optimizer, SimpleOptimizer, adaptor::OptimizerAdaptor},
    tensor::backend::AutodiffBackend,
};

pub trait OptimConfigExt<AutoB, AutoM>
where
    Self: Config,
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
{
    type Optim: SimpleOptimizer<AutoB::InnerBackend>;
    type Adaptor: Optimizer<AutoM, AutoB>;
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
