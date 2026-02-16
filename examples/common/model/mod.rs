use burn::prelude::*;

pub mod bidi;
mod model;

pub use model::{
    MyMamba2Network, MyMamba2NetworkConfig, mamba2_block_config, mamba2_layers_config,
};

pub trait ModelConfigExt<B: Backend>: Config {
    type Model: Module<B>;
    fn init(&self, device: &B::Device) -> Self::Model;
}
