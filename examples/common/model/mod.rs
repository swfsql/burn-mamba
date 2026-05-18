use burn::prelude::*;

pub mod bidi;
mod model;

pub use model::{
    mamba2::{MyMamba2Network, MyMamba2NetworkConfig, mamba2_block_config, mamba2_layers_config},
    mamba3::{MyMamba3Network, MyMamba3NetworkConfig, mamba3_block_config, mamba3_layers_config},
};

pub trait ModelConfigExt<B: Backend>: Config {
    type Model: Module<B>;
    fn init(&self, device: &B::Device) -> Self::Model;
}
