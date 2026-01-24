use crate::mamba1::*;
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::Initializer,
};

#[derive(Module, Debug)]
pub struct Mamba1Cache<B: Backend> {
    /// # Shape
    /// [batch, d_inner, d_conv]
    pub conv: Param<Tensor<B, 3>>,
    /// # Shape
    /// [batch, d_inner, d_state]
    pub ssm: Param<Tensor<B, 3>>,
}

#[derive(Config, Debug)]
pub struct Mamba1CacheConfig {
    pub batch: usize,

    /// latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    #[config(default = 16)]
    pub d_state: usize,

    #[config(default = 4)]
    pub d_conv: usize,

    pub d_inner: usize,
}

impl Mamba1CacheConfig {
    pub fn new_from_block_config(batch: usize, block_config: Mamba1Config) -> Self {
        Self {
            batch,
            d_state: block_config.d_state,
            d_conv: block_config.d_conv,
            d_inner: block_config.d_inner(),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Cache<B> {
        let conv = Initializer::Zeros.init([self.batch, self.d_inner, self.d_conv], device);
        let ssm = Initializer::Zeros.init([self.batch, self.d_inner, self.d_state], device);
        Mamba1Cache { conv, ssm }
    }
}

#[derive(Module, Debug)]
pub struct Mamba1Caches<B: Backend> {
    /// # Shape
    /// [n_layers]
    pub caches: Vec<Mamba1Cache<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba1CachesConfig {
    pub n_layers: usize,
    pub cache: Mamba1CacheConfig,
}

impl Mamba1CachesConfig {
    pub fn new_from_block_config(
        n_layers: usize,
        batch: usize,
        block_config: Mamba1Config,
    ) -> Self {
        Self {
            n_layers,
            cache: Mamba1CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Caches<B> {
        let mut caches: Vec<Mamba1Cache<B>> = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            let cache: Mamba1Cache<B> = self.cache.clone().init(device);
            caches.push(cache);
        }
        Mamba1Caches { caches }
    }
}
