use crate::mamba2::*;
use burn::module::Module;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba2Caches<B: Backend> {
    /// # Shape
    /// [`Mamba2CachesConfig::n_real_caches`]
    pub caches: Vec<Mamba2Cache<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2CachesConfig {
    pub n_real_caches: usize,
    pub cache: Mamba2CacheConfig,
}

impl Mamba2CachesConfig {
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba2Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba2CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Caches<B> {
        let mut caches: Vec<Mamba2Cache<B>> = Vec::with_capacity(self.n_real_caches);
        for _ in 0..self.n_real_caches {
            let cache: Mamba2Cache<B> = self.cache.clone().init(device);
            caches.push(cache);
        }
        Mamba2Caches { caches }
    }
}

#[derive(Module, Debug)]
pub struct Mamba2Cache<B: Backend> {
    /// # Shape
    /// [batch, conv_dim, conv_kernel]
    pub conv_bvk: Tensor<B, 3>,
    /// # Shape
    /// [batch, nheads, per_head_dim, state_rank]
    pub ssm_bhpr: Tensor<B, 4>,
}

#[derive(Config, Debug)]
pub struct Mamba2CacheConfig {
    pub batch: usize,

    #[config(default = 128)]
    pub state_rank: usize,

    /// Convolution kernel size.
    #[config(default = 4)]
    pub conv_kernel: usize,

    pub conv_dim: usize,

    /// Head dimension.
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of heads.
    pub nheads: usize,
}

impl Mamba2CacheConfig {
    pub fn new_from_block_config(batch: usize, block_config: Mamba2Config) -> Self {
        Self {
            batch,
            state_rank: block_config.state_rank,
            conv_kernel: block_config.conv_kernel,
            conv_dim: block_config.conv_dim(),
            per_head_dim: block_config.per_head_dim,
            nheads: block_config.nheads(),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Cache<B> {
        let conv_bvk = Tensor::zeros(
            Shape::new([self.batch, self.conv_dim, self.conv_kernel]),
            device,
        );
        let ssm_bhpr = Tensor::zeros(
            Shape::new([self.batch, self.nheads, self.per_head_dim, self.state_rank]),
            device,
        );
        Mamba2Cache { conv_bvk, ssm_bhpr }
    }
}
