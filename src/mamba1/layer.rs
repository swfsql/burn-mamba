use crate::mamba1::*;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba1Layer<B: Backend> {
    pub norm: RmsNorm<B>,
    pub mamba_block: Mamba1<B>,
}

#[derive(Config, Debug)]
pub struct Mamba1LayerConfig {
    pub mamba_block: Mamba1Config,
}

impl Mamba1LayerConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Layer<B> {
        Mamba1Layer {
            norm: RmsNormConfig::new(self.mamba_block.d_model).init(device),
            mamba_block: Mamba1Config::new(self.mamba_block.d_model)
                .with_d_state(self.mamba_block.d_state)
                .with_dt_rank(self.mamba_block.dt_rank)
                .with_d_conv(self.mamba_block.d_conv)
                .with_d_inner(self.mamba_block.d_inner)
                .init(device),
        }
    }
}

impl<B: Backend> Mamba1Layer<B> {
    /// See also [`Self::step`].
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);

        let x = self.mamba_block.forward(x);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        x + res
    }
}

impl<B: Backend> Mamba1Layer<B> {
    /// See also [`Self::forward`].
    ///
    /// # Shapes
    ///   - Input [batch, d_model]
    ///   - Output [batch, d_model]
    pub fn step(&self, x: Tensor<B, 2>, cache: Mamba1Cache<B>) -> (Tensor<B, 2>, Mamba1Cache<B>) {
        let [batch, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);
        let (x, cache) = self.mamba_block.step(x, cache);
        debug_assert_eq!([batch, d_model], x.dims());

        (x + res, cache)
    }
}
