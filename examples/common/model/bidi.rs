use crate::common::model::{ModelConfigExt, mamba2_block_config};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn_mamba::mamba2::bidi::naive::{
    Mamba2BidiLayers, Mamba2BidiLayersConfig, OutputMergeConfig,
};
use burn_mamba::prelude::*;
use burn_mamba::schedule::BidiSchedule;

/// Basic Mamba network containing input and output heads, with Mamba2Layers in between.
#[derive(Config, Debug)]
pub struct MyMamba2BidiNetworkConfig {
    #[config(default = 1)]
    pub input_size: usize,

    #[config(
        default = "mamba2_bidi_layers_config(1, None, mamba2_block_config(1, 1, 1, 1, 1), OutputMergeConfig::mean(1))"
    )]
    pub layers: Mamba2BidiLayersConfig,

    #[config(default = 1)]
    pub output_size: usize,
}

pub fn mamba2_bidi_layers_config(
    n_real_layers: usize,
    n_virtual_layers: Option<(usize, BidiSchedule)>,
    mamba_block: Mamba2Config,
    outputs_merge: Vec<OutputMergeConfig>,
) -> Mamba2BidiLayersConfig {
    Mamba2BidiLayersConfig::new(n_real_layers, mamba_block, outputs_merge)
        .with_n_virtual_layers(n_virtual_layers)
}

#[derive(Module, Debug)]
pub struct MyMamba2BidiNetwork<B: Backend> {
    pub in_proj: Linear<B>,
    pub layers: Mamba2BidiLayers<B>,
    pub out_proj: Linear<B>,
}

impl<B: Backend> ModelConfigExt<B> for MyMamba2BidiNetworkConfig {
    type Model = MyMamba2BidiNetwork<B>;

    /// Returns the initialized model.
    fn init(&self, device: &B::Device) -> Self::Model {
        let d_model = self.layers.mamba_block.d_model;
        let in_proj = LinearConfig::new(self.input_size, d_model)
            .with_bias(true)
            .init(device);
        let layers = self.layers.init(device);
        let out_proj = LinearConfig::new(d_model, self.output_size)
            .with_bias(true)
            .init(device);
        MyMamba2BidiNetwork {
            in_proj,
            layers,
            out_proj,
        }
    }
}

impl<B: Backend> MyMamba2BidiNetwork<B>
where
    B: mamba2::gpu::BackendExt,
{
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        caches: Option<Mamba2Caches<B>>,
        ssd_path: SsdPath,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
        let [batch_size, sequence_len, _input_dim] = x.dims();
        let [_input_dim, d_model] = self.in_proj.weight.dims();
        let [_d_model, output_dim] = self.out_proj.weight.dims();

        // input projection
        let x = self.in_proj.forward(x);
        assert_eq!([batch_size, sequence_len, d_model], x.dims());

        // layers
        let (x, caches) = self.layers.forward(x, caches, ssd_path);
        assert_eq!([batch_size, sequence_len, d_model], x.dims());

        // output projection
        let x = self.out_proj.forward(x);
        assert_eq!([batch_size, sequence_len, output_dim], x.dims());

        (x, caches)
    }
}
