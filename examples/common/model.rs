use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};
use burn_mamba::prelude::*;
use burn_mamba::schedule::Schedule;

pub trait ModelConfigExt<B: Backend>: Config {
    type Model: Module<B>;
    fn init(&self, device: &B::Device) -> Self::Model;
}

/// Basic Mamba network containing input and output heads, with Mamba2Layers in between.
#[derive(Config, Debug)]
pub struct MyMamba2NetworkConfig {
    #[config(default = 1)]
    pub input_size: usize,

    #[config(default = "mamba2_layers_config(1, None, mamba2_block_config(1, 1, 1, 1, 1))")]
    pub layers: Mamba2LayersConfig,

    #[config(default = 1)]
    pub output_size: usize,
}

pub fn mamba2_block_config(
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    n_heads: usize,
    expand: usize,
) -> Mamba2Config {
    let d_inner = expand * d_model;
    let headdim = d_inner / n_heads;
    assert_eq!(d_inner, headdim * n_heads);
    Mamba2Config::new(d_model)
        .with_d_state(d_state)
        .with_d_conv(d_conv)
        .with_expand(expand)
        .with_headdim(headdim)
        .with_ngroups(1)
        .with_has_proj_bias(true)
}

pub fn mamba2_layers_config(
    n_real_layers: usize,
    n_virtual_layers: Option<(usize, Schedule)>,
    mamba_block: Mamba2Config,
) -> Mamba2LayersConfig {
    Mamba2LayersConfig::new(n_real_layers, mamba_block).with_n_virtual_layers(n_virtual_layers)
}

#[derive(Module, Debug)]
pub struct MyMamba2Network<B: Backend> {
    pub in_proj: Linear<B>,
    pub layers: Mamba2Layers<B>,
    pub out_proj: Linear<B>,
}

impl<B: Backend> ModelConfigExt<B> for MyMamba2NetworkConfig {
    type Model = MyMamba2Network<B>;

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
        MyMamba2Network {
            in_proj,
            layers,
            out_proj,
        }
    }
}

impl<B: Backend> MyMamba2Network<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        caches: Option<Mamba2Caches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
        let [batch_size, sequence_len, _input_dim] = x.dims();
        let [_input_dim, d_model] = self.in_proj.weight.dims();
        let [_d_model, output_dim] = self.out_proj.weight.dims();

        // input projection
        let x = self.in_proj.forward(x);
        assert_eq!([batch_size, sequence_len, d_model], x.dims());

        // layers
        let (x, caches) = self.layers.forward(x, caches, chunk_size);

        // output projection
        let x = self.out_proj.forward(x);
        assert_eq!([batch_size, sequence_len, output_dim], x.dims());

        (x, caches)
    }
}
