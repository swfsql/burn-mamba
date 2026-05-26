//! Bidirectional counterparts of the example networks: `in_proj` →
//! `{Mamba2,Mamba3}BidiLayers` → `out_proj`, for non-autoregressive tasks.

use crate::common::model::ModelConfigExt;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn_mamba::prelude::*;
use burn_mamba::schedule::BidiSchedule;

/// Bidirectional Mamba-2 example network and config builders.
pub mod mamba2 {
    use super::*;
    use crate::common::model::mamba2_block_config;
    use burn_mamba::mamba2::bidi::naive::{
        Mamba2BidiLayers, Mamba2BidiLayersConfig, OutputMergeConfig,
    };

    /// Basic bidirectional Mamba network with input/output heads.
    #[derive(Config, Debug)]
    pub struct MyMamba2BidiNetworkConfig {
        /// Width of the input features fed to `in_proj`.
        #[config(default = 1)]
        pub input_size: usize,

        /// Configuration for the bidirectional Mamba-2 layer stack.
        #[config(
            default = "mamba2_bidi_layers_config(1, None, mamba2_block_config(1, 1, 1, 1, 1, 1), OutputMergeConfig::mean(1))"
        )]
        pub layers: Mamba2BidiLayersConfig,

        /// Width of the output features produced by `out_proj`.
        #[config(default = 1)]
        pub output_size: usize,
    }

    /// Build a [`Mamba2BidiLayersConfig`] with optional virtual-layer scheduling.
    pub fn mamba2_bidi_layers_config(
        n_real_layers: usize,
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        mamba_block: Mamba2Config,
        outputs_merge: Vec<OutputMergeConfig>,
    ) -> Mamba2BidiLayersConfig {
        Mamba2BidiLayersConfig::new(n_real_layers, mamba_block, outputs_merge)
            .with_n_virtual_layers(n_virtual_layers)
    }

    /// `in_proj` → bidirectional Mamba-2 layer stack → `out_proj`.
    #[derive(Module, Debug)]
    pub struct MyMamba2BidiNetwork<B: Backend> {
        /// Linear projection from `input_size` to `d_model`.
        pub in_proj: Linear<B>,
        /// The bidirectional Mamba-2 layer stack.
        pub layers: Mamba2BidiLayers<B>,
        /// Linear projection from `d_model` to `output_size`.
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

    impl<B: Backend + Mamba2BackendExt> MyMamba2BidiNetwork<B> {
        /// `in_proj` → bidirectional layers → `out_proj` over a full sequence
        /// (`[batch, sequence, input_size]` → `[batch, sequence, output_size]`).
        pub fn forward(
            &self,
            x: Tensor<B, 3>,
            caches: Option<Mamba2Caches<B>>,
            ssd_path: Mamba2SsdPath,
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
}

/// Bidirectional Mamba-3 example network and config builders.
pub mod mamba3 {
    use super::*;
    use crate::common::model::mamba3_block_config;
    use burn_mamba::mamba3::bidi::naive::{
        Mamba3BidiLayers, Mamba3BidiLayersConfig, OutputMergeConfig,
    };

    /// Basic bidirectional Mamba network with input/output heads.
    #[derive(Config, Debug)]
    pub struct MyMamba3BidiNetworkConfig {
        /// Width of the input features fed to `in_proj`.
        #[config(default = 1)]
        pub input_size: usize,

        /// Configuration for the bidirectional Mamba-3 layer stack.
        #[config(
            default = "mamba3_bidi_layers_config(1, None, mamba3_block_config(1, 2, 1, 1, 1, 1.0, 1), OutputMergeConfig::mean(1))"
        )]
        pub layers: Mamba3BidiLayersConfig,

        /// Width of the output features produced by `out_proj`.
        #[config(default = 1)]
        pub output_size: usize,
    }

    /// Build a [`Mamba3BidiLayersConfig`] with optional virtual-layer scheduling.
    pub fn mamba3_bidi_layers_config(
        n_real_layers: usize,
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        mamba_block: Mamba3Config,
        outputs_merge: Vec<OutputMergeConfig>,
    ) -> Mamba3BidiLayersConfig {
        Mamba3BidiLayersConfig::new(n_real_layers, mamba_block, outputs_merge)
            .with_n_virtual_layers(n_virtual_layers)
    }

    /// `in_proj` → bidirectional Mamba-3 layer stack → `out_proj`.
    #[derive(Module, Debug)]
    pub struct MyMamba3BidiNetwork<B: Backend> {
        /// Linear projection from `input_size` to `d_model`.
        pub in_proj: Linear<B>,
        /// The bidirectional Mamba-3 layer stack.
        pub layers: Mamba3BidiLayers<B>,
        /// Linear projection from `d_model` to `output_size`.
        pub out_proj: Linear<B>,
    }

    impl<B: Backend> ModelConfigExt<B> for MyMamba3BidiNetworkConfig {
        type Model = MyMamba3BidiNetwork<B>;

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
            MyMamba3BidiNetwork {
                in_proj,
                layers,
                out_proj,
            }
        }
    }

    impl<B: Backend + Mamba3BackendExt> MyMamba3BidiNetwork<B> {
        /// `in_proj` → bidirectional layers → `out_proj` over a full sequence
        /// (`[batch, sequence, input_size]` → `[batch, sequence, output_size]`).
        pub fn forward(
            &self,
            x: Tensor<B, 3>,
            caches: Option<Mamba3Caches<B>>,
            ssd_path: Mamba3SsdPath,
        ) -> (Tensor<B, 3>, Mamba3Caches<B>) {
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
}
