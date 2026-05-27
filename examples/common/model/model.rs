//! The example networks: a linear `in_proj` → a `{Mamba2,Mamba3}Layers` stack →
//! a linear `out_proj`, one variant per family (a regression/feature head rather
//! than a token LM).  Also the small `*_block_config` / `*_layers_config`
//! builders the concrete examples use to assemble their configs.

use crate::common::model::ModelConfigExt;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*
use burn_mamba::prelude::*;
use burn_mamba::schedule::Schedule;
use burn::backend::Backend;

/// Mamba-2 example network and config builders.
pub mod mamba2 {
    use super::*;

    /// Basic Mamba network containing input and output heads, with Mamba2Layers in between.
    #[derive(Config, Debug)]
    pub struct MyMamba2NetworkConfig {
        /// Width of the input features fed to `in_proj`.
        #[config(default = 1)]
        pub input_size: usize,

        /// Configuration for the Mamba-2 layer stack.
        #[config(default = "mamba2_layers_config(1, None, mamba2_block_config(1, 1, 1, 1, 1, 1))")]
        pub layers: Mamba2LayersConfig,

        /// Width of the output features produced by `out_proj`.
        #[config(default = 1)]
        pub output_size: usize,
    }

    /// Build a [`Mamba2Config`] from the example's flat hyperparameters,
    /// deriving `per_head_dim = expand · d_model / nheads`.
    pub fn mamba2_block_config(
        d_model: usize,
        state_rank: usize,
        conv_kernel: usize,
        nheads: usize,
        ngroups: usize,
        expand: usize,
    ) -> Mamba2Config {
        let d_inner = expand * d_model;
        let per_head_dim = d_inner / nheads;
        assert_eq!(d_inner, per_head_dim * nheads);
        Mamba2Config::new(d_model)
            .with_state_rank(state_rank)
            .with_conv_kernel(conv_kernel)
            .with_expand(expand)
            .with_per_head_dim(per_head_dim)
            .with_ngroups(ngroups)
            .with_has_proj_bias(true)
    }

    /// Build a [`Mamba2LayersConfig`] with optional virtual-layer scheduling.
    pub fn mamba2_layers_config(
        n_real_layers: usize,
        n_virtual_layers: Option<(usize, Schedule)>,
        mamba_block: Mamba2Config,
    ) -> Mamba2LayersConfig {
        Mamba2LayersConfig::new(n_real_layers, mamba_block).with_n_virtual_layers(n_virtual_layers)
    }

    /// `in_proj` → Mamba-2 layer stack → `out_proj`.
    #[derive(Module, Debug)]
    pub struct MyMamba2Network {
        /// Linear projection from `input_size` to `d_model`.
        pub in_proj: Linear,
        /// The Mamba-2 layer stack.
        pub layers: Mamba2Layers,
        /// Linear projection from `d_model` to `output_size`.
        pub out_proj: Linear,
    }

    impl ModelConfigExt for MyMamba2NetworkConfig {
        type Model = MyMamba2Network;

        /// Returns the initialized model.
        fn init(&self, device: &Device) -> Self::Model {
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

    impl MyMamba2Network {
        /// `in_proj` → layers → `out_proj` over a full sequence.
        ///
        /// # Shapes
        ///   - Input `[batch, sequence, input_size]`
        ///   - Output `[batch, sequence, output_size]`
        pub fn forward(
            &self,
            x: Tensor<3>,
            caches: Option<Mamba2Caches>,
            ssd_path: Mamba2SsdPath,
        ) -> (Tensor<3>, Mamba2Caches) {
            let [batch_size, sequence_len, _input_dim] = x.dims();
            let [_input_dim, d_model] = self.in_proj.weight.dims();
            let [_d_model, output_dim] = self.out_proj.weight.dims();

            // input projection
            let x = self.in_proj.forward(x);
            assert_eq!([batch_size, sequence_len, d_model], x.dims());

            // layers
            let (x, caches) = self.layers.forward(x, caches, ssd_path);

            // output projection
            let x = self.out_proj.forward(x);
            assert_eq!([batch_size, sequence_len, output_dim], x.dims());

            (x, caches)
        }
    }
}

/// Mamba-3 example network and config builders.
pub mod mamba3 {
    use super::*;

    /// Basic Mamba network containing input and output heads, with Mamba3Layers in between.
    #[derive(Config, Debug)]
    pub struct MyMamba3NetworkConfig {
        /// Width of the input features fed to `in_proj`.
        #[config(default = 1)]
        pub input_size: usize,

        /// Configuration for the Mamba-3 layer stack.
        #[config(
            default = "mamba3_layers_config(1, None, mamba3_block_config(1, 2, 1, 1, 1, 1.0, 1))"
        )]
        pub layers: Mamba3LayersConfig,

        /// Width of the output features produced by `out_proj`.
        #[config(default = 1)]
        pub output_size: usize,
    }

    /// Build a [`Mamba3Config`] from the example's flat hyperparameters,
    /// deriving `per_head_dim = expand · d_model / nheads`.
    pub fn mamba3_block_config(
        d_model: usize,
        state_rank: usize,
        nheads: usize,
        ngroups: usize,
        mimo_rank: usize,
        rope_fraction: f64,
        expand: usize,
    ) -> Mamba3Config {
        let d_inner = expand * d_model;
        let per_head_dim = d_inner / nheads;
        assert_eq!(d_inner, per_head_dim * nheads);
        Mamba3Config::new(d_model)
            .with_state_rank(state_rank)
            .with_expand(expand)
            .with_per_head_dim(per_head_dim)
            .with_ngroups(ngroups)
            .with_mimo_rank(mimo_rank)
            .with_rope_fraction(rope_fraction)
            .with_has_proj_bias(true)
            .with_has_outproj_norm(true)
    }

    /// Build a [`Mamba3LayersConfig`] with optional virtual-layer scheduling.
    pub fn mamba3_layers_config(
        n_real_layers: usize,
        n_virtual_layers: Option<(usize, Schedule)>,
        mamba_block: Mamba3Config,
    ) -> Mamba3LayersConfig {
        Mamba3LayersConfig::new(n_real_layers, mamba_block).with_n_virtual_layers(n_virtual_layers)
    }

    /// `in_proj` → Mamba-3 layer stack → `out_proj`.
    #[derive(Module, Debug)]
    pub struct MyMamba3Network {
        /// Linear projection from `input_size` to `d_model`.
        pub in_proj: Linear,
        /// The Mamba-3 layer stack.
        pub layers: Mamba3Layers,
        /// Linear projection from `d_model` to `output_size`.
        pub out_proj: Linear,
    }

    impl ModelConfigExt for MyMamba3NetworkConfig {
        type Model = MyMamba3Network;

        /// Returns the initialized model.
        fn init(&self, device: &Device) -> Self::Model {
            let d_model = self.layers.mamba_block.d_model;
            let in_proj = LinearConfig::new(self.input_size, d_model)
                .with_bias(true)
                .init(device);
            let layers = self.layers.init(device);
            let out_proj = LinearConfig::new(d_model, self.output_size)
                .with_bias(true)
                .init(device);
            MyMamba3Network {
                in_proj,
                layers,
                out_proj,
            }
        }
    }

    impl MyMamba3Network {
        /// `in_proj` → layers → `out_proj` over a full sequence.
        ///
        /// # Shapes
        ///   - Input `[batch, sequence, input_size]`
        ///   - Output `[batch, sequence, output_size]`
        pub fn forward(
            &self,
            x: Tensor<3>,
            caches: Option<Mamba3Caches>,
            ssd_path: Mamba3SsdPath,
        ) -> (Tensor<3>, Mamba3Caches) {
            let [batch_size, sequence_len, _input_dim] = x.dims();
            let [_input_dim, d_model] = self.in_proj.weight.dims();
            let [_d_model, output_dim] = self.out_proj.weight.dims();

            // input projection
            let x = self.in_proj.forward(x);
            assert_eq!([batch_size, sequence_len, d_model], x.dims());

            // layers
            let (x, caches) = self.layers.forward(x, caches, ssd_path);

            // output projection
            let x = self.out_proj.forward(x);
            assert_eq!([batch_size, sequence_len, output_dim], x.dims());

            (x, caches)
        }
    }
}
