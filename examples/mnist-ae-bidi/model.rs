use crate::common::mnist::dataset::{HEIGHT, WIDTH};
pub use crate::common::model::bidi::{
    MyMamba2BidiNetwork, MyMamba2BidiNetworkConfig, mamba2_bidi_layers_config,
};
pub use crate::common::model::{
    ModelConfigExt, class_token::ClassToken, mamba2_block_config, mamba2_layers_config,
};
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::mamba2::bidi::naive::OutputMergeConfig;
use burn_mamba::prelude::*;

#[derive(Config, Debug)]
pub struct AutoEncoderNetworkConfig {
    pub encoder: MyMamba2BidiNetworkConfig,
    pub decoder_scratchpadding: Option<usize>,
    pub decoder: MyMamba2BidiNetworkConfig,
}

#[derive(Module, Debug)]
pub struct AutoEncoderNetwork<B: Backend> {
    pub encoder: MyMamba2BidiNetwork<B>,
    pub decoder_scratchpadding: Option<usize>,
    // TODO: try RoPE (no extra params needed)
    /// # Shape
    /// [HEIGHT * latent_dim + 2 * decoder_scratchpadding]
    pub decoder_bias: Param<Tensor<B, 1>>,
    pub decoder: MyMamba2BidiNetwork<B>,
}

pub fn model_config(latent_size: usize, decoder_scratchpadding: Option<usize>) -> AutoEncoderNetworkConfig {
    let mamba_block = mamba_block();
    let layers = [16, 36];
    let encoder = MyMamba2BidiNetworkConfig::new()
        // the input is a sequence of a single-dimensioned values
        // the input shape is [batch_size, sequence_len = HEIGHT, input_size = WIDTH]
        .with_input_size(WIDTH)
        // aggregate info to the middle token
        // this is useful for bidi layers
        // the input shape becomes [batch_size, sequence_len = HEIGHT + 1, input_size = WIDTH]
        .with_class_tokens(vec![ClassToken::Middle])
        .with_layers(
            mamba2_bidi_layers_config(
                layers[0],
                None,
                mamba_block.clone(),
                OutputMergeConfig::cat_linear(layers[0]),
            )
            .with_n_virtual_layers(None)
            .with_ignore_first_residual(true) // first input (before flip) is non-bidi
            .with_ignore_last_residual(true), // last output comes only from the state
        )
        // the output are the latent values (last projection from last layer state)
        // the output shape by default would be [batch_size, sequence_len = HEIGHT + 1, latent_size],
        // but a custom implementation narrows it to a single timestep [batch_size, sequence_len = 1, latent_size]
        .with_output_size(latent_size);

    let decoder = MyMamba2BidiNetworkConfig::new()
        // the input is a repeated sequence of the latent
        // the input shape by default would be [batch_size, sequence_len = 1, latent_size]
        // but a custom implementation expands it to [batch_size, sequence_len = HEIGHT, latent_size]
        .with_input_size(latent_size)
        .with_layers(
            mamba2_bidi_layers_config(
                layers[1],
                None,
                mamba_block.clone(),
                OutputMergeConfig::cat_linear(layers[1]),
            )
            .with_n_virtual_layers(None)
            .with_ignore_first_residual(false) // first input (before flip) is bidi
            .with_ignore_last_residual(true), // last output comes only from the state
        )
        // the output is a sequence of a single-dimensioned values (reconstruction)
        // the output shape is [batch_size, sequence_len = HEIGHT, output_size = WIDTH]
        .with_output_size(WIDTH);
    AutoEncoderNetworkConfig::new(encoder, decoder).with_decoder_scratchpadding(decoder_scratchpadding)
}

pub fn mamba_block() -> Mamba2Config {
    mamba2_block_config(
        //
        64,  // d_model
        128, // d_state
        1,   // d_conv
        8,   // n_heads
        4,   // expand
    )
}

impl<B: Backend> ModelConfigExt<B> for AutoEncoderNetworkConfig {
    type Model = AutoEncoderNetwork<B>;

    /// Returns the initialized model.
    fn init(&self, device: &B::Device) -> Self::Model {
        let latent_dim = self.encoder.output_size;
        let dec_padding = self.decoder_scratchpadding.unwrap_or_default();
        let decoder_bias = Tensor::random(
            [(HEIGHT + 2 * dec_padding) * latent_dim],
            burn::tensor::Distribution::Uniform(-0.1, 0.1),
            device,
        );
        AutoEncoderNetwork {
            encoder: self.encoder.init(device),
            decoder_scratchpadding: self.decoder_scratchpadding.clone(),
            decoder_bias: Param::from_tensor(decoder_bias),
            decoder: self.decoder.init(device),
        }
    }
}

impl<B: Backend> AutoEncoderNetwork<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_caches: Option<Mamba2Caches<B>>,
        decoder_caches: Option<Mamba2Caches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>, Mamba2Caches<B>) {
        let [batch_size, sequence_len, _input_dim] = x.dims();
        let [_encoder_d_model, latent_dim] = self.encoder.out_proj.weight.dims();
        let [_decoder_d_model, output_dim] = self.decoder.out_proj.weight.dims();
        assert_eq!([sequence_len, output_dim], [HEIGHT, WIDTH]);

        // encoder
        let extra_tokens = 1;
        let (z, encoder_caches) = self.encoder.forward(x, encoder_caches, chunk_size);
        assert_eq!([batch_size, sequence_len + extra_tokens, latent_dim], z.dims());

        // information aggregation
        // let z = z.mean_dim(1); // single latent value per batch
        let z = z.narrow(1, sequence_len / 2, 1);
        assert_eq!([batch_size, 1, latent_dim], z.dims());

        // expanded for decoder
        // TODO: move this op to inside the decoder
        let dec_padding = self.decoder_scratchpadding.unwrap_or_default();
        let z = if dec_padding > 0 {
            (z).expand([batch_size, sequence_len + 2 * dec_padding, latent_dim])
        } else {
            (z).expand([batch_size, sequence_len, latent_dim])
        }; 
        // decoder bias
        let bias = self.decoder_bias.val();
        assert_eq!([(sequence_len + 2 * dec_padding) * latent_dim], bias.dims());
        let bias = bias.reshape([sequence_len + 2 * dec_padding, latent_dim]);
        let bias = bias
            .unsqueeze_dim::<3>(0)
            .expand([batch_size, sequence_len + 2 * dec_padding, latent_dim]);
        let z = z + bias;

        // decoder
        let (x, decoder_caches) = self.decoder.forward(z.clone(), decoder_caches, chunk_size);
        assert_eq!([batch_size, sequence_len + 2 * dec_padding, output_dim], x.dims());
        // narrow away from dec_padding, taking the centered positions
        let x = x.narrow(1, dec_padding, sequence_len);
        assert_eq!([batch_size, sequence_len, output_dim], x.dims());
        
        (x, encoder_caches, decoder_caches)
    }
}
