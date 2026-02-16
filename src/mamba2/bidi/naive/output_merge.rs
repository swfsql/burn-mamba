use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Used when a Module is expected.
#[derive(Module, Clone, Debug)]
pub struct NoOp;

#[derive(Module, Debug)]
pub enum OutputMerge<B: Backend> {
    Mean(NoOp),
    /// # Shape
    /// - [2 * d_model, d_model]
    CatLinear(Linear<B>),
}

/// # Shapes
///   - Input straight [batch_size, sequence_len, d_model]
///   - Input reverse [batch_size, sequence_len, d_model]
///   - Output [batch_size, sequence_len, d_model]
impl<B: Backend> OutputMerge<B> {
    pub fn forward(&self, straight: Tensor<B, 3>, reverse: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, sequence_len, d_model] = straight.dims();
        assert_eq!(straight.dims(), reverse.dims());
        match self {
            OutputMerge::Mean(_noop) => (straight + reverse) * 0.5,
            OutputMerge::CatLinear(proj) => {
                let cat = Tensor::cat([straight, reverse].to_vec(), 2);
                assert_eq!([batch_size, sequence_len, 2 * d_model], cat.dims());
                let merged = proj.forward(cat);
                assert_eq!([batch_size, sequence_len, d_model], merged.dims());
                merged
            }
        }
    }
}

#[derive(Config, Debug)]
pub enum OutputMergeConfig {
    Mean,
    CatLinear,
}

impl OutputMergeConfig {
    pub fn mean(n_real_layers: usize) -> Vec<Self> {
        vec![Self::Mean; n_real_layers / 2]
    }
    pub fn cat_linear(n_real_layers: usize) -> Vec<Self> {
        vec![Self::CatLinear; n_real_layers / 2]
    }

    pub fn init<B: Backend>(&self, d_model: usize, device: &B::Device) -> OutputMerge<B> {
        match self {
            OutputMergeConfig::Mean => OutputMerge::Mean(NoOp),
            OutputMergeConfig::CatLinear => {
                let cat_linear = LinearConfig::new(d_model * 2, d_model).init(device);
                OutputMerge::CatLinear(cat_linear)
            }
        }
    }
}
