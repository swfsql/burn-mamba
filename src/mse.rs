use crate::utils::div_eps;
use burn::module::Module;
use burn::nn::loss::Reduction;
use burn::tensor::{DType, Element, Tensor, backend::Backend, f16};

/// Calculate the mean squared error loss from the input logits and the targets.
#[derive(Module, Clone, Debug)]
pub struct MseLoss;

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MseLoss {
    /// Create the criterion.
    pub fn new() -> Self {
        Self
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: [batch_size, num_targets]
    /// - targets: [batch_size, num_targets]
    pub fn forward<const D: usize, B: Backend>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        match <B::FloatElem as Element>::dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let tensor = self.forward_no_reduction(logits, targets);
                match reduction {
                    Reduction::Mean | Reduction::Auto => tensor.mean(),
                    Reduction::Sum => tensor.sum(),
                }
            }
            DType::F16 => {
                use burn::tensor::ElementConversion;
                let div_eps: f16 = f16::from_elem(div_eps::<B>()) * f16::from_f32(2.);
                // avoid calculating sub² directly (due to overflow e.g. on 256 * 256)
                let sub = logits.sub(targets);
                let max = sub.clone().no_grad().detach().abs().max();
                let sub_ = sub.clone() / (max.clone().expand(sub.shape()) + div_eps); // sub_.abs() <= 1
                let partial = sub * sub_; // sub² = partial * max
                let reduced_partial = match reduction {
                    Reduction::Mean | Reduction::Auto => partial.mean(),
                    Reduction::Sum => partial.sum(),
                };
                reduced_partial * max
            }
            DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8 => {
                unreachable!()
            }
            DType::Bool => {
                unreachable!()
            }
            DType::QFloat(_) => {
                unimplemented!()
            }
        }
    }

    /// Compute the criterion on the input tensor without reducing.
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        logits.sub(targets).square()
    }
}
