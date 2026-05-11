use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::Element;

/// Applies the log sigmoid function element-wise.
///
/// `log_sigmoid(x) = log(1 / (1 + exp(-x)))`
pub fn log_sigmoid<const D: usize, B: Backend>(x: Tensor<B, D>) -> Tensor<B, D> {
    match <B::FloatElem as Element>::dtype() {
        DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
            // log_sigmoid(x) = log(1 / (1 + exp(-x)))
            (x.neg().exp() + 1.).recip().log()
        }
        DType::F16 => {
            // log_sigmoid(x) = -softplus(-x)
            -crate::utils::softplus::softplus(x.neg())
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
