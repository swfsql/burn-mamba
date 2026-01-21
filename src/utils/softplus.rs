use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::Element;

/// Applies the SoftPlus function element-wise.
///
/// The SoftPlus function is a smooth approximation of the ReLU function.
pub fn softplus<const D: usize, B: Backend>(x: Tensor<B, D>) -> Tensor<B, D> {
    match <B::FloatElem as Element>::dtype() {
        DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
            // softplus = log(e^x + 1)
            x.exp().log1p()
        }
        DType::F16 => {
            // (x.exp() + 1.).log()

            // max(a,b) = (a + b + |a-b|)/2
            // softplus = max(x, 0) + log(e^-|x| + 1)
            //          = (x + |x|) / 2 + log(e^-|x| + 1)
            let xabs = x.clone().abs();
            (x + xabs.clone()) / 2. + xabs.neg().exp().log1p()
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
