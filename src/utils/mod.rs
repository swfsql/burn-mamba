use ElementConversion;
use burn::prelude::*;
use burn::tensor::{DType, Element};

pub mod log_sigmoid;
pub mod loss;
pub mod rms_norm;
pub mod rms_norm_gated;
pub mod silu;
pub mod softplus;

pub fn stable_max<B: Backend>() -> B::FloatElem {
    match <B::FloatElem as Element>::dtype() {
        DType::F64 => f64::MAX.elem(),
        DType::F32 | DType::Flex32 => f32::MAX.elem(),
        DType::F16 => burn::tensor::f16::MAX.elem(),
        DType::BF16 => burn::tensor::bf16::MAX.elem(),
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

pub fn div_eps_f32<B: Backend>() -> f32 {
    match <B::FloatElem as Element>::dtype() {
        // 4.0693917e-16
        DType::F64 => {
            let raw_exp = -(-f64::MIN_EXP as f32 * 2.3f32).powf(0.35f32);
            let eps_exp = (f64::EPSILON as f32).log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 8.1584695e-8
        DType::F32 | DType::Flex32 => {
            let raw_exp = -(-f32::MIN_EXP as f32 * 2.3f32).powf(0.35f32);
            let eps_exp = f32::EPSILON.log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 7.1209995e-4
        DType::F16 => {
            let raw_exp = -(-burn::tensor::f16::MIN_EXP.to_f32() * 2.3f32).powf(0.35f32);
            let eps_exp = burn::tensor::f16::EPSILON.to_f32().log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        // 2.0885676e-5
        DType::BF16 => {
            let raw_exp = -(-burn::tensor::bf16::MIN_EXP.to_f32() * 2.3f32).powf(0.35f32);
            let eps_exp = burn::tensor::bf16::EPSILON.to_f32().log10();
            let avg = (raw_exp + eps_exp) / 2f32;
            10f32.powf(avg)
        }
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8
        | DType::Bool => {
            unreachable!()
        }
        DType::QFloat(_) => {
            unimplemented!()
        }
    }
}

pub fn div_eps<B: Backend>() -> B::FloatElem {
    div_eps_f32::<B>().elem()
}

pub fn contains_nan_or_inf<B: Backend, const D: usize>(_t: &Tensor<B, D>) -> bool {
    // let is_inf = t.clone().is_inf().any().into_scalar().to_bool();
    // let is_nan = t.clone().is_nan().any().into_scalar().to_bool();
    // if is_nan {
    //     println!("got a NAN");
    // }
    // if is_inf {
    //     println!("got an INF");
    // }
    // is_nan || is_inf
    false
}
