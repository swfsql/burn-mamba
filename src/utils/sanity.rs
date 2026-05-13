use crate::{DENY_INF, DENY_NAN};
use burn::prelude::*;

pub fn sanity<B: Backend, const D: usize>(t: &Tensor<B, D>) {
    let mut has_nan = false;
    let mut has_inf = false;

    if DENY_NAN {
        has_nan = t.clone().contains_nan().into_scalar().to_bool();
        if has_nan {
            eprintln!("got a NaN");
        }
    }
    if DENY_INF {
        has_inf = t.clone().is_inf().any().into_scalar().to_bool();
        if has_inf {
            eprintln!("got a INF");
        }
    }

    if has_nan || has_inf {
        panic!("sanity check failed");
    }
}

pub fn sanity_nan<B: Backend, const D: usize>(t: &Tensor<B, D>) {
    let mut has_nan = false;

    if DENY_NAN {
        has_nan = t.clone().contains_nan().into_scalar().to_bool();
        if has_nan {
            eprintln!("got a NaN");
        }
    }

    if has_nan {
        panic!("sanity check failed");
    }
}
