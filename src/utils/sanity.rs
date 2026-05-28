//! Optional `NaN`/`Inf` guards for debugging numerical issues.
//!
//! Both checks are gated by the compile-time flags [`crate::DENY_NAN`] /
//! [`crate::DENY_INF`] (both `false` by default).  When the flags are off the
//! functions compile down to nothing — there is no runtime cost in release
//! builds — so calls can be sprinkled liberally through the forward passes.

use crate::{DENY_INF, DENY_NAN};
use burn::prelude::*;

/// Panics if `t` contains a `NaN` (when [`crate::DENY_NAN`] is set) or an `Inf`
/// (when [`crate::DENY_INF`] is set).  A no-op when both flags are `false`.
pub fn sanity<const D: usize>(t: &Tensor<D>) {
    let mut has_nan = false;
    let mut has_inf = false;

    if DENY_NAN {
        has_nan = t.clone().contains_nan().into_scalar::<bool>().to_bool();
        if has_nan {
            eprintln!("got a NaN");
        }
    }
    if DENY_INF {
        has_inf = t.clone().is_inf().any().into_scalar::<bool>().to_bool();
        if has_inf {
            eprintln!("got a INF");
        }
    }

    if has_nan || has_inf {
        panic!("sanity check failed");
    }
}

/// Like [`sanity`] but checks only for `NaN` (ignores `Inf`).
pub fn sanity_nan<const D: usize>(t: &Tensor<D>) {
    let mut has_nan = false;

    if DENY_NAN {
        has_nan = t.clone().contains_nan().into_scalar::<bool>().to_bool();
        if has_nan {
            eprintln!("got a NaN");
        }
    }

    if has_nan {
        panic!("sanity check failed");
    }
}
