//! Shared test fixtures for the SSD-path agreement tests in Mamba-2 and Mamba-3.
//!
//! Each `*SsdPath` exposes three implementations (`Minimal`, `Serial`,
//! `SerialRecalculated`) which must agree on both forward outputs and input
//! gradients. The tests follow the same pattern: pick a baseline, run two
//! alternatives, and compare per-field. The shape of the per-input gradient
//! struct (`PathRun`) differs between Mamba-2 and Mamba-3 (different field
//! sets), so we only share the small generic primitives here.

#![cfg(test)]

use burn::prelude::*;

/// Element-wise max absolute difference between two same-shape tensors,
/// returned as `f32` (already pulled to host via `into_scalar()`).
pub fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    use burn::tensor::ElementConversion;
    (a - b).abs().max().into_scalar().elem()
}

/// Compare two `PathRun`-style structs field-by-field against a baseline,
/// asserting every named field is within `tol` of the baseline.
///
/// Used by the SSD-path agreement tests in both Mamba-2 and Mamba-3. The
/// field set differs between the two — pass exactly the fields present on
/// your `PathRun`.
///
/// # Example
/// ```ignore
/// check_grads_match_two_paths!(
///     baseline: r_min,
///     alt1: ("Serial", r_ser),
///     alt2: ("SerialRecalculated", r_rec),
///     tol: 1e-3,
///     fields: [d_x => "x", d_dt => "dt", /* ... */],
/// );
/// ```
#[macro_export]
macro_rules! check_grads_match_two_paths {
    (
        baseline: $r_min:ident,
        alt1: ($alt1_label:literal, $r_alt1:ident),
        alt2: ($alt2_label:literal, $r_alt2:ident),
        tol: $tol:expr,
        fields: [ $($field:ident => $name:literal),* $(,)? ] $(,)?
    ) => {{
        let tol: f32 = $tol;
        let mut failures: Vec<String> = Vec::new();
        $(
            let d1 = $crate::utils::test_helpers::max_abs_diff(
                $r_min.$field.clone(), $r_alt1.$field.clone(),
            );
            let d2 = $crate::utils::test_helpers::max_abs_diff(
                $r_min.$field.clone(), $r_alt2.$field.clone(),
            );
            eprintln!(
                "grad {:>14} | min↔{} = {:>10.6} | min↔{} = {:>10.6}",
                $name, $alt1_label, d1, $alt2_label, d2,
            );
            if d1 >= tol {
                failures.push(format!(
                    "Minimal vs {}: grad of {} max abs diff = {:.6} (tol {})",
                    $alt1_label, $name, d1, tol,
                ));
            }
            if d2 >= tol {
                failures.push(format!(
                    "Minimal vs {}: grad of {} max abs diff = {:.6} (tol {})",
                    $alt2_label, $name, d2, tol,
                ));
            }
        )*
        assert!(
            failures.is_empty(),
            "gradient mismatches:\n  {}",
            failures.join("\n  "),
        );
    }};
}
