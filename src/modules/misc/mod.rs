/// Group→head expansion of B/C (GQA-style sharing).
pub mod gqa;
pub mod rope;
/// Optional `NaN`/`Inf` guards gated by [`crate::DENY_NAN`] / [`crate::DENY_INF`].
pub mod sanity;
/// Stable segment-sum → 1-semiseparable mask (log-space prefix-sum differences).
pub mod segsum;
/// Typed-array variant of `split_with_sizes` for clean destructuring.
pub mod split;
