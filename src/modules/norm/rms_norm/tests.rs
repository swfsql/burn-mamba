use super::*;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// RMSNorm's backward must stay finite when a normalised slice collapses to zero
/// norm (`mean(x²) = 0`) — a dead token/channel, or a subnormal flushed to zero
/// on CUDA. The forward guards the *division* (`rms + div_eps`), but the `sqrt`
/// node's own backward is `1/(2·√(mean x²))`, singular at zero unless the
/// epsilon sits *inside* the root. Regression guard for the combined-penalty
/// grokking NaN, localised to this op (the SSD backward's incoming `d_y`).
#[test]
fn rms_norm_gradient_finite_on_collapsed_slice() {
    let device: Device = Default::default();
    let (batch, seq, d_model) = (2, 3, 8);
    let norm = RmsNormConfig::new(d_model).init(&device.clone().autodiff());

    // A normal batch with one token collapsed to exactly zero norm.
    let normal =
        Tensor::<3>::random([batch, seq - 1, d_model], Distribution::Normal(0.0, 1.0), &device);
    let collapsed = Tensor::<3>::zeros([batch, 1, d_model], &device);
    let base = Tensor::cat(vec![collapsed, normal], 1);

    let x = Param::from_tensor(Tensor::from_inner(base));
    let grads = norm.forward(x.val()).sum().backward();
    let g = x.val().grad(&grads).expect("grad exists");
    let gvec = g.into_data().to_vec::<f32>().unwrap();
    assert!(
        gvec.iter().all(|v| v.is_finite()),
        "RMSNorm gradient must stay finite for a zero-norm slice"
    );
}
