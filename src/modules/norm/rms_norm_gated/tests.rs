use super::*;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// Gated RMSNorm's backward must stay finite when the normalised input (the SSD
/// output `y`) collapses to zero norm on a slice — the exact NaN localised in
/// the combined-penalty grokking run (`d_y` arriving NaN at the SSD backward).
/// Same root cause as the ungated [`RmsNorm`](crate::modules::norm::rms_norm):
/// `div_eps` guards the forward division but not the `sqrt` node's `1/(2√·)`
/// backward.
#[test]
fn rms_norm_gated_gradient_finite_on_collapsed_slice() {
    let device: Device = Default::default();
    let (batch, seq, d_model) = (2, 3, 8);
    let norm = RmsNormGatedConfig::new(d_model).init(&device.clone().autodiff());

    // The normalised input (SSD output) has one token collapsed to zero norm;
    // the gate `z` stays healthy.
    let normal =
        Tensor::<3>::random([batch, seq - 1, d_model], Distribution::Normal(0.0, 1.0), &device);
    let collapsed = Tensor::<3>::zeros([batch, 1, d_model], &device);
    let base = Tensor::cat(vec![collapsed, normal], 1);
    let z = Tensor::from_inner(Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));

    let x = Param::from_tensor(Tensor::from_inner(base));
    let grads = norm.forward(x.val(), z).sum().backward();
    let g = x.val().grad(&grads).expect("grad exists");
    let gvec = g.into_data().to_vec::<f32>().unwrap();
    assert!(
        gvec.iter().all(|v| v.is_finite()),
        "gated RMSNorm gradient must stay finite for a zero-norm slice"
    );
}
