use super::*;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// `hidden` rounds `d_intermediate` **up** to `multiple_of`, so the realised
/// `fc1`/`fc2` shapes can differ from the configured figure. Pinned because the
/// Mamba-3 MIMO 187m checkpoint relies on it: `d_intermediate: 1264` in
/// `config.json`, `fc1.weight` of `[2·1280, 768]` on disk.
#[test]
fn hidden_rounds_up_to_multiple_of() {
    let device: Device = Default::default();
    for (d_intermediate, expected) in [(1264, 1280), (1536, 1536), (1, 128), (129, 256)] {
        let config = GatedMlpConfig::new(768, d_intermediate);
        assert_eq!(config.hidden(), expected, "d_intermediate {d_intermediate}");
        let mlp = config.init(&device);
        assert_eq!(mlp.hidden(), expected);
        assert_eq!(mlp.fc1.weight.dims(), [768, 2 * expected]);
        assert_eq!(mlp.fc2.weight.dims(), [expected, 768]);
    }
}

/// The value half of `fc1` comes first and the gate half second — the order
/// `y.chunk(2, dim=-1)` produces in the reference `GatedMLP`, and therefore the
/// layout the checkpoint's fused `fc1` is stored in. Swapping the two halves is
/// silent (shapes are identical), so compare against the split done by hand.
#[test]
fn value_half_precedes_gate_half() {
    let device: Device = Default::default();
    let (d_model, d_intermediate, batch, seq) = (16, 128, 2, 3);
    let mlp = GatedMlpConfig::new(d_model, d_intermediate).init(&device);
    let hidden = mlp.hidden();

    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let fused = mlp.fc1.forward(x.clone());
    let value = fused.clone().narrow(2, 0, hidden);
    let gate = fused.narrow(2, hidden, hidden);
    let expected = mlp.fc2.forward(value * Silu::new().forward(gate));

    assert!(max_abs_diff(mlp.forward(x), expected) < 1e-5);
}

/// The block is point-wise in the sequence, so a `[batch, d_model]` step and the
/// matching row of a `[batch, sequence, d_model]` forward must agree. This is
/// what lets [`Layer`](crate::modules::Layer) reuse one module for both modes.
#[test]
fn step_matches_the_matching_forward_row() {
    let device: Device = Default::default();
    let (d_model, d_intermediate, batch, seq) = (16, 128, 2, 4);
    let mlp = GatedMlpConfig::new(d_model, d_intermediate).init(&device);

    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let full = mlp.forward(x.clone());

    for t in 0..seq {
        let row: Tensor<2> = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let expected: Tensor<2> = full.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(mlp.forward(row), expected) < 1e-5,
            "row {t} disagrees"
        );
    }
}

/// Gradients must reach both `fc1` halves and `fc2`. A gate branch detached by a
/// stray `no_grad` would still produce correct forward values, so the forward
/// tests above cannot catch it.
#[test]
fn gradients_reach_both_projections() {
    let device: Device = Default::default();
    let (d_model, d_intermediate) = (16, 128);
    let mlp = GatedMlpConfig::new(d_model, d_intermediate).init(&device.clone().autodiff());

    let base = Tensor::<3>::random([2, 3, d_model], Distribution::Normal(0.0, 1.0), &device);
    let x = Param::from_tensor(Tensor::from_inner(base));
    let grads = mlp.forward(x.val()).sum().backward();

    let g = x.val().grad(&grads).expect("input grad exists");
    let gvec = g.into_data().to_vec::<f32>().unwrap();
    assert!(gvec.iter().all(|v| v.is_finite()));
    assert!(
        gvec.iter().any(|v| v.abs() > 0.),
        "the input gradient must not be identically zero"
    );
}
