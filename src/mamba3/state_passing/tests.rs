use super::{chunk_cumsum, state_passing};
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

fn reference(intra: Tensor<5>, decay: Tensor<3>, initial: Tensor<4>) -> Tensor<5> {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = intra.dims();
    let mut running = initial;
    let mut states = vec![running.clone()];
    for chunk in 0..nchunks {
        let injection = intra.clone().narrow(1, chunk, 1).squeeze_dim::<4>(1);
        let a = decay
            .clone()
            .narrow(2, chunk, 1)
            .squeeze_dim::<2>(2)
            .unsqueeze_dims::<4>(&[2, 3])
            .expand([batch, nheads, per_head_dim, state_rank]);
        running = a * running + injection;
        states.push(running.clone());
    }
    Tensor::stack(states, 1)
}

fn chunk_cumsum_reference(da: Tensor<4>) -> Tensor<4> {
    da.permute([0, 3, 1, 2]).cumsum(3)
}

#[test]
fn chunk_cumsum_matches_reference_values_and_gradients() {
    let device: Device = Default::default();
    let (batch, nchunks, chunk_len, nheads) = (2, 3, 7, 4);
    let da = Tensor::<4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Normal(0.0, 0.2),
        &device,
    );
    let head = Tensor::<4>::random(
        [batch, nheads, nchunks, chunk_len],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let run = |custom: bool| {
        let da = Param::from_tensor(Tensor::from_inner(da.clone()));
        let prefix = if custom {
            chunk_cumsum(da.val())
        } else {
            chunk_cumsum_reference(da.val())
        };
        let values = prefix.clone().inner();
        let grads = (prefix * Tensor::from_inner(head.clone())).sum().backward();
        (values, da.val().grad(&grads).expect("da grad"))
    };

    let custom = run(true);
    let plain = run(false);
    let value_diff = max_abs_diff(custom.0, plain.0);
    let da_diff = max_abs_diff(custom.1, plain.1);

    assert!(value_diff < 1e-5, "chunk cumsum max abs diff: {value_diff}");
    assert!(
        da_diff < 1e-5,
        "chunk cumsum da grad max abs diff: {da_diff}"
    );
}

#[test]
fn state_passing_matches_reference_values_and_gradients() {
    let device: Device = Default::default();
    let (batch, nchunks, nheads, per_head_dim, state_rank) = (2, 5, 3, 4, 3);
    let intra = Tensor::<5>::random(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 0.2),
        &device,
    );
    let decay = Tensor::<3>::random(
        [batch, nheads, nchunks],
        Distribution::Uniform(0.4, 0.95),
        &device,
    );
    let initial = Tensor::<4>::random(
        [batch, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 0.2),
        &device,
    );
    let head = Tensor::<5>::random(
        [batch, nchunks + 1, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let run = |custom: bool| {
        let intra = Param::from_tensor(Tensor::from_inner(intra.clone()));
        let decay = Param::from_tensor(Tensor::from_inner(decay.clone()));
        let initial = Param::from_tensor(Tensor::from_inner(initial.clone()));
        let states = if custom {
            state_passing(intra.val(), decay.val(), initial.val())
        } else {
            reference(intra.val(), decay.val(), initial.val())
        };
        let values = states.clone().inner();
        let grads = (states * Tensor::from_inner(head.clone())).sum().backward();
        (
            values,
            intra.val().grad(&grads).expect("intra grad"),
            decay.val().grad(&grads).expect("decay grad"),
            initial.val().grad(&grads).expect("initial grad"),
        )
    };

    let custom = run(true);
    let plain = run(false);
    let value_diff = max_abs_diff(custom.0, plain.0);
    let intra_diff = max_abs_diff(custom.1, plain.1);
    let decay_diff = max_abs_diff(custom.2, plain.2);
    let initial_diff = max_abs_diff(custom.3, plain.3);

    assert!(value_diff < 1e-5, "state value max abs diff: {value_diff}");
    assert!(intra_diff < 1e-5, "intra grad max abs diff: {intra_diff}");
    assert!(decay_diff < 1e-5, "decay grad max abs diff: {decay_diff}");
    assert!(
        initial_diff < 1e-5,
        "initial grad max abs diff: {initial_diff}"
    );
}
