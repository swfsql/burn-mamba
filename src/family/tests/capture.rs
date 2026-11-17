//! Tests of the cache write-back of a captured step ([`CacheTensors`]) on
//! the Mamba-3 cache.
//!
//! The backend decides whether the write lands in the same buffer (the
//! in-place `slice_assign` of cubecl). The examples capture their decoding,
//! and a backend that can record the calls (for example, CUDA) replays it.
//! These tests check the traversal of the Mamba-3 cache, with each optional
//! field present. They compare each field directly, not through the
//! traversal. The values move exactly.
//!
//! The contracts of `block_contract.rs` compare no field directly.
//! `cache_traversal_moves_every_field` reads the outputs from a moved cache.

use crate::prelude::*;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;
use burn_stack::utils::test_helpers::test_device;

type Device = burn::prelude::Device;

fn input(batch: usize, d_model: usize, device: &Device) -> Tensor<2> {
    Tensor::random([batch, d_model], Distribution::Normal(0.0, 1.0), device)
}

fn assert_same<const D: usize>(label: &str, a: &Tensor<D>, b: &Tensor<D>) {
    assert_eq!(a.dims(), b.dims(), "{label}: shape");
    let d = max_abs_diff(a.clone(), b.clone());
    assert_eq!(d, 0.0, "{label}: differs by {d}");
}

fn assert_same_option<const D: usize>(label: &str, a: &Option<Tensor<D>>, b: &Option<Tensor<D>>) {
    assert_eq!(a.is_some(), b.is_some(), "{label}: present in one cache only");
    if let (Some(a), Some(b)) = (a, b) {
        assert_same(label, a, b);
    }
}

/// Every optional field present: a two-tap pattern at `u = 2` (the tap
/// FIFO), a Kalman gain (`ln Λ`), the tropical register, and a tensor
/// rotation state.
fn block(device: &Device) -> Mamba3 {
    Mamba3Config::new(16)
        .with_state_rank(4)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_ngroups(2)
        .with_micro_steps(2)
        .with_tap_pattern(TapPattern::VerticalPlusHorizontalCarryOver)
        .with_gain(Gain::Kalman)
        .with_tropical(Tropical::MaxPlus)
        .with_rotation(RotationKind::Quaternion4D)
        .init(device)
}

fn assert_fields(label: &str, a: &Mamba3Cache, b: &Mamba3Cache) {
    assert_same(&format!("{label} ssm"), &a.ssm_bhpr, &b.ssm_bhpr);
    let (a, b) = (&a.frame, &b.frame);
    assert!(a.tap_fifo.is_some() && a.log_precision_bh.is_some() && a.tropical_bh.is_some());
    let fifo = |c: &Mamba3FrameCache| c.tap_fifo.clone().map(|f| (f.b_bumhr, f.x_buhp)).unzip();
    let ((a_b, a_x), (b_b, b_x)) = (fifo(a), fifo(b));
    assert_same_option(&format!("{label} tap FIFO B"), &a_b, &b_b);
    assert_same_option(&format!("{label} tap FIFO x"), &a_x, &b_x);
    assert_same_option(&format!("{label} ln Λ"), &a.log_precision_bh, &b.log_precision_bh);
    assert_same_option(&format!("{label} tropical"), &a.tropical_bh, &b.tropical_bh);
    match (&a.rotation, &b.rotation) {
        (RotationState::Quaternion(a), RotationState::Quaternion(b)) => {
            assert_same(&format!("{label} rotation"), a, b)
        }
        _ => panic!("{label}: expected quaternion rotation states"),
    }
}

/// Two consecutive step caches.
fn two_caches(device: &Device) -> (Mamba3Cache, Mamba3Cache) {
    let model = block(device);
    let (_, first) = model.step(input(2, 16, device), None);
    let (_, second) = model.step(input(2, 16, device), Some(first.clone()));
    (first, second)
}

#[test]
fn owned_buffers_keep_values_and_assign_moves_them() {
    let device = test_device();
    let (first, second) = two_caches(&device);
    let owned = first.clone().into_owned_buffers();
    assert_fields("owned", &owned, &first);
    let assigned = owned.assign_in_place(second.clone());
    assert_fields("assigned", &assigned, &second);
    // The stacked form: the same through the caches of a layer stack.
    let assigned = vec![first.clone()]
        .into_owned_buffers()
        .assign_in_place(vec![second.clone()]);
    let [assigned] = assigned.as_slice() else {
        panic!("the caches of the stack must hold one slot");
    };
    assert_fields("stacked", assigned, &second);
    // The same through the caches of the enum.
    let flat = |c: Mamba3Cache| vec![MambaCache::Mamba3(c)];
    let assigned = flat(first.clone())
        .into_owned_buffers()
        .assign_in_place(flat(second.clone()));
    let [MambaCache::Mamba3(assigned)] = assigned.as_slice() else {
        panic!("the caches of the stack must hold one slot of the Mamba-3 family");
    };
    assert_fields("flat", assigned, &second);
}
