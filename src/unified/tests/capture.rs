//! Tests for the graph-capture cache write-back ([`CacheTensors`]).
//!
//! Whether the write lands in the same buffer is a backend property (cubecl's
//! in-place `slice_assign`), exercised on CUDA by the decode-capture prototype.
//! What is pinned here is the traversal: every tensor field is visited — checked
//! field by field, not through the traversal itself — values move exactly, and
//! mismatched structures panic.

use crate::prelude::*;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

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

#[cfg(feature = "mamba3")]
mod mamba3 {
    use super::*;
    use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;

    /// Every optional field present: a two-tap trapezoid at `u = 2` (the tap
    /// FIFO), a Kalman gain (`ln Λ`), the tropical register, and a tensor
    /// rotation state.
    fn block(device: &Device) -> Mamba3 {
        Mamba3Config::new(16)
            .with_state_rank(4)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_ngroups(2)
            .with_micro_steps(2)
            .with_trapezoid(Trapezoid::VerticalPlusHorizontalCarryOver)
            .with_gain(Gain::Kalman)
            .with_tropical(Tropical::MaxPlus)
            .with_rotation(RotationKind::Quaternion4D)
            .init(device)
    }

    fn assert_fields(label: &str, a: &Mamba3Cache, b: &Mamba3Cache) {
        let fields = |c: &Mamba3Cache| -> Mamba3DoubleSsdCache {
            match c.clone() {
                Mamba3Cache::DoubleSsd(c) => c,
                Mamba3Cache::SingleSsd(c) => c.into(),
            }
        };
        let (a, b) = (fields(a), fields(b));
        assert!(a.k_state_bumhr.is_some() && a.log_precision_bh.is_some() && a.tropical_bh.is_some());
        assert_same(&format!("{label} ssm"), &a.ssm_bhpr, &b.ssm_bhpr);
        assert_same_option(&format!("{label} k slots"), &a.k_state_bumhr, &b.k_state_bumhr);
        assert_same_option(&format!("{label} v slots"), &a.v_state_buhp, &b.v_state_buhp);
        assert_same_option(&format!("{label} ln Λ"), &a.log_precision_bh, &b.log_precision_bh);
        assert_same_option(&format!("{label} tropical"), &a.tropical_bh, &b.tropical_bh);
        match (&a.rotation, &b.rotation) {
            (RotationState::Quaternion(a), RotationState::Quaternion(b)) => {
                assert_same(&format!("{label} rotation"), a, b)
            }
            _ => panic!("{label}: expected quaternion rotation states"),
        }
    }

    /// Two consecutive step caches, on the single- (`step`'s default) and the
    /// double-ssd pathway.
    fn two_caches(double: bool, device: &Device) -> (Mamba3Cache, Mamba3Cache) {
        let model = block(device);
        let (_, first) = model.step(input(2, 16, device), None);
        let first = match (double, first) {
            (true, Mamba3Cache::SingleSsd(c)) => Mamba3Cache::DoubleSsd(c.into()),
            (_, c) => c,
        };
        let (_, second) = model.step(input(2, 16, device), Some(first.clone()));
        (first, second)
    }

    #[test]
    fn owned_buffers_keep_values_and_assign_moves_them() {
        let device = Device::default();
        for double in [false, true] {
            let label = if double { "double" } else { "single" };
            let (first, second) = two_caches(double, &device);
            let owned = first.clone().into_owned_buffers();
            assert_fields(&format!("{label} owned"), &owned, &first);
            let assigned = owned.assign_in_place(second.clone());
            assert_fields(&format!("{label} assigned"), &assigned, &second);
            // The slotted form: the same through `Mamba3Caches` and the enum.
            let stack = |c: Mamba3Cache| MambaCaches::Mamba3(Mamba3Caches::from_vec(vec![c]));
            let assigned = stack(first.clone())
                .into_owned_buffers()
                .assign_in_place(stack(second.clone()));
            let MambaCaches::Mamba3(assigned) = assigned else { unreachable!() };
            let assigned = assigned.into_options().pop().flatten().unwrap();
            assert_fields(&format!("{label} stacked"), &assigned, &second);
        }
    }

    #[test]
    #[should_panic(expected = "SSD pathway")]
    fn pathway_mismatch_panics() {
        let device = Device::default();
        let (single, _) = two_caches(false, &device);
        let (double, _) = two_caches(true, &device);
        let _ = single.assign_in_place(double);
    }
}

#[cfg(feature = "mamba2")]
#[test]
fn mamba2_assign_moves_every_field() {
    let device = Device::default();
    let model = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .init(&device);
    let (_, first) = model.step(input(2, 16, &device), None);
    let (_, second) = model.step(input(2, 16, &device), Some(first.clone()));
    let assigned = first.into_owned_buffers().assign_in_place(second.clone());
    assert_same("conv", &assigned.conv_bvk, &second.conv_bvk);
    assert_same("ssm", &assigned.ssm_bhpr, &second.ssm_bhpr);
}

#[cfg(feature = "mamba1")]
#[test]
fn mamba1_assign_moves_every_field() {
    let device = Device::default();
    let model = Mamba1Config::new(16).with_state_rank(8).init(&device);
    let (_, first) = model.step(input(2, 16, &device), None);
    let (_, second) = model.step(input(2, 16, &device), Some(first.clone()));
    let assigned = first.into_owned_buffers().assign_in_place(second.clone());
    assert_same("conv", &assigned.conv_bik, &second.conv_bik);
    assert_same("ssm", &assigned.ssm_bir, &second.ssm_bir);
}
