//! A right-padded call is each of its slots run alone: the outputs of a slot's
//! real rows, every field of its returned cache, and every parameter's gradient
//! of a loss over both. Each call continues an unpadded one, so the cache it
//! reads is not fresh — and a slot with no real row at all must hand that cache
//! back untouched.

use crate::mamba1::prelude::*;
use crate::mamba2::prelude::*;
use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
use crate::mamba3::prelude::*;
use crate::mamba3::single_ssd::prelude::Mamba3SingleSsdCache;
use burn::module::{ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients};
use burn_stack::modules::{LatentNetworkBuilder, LayersBuilder};
use burn_stack::utils::ClassLatent;
use burn_stack::utils::test_helpers::max_abs_diff;

const D_MODEL: usize = 16;
const VAL_TOL: f32 = 1e-4;
const GRAD_TOL: f32 = 1e-3;

/// Slot lengths of the padded call: one full slot, a short one, one with no
/// real row, and one a single token long.
const LENS: [usize; 4] = [7, 3, 0, 1];
/// Tokens of the unpadded call before it.
const PREFIX: usize = 4;

/// `[batch, padded]`, `true` past each slot's length.
fn pad_mask(lens: &[usize], padded: usize, device: &Device) -> Tensor<2, Bool> {
    let batch = lens.len();
    let lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
    let lens_bs = Tensor::<1, Int>::from_ints(lens.as_slice(), device)
        .reshape([batch, 1])
        .expand([batch, padded]);
    Tensor::<1, Int>::arange(0..padded as i64, device)
        .reshape([1, padded])
        .expand([batch, padded])
        .greater_equal(lens_bs)
}

/// A cache field as `[batch, everything else]`.
fn rows<const D: usize>(t: &Tensor<D>) -> Tensor<2> {
    let batch = t.dims()[0];
    let n = t.shape().num_elements() / batch;
    t.clone().reshape([batch, n])
}

fn assert_close<const D: usize>(got: Tensor<D>, want: Tensor<D>, tol: f32, what: &str) {
    let scale = 1.0 + want.clone().abs().max().into_scalar::<f32>();
    let diff = max_abs_diff(got, want);
    assert!(diff <= tol * scale, "{what}: differs by {diff} (scale {scale})");
}

/// Every float parameter's gradient, in visiting order.
struct Collect<'a> {
    grads: &'a Gradients,
    out: Vec<Option<Tensor<1>>>,
}

impl ModuleVisitor for Collect<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let n = param.val().shape().num_elements();
        self.out.push(param.val().grad(self.grads).map(|g| g.reshape([n])));
    }
}

fn param_grads<M: Module>(module: &M, grads: &Gradients) -> Vec<Option<Tensor<1>>> {
    let mut collect = Collect {
        grads,
        out: Vec::new(),
    };
    module.visit(&mut collect);
    collect.out
}

fn assert_grads_close(padded: Vec<Option<Tensor<1>>>, alone: Vec<Option<Tensor<1>>>) {
    assert_eq!(padded.len(), alone.len());
    for (i, (p, a)) in padded.into_iter().zip(alone).enumerate() {
        match (p, a) {
            (Some(p), Some(a)) => assert_close(p, a, GRAD_TOL, &format!("parameter {i}'s gradient")),
            (None, None) => {}
            (p, a) => panic!(
                "parameter {i}: gradient present padded {} / alone {}",
                p.is_some(),
                a.is_some()
            ),
        }
    }
}

/// Run a padded call (after an unpadded prefix one) and each slot's own
/// calls, and compare each slot's outputs, cache fields (`fields`) and the
/// gradients of a loss over both.
fn check_each_slot_alone<C, M: Module>(
    module: &M,
    forward: impl Fn(Tensor<3>, Option<C>, Option<Tensor<2, Bool>>) -> (Tensor<3>, C),
    fields: impl Fn(&C) -> Vec<Tensor<2>>,
) {
    let device = Device::default().autodiff();
    let batch = LENS.len();
    let padded = *LENS.iter().max().unwrap();
    let normal = Distribution::Normal(0.0, 1.0);
    let x0 = Tensor::<3>::random([batch, PREFIX, D_MODEL], normal, &device);
    let x = Tensor::<3>::random([batch, padded, D_MODEL], normal, &device);

    let (_, cache0) = forward(x0.clone(), None, None);
    let (y, cache) = forward(x.clone(), Some(cache0), Some(pad_mask(&LENS, padded, &device)));
    let fields_padded = fields(&cache);
    let weight_y = Tensor::<3>::random([1, padded, D_MODEL], normal, &device);
    let weights_c: Vec<Tensor<2>> = fields_padded
        .iter()
        .map(|f| Tensor::random([1, f.dims()[1]], normal, &device))
        .collect();

    let mut loss_padded = Tensor::<1>::zeros([1], &device);
    let mut loss_alone = Tensor::<1>::zeros([1], &device);
    for (b, &len) in LENS.iter().enumerate() {
        let (_, cache0_b) = forward(x0.clone().narrow(0, b, 1), None, None);
        let cache_b = match len {
            0 => cache0_b,
            _ => {
                let (y_b, cache_b) =
                    forward(x.clone().narrow(0, b, 1).narrow(1, 0, len), Some(cache0_b), None);
                let y_own = y.clone().narrow(0, b, 1).narrow(1, 0, len);
                assert_close(y_own.clone(), y_b.clone(), VAL_TOL, &format!("slot {b}'s outputs"));
                let weight = weight_y.clone().narrow(1, 0, len);
                loss_padded = loss_padded + (y_own * weight.clone()).sum();
                loss_alone = loss_alone + (y_b * weight).sum();
                cache_b
            }
        };
        let fields_b = fields(&cache_b);
        assert_eq!(fields_padded.len(), fields_b.len());
        for (k, ((f, f_b), w)) in fields_padded.iter().zip(fields_b).zip(&weights_c).enumerate() {
            let f_own = f.clone().narrow(0, b, 1);
            assert_close(f_own.clone(), f_b.clone(), VAL_TOL, &format!("slot {b}'s cache field {k}"));
            loss_padded = loss_padded + (f_own * w.clone()).sum();
            loss_alone = loss_alone + (f_b * w.clone()).sum();
        }
    }
    assert_grads_close(
        param_grads(module, &loss_padded.backward()),
        param_grads(module, &loss_alone.backward()),
    );
}

#[test]
fn padded_mamba1_is_each_slot_alone() {
    let device = Device::default().autodiff();
    let block = Mamba1Config::new(D_MODEL)
        .with_state_rank(8)
        .with_conv_kernel(4)
        .init(&device);
    check_each_slot_alone(
        &block,
        |x, cache, pad| block.forward(x, cache, pad),
        |c: &Mamba1Cache| vec![rows(&c.conv_bik), rows(&c.ssm_bir)],
    );
}

#[test]
fn padded_mamba2_is_each_slot_alone() {
    let device = Device::default().autodiff();
    let block = Mamba2Config::new(D_MODEL)
        .with_state_rank(8)
        .with_per_head_dim(8)
        .init(&device);
    for path in [Mamba2SsdPath::Minimal(Some(4)), Mamba2SsdPath::SerialRecalculated(Some(4))] {
        check_each_slot_alone(
            &block,
            |x, cache, pad| block.forward(x, cache, path.clone(), pad),
            |c: &Mamba2Cache| vec![rows(&c.conv_bvk), rows(&c.ssm_bhpr)],
        );
    }
}

/// Every Mamba-3 cache field, whichever pathway it comes from (the two are
/// field-identical).
macro_rules! mamba3_fields {
    ($c:expr) => {{
        let c = $c;
        let mut out = vec![rows(&c.ssm_bhpr)];
        out.extend(c.k_state_bumhr.as_ref().map(rows));
        out.extend(c.v_state_buhp.as_ref().map(rows));
        match &c.rotation {
            RotationState::Real(_) => {}
            RotationState::Angle(t) => out.push(rows(t)),
            RotationState::Quaternion(t) | RotationState::Rotor(t) => out.push(rows(t)),
        }
        out.extend(c.log_precision_bh.as_ref().map(rows));
        out.extend(c.tropical_bh.as_ref().map(rows));
        out
    }};
}

fn mamba3(
    rotation: RotationKind,
    trapezoid: Trapezoid,
    micro_steps: usize,
    mimo_rank: usize,
    gain: Gain,
    tropical: Tropical,
) -> Mamba3Config {
    Mamba3Config::new(D_MODEL)
        .with_state_rank(8)
        .with_per_head_dim(8)
        .with_rotation(rotation)
        .with_trapezoid(trapezoid)
        .with_micro_steps(micro_steps)
        .with_mimo_rank(mimo_rank)
        .with_gain(gain)
        .with_tropical(tropical)
}

/// Both pathways over a spread of the block's dials: every tap pattern family
/// (none, lag 1, lag `u`, two taps, the token-gated one), every rotation kind,
/// `u > 1`, MIMO, both positive systems, and chunks short enough that the real
/// rows span several.
#[test]
fn padded_mamba3_is_each_slot_alone() {
    use {Gain as G, RotationKind as R, Trapezoid as T, Tropical as Tr};
    let device = Device::default().autodiff();
    let configs = [
        mamba3(R::Complex2D, T::HorizontalCarryOver, 1, 1, G::Projected, Tr::None),
        mamba3(R::Quaternion4D, T::Vertical, 3, 2, G::Kalman, Tr::MaxPlus),
        mamba3(R::Rotor4D, T::VerticalPlusHorizontalCarryOver, 2, 1, G::KalmanProjectedNoise, Tr::None),
        mamba3(R::Real1D, T::None, 2, 2, G::Projected, Tr::MaxPlus),
        mamba3(R::Complex2D, T::HorizontalReset, 3, 1, G::Kalman, Tr::None),
    ];
    let paths = [
        Mamba3SsdPath::Minimal(Some(6)),
        Mamba3SsdPath::Serial(Some(6)),
        Mamba3SsdPath::SerialRecalculated(Some(6)),
    ];
    for (i, config) in configs.iter().enumerate() {
        let block = config.init(&device);
        let path = paths[i % paths.len()].clone();
        check_each_slot_alone(
            &block,
            |x, cache, pad| block.forward_double_ssd(x, cache, &path, pad),
            |c: &Mamba3DoubleSsdCache| mamba3_fields!(c),
        );
        check_each_slot_alone(
            &block,
            |x, cache, pad| block.forward_single_ssd(x, cache, &path, pad),
            |c: &Mamba3SingleSsdCache| mamba3_fields!(c),
        );
    }
}

/// Through a network whose stack closes every sequence with an `End` latent: a
/// real block, run on each slot's rows gathered into that slot's order, sees
/// the `End` right after the slot's own last token — its tap reading that
/// token, not the padding the batch-wide `End` follows.
#[test]
fn padded_mamba3_network_closes_each_slot_at_its_own_end() {
    let device = Device::default().autodiff();
    let block = mamba3(
        RotationKind::Quaternion4D,
        Trapezoid::Vertical,
        2,
        1,
        Gain::Kalman,
        Tropical::MaxPlus,
    );
    let net = LatentNetworkBuilder {
        input_size: D_MODEL,
        layers: LayersBuilder {
            class_latents: vec![ClassLatent::Start, ClassLatent::End],
            ..LayersBuilder::new(2, block)
        },
        output_size: 3,
        final_norm: true,
        class_tokens: Vec::new(),
    }
    .init(&device);
    let path = Mamba3SsdPath::SerialRecalculated(Some(4));

    let lens = [6, 2, 4];
    let batch = lens.len();
    let padded = 6;
    let x = Tensor::<3>::random([batch, padded, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (y, caches) = net.forward(
        x.clone(),
        None,
        path.clone(),
        None,
        Some(pad_mask(&lens, padded, &device)),
    );
    // Rows: `Start`, the tokens, then `End` — after the padding batch-wide.
    assert_eq!(y.dims()[1], padded + 2);
    let Mamba3Caches::SingleSsd(caches) = caches else {
        panic!("a fresh call runs the single-SSD pathway");
    };
    for (b, &len) in lens.iter().enumerate() {
        let (y_b, caches_b) =
            net.forward(x.clone().narrow(0, b, 1).narrow(1, 0, len), None, path.clone(), None, None);
        let Mamba3Caches::SingleSsd(caches_b) = caches_b else {
            panic!("a fresh call runs the single-SSD pathway");
        };
        let y_own = Tensor::cat(
            vec![
                y.clone().narrow(0, b, 1).narrow(1, 0, 1 + len),
                y.clone().narrow(0, b, 1).narrow(1, padded + 1, 1),
            ],
            1,
        );
        assert_close(y_own, y_b, VAL_TOL, &format!("slot {b}'s rows"));
        for (l, (c, c_b)) in caches.caches.iter().zip(&caches_b.caches).enumerate() {
            for (k, (f, f_b)) in mamba3_fields!(c).into_iter().zip(mamba3_fields!(c_b)).enumerate() {
                let what = format!("slot {b}, layer {l}, cache field {k}");
                assert_close(f.narrow(0, b, 1), f_b, VAL_TOL, &what);
            }
        }
    }
}
