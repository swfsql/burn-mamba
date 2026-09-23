//! The positive systems, from the log-semiring primitive up to the block.
//!
//! 1. **The scan**: `lse` is exact beside [`LOG_ZERO`] and splits a tie; the
//!    doubling [`scan::prefix`] is the sequential [`scan::fold`], in values and
//!    gradients; the projective read ignores a common shift.
//! 2. **The members against their textbook recursions**, in f64: the gate is
//!    the covariance-form Kalman filter (and `η/Λ` its mean), and under a
//!    trapezoid the plant's own recurrence on ones; the register is the hard
//!    max-plus recursion to within `ln(t + 1)`.
//! 3. **The invariants the design leans on**: the confidence ceiling, a decay
//!    that only ever adds forgetting, Birkhoff's contraction rate, and finite
//!    gradients where a mass underflows (f32 and f16).
//! 4. **The block**: stock at `κ = 0` / `e = 0`, with live gradients into the
//!    join (the out-norm on or off), and finite ones when a head's `Δ`
//!    vanishes; `forward` ≡ `step` ≡ the other pathway ≡ a split prefill ≡ the other
//!    SSD algorithm, on outputs, every cache slot and gradients, across the
//!    lattice the two systems have to commute with (tap patterns, micro-steps,
//!    rotation kinds, MIMO, the out-norm); the slots survive a no-grad region;
//!    the in-projection's widths agree with the Muon plan.

use super::kalman::{self, GainInput, LogMasses};
use super::scan::{self, Affine, Mobius};
use super::{Gain, LOG_ZERO, Tropical};
use crate::mamba3::cache::Mamba3Cache;
use crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache;
use crate::mamba3::mamba3::{Mamba3, Mamba3Config};
use crate::mamba3::rotation::RotationKind;
use crate::mamba3::ssd_path::Mamba3SsdPath;
use crate::mamba3::trapezoid::Trapezoid;
use burn::module::Param;
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::{DType, Distribution};
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

fn uniform<const D: usize>(dims: [usize; D], lo: f64, hi: f64, device: &Device) -> Tensor<D> {
    Tensor::random(dims, Distribution::Uniform(lo, hi), device)
}

fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

// ---------------------------------------------------------------------------
// 1. The scan
// ---------------------------------------------------------------------------

#[test]
fn lse_is_exact_beside_log_zero_and_splits_a_tie() {
    let device: Device = Default::default();
    let x = uniform([2, 5, 3], -4.0, 4.0, &device);
    let zero = x.full_like(LOG_ZERO);
    assert_eq!(max_abs_diff(scan::lse(x.clone(), zero.clone()), x.clone()), 0.0);
    assert_eq!(max_abs_diff(scan::lse(zero, x.clone()), x.clone()), 0.0);
    let tie = scan::lse(x.clone(), x.clone());
    assert!(max_abs_diff(tie, x.clone() + std::f32::consts::LN_2) < 1e-5);

    // At a tie each side takes half the gradient, not one side all of it.
    let ad = device.autodiff();
    let a = Param::from_tensor(Tensor::<3>::from_inner(x.clone()).to_device(&ad));
    let b = Param::from_tensor(Tensor::<3>::from_inner(x).to_device(&ad));
    let grads = scan::lse(a.val(), b.val()).sum().backward();
    for g in [a.val().grad(&grads).unwrap(), b.val().grad(&grads).unwrap()] {
        assert!(max_abs_diff(g.clone(), g.full_like(0.5)) < 1e-6);
    }
}

fn random_mobius(dims: [usize; 3], device: &Device) -> Mobius {
    Mobius {
        m00: uniform(dims, -3.0, 1.0, device),
        m01: uniform(dims, -3.0, 1.0, device),
        m10: uniform(dims, -3.0, 1.0, device),
        m11: uniform(dims, -3.0, 1.0, device),
    }
}

fn random_affine(dims: [usize; 3], device: &Device) -> Affine {
    Affine {
        a: uniform(dims, -2.0, 1.0, device),
        b: uniform(dims, -3.0, 3.0, device),
    }
}

#[test]
fn prefix_matches_fold_in_values() {
    let device: Device = Default::default();
    for len in [1, 2, 3, 5, 8, 13, 33] {
        let dims = [2, len, 3];
        let carry = uniform([2, 3], -2.0, 2.0, &device);

        let m = random_mobius(dims, &device);
        let d = max_abs_diff(
            scan::prefix(m.clone()).apply(carry.clone()),
            scan::fold(m).apply(carry.clone()),
        );
        assert!(d < 1e-4, "Möbius, len {len}: prefix vs fold {d}");

        let a = random_affine(dims, &device);
        let d = max_abs_diff(
            scan::prefix(a.clone()).apply(carry.clone()),
            scan::fold(a).apply(carry),
        );
        assert!(d < 1e-4, "affine, len {len}: prefix vs fold {d}");
    }
}

#[test]
fn prefix_matches_fold_in_gradients() {
    let device: Device = Default::default();
    let ad = device.clone().autodiff();
    let dims = [2, 11, 3];
    let lift = |t: Tensor<3>| Param::from_tensor(Tensor::<3>::from_inner(t).to_device(&ad));
    let raw = random_mobius(dims, &device);
    let head = Tensor::<3>::from_inner(uniform(dims, -1.0, 1.0, &device)).to_device(&ad);
    let carry = Tensor::<2>::from_inner(uniform([2, 3], -2.0, 2.0, &device)).to_device(&ad);

    let run = |scanner: fn(Mobius) -> Mobius| {
        let p = [raw.m00.clone(), raw.m01.clone(), raw.m10.clone(), raw.m11.clone()].map(lift);
        let m = Mobius {
            m00: p[0].val(),
            m01: p[1].val(),
            m10: p[2].val(),
            m11: p[3].val(),
        };
        let grads = (scanner(m).apply(carry.clone()) * head.clone()).sum().backward();
        p.map(|p| p.val().grad(&grads).unwrap())
    };
    let by_prefix = run(|m| scan::prefix(m));
    let by_fold = run(|m| scan::fold(m));
    for (i, (a, b)) in by_prefix.into_iter().zip(by_fold).enumerate() {
        let d = max_abs_diff(a, b);
        assert!(d < 1e-4, "entry {i}: prefix vs fold gradient {d}");
    }
}

#[test]
fn projective_read_ignores_a_common_shift() {
    let device: Device = Default::default();
    let m = random_mobius([2, 4, 3], &device);
    let carry = uniform([2, 3], -2.0, 2.0, &device);
    let shift = uniform([2, 4, 3], -5.0, 5.0, &device);
    let shifted = Mobius {
        m00: m.m00.clone() + shift.clone(),
        m01: m.m01.clone() + shift.clone(),
        m10: m.m10.clone() + shift.clone(),
        m11: m.m11.clone() + shift,
    };
    assert!(max_abs_diff(m.apply(carry.clone()), shifted.apply(carry)) < 1e-4);
}

// ---------------------------------------------------------------------------
// 2. The members against their textbook recursions
// ---------------------------------------------------------------------------

/// A random run of discretisations and gate inputs: `(Δ, ln α, ln κ, r, ℓ₀)`.
struct GateCase {
    dt: Tensor<3>,
    da: Tensor<3>,
    log_kappa: Tensor<1>,
    noise: Tensor<3>,
    carry: Tensor<2>,
}

fn gate_case(batch: usize, len: usize, device: &Device) -> GateCase {
    let nheads = 3;
    let dt = uniform([batch, len, nheads], 0.05, 2.0, device);
    let da = dt.clone() * -uniform([batch, len, nheads], 0.01, 1.5, device);
    GateCase {
        dt,
        da,
        log_kappa: Tensor::from_floats([0.05f32.ln(), 0.5f32.ln(), 3.0f32.ln()], device),
        noise: uniform([batch, len, nheads], -2.0, 2.0, device),
        carry: uniform([batch, nheads], -1.0, 2.0, device),
    }
}

/// The gate with no trapezoid split (`γ = Δ`, no `ν`): the textbook filter.
fn run_gate(case: &GateCase, carry: Tensor<2>) -> kalman::GainOutput {
    let ln_dt = case.dt.clone().log();
    kalman::gate(
        case.da.clone(),
        LogMasses {
            dt_bsh: ln_dt.clone(),
            gamma_bsh: ln_dt,
            nu_bsh: None,
        },
        GainInput {
            log_kappa_h: case.log_kappa.clone(),
            noise_bsh: Some(case.noise.clone()),
            carry_bh: carry,
        },
    )
}

/// The gate is the scalar Kalman filter of a random walk observed through the
/// token's write, `a² = 1/α`, `q = κ·Δ·eʳ`, `r = 1/Δ`, written in covariance
/// form in f64: `Λₜ = 1/Pₜ` and the log-decay `ln dₜ`, at every position. The
/// same loop checks the information-form identity the plant relies on — the
/// state `η` updated by `dₜ` divided by `Λ` is the filter's mean.
#[test]
fn gate_matches_the_covariance_form_filter() {
    let device: Device = Default::default();
    let (batch, len, nheads) = (2, 24, 3);
    let case = gate_case(batch, len, &device);
    let out = run_gate(&case, case.carry.clone());

    let (dt, da, kappa, noise, carry) = (
        floats(case.dt.clone()),
        floats(case.da.clone()),
        floats(case.log_kappa.clone().exp()),
        floats(case.noise.clone()),
        floats(case.carry.clone()),
    );
    let (lp, da_out) = (floats(out.log_precision_bsh), floats(out.da_bsh));
    let targets: Vec<f64> = (0..len).map(|t| ((t * 7 % 5) as f64) - 2.0).collect();
    for b in 0..batch {
        for h in 0..nheads {
            let mut p = (-(carry[b * nheads + h] as f64)).exp();
            let (mut mean, mut eta) = (0.3f64, 0.3 / p);
            for t in 0..len {
                let i = (b * len + t) * nheads + h;
                let alpha = (da[i] as f64).exp();
                let a2 = 1.0 / alpha;
                let q = kappa[h] as f64 * dt[i] as f64 * (noise[i] as f64).exp();
                let m = dt[i] as f64;
                let lambda_prev = 1.0 / p;
                let d = 1.0 / (a2 + q * lambda_prev);

                let p_prior = a2 * p + q;
                let k = p_prior / (p_prior + 1.0 / m);
                p *= 0.0;
                p += (1.0 - k) * p_prior;
                mean += k * (targets[t] - mean);
                eta = d * eta + m * targets[t];

                let lambda = 1.0 / p;
                assert!(
                    ((lp[i] as f64).exp() / lambda - 1.0).abs() < 2e-3,
                    "Λ at b{b} h{h} t{t}: {} vs {lambda}",
                    (lp[i] as f64).exp()
                );
                assert!(
                    (da_out[i] as f64 - d.ln()).abs() < 2e-3,
                    "ln d at b{b} h{h} t{t}: {} vs {}",
                    da_out[i],
                    d.ln()
                );
                assert!(
                    (eta / lambda - mean).abs() < 1e-9 * (1.0 + mean.abs()),
                    "η/Λ is the mean at b{b} h{h} t{t}"
                );
            }
        }
    }

    // A fresh start holds no evidence: the first decay is stock's, and the first
    // precision is the first mass.
    let fresh = run_gate(&case, case.carry.full_like(LOG_ZERO));
    let first = |t: Tensor<3>| t.narrow(1, 0, 1);
    assert_eq!(max_abs_diff(first(fresh.da_bsh), first(case.da.clone())), 0.0);
    assert!(max_abs_diff(first(fresh.log_precision_bsh), first(case.dt.clone().log())) < 1e-5);
}

/// Under a trapezoid the evidence is the plant's own two installments, and `Λ`
/// is the plant's recurrence on ones, run on the very coefficients the SSD
/// consumes: `Wₜ = αₜ·Wₜ₋₁ + γₜ + αₜ·νⁱⁿᵗₜ + νₜ·Tₜ`, `Tₜ` the far tap's transport
/// across its lag (front factors `1`, as `helpers::interior_gap_decay` pads).
/// Equal for a lag-1 tap, an upper bound for a lag-`u` one; the ceiling
/// `Λₜ < 1/qₜ + γₜ` holds either way.
#[test]
fn the_precision_is_the_plants_weight_on_ones() {
    use crate::mamba3::helpers::trapezoidal_coefficients;
    use crate::mamba3::trapezoid::TrapezoidSpec;
    use Trapezoid as T;
    let device: Device = Default::default();
    let (batch, nheads) = (2, 3);
    for (pattern, u, exact) in [
        (T::HorizontalCarryOver, 1, true),
        (T::HorizontalCarryOver, 3, true),
        (T::HorizontalReset, 2, true),
        (T::VerticalPlusHorizontalCarryOver, 1, true),
        (T::Vertical, 2, false),
        (T::VerticalPlusHorizontalCarryOver, 3, false),
    ] {
        let len = 8 * u;
        let dims = [batch, len, nheads];
        let spec = TrapezoidSpec {
            pattern,
            micro_steps: u,
            dt_limit: (0.0, 6.5504e4),
            a_floor: 1e-4,
        };
        let log_kappa = Tensor::<1>::from_floats([0.05f32.ln(), 0.5f32.ln(), 3.0f32.ln()], &device);
        let carry = uniform([batch, nheads], -1.0, 2.0, &device);
        let coeffs = trapezoidal_coefficients(
            uniform(dims, -2.0, 1.0, &device),
            uniform(dims, -3.0, 1.0, &device),
            Some(uniform(dims, -3.0, 3.0, &device)),
            pattern.has_interior_tap(u).then(|| uniform(dims, -3.0, 3.0, &device)),
            Tensor::zeros([nheads], &device),
            spec,
            Some(GainInput {
                log_kappa_h: log_kappa.clone(),
                noise_bsh: None,
                carry_bh: carry.clone(),
            }),
        );
        let lag = pattern.tap_lag(u);
        let (alpha, gamma, dt) = (floats(coeffs.alpha), floats(coeffs.gamma), floats(coeffs.dt));
        let nu = floats(coeffs.nu.expect("a β tap"));
        let nu_interior = coeffs.nu_interior.map(floats);
        let lp = floats(coeffs.log_precision.expect("a Kalman gain"));
        let (kappa, carry) = (floats(log_kappa.exp()), floats(carry));
        for b in 0..batch {
            for h in 0..nheads {
                let at = |t: usize| (b * len + t) * nheads + h;
                let mut w = (carry[b * nheads + h] as f64).exp();
                for t in 0..len {
                    let transport: f64 =
                        (t.saturating_sub(lag - 1)..=t).map(|j| alpha[at(j)] as f64).product();
                    let interior =
                        nu_interior.as_ref().map_or(0.0, |n| alpha[at(t)] as f64 * n[at(t)] as f64);
                    w = alpha[at(t)] as f64 * w
                        + gamma[at(t)] as f64
                        + nu[at(t)] as f64 * transport
                        + interior;
                    let lambda = (lp[at(t)] as f64).exp();
                    let label = format!("{pattern:?} u{u} b{b} h{h} t{t}");
                    if exact {
                        assert!((lambda / w - 1.0).abs() < 1e-3, "{label}: Λ {lambda} vs plant {w}");
                    } else {
                        assert!(lambda >= w * (1.0 - 1e-3), "{label}: Λ {lambda} under plant {w}");
                    }
                    let q = kappa[h] as f64 * dt[at(t)] as f64;
                    let ceiling = 1.0 / q + gamma[at(t)] as f64;
                    assert!(lambda < ceiling * 1.0001, "{label}: Λ {lambda} above {ceiling}");
                }
            }
        }
    }
}

/// The soft register is the max-plus recursion `c = max(c + a, b)` from above,
/// and by at most `ln(t + 1)` in projection units — the log-sum-exp over the
/// `t + 1` ways the maximum can have been reached.
#[test]
fn register_tracks_max_plus_within_its_log_bound() {
    let device: Device = Default::default();
    let (batch, len, nheads) = (2, 40, 3);
    // Integer-valued projections at a large scale: a counter, a floor and
    // resets, with ties.
    let a = uniform([batch, len, nheads], -3.0, 3.0, &device).round();
    let b = uniform([batch, len, nheads], -3.0, 3.0, &device).round() * 8.0;
    let soft = floats(super::tropical::register(
        a.clone(),
        b.clone(),
        Tensor::full([batch, nheads], LOG_ZERO, &device),
    ));
    let (a, b) = (floats(a), floats(b));
    for bi in 0..batch {
        for h in 0..nheads {
            let mut hard = f64::NEG_INFINITY;
            for t in 0..len {
                let i = (bi * len + t) * nheads + h;
                hard = (hard + a[i] as f64).max(b[i] as f64);
                let gap = soft[i] as f64 - hard;
                assert!(
                    gap > -1e-3 && gap <= ((t + 1) as f64).ln() + 1e-3,
                    "b{bi} h{h} t{t}: soft {} vs hard {hard}",
                    soft[i]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Invariants
// ---------------------------------------------------------------------------

/// `Λₜ < 1/qₜ + γₜ` (here `γ = Δ`) whatever came before, and `dₜ ≤ αₜ`: the gate can add
/// forgetting, never remove it.
#[test]
fn precision_has_a_ceiling_and_the_decay_only_adds_forgetting() {
    let device: Device = Default::default();
    let case = gate_case(3, 40, &device);
    let out = run_gate(&case, case.carry.clone() + 5.0);
    let q = case.log_kappa.clone().unsqueeze::<3>().exp() * case.dt.clone() * case.noise.clone().exp();
    let ceiling = q.recip() + case.dt.clone();
    let excess = out.log_precision_bsh.exp() - ceiling.clone() * 1.0001;
    assert!(excess.max().into_scalar::<f32>() < 0.0, "a precision above its ceiling");
    let raise = out.da_bsh - case.da.clone();
    assert!(raise.max().into_scalar::<f32>() <= 0.0, "a decay above stock's");
}

/// For `q, m > 0` every step is an entrywise-positive matrix, so it contracts
/// the Hilbert distance `|ln Λ − ln Λ'|` by at least
/// `tanh(¼·ln(1 + 1/(q·γ)))` (here `γ = Δ`) — with no `α` in it — and two caches that disagree
/// about the evidence held forget the disagreement.
#[test]
fn a_disagreement_about_the_evidence_contracts_at_the_birkhoff_rate() {
    let device: Device = Default::default();
    let (batch, len, nheads) = (2, 30, 3);
    let case = gate_case(batch, len, &device);
    let a = floats(run_gate(&case, case.carry.clone()).log_precision_bsh);
    let b = floats(run_gate(&case, case.carry.clone() - 4.0).log_precision_bsh);
    let (dt, kappa, noise) = (
        floats(case.dt.clone()),
        floats(case.log_kappa.clone().exp()),
        floats(case.noise.clone()),
    );
    for bi in 0..batch {
        for h in 0..nheads {
            let mut bound = 4.0f64;
            for t in 0..len {
                let i = (bi * len + t) * nheads + h;
                let qm = kappa[h] as f64 * (dt[i] as f64).powi(2) * (noise[i] as f64).exp();
                bound *= (0.25 * (1.0 + 1.0 / qm).ln()).tanh();
                let dist = (a[i] - b[i]).abs() as f64;
                assert!(dist <= bound + 2e-3, "b{bi} h{h} t{t}: {dist} > {bound}");
            }
        }
    }
}

fn all_finite<const D: usize>(t: Tensor<D>) -> bool {
    floats(t.cast(DType::F32)).iter().all(|v| v.is_finite())
}

/// A mass that underflows to `0` — `Δ` past softplus's range (≈ −104 of
/// pre-activation in f32, ≈ −17 in f16) or a saturated `λ` — leaves the
/// gradient finite, in both dtypes: the gate's masses are logs of
/// pre-activations, never `ln` of a zero.
#[test]
fn an_underflowing_mass_keeps_the_gradient_finite() {
    let device: Device = Default::default();
    let ad = device.autodiff();
    let dt_raw = [-1e4f32, -200.0, -40.0, -17.5, -8.0, 0.0, 30.0];
    let lambda_raw = [0.0f32, 200.0, -200.0, 40.0, -40.0, 1.0, -1.0];
    for dtype in [DType::F32, DType::F16] {
        let on = |t: Tensor<3>| t.cast(dtype);
        let lift = |v: [f32; 7]| Param::from_tensor(on(Tensor::<1>::from_floats(v, &ad).reshape([1, 7, 1])));
        let (x, l) = (lift(dt_raw), lift(lambda_raw));
        let masses = LogMasses::new(x.val(), (0.0, 6.5504e4), Some(l.val()), None);
        let out = kalman::gate(
            on(Tensor::full([1, 7, 1], -0.5, &ad)),
            masses,
            GainInput {
                log_kappa_h: Tensor::<1>::full([1], 0.5f32.ln(), &ad).cast(dtype),
                noise_bsh: None,
                carry_bh: Tensor::<2>::full([1, 1], LOG_ZERO, &ad).cast(dtype),
            },
        );
        assert!(all_finite(out.log_precision_bsh.clone()), "{dtype:?}: ln Λ");
        let grads = (out.da_bsh.sum() + out.log_precision_bsh.sum()).backward();
        for (name, g) in [("Δ", x.val().grad(&grads)), ("λ", l.val().grad(&grads))] {
            assert!(all_finite(g.expect(name)), "{dtype:?}: the {name} gradient");
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The block
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Case {
    gain: Gain,
    tropical: Tropical,
    trapezoid: Trapezoid,
    u: usize,
    rotation: RotationKind,
    mimo: usize,
    norm: bool,
}

impl Case {
    fn config(self) -> Mamba3Config {
        Mamba3Config::new(16)
            .with_state_rank(4)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_ngroups(2)
            .with_gain(self.gain)
            .with_tropical(self.tropical)
            .with_trapezoid(self.trapezoid)
            .with_micro_steps(self.u)
            .with_rotation(self.rotation)
            .with_mimo_rank(self.mimo)
            .with_has_outproj_norm(self.norm)
    }
}

/// The lattice the positive systems must commute with, covered pairwise rather
/// than as a product: every gain, both registers, the tap patterns that change
/// the algorithm (none, lag 1, lag `u`, two taps, a gated one), `u ∈ {1, 2, 3}`,
/// all four rotation kinds, MIMO and the out-norm.
fn lattice() -> Vec<Case> {
    use Gain::*;
    use RotationKind::*;
    use Trapezoid as T;
    let case = |gain, tropical, trapezoid, u, rotation, mimo, norm| Case {
        gain,
        tropical,
        trapezoid,
        u,
        rotation,
        mimo,
        norm,
    };
    vec![
        case(Kalman, Tropical::None, T::HorizontalCarryOver, 1, Complex2D, 1, false),
        case(KalmanProjectedNoise, Tropical::MaxPlus, T::HorizontalCarryOver, 2, Real1D, 1, false),
        case(Kalman, Tropical::MaxPlus, T::Vertical, 2, Complex2D, 2, true),
        case(KalmanProjectedNoise, Tropical::None, T::VerticalPlusHorizontalCarryOver, 3, Quaternion4D, 1, false),
        case(Projected, Tropical::MaxPlus, T::None, 2, Real1D, 1, false),
        case(Kalman, Tropical::None, T::HorizontalReset, 2, Rotor4D, 2, false),
        case(KalmanProjectedNoise, Tropical::MaxPlus, T::None, 1, Real1D, 1, true),
    ]
}

/// Move the block off its (stock-adjacent) init so every port and every
/// in-projection channel matters: `κ` up to `O(1)`, `ω` and `e` non-zero.
fn exercised(mut model: Mamba3, device: &Device) -> Mamba3 {
    let nheads = model.nheads();
    model.kalman_log_kappa_h = model
        .kalman_log_kappa_h
        .map(|_| Param::from_tensor(uniform([nheads], -2.0, 1.0, device)));
    model.kalman_read_h = model
        .kalman_read_h
        .map(|_| Param::from_tensor(uniform([nheads], -1.0, 1.0, device)));
    model.tropical_readout_hp = model
        .tropical_readout_hp
        .map(|p| Param::from_tensor(uniform(p.dims(), -0.3, 0.3, device)));
    model
}

fn assert_slot(label: &str, name: &str, a: &Option<Tensor<2>>, b: &Option<Tensor<2>>, tol: f32) {
    assert_eq!(a.is_some(), b.is_some(), "{label}: {name} present in one cache only");
    if let (Some(a), Some(b)) = (a, b) {
        let d = max_abs_diff(a.clone(), b.clone());
        assert!(d <= tol, "{label}: {name} differs by {d}");
    }
}

fn assert_caches_match(label: &str, a: &Mamba3DoubleSsdCache, b: &Mamba3DoubleSsdCache) {
    let d = max_abs_diff(a.ssm_bhpr.clone(), b.ssm_bhpr.clone());
    assert!(d < 1e-4, "{label}: ssm state differs by {d}");
    if let (Some(a), Some(b)) = (&a.v_state_buhp, &b.v_state_buhp) {
        assert!(max_abs_diff(a.clone(), b.clone()) < 1e-4, "{label}: tap slots");
    }
    assert_slot(label, "ln Λ", &a.log_precision_bh, &b.log_precision_bh, 1e-4);
    assert_slot(label, "tropical c", &a.tropical_bh, &b.tropical_bh, 1e-3);
}

/// `forward` ≡ unrolled `step` ≡ the single-SSD pathway ≡ a split prefill ≡
/// the minimal SSD algorithm — outputs and every cache field, across the lattice.
#[test]
fn forward_step_pathways_and_prefill_agree_across_the_lattice() {
    let device: Device = Default::default();
    for case in lattice() {
        let label = format!("{case:?}");
        let config = case.config();
        let model = exercised(config.init(&device), &device);
        let (batch, tokens, split) = (2, 5, 3);
        let input = uniform([batch, tokens, config.d_model], -1.5, 1.5, &device);
        let path = Mamba3SsdPath::default();

        let (out, cache) = model.forward_double_ssd(input.clone(), None, &path, None);
        cache.sanity();

        let mut step_cache = None;
        let mut outs = Vec::new();
        for t in 0..tokens {
            let (o, c) =
                model.step_double_ssd(input.clone().narrow(1, t, 1).squeeze_dim(1), step_cache);
            outs.push(o.unsqueeze_dim::<3>(1));
            step_cache = Some(c);
        }
        let d = max_abs_diff(out.clone(), Tensor::cat(outs, 1));
        assert!(d < 1e-4, "{label}: forward vs step outputs {d}");
        assert_caches_match(&format!("{label} forward/step"), &cache, &step_cache.unwrap());

        let (out_single, cache_single) = model.forward_single_ssd(input.clone(), None, &path, None);
        let d = max_abs_diff(out.clone(), out_single);
        assert!(d < 1e-4, "{label}: double vs single outputs {d}");
        assert_caches_match(&format!("{label} double/single"), &cache, &cache_single.into());

        let (head, mid) = model.forward_single_ssd(input.clone().narrow(1, 0, split), None, &path, None);
        let (tail, last) =
            model.forward_single_ssd(input.clone().narrow(1, split, tokens - split), Some(mid), &path, None);
        let d = max_abs_diff(out.clone(), Tensor::cat(vec![head, tail], 1));
        assert!(d < 1e-4, "{label}: split prefill outputs {d}");
        assert_caches_match(&format!("{label} split prefill"), &cache, &last.into());

        let (out_minimal, _) =
            model.forward_double_ssd(input, None, &Mamba3SsdPath::Minimal(Some(2 * case.u)), None);
        let d = max_abs_diff(out, out_minimal);
        assert!(d < 1e-4, "{label}: SerialRecalculated vs Minimal outputs {d}");
    }
}

/// The third leg of forward ≡ step: backprop through the chunked pass and
/// through the unrolled recurrence agrees on the input, the in-projection
/// (whose tail carries the noise and register channels) and the positive
/// systems' own parameters.
#[test]
fn forward_and_step_gradients_agree() {
    let device: Device = Default::default();
    let ad = device.clone().autodiff();
    for case in [lattice()[1], lattice()[2], lattice()[3]] {
        let label = format!("{case:?}");
        let config = case.config();
        let model = exercised(config.init(&ad), &ad);
        let (batch, tokens) = (2, 4);
        let input = uniform([batch, tokens, config.d_model], -1.5, 1.5, &device);
        let head = Tensor::<3>::from_inner(uniform([batch, tokens, config.d_model], -1.0, 1.0, &device))
            .to_device(&ad);

        let p_fwd = Param::from_tensor(Tensor::<3>::from_inner(input.clone()).to_device(&ad));
        let (out, _) = model.forward(p_fwd.val(), None, Mamba3SsdPath::Minimal(None), None);
        let g_fwd = (out * head.clone()).sum().backward();

        let p_step = Param::from_tensor(Tensor::<3>::from_inner(input).to_device(&ad));
        let mut cache: Option<Mamba3Cache> = None;
        let mut outs = Vec::new();
        for t in 0..tokens {
            let (o, c) = model.step(p_step.val().narrow(1, t, 1).squeeze_dim::<2>(1), cache);
            outs.push(o.unsqueeze_dim::<3>(1));
            cache = Some(c);
        }
        let g_step = (Tensor::cat(outs, 1) * head).sum().backward();

        let check = |name: &str, a: Option<Tensor<3>>, b: Option<Tensor<3>>| {
            let d = max_abs_diff(a.expect(name), b.expect(name));
            assert!(d < 1e-2, "{label}: {name} gradient differs by {d}");
        };
        check(
            "input",
            p_fwd.val().grad(&g_fwd),
            p_step.val().grad(&g_step),
        );
        let w = model.in_proj.weight.val();
        check(
            "in_proj",
            w.clone().grad(&g_fwd).map(|g| g.unsqueeze::<3>()),
            w.grad(&g_step).map(|g| g.unsqueeze::<3>()),
        );
        if let Some(k) = &model.kalman_log_kappa_h {
            check(
                "ln κ",
                k.val().grad(&g_fwd).map(|g| g.unsqueeze::<3>()),
                k.val().grad(&g_step).map(|g| g.unsqueeze::<3>()),
            );
        }
        if let Some(w) = &model.kalman_read_h {
            check(
                "ω",
                w.val().grad(&g_fwd).map(|g| g.unsqueeze::<3>()),
                w.val().grad(&g_step).map(|g| g.unsqueeze::<3>()),
            );
        }
        if let Some(e) = &model.tropical_readout_hp {
            check(
                "e",
                e.val().grad(&g_fwd).map(|g| g.unsqueeze::<3>()),
                e.val().grad(&g_step).map(|g| g.unsqueeze::<3>()),
            );
        }
    }
}

/// The same block with its positive systems removed: the in-projection's
/// trailing (noise, register) rows dropped and the structural fields reset.
fn as_stock(model: &Mamba3) -> Mamba3 {
    let extra = model.noise_channels_total() + model.tropical_channels_total();
    let Linear { weight, bias } = model.in_proj.clone();
    let [d_model, width] = weight.dims();
    let in_proj = Linear {
        weight: Param::from_tensor(weight.val().narrow(1, 0, width - extra)),
        bias: bias.map(|b| Param::from_tensor(b.val().narrow(0, 0, width - extra))),
    };
    let _ = d_model;
    Mamba3 {
        in_proj,
        gain: Gain::Projected,
        tropical: Tropical::None,
        kalman_log_kappa_h: None,
        kalman_read_h: None,
        tropical_readout_hp: None,
        ..model.clone()
    }
}

/// At `κ = 0` (and the init's `ω = 0`, `e = 0`) the block **is** stock. For
/// the tied gain, whose in-projection is stock's, bit for bit on outputs and
/// state, through both pathways and `step`; the widened in-projections (noise,
/// register) round their shared rows through a wider matmul, so to 1e-6.
#[test]
fn zero_kappa_and_zero_readout_are_the_stock_block() {
    let device: Device = Default::default();
    use Trapezoid as T;
    for (gain, tropical, trapezoid, u, norm) in [
        (Gain::Kalman, Tropical::None, T::HorizontalCarryOver, 1, false),
        (Gain::Kalman, Tropical::None, T::Vertical, 2, true),
        (Gain::Kalman, Tropical::None, T::None, 2, false),
        (Gain::KalmanProjectedNoise, Tropical::MaxPlus, T::HorizontalCarryOver, 2, false),
        (Gain::Projected, Tropical::MaxPlus, T::None, 1, true),
    ] {
        let case = Case {
            gain,
            tropical,
            trapezoid,
            u,
            rotation: RotationKind::Complex2D,
            mimo: 1,
            norm,
        };
        let label = format!("{case:?}");
        let exact = !gain.projects_noise() && !tropical.is_on();
        let tol = if exact { 0.0 } else { 1e-6 };
        let mut model = case.config().init(&device);
        model.kalman_log_kappa_h = model
            .kalman_log_kappa_h
            .map(|p| Param::from_tensor(p.val().full_like(f32::NEG_INFINITY)));
        let stock = as_stock(&model);
        let input = uniform([2, 6, 16], -1.5, 1.5, &device);
        let path = Mamba3SsdPath::default();

        let (a, ca) = model.forward_double_ssd(input.clone(), None, &path, None);
        let (b, cb) = stock.forward_double_ssd(input.clone(), None, &path, None);
        assert!(max_abs_diff(a, b) <= tol, "{label}: double-SSD output");
        assert!(max_abs_diff(ca.ssm_bhpr, cb.ssm_bhpr) <= tol, "{label}: state");

        let (a, _) = model.forward_single_ssd(input.clone(), None, &path, None);
        let (b, _) = stock.forward_single_ssd(input.clone(), None, &path, None);
        assert!(max_abs_diff(a, b) <= tol, "{label}: single-SSD output");

        let x0 = input.narrow(1, 0, 1).squeeze_dim::<2>(1);
        let (a, _) = model.step(x0.clone(), None);
        let (b, _) = stock.step(x0, None);
        assert!(max_abs_diff(a, b) <= tol, "{label}: step output");
    }
}

/// The join is learnable from stock's side: at the init (`κ` small, `ω = 0`,
/// `e = 0`) every positive-system parameter receives gradient.
#[test]
fn the_join_has_live_gradients_at_init() {
    let device: Device = Default::default();
    let ad = device.clone().autodiff();
    // With the out-norm too: `ω` scales the SSD's share against the `D` skip,
    // which the norm keeps.
    for norm in [false, true] {
        let config = Case {
            gain: Gain::KalmanProjectedNoise,
            tropical: Tropical::MaxPlus,
            trapezoid: Trapezoid::HorizontalCarryOver,
            u: 1,
            rotation: RotationKind::Complex2D,
            mimo: 1,
            norm,
        }
        .config();
        let model = config.init(&ad);
        let input = Tensor::<3>::from_inner(uniform([2, 8, 16], -1.5, 1.5, &device)).to_device(&ad);
        let head = Tensor::<3>::from_inner(uniform([2, 8, 16], -1.0, 1.0, &device)).to_device(&ad);
        let (out, _) = model.forward(input, None, Mamba3SsdPath::default(), None);
        let grads = (out * head).sum().backward();
        let live = |name: &str, g: Option<f32>| {
            let g = g.unwrap_or_else(|| panic!("norm {norm}, {name}: no gradient"));
            assert!(g > 0.0, "norm {norm}, {name}: dead gradient at init");
        };
        let norm1 = |t: Option<Tensor<1>>| t.map(|g| g.abs().sum().into_scalar::<f32>());
        let norm2 = |t: Option<Tensor<2>>| t.map(|g| g.abs().sum().into_scalar::<f32>());
        live("ln κ", norm1(model.kalman_log_kappa_h.as_ref().unwrap().val().grad(&grads)));
        live("ω", norm1(model.kalman_read_h.as_ref().unwrap().val().grad(&grads)));
        live("e", norm2(model.tropical_readout_hp.as_ref().unwrap().val().grad(&grads)));
        // The noise rows are the in-projection's; they see `κ`'s gradient.
        let noise = model.noise_channels_total();
        let tropical = model.tropical_channels_total();
        let w_grad = model.in_proj.weight.val().grad(&grads).unwrap();
        let width = w_grad.dims()[1];
        live(
            "noise rows",
            Some(w_grad.narrow(1, width - tropical - noise, noise).abs().sum().into_scalar::<f32>()),
        );
    }
}

/// The block-level regression for [`an_underflowing_mass_keeps_the_gradient_finite`]:
/// heads whose `Δ` underflows at every position (`dt_bias` far past softplus's
/// range) still backpropagate finite gradients into every parameter.
#[test]
fn a_vanished_step_keeps_the_block_gradient_finite() {
    let device: Device = Default::default();
    let ad = device.clone().autodiff();
    let config = lattice()[1].config();
    let mut model = exercised(config.init(&ad), &ad);
    let bias: Vec<f32> = (0..model.nheads()).map(|h| if h % 2 == 0 { -200.0 } else { 0.0 }).collect();
    model.dt_bias_h = Param::from_tensor(Tensor::from_floats(bias.as_slice(), &ad));
    let input = Tensor::<3>::from_inner(uniform([2, 5, config.d_model], -1.5, 1.5, &device)).to_device(&ad);
    let (out, _) = model.forward(input, None, Mamba3SsdPath::default(), None);
    let grads = out.sum().backward();
    let finite = |name: &str, g: Option<bool>| assert!(g.expect(name), "the {name} gradient");
    finite("in_proj", model.in_proj.weight.val().grad(&grads).map(all_finite));
    finite("dt_bias", model.dt_bias_h.val().grad(&grads).map(all_finite));
    finite("ln κ", model.kalman_log_kappa_h.as_ref().unwrap().val().grad(&grads).map(all_finite));
    finite("ω", model.kalman_read_h.as_ref().unwrap().val().grad(&grads).map(all_finite));
}

/// A no-grad region moves every cache tensor to the inner backend and back by
/// hand ([`CacheStack`](burn_stack::modules::CacheStack)); the two new slots
/// must make the trip.
#[test]
fn the_slots_survive_a_no_grad_round_trip() {
    use crate::mamba3::cache::Mamba3Caches;
    use burn_stack::modules::CacheStack;
    let device: Device = Default::default();
    let ad = device.clone().autodiff();
    let config = lattice()[1].config();
    let model = exercised(config.init(&ad), &ad);
    let input = Tensor::<3>::from_inner(uniform([2, 3, 16], -1.5, 1.5, &device)).to_device(&ad);
    let (_, cache) = model.forward(input, None, Mamba3SsdPath::default(), None);
    let back = <Mamba3Caches as CacheStack>::cache_from_inner(
        <Mamba3Caches as CacheStack>::cache_to_inner(cache.clone()),
    );
    let (Mamba3Cache::SingleSsd(before), Mamba3Cache::SingleSsd(after)) = (cache, back) else {
        panic!("a missing cache is a single-SSD one");
    };
    assert_slot("no-grad trip", "ln Λ", &before.log_precision_bh, &after.log_precision_bh, 0.0);
    assert_slot("no-grad trip", "tropical c", &before.tropical_bh, &after.tropical_bh, 0.0);
    assert!(after.log_precision_bh.is_some() && after.tropical_bh.is_some());
}

/// The Muon plan splits the width of the in-projection segment for segment,
/// with the positive-system scalar channels on the fallback (tied and untied
/// tail alike).
#[cfg(feature = "optim")]
#[test]
fn the_muon_plan_covers_the_positive_channels() {
    use crate::mamba3::mamba3::Mamba3Untied;
    for untied in [vec![], vec![Mamba3Untied::InProjTail]] {
        let config = lattice()[1].config().with_untied(untied.clone());
        let specs = config.muon_projections();
        let in_proj_width: usize = specs
            .iter()
            .filter(|s| s.path.starts_with("in_proj"))
            .map(|s| s.width())
            .sum();
        assert_eq!(in_proj_width, config.d_in_proj(), "untied {untied:?}");
        let names: Vec<_> = specs
            .iter()
            .flat_map(|s| s.segments.iter())
            .filter(|seg| seg.name.starts_with("kalman") || seg.name.starts_with("tropical"))
            .collect();
        assert_eq!(names.len(), 3 * config.micro_steps, "untied {untied:?}");
        assert!(names.iter().all(|seg| !seg.muon), "the scalar channels stay on AdamW");
    }
}
