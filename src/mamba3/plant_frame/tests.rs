use super::*;
use crate::mamba3::prelude::*;
use burn::module::Param;
use burn::tensor::{Distribution, TensorData};
use burn_stack::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};

// ---------------------------------------------------------------------------
// The split of the in-projection
// ---------------------------------------------------------------------------

/// The optional channel groups that a config projects.
#[derive(Clone, Copy, Debug)]
struct Optional {
    lambda: bool,
    mu: bool,
    rotation: bool,
    noise: bool,
    tropical: bool,
}

/// One entry of the layout in the doc of [`Mamba3::in_proj`].
#[derive(Clone, Copy)]
struct Group {
    name: &'static str,
    /// The channels of one micro-step, or of the token for a per-token group.
    width: usize,
    per_micro_step: bool,
}

/// The layout in the doc of [`Mamba3::in_proj`], without the optional groups
/// that the config does not project.
fn documented_layout(model: &Mamba3, on: Optional) -> Vec<Group> {
    let d_inner = model.d_inner();
    let nheads = model.nheads();
    let bc = model.mimo_rank * model.ngroups * model.state_rank;
    let group = |name, width, per_micro_step| Group { name, width, per_micro_step };
    let mut layout = vec![
        group("z", d_inner, false),
        group("x", d_inner, true),
        group("b_raw", bc, true),
        group("c_raw", bc, false),
        group("dd_dt", nheads, true),
        group("dd_a_raw", nheads, true),
    ];
    if on.lambda {
        layout.push(group("lambda_raw", nheads, true));
    }
    if on.mu {
        layout.push(group("mu_raw", nheads, true));
    }
    if on.rotation {
        layout.push(group("rotation", model.num_rotation_channels, true));
    }
    if on.noise {
        layout.push(group("kalman_noise", nheads, true));
    }
    if on.tropical {
        layout.push(group("tropical_a", nheads, true));
        layout.push(group("tropical_b", nheads, true));
    }
    layout
}

/// The marker of a group at one token and micro-step. All values are exact in
/// fp16.
fn marker(token: usize, group: usize, micro: usize) -> f32 {
    (256 * token + 16 * group + micro) as f32
}

/// A one-row in-projection output with a distinct marker in each group.
fn marked_projection(layout: &[Group], tokens: usize, u: usize, device: &Device) -> Tensor<3> {
    let mut values = Vec::new();
    for token in 0..tokens {
        for (g, group) in layout.iter().enumerate() {
            for micro in 0..if group.per_micro_step { u } else { 1 } {
                values.extend(std::iter::repeat_n(marker(token, g, micro), group.width));
            }
        }
    }
    let width = values.len() / tokens;
    Tensor::from_data(TensorData::new(values, [1, tokens, width]), device)
}

/// Check that a stream holds the markers of the group with this name.
fn assert_stream(label: &str, layout: &[Group], name: &str, stream: Tensor<3>, tokens: usize, u: usize) {
    let (g, group) = layout
        .iter()
        .enumerate()
        .find(|(_, group)| group.name == name)
        .unwrap_or_else(|| panic!("{label}: the layout has no group {name}"));
    let len = if group.per_micro_step { tokens * u } else { tokens };
    assert_eq!([1, len, group.width], stream.dims(), "{label}: shape of {name}");
    let values = stream.into_data().try_into_vec_as::<f32>().unwrap();
    for position in 0..len {
        let (token, micro) = if group.per_micro_step {
            (position / u, position % u)
        } else {
            (position, 0)
        };
        for channel in 0..group.width {
            let got = values[position * group.width + channel];
            let want = marker(token, g, micro);
            assert_eq!(want, got, "{label}: {name}, position {position}, channel {channel}");
        }
    }
}

fn assert_presence(label: &str, name: &str, present: bool, expected: bool) {
    assert_eq!(expected, present, "{label}: presence of {name}");
}

fn assert_split(label: &str, cfg: Mamba3Config, on: Optional) {
    let device = test_device();
    let tokens = 3;
    let model = cfg.init(&device);
    let u = model.micro_steps;
    let layout = documented_layout(&model, on);
    let proj_btd = marked_projection(&layout, tokens, u, &device);
    assert_eq!(cfg.d_in_proj(), proj_btd.dims()[2], "{label}: the doc and the config disagree");

    let s = model.split_in_proj(proj_btd);
    let check = |name: &str, stream: Tensor<3>| assert_stream(label, &layout, name, stream, tokens, u);
    check("z", s.z_bti);
    check("x", s.x_bsi);
    check("b_raw", s.b_raw_bsMGR);
    check("c_raw", s.c_raw_btMGR);
    check("dd_dt", s.dd_dt_bsh);
    check("dd_a_raw", s.dd_a_raw_bsh);
    assert_presence(label, "lambda_raw", s.lambda_raw_bsh.is_some(), on.lambda);
    assert_presence(label, "mu_raw", s.mu_raw_bsh.is_some(), on.mu);
    assert_presence(label, "rotation", s.rot_bsa.is_some(), on.rotation);
    assert_presence(label, "kalman_noise", s.noise_bsh.is_some(), on.noise);
    assert_presence(label, "tropical", s.tropical_ab_bsh.is_some(), on.tropical);
    s.lambda_raw_bsh.map(|t| check("lambda_raw", t));
    s.mu_raw_bsh.map(|t| check("mu_raw", t));
    s.rot_bsa.map(|t| check("rotation", t));
    s.noise_bsh.map(|t| check("kalman_noise", t));
    s.tropical_ab_bsh.map(|(a, b)| {
        check("tropical_a", a);
        check("tropical_b", b);
    });
}

/// A config that projects each optional group, at `u = 2`.
fn all_on() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(16)
        .with_ngroups(2)
        .with_mimo_rank(2)
        .with_micro_steps(2)
        .with_tap_pattern(TapPattern::VerticalPlusHorizontalCarryOver)
        .with_gain(Gain::KalmanProjectedNoise)
        .with_tropical(Tropical::MaxPlus)
}

const ALL: Optional = Optional { lambda: true, mu: true, rotation: true, noise: true, tropical: true };

#[test]
fn split_with_all_optional_groups() {
    assert_split("all on, u = 2", all_on(), ALL);
    assert_split("all on, u = 3", all_on().with_micro_steps(3), ALL);
}

#[test]
fn split_without_lambda() {
    // `TapPattern::None` has no `λ`, and so no `μ`.
    let cfg = all_on().with_tap_pattern(TapPattern::None);
    assert_split("no lambda", cfg, Optional { lambda: false, mu: false, ..ALL });
}

#[test]
fn split_without_mu() {
    let cfg = all_on().with_tap_pattern(TapPattern::HorizontalCarryOver);
    assert_split("no mu", cfg, Optional { mu: false, ..ALL });
}

#[test]
fn split_without_rotation() {
    let cfg = all_on().with_rotation(RotationKind::Real1D);
    assert_split("no rotation", cfg, Optional { rotation: false, ..ALL });
}

#[test]
fn split_without_noise() {
    let cfg = all_on().with_gain(Gain::Kalman);
    assert_split("no noise", cfg, Optional { noise: false, ..ALL });
}

#[test]
fn split_without_tropical() {
    let cfg = all_on().with_tropical(Tropical::None);
    assert_split("no tropical", cfg, Optional { tropical: false, ..ALL });
}

#[test]
fn split_at_one_micro_step() {
    // At `u = 1`, a two-tap pattern folds and projects no `μ`.
    let cfg = all_on().with_micro_steps(1);
    assert_split("u = 1", cfg, Optional { mu: false, ..ALL });
}

// ---------------------------------------------------------------------------
// The Muon plan of the in-projection
// ---------------------------------------------------------------------------

/// The groups that Muon updates. Each other group uses the fallback optimizer.
#[cfg(feature = "optim")]
const MUON_GROUPS: [&str; 5] = ["z", "x", "b_raw", "c_raw", "rotation"];

/// The groups that `in_proj` keeps under [`Mamba3Untied::InProjTail`].
#[cfg(feature = "optim")]
const TIED_GROUPS: [&str; 4] = ["z", "x", "b_raw", "c_raw"];

/// The sub-projections of a part of the layout: one for each micro-step of a
/// per-micro-step group, and one for a per-token group.
#[cfg(feature = "optim")]
fn expected_sub_projections(layout: &[Group], u: usize) -> Vec<burn_stack::optim::SubProjection> {
    use burn_stack::optim::SubProjection;
    layout
        .iter()
        .flat_map(|group| {
            let sub = if MUON_GROUPS.contains(&group.name) {
                SubProjection::muon(group.name, group.width)
            } else {
                SubProjection::adamw(group.name, group.width)
            };
            std::iter::repeat_n(sub, if group.per_micro_step { u } else { 1 })
        })
        .collect()
}

/// The `(path, tiled, sub-projections)` of the specs of the in-projection.
#[cfg(feature = "optim")]
fn in_proj_specs(cfg: &Mamba3Config) -> Vec<(String, bool, Vec<burn_stack::optim::SubProjection>)> {
    cfg.muon_projections()
        .into_iter()
        .filter(|spec| spec.path.starts_with("in_proj"))
        .map(|spec| (spec.path, spec.tiled, spec.sub_projections))
        .collect()
}

/// Compare the Muon plan with the documented layout, entry by entry: the order,
/// the name, the width and the Muon flag. The tied and the untied tail both
/// get the check.
#[cfg(feature = "optim")]
fn assert_muon_plan(label: &str, cfg: Mamba3Config, on: Optional) {
    let model = cfg.init(&test_device());
    let u = model.micro_steps;
    let layout = documented_layout(&model, on);

    let tied = vec![("in_proj.weight".to_string(), false, expected_sub_projections(&layout, u))];
    assert_eq!(tied, in_proj_specs(&cfg), "{label}: tied tail");

    let (always_tied, tail): (Vec<Group>, Vec<Group>) =
        layout.iter().copied().partition(|group| TIED_GROUPS.contains(&group.name));
    let untied = vec![
        ("in_proj.weight".to_string(), false, expected_sub_projections(&always_tied, u)),
        ("in_proj_tail.weight".to_string(), true, expected_sub_projections(&tail, u)),
    ];
    let untied_cfg = cfg.with_untied(vec![Mamba3Untied::InProjTail]);
    assert_eq!(untied, in_proj_specs(&untied_cfg), "{label}: untied tail");
}

#[cfg(feature = "optim")]
#[test]
fn muon_plan_with_all_optional_groups() {
    assert_muon_plan("all on, u = 2", all_on(), ALL);
    assert_muon_plan("all on, u = 3", all_on().with_micro_steps(3), ALL);
}

#[cfg(feature = "optim")]
#[test]
fn muon_plan_without_each_optional_group() {
    let no_lambda = all_on().with_tap_pattern(TapPattern::None);
    assert_muon_plan("no lambda", no_lambda, Optional { lambda: false, mu: false, ..ALL });
    let no_mu = all_on().with_tap_pattern(TapPattern::HorizontalCarryOver);
    assert_muon_plan("no mu", no_mu, Optional { mu: false, ..ALL });
    let no_rotation = all_on().with_rotation(RotationKind::Real1D);
    assert_muon_plan("no rotation", no_rotation, Optional { rotation: false, ..ALL });
    let no_noise = all_on().with_gain(Gain::Kalman);
    assert_muon_plan("no noise", no_noise, Optional { noise: false, ..ALL });
    let no_tropical = all_on().with_tropical(Tropical::None);
    assert_muon_plan("no tropical", no_tropical, Optional { tropical: false, ..ALL });
}

#[cfg(feature = "optim")]
#[test]
fn muon_plan_at_one_micro_step() {
    // At `u = 1`, a two-tap pattern folds and projects no `μ`.
    assert_muon_plan("u = 1", all_on().with_micro_steps(1), Optional { mu: false, ..ALL });
}

// ---------------------------------------------------------------------------
// The tail
// ---------------------------------------------------------------------------

fn random<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::random(dims, Distribution::Normal(0.0, 1.0), device)
}

fn host<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.into_data().try_into_vec_as::<f64>().unwrap()
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// The tail, against its definition written as a loop over the indices:
///
/// ```text
///   yₘ[h, p]  = y[m, h, p] + D[h] · x[h, p] · mimo_x[h, m, p]
///   gₘ[h, p]  = yₘ[h, p] · silu(z[h, p] · mimo_z[h, m, p])
///               (with `out_norm`: yₘ / rms(yₘ[h, ·]) · γ[p] · silu(…))
///   out[h, p] = Σₘ mimo_o[h, m, p] · gₘ[h, p]
///   result[k] = Σᵢ out[i] · W[i, k] + bias[k]
/// ```
///
/// A SISO block has no `mimo_*` weights, and each is `1`. The `ε` of the RMS is
/// about `1e-7`, against `mean(y²) ≈ 1`, so the loop omits it. `x` is read
/// at the last micro-step of each token.
fn assert_tail(mimo_rank: usize, out_norm: bool, u: usize) {
    let label = format!("mimo_rank = {mimo_rank}, out_norm = {out_norm}, u = {u}");
    let device = test_device();
    let mut model = Mamba3Config::new(8)
        .with_state_rank(4)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_mimo_rank(mimo_rank)
        .with_micro_steps(u)
        .with_has_outproj_norm(out_norm)
        .init(&device);
    let (nheads, per_head) = (model.nheads(), model.per_head_dim());
    let (d_inner, d_model) = (model.d_inner(), model.out_proj.weight.dims()[1]);
    // Random parameters, so that each index matters.
    model.d_h = Param::from_tensor(random([nheads], &device));
    let hmp = [nheads, mimo_rank, per_head];
    model.mimo_x_hmp = model.mimo_x_hmp.map(|_| Param::from_tensor(random(hmp, &device)));
    model.mimo_z_hmp = model.mimo_z_hmp.map(|_| Param::from_tensor(random(hmp, &device)));
    model.mimo_o_hmp = model.mimo_o_hmp.map(|_| Param::from_tensor(random(hmp, &device)));
    if let Some(norm) = model.out_norm.as_mut() {
        norm.gamma = Param::from_tensor(random([per_head], &device));
    }

    let (batch, tokens) = (2, 3);
    let y_btmhp = random([batch, tokens, mimo_rank, nheads, per_head], &device);
    let x_bshp = random([batch, tokens * u, nheads, per_head], &device);
    let z_bti = random([batch, tokens, d_inner], &device);
    let b_last_bumhr = random([batch, 1, mimo_rank, nheads, 4], &device);
    let x_last_buhp = random([batch, 1, nheads, per_head], &device);
    let angle_bha = random([batch, nheads, model.num_rotation_angles], &device);
    let ssm_bhpr = random([batch, nheads, per_head, 4], &device);

    let (out_btd, cache) = model.plant_tail(
        PlantTail {
            y_btmhp: y_btmhp.clone(),
            z_bti: z_bti.clone(),
            x_bshp: x_bshp.clone(),
            log_precision_bsh: None,
            tropical_ab_bsh: None,
            end_b: None,
            packed: None,
            new_rotation: RotationState::Angle(angle_bha.clone()),
            new_tap_fifo: Some(TapFifo { b_bumhr: b_last_bumhr.clone(), x_buhp: x_last_buhp.clone() }),
        },
        Mamba3DoubleSsdCache {
            ssm_bhpr: ssm_bhpr.clone(),
            tap_fifo: None,
            rotation: RotationState::Angle(angle_bha.clone().zeros_like()),
            log_precision_bh: None,
            tropical_bh: None,
        },
    );

    let (y, x, z) = (host(y_btmhp), host(x_bshp), host(z_bti));
    let d = host(model.d_h.val());
    let weight = |p: &Option<Param<Tensor<3>>>| p.as_ref().map(|p| host(p.val()));
    let (mimo_x, mimo_z, mimo_o) =
        (weight(&model.mimo_x_hmp), weight(&model.mimo_z_hmp), weight(&model.mimo_o_hmp));
    let gamma = model.out_norm.as_ref().map(|norm| host(norm.gamma.val()));
    let w = host(model.out_proj.weight.val());
    let bias = model.out_proj.bias.as_ref().map(|b| host(b.val()));
    let at_hmp = |t: &Option<Vec<f64>>, h: usize, m: usize, p: usize| {
        t.as_ref().map_or(1.0, |t| t[(h * mimo_rank + m) * per_head + p])
    };

    let mut want = Vec::with_capacity(batch * tokens * d_model);
    for row in 0..batch {
        for token in 0..tokens {
            let read = token * u + u - 1;
            let mut out_i = vec![0.0; d_inner];
            for h in 0..nheads {
                for m in 0..mimo_rank {
                    let y_p: Vec<f64> = (0..per_head)
                        .map(|p| {
                            let y_at = ((row * tokens + token) * mimo_rank + m) * nheads + h;
                            let x_at = ((row * tokens * u + read) * nheads + h) * per_head + p;
                            y[y_at * per_head + p] + d[h] * x[x_at] * at_hmp(&mimo_x, h, m, p)
                        })
                        .collect();
                    let rms = (y_p.iter().map(|v| v * v).sum::<f64>() / per_head as f64).sqrt();
                    for p in 0..per_head {
                        let i = h * per_head + p;
                        let z_m = z[(row * tokens + token) * d_inner + i] * at_hmp(&mimo_z, h, m, p);
                        let value = match &gamma {
                            Some(gamma) => y_p[p] / rms * gamma[p],
                            None => y_p[p],
                        };
                        out_i[i] += at_hmp(&mimo_o, h, m, p) * value * silu(z_m);
                    }
                }
            }
            for k in 0..d_model {
                let sum: f64 = (0..d_inner).map(|i| out_i[i] * w[i * d_model + k]).sum();
                want.push(sum + bias.as_ref().map_or(0.0, |b| b[k]));
            }
        }
    }
    let want: Vec<f32> = want.into_iter().map(|v| v as f32).collect();
    let want_btd = Tensor::<3>::from_data(TensorData::new(want, [batch, tokens, d_model]), &device);
    let diff = max_rel_diff(out_btd, want_btd);
    assert!(diff < dtype_tol(1e-4), "{label}: the tail differs from its definition by {diff}");

    // The tail fills the fields of the next call, and keeps the state of the scan.
    assert_eq!(host(ssm_bhpr), host(cache.ssm_bhpr), "{label}: state");
    let fifo = cache.tap_fifo.unwrap();
    assert_eq!(host(b_last_bumhr), host(fifo.b_bumhr), "{label}: tap FIFO B");
    assert_eq!(host(x_last_buhp), host(fifo.x_buhp), "{label}: tap FIFO x");
    assert_eq!(host(angle_bha), host(cache.rotation.angle()), "{label}: rotation");
}

#[test]
fn tail_siso() {
    assert_tail(1, false, 1);
    assert_tail(1, false, 2);
}

#[test]
fn tail_siso_with_out_norm() {
    assert_tail(1, true, 1);
    assert_tail(1, true, 2);
}

#[test]
fn tail_mimo() {
    assert_tail(2, false, 1);
    assert_tail(3, false, 2);
}

#[test]
fn tail_mimo_with_out_norm() {
    assert_tail(2, true, 1);
    assert_tail(3, true, 2);
}
