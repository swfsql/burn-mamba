//! A packed call is its segments run as one batch, one segment per row, from
//! the same incoming cache. Both calls start from one cache (made by an
//! earlier call, then made a leaf), broadcast to their rows. The test
//! compares:
//!
//! - the outputs of the real rows of each segment,
//! - every field of the cache that the packed call returns (that of the last
//!   segment of each packed row) against the row of that segment,
//! - the gradients of a loss over both: of every parameter, of the inputs,
//!   and of the fields of the incoming cache. Every segment sends its
//!   gradient to that cache, and none to the segment before it.

use crate::mamba2::prelude::*;
use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
use crate::mamba3::prelude::*;
use crate::mamba3::single_ssd::prelude::Mamba3SingleSsdCache;
use burn::module::{ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients, TensorData};
use burn_stack::utils::test_helpers::max_abs_diff;

const D_MODEL: usize = 16;
const VAL_TOL: f32 = 1e-4;
const GRAD_TOL: f32 = 1e-3;
/// Tokens of the call that makes the incoming cache.
const PREFIX: usize = 4;

/// The packed rows of each layout: the lengths of their segments, in tokens.
/// A length of `0` is a segment of pad rows only (one chunk).
///
/// - One packed row of three segments: the reference is a batch of 3.
/// - Two packed rows: a long segment, a short one (shorter than the conv
///   window), a pad-only one, and a one-token one.
const LAYOUTS: [&[&[usize]]; 2] = [&[&[5, 8, 3]], &[&[9, 2, 0], &[1, 6]]];

/// One segment: its packed row, its first token there, and its real tokens.
/// Its index in [`Layout::segments`] is its row in the reference batch.
struct Segment {
    row: usize,
    start: usize,
    len: usize,
}

/// A packed batch: each segment starts at a chunk start, and pad rows fill it
/// up to the next one. A row that is shorter than the longest one ends with
/// more pad rows.
struct Layout {
    segments: Vec<Segment>,
    rows: usize,
    tokens: usize,
    pad: Vec<bool>,
    reset: Vec<bool>,
    /// The index of the last segment of each packed row.
    last: Vec<usize>,
}

impl Layout {
    fn new(rows: &[&[usize]], chunk_tokens: usize) -> Self {
        let span = |len: usize| len.max(1).next_multiple_of(chunk_tokens);
        let tokens = rows
            .iter()
            .map(|lens| lens.iter().map(|&l| span(l)).sum::<usize>())
            .max()
            .unwrap();
        let (mut segments, mut pad, mut reset, mut last) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (row, lens) in rows.iter().enumerate() {
            let (mut row_pad, mut row_reset) = (vec![true; tokens], vec![false; tokens]);
            let mut start = 0;
            for &len in lens.iter() {
                segments.push(Segment { row, start, len });
                row_pad[start..start + len].fill(false);
                row_reset[start] = true;
                start += span(len);
            }
            last.push(segments.len() - 1);
            pad.extend(row_pad);
            reset.extend(row_reset);
        }
        Self {
            segments,
            rows: rows.len(),
            tokens,
            pad,
            reset,
            last,
        }
    }

    fn mask(&self, flags: &[bool], device: &Device) -> Tensor<2, Bool> {
        let data = TensorData::new(flags.to_vec(), [self.rows, self.tokens]);
        Tensor::from_data(data, device)
    }

    /// The longest segment (at least one token).
    fn max_len(&self) -> usize {
        self.segments.iter().map(|s| s.len).max().unwrap().max(1)
    }

    /// `[segments, max_len]`: the right padding of the reference batch.
    fn reference_pad(&self, device: &Device) -> Tensor<2, Bool> {
        let max_len = self.max_len();
        let flags: Vec<bool> = self
            .segments
            .iter()
            .flat_map(|s| (0..max_len).map(move |t| t >= s.len))
            .collect();
        Tensor::from_data(TensorData::new(flags, [self.segments.len(), max_len]), device)
    }
}

// ---------------------------------------------------------------------------
// The fields of a cache
// ---------------------------------------------------------------------------

/// A function applied to every tensor field of a cache.
trait FieldMap {
    fn map<const D: usize>(&self, t: Tensor<D>) -> Tensor<D>;
}

/// A reader of every tensor field of a cache.
trait FieldVisit {
    fn visit<const D: usize>(&mut self, t: &Tensor<D>);
}

trait TestCache: Clone {
    fn map(self, f: &impl FieldMap) -> Self;
    fn visit(&self, v: &mut impl FieldVisit);
}

impl TestCache for Mamba2Cache {
    fn map(self, f: &impl FieldMap) -> Self {
        Mamba2Cache {
            conv_bvk: f.map(self.conv_bvk),
            ssm_bhpr: f.map(self.ssm_bhpr),
        }
    }

    fn visit(&self, v: &mut impl FieldVisit) {
        v.visit(&self.conv_bvk);
        v.visit(&self.ssm_bhpr);
    }
}

/// The two Mamba-3 caches have the same fields.
macro_rules! impl_mamba3_cache {
    ($cache:ty) => {
        impl TestCache for $cache {
            fn map(self, f: &impl FieldMap) -> Self {
                Self {
                    ssm_bhpr: f.map(self.ssm_bhpr),
                    k_state_bumhr: self.k_state_bumhr.map(|t| f.map(t)),
                    v_state_buhp: self.v_state_buhp.map(|t| f.map(t)),
                    rotation: match self.rotation {
                        RotationState::Real(r) => RotationState::Real(r),
                        RotationState::Angle(t) => RotationState::Angle(f.map(t)),
                        RotationState::Quaternion(t) => RotationState::Quaternion(f.map(t)),
                        RotationState::Rotor(t) => RotationState::Rotor(f.map(t)),
                    },
                    log_precision_bh: self.log_precision_bh.map(|t| f.map(t)),
                    tropical_bh: self.tropical_bh.map(|t| f.map(t)),
                }
            }

            fn visit(&self, v: &mut impl FieldVisit) {
                v.visit(&self.ssm_bhpr);
                self.k_state_bumhr.iter().for_each(|t| v.visit(t));
                self.v_state_buhp.iter().for_each(|t| v.visit(t));
                match &self.rotation {
                    RotationState::Real(_) => {}
                    RotationState::Angle(t) => v.visit(t),
                    RotationState::Quaternion(t) | RotationState::Rotor(t) => v.visit(t),
                }
                self.log_precision_bh.iter().for_each(|t| v.visit(t));
                self.tropical_bh.iter().for_each(|t| v.visit(t));
            }
        }
    };
}
impl_mamba3_cache!(Mamba3DoubleSsdCache);
impl_mamba3_cache!(Mamba3SingleSsdCache);

/// A leaf that takes gradients: the incoming cache of the test.
struct Leaf;
impl FieldMap for Leaf {
    fn map<const D: usize>(&self, t: Tensor<D>) -> Tensor<D> {
        t.detach().require_grad()
    }
}

/// A batch of one, broadcast to `n` rows.
struct Broadcast(usize);
impl FieldMap for Broadcast {
    fn map<const D: usize>(&self, t: Tensor<D>) -> Tensor<D> {
        let mut dims = t.dims();
        dims[0] = self.0;
        t.expand(dims)
    }
}

/// Row `b` of a batch.
struct Row(usize);
impl FieldMap for Row {
    fn map<const D: usize>(&self, t: Tensor<D>) -> Tensor<D> {
        t.narrow(0, self.0, 1)
    }
}

/// Every field as `[batch, everything else]`.
struct Rows(Vec<Tensor<2>>);
impl FieldVisit for Rows {
    fn visit<const D: usize>(&mut self, t: &Tensor<D>) {
        let batch = t.dims()[0];
        let n = t.shape().num_elements() / batch;
        self.0.push(t.clone().reshape([batch, n]));
    }
}

/// The gradient of every field (a leaf), flattened.
struct Grads<'a>(&'a Gradients, Vec<Option<Tensor<1>>>);
impl FieldVisit for Grads<'_> {
    fn visit<const D: usize>(&mut self, t: &Tensor<D>) {
        let n = t.shape().num_elements();
        self.1.push(t.grad(self.0).map(|g| Tensor::from_inner(g).reshape([n])));
    }
}

fn rows<C: TestCache>(cache: &C) -> Vec<Tensor<2>> {
    let mut rows = Rows(Vec::new());
    cache.visit(&mut rows);
    rows.0
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

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

fn assert_grads_close(packed: Vec<Option<Tensor<1>>>, batched: Vec<Option<Tensor<1>>>, what: &str) {
    assert_eq!(packed.len(), batched.len());
    for (i, (p, b)) in packed.into_iter().zip(batched).enumerate() {
        match (p, b) {
            (Some(p), Some(b)) => assert_close(p, b, GRAD_TOL, &format!("{what} {i}: gradient")),
            (None, None) => {}
            (p, b) => panic!(
                "{what} {i}: gradient present packed {} / batched {}",
                p.is_some(),
                b.is_some()
            ),
        }
    }
}

/// Run a packed call and the batch of its segments from the same incoming
/// cache, and compare them (see the module header). `forward` takes `(input,
/// cache, pad, reset)`.
fn check_packed_is_batched<C: TestCache, M: Module>(
    module: &M,
    chunk_tokens: usize,
    forward: impl Fn(Tensor<3>, Option<C>, Option<Tensor<2, Bool>>, Option<Tensor<2, Bool>>) -> (Tensor<3>, C),
) {
    let device = Device::default().autodiff();
    let normal = Distribution::Normal(0.0, 1.0);
    for rows_spec in LAYOUTS {
        let layout = Layout::new(rows_spec, chunk_tokens);
        let nsegments = layout.segments.len();
        let max_len = layout.max_len();

        // The incoming cache: a real one, then a leaf.
        let x0 = Tensor::<3>::random([1, PREFIX, D_MODEL], normal, &device);
        let cache0 = forward(x0, None, None, None).1.map(&Leaf);

        // The reference batch holds the inputs. The packed rows are made of
        // them, so both calls send their input gradients to the same leaf.
        let x_ref = Tensor::<3>::random([nsegments, max_len, D_MODEL], normal, &device).require_grad();
        let filler = |len: usize| Tensor::<3>::random([1, len, D_MODEL], normal, &device);
        let x_packed = Tensor::cat(
            (0..layout.rows)
                .map(|row| {
                    let mut parts = Vec::new();
                    let mut end = 0;
                    for (i, s) in layout.segments.iter().enumerate().filter(|(_, s)| s.row == row) {
                        if s.start > end {
                            parts.push(filler(s.start - end));
                        }
                        if s.len > 0 {
                            parts.push(x_ref.clone().narrow(0, i, 1).narrow(1, 0, s.len));
                        }
                        end = s.start + s.len;
                    }
                    if layout.tokens > end {
                        parts.push(filler(layout.tokens - end));
                    }
                    Tensor::cat(parts, 1)
                })
                .collect(),
            0,
        );

        let (y_packed, cache_packed) = forward(
            x_packed,
            Some(cache0.clone().map(&Broadcast(layout.rows))),
            Some(layout.mask(&layout.pad, &device)),
            Some(layout.mask(&layout.reset, &device)),
        );
        let (y_ref, cache_ref) = forward(
            x_ref.clone(),
            Some(cache0.clone().map(&Broadcast(nsegments))),
            Some(layout.reference_pad(&device)),
            None,
        );

        let weight_y = Tensor::<3>::random([1, max_len, D_MODEL], normal, &device);
        let mut loss_packed = Tensor::<1>::zeros([1], &device);
        let mut loss_ref = Tensor::<1>::zeros([1], &device);
        for (i, s) in layout.segments.iter().enumerate().filter(|(_, s)| s.len > 0) {
            let what = format!("layout {rows_spec:?}, segment {i}'s outputs");
            let y_p = y_packed.clone().narrow(0, s.row, 1).narrow(1, s.start, s.len);
            let y_r = y_ref.clone().narrow(0, i, 1).narrow(1, 0, s.len);
            assert_close(y_p.clone(), y_r.clone(), VAL_TOL, &what);
            let weight = weight_y.clone().narrow(1, 0, s.len);
            loss_packed = loss_packed + (y_p * weight.clone()).sum();
            loss_ref = loss_ref + (y_r * weight).sum();
        }
        for (row, &i) in layout.last.iter().enumerate() {
            let fields_p = rows(&cache_packed.clone().map(&Row(row)));
            let fields_r = rows(&cache_ref.clone().map(&Row(i)));
            assert_eq!(fields_p.len(), fields_r.len());
            for (k, (f_p, f_r)) in fields_p.into_iter().zip(fields_r).enumerate() {
                let what = format!("layout {rows_spec:?}, packed row {row}'s cache field {k}");
                assert_close(f_p.clone(), f_r.clone(), VAL_TOL, &what);
                let weight = Tensor::<2>::random(f_r.dims(), normal, &device);
                loss_packed = loss_packed + (f_p * weight.clone()).sum();
                loss_ref = loss_ref + (f_r * weight).sum();
            }
        }

        let grads_packed = loss_packed.backward();
        let grads_ref = loss_ref.backward();
        let input_grad = |grads: &Gradients| {
            x_ref
                .grad(grads)
                .map(Tensor::from_inner)
                .unwrap_or_else(|| Tensor::zeros(x_ref.dims(), &device))
        };
        assert_close(
            input_grad(&grads_packed),
            input_grad(&grads_ref),
            GRAD_TOL,
            &format!("layout {rows_spec:?}: the gradient of the inputs"),
        );
        let cache_grads = |grads: &Gradients| {
            let mut visit = Grads(grads, Vec::new());
            cache0.visit(&mut visit);
            visit.1
        };
        assert_grads_close(
            cache_grads(&grads_packed),
            cache_grads(&grads_ref),
            &format!("layout {rows_spec:?}: incoming cache field"),
        );
        assert_grads_close(
            param_grads(module, &grads_packed),
            param_grads(module, &grads_ref),
            &format!("layout {rows_spec:?}: parameter"),
        );
    }
}

#[test]
fn packed_mamba2_is_the_batch_of_its_segments() {
    let device = Device::default().autodiff();
    let block = Mamba2Config::new(D_MODEL)
        .with_state_rank(8)
        .with_per_head_dim(8)
        .init(&device);
    for path in [Mamba2SsdPath::Serial(Some(4)), Mamba2SsdPath::SerialRecalculated(Some(4))] {
        check_packed_is_batched(&block, 4, |x, cache, pad, reset| {
            block.forward_packed(x, cache, path.clone(), pad, reset)
        });
    }
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

/// Both pathways and both serial paths, over a spread of the block's dials:
///
/// - every tap pattern family (none, lag 1, lag `u`, two taps, the token-gated
///   one),
/// - every rotation kind, `u > 1`, MIMO, and both positive systems.
#[test]
fn packed_mamba3_is_the_batch_of_its_segments() {
    use {Gain as G, RotationKind as R, Trapezoid as T, Tropical as Tr};
    let device = Device::default().autodiff();
    let configs = [
        mamba3(R::Complex2D, T::HorizontalCarryOver, 1, 1, G::Projected, Tr::None),
        mamba3(R::Quaternion4D, T::Vertical, 3, 2, G::Kalman, Tr::MaxPlus),
        mamba3(R::Rotor4D, T::VerticalPlusHorizontalCarryOver, 2, 1, G::KalmanProjectedNoise, Tr::None),
        mamba3(R::Real1D, T::None, 2, 2, G::Projected, Tr::MaxPlus),
        mamba3(R::Complex2D, T::HorizontalReset, 3, 1, G::Kalman, Tr::None),
    ];
    let paths = [Mamba3SsdPath::Serial(Some(4)), Mamba3SsdPath::SerialRecalculated(Some(4))];
    for config in &configs {
        let block = config.init(&device);
        for path in &paths {
            let chunk_tokens =
                Mamba3SsdPath::chunk_tokens(path.chunk_len_or_optimal(&block), block.micro_steps);
            check_packed_is_batched(&block, chunk_tokens, |x, cache, pad, reset| {
                block.forward_double_ssd_packed(x, cache, path, pad, reset)
            });
            check_packed_is_batched(&block, chunk_tokens, |x, cache, pad, reset| {
                block.forward_single_ssd_packed(x, cache, path, pad, reset)
            });
        }
    }
}

/// [`restart_heads`](crate::packing::restart_heads) puts the head at the start
/// of every reset chunk, so each head entry gets the sum of the gradients of
/// its copies. The check is by hand, on the host: a tiling by `repeat_dim`
/// has the right forward but a wrong backward in this Burn.
#[test]
fn restart_heads_gradient_sums_over_the_chunks() {
    let device = Device::default().autodiff();
    let (width, chunk_len, nchunks) = (3, 4, 3);
    let len = chunk_len * nchunks;
    let head = Tensor::<3>::random([1, 2, width], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let x = Tensor::<3>::zeros([1, 2, len], &device);
    // Chunks 0 and 2 restart, chunk 1 does not.
    let reset_bn = Tensor::<2, Bool>::from_data(TensorData::new(vec![true, false, true], [1, nchunks]), &device);
    let y = crate::packing::restart_heads(x, 2, head.clone(), &reset_bn, chunk_len);
    let w = Tensor::<3>::random([1, 2, len], Distribution::Normal(0.0, 1.0), &device);
    let grads = (y * w.clone()).sum().backward();
    let got = Tensor::<3>::from_inner(head.grad(&grads).unwrap());
    let w_host: Vec<f32> = w.to_data().try_to_vec().unwrap();
    let want: Vec<f32> = (0..2)
        .flat_map(|v| (0..width).map(move |j| (v, j)))
        .map(|(v, j)| [0, 2].iter().map(|c| w_host[v * len + c * chunk_len + j]).sum())
        .collect();
    let want = Tensor::<3>::from_data(TensorData::new(want, [1, 2, width]), &device);
    assert_close(got, want, 1e-6, "the gradient of the head");
}

/// `Minimal` builds its inter-chunk decay as one `nchunks × nchunks` matrix
/// and takes no resets.
#[test]
#[should_panic(expected = "Minimal does not take resets")]
fn packed_minimal_panics() {
    let device = Device::default();
    let block = Mamba2Config::new(D_MODEL)
        .with_state_rank(8)
        .with_per_head_dim(8)
        .init(&device);
    let x = Tensor::<3>::zeros([1, 8, D_MODEL], &device);
    let reset = Tensor::<2, Int>::zeros([1, 8], &device).bool();
    let _ = block.forward_packed(x, None, Mamba2SsdPath::Minimal(Some(4)), None, Some(reset));
}

/// A reset inside a chunk would need the masks of the official kernels inside
/// the chunk. So the block refuses it.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a reset must be at the first token of a chunk")]
fn reset_inside_a_chunk_panics() {
    let device = Device::default();
    let block = Mamba2Config::new(D_MODEL)
        .with_state_rank(8)
        .with_per_head_dim(8)
        .init(&device);
    let x = Tensor::<3>::zeros([1, 8, D_MODEL], &device);
    let reset = Tensor::<1, Int>::arange(0..8, &device)
        .equal_elem(2)
        .reshape([1, 8]);
    let _ = block.forward_packed(x, None, Mamba2SsdPath::Serial(Some(4)), None, Some(reset));
}
