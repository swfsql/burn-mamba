//! Shared helpers used by both [`Mamba3::forward`](super::mamba3::Mamba3::forward)
//! and [`Mamba3::step`](super::mamba3::Mamba3::step). They isolate three blocks
//! that previously appeared in both methods at different ranks:
//!
//! 1. Trapezoidal discretisation: `dt`, `α`, `β`, `γ`, `da`.
//! 2. QK-norm + GQA expansion + per-(head, mimo-rank) bias on B / C.
//! 3. MIMO `V` construction: broadcast-multiply `x` by `mimo_x_hmp`.
//! 4. The rank-summed outer product `Σₘ v[m] ⊗ k[m]` feeding the SSM state
//!    (SISO-branched).
//! 5. Peeling the rotation channels off the in-projection.
//! 6. The trapezoid tap's lag arithmetic: the shift, the gap transport, the
//!    decay a cached slot carries across a call boundary, and the per-position
//!    gate a tap pattern admits its taps by.
//! 7. [`prefix_sum`]: the log-depth inclusive scan every sequence-length
//!    cumulative sum goes through instead of `Tensor::cumsum`.
//!
//! Most helpers are generic over the rank `D` of the data tensors so a single
//! definition serves both the sequence-aware (`forward`) and single-token
//! (`step`) code paths. The discretisation is the exception: MambaProduct gives
//! `step` a `u` axis of its own, so both paths reach it at rank 3.

use crate::mamba3::trapezoid::TrapezoidSpec;
use burn_stack::modules::RmsNorm;
use burn_stack::modules::gqa_expand_to_heads;
use burn_stack::modules::softplus;
use burn::prelude::*;

/// Peel a **trailing** in-projection segment of `width` channels off `proj`,
/// yielding `None` for it when the block projects none.
///
/// Used for the two segments a block may omit entirely — the rotation channels
/// ([`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D)
/// projects none) and then `λ`
/// ([`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) projects
/// none) — which is why the in-projection lays them out last, in that order.
///
/// An optional slice cannot simply be one more entry in the main `split_into`:
/// Burn has no zero-width tensors, and `split_with_sizes` *drops* a zero-length
/// segment rather than returning an empty one, so the destructuring would come
/// up one part short.
///
/// # Shapes
/// - `proj` : `[..., w]` along `dim`
/// - out    : `[..., w − width]` and, if any, `[..., width]`
pub fn split_trailing<const D: usize>(
    proj: Tensor<D>,
    width: usize,
    dim: usize,
) -> (Tensor<D>, Option<Tensor<D>>) {
    if width == 0 {
        return (proj, None);
    }
    let rest = proj.dims()[dim] - width;
    let tail = proj.clone().narrow(dim, rest, width);
    (proj.narrow(dim, 0, rest), Some(tail))
}

/// Shift a per-position stream back by `lag`, seeding the first `lag` positions
/// from the cache's tap slots: `out[p] = stream[p − lag]`, and `prefix[p]` where
/// that index is before the call.
///
/// This is the "shift-before-chunking" of the double-SSD pathway generalised
/// from lag 1 to [`Trapezoid::tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag).
/// A folded sequence is `tokens · u` long and `lag ∈ {1, u}`, so `sequence ≥
/// lag` always holds; at equality the whole call is prefix.
///
/// # Shapes
/// - `stream` : `[batch, sequence, …]`
/// - `prefix` : `[batch, lag, …]`
/// - out      : `[batch, sequence, …]`
pub fn shift_stream<const D: usize>(
    stream: Tensor<D>,
    prefix: Tensor<D>,
    lag: usize,
) -> Tensor<D> {
    let sequence = stream.dims()[1];
    assert_eq!(prefix.dims()[1], lag, "one prefix slot per lagged position");
    assert!(sequence >= lag, "a call is at least one token, i.e. `lag` long");
    if sequence == lag {
        prefix
    } else {
        Tensor::cat(vec![prefix, stream.narrow(1, 0, sequence - lag)], 1)
    }
}

/// The in-block length [`prefix_sum`] scans directly.
///
/// A `cumsum` of length `L` over `N` elements costs `∝ N·L` (below), so a
/// blocked scan costs `∝ N·(block + len/block²)` — one pass inside each block
/// plus one over the `len/block` block totals. That is least at
/// `block = ∛(2·len)`, which this walks up to on a power-of-two grid. The floor
/// of 16 keeps the fixed cost (a reshape, a subtract, a broadcast add) amortised
/// on short axes; the ceiling bounds the in-block quadratic.
///
/// `scan_block(len) < len` is exactly "this length takes the blocked branch",
/// which the parity tests assert so that re-tuning the rule cannot silently stop
/// covering it.
pub(crate) fn scan_block(len: usize) -> usize {
    let mut block = 16;
    while block < 256 && block * block * block < 2 * len {
        block *= 2;
    }
    block
}

/// Inclusive prefix sum along `dim`, continued from `init`:
/// `out[i] = init + Σ_{j ≤ i} t[j]`.
///
/// The values [`Tensor::cumsum`] computes, but **blocked**: the scanned axis is
/// split into runs of [`scan_block`], each run scanned on its own, then offset
/// by the exclusive prefix of the run totals. Both of those scans are short and
/// bounded, and the whole thing is a handful of ops whatever `len` is.
///
/// `init` is the scan's carry-in — for the cumulative rotation angle, the one
/// the cache brings from the previous call, exactly as `quat_cumprod` takes its
/// `init` quaternion. It rides the block offset rather than costing a pass of
/// its own, which is what keeps this at parity with a plain `cumsum` **plus the
/// caller's add** on a backend whose `cumsum` is already linear.
///
/// It exists because Burn's `cumsum` is **quadratic in the scanned length** on
/// the cubecl backends: measured on CUDA over `[2, len, 32, 32]` along `dim 1`,
/// 0.27 ms at `len = 256` rising to 143 ms at 4096, ≈4× per doubling — and the
/// same scan on the trailing, contiguous axis is only 2.6× cheaper and still
/// ≈4× per doubling, so it is the scan and not the stride. Every other scan in
/// this crate runs over `chunk_len`, `lag` or `micro_steps`, all bounded, which
/// is the regime this restores the angle scan to: the cumulative rotation angle
/// is the one that runs over the whole **folded** sequence, and `micro_steps`
/// multiplies that axis, so `u > 1` reaches the quadratic `u`× sooner. Blocked,
/// the same case is 1.9× / 6.7× / 23× / 46× faster at `len = 256 / 512 / 1024 /
/// 2048`, and the forward is near-flat in `len` (0.14 → 0.51 ms over that span).
///
/// The non-abelian sibling ([`quat_scan`](crate::mamba3::quat_scan)) solves the
/// same problem with a Hillis–Steele doubling instead, having no `cumprod` to
/// block with. Doubling works here too, and blocking beats it at **every** length
/// measured (256 … 8192): 4.8× → 6.6× on CUDA forward, 2.3× → 3.3× with the
/// backward, and 15× → 41× on the CPU backend. `O(len·log len)` work in
/// `3·⌈log₂ len⌉` full-tensor kernels loses to `O(len·∛len)` in six, on hardware
/// that is bandwidth- and launch-bound rather than FLOP-bound. So there is
/// nothing here for a runtime knob to pick between.
///
/// The additions associate differently from a sequential scan, so the two agree
/// to rounding rather than bit-for-bit.
///
/// # Shapes
/// - `t`    : any rank, scanned along `dim`; `DP1` is `D + 1`
/// - `init` : `t`'s shape with `1` along `dim` (it broadcasts along it)
/// - out    : `t`'s shape
pub fn prefix_sum<const D: usize, const DP1: usize>(
    t: Tensor<D>,
    dim: usize,
    init: Option<Tensor<D>>,
) -> Tensor<D> {
    let len = t.dims()[dim];
    let block = scan_block(len);
    if len <= block {
        let scanned = t.cumsum(dim);
        return match init {
            Some(init) => scanned + init,
            None => scanned,
        };
    }
    let device = t.device();

    // Pad up to a whole number of blocks. Zero is the additive identity, so the
    // padding contributes to no prefix, and it is narrowed off at the end.
    let nblocks = len.div_ceil(block);
    let pad = nblocks * block - len;
    let t = if pad == 0 {
        t
    } else {
        let mut pad_dims = t.dims();
        pad_dims[dim] = pad;
        Tensor::cat(vec![t, Tensor::zeros(pad_dims, &device)], dim)
    };

    // Split the scanned axis into (which block, where in it). Row-major, so the
    // reshape moves nothing.
    let dims = t.dims();
    let mut split = [0usize; DP1];
    split[..dim].copy_from_slice(&dims[..dim]);
    split[dim] = nblocks;
    split[dim + 1] = block;
    split[dim + 2..].copy_from_slice(&dims[dim + 1..]);

    let inner = t.reshape(split).cumsum(dim + 1);
    // Each block's total is its last in-block prefix; kept at width 1 on that
    // axis so the carry broadcasts back over the block. `init` joins the carry
    // here, for free — one add serves both.
    let totals = inner.clone().narrow(dim + 1, block - 1, 1);
    let carry = totals.clone().cumsum(dim) - totals;
    let carry = match init {
        Some(init) => carry + init.unsqueeze_dim::<DP1>(dim + 1),
        None => carry,
    };

    let joined = (inner + carry).reshape(dims);
    if pad == 0 {
        joined
    } else {
        joined.narrow(dim, 0, len)
    }
}

// ---------------------------------------------------------------------------
// The read axis
// ---------------------------------------------------------------------------
//
// A chunk has two axes, and `micro_steps` widens only one of them. Every one of
// a token's `u` micro-steps **writes** to the state, so the source axis is the
// folded `chunk_len`; the token is **read** once, at its last micro-step, so
// the target axis is `chunk_len / u` — the rows whose `y` survives, everything
// else being multiplied by a zero gradient. `read_stride` is that `u` (and `1`
// wherever there is no fold: Mamba-2, `micro_steps = 1`, the `step` path),
// which makes both helpers below the identity there.
//
// The alignment is what keeps this a reshape rather than a gather:
// `chunk_len` is a multiple of `read_stride`
// ([`Mamba3SsdPath::chunk_len_or_optimal`](crate::mamba3::ssd_path::Mamba3SsdPath::chunk_len_or_optimal)),
// so chunk `c` covers folded positions `[c·L, (c+1)·L)` = tokens
// `[c·T, (c+1)·T)` and its read rows are a **contiguous run** of them.

/// The read rows of a folded axis: index `i·stride + (stride − 1)` for each
/// `i`, i.e. the last position of every run of `stride`.
///
/// The identity at `stride = 1`.
///
/// # Shapes
/// - `t`   : `[…, len, …]` at `dim`, `len` a multiple of `stride`
/// - out   : `[…, len / stride, …]`
///
/// `DP1 = D + 1`.
pub fn read_rows<const D: usize, const DP1: usize>(
    t: Tensor<D>,
    dim: usize,
    stride: usize,
) -> Tensor<D> {
    if stride <= 1 {
        return t;
    }
    let dims = t.dims();
    let len = dims[dim];
    assert_eq!(
        len % stride,
        0,
        "a folded axis of {len} is not a whole number of {stride}-step tokens",
    );
    let mut split = [0usize; DP1];
    split[..dim].copy_from_slice(&dims[..dim]);
    split[dim] = len / stride;
    split[dim + 1] = stride;
    split[dim + 2..].copy_from_slice(&dims[dim + 1..]);

    let mut out = dims;
    out[dim] = len / stride;
    // Row-major, so the split moves nothing and the narrow is a strided view.
    t.reshape(split).narrow(dim + 1, stride - 1, 1).reshape(out)
}

/// The additive causal mask over `(read row, folded source)`:
/// `0` where the source is early enough to contribute, `−∞` where it is not.
///
/// `diagonal` is passed to [`Tensor::triu`] on the **folded** grid, so it says
/// the same thing it says today: `0` excludes the same position (the single-SSD
/// strict mask, whose diagonal `ssd::diag` adds back with `γ`), `1` keeps it
/// (the double-SSD inclusive mask).
///
/// `−∞` rather than a `0/1` multiply because the mask is added *before* the
/// `exp`: for a source after the row the decay difference is positive and would
/// overflow.
///
/// # Shapes
/// - out : `[chunk_len / stride, chunk_len]`
pub fn read_causal_mask(
    chunk_len: usize,
    stride: usize,
    diagonal: i64,
    device: &Device,
) -> Tensor<2> {
    // Rows of the full folded mask, at the positions the readout happens at —
    // which is the same statement as `read_rows`, so it is the same op.
    let full = Tensor::<2>::full([chunk_len, chunk_len], f32::NEG_INFINITY, device).triu(diagonal);
    read_rows::<2, 3>(full, 0, stride)
}

/// The read axis on primitives, for the recompute-backward math.
///
/// Same definitions as [`read_rows`] / [`read_causal_mask`] one module up, on
/// [`F`] instead of `Tensor` — the split every kernel in this crate already has
/// between its `Tensor` body and the `F<B, _>` one a
/// [`Backward`](burn::backend::autodiff::ops::Backward) node can run (see
/// [`fprim`](burn_stack::utils::fprim)). Both are reshape-and-narrow, so
/// neither needs an op `F` does not carry.
pub mod prim {
    use burn_stack::utils::fprim::F;
    use burn::backend::Backend;
    use burn::backend::tensor::Device;
    use burn::backend::FloatDType;

    /// [`super::read_rows`] on primitives.
    pub fn read_rows<B: Backend, const D: usize, const DP1: usize>(
        t: F<B, D>,
        dim: usize,
        stride: usize,
    ) -> F<B, D> {
        if stride <= 1 {
            return t;
        }
        let dims = t.dims();
        let len = dims[dim];
        assert_eq!(
            len % stride,
            0,
            "a folded axis of {len} is not a whole number of {stride}-step tokens",
        );
        let mut split = [0usize; DP1];
        split[..dim].copy_from_slice(&dims[..dim]);
        split[dim] = len / stride;
        split[dim + 1] = stride;
        split[dim + 2..].copy_from_slice(&dims[dim + 1..]);

        let mut out = dims;
        out[dim] = len / stride;
        t.reshape::<DP1>(split)
            .narrow(dim + 1, stride - 1, 1)
            .reshape::<D>(out)
    }

    /// The transpose of [`read_rows`]: scatter a read-row tensor back onto the
    /// folded axis, zero at the positions no readout happens at.
    ///
    /// The gradient of `read_rows` — the hand-written backward's counterpart to
    /// what autodiff does for the `Minimal` / `Serial` paths. The identity at
    /// `stride = 1`.
    ///
    /// # Shapes
    /// - `t`  : `[…, len / stride, …]` at `dim`
    /// - out  : `[…, len, …]`
    pub fn scatter_read_rows<B: Backend, const D: usize, const DP1: usize>(
        t: F<B, D>,
        dim: usize,
        stride: usize,
    ) -> F<B, D> {
        if stride <= 1 {
            return t;
        }
        let dims = t.dims();
        let device = t.device();
        let dtype = t.dtype();

        let mut split = [0usize; DP1];
        split[..dim].copy_from_slice(&dims[..dim]);
        split[dim] = dims[dim];
        split[dim + 1] = 1;
        split[dim + 2..].copy_from_slice(&dims[dim + 1..]);
        let mut pad = split;
        pad[dim + 1] = stride - 1;

        let mut out = dims;
        out[dim] = dims[dim] * stride;
        // The row is the *last* of its run, so the zeros go in front of it.
        F::cat(
            vec![F::<B, DP1>::zeros(pad, &device, dtype), t.reshape::<DP1>(split)],
            dim + 1,
        )
        .reshape::<D>(out)
    }

    /// [`super::read_causal_mask`] on primitives.
    pub fn read_causal_mask<B: Backend>(
        chunk_len: usize,
        stride: usize,
        diagonal: i64,
        device: &Device<B>,
        dtype: FloatDType,
    ) -> F<B, 2> {
        let full = F::<B, 2>::full(
            [chunk_len, chunk_len],
            f32::NEG_INFINITY,
            device,
            dtype,
        )
        .triu(diagonal);
        read_rows::<B, 2, 3>(full, 0, stride)
    }
}

/// The **interior** of a lag-`lag` tap's gap: `Πᵈ⁼¹..ˡᵃᵍ⁻¹ αₚ₋ᵈ`, or `None` at
/// `lag = 1` (where the gap has no interior).
///
/// A tap at lag `L` is transported across its own gap, `Πᵈ⁼⁰..ᴸ⁻¹ αₚ₋ᵈ`
/// (`info/trapezoid-as-integration.md` §9). `β = (1−λ)Δα` already carries the
/// `d = 0` factor, so this is the rest of it.
///
/// The front is **zero-padded**, not clamped to what the call happens to hold:
/// for `p < L` the missing factors are exactly the ones the cache's `v` slots
/// were already scaled by when they were stored (see [`tail_decay`]).
///
/// # Shapes
/// - `da_bsh` : `[batch, sequence, nheads]` (`Δ·A`, the log-decay)
/// - out      : `[batch, sequence, nheads]`
pub fn interior_gap_decay(da_bsh: Tensor<3>, lag: usize) -> Option<Tensor<3>> {
    if lag <= 1 {
        return None;
    }
    let [batch, sequence, nheads] = da_bsh.dims();
    let device = da_bsh.device();
    let mut window_bsh = Tensor::zeros([batch, sequence, nheads], &device);
    for d in 1..lag {
        let zeros_bdh = Tensor::zeros([batch, d, nheads], &device);
        let shifted = Tensor::cat(vec![zeros_bdh, da_bsh.clone().narrow(1, 0, sequence - d)], 1);
        window_bsh = window_bsh + shifted;
    }
    Some(window_bsh.exp())
}

/// The decay each cached tap slot has already accumulated: `Πᵣ₌q₊₁^{S−1} αᵣ` for
/// the last `lag` positions `q` (oldest first), or `None` at `lag = 1` (the one
/// slot *is* the last position, so the product is empty).
///
/// Storing the tap slots pre-scaled by this is what lets a lag-`L` tap span a
/// call boundary: the next call supplies the in-call part of the gap
/// ([`interior_gap_decay`], front-zero-padded) and the slot carries the rest.
///
/// # Shapes
/// - `da_bsh` : `[batch, sequence, nheads]`
/// - out      : `[batch, lag, nheads]`
pub fn tail_decay(da_bsh: Tensor<3>, lag: usize) -> Option<Tensor<3>> {
    if lag <= 1 {
        return None;
    }
    let sequence = da_bsh.dims()[1];
    // Only the tail window matters, so the cumulative sum is over `lag` terms
    // and never accumulates the whole sequence's decay.
    let tail_blh = da_bsh.narrow(1, sequence - lag, lag);
    let cumulative_blh = tail_blh.cumsum(1);
    let total_b1h = cumulative_blh.clone().narrow(1, lag - 1, 1);
    Some((total_b1h - cumulative_blh).exp())
}

/// A `[1, len, 1]` gate over a folded axis: `0` at each token's **first**
/// micro-step (`p ≡ 0 mod u`) and `1` elsewhere.
///
/// Broadcasts against any `[batch, len, nheads]` stream, in `forward` (`len` is
/// the folded sequence, which starts at a token boundary) and in `step` (`len`
/// is `u` itself, so the gate is its first entry). At `u = 1` every position is
/// a token start and the gate is all zeros — which is the whole of why
/// [`Trapezoid::HorizontalReset`](crate::mamba3::trapezoid::Trapezoid::HorizontalReset)
/// degenerates to [`None`](crate::mamba3::trapezoid::Trapezoid::None) there.
///
/// This is where a pattern's admissibility rule becomes a tensor; *which* mass
/// it applies to is the caller's — `λ` for
/// [`Trapezoid::far_tap_crosses_tokens`](crate::mamba3::trapezoid::Trapezoid::far_tap_crosses_tokens),
/// `μ` for
/// [`interior_tap_crosses_tokens`](crate::mamba3::trapezoid::Trapezoid::interior_tap_crosses_tokens).
pub fn token_start_gate(len: usize, micro_steps: usize, device: &Device) -> Tensor<3> {
    let open: Vec<f32> = (0..len)
        .map(|p| if p.is_multiple_of(micro_steps) { 0.0 } else { 1.0 })
        .collect();
    Tensor::<1>::from_floats(open.as_slice(), device).reshape([1, len, 1])
}

/// Output of [`trapezoidal_coefficients`] — the step's mass `Δₜ`, split.
///
/// The masses are **untransported**: `α` and any wider gap decay are the
/// consumer's to apply, since which one a tap needs depends on the pathway (the
/// double-SSD pass multiplies them in, the single-SSD key scale never does).
pub struct TrapezoidCoeffs {
    /// `Δₜ = softplus(dd_dt + dt_bias)`, clamped.
    pub dt: Tensor<3>,
    /// `Δₜ · Aₜ` (negative; the log-decay).
    pub da: Tensor<3>,
    /// `αₜ = exp(Δₜ · Aₜ) ∈ (0, 1]` — decay.
    pub alpha: Tensor<3>,
    /// `νₜ` — the mass of the tap at
    /// [`tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag), before its
    /// transport. `None` under
    /// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None), which has
    /// no left endpoint: the tensor is not formed rather than formed as zeros.
    pub nu: Option<Tensor<3>>,
    /// `νⁱⁿᵗₜ` — the mass of the extra lag-1 tap, `None` unless
    /// [`has_interior_tap`](crate::mamba3::trapezoid::Trapezoid::has_interior_tap).
    pub nu_interior: Option<Tensor<3>>,
    /// `γₜ = λₜ · Δₜ` — right-endpoint weight, and **`Δₜ` itself** when there is
    /// no `λ` (the whole step is paid at the right endpoint).
    pub gamma: Tensor<3>,
}

/// Compute the trapezoidal discretisation coefficients from the raw
/// (data-dependent) projections. See the top-of-`mamba3.rs` docs for the
/// formulas and `trapezoid.rs`'s header for how the masses divide.
///
/// All data tensors are `[batch, len, nheads]`, `len` being the folded sequence
/// (`forward`) or the `u` micro-steps of one token (`step`); `dt_bias_h` is
/// broadcast to match. Axis 1 is what the patterns' gates are periodic in, in
/// both cases.
///
/// `lambda_raw` is `None` under
/// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) — the block
/// projects no `λ` channels — and the coefficients then take their `λ ≡ 1`
/// values *by construction*: no mass is formed and `γ` **is** `Δ`, sharing its
/// tensor rather than recomputing `1 · Δ`. `mu_raw` is `None` unless the pattern
/// [`has_interior_tap`](crate::mamba3::trapezoid::Trapezoid::has_interior_tap),
/// and the same holds one level down: no `μ`, no second mass.
///
/// A gate closes by handing its mass back, not by dropping it: `λ` is set to `1`
/// where the far tap is inadmissible (so `γ` takes the step whole) and `μ` to
/// `0` where the interior one is (so the far tap takes the left endpoint whole).
/// Both are exact at the ends, so the degenerate members are their targets bit
/// for bit.
pub fn trapezoidal_coefficients(
    dd_dt: Tensor<3>,
    dd_a_raw: Tensor<3>,
    lambda_raw: Option<Tensor<3>>,
    mu_raw: Option<Tensor<3>>,
    dt_bias_h: Tensor<1>,
    spec: TrapezoidSpec,
) -> TrapezoidCoeffs {
    let (dt_limit, a_floor) = (spec.dt_limit, spec.a_floor);
    // Broadcast dt_bias_h [nheads] → [1, 1, nheads] so the addition aligns on
    // the last dim.
    let dt_bias_broadcast = dt_bias_h.unsqueeze::<3>();
    let dt = softplus(dd_dt + dt_bias_broadcast).clamp(dt_limit.0, dt_limit.1);
    // `A = −max(softplus(·), a_floor) ∈ (−∞, −a_floor]`. The floor must be
    // applied to the (positive) softplus *before* negating: a method call
    // binds tighter than unary minus, so `-softplus(x).clamp(NEG_INFINITY,
    // -a_floor)` would collapse the positive softplus to the constant
    // `-a_floor` and yield `A ≡ +a_floor` — a *growing* state (`α > 1`) with a
    // dead `dd_A` projection.
    let a = -softplus(dd_a_raw).clamp(a_floor, f64::INFINITY);
    let da = dt.clone() * a;
    let alpha = da.clone().exp();
    let gate = |crosses: bool| {
        (!crosses).then(|| token_start_gate(dt.dims()[1], spec.micro_steps, &dt.device()))
    };
    let (nu, nu_interior, gamma) = match lambda_raw {
        Some(lambda_raw) => {
            let lambda = burn::tensor::activation::sigmoid(lambda_raw);
            // A closed far tap means λ = 1 there: the whole step is paid at the
            // right endpoint, which is what makes the gated pattern a submodel
            // of the ungated one rather than a lossy version of it.
            let lambda = match gate(spec.pattern.far_tap_crosses_tokens()) {
                Some(open_1s1) => lambda * open_1s1.clone() + (-open_1s1 + 1.0),
                None => lambda,
            };
            let nu = (-lambda.clone() + 1.0) * dt.clone();
            let gamma = lambda * dt.clone();
            // …and a closed interior tap means μ = 0 there: its share of the
            // left endpoint returns to the far tap, one level up.
            let (nu, nu_interior) = match mu_raw {
                Some(mu_raw) => {
                    let mu = burn::tensor::activation::sigmoid(mu_raw);
                    let mu = match gate(spec.pattern.interior_tap_crosses_tokens()) {
                        Some(open_1s1) => mu * open_1s1,
                        None => mu,
                    };
                    let nu_interior = nu.clone() * mu.clone();
                    (nu * (-mu + 1.0), Some(nu_interior))
                }
                None => (nu, None),
            };
            (Some(nu), nu_interior, gamma)
        }
        // λ ≡ 1: the left endpoint is unweighted and the right one takes Δ whole.
        None => (None, None, dt.clone()),
    };
    TrapezoidCoeffs {
        dt,
        da,
        alpha,
        nu,
        nu_interior,
        gamma,
    }
}

/// QK-Norm → GQA-expand groups→heads → add per-(head, mimo-rank) bias.
///
/// The input is the raw B/C projection already reshaped to expose the group
/// dim, with last dim = `state_rank`. The output replaces the group dim with
/// the head dim, leaving the last dim untouched.
///
/// `DP1 = D + 1` (required by [`gqa_expand_to_heads`]'s intermediate rank).
pub fn qk_norm_expand_bias<const D: usize, const DP1: usize>(
    raw_mgr: Tensor<D>,
    norm: &RmsNorm,
    bias_hmr: Tensor<3>,
    group_dim: usize,
    nheads: usize,
) -> Tensor<D> {
    // RmsNorm operates on the last dim only, so the leading shape passes through.
    let normed = norm.forward(raw_mgr);
    let expanded = gqa_expand_to_heads::<D, DP1>(normed, group_dim, nheads);
    // Broadcast bias [nheads, mimo_rank, state_rank] → [1, ..., 1, mimo_rank, nheads, state_rank].
    let bias = bias_hmr.swap_dims(0, 1).unsqueeze::<D>();
    expanded + bias
}

/// Rank-summed outer product `state[b, h, p, r] = Σₘ v[b, m, h, p] · k[b, m, h, r]`
/// (`einsum('bmhp,bmhr->bhpr')`).
///
/// This is the per-token state contribution: each MIMO rank contributes an
/// outer product `v[m] ⊗ k[m]` and the shared state accumulates their sum.
///
/// At `mimo_rank == 1` the contracted dimension is 1, so the GEMM is rank-1 and
/// the sum is a single outer product — [`mimo_outer_sum_siso`] writes it as a
/// broadcast multiply instead. Which form wins is backend-dependent, and by a
/// wide margin at these (tiny, decode-sized) shapes: the broadcast is the
/// faster one on CUDA and *much* slower on the portable CPU backends, whose
/// broadcast-elementwise path trails their matmul by more than an order of
/// magnitude here. The choice is therefore
/// [`Mamba3Config::siso_specialization_decode`](crate::mamba3::mamba3::Mamba3Config::siso_specialization_decode)'s
/// — the per-token flag, *not* the chunkwise one, whose verdict is the opposite;
/// both branches compute the same values and gradients.
pub fn mimo_outer_sum(
    v_bmhp: Tensor<4>,
    k_bmhr: Tensor<4>,
    siso_specialization: bool,
) -> Tensor<4> {
    let [_batch, mimo_rank, _nheads, _per_head_dim] = v_bmhp.dims();
    if mimo_rank == 1 && siso_specialization {
        mimo_outer_sum_siso(v_bmhp, k_bmhr)
    } else {
        mimo_outer_sum_mimo(v_bmhp, k_bmhr)
    }
}

/// SISO (`mimo_rank == 1`) rank-summed outer product: the single `v ⊗ k` as a
/// broadcast multiply of `[b, h, p, 1]` by `[b, h, 1, r]`.
pub fn mimo_outer_sum_siso(v_bmhp: Tensor<4>, k_bmhr: Tensor<4>) -> Tensor<4> {
    let v_bhp1: Tensor<4> = v_bmhp.squeeze_dim::<3>(1).unsqueeze_dim(3);
    let k_bh1r: Tensor<4> = k_bmhr.squeeze_dim::<3>(1).unsqueeze_dim(2);
    v_bhp1 * k_bh1r
}

/// General MIMO rank-summed outer product: a matmul contracting `mimo_rank`.
pub fn mimo_outer_sum_mimo(v_bmhp: Tensor<4>, k_bmhr: Tensor<4>) -> Tensor<4> {
    // v_bmhp [b, m, h, p] -> v_bhpm [b, h, p, m]; k_bmhr [b, m, h, r] -> k_bhmr.
    let v_bhpm = v_bmhp.permute([0, 2, 3, 1]);
    let k_bhmr = k_bmhr.swap_dims(1, 2);
    v_bhpm.matmul(k_bhmr)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

/// Build the MIMO value tensor `v = x ⊙ mimo_x` with broadcasting.
///
/// Inserts a `mimo_rank` axis at `insert_dim`. When `mimo_x_hmp` is `None`
/// (SISO), the inserted axis has size 1 and `x` is passed through; otherwise
/// broadcasting fills the inserted axis to size `mimo_rank`.
///
/// `DP1 = D + 1`.
pub fn build_v_with_mimo<const D: usize, const DP1: usize>(
    x: Tensor<D>,
    mimo_x_hmp: Option<&Tensor<3>>,
    insert_dim: usize,
) -> Tensor<DP1> {
    let x_with_rank_axis = x.unsqueeze_dim::<DP1>(insert_dim);
    match mimo_x_hmp {
        None => x_with_rank_axis,
        Some(mimo_x_hmp) => {
            // mimo_x_hmp [nheads, mimo_rank, per_head_dim] → swap_dims to
            // [mimo_rank, nheads, per_head_dim] → unsqueeze leading 1s. The
            // result broadcasts against `x_with_rank_axis` over (batch, seq, …).
            let mimo_x_broadcast = mimo_x_hmp.clone().swap_dims(0, 1).unsqueeze::<DP1>();
            x_with_rank_axis * mimo_x_broadcast
        }
    }
}
