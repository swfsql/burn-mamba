# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

A Rust library implementing [Mamba-1](https://arxiv.org/abs/2312.00752),
[Mamba-2](https://arxiv.org/abs/2405.21060), and
[Mamba-3](https://arxiv.org/abs/2603.15569) SSM (Structured State Space Model)
architectures on top of the [Burn](https://github.com/tracel-ai/burn/) deep
learning framework. The goal is a **minimal, readable reference
implementation** that ports the official CUDA/Triton kernels down to standard
Burn tensor ops — not a production-optimized one. There are **no custom
kernels**; everything is portable Burn tensor ops, so the same code runs on
every backend (CPU, WGPU, CUDA, Metal, LibTorch, …).

## Build & Test Commands

```bash
cargo check                                   # type-check the lib surface
cargo test --features "backend-flex"          # run tests (any backend; flex = CPU default)
cargo doc --all --no-deps                     # build docs
# An example (flex backend); see examples/README.md for the full set:
cargo run --example fibonacci --features "backend-flex" -- --training --inference
```

- **Feature flags select the backend**: `backend-flex` (preferred for
  checks/tests), `backend-cpu`, `backend-wgpu`, `backend-metal`,
  `backend-vulkan`, `backend-cuda`, `backend-rocm`, `backend-tch-{cpu,gpu}`,
  `backend-remote`, `backend-ndarray` (deprecated). Each flag just enables the
  matching `burn/<backend>`; several may be compiled in at once and
  `Device::default()` resolves which to use (honouring `BURN_DEVICE`). The
  tensor tests compile under whichever backend is enabled and pick the device at
  runtime.
- `mamba1`/`mamba2`/`mamba3`/`autodiff` are on by default; `mamba2`/`mamba3`
  imply `autodiff` (their custom backward needs it). `cubecl`/`fusion` enable the
  memory-saving custom backward on those backend families. `dev-f16`,
  `dev-simd`, `dev-autotune` are example/test conveniences.
- `bacon.toml` configures [bacon](https://github.com/Canop/bacon) watch jobs;
  `cubecl.toml` configures the CubeCL runtime.

## File Map

`refs/` and `doc/` are external reference material (paper TeX, official Python
impl, a minimal third-party impl, Burn source) — see [Extra
References](#extra-references); not listed here. `examples/` is documented by its
own `examples/README.md`. Every leaf module also has a sibling `tests.rs` (or
`*/tests.rs`) asserting forward/step parity, gradients, and cross-variant
agreement; those are not listed individually.

```text
src/
├─ lib.rs            crate root: module decls, prelude, DENY_NAN/DENY_INF sanity flags
├─ mamba1/           original selective SSM (conv1d + sequential selective scan)
│  ├─ mamba1.rs      Mamba1 block + Config: forward()(selective_scan) / step()
│  └─ cache.rs       Mamba1Cache(s): conv window (bik) + SSM state (bir)
├─ mamba2/           SSD (Structured State Space Duality)
│  ├─ mamba2.rs      Mamba2 block + Config: chunkwise forward() / recurrent step()
│  ├─ cache.rs       Mamba2Cache(s): conv window (bvk) + SSM state (bhpr)
│  └─ ssd/           ssd_path.rs selector; minimal / serial / serial_recalculated (custom backward)
├─ mamba3/           trapezoidal SSD + data-dependent RoPE + MIMO
│  ├─ mamba3.rs      Mamba3 block + Config; forward()/step() dispatch by cache variant
│  ├─ helpers.rs     shared: trapezoid coeffs, QK-norm+GQA+bias, MIMO-V build
│  ├─ cache.rs       Mamba3Cache(s) ENUMS dispatching DoubleSsd vs SingleSsd
│  ├─ ssd_path.rs    pathway-agnostic Mamba3SsdPath (From<>/Into<> both sub-paths)
│  ├─ double_ssd/    two-pass trapezoid (γ-SSD + β-SSD); cache.rs + ssd/ kernels
│  ├─ single_ssd/    one-pass official-kernel form (≈½ memory); cache.rs (h' semantics) + ssd/
│  ├─ rotation/      quaternion non-abelian RoPE (Complex2D | Quaternion4D) + algebra/scan
│  └─ quat_scan/     memory-efficient quaternion cumprod scan (recompute backward)
├─ modules/          family-generic composition + shared NN modules
│  ├─ mod.rs         MambaBlock / MambaBlockConfig traits; MambaSsdPath enum
│  ├─ layer.rs       Layer<M>: Pre-LN residual block
│  ├─ layers.rs      Layers<M>: virtual-layer stack over real weight sets
│  ├─ network.rs     LatentNetwork / VocabNetwork + MambaLatentNet / MambaVocabNet enums
│  ├─ bidi.rs        BidiLayers<M> + OutputMerge + MambaBidiLayers enum
│  ├─ cache.rs       CacheStack trait + MambaCaches enum
│  ├─ activation/    silu, softplus, log_sigmoid (fp16-aware)
│  ├─ norm/          rms_norm (also Mamba-3 QK-Norm), rms_norm_gated
│  ├─ loss/          bce, cross_entropy, mse
│  └─ misc/          gqa, segsum, split, sanity
└─ utils/            lower-level plumbing
   ├─ mod.rs         div_eps (per-dtype epsilon, takes runtime DType)
   ├─ class/         ClassToken / ClassLatent insertion (CLS-style registers)
   ├─ schedule/      Schedule + BidiSchedule (virtual→real index mapping)
   ├─ scheduler/     LR schedulers (cosine-annealing + warmup, constant) — example use
   ├─ backend_macros.rs  per-backend BackendExt impls + autodiff marker traits
   ├─ combined_grad.rs   flatten/unflatten (y, final_state) for custom backward
   ├─ fprim.rs           F<B,D>: rank-tagged FloatTensor-primitive wrapper
   └─ test_helpers.rs    max_abs_diff + grad-comparison macros
```

`files.md` holds the per-file signature reference (the items each important file
defines and the non-obvious decisions behind them).

---

## Architecture

### Layer → Network hierarchy (all three families)

All three families share **one** set of generic composition types in
[`src/modules/`](#), parameterised by the SSM core block `M` (`Mamba1` /
`Mamba2` / `Mamba3`):

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → out_proj            (continuous I/O)
Layers<M>         a stack of N (virtual) layers over R real weight sets
Layer<M>          Pre-LN residual:  y = x·residual_scale + Block(RMSNorm(x))
M (Block)         the SSM core itself (mamba1.rs / mamba2.rs / mamba3.rs)
```

- `Layer<M>` (`layer.rs`) wraps the core block with an input RMSNorm + residual
  add. `Layers<M>` (`layers.rs`) owns **virtual-layer scheduling** (see below)
  plus `ignore_first/last_residual` flags. The **bidirectional** wrapper
  `BidiLayers<M>` (`bidi.rs`) serves every family.
- `VocabNetwork<M>` (`network.rs`): `Embedding` → `Layers<M>` → `norm_f` → LM
  head, where the head is **tied** to the embeddingᵀ (`missing_lm_head`) or a
  separate `Linear`; vocab is padded to `pad_vocab_size_multiple`.
  `LatentNetwork<M>` is the continuous-I/O sibling (linear `in_proj`/`out_proj`).
  Runtime-dispatch enums `MambaVocabNet` / `MambaLatentNet` / `MambaBidiLayers`
  (each with a `#[derive(Config)]` `*Config`) pick the family at construction.

### Dual execution modes

Every block / layer / network exposes:
- **`forward()`** — parallel (chunkwise) mode for training and prompt prefill;
  linear in sequence but expressed as batched GEMMs.
- **`step()`** — recurrent mode for token-by-token decoding; O(state) per token,
  constant memory, no growing KV cache.

`forward()` from any initial cache is mathematically equal to recurrent `step()`
unrolling from that same cache — parity on **outputs, final cache, and
gradients** is what the per-block test suites assert (it subsumes the
chunked-prefill split-vs-full guarantee).

### Caches

A cache carries streaming state between calls (prefill→decode, or chunked
prefill). Mamba-1/2 caches hold a conv window + SSM state. **Mamba-3 has no conv
cache** (the short convolution is removed).

### SSD algorithm selection (Mamba-2 & Mamba-3)

The chunkwise scan is pluggable via an `…SsdPath` enum; each variant carries an
optional chunk length (`None` ⇒ optimal default ≈ `√(state_rank·per_head_dim)`,
rounded to a multiple of 32, capped at 512):

| Variant | Algorithm | Backward |
|---------|-----------|----------|
| `Minimal(chunk)` | batched matmuls + `segsum` mask | autodiff |
| `Serial(chunk)` | serial loop over chunks + matmuls (mirrors Triton K1–K5) | autodiff |
| `SerialRecalculated(chunk)` | serial loop, recompute backward (mirrors `ssd_combined.py`) | **custom** (~⅓ less training memory) |

`Default = SerialRecalculated(None)`. `SsdPath::run(input)` dispatches; all three
are exact reformulations and must agree on values **and** gradients (asserted by
the `ssd_path` tests).

**Backend extension traits.** Each SSD family has a `…BackendExt` trait whose
default body works for every plain Burn backend; only `Autodiff<B>` gets the
custom backward (`serial_recalculated/backward.rs` + `combined_backward.rs`).
`utils/backend_macros.rs` emits the per-backend default impls + the autodiff
marker; `Mamba3BackendExt` aggregates the double- and single-ssd ext traits. The
custom backward hands Burn one tracked tensor, so `utils/combined_grad.rs`
flattens `(y, final_state)` and splits it back.

### Mamba-1 (`src/mamba1/`)

Original selective SSM. `mamba1.rs`: in-proj → causal depthwise `conv1d`
(left-padded from the cache for strict causality) → SiLU → `x_proj`/`dt_proj`
selective projections → a **sequential `selective_scan`** (ZOH for A, Euler for
B) → SiLU gate → out-proj. `step()` shares the cache (`conv_bik` window +
`ssm_bir` state). The layer stack / network / bidi come from `src/modules/`.

### Mamba-2 (`src/mamba2/`)

Structured State Space Duality. `mamba2.rs` is heavily documented (read its
module header for the full SSD math). `forward()`: in-projection
`d_model → [z | xbc | dt_raw]` → causal depthwise Conv1d + SiLU over `xbc` →
split `(x, B, C)` → discretise (`Δ = softplus(dt_raw+dt_bias)`, `Ā = exp(Δ·A)`
with `A = -exp(a_log)`, `B̄ = Δ·B`) → zero-pad to a multiple of `chunk_len`
(exact: Δ=0 ⇒ Ā=1, B̄=0) → GQA-expand B/C → run the selected **SSD path** →
gated RMSNorm with the `z` gate → out-proj. `step()` runs the recurrence
`hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜᵀ`, `yₜ = Cₜᵀ hₜ + D xₜ`; optional learnable
`init_state_hpr`.

### Mamba-3 (`src/mamba3/`)

Extends Mamba-2 with three independent additions (each works alone or combined):
**trapezoidal discretisation**, **data-dependent RoPE** on B/C, and **MIMO**
rank expansion. Read the `mamba3.rs` module header for the full combined math.

#### Differences from Mamba-2

| Aspect | Mamba-2 | Mamba-3 |
|--------|---------|---------|
| Recurrence | 2-term `h = Ā h + B̄ x` | 3-term trapezoidal `h = α h + β B₋₁x₋₁ + γ Bₜxₜ` |
| `A` | fixed per-head `a_log_h` | data-dependent `Aₜ = −softplus(dd_A)`, clamped `≤ −a_floor` |
| `λ` (trapezoid split) | — | per-head data-dependent `λ = σ(λ̂)` |
| Short conv | present | **removed** (no conv cache) |
| B/C norm | post-SSD gated RMSNorm | **QK-Norm** (`RmsNorm` over `state_rank`) **before** SSD |
| B/C bias | none | learnable `[nheads, mimo_rank, state_rank]`, init = 1 |
| Positional | none | data-dependent cumulative RoPE on B/C |
| MIMO | none | `mimo_rank > 1`: parallel B/C rank channels + `mimo_x/z/o` projections |
| Out gate | gated RMSNorm | SiLU gate, or optional per-head gated RMSNorm |

#### Trapezoidal coefficients (`helpers::trapezoidal_coefficients`)

```text
Δₜ = softplus(dd_dt + dt_bias)   (clamped to dt_limit)
Aₜ = −softplus(dd_A)             (clamped ≤ −a_floor)
αₜ = exp(Δₜ·Aₜ)                  decay
βₜ = (1 − λₜ)·Δₜ·αₜ              left-endpoint weight  (Bₜ₋₁xₜ₋₁)
γₜ = λₜ·Δₜ                       right-endpoint weight (Bₜxₜ)
```

Setting `λ ≡ 1` collapses to the Mamba-2 (Euler) form.

#### In-projection layout

```text
d_in_proj = 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rotation_channels
split:      [ z | x | B_raw | C_raw | dd_dt | dd_A | λ_raw | θ ]
```

`B_raw`/`C_raw` carry `mimo_rank · ngroups · state_rank` channels each;
`ngroups < nheads` shares B/C GQA-style.

#### Data-dependent RoPE

The default `Complex2D` (abelian `SO(2)`) rotation: angles are projected
(`num_rope_angles = rope_dim/2`), squashed by `tanh(θ)·π`, scaled by Δ per head,
and **cumulatively summed** along the sequence (continued from the cache):
`cum_angle[t] = cum_angle[t−1] + Δₜ·π·tanh(θₜ)`. Pairing is interleaved/NeoX for
SISO (`mimo_rank == 1`) and half-and-half/GPT-J for MIMO. `rope_fraction = 0.5`
rotates only the first `rope_dim` entries (rest pass through); `0` disables RoPE
(`num_rope_angles` floored at 1 since Burn has no zero-width tensors). Angles are
reduced mod `2π` into `[−π, π]` by `wrap_angle` (value-exact, the subtracted
multiple is `detach`ed so gradients are unchanged), keeping the accumulator
bounded for long sequences and fp16-stable.

`mamba3/rotation/` holds the **non-abelian** generalisation: a quaternion
(`k = 4`, `SU(2) ⊂ SO(4)`) rotational state where the cumulative rotation is an
associative *scan* (with a cross-chunk carry) instead of a `cumsum`, while the
B/C-factoring (hence the scalar-decay SSD core) is unchanged. Selected by
`Mamba3Config.rotation: RotationKind` (`Complex2D` default | `Quaternion4D`):
the cache accumulator field is a `RotationState` (`Angle` | `Quaternion`); the
in-projection devotes `num_rotation_channels` to it; forward/step branch via the
shared `rotate_bc_forward`/`rotate_bc_step` helpers. **Quaternion4D runs on both
SSD pathways** (rotation is applied to B/C before chunking, so the SSD core only
sees rotated `B̄`/`C̄`); a missing cache defaults to single-ssd, and `step`
round-trips through the double-ssd recurrence.

#### Two SSD pathways — the central Mamba-3 design point

The trapezoidal recurrence is realised by **two interchangeable algorithms**,
selected at runtime by which **cache variant** is supplied. `Mamba3Cache` /
`Mamba3Caches` are **enums** (`DoubleSsd | SingleSsd`); `forward`/`step` match
and delegate. A missing cache defaults to **SingleSsd**.

- **Double-SSD** (`double_ssd/`, VikramLex-style): splits the trapezoid into two
  **standard** SSD calls reusing the Mamba-2-like kernels — γ-SSM
  (`hᵞₜ = αₜ hᵞₜ₋₁ + γₜ Bₜxₜ`, current token) + β-SSM
  (`hᵝₜ = αₜ hᵝₜ₋₁ + βₜ Bₜ₋₁xₜ₋₁`, previous token, "shift-before-chunking"),
  summed. Simple/verifiable but ~2× intra-chunk + chunk-state memory; both passes
  feed `da = Δ·A`. `step()` runs this recurrence directly.
- **Single-SSD** (`single_ssd/`, official Triton-SISO / Tilelang-MIMO form): one
  SSD call using `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁` as key scale, a strict
  lower-triangular intra-chunk mask, a same-step γ correction, and a **boundary β
  seed** `(1−λ₀)·Δ₀·Kₜ₋₁⊗xₜ₋₁` folded into the initial state. ≈ half the training
  memory. Its accumulator `h'` has **different semantics** mid-sequence (distinct
  `Mamba3SingleSsdCache` type so the two can't be mixed in a chunked pass), but at
  boundaries `scaleₜ = γₜ` so `h'` coincides with the double-ssd state — hence the
  field-identity `From` conversions (`mamba3/cache.rs`). `step_single_ssd` decodes
  by converting to the double-ssd cache, running `step_double_ssd`, converting back.

`Mamba3SsdPath` (top-level) is pathway-agnostic and converts to/from
`Mamba3DoubleSsdPath` / `Mamba3SingleSsdPath` via `From`. The SSD inputs differ:
double feeds pre-scaled `v_bnlmhp` (× γ or β); single feeds raw `v` plus
`gamma_bnlh` and `scale_bnlh` so the kernel applies them.

#### Mamba-3 cache fields (both variants)

`ssm_bhpr` (SSM hidden state — semantics differ Double vs Single, see above),
`k_state_bmhr` (previous-token B per rank, for the β term), `v_state_bhp`
(previous-token x), `rotation` (a `RotationState`: cumulative RoPE angle for
`Complex2D`, cumulative quaternion for `Quaternion4D`; identical in both
variants and moved by the `From` conversions). No conv cache.

### Virtual layer scheduling (`src/utils/schedule/`)

`Layers<M>` can run `n_virtual_layers` logical passes over `n_real_layers`
weight sets (e.g. 48 logical from 12 real). Each virtual layer keeps its **own
cache** but shares parameters. `Schedule` maps virtual→real: `Cyclic` (wrap),
`Stretched` (block-repeat), `Custom(Vec)`. For bidi stacks, `BidiSchedule` pairs
forward/backward layers (`StridedCyclic`, `StridedStretched`, `SymmetricCyclic`,
`SymmetricStretched`, `Custom`; even virtual indices = →, odd = ←).

### Bidirectional support (`BidiLayers<M>` in `src/modules/bidi.rs`)

For non-autoregressive tasks. A `BidiLayerPair<M>` runs a straight (→) and a
reversed (← via `flip` on the sequence axis, then flip back) pass, merged by
`OutputMerge` (`Mean` or `CatLinear`). `BidiLayers<M>` stacks pairs with a
`BidiSchedule`. Generic over `M`, so it serves all three families (the
`MambaBidiLayers` enum dispatches).

### Class tokens / latents (`src/utils/class/`)

Learnable `[CLS]`-style embeddings spliced into the sequence. A **`ClassToken`**
lives on a *network* (`LatentNetwork` at `input_size`, before `in_proj`;
`VocabNetwork` at `d_model`, after the embedding); a **`ClassLatent`** lives on a
*layer* container (`Layer` / `Layers` / `BidiLayerPair` / `BidiLayers`, at
`d_model`). Each container is independent; markers (`Start | Middle | End |
Custom(idx)`) say *where* each lands, while one `Param<Tensor<2>>`
(`[num_markers, width]`, row `i` ↔ marker `i`) holds the embeddings. Insertion
(relative to original length `L`): all `Start` (index 0), then `Middle` (`L/2`),
`End` (`L`), `Custom(idx)` last; ties keep `Vec` order. The sequence permanently
lengthens; `forward` returns the lengthened sequence and the caller reads tokens
back via `class_{token,latent}_output_indices`.

`step` injects class tokens via optional position **cursors** (`Option<&mut
usize>` / `Option<&mut Vec<usize>>`): when a cursor reaches a class position the
embedding is stepped first (advancing the cursor), then the user token — only the
user token's output/cache is returned. `Layer` takes one cursor; `Layers` takes
a stack-level cursor **and** a per-virtual-layer `Vec`; `LatentNetwork` adds a
third for its own class tokens; `VocabNetwork` forwards the two `Layers` cursors.
`Start`/`Custom` are length-independent (inject in `step`); `Middle`/`End` need
the full length and panic for the cursored level (use `forward`).

### Shared modules (`src/modules/`) & utilities (`src/utils/`)

`modules/` also holds the shared neural building blocks: custom activations
(`silu`, `softplus`, `log_sigmoid`) and norms (`rms_norm` — also Mamba-3's
QK-Norm — and `rms_norm_gated`), `loss/` (bce/cross_entropy/mse), and `misc/`
(`segsum` 1-semiseparable mask, `gqa` group→head expansion, typed `split`,
`sanity` guards). These exist because Burn lacks them or needs an fp16-stable
variant (the fp16 RMSNorm paths normalise against `max(|x|)` to avoid `x²`
overflow). `utils/` holds lower-level plumbing: `class/`, `schedule/`,
`scheduler/`, `combined_grad`/`backend_macros`/`fprim` (custom-backward), and
`div_eps` (per-dtype epsilon; `utils/mod.rs`). NaN/Inf guards are gated by the
crate-level `DENY_NAN`/`DENY_INF` constants in `lib.rs`.

---

## Key Design Decisions

- **No optimized kernels** — relies entirely on Burn's portable tensor ops, so
  the same code runs on every backend.
- **Dispatch backend (Burn 0.22+)** — the high-level `Tensor` (and every
  `Module`) is pinned to the global `Dispatch` backend, so library types are **no
  longer backend-generic** (`struct Mamba2`, `Mamba2Cache`, … carry no `<B>`).
  The backend is chosen at runtime via `Device`; autodiff and dtype are device
  properties. Only the custom-backward internals stay generic over `B` (`F<B,D>`,
  the `Backward<B,_>` nodes, the `Autodiff<B>` ext impls).
- **Two Mamba-3 SSD pathways** — double-ssd (simple, verifiable) vs single-ssd
  (official-kernel form, ~½ training memory). The cache type selects the pathway;
  accumulators differ mid-sequence but coincide at boundaries, so caches
  inter-convert via field-identity `From`. `step()` runs the double-ssd recurrence
  for both.
- **Three SSD algorithm variants** (`Minimal`/`Serial`/`SerialRecalculated`),
  the last with a custom recompute backward; all proven equal on values +
  gradients by tests.
- **`#![warn(missing_docs)]`** (in `src/lib.rs`): every public item should carry
  a doc comment, and any gap surfaces as a build warning. Keep the crate
  warning-clean — document public surface as you add it.
- **Burn 0.22+** — a recent version; APIs may differ from older online docs.
- The root of the project is `/shared/claude/burn-mamba/`; do not read/write
  outside it.
- Whenever a source file is created/removed/updated, update its entry in this
  [File Map](#file-map) and in `files.md` ([File Map](#file-map) lists all source
  files; `files.md` covers only the most important).

---

## Notation

Tensor names carry a shape suffix and the codebase is **deliberately
verbose/pedantic** about this — it is desired, backed by frequent shape
`assert`s. Conventions:

- A name whose suffix already encodes the shape needs no extra shape comment.
  In commentary, a shape can be written underscore-style (`_bhl`) or spelled out
  by name; expand to explicit `[...]` form only when it deserves attention
  (usually when an upper-case letter is involved) or when describing indexing.
- **Paper** style (upper-case `A, B, C, H, Y, L, …`) may appear in
  documentation/comments, but **code identifiers must never** use it.
- **Lower-case** letters are base dimensions (table below). **Upper-case** letters
  denote a *relation* of them (offset/multiple/concat/stacking): e.g. `X` may be
  `x±1` or `x*2`; `S` is the padded sequence; `K = conv_kernel − 1`; `XY` may be
  `x+y` or `x*y`. Upper-case may also refer to paper elements.

| Letter | Dimension | Paper | Python | Typical |
|--------|-----------|-------|--------|---------|
| `b` | `batch` | — | `batch` | varies |
| `s` | `sequence` length | `T` | `seqlen` | varies |
| `d` | `d_model` | `D` | `d_model` | 768, 1024 |
| `i` | `d_inner` = `expand`·`d_model` | `E·D` | `d_inner` | 2·`d_model` |
| `h` | `nheads` | `H` | `nheads` | `d_inner`/`per_head_dim` |
| `p` | `per_head_dim` | `P` | `headdim` | 64, 128 |
| `r` | `state_rank` | `N` | `d_state` | 64, 128, 256 |
| `m` | `mimo_rank` (Mamba-3) | `M` | `mimo_rank` | 1–8 |
| `n` | `nchunks` = `sequence`/`chunk_len` | — | `nchunks` | varies |
| `g` | `ngroups` | `G` | `ngroups` | 1 … `nheads` |
| `l` | `chunk_len` | `Q` | `chunk_size` | 64 … 256 |
| `a` | `num_rope_angles` = `rope_dim`/2 | — | `num_rope_angles` | varies |
| `v` | `conv_dim` = `d_inner`+2·`ngroups`·`state_rank` (Mamba-2) | — | `conv_dim` | — |
| `k` | `conv_kernel` (Mamba-1/2) | — | `d_conv` | 4 |

Per-file module headers (notably `mamba2.rs` and `mamba3.rs`) carry their own
notation tables tailored to that file — consult them first when editing.

## Extra References

External material under `refs/` (not analyzed here):
- **Mamba-3 paper** (TeX): `refs/mamba-3-paper/`.
- **Official Mamba-1/2/3 impl** (Python, authoritative): `refs/state-spaces/mamba/`
  — the Triton SISO / Tilelang MIMO kernels are the reference for the single-ssd path.
- **Mamba-3 minimal** (unofficial): `refs/VikramLex/mamba3-minimal/` — basis of
  the double-ssd decomposition.
- **Burn**: `refs/burn/`.
