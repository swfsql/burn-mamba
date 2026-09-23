# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

A Rust library that implements [Mamba-1](https://arxiv.org/abs/2312.00752),
[Mamba-2](https://arxiv.org/abs/2405.21060), and
[Mamba-3](https://arxiv.org/abs/2603.15569) SSM (Structured State Space Model)
architectures on top of the [Burn](https://github.com/tracel-ai/burn/) framework.
The goal is a **minimal, readable reference**. It ports the official
CUDA/Triton kernels to standard, portable Burn tensor ops, with **no custom
kernels**, so the same code runs on every backend (CPU, WGPU, CUDA, Metal,
LibTorch, …).

**[`burn-stack`](../burn-stack)** (`../burn-stack/CLAUDE.md`) holds everything
*around* the block: layers, (virtual-)layer stacks, bidirectional pairs,
latent/vocab networks, multi-gate residuals, class tokens, schedules, the Muon
plan. It is block-agnostic by construction. This crate supplies the three
`Block` implementations and the runtime-selectable `Mamba*` enums in
`src/unified/`. **Never put anything mamba-specific into `burn-stack`**: no
name, no shape assumption, no doc reference. If it needs one, it belongs here.

## Build & Test Commands

```bash
cargo check                 # type-check the lib surface
cargo test --lib -- --test-threads=1 # run lib tests (any backend; flex = CPU default)
cargo test --examples -j 1 -- --test-threads=1 # run example tests (any backend; flex = CPU default)
cargo doc --all --no-deps   # build docs
cargo run --example reset-majority -- --training --inference
# benches run by the user, not by you:
./bench.sh
./kernels.sh
```

- **Feature flags select the backend**: `backend-{flex,cpu,wgpu,metal,vulkan,cuda,
  rocm,tch-cpu,tch-gpu,remote,ndarray}`. Use flex for checks and tests (it is
  on by default). Each feature enables the matching `burn/<backend>`. Several
  can be compiled in at once, and `Device::default()` resolves which one to use
  (it reads `BURN_DEVICE`).
- `mamba1`/`mamba2`/`mamba3`/`autodiff`/`optim` are on by default.
  `mamba2`/`mamba3` imply `autodiff`. `optim` (Muon parameter groups) implies
  `burn/optim` + `burn/std`. `cubecl`/`fusion` enable the memory-saving custom
  backward on those backend families. `check-nan`/`check-inf` enable the NaN/Inf
  guards. `dev-f16`/`dev-simd`/`dev-autotune` are example/test conveniences.
- Every feature above **forwards to `burn-stack`** (see `Cargo.toml`), and it
  must. The `backend-*` cfgs are evaluated where
  `burn_stack::impl_backend_ext_for_burn_backends!` expands, which is in *this*
  crate. If a backend is added on one side only, it silently loses its
  `BackendExt` impls.
- `Cargo.toml` `[patch]`es every burn and cubecl crate to the swfsql forks with
  the tracel-ai/burn#5772 memory fix (a captured graph holds the memory of one
  pass, not ~3), as burn-stack does. A crate that is missing from the list
  links a second copy.

## Writing Style

- **Always load the `asd-ste100` skill** before you write code comments,
  rustdoc or markdown documents. Write all of them in its style.

## Documentation Maintenance (CLAUDE.md & files.md)

- Keep **both files as small as possible**, but still usable. Point to the
  source (each module header carries the detailed math and notation). Do not
  copy it here. When a source file changes, update its one entry. Do not grow
  these files.
- **Never use either file as a changelog.** They describe the code as it *is
  now*. They must not record individual changes, migrations, "used to be /
  now", "verified by", dates, or PR history. If you find changelog-style prose,
  delete it.
- Always be **extremely succinct** when you add content to either file.
- `examples/README.md` documents `examples/`, not this file.
- **Commit messages**: the user can ask for a commit message for the session.
  **Write the message as text only** (a title line + a short body) for the user
  to copy. Do NOT run `git commit` or any git command to create the commit.
  End the message with the `Co-Authored-By:` trailer.

## File Map

`../` contains external reference material (see
[Extra References](#extra-references)). Every leaf module has a sibling
`tests.rs` (forward/step parity, gradients, cross-variant agreement). This map
does not list them. The composition layer is `../burn-stack/`, with its own
File Map.

```text
src/
├─ lib.rs            crate root: module decls, prelude, `pub use burn_stack`
├─ mamba1/           original selective SSM (conv1d + sequential selective scan)
│  ├─ mamba1.rs      Mamba1 block + Config: forward() (selective_scan) / step();
│  │                 Mamba1Untied
│  └─ cache.rs       Mamba1Cache(s): conv window (bik) + SSM state (bir)
├─ mamba2/           SSD (Structured State Space Duality)
│  ├─ mamba2.rs      Mamba2 block + Config: chunkwise forward() / recurrent step();
│  │                 Mamba2Untied (InProjTail ⇒ in_proj_tail)
│  ├─ cache.rs       Mamba2Cache(s): conv window (bvk) + SSM state (bhpr)
│  └─ ssd/           ssd_path.rs selector; minimal / serial / serial_recalculated
├─ mamba3/           trapezoidal SSD + data-dependent RoPE + MIMO + MambaProduct
│  ├─ mamba3.rs      Mamba3 block + Config (header = the math reference);
│  │                 forward()/step() dispatch by cache variant; Mamba3Untied
│  ├─ helpers.rs     shared by both pathways and both modes: trapezoid masses,
│  │                 QK-norm+GQA+bias, MIMO-V, mimo_outer_sum, split_trailing,
│  │                 the tap-lag helpers, prefix_sum (the blocked scan for every
│  │                 sequence-length cumsum), and the read axis (read_rows /
│  │                 read_causal_mask, + `prim` twins with scatter_read_rows)
│  ├─ cache.rs       Mamba3Cache(s) ENUMS: DoubleSsd | SingleSsd, + From moves
│  ├─ ssd_path.rs    pathway-agnostic Mamba3SsdPath (From<> both sub-paths);
│  │                 optimal_chunk_len, chunk_tokens, backward_chunk_group
│  ├─ trapezoid.rs   Trapezoid (the β tap pattern, a closed 2×3 lattice),
│  │                 tap_lag / has_interior_tap, TrapezoidSpec
│  ├─ double_ssd/    one SSD pass per trapezoid term; cache.rs + ssd/ kernels;
│  │                 step_double_ssd is the decode for both cache variants
│  ├─ single_ssd/    one-pass official-kernel form (≈½ memory); cache.rs (h') +
│  │                 ssd/ (ssd/diag.rs: same-step γ-correction, SISO-branched) +
│  │                 token_band.rs (the lag-u band, outside the kernel)
│  ├─ rotation/      RotationKind (Real1D | Complex2D | Quaternion4D | Rotor4D),
│  │                 RotationState, RotationSpec, rotate_bc_forward (the one
│  │                 entry point, also for step), quat algebra; rope.rs
│  ├─ product/       MambaProduct: `micro_steps` (u) steps per token, folded into
│  │                 the sequence axis; header = the reference for the dial
│  ├─ positive/      per-head scalar systems beside the plant: scan.rs (one
│  │                 log-semiring scan), kalman.rs (the computed decay),
│  │                 tropical.rs (the max-plus register)
│  └─ quat_scan/     memory-efficient quaternion cumprod scan (recompute backward)
├─ padding.rs        right padding in a block: per-slot `window` gather,
│                    `fill_padded`, `repeat_rows` (token mask → folded axis)
└─ unified/          the runtime-selectable API + where the families plug in
   ├─ mod.rs         MambaSsdPath; header = why the MIMO 3-D tensors are not
   │                 stacked matrices for Muon
   ├─ cache.rs       MambaCaches (+ detach()) + impl Block / BlockConfig /
   │                 CacheStack for each family
   ├─ capture.rs     CacheTensors traversal of each family (burn-stack's trait),
   │                 through which a CapturedStep writes its new cache back
   ├─ network.rs     MambaLatentNet / MambaVocabNet (+ Configs, ModelConfigExt)
   ├─ bidi.rs        MambaBidiLayers (+ Config)
   └─ tests/         the burn-stack containers with real blocks: layer, layers
                     (grad_horizon), multi_gate, bidi, class, optim, untied,
                     capture
```

The generic containers (`Layer`/`Layers`/networks/bidi/multi_gate/class
tokens/schedules/norms/losses/Muon) are in `burn-stack` (see its File Map).

```text
benches/layer.rs     single-block benches (forward/train/step) — see bench.sh
bench.sh             runs them per backend, writes bench.md
kernels.sh           counts kernel launches per case, writes kernels.md
info/                standalone reference notes (committed); see files.md
scripts/             their numerical checks (python3 + numpy, standalone)
```

`files.md` is the per-file signature reference for **this** crate (what each
important file defines + the non-obvious decisions). The `mamba2.rs` /
`mamba3.rs` module headers hold the detailed math. Start a search from
`files.md`.

---

## Architecture

### Layer → Network hierarchy (all families)

All three families share **one** set of generic composition types. They are
in `burn-stack`, parameterised by the SSM core block `M`
(`Mamba1`/`Mamba2`/`Mamba3`):

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj (continuous I/O)
Layers<M>         a stack of N (virtual) layers over R real weight sets
Layer<M>          Pre-LN residual:  y = x·residual_scale + Block(RMSNorm(x))
M (Block)         the SSM core (mamba1.rs / mamba2.rs / mamba3.rs) — this crate
```

A family joins the stack when it implements `burn_stack::modules::{Block,
BlockConfig}` and `CacheStack` on its `Caches` (all in `src/unified/cache.rs`).
`Block::Options` is the per-call SSD-path selector.

The LM head of `VocabNetwork` is tied to the embeddingᵀ (`missing_lm_head`) or
is a separate `Linear`. The runtime enums `MambaVocabNet` / `MambaLatentNet` /
`MambaBidiLayers` (each with a `#[derive(Config)]` `*Config`) select the family
at construction. They panic on a cache or SSD path of the wrong family.

### Dual execution modes

Every block/layer/network has **`forward()`** (parallel chunkwise: training +
prefill) and **`step()`** (recurrent: token-by-token decode, O(state) per
token, no growing KV cache). `forward()` from any cache equals `step()`
unrolled from that same cache. The test suites assert parity on **outputs,
final cache, and gradients**.

CUDA graph capture (burn-stack) replays fixed-shape work from one captured
graph:

- A decode `step()` (`CapturedStep`, through `unified/capture.rs`), in the
  `generate` of tiny-stories. `--no-graph` runs eagerly in every example, with
  the same text.
- A fixed-shape `forward` with caches `()`: the validation of mnist-class and
  mnist-ae, one capture per pass.
- A prompt prefill as right-padded fixed-shape chunks that carry the cache
  (burn-stack's `Prefill`, in the `infer` of tiny-stories).
- Under plain SGD (`--sgd`), the whole training step of every example except
  tiny-stories (burn-stack's `examples::trainer::Trainer`), bit-identical to
  eager.

Dataloader workers build batches on the host (`loader_device`). A worker that
uploads from its own thread silently breaks a capture.

Layer containers and networks also have **`prime()`**: `step()` without a user
token. It emits the class tokens/latents that wait for the next token and
returns the last of them (`None` if there were none), for seedless generation.
`prime` then `step` runs exactly what that `step` alone would run.

`forward()` also takes `pad: Option<Tensor<2, Bool>>` (`true` at padding,
**right** padding per slot). A padded row is absent: the real outputs and the
cache of each slot are those of that slot run alone. Burn-stack keeps the mask
right-padded around class markers. A family zeroes the decay and the writes of
the step where its discretisation forms them:

- Mamba-1/2: `Δ = 0`.
- Mamba-3: `TrapezoidCoeffs::padded`, per token over the `u` micro-steps. The
  Kalman gate, the single-SSD key scale and the Δ-paced rotation all follow.

A family reads every "last samples" cache field (conv window, tap FIFO,
`positive/` carries) at the end of each slot (`padding::window`).

### Caches

Caches carry streaming state between calls. Mamba-1/2 caches hold a conv
window + SSM state. **Mamba-3 has no conv cache** (it removes the short conv).

### SSD algorithm selection (Mamba-2 & Mamba-3)

An `…SsdPath` enum makes the chunkwise scan pluggable. Each variant carries an
optional chunk length. `None` ⇒ the optimum ≈ `√(state_rank·per_head_dim)`, a
multiple of 32, capped at 512. Mamba-3 divides that by `mimo_rank`, then rounds
it up to a multiple of `micro_steps`: `m` widens both chunk axes, `u` only the
write axis (`info/mamba-3/architecture-deltas.md` §8).

| Variant | Algorithm | Backward |
|---------|-----------|----------|
| `Minimal` | batched matmuls + `segsum` mask | autodiff |
| `Serial` | serial loop over chunks (mirrors Triton K1–K5) | autodiff |
| `SerialRecalculated` | serial loop, recompute backward | **custom** (~⅓ less memory) |

`Default = SerialRecalculated(None)`. All three are exact reformulations and
must agree on values **and** gradients (the `ssd_path` tests assert this). A
learnable initial state works only with `Minimal`. Each family has a
`…BackendExt` trait whose default body works for any plain backend. Only
`Autodiff<B>` gets the custom backward. `backend_macros.rs` (burn-stack) emits
the per-backend impls. `combined_grad.rs` (burn-stack) flattens
`(y, final_state)` into the one tracked tensor that Burn's `prep.finish` wants.

### The three families

Read the `mamba2.rs` and `mamba3.rs` module headers for the full math and the
notation. The essentials:

- **Mamba-1**: selective SSM. in-proj → causal conv → SiLU →
  `x_proj`/`dt_proj` → **sequential `selective_scan`** (ZOH A, Euler B) → SiLU
  gate → out-proj. A is input-independent.
- **Mamba-2**: SSD. in-proj `[z|xbc|dt]` → conv+SiLU → split `(x,B,C)` →
  discretise (`Ā=exp(Δ·A)`, `B̄=Δ·B`) → zero-pad to a `chunk_len` multiple
  (exact) → GQA-expand B/C → SSD path → gated RMSNorm(z) → out-proj. `step()` is
  the recurrence `hₜ = Āₜhₜ₋₁ + B̄ₜxₜᵀ`, `yₜ = Cₜᵀhₜ + Dxₜ`.
- **Mamba-3**: Mamba-2 plus four independent additions:
  1. **trapezoidal** discretisation (`h = αh + βB₋₁x₋₁ + γBₜxₜ`,
     data-dependent `A`/`λ`; `λ ≡ 1` gives Mamba-2),
  2. a **complex transition** (`A+iθ`), applied as **data-dependent RoPE** on
     B/C,
  3. **MIMO** (`mimo_rank > 1`),
  4. **MambaProduct** (`micro_steps = u > 1`).

  B/C get **QK-Norm before** the SSD (not a post gated norm), then a learnable
  ones-init per-(head, rank) bias, then the rotation. There is no short conv.
  The in-projection is
  `[z|x·u|B_raw·u|C_raw|dd_dt·u|dd_A·u|λ_raw·u|μ_raw·u|θ·u|r·u|a·u|b·u]`: only
  the per-micro-step segments widen (the last three are `positive/`).

  Cite these notes, do not restate them:
  - `info/mamba-3/architecture-deltas.md`: why the deleted conv and output
    norm are affordable, why the bias order and init are mechanisms, what
    data-dependent `A` buys.
  - `info/mamba-3/trapezoid-as-integration.md`: the trapezoid changes only the
    *linear* term of the local objective (`λ` is an operator-splitting
    parameter, and the single-SSD key scale `Δ̃ₛ` is where its two
    installments collapse). So it is orthogonal to the rotation and to
    `micro_steps`.
  - `info/mamba-3/mimo-as-batch.md`: MIMO widens the same linear term along
    **rank** (a minibatch of `M`, free keys, tied values, `G` unchanged). So it
    composes with everything, and a MIMO block *is* its SISO block at init.

  `Mamba3Config.trapezoid` (`mamba3/trapezoid.rs`) selects *which* earlier
  sample(s) the `β` tap reads. The choice exists only at `u > 1`, and it selects
  an algorithm and a cache layout. The lattice is **closed**: (lag-1 tap absent
  | `Reset`, gated to within a token | `CarryOver`) × (lag-`u` tap absent |
  present) gives `None`, `HorizontalReset`, `HorizontalCarryOver` (default),
  `Vertical`, and `VerticalPlusHorizontal{Reset,CarryOver}`. Key facts (the
  module header has the rest):
  - The single-tap members are **one algorithm at two lags**:
    `Trapezoid::tap_lag(u)` is what every tap site reads.
  - The two-tap members add a lag-1 tap, mixed in by a second per-head mass
    `μ`. That is still one scalar per sample, so single-ssd stays **one** pass,
    and double-ssd needs one pass per tap.
  - A closed tap gives its mass back (interior → far, far → `γ`). So
    `HorizontalReset` *is* `HorizontalCarryOver` with `λ = 1` at the start of
    each token, and `VerticalPlusHorizontalCarryOver` *contains* both
    single-tap members (`μ ≡ 1` / `μ ≡ 0`).
  - At `u = 1` everything folds bit-exactly: the lag-`u` members are the
    carry-over (with no `μ`), and `HorizontalReset` is `None`.
  - `None` is structural: no `λ` segment in the in-proj or Muon, no `β`
    tensor, no tap slots in either cache, **one** SSD call in `forward` (so
    the pathways are the same and `forward_single_ssd` delegates), and one
    outer product in `step`.

### Mamba-3: two SSD pathways (the central design point)

Two **interchangeable algorithms** compute the trapezoidal recurrence. The
**cache variant** that the caller supplies selects one at runtime.
`Mamba3Cache`/`Mamba3Caches` are `DoubleSsd | SingleSsd` enums. A missing cache
selects SingleSsd.

- **Double-SSD** (`double_ssd/`): the trapezoid as **standard** SSD calls: a
  γ-SSM for the current token + one β-SSM per tap ("shift-before-chunking"),
  summed. Simple and easy to verify, ~2× memory. `step()` runs this recurrence
  directly, for both cache variants.
- **Single-SSD** (`single_ssd/`): one SSD call (the official Triton/Tilelang
  form) with a composite key scale, a strict lower-triangular mask, a
  same-step γ correction, and a boundary-β seed. ≈½ the training memory. Its
  accumulator `h'` has different semantics mid-sequence (a distinct cache
  type, so a chunked pass cannot mix the two). It is equal to the double-ssd
  state at call boundaries, hence the field-identity `From` conversions in
  `mamba3/cache.rs`. `step_single_ssd` decodes through the double-ssd cache.
  Under a lag-`u` pattern the γ correction widens to a `u`-band (of the lag-`u`
  mass only), which *is* the token at the reads that survive. So it runs
  outside the kernel (`single_ssd/token_band.rs`). The pathways then agree on
  everything that a caller observes (output + every cache field). The
  mid-token partial sums that would differ are never computed.

`Mamba3SsdPath` is pathway-agnostic and `From`-converts to either. The inputs
are different: double takes pre-scaled `v_bnlmhp`, and single takes raw `v` +
`gamma_bnth` + `scale_bnlh`. Both take `C` on the read axis (`c_bntmhr`) and a
`read_stride`, and return `y` at token resolution.

### Mamba-3: rotation (complex transition, a.k.a. "RoPE")

**Not a positional encoding.** It is the imaginary part of the *state
transition* (`hₜ = αₜRₜhₜ₋₁ + …`). `α` is scalar and `R` orthogonal, so the
cumulative rotation telescopes out of the state, and B/C absorb it (the "RoPE
trick"). The SSD core stays the plain scalar-decay kernel. The angles are
data-dependent, not a fixed frequency schedule. That buys state-tracking
(parity, mod-k), and it is why a step difference `θⱼ−θᵢ` is accumulated
rotation, never a position. The same argument holds for the quaternion kinds.

The rotation is per (head, plane) and is **broadcast over the MIMO ranks**.
This is necessary: the `M` ranks share one state, so they share its
transition, and per-rank angles have no state-space preimage
(`info/mamba-3/mimo-as-batch.md` §7).

`mamba3/rotation/` has the details. Key points:

- **`Complex2D`** (default, abelian `SO(2)`): angles projected, squashed to
  `range·π·tanh(·)`, Δ-scaled per head, then **`helpers::prefix_sum`** along
  the sequence (not `cumsum`, which costs `O(len²)` on cubecl), absorbed into
  B/C. `wrap_angle` reduces mod `2π` for fp16 stability. `rope_fraction`
  (0.5 | 1, default 1) rotates a prefix. SISO uses interleaved pairs, MIMO
  half-and-half.
- **`Real1D`** is the trivial group (a real transition). To switch the
  rotation off, select this *kind* (`rope_fraction` has no 0). It is
  structural: no rotation segment in the in-proj (hence `split_trailing`), the
  tensor-less `RotationState::Real`, B/C unchanged, and no rotation segment in
  `muon_projections()`. It is the only kind that accepts an **odd**
  `state_rank` (down to the scalar `1` of `reset-majority`). Ladder:
  `Real1D ⊂ Complex2D ⊂ Rotor4D` and `Real1D ⊂ Quaternion4D ⊂ Rotor4D`.
- `rotation_range` bounds one step to `range·π·Δ`. The default is **2** for
  every kind: one full traverse of the group per unit Δ (per factor for
  `Rotor4D`). The bound buys gradients, not reach: a rotation *at* the bound
  is on the asymptote of `tanh`, where the f32 derivative is exactly zero. So
  at `range = 1`, descent cannot reach the half-turn. `rotation_range = 1` +
  `rope_fraction = 0.5` is the reference model.
- **`Quaternion4D`** (`SU(2)`, non-abelian): the cumulative rotation is an
  associative **scan** with a cross-chunk carry (`quat_scan/` has the
  recompute backward). The B/C factoring is unchanged. It runs on **both**
  pathways. The generator bounds its **magnitude** (so the axis is the
  direction of the projection), and the axis is projected **per head** (heads
  with one shared axis track one word at different speeds).
- **`Rotor4D`** is the **whole** `SO(4)` of a 4-block, the two-sided
  `v ↦ q⊗v⊗p̄`. The conjugation reverses the right-hand order *twice*, so `T`
  accumulates by the **same left fold** as `Q`. Both factors stack on one block
  axis, and every quaternion primitive runs once over `2·blocks`. `L_q` is
  *isoclinic*, so `Quaternion4D` cannot express two independent plane angles
  and does **not** contain `Complex2D`. `Rotor4D` contains both (plane angles
  `a∓b`) and the adjoint `SO(3)` (`p = q`). `SO(4)` is the ceiling for `k = 4`
  (`k = 8` would break the scan: octonions are non-associative).

### Mamba-3: MambaProduct (`micro_steps`)

The dial of DeltaProduct, not its mechanism (`mamba3/product/`, whose header is
the reference). `u = micro_steps` full Mamba-3 steps per token, each with its
own `x`, `B`, `Δ`, `A`, `λ` and rotation. The transition of a *token* is the
**product** `(∏ⱼαⱼ)·R_{u−1}⋯R₀`. `u = 1` is stock, byte for byte.

- Both families are `Mₜ = ∏ⱼ (I − ηⱼ∇²Lⱼ)`, but they turn different dials:
  DeltaProduct turns the **curvature** (non-commuting rank-one factors). The
  curvature of Mamba is isotropic, so every factor is a scalar and cannot
  rotate. The rotation must come from the **step size** leaving `ℝ`, which is
  `RotationKind` (`info/mamba-3/rotation-as-optimization.md`).
- Evaluation: the micro-steps fold into the **sequence axis** (the existing
  pipeline runs at length `sequence·u`). No new kernel, no cache change. The
  fold is on the **writes only**: `C`, the gate `z`, the `D` skip and the
  output stay at token resolution. That is the **read axis** of the chunk
  (`helpers::read_rows`, `read_stride`): a chunk is `chunk_len` writes by
  `chunk_tokens = chunk_len/u` reads, so the score and the readout are
  u-invariant.
- `step` solves the `u`-position block **in closed form**, so a decode step
  costs the same launches at every `u`. Its taps, FIFO and rotation are
  `forward`'s own helpers.
- What `u` buys depends on the `RotationKind`:
  - `Real1D` widens only the write (`MambaProduct(u=M)` ⊇ `MIMO(M)`, which is
    also why the dial is not on Mamba-2).
  - `Complex2D` gets `u`× the angle reach, with a live gradient at every
    factor.
  - The non-abelian kinds get a product that no single bounded step can
    express.

### Mamba-3: positive systems beside the plant

`positive/` adds a per-head **scalar** recurrence that reads only the
in-projection and sets the coefficients of the plant. It is a cascade, so the
chunkwise pass still works (a gain that read the state would not). Both members
are nonnegative 2×2 matrices that act projectively, in log coordinates, so one
scan serves both:

- `Gain::{Projected (default), Kalman, KalmanProjectedNoise}` computes the
  decay from an accumulated precision `Λ`. The block substitutes it inside
  `helpers::trapezoidal_coefficients` *before* `α` is formed, so every consumer
  follows. `Λ` is exact for a lag-1 tap, an upper bound at lag `u`.
- `Tropical::{None (default), MaxPlus}` adds a soft `max(c + a, b)` register
  to the readout.

The ports are the decay, the read `(Λ+ε)^(−ω)` (before the `D` skip, so
`has_outproj_norm` keeps it), and `y += c·e`. The masses of the gate are logs
of pre-activations, never `ln` of a mass (it underflows, and `0·∞` is a NaN
gradient). Both are structural, stock exactly at `κ = 0` / `e = 0`, with one
cache slot each. What each buys: `info/kalman/gate-as-positive-system.md`
(cite it, do not restate it).

### Virtual layers, bidirectional, class tokens, multi-gate

These are `burn-stack` features, documented in `../burn-stack/CLAUDE.md`. Here:
they are family-agnostic, `src/unified/tests/` tests each one against real
blocks, and the runtime enums (`MambaLatentNet` / `MambaBidiLayers`) wrap the
generic containers (they do not reimplement them).

---

## Key Design Decisions

- **No optimized kernels**: only the portable tensor ops of Burn, so one code
  path runs on every backend.
- **Dispatch backend (Burn 0.22+)**: the high-level `Tensor` (every `Module`)
  is pinned to the global `Dispatch` backend, so library types are **not
  backend-generic** (`Mamba2`, `Mamba2Cache`, … have no `<B>`). The backend is
  a runtime `Device`. Autodiff and dtype are device properties. Only the
  custom-backward internals stay generic over `B` (`F<B,D>`, the
  `Backward<B,_>` nodes, the `Autodiff<B>` ext impls).
- **A no-grad region means the inner backend, not `detach`** (`burn-stack`,
  see its `utils/detach.rs`). So each family's `Caches` must implement
  `CacheStack::cache_to_inner`/`cache_from_inner` **by hand**. `Module::map`
  does nothing on plain `Tensor` fields, which is all a cache holds, so a
  `Module`-based conversion would silently skip every field.
- **Two Mamba-3 SSD pathways**: the cache type selects double-ssd (simple) or
  single-ssd (~½ memory). The accumulators are equal at boundaries, so the
  caches convert.
- **SISO is `mimo_rank = 1`, not a separate implementation.** The fused `L·M`
  axis is then `chunk_len`, so each kernel already *is* its SISO form. Code
  branches on `m` only where it is a real matmul dimension: RoPE pairing
  (semantic), and two **performance-only** flags (identical values and
  gradients) with different backend preferences:
  - `Mamba3Config.siso_specialization`, for the chunkwise γ-correction
    (`single_ssd/ssd/diag.rs`). It deletes thousands of tiny GEMMs and wins
    everywhere.
  - `siso_specialization_decode`, for the per-token sites
    (`helpers::mimo_outer_sum`, `step_readout`, through
    `Mamba3::use_siso_decode_kernels`). It replaces one good GEMM with a
    broadcast, so it wins on GPU and loses badly on CPU.
- **Three SSD algorithm variants**, the last with a custom recompute backward.
  Tests prove them equal on values + gradients.
- **MambaProduct is a sequence fold, not a kernel.** The `u` micro-steps of a
  token are `u` consecutive positions of the existing recurrence, so only the
  per-micro-step in-proj segments widen, and the state and caches do not
  change. The SSD kernels take one number from it, `read_stride`. Unlike
  DeltaProduct, every micro-step has its own decay: Mamba has no forget gate
  separate from its step size (`α = exp(ΔA)`, and `Δ` also weights the write
  and paces the rotation), so `α ≡ 1` on the interior steps would also stop
  the rotation. The `u` steps are full-size: the effective interval of a token
  is `u`× longer, not subdivided. That buys the reach. The consistent
  alternative (`Δⱼ = Δ/u`) stays reachable, so this is a superset.
- **A gate can read the inputs, never the plant.** The systems of `positive/`
  are a cascade before the LPV plant, which keeps the pass chunkwise.
- **Muon sees split projections, the model does not.** The machinery is
  `burn_stack::optim`. This crate owns the **allowlist**: one
  `muon_projections()` per family config, with the same column widths as the
  `split_into` of the forward. Per-head *scalar* channels
  (Δ/`A`/`λ`/`μ`/`r`/`a`/`b`), every 1-D/3-D tensor, and the boundary weights
  stay on the fallback (AdamW or SGD). The `src/unified/mod.rs` header argues
  why the MIMO 3-D tensors are diagonals, not stacked matrices.
- **Untied parameters are declared, not re-plumbed.** The mechanism is
  `burn_stack::utils::untied`. Each family lists what it can hold once per
  application (`Mamba{1,2,3}Untied` on its config), tiles it in
  `init_applications`, and reports it with `untied_params`. The `InProjTail`
  of Mamba-2/3 moves the trailing scalar (+ rotation) segments of `in_proj`
  into `in_proj_tail` for any application count, so one tiled Muon spec fits
  every real layer. `project_in` joins them again. `init_state_hpr` is
  untiable, but `step` never reads it.
- **`#![warn(missing_docs)]`**: keep the crate free of warnings. Document the
  public surface when you add it. `cargo doc --all --no-deps` must also be free
  of warnings.
- The project root is `/shared/claude/burn-mamba/`. Do not read or write
  outside it.
- When a source file is added, removed or changed, prepare an update to its
  entry in the [File Map](#file-map) and in `files.md` (per the maintenance
  rules above). A change to a composition type updates
  `../burn-stack/CLAUDE.md` instead. Important rule: do this at the end of the
  work. If you have not read those files by then, **do not** read them: the
  context is still big from the work, and reading big files then is expensive.
  Instead, write a `tmp.md` file with the new [File Map](#file-map) entry and
  a short overview of the most important aspects of the created/removed/
  updated files. After a full context reset (the user triggers it), those
  files get their update.

---

## Notation

Tensor names carry a shape suffix. The codebase is **deliberately verbose**
about this (backed by shape `assert`s). A name whose suffix encodes its shape
needs no extra comment. In commentary, a shape can be underscore-style (`_bhl`)
or expanded to `[...]`. **Paper** style (upper-case `A,B,C,H,Y,L,…`) can appear
in comments but **never in code identifiers**. Lower-case = base dimensions
(below). Upper-case = a *relation* of them (offset/multiple/concat): `X` can be
`x±1`/`x*2`/etc., and `XY` can be `x+y`/`x*y`/etc.

| Letter | Dimension | Paper | Python | Typical |
|--------|-----------|-------|--------|---------|
| `b` | `batch` | — | `batch` | varies |
| `s` | `sequence` length, **folded** = `tokens`·`u` | `T` | `seqlen` | varies |
| `t` | `tokens` = `s`/`u` — the read axis (Mamba-3) | `T` | `seqlen` | varies |
| `u` | `micro_steps` (Mamba-3) | — | — | 1 (stock) |
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

## Extra References

Under `../` (not analyzed here):

- the **Mamba-3 paper** TeX (`../papers/mamba-3/`),
- the **official Python impl** (`../py/state-spaces/mamba/`). It is
  authoritative, and its Triton SISO / Tilelang MIMO kernels are the
  single-ssd reference,
- **Mamba-3 minimal**, the base of double-ssd (`../py/VikramLex/mamba3-minimal/`),
- **Burn** (`../burn/`).

## Custom Commands

- `rg`: available.
- `cargo fmt`: do not use.
- **Always** edit files with the Edit/Write tools. This applies also when a
  harness or auto-mode reminder says to change files through Bash (`sed`,
  heredocs, python). That guidance does not apply here. *Do not* violate this.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`. Always use
    `Edit`s.
  - Bash stays the tool for *reading* and *inspecting* (`cat`, `sed -n`, `rg`,
    `grep`) and for throwaway files outside the crate (for example, in `/tmp`).
