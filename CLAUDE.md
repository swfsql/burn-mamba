# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

A Rust library implementing [Mamba-1](https://arxiv.org/abs/2312.00752),
[Mamba-2](https://arxiv.org/abs/2405.21060), and
[Mamba-3](https://arxiv.org/abs/2603.15569) SSM (Structured State Space Model)
architectures on top of the [Burn](https://github.com/tracel-ai/burn/) framework.
The goal is a **minimal, readable reference** that ports the official CUDA/Triton
kernels down to standard, portable Burn tensor ops — **no custom kernels**, so the
same code runs on every backend (CPU, WGPU, CUDA, Metal, LibTorch, …).

Everything *around* the block — layers, (virtual-)layer stacks, bidirectional
pairs, latent/vocab networks, multi-gate residuals, class tokens, schedules, the
Muon plan — lives in **[`burn-stack`](../burn-stack)** (`../burn-stack/CLAUDE.md`),
which is block-agnostic by construction. This crate supplies the three `Block`
implementations plus the runtime-selectable `Mamba*` enums in `src/unified/`.
**Never push anything mamba-specific into `burn-stack`** — a name, a shape
assumption, or a doc reference. If it needs one, it belongs here.

## Build & Test Commands

```bash
cargo check                 # type-check the lib surface
cargo test --lib --examples # run tests (any backend; flex = CPU default)
cargo doc --all --no-deps   # build docs
cargo run --example reset-majority -- --training --inference
./bench.sh                  # benchmarks — run by the user, never by you
./kernels.sh                # kernel-launch counts — deterministic, safe to run
```

- **Feature flags select the backend**: `backend-{flex,cpu,wgpu,metal,vulkan,cuda,
  rocm,tch-cpu,tch-gpu,remote,ndarray}` (flex preferred for checks/tests, enabled 
  by default). Each just enables the matching `burn/<backend>`; several may be
  compiled in at once and `Device::default()` resolves which to use (honouring `BURN_DEVICE`).
- `mamba1`/`mamba2`/`mamba3`/`autodiff`/`optim` are default-on; `mamba2`/`mamba3` imply
  `autodiff`, and `optim` (Muon parameter groups) implies `burn/optim`+`burn/std`.
  `cubecl`/`fusion` enable the memory-saving custom backward on those backend families.
  `dev-f16`/`dev-simd`/`dev-autotune` are example/test conveniences.
- Every feature above **forwards to `burn-stack`** (see `Cargo.toml`). It must: the
  `backend-*` cfgs are evaluated where `burn_stack::impl_backend_ext_for_burn_backends!`
  expands, i.e. in *this* crate. A backend added on one side and not the other
  silently loses its `BackendExt` impls.

## Documentation Maintenance (CLAUDE.md & files.md)

- Keep **both files as minimal as possible while still viable**. Prefer pointing to
  the source (per-file module headers carry the detailed math/notation) over
  duplicating it here. When a source file changes, update its one entry — don't grow
  these files.
- **Never use either file as a changelog.** They describe the code as it *is now*;
  they must not record individual changes, migrations, "used to be / now", "verified
  by", dates, or PR history. If you catch changelog-style prose, delete it.
- Always be **extremely succint** when adding content to either file.
- `examples/` is documented by `examples/README.md`, not here.
- **Commit messages**: the user may ask for a commit message for the session. 
  **Just write the message as text** (a title line + a short body) for the user to copy
  — do NOT run `git commit` or any git command to create the commit.
  End the message with the `Co-Authored-By:` trailer.

## File Map

`../` contain external reference material (see [Extra References](#extra-references)).
Every leaf module has a sibling `tests.rs` (forward/step parity, gradients,
cross-variant agreement) — not listed individually. The composition layer is
`../burn-stack/` and has its own File Map.

```text
src/
├─ lib.rs            crate root: module decls, prelude, DENY_NAN/DENY_INF guards
├─ mamba1/           original selective SSM (conv1d + sequential selective scan)
│  ├─ mamba1.rs      Mamba1 block + Config: forward()(selective_scan) / step()
│  └─ cache.rs       Mamba1Cache(s): conv window (bik) + SSM state (bir)
├─ mamba2/           SSD (Structured State Space Duality)
│  ├─ mamba2.rs      Mamba2 block + Config: chunkwise forward() / recurrent step()
│  ├─ cache.rs       Mamba2Cache(s): conv window (bvk) + SSM state (bhpr)
│  └─ ssd/           ssd_path.rs selector; minimal / serial / serial_recalculated
├─ mamba3/           trapezoidal SSD + data-dependent RoPE + MIMO
│  ├─ mamba3.rs      Mamba3 block + Config; forward()/step() dispatch by cache variant
│  ├─ helpers.rs     shared: trapezoid coeffs, QK-norm+GQA+bias, MIMO-V build,
│  │                 split_rotation_channels (peels the in-proj's rotation tail)
│  ├─ cache.rs       Mamba3Cache(s) ENUMS dispatching DoubleSsd vs SingleSsd
│  ├─ ssd_path.rs    pathway-agnostic Mamba3SsdPath (From<> both sub-paths)
│  ├─ trapezoid.rs   Trapezoid: which earlier sample the β tap reads (None |
│  │                 Vertical | HorizontalReset | HorizontalCarryOver (default,
│  │                 the only implemented one) | VerticalPlusHorizontalReset).
│  │                 Structural: picks the shift, the γ-correction band and the
│  │                 cache's tap slots; init panics on the rest
│  ├─ double_ssd/    two-pass trapezoid (γ-SSD + β-SSD); cache.rs + ssd/ kernels
│  ├─ single_ssd/    one-pass official-kernel form (≈½ memory); cache.rs (h') + ssd/
│  │                 (ssd/diag.rs: same-step γ-correction, SISO-branched)
│  ├─ rotation/      transition rotation (Real1D | Complex2D | Quaternion4D | Rotor4D)
│  │                 + rope.rs (the mechanical pairwise rotation of the abelian path)
│  │                 + quat algebra; RotationSpec {kind,rope_dim,range}: the one
│  │                 per-step definition. Real1D = the trivial group: no in-proj
│  │                 channels, no cache accumulator, odd state_rank ok (scalar 1).
│  │                 Rotor4D = full SO(4), two-sided q⊗v⊗p̄ (both factors stacked
│  │                 on one block axis ⇒ one scan)
│  ├─ product/       MambaProduct: `micro_steps` (u) recurrence steps per token,
│  │                 folded into the sequence axis (no new kernel); u=1 is stock
│  ├─ quat_scan/     memory-efficient quaternion cumprod scan (recompute backward)
│  └─ step_constant/ constant-input shortcut: step_infinite (stationary fixed point)
└─ unified/          the runtime-selectable API + where the families plug in
   ├─ mod.rs         MambaSsdPath; module doc carries the Muon "3-D tensors are
   │                 not stacked matrices" argument (MIMO diagonals)
   ├─ cache.rs       MambaCaches enum (+ detach()) + impl Block / BlockConfig /
   │                 CacheStack for Mamba{1,2,3}(Config|Caches)
   ├─ network.rs     MambaLatentNet / MambaVocabNet (+ Configs)
   ├─ bidi.rs        MambaBidiLayers (+ Config)
   └─ tests/         the burn-stack containers exercised through real blocks:
                     layer, layers (grad_horizon), multi_gate, bidi, class, optim
```

The generic containers themselves (`Layer`/`Layers`/networks/bidi/multi_gate/
class tokens/schedules/norms/losses/Muon) are `burn-stack`; see its File Map.

```text
benches/layer.rs     single-block benches (forward/train/step) — see bench.sh
bench.sh             runs them per backend, writes bench.md
kernels.sh           counts kernel launches per case, writes kernels.md
info/                standalone reference notes (committed); see files.md
scripts/             their numerical checks (python3 + numpy, standalone)
```

`files.md` is the per-file signature reference for **this** crate (what each
important file defines + the non-obvious decisions). The detailed per-family math lives in the `mamba2.rs` /
`mamba3.rs` module headers. Always consider starting-off searching from `files.md`.

---

## Architecture

### Layer → Network hierarchy (all families)

All three families share **one** set of generic composition types, which live in
`burn-stack` and are parameterised by the SSM core block `M`
(`Mamba1`/`Mamba2`/`Mamba3`):

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj (continuous I/O)
Layers<M>         a stack of N (virtual) layers over R real weight sets
Layer<M>          Pre-LN residual:  y = x·residual_scale + Block(RMSNorm(x))
M (Block)         the SSM core (mamba1.rs / mamba2.rs / mamba3.rs) — this crate
```

A family joins the stack by implementing `burn_stack::modules::{Block,
BlockConfig}` and `CacheStack` on its `Caches` (all three impls live in
`src/unified/cache.rs`). `Block::Options` is the per-call SSD-path selector.

`VocabNetwork`'s LM head is tied to the embeddingᵀ (`missing_lm_head`) or a separate
`Linear`. Runtime-dispatch enums `MambaVocabNet` / `MambaLatentNet` /
`MambaBidiLayers` (each with a `#[derive(Config)]` `*Config`) pick the family at
construction and panic on a family-mismatched cache/ssd_path.

### Dual execution modes

Every block/layer/network exposes **`forward()`** (parallel chunkwise: training +
prefill) and **`step()`** (recurrent: token-by-token decode, O(state)/token, no
growing KV cache). `forward()` from any cache equals `step()` unrolled from that same
cache — parity on **outputs, final cache, and gradients** is what the test suites
assert.

Layer containers and networks additionally expose **`prime()`** — `step()` without
a user token: it emits the class tokens/latents waiting for the next one and
returns the last of them (`None` if none were), for seedless generation. `prime`
then `step` runs exactly what that `step` alone would.

Mamba-3 additionally exposes **`step_infinite(x)`** (the stationary fixed-point
output under a constant token; no cache — the state orbits, only the output
converges; the limit composes exactly through `Layer`/`Layers`/networks and the
runtime enums, which panic for Mamba-1/2).

### Caches

Carry streaming state between calls. Mamba-1/2 caches hold a conv window + SSM state.
**Mamba-3 has no conv cache** (the short conv is removed).

### SSD algorithm selection (Mamba-2 & Mamba-3)

The chunkwise scan is pluggable via an `…SsdPath` enum; each variant carries an
optional chunk length (`None` ⇒ optimal ≈ `√(state_rank·per_head_dim)`, mult-of-32,
capped 512):

| Variant | Algorithm | Backward |
|---------|-----------|----------|
| `Minimal` | batched matmuls + `segsum` mask | autodiff |
| `Serial` | serial loop over chunks (mirrors Triton K1–K5) | autodiff |
| `SerialRecalculated` | serial loop, recompute backward | **custom** (~⅓ less memory) |

`Default = SerialRecalculated(None)`. All three are exact reformulations and must
agree on values **and** gradients (asserted by `ssd_path` tests). Each family has a
`…BackendExt` trait whose default body works for any plain backend; only `Autodiff<B>`
gets the custom backward. `backend_macros.rs` emits the per-backend impls;
`combined_grad.rs` flattens `(y, final_state)` into the one tracked tensor Burn's
`prep.finish` wants.

### The three families

Read the `mamba2.rs` and `mamba3.rs` module headers for the full math + per-file
notation tables; the essentials:

- **Mamba-1** — selective SSM: in-proj → causal conv → SiLU → `x_proj`/`dt_proj` →
  **sequential `selective_scan`** (ZOH A, Euler B) → SiLU gate → out-proj. A is
  input-independent.
- **Mamba-2** — SSD: in-proj `[z|xbc|dt]` → conv+SiLU → split `(x,B,C)` → discretise
  (`Ā=exp(Δ·A)`, `B̄=Δ·B`) → zero-pad to a `chunk_len` multiple (exact) → GQA-expand
  B/C → SSD path → gated RMSNorm(z) → out-proj. `step()` is the recurrence
  `hₜ = Āₜhₜ₋₁ + B̄ₜxₜᵀ`, `yₜ = Cₜᵀhₜ + Dxₜ`.
- **Mamba-3** — Mamba-2 plus four independent additions: **trapezoidal**
  discretisation (3-term `h = αh + βB₋₁x₋₁ + γBₜxₜ`, data-dependent `A`/`λ`; `λ≡1`
  collapses to Mamba-2), a **complex transition** (`A+iθ`) realised as
  **data-dependent RoPE** on B/C, **MIMO** (`mimo_rank>1`), and **MambaProduct**
  (`micro_steps=u>1`, below). B/C use **QK-Norm before** the SSD (not a post gated
  norm); no short conv. The in-projection splits
  `[z|x·u|B_raw·u|C_raw|dd_dt·u|dd_A·u|λ_raw·u|θ·u]` — only the per-micro-step
  segments widen. The trapezoid touches only the *linear* term of the local objective
  (`λ` is an operator-splitting parameter; `Δ̃ₛ`, single-ssd's key scale, is where its two
  installments collapse), so it is orthogonal to the rotation and to `micro_steps`:
  `info/trapezoid-as-integration.md` — cite it, don't restate it. MIMO widens that *same*
  linear term along **rank** — a minibatch of `M` with free keys and tied values, `G`
  untouched — which is why it composes with everything else and why a MIMO block *is* its
  SISO block at init: `info/mimo-as-batch.md` — cite it, don't restate it.
  *Which* earlier sample the trapezoid's `β` tap reads is `Mamba3Config.trapezoid`
  (`mamba3/trapezoid.rs`) — a lattice that exists only at `u > 1`, selecting an algorithm
  and a cache layout; the default `HorizontalCarryOver` (lag 1 on the **folded** sequence,
  so `1/u` of the taps cross a token) is the only implemented member, the rest panic in
  `init`.

### Mamba-3: two SSD pathways (the central design point)

The trapezoidal recurrence is realised by **two interchangeable algorithms**, chosen
at runtime by which **cache variant** is supplied (`Mamba3Cache`/`Mamba3Caches` are
`DoubleSsd | SingleSsd` enums; a missing cache defaults to SingleSsd):

- **Double-SSD** (`double_ssd/`) — splits the trapezoid into two **standard** SSD calls
  (γ-SSM current-token + β-SSM previous-token, "shift-before-chunking"), summed.
  Simple/verifiable, ~2× memory. `step()` runs this recurrence directly.
- **Single-SSD** (`single_ssd/`) — one SSD call (official Triton/Tilelang form) with a
  composite key scale, strict-lower-triangular mask, same-step γ correction, and a
  boundary-β seed. ≈½ the training memory. Its accumulator `h'` has different
  semantics mid-sequence (distinct cache type so the two can't be mixed in a chunked
  pass), but coincides with the double-ssd state at boundaries — hence the
  field-identity `From` conversions in `mamba3/cache.rs`. `step_single_ssd` decodes by
  round-tripping through the double-ssd cache.

`Mamba3SsdPath` is pathway-agnostic and `From`-converts to either. The inputs differ:
double feeds pre-scaled `v_bnlmhp`; single feeds raw `v` + `gamma_bnlh` + `scale_bnlh`.

### Mamba-3: rotation (complex transition, a.k.a. "RoPE")

**Not a positional encoding** — it is the imaginary part of the *state transition*
(`hₜ = αₜRₜhₜ₋₁ + …`). Since `α` is scalar and `R` orthogonal, the cumulative rotation
telescopes out of the state and is absorbed into B/C ("RoPE trick"), leaving the plain
scalar-decay SSD core. The angles are data-dependent, not a fixed frequency schedule —
that is what buys state-tracking (parity/mod-k), and why a step difference `θⱼ−θᵢ` is
rotation accumulated, never a position. Same argument for `Quaternion4D` below.

The rotation is per (head, plane) and **broadcast over the MIMO ranks**, necessarily: the `M`
ranks share one state, so they share its transition, and per-rank angles have no state-space
preimage at all (`info/mimo-as-batch.md` §7).

Default **`Complex2D`** (abelian `SO(2)`): angles projected, squashed to
`range·π·tanh(·)`, Δ-scaled per head, then **`cumsum`** along the sequence (continued
from the cache), absorbed into B/C. `wrap_angle` reduces mod `2π` (value-exact, the
offset `detach`ed) to stay fp16-stable over long sequences. `rope_fraction` (0.5/1,
default 1) rotates a prefix; SISO uses interleaved/NeoX pairing, MIMO half-and-half/GPT-J.

**`Real1D`** is the bottom rung: the trivial group, i.e. a real transition. Switching the
rotation off is a choice of *kind*, not a fraction of zero — `rope_fraction` only narrows a
rotation that exists, and `init` asserts a rotating kind turns at least one pair. The kind
is structural: every rotation count is `0`, so the in-projection has no rotation segment at
all (Burn drops a zero-length `split_with_sizes` part, hence `split_rotation_channels`), the
cache slot is the tensor-less `RotationState::Real`, `B`/`C` reach the SSD core untouched,
and `muon_projections()` omits the rotation segment. It has no pair to make, so it is also
the only kind `init` lets carry an **odd** `state_rank` — down to the scalar state `1`, the
`reset-majority` example. Ladder: `Real1D ⊂ Complex2D ⊂ Rotor4D`
and `Real1D ⊂ Quaternion4D ⊂ Rotor4D`.

`rotation_range` bounds one step to `range·π·Δ` and defaults to **2** for every kind:
one full traverse of the rotation group per unit Δ (`2π` is a whole turn of `SO(2)`, and
reaches every element of `SU(2)`, whose period is `4π`; for `Rotor4D` the bound applies
per factor, so the pair reaches all of `SO(4)`). The bound buys gradients, not
reach — a rotation *at* it sits on `tanh`'s asymptote, where f32's derivative is exactly
zero, so at `range=1` the half-turn state-tracking wants is unreachable by descent.
`rotation_range=1` + `rope_fraction=0.5` is the reference model.

`mamba3/rotation/` adds the two **non-abelian** kinds. `Quaternion4D` (`SU(2)`): the
cumulative rotation becomes an associative **scan** (with cross-chunk carry) instead
of a `cumsum`, while the B/C-factoring (so the scalar-decay SSD core) is unchanged.
Selected by `Mamba3Config.rotation: RotationKind`; the cache accumulator is a
`RotationState`. It runs on **both** SSD pathways (applied to B/C before chunking).
`quat_scan/` provides the memory-efficient recompute-backward version of the scan.
Two further differences from the abelian path: the generator's **magnitude** is bounded
(not each channel), so the axis is exactly the direction the projection names; and the
axis is projected **per head** (`nheads·3·num_rotation_blocks` channels), because for a
non-abelian transition the axis is the expressive part — heads sharing one and differing
only in Δ track one word at different speeds instead of different words.

`Rotor4D` is the **whole** rotation group of a 4-block, `SO(4) ≅ (SU(2)×SU(2))/±1`: the
two-sided `v ↦ q⊗v⊗p̄`. The factoring needs only that the per-step maps compose and are
orthogonal, so it survives verbatim — `Pₜ(v) = Qₜ v T̄ₜ` and `B̄ᵢ = Qᵢ*⊗Bᵢ⊗Tᵢ` — and the
conjugation reverses the right-hand order *twice*, so `T` accumulates by the **same left
fold** as `Q`. Both factors therefore stack on one block axis and every quaternion
primitive (generator split, exp map, scan, renormalise) runs once over `2·blocks`,
unbranched; only the application to B/C differs, at one extra `quat_mul`. Its cache
accumulator is `RotationState::Rotor`, and `rotation_range` bounds **each** factor, so
the default reaches every element of the group. Why it exists: `L_q` is *isoclinic* —
it turns both invariant planes by the same angle — so `Quaternion4D` cannot express two
independent per-pair angles and does **not** contain `Complex2D`; the two middle kinds
are incomparable and `Rotor4D` contains both (plane angles `a∓b`). It also contains the
adjoint `SO(3)` (`p=q`), where `±q` act identically — the difference between tracking a
group and tracking its double cover. `SO(4)` is the ceiling for `k=4`; `k=8` would break
the scan (octonions are non-associative).

### Mamba-3: MambaProduct (`micro_steps`)

DeltaProduct's **dial**, not its mechanism (`mamba3/product/`): `u = micro_steps` full
Mamba-3 steps per token, each with its own `x`, `B`, `Δ`, `A`, `λ` and rotation, so a
*token*'s transition is the **product** `(∏ⱼαⱼ)·R_{u−1}⋯R₀` and its write a sum of `u`
outer products staggered along it. `u=1` is stock, byte for byte.

Both families are `Mₜ = ∏ⱼ (I − ηⱼ∇²Lⱼ)`; they turn different dials in it. DeltaProduct
turns the **curvature** (rank-one `kⱼkⱼᵀ`, so factors with different `kⱼ` don't commute);
Mamba's curvature is isotropic, so every factor is a scalar and no product of them can
rotate — the rotation must come from the **step size** leaving `ℝ`, which is what
`RotationKind` is. Hence DeltaProduct's mechanism has no instance here, and the
`RotationKind` split below follows from the algebra of `η` alone. Derived, with the
2×2 design space and the two other readings of the same recurrence (min–max on a harmonic
potential; momentum), in `info/rotation-as-optimization.md` — cite it, don't restate it.

Evaluated by folding the micro-steps into the **sequence axis** — the `u`-wide
in-projection segments become `u` consecutive positions and the existing pipeline
(trapezoid, rotation scan, chunked SSD, padding, caches) runs at length `sequence·u`.
No kernel, no cache change: the state is one matrix at every `u`. The read `C` (broadcast
so its last copy carries the right cumulative rotation), the gate `z`, the `D` skip and
the output are per token. Cost is `u`× the recurrence plus `(u−1)·(d_inner+bc+3·nheads+
rot)` in-proj columns.

What `u` buys is decided by the `RotationKind`, and the split is sharp. `Real1D`: the
factors are scalars and commute, so it widens only the write — the *sequential* reading of
the cell `mimo_rank` occupies *jointly* (exactly: `MambaProduct(u=M)` reproduces a whole
`MIMO(M)` trajectory and the converse fails — `u` is MIMO with the values, step sizes and
rotation *untied*, at `u`× the recurrence), which is also why the dial does not exist on
Mamba-2. `Complex2D`: `u`× the per-token angle reach with a live gradient at every factor,
lifting exactly the `tanh`-asymptote bound `rotation_range` documents. `Quaternion4D`/
`Rotor4D`: a non-commuting product no single bounded step can express — DeltaProduct's own
argument, with the group given directly rather than factored into reflections.

`step_infinite` generalises to a sum of `u` terms but **exists only for the abelian
kinds** at `u>1`: the read-to-write relative rotation `P⁻ᵗQⱼPᵗ⁻ⁿ⁻¹` depends on `n` alone
iff `Qⱼ` commutes with `P`, so the quaternion kinds are almost-periodic, not convergent.
It asserts rather than approximating.

### Virtual layers, bidirectional, class tokens, multi-gate

All four are `burn-stack` features and documented in `../burn-stack/CLAUDE.md`.
What matters here: they are family-agnostic, every one of them is exercised
against real blocks by `src/unified/tests/`, and the runtime enums
(`MambaLatentNet` / `MambaBidiLayers`) wrap the generic containers rather than
reimplementing them.

---

## Key Design Decisions

- **No optimized kernels** — only Burn's portable tensor ops, so one code path runs on
  every backend.
- **Dispatch backend (Burn 0.22+)** — the high-level `Tensor` (every `Module`) is pinned
  to the global `Dispatch` backend, so library types are **not backend-generic**
  (`Mamba2`, `Mamba2Cache`, … carry no `<B>`). The backend is a runtime `Device`;
  autodiff and dtype are device properties. Only the custom-backward internals stay
  generic over `B` (`F<B,D>`, the `Backward<B,_>` nodes, `Autodiff<B>` ext impls).
- **A no-grad region means the inner backend, not `detach`** (`burn-stack`, see
  its `utils/detach.rs`). The consequence *here* is that each family's `Caches`
  must implement `CacheStack::cache_to_inner`/`cache_from_inner` **by hand**:
  `Module::map` is a no-op on plain `Tensor` fields, which is all a cache holds,
  so a `Module`-based conversion would silently skip every one of them.
- **Two Mamba-3 SSD pathways** — cache type selects double-ssd (simple) vs single-ssd
  (~½ memory); accumulators coincide at boundaries so caches inter-convert.
- **SISO is `mimo_rank = 1`, not a separate implementation** — the fused `L·M` axis is
  then `chunk_len`, so each kernel already *is* its SISO form. Only where `m` is a real
  matmul dimension do scopes branch on it: RoPE pairing (semantic), plus two
  **performance-only** flags (identical values and gradients) whose backend preferences
  differ — `Mamba3Config.siso_specialization` for the chunkwise γ-correction
  (`single_ssd/ssd/diag.rs`; deletes thousands of tiny GEMMs, wins everywhere) and
  `siso_specialization_decode` for the per-token sites (`helpers::mimo_outer_sum`,
  `step_readout`, `step_infinite`'s Gram, via `Mamba3::use_siso_decode_kernels`; replaces
  one good GEMM with a broadcast, so it wins on GPU and loses badly on CPU).
- **Three SSD algorithm variants**, the last with a custom recompute backward; proven
  equal on values + gradients by tests.
- **MambaProduct is a sequence fold, not a kernel** — `u` micro-steps per token are `u`
  consecutive positions of the existing recurrence, so only the per-micro-step in-proj
  segments widen and the state/caches/SSD paths are untouched. Unlike DeltaProduct every
  micro-step carries its own decay: Mamba has no forget gate separate from its step size
  (`α=exp(ΔA)`, and `Δ` also weights the write and paces the rotation), and pinning
  `α ≡ 1` on the interior steps would silence the rotation with it. A scalar decay
  composes, so the uniform placement costs nothing. The `u` steps are full-size, so a
  token's effective interval is inflated `u`×, not subdivided — that is what buys the
  reach; the consistent alternative (`Δⱼ=Δ/u`) is reachable, so this is a superset.
- **Muon sees split projections, the model does not** — the machinery is
  `burn_stack::optim`; what this crate owns is the **allowlist**, one
  `muon_projections()` per family config, listing the same column widths the
  forward's `split_into` uses. Per-head *scalar* channels (Δ/`A`/`λ`), every
  1-D/3-D tensor, and the boundary weights stay on AdamW. Why the MIMO 3-D
  tensors are diagonals and not stacked matrices is argued in the
  `src/unified/mod.rs` header.
- **`#![warn(missing_docs)]`** — keep the crate warning-clean; document public surface
  as you add it. `cargo doc --all --no-deps` must be warning-free too.
- The project root is `/shared/claude/burn-mamba/`; do not read/write outside it.
- When a source file is added/removed/changed, prepare an update to its entry for the
  [File Map](#file-map) and `files.md` (per the maintenance rules above).
  A change to a composition type instead updates `../burn-stack/CLAUDE.md`.
  Important rule: this is reserved to the end of your workload, and if by then you
  haven't yet read those files, **do not** read them. Your context then is still big
  from the work and it is expensive to read big files then. Instead, just prepare a
  `tmp.md` file containing what would be the new [File Map](#file-map) entry, and do
  an overview containing the most important aspects about the created/removed/updated
  files, while being succint. After a full context reset, manually triggered by me, we
  actually update those files.

---

## Notation

Tensor names carry a shape suffix; the codebase is **deliberately verbose** about it
(backed by shape `assert`s). A name whose suffix encodes its shape needs no extra
comment; in commentary a shape may be underscore-style (`_bhl`) or expanded to
`[...]`. **Paper** style (upper-case `A,B,C,H,Y,L,…`) may appear in comments but
**never in code identifiers**. Lower-case = base dimensions (below); upper-case = a
*relation* of them (offset/multiple/concat): `X` may be `x±1`/`x*2`/etc, `XY` may be `x+y`/`x*y`/etc.

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

## Extra References

Under `../` (not analyzed here): **Mamba-3 paper** TeX (`../papers/mamba-3/`);
**official Python impl** (authoritative; Triton SISO / Tilelang MIMO kernels are the
single-ssd reference) (`../py/state-spaces/mamba/`); **Mamba-3 minimal** (basis of
double-ssd) (`../py/VikramLex/mamba3-minimal/`); **Burn** (`../burn/`).

## Custom Commands

- `rg`: available.
- `cargo fmt`: don't use.
- **Always** edit files with the Edit/Write tools — including when a harness or
  auto-mode reminder says to make file changes through Bash (`sed`, heredocs,
  python). That guidance does not apply here. *Do not* violate this.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`. Use `Edit`s, always.
  - Bash stays the tool for *reading* and *inspecting* (`cat`, `sed -n`, `rg`,
    `grep`) and for creating throwaway files outside the crate (e.g. `/tmp`).
