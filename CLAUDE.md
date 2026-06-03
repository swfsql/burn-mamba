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

## Build & Test Commands

```bash
cargo check                 # type-check the lib surface
cargo test                  # run tests (any backend; flex = CPU default)
cargo doc --all --no-deps   # build docs
cargo run --example fibonacci -- --training --inference
```

- **Feature flags select the backend**: `backend-{flex,cpu,wgpu,metal,vulkan,cuda,
  rocm,tch-cpu,tch-gpu,remote,ndarray}` (flex preferred for checks/tests, enabled 
  by default). Each just enables the matching `burn/<backend>`; several may be
  compiled in at once and `Device::default()` resolves which to use (honouring `BURN_DEVICE`).
- `mamba1`/`mamba2`/`mamba3`/`autodiff` are default-on; `mamba2`/`mamba3` imply
  `autodiff`. `cubecl`/`fusion` enable the memory-saving custom backward on those
  backend families. `dev-f16`/`dev-simd`/`dev-autotune` are example/test conveniences.

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

## File Map

`refs/` is external reference material (see [Extra References](#extra-references)).
Every leaf module has a sibling `tests.rs` (forward/step parity, gradients,
cross-variant agreement) — not listed individually.

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
│  ├─ helpers.rs     shared: trapezoid coeffs, QK-norm+GQA+bias, MIMO-V build
│  ├─ cache.rs       Mamba3Cache(s) ENUMS dispatching DoubleSsd vs SingleSsd
│  ├─ ssd_path.rs    pathway-agnostic Mamba3SsdPath (From<> both sub-paths)
│  ├─ double_ssd/    two-pass trapezoid (γ-SSD + β-SSD); cache.rs + ssd/ kernels
│  ├─ single_ssd/    one-pass official-kernel form (≈½ memory); cache.rs (h') + ssd/
│  ├─ rotation/      quaternion non-abelian RoPE (Complex2D | Quaternion4D) + algebra
│  └─ quat_scan/     memory-efficient quaternion cumprod scan (recompute backward)
├─ modules/          family-generic composition + shared NN modules
│  ├─ mod.rs         MambaBlock / MambaBlockConfig traits; MambaSsdPath enum
│  ├─ layer.rs       Layer<M>: Pre-LN residual block
│  ├─ layers.rs      Layers<M>: virtual-layer stack over real weight sets
│  ├─ multi_gate.rs  Multi-Gate Residuals (Standard|MultiGate)
│  ├─ network.rs     LatentNetwork / VocabNetwork + MambaLatentNet / MambaVocabNet enums
│  ├─ bidi.rs        BidiLayers<M> + OutputMerge + MambaBidiLayers enum
│  ├─ cache.rs       CacheStack trait + MambaCaches enum
│  ├─ activation/    silu, softplus, log_sigmoid (fp16-aware)
│  ├─ norm/          rms_norm (also Mamba-3 QK-Norm), rms_norm_gated
│  ├─ loss/          bce, cross_entropy, mse
│  └─ misc/          gqa, segsum, split, sanity
└─ utils/            lower-level plumbing
   ├─ mod.rs         div_eps (per-dtype epsilon)
   ├─ class/         ClassToken / ClassLatent insertion (CLS-style registers)
   ├─ schedule/      Schedule + BidiSchedule (virtual→real index mapping)
   ├─ scheduler/     LR schedulers (cosine + warmup, constant) — example use
   ├─ backend_macros.rs  per-backend BackendExt impls + autodiff marker traits
   ├─ combined_grad.rs   flatten/unflatten (y, final_state) for custom backward
   ├─ fprim.rs           F<B,D>: rank-tagged FloatTensor-primitive wrapper
   └─ test_helpers.rs    max_abs_diff + grad-comparison macros
```

`files.md` is the per-file signature reference (what each important file defines +
the non-obvious decisions). The detailed per-family math lives in the `mamba2.rs` /
`mamba3.rs` module headers. Always consider starting-off searching from `files.md`.

---

## Architecture

### Layer → Network hierarchy (all families)

All three families share **one** set of generic composition types in `src/modules/`,
parameterised by the SSM core block `M` (`Mamba1`/`Mamba2`/`Mamba3`):

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → out_proj            (continuous I/O)
Layers<M>         a stack of N (virtual) layers over R real weight sets
Layer<M>          Pre-LN residual:  y = x·residual_scale + Block(RMSNorm(x))
M (Block)         the SSM core (mamba1.rs / mamba2.rs / mamba3.rs)
```

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
- **Mamba-3** — Mamba-2 plus three independent additions: **trapezoidal**
  discretisation (3-term `h = αh + βB₋₁x₋₁ + γBₜxₜ`, data-dependent `A`/`λ`; `λ≡1`
  collapses to Mamba-2), **data-dependent RoPE** on B/C, and **MIMO** (`mimo_rank>1`).
  B/C use **QK-Norm before** the SSD (not a post gated norm); no short conv. The
  in-projection splits `[z|x|B_raw|C_raw|dd_dt|dd_A|λ_raw|θ]`.

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

### Mamba-3: rotation (RoPE)

Default **`Complex2D`** (abelian `SO(2)`): angles projected, `tanh·π`-squashed,
Δ-scaled per head, then **`cumsum`** along the sequence (continued from the cache),
absorbed into B/C. `wrap_angle` reduces mod `2π` (value-exact, the offset `detach`ed)
to stay fp16-stable over long sequences. `rope_fraction` (0.5/1.0) rotates a prefix;
SISO uses interleaved/NeoX pairing, MIMO half-and-half/GPT-J.

`mamba3/rotation/` adds the **non-abelian** `Quaternion4D` (`SU(2) ⊂ SO(4)`): the
cumulative rotation becomes an associative **scan** (with cross-chunk carry) instead
of a `cumsum`, while the B/C-factoring (so the scalar-decay SSD core) is unchanged.
Selected by `Mamba3Config.rotation: RotationKind`; the cache accumulator is a
`RotationState`. It runs on **both** SSD pathways (applied to B/C before chunking).
`quat_scan/` provides the memory-efficient recompute-backward version of the scan.

### Virtual layers, bidirectional, class tokens

- **Virtual layers** (`utils/schedule/`): `Layers<M>` runs `n_virtual_layers` logical
  passes over `n_real_layers` weight sets, each virtual layer keeping its own cache.
  `Schedule` maps virtual→real (`Cyclic`/`Stretched`/`Custom`); `BidiSchedule` pairs
  forward/backward layers.
- **Bidirectional** (`modules/bidi.rs`): `BidiLayerPair<M>` runs a straight (→) and a
  reversed (← via `flip`) pass merged by `OutputMerge` (`Mean`|`CatLinear`);
  `BidiLayers<M>` stacks pairs. Generic over `M` → serves all families.
- **Class tokens/latents** (`utils/class/`): learnable `[CLS]`-style embeddings spliced
  into the sequence. `ClassToken` on a *network*, `ClassLatent` on a *layer container*;
  markers (`Start|Middle|End|Custom`) say where each lands. `forward` returns the
  lengthened sequence; the caller reads tokens via `class_*_output_indices`. `step`
  injects via position **cursors** (`Start`/`Custom` only; `Middle`/`End` need the full
  length and panic there).
- **Multi-Gate Residuals** (`modules/multi_gate.rs`): `Layers<M>.residuals` picks plain
  additive (`Standard`) vs `MultiGate` — `n_stream` streams gated/attention-pooled between
  layers instead of one additive skip. See the module header.

---

## Key Design Decisions

- **No optimized kernels** — only Burn's portable tensor ops, so one code path runs on
  every backend.
- **Dispatch backend (Burn 0.22+)** — the high-level `Tensor` (every `Module`) is pinned
  to the global `Dispatch` backend, so library types are **not backend-generic**
  (`Mamba2`, `Mamba2Cache`, … carry no `<B>`). The backend is a runtime `Device`;
  autodiff and dtype are device properties. Only the custom-backward internals stay
  generic over `B` (`F<B,D>`, the `Backward<B,_>` nodes, `Autodiff<B>` ext impls).
- **Two Mamba-3 SSD pathways** — cache type selects double-ssd (simple) vs single-ssd
  (~½ memory); accumulators coincide at boundaries so caches inter-convert.
- **Three SSD algorithm variants**, the last with a custom recompute backward; proven
  equal on values + gradients by tests.
- **`#![warn(missing_docs)]`** — keep the crate warning-clean; document public surface
  as you add it.
- The project root is `/shared/claude/burn-mamba/`; do not read/write outside it.
- When a source file is added/removed/changed, prepare an update to its entry for the
  [File Map](#file-map) and `files.md` (per the maintenance rules above).
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
*relation* of them (offset/multiple/concat): `X` may be `x±1`/`x*2`, `S` = padded
sequence, `K = conv_kernel − 1`.

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

Under `refs/` (not analyzed here): **Mamba-3 paper** TeX (`refs/mamba-3-paper/`);
**official Python impl** (authoritative; Triton SISO / Tilelang MIMO kernels are the
single-ssd reference) (`refs/state-spaces/mamba/`); **Mamba-3 minimal** (basis of
double-ssd) (`refs/VikramLex/mamba3-minimal/`); **Burn** (`refs/burn/`).

## Custom Commands

- `rg`: available.
- `git`: forbidden.
