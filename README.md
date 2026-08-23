# burn-mamba &emsp; [![deepwiki]][deepwikiurl] [![docs]][docsurl] &emsp; <img src="https://raw.githubusercontent.com/swfsql/burn-mamba/main/assets/logo-small.png?raw=true" alt="Logo" width="20px"/>

[deepwiki]: https://deepwiki.com/badge.svg
[deepwikiurl]: https://deepwiki.com/swfsql/burn-mamba
[docs]: https://img.shields.io/badge/-docs-brightgreen
[docsurl]: https://swfsql.github.io/burn-mamba/doc/burn_mamba/index.html

> A minimal, readable reference implementation of **Mamba-1, Mamba-2, and Mamba-3**
> for the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

`burn-mamba` ports the selective state space model (SSM) architectures from
[Mamba-1](https://arxiv.org/abs/2312.00752),
[Mamba-2](https://arxiv.org/abs/2405.21060), and
[Mamba-3](https://arxiv.org/abs/2603.15569) down to **standard, portable Burn
tensor operations**. There are no custom CUDA/Triton kernels — the exact same
code runs on every Burn backend (CPU, WGPU, CUDA, Metal, LibTorch, …). The goal
is clarity: a faithful, well-documented translation of the official
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) kernels that is
easy to read, verify, and learn from.

---

## What is Mamba?

Mamba is a family of **selective state space models** for sequence modeling.
Like an RNN it carries a fixed-size hidden state, but its *selective* parameters
are input-dependent, letting it choose what to remember or forget at each step.
This gives it two complementary modes:

- a **parallel** form for training and prompt prefill, linear in sequence length
  but expressed as batched matrix multiplies; and
- a **recurrent** form for decoding, which emits one token at a time in constant
  memory — no growing attention KV-cache.

Each generation in this crate builds on the last:

- **Mamba-1** — the original selective SSM (a sequential selective scan).
- **Mamba-2** — recasts the recurrence as **Structured State Space Duality
  (SSD)**, a chunkwise algorithm built from GEMMs.
- **Mamba-3** — extends SSD with trapezoidal discretisation, a **complex-valued**
  state transition (a data-dependent rotation alongside the scalar decay), and
  multi-input/multi-output (MIMO) state.

Despite the name, Mamba-3's *data-dependent RoPE* is not a positional encoding —
it is the imaginary part of the state transition. A complex SSM is exactly a real
SSM whose transition is a scalar decay times a block-diagonal of `2×2` rotations,
and because those rotations are orthogonal (and, being `SO(2)`, commute) the
cumulative rotation factors out of the recurrence and is absorbed into `B`/`C` —
the "RoPE trick", which leaves the plain scalar-decay SSD kernel untouched. So it
borrows RoPE's machinery and lands on the query/key side (by SSD duality), but the
angles are data-dependent rather than a fixed frequency schedule, and the purpose
is rotational state dynamics: that is what lets Mamba-3 track state (parity,
mod-k) which a real, non-negative-eigenvalue SSM provably cannot.

That rotation need not be abelian, and `RotationKind` picks the group it turns
in: `Real1D` is the trivial group (no rotation at all — a real transition, and
the ablation the rest is measured against), `Complex2D` is the paper's `SO(2)`
(a `cumsum` of angles), `Quaternion4D` is `SU(2)` (an associative scan of unit
quaternions, so the state composes a *word* in a group rather than counting),
and `Rotor4D` is the whole `SO(4)` (`v ↦ q ⊗ v ⊗ p̄`), which contains both. The
factoring onto `B`/`C` — and so the plain scalar-decay SSD kernel — is identical
in all of them. The `reset-*` examples are a ladder isolating exactly what each
one buys.

## Highlights

- **All three families** — Mamba-1, Mamba-2, and Mamba-3, each as a block, a
  Pre-LN residual layer, a layer stack, and a full language-model network.
- **Backend-agnostic** — pure Burn tensor ops; no custom kernels, so it runs
  unchanged on every backend.
- **Dual execution modes** — a parallel `forward()` and a recurrent `step()`
  that are mathematically equivalent (asserted by the test suite on outputs,
  final state, *and* gradients).
- **Pluggable SSD algorithms** (Mamba-2/3) — including a custom
  recompute backward that trades a little compute for roughly a third less
  training memory.
- **Virtual layers** — run many logical layers over a smaller set of shared
  weights via a configurable schedule.
- **Bidirectional wrappers** (Mamba-2/3) for non-autoregressive tasks.
- **Class tokens / latents** — learnable `[CLS]`-style registers spliced into the
  sequence, landing identically however it is split into `forward()` chunks and
  `step()`s, plus a `prime()` that steps them without a user token.

## Installation

```toml
[dependencies]
# burn = "0.22.0-pre.1"
# burn 0.22.0-pre.1 is not yet released, so pin to the same version that burn-mamba uses:
burn = { git = "https://github.com/tracel-ai/burn.git", rev = "ed4d313b16ac348093cfa0f979774b4312b17058" }

# pin to a specific revision:
burn-mamba = { git = "https://github.com/swfsql/burn-mamba.git", rev = "abc..." }
```

Enable at least one `backend-*` feature to pick a runtime backend (the same
backend selection Burn uses). Several may be enabled at once; the running program
chooses the backend by constructing the matching `Device`.

<details>
<summary><b>Feature flags</b></summary>

| Feature | Purpose |
|---------|---------|
| `mamba1` / `mamba2` / `mamba3` | Enable each family (all on by default). `mamba2`/`mamba3` imply `autodiff`. |
| `autodiff` | Required for Mamba-2/3; enables the memory-saving custom backward. |
| `cubecl` | Enables the custom backward on CubeCL backends. |
| `fusion` | Enables the custom backward under `burn/fusion`. |
| `backend-*` | Select the backend (e.g. `backend-flex`, `backend-cuda`, `backend-wgpu`, `backend-tch-cpu`, …). |
| `dev-f16` / `dev-simd` / `dev-autotune` | Example/test conveniences (fp16, SIMD, autotune). |

See `Cargo.toml` for the full list. `backend-flex` is the recommended choice for
quick checks and tests.

</details>

## Quick start

Every block exposes the two execution modes. Training/prefill runs `forward()`
over a whole sequence; decoding advances `step()` one token at a time, threading
the returned cache:

```rust
use burn::prelude::*;
use burn_mamba::prelude::*;

// The backend is chosen at runtime by the `Device`; tensors and modules are not
// backend-generic. Construct a device for the enabled backend, e.g.
// `Device::flex()` / `Device::cuda(0)` (or `device.autodiff()` for training).
fn main() {
    // Create a default Device
    let device = Device::default();
    
    // A single Mamba-2 SSM block with d_model = 256.
    let block = Mamba2Config::new(256).init(&device);

    // forward: parallel over the full sequence — [batch, sequence, d_model].
    let x = Tensor::<3>::zeros([2, 64, 256], &device);
    let (y, cache) = block.forward(x, None, Mamba2SsdPath::default());
    assert_eq!([2, 64, 256], y.dims());

    // step: one token at a time, constant memory — [batch, d_model].
    let x_t = Tensor::<2>::zeros([2, 256], &device);
    let (y_t, _next_cache) = block.step(x_t, Some(cache));
    assert_eq!([2, 256], y_t.dims());
}
```

`Mamba1Config` / `Mamba3Config` (and the `…Layer`, `…Layers`, `…Network`
variants) follow the same shape. See the [examples](#examples) for complete,
runnable training and inference programs.

## Two execution modes

| Method | Mode | Best for | Cost per token |
|--------|------|----------|----------------|
| `forward()` | parallel / chunkwise | training, prompt prefill | amortised via batched GEMMs |
| `step()` | recurrent | autoregressive decoding | O(state), independent of sequence length |

A `forward()` over a sequence is exactly equal to unrolling `step()` token by
token from the same initial cache — the parity property the test suite verifies
on outputs, final cache, and gradients.

Models carrying class tokens/latents gain a third recurrent entry point,
`prime()`: it steps the class markers waiting for the next user token *without*
one, returning the last of them — how a seedless generation loop starts, when
there is nothing to hand `step()` yet. A `prime()` plus the `step()` after it
runs exactly what that `step()` alone would have.

API references:
[`Mamba1`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba1/mamba1/struct.Mamba1.html) ·
[`Mamba2`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/mamba2/struct.Mamba2.html) ·
[`Mamba3`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/mamba3/struct.Mamba3.html).

## The three families at a glance

| | Mamba-1 | Mamba-2 | Mamba-3 |
|---|---|---|---|
| Core algorithm | sequential selective scan | chunkwise SSD | trapezoidal SSD |
| State transition | diagonal | scalar (SSD) | complex: data-dependent scalar decay × rotation |
| Rotational state (the "RoPE trick") | — | — | data-dependent rotation factored onto B/C (`SO(2)`, `SU(2)` or `SO(4)`) |
| MIMO state | — | — | optional (`mimo_rank > 1`) |
| Short convolution | yes | yes | removed |
| Pluggable SSD algorithms | — | yes | yes |
| Bidirectional wrappers | — | yes | yes |
| Virtual-layer scheduling | yes | yes | yes |

Mamba-2 and Mamba-3 are the modern baselines; Mamba-1 is kept as the original,
simplest reference.

## Choosing an SSD algorithm (Mamba-2 / Mamba-3)

The chunkwise scan is pluggable via an `…SsdPath` selector. All three variants
are exact reformulations of the same math and agree on values **and** gradients;
they differ only in their memory/compute trade-off:

| Variant | Approach | Backward |
|---------|----------|----------|
| `Minimal` | mostly batched matmuls + a segment-sum mask | autodiff |
| `Serial` | a serial loop over chunks (mirrors the reference Triton kernels) | autodiff |
| `SerialRecalculated` *(default)* | serial loop with a recompute backward | custom — ~⅓ less training memory |

See
[`Mamba2SsdPath`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/ssd/ssd_path/enum.Mamba2SsdPath.html)
and
[`Mamba3SsdPath`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/ssd_path/enum.Mamba3SsdPath.html).
In Mamba-3, the algorithm is chosen independently of the *pathway* (double- vs
single-SSD), which is selected by the cache variant supplied.

## Examples

The [`examples/`](examples/) directory contains small, self-contained models on
synthetic or canonical data:

- **`reset-majority`** — the smallest demo: a single Mamba-2 block (62
  parameters) on the sign of a running vote since the last reset, exercising the
  full train → save → infer flow.
- **`mnist-class`** — a Mamba-3 classifier that reads each MNIST image as a
  sequence of pixels.

```bash
# train the smallest example (flex backend, fp32), then run inference
cargo run --example reset-majority --features "backend-flex" -- --training --inference
```

For browser/wasm inference of the smallest pretrained Mamba-1/2/3 models from
`huggingface.co/state-spaces`, see
[`swfsql/burn-mamba-example`](https://github.com/swfsql/burn-mamba-example).

## Benchmarks

`./bench.sh` times a single block of each family — `forward` (prefill), `train`
(forward + backward) and `step` (decode) — on flex, CUDA, and CUDA with fusion and
autotuning, writing the comparison to [`bench.md`](bench.md). The cases and the
knobs live in [`benches/layer.rs`](benches/layer.rs); `cargo bench` runs them for a
single configuration.

## Documentation

- **[API docs][docsurl]** — the rendered `rustdoc`; every public item is
  documented, and the per-block module headers carry the full math and notation.
- **[DeepWiki][deepwikiurl]** — an explorable overview of the codebase.
- Contributors: `CLAUDE.md` and `files.md` map the repository's structure,
  architecture, and conventions.

## Scope

This is a **readable reference implementation**, not a performance-tuned one. It
deliberately relies only on portable Burn ops (no hand-written kernels), so it
favours clarity and backend portability over raw throughput. Correctness is
guarded by extensive forward/step parity and gradient-agreement tests.

<details>
<summary><b>References &amp; learning resources</b></summary>

#### Structured State Spaces (S4)

- [Stanford MLSys #46 — Efficiently Modeling Long Sequences with Structured State Spaces (Albert Gu)](https://www.youtube.com/watch?v=EvQ3ncuriCM)
- [Stanford MedAI #41 — Efficiently Modeling Long Sequences with Structured State Spaces (Albert Gu)](https://www.youtube.com/watch?v=luCBXCErkCs)
- [Yingzhen Li — Structured State Space Models for Deep Sequence Modeling (Albert Gu, CMU)](https://www.youtube.com/watch?v=OpJMn8T7Z34)

#### Mamba

- [Samuel Albanie — Mamba: a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY)
- [Umar Jamil — Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU)
- [Algorithmic Simplicity — Mamba from scratch](https://www.youtube.com/watch?v=N6Piou4oYx8)
- [Tri Dao — State Space Duality (Mamba-2)](https://tridao.me/blog/2024/mamba2-part1-model/)

#### Implementation references

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — the official, authoritative implementation.
- [huggingface/candle — mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/)
- [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/)
- [kroggen/mamba.c](https://github.com/kroggen/mamba.c/blob/learning/mamba.c)
- [kroggen/mamba-cpu](https://github.com/kroggen/mamba-cpu/blob/recurrent-only/mamba_ssm/mamba_simple.py)
- [tommyip/mamba2-minimal](https://github.com/tommyip/mamba2-minimal)
- [VikramLex/mamba3-minimal](https://github.com/VikramLex/mamba3-minimal)

</details>
