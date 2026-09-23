# burn-mamba &emsp; [![deepwiki]][deepwikiurl] [![docs]][docsurl] &emsp; <img src="https://raw.githubusercontent.com/swfsql/burn-mamba/main/assets/logo-small.png?raw=true" alt="Logo" width="20px"/>

[deepwiki]: https://deepwiki.com/badge.svg
[deepwikiurl]: https://deepwiki.com/swfsql/burn-mamba
[docs]: https://img.shields.io/badge/-docs-brightgreen
[docsurl]: https://swfsql.github.io/burn-mamba/doc/burn_mamba/index.html

> A minimal, readable reference implementation of **Mamba-1, Mamba-2, and Mamba-3**
> for the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

`burn-mamba` ports the selective state space model (SSM) architectures of
[Mamba-1](https://arxiv.org/abs/2312.00752),
[Mamba-2](https://arxiv.org/abs/2405.21060), and
[Mamba-3](https://arxiv.org/abs/2603.15569) to **standard, portable Burn tensor
operations**. It has no custom CUDA/Triton kernels, so the same code runs on
every Burn backend (CPU, WGPU, CUDA, Metal, LibTorch, …). The goal is clarity:
a faithful, documented translation of the official
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) kernels that is
easy to read, verify, and learn from.

---

## What is Mamba?

Mamba is a family of **selective state space models** for sequence modeling.
Like an RNN, it carries a fixed-size hidden state. But its *selective*
parameters depend on the input, so at each step it can choose what to remember
or forget. This gives it two modes:

- a **parallel** form for training and prompt prefill: linear in sequence
  length, and expressed as batched matrix multiplications,
- a **recurrent** form for decoding: one token at a time, in constant memory
  (no growing attention KV-cache).

Each generation in this crate builds on the one before:

- **Mamba-1**: the original selective SSM (a sequential selective scan).
- **Mamba-2**: recasts the recurrence as **Structured State Space Duality
  (SSD)**, a chunkwise algorithm made of GEMMs.
- **Mamba-3**: extends SSD with trapezoidal discretisation, a
  **complex-valued** state transition (a data-dependent rotation beside the
  scalar decay), and multi-input/multi-output (MIMO) state. This crate adds
  **MambaProduct** (`micro_steps` recurrence steps per token) and optional
  **positive systems** beside the recurrence (a Kalman gate, a max-plus
  register).

Despite the name, the *data-dependent RoPE* of Mamba-3 is not a positional
encoding. It is the imaginary part of the state transition. A complex SSM is
exactly a real SSM whose transition is a scalar decay times a block-diagonal of
`2×2` rotations. Those rotations are orthogonal (and, in `SO(2)`, they
commute), so the cumulative rotation factors out of the recurrence, and `B`/`C`
absorb it. This is the "RoPE trick": the plain scalar-decay SSD kernel stays
unchanged. So the transition uses the machinery of RoPE, on the query/key side
(by SSD duality). But the angles are data-dependent, not a fixed frequency
schedule, and the purpose is rotational state dynamics. That lets Mamba-3 track
state (parity, mod-k), which a real SSM with non-negative eigenvalues provably
cannot do.

That rotation need not be abelian. `RotationKind` selects its group:

- `Real1D`: the trivial group (no rotation: a real transition, and the
  ablation for the other kinds),
- `Complex2D`: the `SO(2)` of the paper (a prefix sum of angles),
- `Quaternion4D`: `SU(2)` (an associative scan of unit quaternions, so the
  state composes a *word* in a group, not only a count),
- `Rotor4D`: the whole `SO(4)` (`v ↦ q ⊗ v ⊗ p̄`), which contains both.

The factoring onto `B`/`C`, and so the plain scalar-decay SSD kernel, is the
same for all of them. The `reset-*` examples are a ladder that isolates what
each kind buys.

## Highlights

- **All three families**: Mamba-1, Mamba-2, and Mamba-3, each as a block that
  plugs into Pre-LN residual layers, layer stacks, and full networks (from
  [`burn-stack`](https://github.com/swfsql/burn-stack)).
- **Backend-agnostic**: pure Burn tensor ops, no custom kernels, so it runs
  unchanged on every backend.
- **Two execution modes**: a parallel `forward()` and a recurrent `step()`
  that are mathematically equivalent. The test suite asserts this on outputs,
  final state, *and* gradients.
- **Pluggable SSD algorithms** (Mamba-2/3), including a custom recompute
  backward that costs a little compute and saves roughly a third of the
  training memory.
- **Right padding**: `forward()` takes a padding mask. Each slot then gives the
  outputs and cache that it would give alone.
- **Virtual layers**: many logical layers over a smaller set of shared weights,
  with a configurable schedule.
- **Bidirectional stacks** for non-autoregressive tasks.
- **Class tokens / latents**: learnable `[CLS]`-style registers in the
  sequence. They land identically for any split of the sequence into
  `forward()` chunks and `step()`s. `prime()` steps them without a user token.
- **CUDA graph replay** (optional): a decode `step()`, a fixed-shape
  `forward()`, and a plain-SGD training step can replay from one captured
  graph.

## Installation

```toml
[dependencies]
# Note: see Cargo.toml for the burn rev in use
burn = { git = "https://github.com/tracel-ai/burn.git", rev = "abc..." }
burn-mamba = { git = "https://github.com/swfsql/burn-mamba.git", rev = "abc..." }
```

Enable at least one `backend-*` feature to select a runtime backend (the same
backend selection as Burn). You can enable several at once. The program
selects the backend with the `Device` that it constructs.

<details>
<summary><b>Feature flags</b></summary>

| Feature | Purpose |
|---------|---------|
| `mamba1` / `mamba2` / `mamba3` | Enable each family (all on by default). `mamba2`/`mamba3` imply `autodiff`. |
| `autodiff` | Required for Mamba-2/3. Enables the memory-saving custom backward. |
| `optim` | Muon parameter groups over the fused projections (on by default). |
| `cubecl` | Enables the custom backward on CubeCL backends. |
| `fusion` | Enables the custom backward under `burn/fusion`. |
| `check-nan` / `check-inf` | Assert that intermediate tensors have no NaN / Inf (debugging). |
| `backend-*` | Select the backend (e.g. `backend-flex`, `backend-cuda`, `backend-wgpu`, `backend-tch-cpu`, …). `backend-flex` is on by default. |
| `dev-f16` / `dev-simd` / `dev-autotune` | Example/test conveniences (fp16, SIMD, autotune). |

See `Cargo.toml` for the full list. `backend-flex` is the recommended choice
for quick checks and tests.

</details>

## Quick start

Every block has the two execution modes. Training/prefill runs `forward()`
over a whole sequence. Decoding runs `step()` one token at a time, and passes
the returned cache to the next call:

```rust
use burn::prelude::*;
use burn_mamba::prelude::*;

// The `Device` selects the backend at runtime: tensors and modules are not
// backend-generic. Construct a device for an enabled backend, for example
// `Device::flex()` / `Device::cuda(0)` (or `device.autodiff()` for training).
fn main() {
    // Create a default Device
    let device = Device::default();

    // A single Mamba-2 SSM block with d_model = 256.
    let block = Mamba2Config::new(256).init(&device);

    // forward: parallel over the full sequence — [batch, sequence, d_model].
    // The last argument is an optional right-padding mask.
    let x = Tensor::<3>::zeros([2, 64, 256], &device);
    let (y, cache) = block.forward(x, None, Mamba2SsdPath::default(), None);
    assert_eq!([2, 64, 256], y.dims());

    // step: one token at a time, constant memory — [batch, d_model].
    let x_t = Tensor::<2>::zeros([2, 256], &device);
    let (y_t, _next_cache) = block.step(x_t, Some(cache));
    assert_eq!([2, 256], y_t.dims());
}
```

`Mamba1` and `Mamba3` have the same API (`Mamba1::forward` has no SSD-path
argument). The generic containers of `burn-stack` (`Layer<M>`, `Layers<M>`,
`LatentNetwork<M>`, `VocabNetwork<M>`, `BidiLayers<M>`) compose any of them.
The runtime-selectable `MambaLatentNet`, `MambaVocabNet` and `MambaBidiLayers`
select the family at run time. See the [examples](#examples) for complete
training and inference programs.

## Two execution modes

| Method | Mode | Best for | Cost per token |
|--------|------|----------|----------------|
| `forward()` | parallel / chunkwise | training, prompt prefill | amortised by batched GEMMs |
| `step()` | recurrent | autoregressive decoding | O(state), independent of sequence length |

A `forward()` over a sequence is exactly equal to `step()` unrolled token by
token from the same initial cache. The test suite checks this parity on
outputs, final cache, and gradients.

Models with class tokens/latents have a third recurrent entry point,
`prime()`. It steps the class markers that wait for the next user token,
*without* that token, and returns the last of them. A seedless generation
loop starts this way, when it has nothing to give `step()` yet. A `prime()`
followed by a `step()` runs exactly what that `step()` alone would run.

API references:
[`Mamba1`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba1/mamba1/struct.Mamba1.html) ·
[`Mamba2`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/mamba2/struct.Mamba2.html) ·
[`Mamba3`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/mamba3/struct.Mamba3.html).

## The three families at a glance

| | Mamba-1 | Mamba-2 | Mamba-3 |
|---|---|---|---|
| Core algorithm | sequential selective scan | chunkwise SSD | trapezoidal SSD |
| State transition | diagonal | scalar (SSD) | data-dependent scalar decay × rotation |
| Rotational state (the "RoPE trick") | — | — | data-dependent rotation factored onto B/C (`SO(2)`, `SU(2)` or `SO(4)`) |
| MIMO state | — | — | optional (`mimo_rank > 1`) |
| Micro-steps per token (MambaProduct) | — | — | optional (`micro_steps > 1`) |
| Short convolution | yes | yes | removed |
| Pluggable SSD algorithms | — | yes | yes |
| Bidirectional stacks | yes | yes | yes |
| Virtual-layer scheduling | yes | yes | yes |

Mamba-2 and Mamba-3 are the modern baselines. Mamba-1 stays as the original,
simplest reference.

## Choosing an SSD algorithm (Mamba-2 / Mamba-3)

An `…SsdPath` selector makes the chunkwise scan pluggable. All three variants
are exact reformulations of the same math, and they agree on values **and**
gradients. They differ only in their memory/compute trade-off:

| Variant | Approach | Backward |
|---------|----------|----------|
| `Minimal` | mostly batched matmuls + a segment-sum mask | autodiff |
| `Serial` | a serial loop over chunks (mirrors the reference Triton kernels) | autodiff |
| `SerialRecalculated` *(default)* | a serial loop with a recompute backward | custom: ~⅓ less training memory |

See
[`Mamba2SsdPath`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/ssd/ssd_path/enum.Mamba2SsdPath.html)
and
[`Mamba3SsdPath`](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/ssd_path/enum.Mamba3SsdPath.html).
In Mamba-3 the algorithm is independent of the *pathway* (double- or
single-SSD). The cache variant selects the pathway.

## Examples

The [`examples/`](examples/) directory has small, self-contained models on
synthetic or canonical data (see [`examples/README.md`](examples/README.md)):

- **[`reset-*`](examples/reset/)**: a ladder on one stream shape. Each rung is
  the smallest task that its block is *needed* for: `reset-majority` (a single
  64-parameter Mamba-3 block on the sign of a running vote since the last
  reset, and the full train → save → infer flow), then `reset-rotor`,
  `reset-spinor` and `reset-swap` for the complex, quaternion and two-sided
  `SO(4)` rotations. `spinor-product` turns the `micro_steps` dial, and
  `reset-quintic` asks for the non-solvable `A₅` and `S₅`. Every rung has a
  hand-built exact solution and the ablations that separate it from the rung
  below.
- **[`tally-*`](examples/tally/)**: a ladder for the positive systems beside
  the recurrence (the max-plus register and the Kalman gate).
- **`mnist-class`**: a Mamba-3 classifier that reads each MNIST image as a
  sequence of pixels.
- **`mnist-ae`**: a bidirectional Mamba-3 autoencoder over the same pixel
  sequence.
- **`tiny-stories`**: a tiny character-level Mamba-3 language model.

```bash
# train the smallest example (flex backend, fp32), then run inference
cargo run --example reset-majority --features "backend-flex" -- --training --inference
```

For browser/wasm inference of the smallest pretrained Mamba-1/2/3 models from
`huggingface.co/state-spaces`, see
[`swfsql/burn-mamba-example`](https://github.com/swfsql/burn-mamba-example).

## Benchmarks

`./bench.sh` times a single block of each family (`forward` for prefill,
`train` for forward + backward, `step` for decode) on flex, CUDA, and CUDA with
fusion and autotuning. It writes the comparison to [`bench.md`](bench.md). The
cases and the knobs are in [`benches/layer.rs`](benches/layer.rs). `cargo
bench` runs them for a single configuration. `./kernels.sh` counts the kernel
launches per case and writes [`kernels.md`](kernels.md).

## Documentation

- **[API docs][docsurl]**: the rendered `rustdoc`. Every public item has
  documentation, and the per-block module headers carry the full math and
  notation.
- **[DeepWiki][deepwikiurl]**: an explorable overview of the codebase.
- **[`info/`](info/)**: standalone notes. A sibling script in
  [`scripts/`](scripts/) checks each note in float64 (`numpy` only, no import
  of the crate, non-zero exit on failure). Four notes are about what Mamba-3
  changed and why. Three classify the recurrence, and one classifies the block
  around it:
  - **[Rotation as Optimization](info/mamba-3/rotation-as-optimization.md)**:
    the *quadratic* term. What a step optimizes, in what sense the complex
    transition is a learning rate (or a saddle, or momentum), and what
    `micro_steps` composes, compared with DeltaProduct.
  - **[Trapezoid as Integration](info/mamba-3/trapezoid-as-integration.md)**:
    the *linear* term along time. `λ` as an operator-splitting parameter, the
    two-installment collapse under the single-SSD pathway, and the tap lattice
    that `micro_steps > 1` opens.
  - **[MIMO as Batch Size](info/mamba-3/mimo-as-batch.md)**: the linear term
    along *rank*. The minibatch reading, the cost of the value tying, and why
    the ranks must share the rotation.
  - **[Architecture Deltas](info/mamba-3/architecture-deltas.md)**: everything
    outside the recurrence. BCNorm and the `B`/`C` biases (a fresh block is a
    convolution), the deleted short conv, the output-norm placements, and the
    chunk-length schedule.

  And one note on what this crate adds beside the plant:
  - **[Gate as Positive System](info/kalman/gate-as-positive-system.md)**: the
    Kalman gate and the max-plus register. One log-semiring scan, the ceiling
    and contraction of the computed decay, and what a classifier gets for free.
- Contributors: `CLAUDE.md` and `files.md` map the structure, architecture, and
  conventions of the repository.

## Scope

This is a **readable reference implementation**, not a performance-tuned one.
It uses only portable Burn ops (no hand-written kernels), so it puts clarity and
backend portability before raw throughput. Extensive forward/step parity and
gradient-agreement tests guard its correctness.

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
