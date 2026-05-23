# <img src="https://raw.githubusercontent.com/swfsql/burn-mamba/main/assets/logo-small.png"/> burn-mamba &emsp; [![deepwiki]][deepwikiurl] [![docs]][docsurl]

[deepwiki]: https://deepwiki.com/badge.svg
[deepwikiurl]: https://deepwiki.com/swfsql/burn-mamba
[docs]: https://img.shields.io/badge/-docs-brightgreen
[docsurl]: https://swfsql.github.io/burn-mamba/doc/burn_mamba/index.html

Ports [Mamba-1](https://arxiv.org/abs/2312.00752)/[2](https://arxiv.org/abs/2405.21060)/[3](https://arxiv.org/abs/2603.15569) to standard tensor operations for the burn framework. The official reference implementation is at [state-spaces/mamba](https://github.com/state-spaces/mamba).

Mamba is a fast, efficient model for handling long data sequences such as in language and time-series tasks, competitive with traditional Transformers. It uses a smart selection process to focus on key information, scaling linearly with strong performance.

##### Cargo.toml

```toml
[dependencies]
burn = "0.21.0"
burn-mamba = { git = 'https://github.com/swfsql/burn-mamba.git', rev = "abc..." } # add frozen rev
```

##### Features

- `mamba1`/`2`/`3`: Enable the Mamba-1/2/3 types. All enabled by default.
- `autodiff`: Required if using Mamba-2/3, enables an optional backwards algorithm that saves training memory by ~1/3.
- `cubecl`: Required if using Mamba-2/3 in a cubecl backend, enables an optional backwards algorithm that saves memory.
- `fusion`: Required if using Mamba-2/3 with fusion, enables an optional backwards algorithm that saves memory.
- `backend-*`: Required if using Mamba-2/3 with a specific backend. The backend selection is also required by the examples.
- `dev-f16`: Enables `f16` support in examples.
- `dev-simd`: Enables `burn/simd` support in examples.
- `dev-autotune`: Enables `burn/autotune` support in examples.

Please check `Cargo.toml` for more info.

##### Usage

The models can be used with two methods:

- `forward`: preferred for training, this is a parallel mode that generates a causal-autoregressive output for each timestep. See [Mamba1::forward](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba1/mamba1/struct.Mamba1.html#method.forward)/[2](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/mamba2/struct.Mamba2.html#method.forward)/[3](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/mamba3/struct.Mamba3.html#method.forward) for more info.
  - For Mamba-2/3, it is required an ssd algorithm selection. See [Mamba2SsdPath](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/ssd/ssd_path/enum.Mamba2SsdPath.html)/[3](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/ssd/ssd_path/enum.Mamba3SsdPath.html) for more info.
- `step`: preferred for inference, this is a mode that generates a single causal output in constant time and memory. See [Mamba1::step](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba1/mamba1/struct.Mamba1.html#method.step)/[2](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba2/mamba2/struct.Mamba2.html#method.step)/[3](https://swfsql.github.io/burn-mamba/doc/burn_mamba/mamba3/mamba3/struct.Mamba3.html#method.step) for more info.

###### Examples

- `examples/` directory contains some small-model examples on synthetic or canonical data (e.g. mnist).
- [`swfsql/burn-mamba-example`](https://github.com/swfsql/burn-mamba-example) shows inference for the smallest Mamba-1/2 models from `huggingface.co/state-spaces`, which can also be tested in the browser via wasm.

##### Learn More
###### S4
- [Stanford MLSys Seminars - Efficiently Modeling Long Sequences with Structured State Spaces - Albert Gu | Stanford MLSys #46](https://www.youtube.com/watch?v=EvQ3ncuriCM).
- [Stanford MedAI - MedAI #41: Efficiently Modeling Long Sequences with Structured State Spaces | Albert Gu](https://www.youtube.com/watch?v=luCBXCErkCs).
- [Yingzhen Li - Structured State Space Models for Deep Sequence Modeling (Albert Gu, CMU)](https://www.youtube.com/watch?v=OpJMn8T7Z34).

###### Mamba
- [Samuel Albanie - Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY).
- [Umar Jamil - Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU).
- [Algorithmic Simplicity - Mamba from scratch: Neural nets better and faster than transformers](https://www.youtube.com/watch?v=N6Piou4oYx8).
- [State Space Duality (Mamba-2)](https://tridao.me/blog/2024/mamba2-part1-model/).

###### Implementation References

- [state-spaces/mamba](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py).
- [huggingface/candle-examples/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/).
- [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/).
- [kroggen/mamba.c](https://github.com/kroggen/mamba.c/blob/learning/mamba.c).
- [kroggen/mamba-cpu](https://github.com/kroggen/mamba-cpu/blob/recurrent-only/mamba_ssm/mamba_simple.py).
- [tommyip/mamba2-minimal](https://github.com/tommyip/mamba2-minimal).
- [VikramLex/mamba3-minimal](https://github.com/VikramLex/mamba3-minimal).