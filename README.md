# burn-mamba

Ports a minimal (non-optimized) implementation of [Mamba](https://arxiv.org/abs/2312.00752) and [Mamba2](https://arxiv.org/abs/2405.21060) (WIP).

Mamba is a fast, efficient model for handling long data sequences such as in language and time-series tasks, competitive with traditional Transformers. It uses a smart selection process to focus on key information, scaling linearly with strong performance.

##### Cargo.toml

```toml
[dependencies.burn-mamba]
git = 'https://github.com/swfsql/burn-mamba.git'
branch = "main"
## instead of using a branch, you can pin to a specific commit:
# rev = ""
```

##### Features
- `mamba1`, `mamba2` to select the mamba version.

##### Example

You can check an example using this mamba block for inference in [here](https://github.com/swfsql/burn-mamba-example).

##### Implementation References

- [state-spaces/mamba](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py).
- [huggingface/candle-examples/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/).
- [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/).
- [kroggen/mamba.c](https://github.com/kroggen/mamba.c/blob/learning/mamba.c).
- [kroggen/mamba-cpu](https://github.com/kroggen/mamba-cpu/blob/recurrent-only/mamba_ssm/mamba_simple.py).

##### Learn More
###### S4
- [Stanford MLSys Seminars - Efficiently Modeling Long Sequences with Structured State Spaces - Albert Gu | Stanford MLSys #46](https://www.youtube.com/watch?v=EvQ3ncuriCM).
- [Stanford MedAI - MedAI #41: Efficiently Modeling Long Sequences with Structured State Spaces | Albert Gu](https://www.youtube.com/watch?v=luCBXCErkCs).
- [Yingzhen Li - Structured State Space Models for Deep Sequence Modeling (Albert Gu, CMU)](https://www.youtube.com/watch?v=OpJMn8T7Z34).

###### Mamba
- [Samuel Albanie - Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY).
- [Umar Jamil - Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU).
- [Algorithmic Simplicity - Mamba from scratch: Neural nets better and faster than transformers](https://www.youtube.com/watch?v=N6Piou4oYx8).
