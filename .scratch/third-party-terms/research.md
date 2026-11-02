# Third-party terms against the two glossaries

This file compares the vocabulary of the primary sources with
`burn-mamba/GLOSSARY.md` and `burn-stack/GLOSSARY.md`. It reports facts. It does
not recommend a rename.

This file is a best-effort draft. Check a claim against its citation before you
rely on it.

## 1. Sources

### Path aliases

| Alias | Path |
|---|---|
| `M1/` | `/shared/claude/papers/mamba/mamba-1/src/` |
| `M2/` | `/shared/claude/papers/mamba/mamba-2/structure/` |
| `M3/` | `/shared/claude/papers/mamba/mamba-3/` |
| `BLOG/` | `/shared/claude/papers/mamba/blogs/goombalab/mamba2/` |
| `SSM/` | `/shared/claude/py/state-spaces/mamba/mamba_ssm/` (commit `a14b1df`) |
| `TRI/` | `SSM/ops/triton/mamba3/` |
| `TIL/` | `SSM/ops/tilelang/mamba3/` |
| `SSR` | `/shared/claude/py/state-spaces/mamba/README.md` |
| `MM/` | `/shared/claude/py/VikramLex/mamba3-minimal/` (commit `15ba998`) |
| `HN/` | `/shared/claude/papers/hnet/` |
| `HNM` | `HN/sections/2_method_v2.tex` |
| `MGR` | `/shared/claude/papers/multi-gate-res/tex/main.tex` |
| `BM/` | `/shared/claude/burn-mamba/` |
| `BS/` | `/shared/claude/burn-stack/` |

A citation has the form `alias/file:line`. In one table cell, `:N` after a
citation is another line of the same file.

### Read

Read in full:

- `M3/structure/methods_complex.tex`, `methods_mimo.tex`, `methods_trap.tex`,
  `methods_arch.tex`, `preliminaries.tex`, `M3/appendix/mimo.tex`.
- `M2/glossary.tex`, `M2/architecture.tex`, `M2/efficient.tex`, `M2/code.py`.
- `SSM/modules/block.py`, `mamba3.py`, `mamba2.py`, `mamba_simple.py`,
  `ssd_minimal.py`.
- `SSM/models/mixer_seq_simple.py`, `SSM/models/config_mamba.py`.
- `TRI/mamba3_siso_combined.py`, `TRI/angle_dt.py`,
  `TRI/mamba3_mimo_rotary_step.py`.
- `MM/mamba3.py`, `MM/README.md`.
- `HNM`, `HN/tab/architecture.tex`, the macros of `HN/main.tex`.
- `MGR` lines 114-213 and 571-614.
- `/shared/claude/py/state-spaces/mamba/usage.md`.

Searched with `rg`, with the matched lines read:

- The rest of `M1/`, `M2/`, `M3/structure/`, `M3/appendix/`, `M3/tables/`.
- `BLOG/part1.md` to `part4.md`, `SSR`.
- `TRI/mamba3_siso_fwd.py`, `TIL/mamba3_mimo.py`, `TIL/mamba3_mimo_fwd.py`,
  `TIL/mamba3_mimo_bwd.py`.
- `SSM/ops/triton/ssd_combined.py`, `SSM/ops/triton/layernorm_gated.py`,
  `SSM/ops/utils/generation.py`, `SSM/modules/mamba2_simple.py`,
  `SSM/modules/mha.py`.
- `MM/demo.py`, `MM/tests/`.
- The other sections of `HN/sections/`, `HN/supplementary/supplementary.tex`,
  the caption in `HN/fig/main.tex`.
- The rest of `MGR`.

The comparison also read our side: `BM/GLOSSARY.md`, `BS/GLOSSARY.md`, the
Notation table of `BM/CLAUDE.md`, `BS/src/modules/layer.rs`,
`BM/src/mamba3/mamba3.rs`, `BM/src/mamba2/ssd/ssd_path.rs`, `BM/src/packing.rs`,
`BM/docs/adr/0003-two-mamba3-pathways.md`.

### Not read

| Source | Reason |
|---|---|
| The figure PDFs (`M3/fig/mamba3-arch.pdf`, `M1/../fig/architecture.pdf`, `MGR` `images/flow_mgr.pdf`, tried) | The Read tool failed: `pdftoppm` is not installed. I used the TeX captions. No claim here cites a figure. |
| `/shared/claude/papers/misc/tiny-stories/` | The directory holds only the TinyStories paper. No glossary term comes from it. I read its title line only. |
| A Muon paper | Not found. `papers/misc/` has none. The approved sources only cite Muon (`MGR:285`, `MGR:621`). |
| `/shared/claude/papers/multi-gate-res/html/` | Same title as the TeX. I read the title and counted a term. I did not compare the text. |
| `M3/ref/*.rs` | First 12 lines of `data.rs` only: music data code. The term searches excluded `*.rs`. |
| `M3/.structure.tex.swp`, `*.bib`, `*.bbl` | Not read. The term searches excluded them. |
| `papers/deltanet`, `papers/hrm`, `/shared/claude/hnet` | Excluded by the task. So `DeltaProduct` is checked only against the approved sources. |

## 2. Comparison tables

### Terms with no counterpart

I searched all approved sources for these terms. Each result is "not found":

- `Plant` (the word), `Micro-step`, `Lag`, `Tap` (the word), `Mass` (the word),
  `Tap pattern`, `Write axis`, `Read axis`.
- `MambaProduct`, `Quaternion4D`, `Rotor4D`, `Positive system`, `Gain` (as a
  system), `Tropical register`. `M1/background.tex:126` cites Kalman filters
  only as related work.
- `Real layer`, `Virtual layer`, `Application`, `Untied parameter`, `Layer
  schedule`, `Prime`, `Length hint`, `Marker level`, `Class marker`, `Bucket`.
- `Bidirectional pair`, `Gradient horizon`, `Tracked layer`, `Untracked span`,
  `Cut`, `Ghost latent`, `Start embedding`.

### 2.1 Mamba papers and blogs

Check of rotation in the paper: see 2.2, which holds the paper, the code and
mamba3-minimal side by side. The rows here give only the paper terms.

| Our term (glossary) | Source term | Relation | Difference | Citation |
|---|---|---|---|---|
| Layer (burn-stack) | "Mamba-2 layer", "Mamba-3 layer", "entire layer", "layer output" | false friend | The word names our block. The paper has no word for a Pre-LN residual around one block. | `M3/structure/preliminaries.tex:28`, `M3/structure/intro.tex:31`, `M3/structure/experiments.tex:39`, `M3/structure/methods_mimo.tex:71`, `M3/appendix/mimo.tex:23` |
| Layer (burn-stack) | "SSD layer", "selective SSM layer", "S6 layer", "core SSD layer" | false friend | The word names the recurrence inside the block, our plant. | `M2/architecture.tex:26`:36:167, `M3/structure/methods_arch.tex:14`, `BLOG/part1.md:29`:55 |
| Layer stack (burn-stack) | "Mamba architecture", "overall architecture" | renamed | The paper repeats a block with norm and residual, and gives the result no other name. | `M1/method.tex:235`, `M3/structure/methods_arch.tex:10` |
| Block (burn-stack) | "MLP block", "SwiGLU block" | split | The source word covers our block and the feed-forward part of our layer. | `M1/method.tex:223`:237, `M3/structure/methods_arch.tex:10`, `BS/src/modules/layer.rs:18` |
| Block (burn-stack) | "block decomposition", "diagonal block", "block size" | false friend | The word names a submatrix of `M` or a part of the sequence, our chunk. | `M2/efficient.tex:27`:40:103, `M2/code.py:11`:24, `BLOG/part3.md:95` |
| Family (burn-stack) | "Mamba variants", "Mamba-3 variants" | split | The source term covers families and options: Mamba-2, Mamba-3 SISO, Mamba-3 MIMO. | `M3/appendix/results.tex:46`:175, `M3/structure/intro.tex:45`, `M3/tables/decode_cudagraph_timing.tex:7` |
| Plant (burn-mamba) | "SSM", "SSM recurrence", "core recurrence" | split | The source counts several SSMs in one layer, one for each head or channel. Our block has one plant. | `M3/structure/preliminaries.tex:39`, `M3/structure/methods_mimo.tex:21`:66, `M3/appendix/mimo.tex:29`, `M1/background.tex:8` |
| Plant: its write (burn-mamba) | "state-input", "state projection", "expansion" | renamed | `B` in `M2` already holds the step size. `B` in `M3` does not. | `M3/structure/preliminaries.tex:19`:85, `M3/structure/methods_mimo.tex:59`, `M2/glossary.tex:24` |
| Plant: its readout (burn-mamba) | "state emission", "contraction", "output projection" | renamed | Same concept. | `M3/structure/methods_mimo.tex:31`:59, `M2/glossary.tex:25`, `M2/architecture.tex:152` |
| State (burn-mamba) | "hidden state", "latent state", "SSM state", "recurrent state" | renamed | Same `h`. All four are `_Avoid_` entries of ours. | `M3/structure/preliminaries.tex:18`, `M1/background.tex:27`, `M2/related.tex:30`, `M3/structure/related.tex:16` |
| Gate (burn-mamba) | "the gate" `g_t` | false friend | Mamba-1 names the scalar of the recurrence at `N=1`. It is not `z`. | `M1/method.tex:277`:290:332 |
| Gate (burn-mamba) | "data-dependent gate" `λ_t` | false friend | Mamba-3 names the split of the trapezoid. It is not `z`. | `M3/appendix/discretization.tex:153` |
| Gate (burn-mamba) | "residual gate term" `Z_t` | renamed | Same `z`. "Residual" here does not mean the skip of a layer. | `M3/appendix/mimo.tex:13`:22 |
| SSD path (burn-mamba) | "SSD algorithm", "chunked algorithm", "state-passing mode" | split | One source term covers our three SSD paths. "Mode" also names forms with no chunks. | `M2/efficient.tex:9`, `M2/scan.tex:114`:118, `M3/structure/methods_mimo.tex:53` |
| Segment (burn-stack) | "segments", "local segment", `segsum` | false friend | `M3`: parts of one long sequence, our windows. `M2`: a chunk. `M1`: a DNA sample. `segsum`: a sum over an index range of `A`. | `M3/appendix/results.tex:183`, `M2/scan.tex:191`, `M1/appendix.tex:369`, `M2/code.py:1` |
| Packed row (burn-stack) | "treating the whole batch as one long sequence", "variable sequence lengths" | renamed | The source joins the whole batch. Ours joins sequences inside each row. | `M2/systems.tex:110`:115:116 |
| Reset (burn-stack) | "reset their state" | false friend | The source names a learned behaviour (`Δ→∞` or `g→1`), not a mark on a token. | `M1/method.tex:313`:320 |
| Tied parameter (burn-stack) | "tied across heads" | false friend | The source shares across heads. Ours shares across the applications of a real layer. | `M2/architecture.tex:70` |
| Fused projection (burn-stack) | "single projection", "parallel projections" | renamed | Same concept, with no noun for the weight itself. | `M2/architecture.tex:28`:37 |
| Fused projection (burn-stack) | "fused selective scan layer", "fusion structure" | false friend | The source fuses several operations into one kernel, not several maps into one weight. | `M1/method.tex:201`, `M3/appendix/results.tex:110`:119 |
| Trapezoid (burn-mamba) | "exponential-trapezoidal discretization", "generalized trapezoidal rule" | renamed | Same rule, with one tap at `t−1`. | `M3/structure/methods_trap.tex:1`:119 |
| Trapezoid (burn-mamba) | "Trapezoidal", "classical trapezoid rule" | false friend | The source names the classic LTI rule with a fixed average, as in S4. | `M3/structure/methods_trap.tex:42`:121:142 |
| Tap (burn-mamba) | "interval endpoint", "convolution of size two", "2-band" | renamed | The source has one tap only, at `t−1`. | `M3/structure/methods_trap.tex:120`:156:171 |
| Mass (burn-mamba) | `γ_t`, `β_t`, "weights β, γ" | split | Mass covers `λΔ` and `(1−λ)Δ`. `β_t` is `(1−λ)Δ·α`, so it holds the decay. | `M3/structure/methods_trap.tex:138`:200, `BM/src/mamba3/mamba3.rs:27` |
| Rotation (burn-mamba) | "data-dependent RoPE", "data-dependent rotation matrices", "complex SSM", "complex transitions" | renamed | Same factor. The paper does not separate it from the RoPE word. | `M3/structure/methods_complex.tex:13`:16:58:96, `M3/structure/experiments.tex:8` |
| Rotation (burn-mamba) | "RoPE" (alone), "RoPE-like" | split | The word covers our rotation, our RoPE trick and our rotary position embedding. | `M3/structure/experiments.tex:54`, `M3/structure/intro.tex:40`, `M3/tables/ablations.tex:39`, `M3/tables/retrieval.tex:6` |
| Rotary position embedding (burn-mamba) | "vanilla RoPE", "standard RoPE", "Std. RoPE" | renamed | Same fixed-frequency encoding. | `M3/structure/methods_complex.tex:73`:96, `M3/tables/ablations.tex:38` |
| Rotary position embedding (burn-mamba) | "data-dependent rotary embeddings", "rotary embedding" | false friend | The source uses the embedding word for the data-dependent rotation. | `M3/structure/methods_complex.tex:15`:62:80, `M3/structure/methods_arch.tex:6` |
| Rotation kind (burn-mamba) | "complex SSM", "real-valued SSM", "without RoPE" | split | The paper has two forms: our `Complex2D` and our `Real1D`. | `M3/structure/methods_complex.tex:30`:96, `M3/structure/experiments.tex:54` |
| SISO (burn-mamba) | "SISO SSM", "SISO variant" | false friend | The paper treats SISO as its own default form. Ours says that it is not a separate form. | `M3/structure/methods_arch.tex:15`, `M3/structure/methods_mimo.tex:21`, `M3/appendix/mimo.tex:32` |
| Micro-step (burn-mamba) | "time step", "recurrent step" | renamed | The source has exactly one for each token. | `M3/structure/methods_trap.tex:5`, `M3/structure/preliminaries.tex:49`, `M3/structure/related.tex:16` |
| Write axis, Read axis (burn-mamba) | `T`, "time axis", "target sequence axis" | split | One source axis covers our two axes. | `M2/glossary.tex:15`, `M2/background.tex:116`, `M3/structure/preliminaries.tex:6` |
| Origin (burn-mamba) | "initial state (to the chunk)", "each worker takes an initial state" | split | The source term names the start of a chunk or of a worker part, not a cache for each segment. | `M2/efficient.tex:108`, `M2/systems.tex:102` |
| Training step (burn-stack) | "gradient step" | renamed | Same concept. `M1` also uses "training step". | `M1/appendix.tex:422`:456 |

### 2.2 Official code (state-spaces/mamba)

**Check 1: does `Block` wrap a `mixer` with a norm and a residual?** Yes. The
docstring says "Simple block wrapping a mixer class with LayerNorm/RMSNorm and
residual connection" (`SSM/modules/block.py:15`). The class holds `norm`,
`mixer`, and an optional `norm2` with `mlp` (`SSM/modules/block.py:29`:30:32:33).

- Their `Block` maps to our layer. Their `mixer` maps to our block.
- The order differs from the claim. It is "Add -> LN -> Mixer"
  (`SSM/modules/block.py:21`). `forward` adds the residual of the previous
  `Block` first, and it returns `(hidden_states, residual)`
  (`SSM/modules/block.py:52`:88). `MixerModel.forward` runs the last add
  (`SSM/models/mixer_seq_simple.py:197`). A comment says "The model definition
  is unchanged" (`SSM/models/mixer_seq_simple.py:145`).
- Our `Layer` computes `M(RMSNorm(x))` and "does **not** apply the residual".
  `Layers` owns the outer add (`BS/src/modules/layer.rs:13`:14). The glossary
  line "A Pre-LN residual around one block" (`BS/GLOSSARY.md:41`) describes the
  pair of `Layer` and `Layers`. Our `Layer` also has an optional MLP
  (`BS/src/modules/layer.rs:18`).
- `create_block` accepts only `"Mamba1"` and `"Mamba2"`
  (`SSM/models/mixer_seq_simple.py:53`:54). `Mamba3` is a standalone module
  (`SSR:106`). No code wraps `Mamba3` in a `Block`.
- A `mixer` can also be `MHA` (`SSM/models/mixer_seq_simple.py:63`).
- `SSR` uses "Mamba block" for the class `Mamba` (`SSR:55`:57:59), which is our
  block. So "block" has two meanings inside this repository.

**Check 2: rotation terms across the three Mamba-3 sources.**

| Ours | Mamba-3 paper | Official code | mamba3-minimal |
|---|---|---|---|
| Rotation | "data-dependent rotary embeddings" (`M3/structure/methods_complex.tex:15`), "data-dependent RoPE" (:16:96), block-diagonal "rotation matrix" `R_t` (:58:91) | `Angles`: "Rotary position embeddings" (`TIL/mamba3_mimo.py:262`), "rotary position embeddings" (`TRI/mamba3_siso_combined.py:311`), "RoPE State" (`SSM/modules/mamba3.py:418`) | "Data-Dependent RoPE" (`MM/mamba3.py:15`:358:536), "cumulative rotation" (:449) |
| RoPE trick | "RoPE trick" with the same meaning (`M3/structure/methods_complex.tex:17`:68) | No hit for "trick" in the code. The kernels do it in "Phase 1" (`TRI/mamba3_siso_fwd.py:256`:312) | "RoPE trick" (`MM/mamba3.py:665`) |
| Rotation kind | Not found as a term. Complex against real only (`M3/structure/methods_complex.tex:30`). `Quaternion4D`, `Rotor4D`: not found | No selector. `assert self.num_rope_angles > 0` excludes `Real1D` (`SSM/modules/mamba3.py:83`) | No selector. `θ` always exists (`MM/mamba3.py:271`) |
| Rotation fraction | Not found. `θ` has `N/2` entries (`M3/structure/methods_complex.tex:29`) | `rope_fraction`, only 0.5 or 1.0 (`SSM/modules/mamba3.py:36`:77). `rotary_dim_divisor = 2/rope_fraction` (:78). `split_tensor_size` (:79). `rotary_dim` (`TRI/mamba3_mimo_rotary_step.py:183`) | No knob. Always `d_state // 2` angles (`MM/mamba3.py:276`:343) |
| Rotation range | Not found. The propositions state no bound on `θ` (`M3/structure/methods_complex.tex:23`:44) | No word and no knob. The value is `tanh(angle)·π·dt` (`TRI/angle_dt.py:94`:101, `TRI/mamba3_mimo_rotary_step.py:75`:77) | No bound. `Δ·θ` with no squash (`MM/mamba3.py:363`) |
| Rotary position embedding | "vanilla RoPE" (`M3/structure/methods_complex.tex:73`), "standard RoPE" (:96) | "original RoPE paper" (`TRI/mamba3_mimo_rotary_step.py:46`) | "standard RoPE" (`MM/mamba3.py:666`) |

The cumulative angle has three names: `angle_dt_state` (`SSM/modules/mamba3.py:419`),
`Angles_Cumsum` (`TRI/mamba3_siso_combined.py:315`), `cum_angle`
(`MM/mamba3.py:96`). Its sign differs. The official kernel accumulates `+Δθ`
(`TRI/angle_dt.py:104`). mamba3-minimal accumulates `−Δθ` (`MM/mamba3.py:367`).
The paper writes `R^⊤` (`M3/structure/methods_complex.tex:64`).

The official pair layout has its own names, and it is not a rotation kind:
`rotate_pairwise`, "Interleaved rotary", "Halved rotary"
(`TRI/mamba3_mimo_rotary_step.py:46`:279:301).

| Our term (glossary) | Source term | Relation | Difference | Citation |
|---|---|---|---|---|
| Block (burn-stack) | class `Block`, `create_block` | false friend | It holds norm, mixer and an optional MLP. That is our layer. | `SSM/modules/block.py:10`:15, `SSM/models/mixer_seq_simple.py:29`:73 |
| Block (burn-stack) | `mixer`, `mixer_cls`, classes `Mamba`, `Mamba2`, `Mamba3`, `MHA` | renamed | `mixer` is an `_Avoid_` entry of ours. | `SSM/modules/block.py:30`:67, `SSM/models/mixer_seq_simple.py:56`:63 |
| Block (burn-stack) | `BLOCK_D`, `BLOCK_SIZE_M`, `q_pre_block`, `angle_block` | false friend | The word names a memory tile of a kernel. | `TRI/angle_dt.py:166`, `SSM/ops/triton/ssd_combined.py:80`, `TRI/mamba3_siso_fwd.py:264`:268 |
| Chunk (burn-stack) | `block_len`, "blocks/chunks" | renamed | `chunk_size` is the usual name. `block_len` is in the Listing 1 form only. | `SSM/modules/ssd_minimal.py:34`:47 |
| Layer (burn-stack) | class `Block`, `self.layers`, `layer_idx`, `n_layer` | renamed | Their add runs at the entry of the next `Block`. Ours runs in `Layers`. | `SSM/modules/block.py:21`:52, `SSM/models/mixer_seq_simple.py:152`:193, `BS/src/modules/layer.rs:13` |
| Layer (burn-stack) | `ssm_cfg["layer"]`, `ssm_layer`, "selective SSM layer" | false friend | The config key selects a family. `SSR` uses the word for the recurrence. | `SSM/models/mixer_seq_simple.py:53`:54, `SSR:51` |
| Family (burn-stack) | `ssm_layer` values `"Mamba1"`, `"Mamba2"`, `mixer_cls` | renamed | Same concept. | `SSM/models/mixer_seq_simple.py:53`:57 |
| Layer stack (burn-stack) | `MixerModel`, `backbone` | split | `MixerModel` holds the embedding, the layers and a final norm, so it crosses our layer stack and our network. | `SSM/models/mixer_seq_simple.py:118`:140:152:171:241 |
| Network (burn-stack) | `MambaLMHeadModel` | renamed | Same concept: `backbone` plus `lm_head`. | `SSM/models/mixer_seq_simple.py:215`:241:255 |
| State (burn-mamba) | `ssm_state`, `last_state` | renamed | `ssm_state` is the full `[batch, nheads, headdim, d_state]` tensor. | `SSM/modules/mamba3.py:426`:427, `SSM/modules/mamba2.py:169` |
| State (burn-mamba) | `hidden_states`, `hidden_dim` | false friend | The word names the `d_model` tensor that passes through the layers. | `SSM/modules/block.py:43`:67, `SSM/modules/mamba_simple.py:119`:121, `SSM/modules/mamba2.py:156` |
| Accumulator (burn-mamba) | `acc_ssm_states`, `Final_SSM_State`, `ssm_state` | split | `ssm_state` holds the accumulator after a forward and the state in `step`. The two are equal at call boundaries. | `TRI/mamba3_siso_fwd.py:336`:352:423:431, `SSM/modules/mamba3.py:241`:286, `BM/docs/adr/0003-two-mamba3-pathways.md:8`:13 |
| Accumulator (burn-mamba) | "Output Accumulators", `accum_dtype` | false friend | The word names a gradient buffer and a dtype. | `TIL/mamba3_mimo_bwd.py:140`, `TIL/mamba3_mimo_fwd.py:57` |
| Pathway (burn-mamba) | "Mamba-3 Combined Kernel" | renamed | The repository has no pathway term. Its only form is our single-SSD. | `SSM/modules/mamba3.py:425`, `TRI/mamba3_siso_fwd.py:285`:287, `BM/docs/adr/0003-two-mamba3-pathways.md:5` |
| Mass (burn-mamba) | `gamma`, `shifted_gamma`, `scale` | split | `gamma` is `λΔ`. `shifted_gamma` is `(1−λ)Δ` of the next step, which is a tap mass and not `γ`. | `TRI/mamba3_siso_fwd.py:285`:286:287, `TIL/mamba3_mimo_fwd.py:179`:193 |
| Tap (burn-mamba) | `k_state`, `v_state`, `input_k_state`, `input_v_state` | renamed | "K (=B) State" and "V (=x) State" hold the previous sample. | `SSM/modules/mamba3.py:433`:447, `TRI/mamba3_siso_fwd.py:351`:352 |
| Trapezoid (burn-mamba) | `trap`, `Trap`, "Trapezoidal factor" | false friend | The word names `λ`, not the rule. It is pre-sigmoid in `forward` and post-sigmoid in `_preprocess`. | `TRI/mamba3_siso_combined.py:323`, `TIL/mamba3_mimo.py:255`, `SSM/modules/mamba3.py:166`:257 |
| Cache (burn-stack) | `InferenceParams`, `inference_params`, `key_value_memory_dict`, `allocate_inference_cache` | renamed | The dict holds all layers, keyed by `layer_idx`. `MambaCaches` is our closest form. | `SSM/ops/utils/generation.py:18`:26, `SSM/modules/block.py:90`, `SSM/modules/mamba3.py:489` |
| Forward (burn-stack) | `forward()` | false friend | At `seqlen_offset > 0`, `forward` runs `step`. So it also serves one-token calls. | `SSM/modules/mamba3.py:146`:147, `SSM/modules/mamba2.py:173`:175, `SSM/modules/mamba_simple.py:129`:131 |
| Packed row (burn-stack) | `cu_seqlens`, "varlen mode", `seq_idx` | renamed | `cu_seqlens` requires batch 1. Ours keeps the batch axis. | `TRI/mamba3_siso_combined.py:341`:344:379, `SSM/ops/triton/ssd_combined.py:637` |
| Segment (burn-stack) | `num_sequences`, "sequences" | renamed | Same concept. | `TRI/mamba3_siso_combined.py:342`:362 |
| Segment (burn-stack) | `segsum` | false friend | The word names a sum over an index range of `A`. | `SSM/modules/ssd_minimal.py:14`:23 |
| Reset (burn-stack) | a change of `seq_idx`, an entry of `cu_seqlens` | renamed | The source holds an index or an offset, not a mark. | `SSM/ops/triton/ssd_combined.py:155`:158, `TRI/angle_dt.py:57`:58 |
| Reset (burn-stack) | `InferenceParams.reset()` | false friend | The method clears the counters of a whole cache object. | `SSM/ops/utils/generation.py:29` |
| Origin (burn-mamba) | `initial_states`, `Input_States`, `init_state`, `init_states` | split | `Input_States` has one entry for each sequence. `init_states` is learnable. Ours is one cache that each segment starts from, and it is not learnable. | `SSM/ops/triton/ssd_combined.py:636`, `TRI/mamba3_siso_combined.py:334`:335, `SSM/modules/mamba2_simple.py:87`:88, `BM/src/packing.rs:2`:6 |
| Gate (burn-mamba) | `z`, `Z`, "Gating tensor", `RMSNormGated` | renamed | In Mamba-2 with `rmsnorm=True`, the gate runs inside the norm. | `TRI/mamba3_siso_combined.py:332`, `SSM/modules/mamba2.py:269`:270, `SSM/ops/triton/layernorm_gated.py:343` |
| Rotation (burn-mamba) | `rope`, `rotary`, `Angles`, "Rotary angles" | renamed | Same factor. | `SSM/modules/mamba3.py:76`:418, `TRI/mamba3_siso_fwd.py:267`:470 |
| Rotary position embedding (burn-mamba) | "rotary position embeddings", "Rotary embeddings" | false friend | The code uses the transformer name for the data-dependent rotation. | `TRI/mamba3_siso_combined.py:311`, `TIL/mamba3_mimo.py:262`, `TRI/mamba3_siso_fwd.py:312` |
| Rotation fraction (burn-mamba) | `rope_fraction`, `rotary_dim_divisor`, `split_tensor_size`, `rotary_dim` | renamed | The divisor is `2/fraction`, so divisor 2 means the whole state. | `SSM/modules/mamba3.py:36`:77:78:79, `TIL/mamba3_mimo.py:266`:299 |
| Rotation range (burn-mamba) | fixed `tanh(angle)·π` | renamed | The code has no word and no knob. | `TRI/angle_dt.py:94`:101 |
| SISO (burn-mamba) | `is_mimo=False`, `mamba3_siso_*` | split | The code sets `mimo_rank = 1` and still runs a separate kernel. | `SSM/modules/mamba3.py:66`:67:184:221 |
| MIMO rank (burn-mamba) | `mimo_dim`, `R` | renamed | `mimo_rank` is the same word in the module. The kernels use these two. | `TRI/mamba3_mimo_rotary_step.py:29`, `TIL/mamba3_mimo_fwd.py:45` |
| SSD path (burn-mamba) | `ssd_minimal_discrete`, `mamba_chunk_scan`, `mamba_chunk_scan_combined`, `use_mem_eff_path` | renamed | Our three paths cite these kernels as their base. `use_mem_eff_path` selects the split-conv combined kernel. | `SSM/modules/ssd_minimal.py:34`, `SSM/ops/triton/ssd_combined.py:624`:646:977, `SSM/modules/mamba2.py:60`:184, `BM/src/mamba2/ssd/ssd_path.rs:27`:36:50 |
| Fused projection (burn-stack) | `in_proj`, `zxbcdt`, `zxBCdtAtrap` | renamed | Same concept. | `SSM/modules/mamba3.py:85`:87:151, `SSM/modules/mamba2.py:95`:178 |
| Sub-projection (burn-stack) | the parts of `torch.split`: `z`, `x`, `B`, `C`, `dd_dt`, `dd_A`, `trap`, `angles` | renamed | The source has no noun for one part. | `SSM/modules/mamba3.py:152`:161, `SSM/modules/mamba2.py:211` |
| Fused projection (burn-stack) | `fused_add_norm`, "Fused kernel", `use_fast_path` | false friend | The source fuses operations into one kernel. | `SSM/modules/block.py:12`:23, `SSM/modules/mamba2.py:58`, `SSM/modules/mamba_simple.py:46` |
| Tied parameter (burn-stack) | `tie_embeddings`, `tie_weights` | false friend | The LM head shares the weight of the embedding. | `SSM/models/config_mamba.py:18`, `SSM/models/mixer_seq_simple.py:267`:269 |
| Captured step (burn-stack) | "CUDA graph decoding", `capture_graph`, `cg` | renamed | The source records decode calls only. | `SSM/ops/utils/generation.py:133`:310:342 |
| Recording (burn-stack) | `graph`, `torch.cuda.CUDAGraph` | renamed | Same concept. `graph.replay()` uses our word "replay". | `SSM/ops/utils/generation.py:372`:385 |
| Padding (burn-stack) | `pad_vocab_size_multiple` | false friend | The word names a round-up of the vocabulary size. | `SSM/models/config_mamba.py:17`, `SSM/models/mixer_seq_simple.py:235`:239 |
| Write axis, Read axis (burn-mamba) | `seqlen`, `S` | split | One source axis covers our two axes. | `TRI/mamba3_siso_combined.py:318`, `TIL/mamba3_mimo_fwd.py:40` |

### 2.3 mamba3-minimal

| Our term (glossary) | Source term | Relation | Difference | Citation |
|---|---|---|---|---|
| Block (burn-stack) | `mixer`, class `Mamba3`, "the core mixer", "Mamba-3 SSM block" | renamed | Same concept. | `MM/mamba3.py:136`:177:241:246 |
| Block (burn-stack) | "SSM blocks and SwiGLU MLP blocks", "before each block" | split | The source word covers our block and the feed-forward part of our layer. | `MM/mamba3.py:21`:855 |
| Layer (burn-stack) | `layers[i]`, `n_layer`, "each = SSM mixer + SwiGLU MLP" | renamed | The source layer always has an MLP. The model loop runs both adds. | `MM/mamba3.py:63`:130:175:181 |
| Network (burn-stack) | `Mamba3LMHeadModel`, `backbone` | renamed | Same concept. | `MM/mamba3.py:119`:127:147 |
| Cache (burn-stack) | `InferenceCache`, with `ssm_state`, `prev_Bx`, `cum_angle` | renamed | Same concept. It holds the product `B⊗x`. | `MM/mamba3.py:93`:94:95:96 |
| State (burn-mamba) | `ssm_state`, "SSM hidden state h_t" | renamed | Same concept. | `MM/mamba3.py:94` |
| State (burn-mamba) | `h`, "hidden states", "hidden state" | false friend | `h` holds the whole `InferenceCache`, not only the state. | `MM/mamba3.py:154`:159:317:501 |
| Pathway (burn-mamba) | "two-SSD decomposition", "γ SSD", "β SSD" | renamed | The source has one β SSD at `t−1`. Ours has one β-SSD for each tap. | `MM/README.md:61`:69:70, `MM/mamba3.py:404`:453:723, `BM/src/mamba3/mamba3.rs:166` |
| Mass (burn-mamba) | "Trapezoidal coefficients", "left-endpoint coefficient", "right-endpoint coefficient" | split | `beta` holds `alpha`, so it is not a share of `Δ`. | `MM/mamba3.py:369`:375:376:547:548 |
| Tap (burn-mamba) | "left-endpoint (previous input)", `B_prev`, `x_prev`, `prev_Bx`, "β term" | renamed | One tap at `t−1`. The cache holds the product. Ours keeps `B` and `x` in slots. | `MM/mamba3.py:95`:371:411:459, `MM/README.md:70`, `BM/src/mamba3/mamba3.rs:1854` |
| Gate (burn-mamba) | "forget-gate (α)", "input-gate (γ)" | false friend | A comment uses the word for the decay and the current mass. | `MM/mamba3.py:349` |
| Trapezoid (burn-mamba) | "Trapezoidal Discretization" | renamed | Same rule, with one tap. | `MM/mamba3.py:14`, `MM/README.md:20` |
| Rotation (burn-mamba) | "Data-Dependent RoPE", "cumulative rotation", `cum_angle`, `apply_rope`, "Complex SSM / RoPE" | renamed | No bound on the angle and no fraction. | `MM/mamba3.py:15`:96:358:449:661, `MM/README.md:21` |
| Rotary position embedding (burn-mamba) | "rotary position embedding" | false friend | The docstring of `apply_rope` uses the word for the data-dependent rotation. | `MM/mamba3.py:662` |
| Rotary position embedding (burn-mamba) | "standard RoPE" | renamed | Same fixed-frequency encoding. | `MM/mamba3.py:666` |
| Chunk (burn-stack) | `chunk_size`, "Matrix partition size", "partition size Q" | renamed | Same concept. | `MM/mamba3.py:67`:732 |
| Forward (burn-stack) | `forward()` | false friend | With `h` present, `forward` runs the one-token path. | `MM/mamba3.py:159`:323 |
| Tied parameter (burn-stack) | "Weight tying" | false friend | The LM head shares the weight of the embedding. | `MM/mamba3.py:150`:151 |
| SISO (burn-mamba) | `use_mimo=False`, "SISO path", `ssd` against `ssd_mimo` | split | The source has a separate branch and a separate SSD function. | `MM/mamba3.py:70`:444:719:785 |
| SSD path (burn-mamba) | `ssd`, "the chunked parallel algorithm" | renamed | One path only, the form of our `Minimal`. | `MM/mamba3.py:693`:719:720 |
| Prefill (burn-stack) | "Process prompt" | renamed | The tail of the prompt runs with one-token calls. | `MM/mamba3.py:199`:203:211 |
| Decode (burn-stack) | `generate`, "Auto-regressive generation" | renamed | Same concept. | `MM/mamba3.py:187`:196 |

### 2.4 H-Net

**Check 4: source terms.** The source defines these words through macros
(`HN/main.tex:97`:103:105:107:109:112).

| Ours | Source term | Citation |
|---|---|---|
| H-Net | "H-Net": same word, same meaning | `HNM:6` |
| Stage | "stage", index `s`. An `S`-stage model has `M` at `s=S` | `HNM:22` |
| Encoder | "encoder network", `E^s` | `HN/main.tex:103`, `HNM:17` |
| Decoder | "decoder network", `D^s` | `HN/main.tex:107`, `HNM:17` |
| Router | "routing module". Also "chunking router", "boundary predictor" | `HN/main.tex:109`, `HNM:53`, `HN/supplementary/supplementary.tex:65` |
| Main stack | "main network", `M`. Also "main stage" | `HN/main.tex:105`, `HNM:53` |
| Chunk | "chunk": same word, same meaning. One retained vector stands for one chunk | `HNM:61`:124 |
| Chunk boundary | "chunk boundaries", "chunking boundaries", "boundaries", `b_t` | `HNM:27`:111:133 |

| Our term (glossary) | Source term | Relation | Difference | Citation |
|---|---|---|---|---|
| Stage (burn-stack) | "stage", "main stage" | false friend | Index `s=S` holds only the main network. Ours always has an encoder, a router and a decoder. | `HNM:22`:53, `HN/supplementary/supplementary.tex:108` |
| Stage (burn-stack) | "level", "multi-level" | renamed | `level` is an `_Avoid_` entry of ours. | `HNM:7`:19, `HN/sections/0_abstract.tex:9` |
| Encoder (burn-stack) | "encoder network" | renamed | The source puts an RMSNorm at the end of each network. | `HN/main.tex:103`, `HNM:17`:229 |
| Decoder (burn-stack) | "decoder network" | renamed | Same note as the encoder. | `HN/main.tex:107`, `HNM:17`:229 |
| Router (burn-stack) | "routing module", "chunking router", "boundary predictor" | renamed | It computes `p_t` and `b_t`. The "chunking layer" holds it with the downsampler. | `HN/main.tex:109`, `HNM:53`:105:111, `HN/supplementary/supplementary.tex:65` |
| Main stack (burn-stack) | "main network" | split | One source term covers our main stack and our inner H-Net. | `HNM:17`:22:75 |
| Inner H-Net (burn-stack) | "main network", "inner stages" | split | "the main network can itself be an H-Net". | `HNM:19`:256, `HN/sections/1_introduction.tex:62` |
| Chunk boundary (burn-stack) | "boundaries", `b_t` | renamed | `b_t = 1` marks a position, and `p_1 = 1.0`. An intro line says "between adjacent elements". | `HNM:111`:117, `HN/sections/1_introduction.tex:41` |
| Layer stack (burn-stack) | "network", "stack of sequence mixing layers" | renamed | Same concept, plus the RMSNorm at its end. | `HNM:17`:229 |
| Network (burn-stack) | "network", "network component", "sub-networks" | false friend | The source network is our layer stack. Ours adds an input map and an output map. | `HNM:17`:94:229, `BS/GLOSSARY.md:70` |
| Layer (burn-stack) | "sequence mixing layer", "residual block", `M`, `T` | renamed | A Mamba layer has a gated MLP in `M`, and none in `E` or `D`. | `HNM:17`:228, `HN/sections/3_experiments.tex:50`, `HN/tab/architecture.tex:4` |
| State (burn-mamba) | "state", "DC state", "isotropic state" | false friend | The word names what each component keeps between steps, including a KV cache. | `HNM:279`:282:283 |
| Cache (burn-stack) | "state" | renamed | Same concept as the row above. | `HNM:279` |
| Residual stream (burn-stack) | "Separation of Two Streams" | false friend | The two streams are the two uses of an encoder output. | `HNM:237`:238 |
| Residual scheme (burn-stack) | "residual connection" | false friend | The source connection runs from an encoder to its decoder, through a `Linear`. | `HNM:46`:61:233:242 |
| Window (burn-stack) | "window" of Sliding Window Attention, "width-k window" | false friend | The word names an attention range or a pool width. | `HN/tab/architecture.tex:8`, `HN/sections/A_background.tex:45` |
| Segment (burn-stack) | "segmentation", "segment data" | false friend | The word names chunking. | `HN/sections/0_abstract.tex:6`, `HN/sections/1_introduction.tex:40` |
| Straight-through carry (burn-stack) | "Straight-Through Estimator", `STE` | false friend | STE rounds a confidence to 1.0 and keeps its gradient. It adds no value-zero input at a cut. | `HNM:170`:189 |
| Token (burn-stack) | "token", "BPE token", "Tokenizer" | split | A footnote gives "one time step". Elsewhere it means a BPE token. The `Tokenizer` column lists the chunking mechanism. | `HNM:80`:285, `HN/tab/architecture.tex:5` |

Same word with the same meaning: "prefill" and "step" (`HNM:279`:285), "residual
stream" (`HNM:232`). `Start embedding`: not found in `HN/sections/`,
`HN/supplementary/` or `HN/tab/`. The nearest fact is `p_1 = 1.0`, which forces a
boundary at the first position (`HNM:117`).

### 2.5 Multi-Gate Residuals

**Check 5: source terms.**

- Residual stream: "residual stream", same word and same meaning
  (`MGR:141`:150:172). Also "stream" `s_i`, "stream state", `streams`,
  `n_stream` (`MGR:105`:171:574:595). "Depth-stream" and "lane": not found.
- β gate: the coefficient is `β_{i←l}`, "interpolation coefficient"
  (`MGR:171`:174:179). The structure is "gating layer" and "mixer"
  (`MGR:148`:172:264:275). The update is `s_i' = (1−β_i)s_i + β_i F_l(h_l)`
  (`MGR:193`). This matches "how far that stream moves towards the output of a
  layer".

| Our term (glossary) | Source term | Relation | Difference | Citation |
|---|---|---|---|---|
| β gate (burn-stack) | `β_{i←l}`, "interpolation coefficient", "gating layer", "mixer", "residual stream mixers" | renamed | The source has two forms: sigmoid ("independent") and softmax ("competitive"). `mixer` is an `_Avoid_` entry of ours. | `MGR:148`:171:172:177:264:600 |
| Residual scheme (burn-stack) | "residual connection", "residual structure", "additive residual update" | renamed | Same concept. | `MGR:116`:117:290:551 |
| State (burn-mamba) | "hidden state" `h_l` | false friend | The word names the input of a layer, pooled from the streams. | `MGR:105`:155:432:468 |
| Block (burn-stack) | "block", "Transformer block", "block depth" | false friend | The word names a whole Transformer block. | `MGR:268`:318:339:424 |
| Layer (burn-stack) | "attention layer", "feedforward layer", `F_l` | split | Streams merge from attention layers and feedforward layers. So one block has more than one source layer. | `MGR:193`:432:467:487 |
| Gate (burn-mamba) | "gate", "forget gate score", "gate bias" | false friend | The word names `β`, not the SiLU gate `z`. | `MGR:189`:235:544 |
| Straight-through carry (burn-stack) | "carry behavior" | false friend | From Highway Networks: `1−β` near 1 at the start of training. | `MGR:197` |
| Stage (burn-stack) | "lerping stage" | false friend | The word names the layers after the first `n`, where the gates run. | `MGR:171`:266:420 |
| Fused projection (burn-stack) | "Kernel Fusion", "fuse core operators" | false friend | The source fuses operations into one kernel. | `MGR:495`:498:499 |
| Muon plan (burn-stack) | "dual-parameter-group setup" | renamed | The source sets Muon on "all 2D weight matrices". Ours is an allowlist with sub-projections. | `MGR:285`:621, `BS/GLOSSARY.md:252` |
| Fallback optimizer (burn-stack) | AdamW for "all remaining scalar parameters" | renamed | The source has no noun for the role. It names only AdamW. | `MGR:285`:621 |
| Fallback optimizer (burn-stack) | "Fallback Inversion" | false friend | The word names a memory scheme for the backward pass. | `MGR:544`:545:547 |
| I/O weight (burn-stack) | "embeddings" | split | The source names only embeddings. Ours also covers the LM head, the projections and the class marker tables. | `MGR:621`, `BS/GLOSSARY.md:266` |

## 3. False friends, ranked

Rank 1 has the highest risk of a wrong port. The rank is my judgement from the
facts in part 2. Each entry groups the rows of one word.

1. **`Block`, the official class** (`SSM/modules/block.py:10`). A port that maps
   it to our block adds a second Pre-LN and a second residual around the real
   block.
2. **`hidden_states`, `h`, "state"** (`SSM/modules/block.py:43`,
   `MM/mamba3.py:154`, `MGR:155`, `HNM:279`). A port that reads these as our state
   can store, carry or reset the wrong tensor.
3. **Kernel and einops letters** (part 5.3). A port can bind a letter to the wrong
   axis or tensor, and it can pass the shape checks when two dimensions have equal
   sizes.
4. **"layer"** (`M3/structure/preliminaries.tex:28`, `M2/architecture.tex:26`,
   `SSM/models/mixer_seq_simple.py:53`). The word can mean our block, our plant or
   our family, so a port can place a norm or a residual at the wrong level.
5. **"rotary position embeddings", "RoPE"** (`TRI/mamba3_siso_combined.py:311`,
   `M3/structure/methods_complex.tex:15`). A port can install a fixed-frequency
   table in place of the data-dependent angle.
6. **`trap`, "Trapezoidal"** (`SSM/modules/mamba3.py:166`:257,
   `M3/structure/methods_trap.tex:42`). A port can apply the sigmoid twice, because
   `trap` is pre-sigmoid in `forward` and post-sigmoid in `_preprocess`.
7. **"block" for a chunk, a submatrix or a tile** (`M2/code.py:11`,
   `TRI/angle_dt.py:166`). A port can bind `block_len` or `BLOCK_D` to a setting of
   our block instead of a chunk length or a kernel detail.
8. **"gate"** (`M1/method.tex:290`, `M3/appendix/discretization.tex:153`,
   `MM/mamba3.py:349`, `MGR:189`). A port can read `g_t`, `λ_t`, `α`, `γ` or `β` as
   the SiLU gate `z`.
9. **`forward()`** (`SSM/modules/mamba3.py:146`, `MM/mamba3.py:323`). A port that
   maps it to our Forward can miss the branch that runs `step`.
10. **"stage"** (`HNM:22`, `MGR:266`). A port can build an encoder, a router and a
    decoder at index `s=S`, where the source has only the main network.
11. **"network"** (`HNM:17`). A port can add an embedding and an LM head to each
    encoder and decoder.
12. **"residual connection", "two streams"** (`HNM:46`:237). A port can send the
    encoder-to-decoder link through the residual scheme of a layer stack.
13. **`SISO`** (`M3/structure/methods_arch.tex:15`, `SSM/modules/mamba3.py:184`). A
    port can keep a separate SISO branch that our glossary defines as rank 1.
14. **"segment", `segsum`** (`SSM/modules/ssd_minimal.py:14`,
    `M3/appendix/results.tex:183`, `HN/sections/0_abstract.tex:6`). A port can link
    these to packed segments, although they name a sum, a window or chunking.
15. **"reset"** (`M1/method.tex:320`, `SSM/ops/utils/generation.py:29`). A port can
    read a learned behaviour or a counter reset as the mark on the first token of a
    segment.
16. **"fused"** (`SSM/modules/block.py:12`, `MGR:498`). A port can read a kernel
    option as a fact about a fused projection, and so about the Muon plan.
17. **"tied"** (`M2/architecture.tex:70`, `SSM/models/config_mamba.py:18`). A port
    can read head sharing or embedding sharing as sharing across applications.
18. **"accumulator"** (`TIL/mamba3_mimo_bwd.py:140`). A port can read a gradient
    buffer as the running sum of single-SSD.
19. **"window"** (`HN/tab/architecture.tex:8`). A port can read an attention range
    as the part of a sequence that one forward takes.
20. **"straight-through", "carry"** (`HNM:170`, `MGR:197`). A port can read the
    estimator or the Highway carry as the value-zero input at a cut.
21. **`pad_vocab_size_multiple`** (`SSM/models/config_mamba.py:17`). A port can
    read a vocabulary round-up as padding of a row.
22. **"Fallback Inversion"** (`MGR:547`). A port can read a backward memory scheme
    as the fallback optimizer.

## 4. Source terms with no term in our glossaries

| Source term | Meaning | Closest concept of ours | Citation |
|---|---|---|---|
| `λ_t`, `trap`, `lam`, "trapezoidal parameter", "mixing factor", "convex-combination modulator" | The data-dependent split between the two endpoints | Mass, Trapezoid | `M3/structure/methods_trap.tex:138`:146, `TRI/mamba3_siso_combined.py:324`, `TIL/mamba3_mimo_fwd.py:114`, `MM/mamba3.py:270` |
| `Δ_t`, `dt`, `DT`, "step size" | The learned step of the recurrence | Mass (its definition uses "step size") | `M3/structure/preliminaries.tex:20`, `MM/mamba3.py:269` |
| `α_t`, `A_t`, `ADT`, `dA`, "decay" | The scalar decay of the transition | Plant (its transition) | `M3/structure/preliminaries.tex:32`, `SSM/modules/mamba3.py:172`, `MM/mamba3.py:370` |
| `θ_t`, `angles`, `angle_proj`, "Rotary angle rates" | The projected rate of the rotation | Rotation | `M3/structure/methods_complex.tex:23`, `TRI/mamba3_siso_combined.py:327`, `MM/mamba3.py:271` |
| `rotate_pairwise`, "Interleaved rotary", "Halved rotary" | How state elements pair for one angle | Rotation | `TRI/mamba3_mimo_rotary_step.py:46`:279:301 |
| "head", "head pattern", "MVA", "MIS", "GVA", "GIS" | A copy of the recurrence, and which parameters the heads share | Notation `h`, `g` | `M2/architecture.tex:66`:70:162:185 |
| `num_bc_heads`, `nheads_qk` | The count of `B`, `C` heads | Notation `g` | `SSM/modules/mamba3.py:74`, `TRI/mamba3_siso_combined.py:318` |
| "BCNorm", "QKNorm", `B_norm`, `C_norm` | RMS norm on the `B` and `C` projections | none | `M3/structure/methods_arch.tex:19`:20, `SSM/modules/mamba3.py:105` |
| "BC bias", `B_bias`, `C_bias` | A learnable bias on `B` and `C` for each head | none | `M3/structure/methods_arch.tex:24`:25, `SSM/modules/mamba3.py:100` |
| `D` "skip" | A direct path from `x` to the output, for each head | Plant (its readout) | `SSM/modules/mamba3.py:118`, `MM/mamba3.py:471` |
| `mimo_x`, `mimo_z`, `mimo_o`, Psi, Zeta, Phi, `W_X`, `W_Z`, `W_{O'}` | The rank up-projections and the down-projection | MIMO | `SSM/modules/mamba3.py:114`:116, `TIL/mamba3_mimo_fwd.py:111`:113, `M3/appendix/mimo.tex:11`:13:25 |
| "selective SSM", "S6", "selective scan" | The core recurrence of Mamba-1 | Family (Mamba-1), Plant | `M1/method.tex:108`:397, `M2/architecture.tex:167` |
| "exponential-Euler", "ZOH" | The discretizations of Mamba-1 and Mamba-2 | Trapezoid | `M3/structure/methods_trap.tex:64`:100 |
| "structured mask", `L`, "2-band matrix" | The mask of the parallel form | Tap, SSD path | `M3/structure/preliminaries.tex:62`, `M3/structure/methods_trap.tex:171` |
| "intra-chunk", "inter-chunk", "diagonal blocks", "low-rank blocks" | The two parts of the chunkwise computation | Chunk, SSD path | `M2/efficient.tex:101`:111, `M3/structure/methods_mimo.tex:56`, `SSM/modules/ssd_minimal.py:53`:62 |
| "state passing" | Passing the final state to the next chunk or part | SSD path (`Serial`), Window | `M2/scan.tex:114`:191, `M3/appendix/results.tex:183` |
| `seqlen_offset` | The count of tokens already seen. It selects `step` | Cursor (a similar count, for another purpose) | `SSM/ops/utils/generation.py:24`, `SSM/modules/mamba3.py:146` |
| `d_ssm` | The width under the SSM. The rest uses a gated MLP path | none | `SSM/modules/mamba2.py:46`:272 |
| `dt_rank` | The width of the `Δ` projection in Mamba-1 | none | `SSM/modules/mamba_simple.py:38`:58 |
| `d_intermediate`, `GatedMLP`, "SwiGLU" | The optional feed-forward part of a layer | Layer | `SSM/models/mixer_seq_simple.py:67`:71, `M3/structure/methods_arch.tex:10` |
| "pre-gate", "post-gate", `norm_before_gate`, `is_outproj_norm` | The place of the extra RMSNorm around the gate | Gate | `M3/appendix/results.tex:15`, `SSM/modules/mamba2.py:51`, `SSM/modules/mamba3.py:41` |
| "chunking layer" | The routing module plus the downsampler | Router, Stage | `HN/main.tex:97`, `HNM:105` |
| "dechunking layer" | The smoothing module plus the upsampler | Stage | `HN/main.tex:100`, `HNM:130` |
| "downsampler" | It keeps the vectors with `b_t = 1` | Stage, Chunk | `HNM:121`:124 |
| "upsampler" | It repeats a vector until the next boundary | Stage, Chunk | `HNM:158`:191 |
| "smoothing module" | An EMA over chunks, weighted by `P_t` | Stage | `HNM:136`:139 |
| `p_t`, "boundary probability" | The score from cosine similarity | Router, Chunk boundary | `HNM:111`:115 |
| `c_t`, "confidence score" | The scale on the upsampled vector | Router | `HNM:164`:185 |
| "ratio loss", `N` | A loss toward a target compression ratio | Router | `HNM:204`:206 |
| "dynamic chunking", "DC" | The chunking and dechunking mechanism | Router, Chunk | `HNM:97`:218 |
| "network normalization", "norm balance" | An RMSNorm at the end of each network | Layer stack | `HNM:226`:229 |
| "learning rate modulation" | A learning-rate scale for each stage | LR schedule | `HNM:253`:258 |
| "isotropic" | A model with no hierarchy, or a component in that form | Layer stack, Main stack | `HN/sections/1_introduction.tex:16`, `HNM:279` |
| "aggregator", "AttnPool", `α_{i→l}` | It pools the streams into the input of a layer | Residual stream, Multi-Gate Residuals | `MGR:148`:150:154:155 |
| "independent", "competitive", `gate_variant` | The sigmoid form and the softmax form of the gate | β gate | `MGR:172`:177:577 |
| "gate bias", "forget gate score", `b_{l,0}` | The bias of each stream, and the extra softmax entry | β gate | `MGR:189`:235:275 |
| "direct concatenation" | The first `n` layers append streams and do not gate | Residual stream | `MGR:171` |
| "lerp" | The update of `s_i` | β gate | `MGR:193`:194 |

## 5. Notation table

### 5.1 Errors

Each row names one row of the Notation table in `BM/CLAUDE.md`.

| Row | Column | Table says | Source shows | Citation |
|---|---|---|---|---|
| `m` | Paper | `M` | **Wrong.** The Mamba-3 paper uses `R` everywhere. No symbol `M` exists in its TeX. In Mamba-2, `M` is the sequence transformation matrix. | `M3/structure/methods_mimo.tex:25`:31:67, `M3/appendix/mimo.tex:5`, `M2/glossary.tex:22` |
| `l` | Paper | `Q` | **Mamba-2 only.** The Mamba-3 paper uses `C`. | `M2/efficient.tex:40`:105, `M3/structure/methods_mimo.tex:53`:56 |
| `b` | Paper | `—` | **Incomplete.** Mamba-1 and Mamba-2 use `B`. A code comment also says "(B, T, Q, D, P) in the paper". The Mamba-3 TeX has no batch symbol. | `M1/method.tex:87`, `M2/efficient.tex:173`, `SSM/modules/ssd_minimal.py:86` |
| `s`, `t` | Paper | `T` | **Incomplete.** Mamba-2 and Mamba-3 use `T`. Mamba-1 uses `L`, and so does a Mamba-3 appendix. No source separates `s` from `t`. | `M2/glossary.tex:15`, `M3/structure/preliminaries.tex:6`, `M1/method.tex:87`, `M3/appendix/results.tex:179` |
| `i` | Paper | `E·D` | **Mamba-1 only.** Mamba-2 writes `e` and `d` in one section. `E` is not found in the Mamba-3 TeX as an expansion factor. mamba3-minimal comments use `E`. | `M1/method.tex:227`, `M2/systems.tex:15`, `MM/mamba3.py:65` |
| `h` | Paper | `H` | **Mamba-2 only.** Its formula is `H = D/P`, with `D` the model dimension. The code uses `d_inner // headdim`. The Mamba-3 TeX has no symbol for the count of heads. | `M2/architecture.tex:134`, `SSM/modules/mamba3.py:73` |
| `g` | Python | `ngroups` | **Incomplete.** Mamba-3 keeps the argument `ngroups` and stores `num_bc_heads`. Its kernels use `nheads_qk`. The Mamba-3 TeX has no `G`. | `SSM/modules/mamba3.py:33`:74, `TRI/mamba3_siso_combined.py:318` |
| `p` | Python | `headdim` | **Incomplete.** The modules use `headdim`. The Triton kernels use `headdim_v`. In `mamba3_mimo_rotary_step.py`, `headdim` is the state dimension. | `SSM/modules/mamba2.py:45`, `TRI/mamba3_siso_combined.py:320`, `TRI/mamba3_mimo_rotary_step.py:29`:168 |
| `r` | Python | `d_state` | **Incomplete.** The modules use `d_state`. Kernels use `dstate` and `headdim_qk`. | `SSM/modules/mamba3.py:30`, `SSM/ops/triton/ssd_combined.py:99`, `TRI/mamba3_siso_combined.py:318` |
| `a` | Python | `num_rope_angles` | **Incomplete.** The module uses this name. Kernels use `headdim_angles` and `headangles`. | `SSM/modules/mamba3.py:82`, `TRI/mamba3_siso_combined.py:327`, `TIL/mamba3_mimo.py:262` |
| `v` | Dimension | `d_inner`+2·`ngroups`·`state_rank` | **Default case only.** `SSM/modules/mamba2.py` uses `d_ssm`, which equals `d_inner` when the `d_ssm` argument is `None`. | `SSM/modules/mamba2.py:81`:104, `SSM/modules/mamba2_simple.py:73` |

### 5.2 Rows with correct columns

| Row | Result | Citation |
|---|---|---|
| `u` | `—` and `—` are correct: "micro" and "sub-step" are not found. | see 2, "Terms with no counterpart" |
| `d` | `D` and `d_model` are correct. | `M2/glossary.tex:17`, `M3/structure/preliminaries.tex:6`, `SSM/modules/mamba3.py:29` |
| `n` | `—` and `nchunks` are correct. The papers write `T/Q` or `T/C`. | `M2/efficient.tex:39`, `M3/structure/methods_mimo.tex:56`, `SSM/ops/triton/ssd_combined.py:264` |
| `k` | `—` and `d_conv` are correct. | `SSM/modules/mamba_simple.py:36`, `SSM/modules/mamba2.py:42` |

The Paper column of `p` (`P`), `r` (`N`) and `g` (`G`) is correct
(`M3/structure/methods_mimo.tex:20`, `M2/glossary.tex:18`,
`M2/architecture.tex:181`). The Python column of `b`, `s`, `t`, `i`, `h`, `m` and
`l` is correct (`SSM/modules/mamba3.py:140`:71:73:43:60).

### 5.3 Letters with another meaning in a source

The columns above can be correct, and a letter can still mean something else in a
source.

| Our letter | Source use | Citation |
|---|---|---|
| `r` state rank | `r` is the MIMO rank in the official and the minimal einops | `SSM/modules/mamba3.py:164`, `MM/mamba3.py:71`:385 |
| `n` nchunks | `n` is the state dimension in einops. `N` is the state in papers and in Tilelang, the count of heads in the rotary step, and the count of streams in MGR | `SSM/modules/mamba3.py:164`, `TIL/mamba3_mimo_fwd.py:43`, `TRI/mamba3_mimo_rotary_step.py:240`, `MGR:595` |
| `l` chunk length | `l` is the time axis at module level (`b l h p`). It is a position in a chunk only inside SSD (`b c l h p`). `L` is a mask matrix or a sequence length | `SSM/modules/mamba3.py:162`, `SSM/modules/ssd_minimal.py:48`, `M2/glossary.tex:32`, `M1/method.tex:87`, `HNM:52` |
| `s` write-axis length | `s` is the state in `"b (r g s)"`. `S` is the sequence in Tilelang, the angles in the rotary step, the source axis in Mamba-2, and the count of stages in H-Net | `SSM/modules/mamba3.py:260`, `TIL/mamba3_mimo_fwd.py:40`, `TRI/mamba3_mimo_rotary_step.py:242`, `M2/glossary.tex:16`, `HNM:22` |
| `u` micro-steps | `u` is the input tensor. `u_t` is the projection behind `λ_t` | `SSM/modules/mamba3.py:135`, `SSM/modules/mamba2.py:154`, `MM/mamba3.py:313`, `M2/systems.tex:15`, `M3/appendix/discretization.tex:154` |
| `h` nheads | `h` is the state in every Mamba paper (our State also says so: `BM/GLOSSARY.md:19`). `H` is the state matrix. `h` is the whole cache in mamba3-minimal. `h_l` is a layer input in MGR | `M3/structure/preliminaries.tex:18`, `M2/ssm.tex:247`, `M3/appendix/mimo.tex:18`, `MM/mamba3.py:154`, `MGR:155` |
| `k` conv kernel | `K` is `B` in the kernels. `k_state` holds `B`. `k_t` is a key in the H-Net router | `SSM/modules/mamba3.py:187`:298, `HNM:115` |
| `v` conv dim | `V` is `x` in the kernels. `v_state` holds `x`. `v_t` is the state-input in the Mamba-3 paper | `SSM/modules/mamba3.py:188`:299, `M3/structure/methods_trap.tex:156` |
| `d` d_model | `d` is an inner width in Mamba-1 einops. `D` is the state dimension in the rotary step | `SSM/modules/mamba_simple.py:137`:182, `TRI/mamba3_mimo_rotary_step.py:240` |
| `m` MIMO rank | `M` is a matrix in Mamba-2 and the main network in H-Net | `M2/glossary.tex:22`, `HNM:17` |
| `g` ngroups | `G` is both the group count and the Gram matrix in Mamba-2 | `M2/architecture.tex:181`, `M2/glossary.tex:31` |
| `b` batch | `B` is both the batch and the input projection. `b_t` is a boundary in H-Net. `b` is a gate bias in MGR | `M1/method.tex:87`:116, `HNM:111`, `MGR:184` |
| Paper `Q` for chunk length | `Q` is the query in Mamba-2. `Q` is `C` in the kernels | `M2/glossary.tex:28`, `SSM/modules/mamba3.py:186` |
| Paper `C` (Mamba-3 chunk length) | `C` is also the output projection in the same paper. `c` is the chunk index in einops, which is our `n` | `M3/structure/methods_mimo.tex:31`:53, `SSM/modules/ssd_minimal.py:48` |
| `β` in Tap (burn-mamba) and in β gate (burn-stack) | The two glossaries use one letter for two sources: the Mamba-3 `β_t` and the MGR `β_i` | `M3/structure/methods_trap.tex:138`, `MGR:193`, `BM/GLOSSARY.md:81`, `BS/GLOSSARY.md:192` |
| Paper `A`, `B` | Mamba-2 `A` and `B` are discrete. Mamba-3 `A_t` and `B_t` are continuous | `M2/glossary.tex:23`, `M3/structure/preliminaries.tex:85` |

## 6. `_Avoid_` entries that a source uses as its own term

| `_Avoid_` entry | Our term (glossary) | Source term and meaning | Citation |
|---|---|---|---|
| hidden state | State (burn-mamba) | `h` of the SSM in the papers. `hidden_states` is a `d_model` tensor in the code. `h_l` is a layer input in MGR | `M3/structure/preliminaries.tex:18`, `SSM/modules/block.py:43`, `MGR:155` |
| latent state | State (burn-mamba) | `h` of the SSM | `M1/background.tex:27`, `M2/background.tex:11` |
| SSM state | State (burn-mamba) | `h` of the SSM. `ssm_state` in the code | `M2/related.tex:30`, `SSM/modules/mamba3.py:426` |
| recurrent state | State (burn-mamba) | `h` of the SSM | `M2/scan.tex:187`, `M3/structure/related.tex:16` |
| state (for the accumulator) | Accumulator (burn-mamba) | `acc_ssm_states`, `Final_SSM_State`: the running sum of the official kernel | `TRI/mamba3_siso_fwd.py:336`:431 |
| SSM core, core block, mixer core | Plant, Block | Literal forms: not found. Near forms: "core SSD layer" (plant), "the core mixer" (block), "core recurrence" (plant) | `M2/architecture.tex:26`, `MM/mamba3.py:241`, `M3/structure/methods_trap.tex:160` |
| system (alone) | Plant (burn-mamba) | "continuous state space system": the continuous plant | `M3/structure/methods_trap.tex:4` |
| algorithm | SSD path, Pathway (burn-mamba) | "SSD algorithm", "chunked algorithm": the chunkwise computation | `M2/efficient.tex:9`, `M3/structure/methods_mimo.tex:53` |
| mode | SSD path, Pathway (burn-mamba) | "state-passing mode", "block decomposition mode": ways to compute the matrix product | `M2/scan.tex:114`:116 |
| variant | SSD path, Pathway (burn-mamba), Family (burn-stack) | "MIMO variant": an option. "Mamba variants": families and options | `M3/structure/methods_arch.tex:15`, `M3/appendix/results.tex:46` |
| path | Pathway (burn-mamba) | `use_mem_eff_path`, `use_fast_path`: kernel choices. "forward (prefill) path": a call kind | `SSM/modules/mamba2.py:60`, `SSM/modules/mamba_simple.py:46`, `M3/structure/experiments.tex:139` |
| step (alone) | Micro-step (burn-mamba), Step (burn-stack) | "step size": `Δ`. "recurrent steps": a recurrence step | `M3/structure/preliminaries.tex:20`, `M3/structure/related.tex:16` |
| sequence axis | Write axis (burn-mamba) | "target sequence axis": `T`. "Source sequence axis": `S` | `M2/glossary.tex:15`:16 |
| weight, coefficient | Mass (burn-mamba) | "weights β, γ". "Trapezoidal coefficients", "left-endpoint coefficient" | `M3/structure/methods_trap.tex:200`, `MM/mamba3.py:369`:547 |
| RoPE | Rotation (burn-mamba), Rotary position embedding (burn-mamba) | The word covers both, and the trick | `M3/structure/experiments.tex:54`, `M3/tables/retrieval.tex:6` |
| positional encoding | Rotation (burn-mamba) | "positional embeddings": the encodings of transformers | `M3/tables/retrieval.tex:6`, `M2/background.tex:215` |
| complex transition | Rotation (burn-mamba) | "complex transitions", "complex-valued state-transition matrices" | `M3/structure/experiments.tex:8`, `M3/structure/methods_complex.tex:17` |
| vanilla RoPE, standard RoPE | Rotary position embedding (burn-mamba) | The fixed-frequency encoding of RoFormer | `M3/structure/methods_complex.tex:73`:96, `M3/structure/experiments.tex:54`, `MM/mamba3.py:666` |
| RoPE (as a name for the rotation) | RoPE trick (burn-mamba) | "Mamba-3 without RoPE": no rotation | `M3/structure/experiments.tex:54`, `M3/tables/ablations.tex:39` |
| RoPE fraction | Rotation fraction (burn-mamba) | `rope_fraction` | `SSM/modules/mamba3.py:36` |
| initial state | Origin (burn-mamba) | The start of a chunk (paper). `initial_states`: an SSD argument (code) | `M2/efficient.tex:108`, `SSM/ops/triton/ssd_combined.py:636` |
| boundary (alone) | Chunk boundary, Cut (burn-stack) | H-Net: the chunk boundary. mamba3-minimal: a chunk boundary. Mamba-1: the end of a context | `HNM:108`:111, `MM/README.md:63`, `M1/method.tex:320` |
| mixer | Block, Family, β gate (burn-stack) | Official and minimal: our block. "recurrent mixers": our blocks. MGR: the stream update | `SSM/modules/block.py:30`, `MM/mamba3.py:136`, `M3/appendix/results.tex:186`, `MGR:148`:172 |
| version | Family (burn-stack) | "Early versions of SSMs": older models | `M3/structure/related.tex:20` |
| KV cache | Cache (burn-stack) | The cache of attention layers | `M2/architecture.tex:159`, `HNM:279` |
| memory | Cache (burn-stack) | `key_value_memory_dict`: the cache of all layers. "its memory (SSM states)": the state | `SSM/ops/utils/generation.py:26`, `M2/experiments.tex:173` |
| graph (alone), CUDA graph | Captured step, Recording (burn-stack) | "CUDA graph decoding", `graph`, `CUDAGraph`: captured decode calls | `SSM/ops/utils/generation.py:310`:372 |
| level | Stage (burn-stack) | "multi-level", "levels of abstraction": the depth of the hierarchy | `HNM:7`:19, `HN/sections/0_abstract.tex:9` |
| encoder network | Encoder (burn-stack) | The macro `\encoder` expands to this term | `HN/main.tex:103`, `HNM:17` |
| decoder network | Decoder (burn-stack) | The macro `\decoder` expands to this term | `HN/main.tex:107`, `HNM:17` |
| main network | Main stack (burn-stack) | The macro `\main` expands to this term. It also covers our inner H-Net | `HN/main.tex:105`, `HNM:19`:22 |
| carry (alone) | Straight-through carry (burn-stack) | "carry behavior": the Highway Networks carry path | `MGR:197` |

Entries with a near form only:

- `per-layer parameter`: not found. Near form: "layer-specific learnable vector"
  (`MGR:159`).
- `inner network`: not found. Near forms: "inner stages" (`HNM:256`), "outer
  networks" (`HNM:89`).
- `CLS`: one mention, for the pooling of another model (eByte), not as a source
  term (`HN/sections/A_background.tex:127`).
- `episode`: one mention, "episode boundaries in reinforcement learning"
  (`M1/method.tex:322`).

Entries not found in the approved sources:

- `folded axis` ("folded" exists in another sense: `M3/structure/preliminaries.tex:85`).
- `sub-step`, `inner step`, `horizontal tap`, `interior tap`, `near tap`,
  `vertical tap`, `far tap`.
- `RoPE kind`, `side system`, `Kalman gate`, `max-plus gate`, `DeltaProduct`,
  `initial cache`.
- `weight set`, `unshared parameter`, `optimizer step`, `special token`, `register
  token`, `start token`, `start row`, `restart`, `lane`, `depth-stream`, `truncated
  BPTT`, `ghost row`, `fused weight`, `column block`.
- `generation` in the sense of a family. In the sources, it means decode
  (`SSM/ops/utils/generation.py:246`).
- `forward` and `backward` as directions. In the sources, they mean the two
  autodiff passes (`TRI/mamba3_siso_combined.py:54`:153).
