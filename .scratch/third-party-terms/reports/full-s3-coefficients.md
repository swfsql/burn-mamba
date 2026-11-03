Batch `s3-coefficients`: 41 confirmed, 2 wrong, 0 not found. All 6 rows have a correction: L137, L138, L139, L140, L141, L142. Every row is finished. I did not write a report file, as the coordinator instructed. The report follows.

```text
L137 | `M2/glossary.tex:24` | wrong | 24: "State space model input projection (expansion) matrix". No word on the step size. The proof is at M2/background.tex:61: "``continuous parameters'' $(\dt, \mathring{A}, \mathring{B})$ are converted to ``discrete parameters'' $(A, B)$".
L137 | `M3/structure/preliminaries.tex:85` | confirmed | 85: "in Mamba-2, $\bB_t$ represents the discretized parameter which is equivalent to $\gamma_t \bB_t$".
L137 | `TRI/mamba3_siso_fwd.py:286` | confirmed | 286: gamma = dt * trap
L137 | `MM/mamba3.py:376` | confirmed | 376: gamma = lam * dt
L138 | `M2/glossary.tex:23` | confirmed | 23: "Discrete SSM recurrent (state) matrix"
L138 | `M2/glossary.tex:24` | wrong | 24: no word "discrete" for `B`. The proof is at M2/background.tex:61: "``discrete parameters'' $(A, B)$".
L138 | `M3/structure/preliminaries.tex:85` | confirmed | 85: "$\bB_t$ represents the continuous parameter". It covers `B` only. For `A`, see the row correction.
L138 | `SSM/modules/mamba3.py:172` | confirmed | 172: ADT = _A * DT
L138 | `MM/mamba3.py:373` | confirmed | 373: dA = dt * rearrange(A, "h -> 1 1 h")
L139 | `M3/structure/methods_trap.tex:138` | confirmed | 138: "$\lambda_t \in [0,1]$ is a data-dependent scalar"
L139 | `M3/structure/methods_trap.tex:146` | confirmed | 146: "the trapezoidal parameter satisfies"
L139 | `M3/appendix/discretization.tex:153` | confirmed | 153: "a data-dependent gate"
L139 | `SSM/modules/mamba3.py:166` | confirmed | 166: trap = rearrange(trap, "b l h -> b h l"). No sigmoid. The kernels apply it, see the row correction.
L139 | `SSM/modules/mamba3.py:257` | confirmed | 257: trap = torch.sigmoid(trap_proj)
L139 | `TRI/mamba3_siso_combined.py:323` | confirmed | 323: "Trap: Trapezoidal factor"
L139 | `TRI/mamba3_siso_combined.py:324` | confirmed | 324: "Mixing factor in [0, 1] for trapezoidal discretization."
L139 | `TIL/mamba3_mimo_fwd.py:114` | confirmed | 114: "Trap: convex-combination modulator used in exponential-trapezoidal discretization."
L139 | `MM/mamba3.py:270` | confirmed | 270: "trapezoidal interpolation parameter"
L139 | `MM/mamba3.py:375` | confirmed | 375: beta = (1 - lam) * dt * alpha
L140 | `M3/structure/methods_trap.tex:138` | confirmed | 138: "$\beta_t \coloneqq (1-\lambda_t)\Delta_t e^{\Delta_t A_t}$"
L140 | `TRI/mamba3_siso_fwd.py:285` | confirmed | 285: shifted_gamma = dt_shifted * (1 - trap_shifted). This is the forward only, see the row correction.
L140 | `TRI/mamba3_siso_fwd.py:287` | confirmed | 287: scale = shifted_gamma + gamma
L140 | `MM/mamba3.py:375` | confirmed | 375: beta = (1 - lam) * dt * alpha
L140 | `MM/mamba3.py:547` | confirmed | 547: beta = (1 - lam) * dt * alpha
L141 | `M3/structure/methods_arch.tex:10` | confirmed | 10: "alternating Mamba-3 and SwiGLU blocks with pre-norm"
L141 | `SSM/modules/block.py:21` | confirmed | 21: "Here we have: Add -> LN -> Mixer"
L141 | `SSM/modules/block.py:52` | confirmed | 52: residual = (hidden_states + residual) if residual is not None else hidden_states
L141 | `SSM/modules/block.py:88` | confirmed | 88: return hidden_states, residual
L141 | `SSM/models/mixer_seq_simple.py:197` | confirmed | 197: if not self.fused_add_norm: starts the last add. The add is on 198, inside MixerModel.forward (def at 190).
L141 | `SSM/models/mixer_seq_simple.py:53` | confirmed | 53: ssm_layer = ssm_cfg.pop("layer", "Mamba1")
L141 | `SSM/models/mixer_seq_simple.py:54` | confirmed | 54: if ssm_layer not in ["Mamba1", "Mamba2"]:
L141 | `SSR:106` | confirmed | 106: "The Mamba-3 block is implemented at modules/mamba3.py". The usage at README.md:113 calls Mamba3 directly.
L141 | `MM/mamba3.py:175` | confirmed | 175: for i, layer in enumerate(self.backbone.layers):
L141 | `MM/mamba3.py:178` | confirmed | 178: x = y + x  # Residual connection
L141 | `MM/mamba3.py:181` | confirmed | 181: x = x + layer.mlp(layer.mlp_norm(x))
L142 | `M3/structure/methods_arch.tex:10` | confirmed | 10: "alternating Mamba-3 and SwiGLU blocks"
L142 | `SSM/modules/block.py:32` | confirmed | 32: self.norm2 = norm_cls(dim)
L142 | `SSM/modules/block.py:33` | confirmed | 33: self.mlp = mlp_cls(dim)
L142 | `SSM/models/mixer_seq_simple.py:67` | confirmed | 67: if d_intermediate == 0:
L142 | `SSM/models/mixer_seq_simple.py:71` | confirmed | 71: GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
L142 | `MM/mamba3.py:63` | confirmed | 63: "each = SSM mixer + SwiGLU MLP"
L142 | `MM/mamba3.py:130` | confirmed | 130: layers=nn.ModuleList( starts the statement. The feed-forward is on 138: mlp=SwiGLU(...).
L142 | `MM/mamba3.py:181` | confirmed | 181: x = x + layer.mlp(layer.mlp_norm(x))
```

## Row corrections

### L137

Cell: Papers
Old: `M2/glossary.tex:24`
New: `M2/background.tex:61`
Proof: M2/background.tex:61: "``discrete parameters'' $(A, B)$", through "$B = f_B(\dt, \mathring{B})$".

Cell: Official code
Old: Mamba-3: `gamma = dt * trap` is a separate value (`TRI/mamba3_siso_fwd.py:286`).
New: Mamba-2: `dBx` multiplies `dt`, `B` and `x`, so `B` does not hold the step size (`SSM/modules/mamba2.py:316`). Mamba-3: `gamma = dt * trap` is a separate value (`TRI/mamba3_siso_fwd.py:286`).
Proof: SSM/modules/mamba2.py:316: dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x).

Cell: Our code follows
Old: ?
New: Mamba-2: the official code, not the paper. `Mamba2SsdInput` holds `dt_bnlh` and `b_bnlhr` as separate values. The `B` of the Mamba-2 paper is our `B̄ = Δ·B`. Mamba-3: all three sources. `TrapezoidCoeffs::gamma` is a separate value, `λ·Δ`. Under `TapPattern::None`, it is `Δ`.
Proof: src/mamba2/ssd/ssd_path.rs:59, :65, :73. src/mamba2/ssd/minimal.rs:113. src/mamba2/mamba2.rs:22. src/mamba3/helpers.rs:587, :709, :728.

### L138

Cell: Papers
Old: (`M2/glossary.tex:23`, `M2/glossary.tex:24`, `M3/structure/preliminaries.tex:85`)
New: (`M2/glossary.tex:23`, `M2/background.tex:61`, `M3/structure/preliminaries.tex:85`, `M3/structure/methods_trap.tex:138`)
Proof: M2/background.tex:61 gives `B` as discrete. M3/structure/methods_trap.tex:138: "$\alpha_t \coloneqq e^{\Delta_t A_t}$" gives `A` of Mamba-3. Line 85 names only `B`.

Cell: Official code
Old: Mamba-3: `ADT = _A * DT`, so `_A` is continuous (`SSM/modules/mamba3.py:172`).
New: Mamba-2: `dA = torch.exp(dt * A)`, so `A` is continuous (`SSM/modules/mamba2.py:314`). Mamba-3: `ADT = _A * DT`, so `_A` is continuous (`SSM/modules/mamba3.py:172`).
Proof: SSM/modules/mamba2.py:314.

Cell: Our code follows
Old: ?
New: The official code. `A` and `B` are continuous in both families. Mamba-2: `Mamba2::a_log_h` gives one `A` for each head, with no data dependence. Mamba-3: `helpers::trapezoidal_coefficients` forms `A` from the `dd_A` projection, and `TrapezoidCoeffs::da` is `Δ·A`. Differences: the `A` of the Mamba-2 paper is our `Ā`. mamba3-minimal has one `A` for each head (`MM/mamba3.py:330`), but the Mamba-3 paper makes `A_t` data-dependent (`M3/structure/preliminaries.tex:44`). Under `Gain::Kalman`, the gain sets `TrapezoidCoeffs::da`.
Proof: src/mamba2/mamba2.rs:159, :165. src/mamba3/helpers.rs:569, :676, :677, :691. src/mamba3/positive/mod.rs:138. SSM/modules/mamba3.py:169.

### L139

Cell: Official code
Old: `trap` is pre-sigmoid in `forward` and post-sigmoid in `_preprocess` (`SSM/modules/mamba3.py:166`, `SSM/modules/mamba3.py:257`).
New: `trap` is pre-sigmoid in `forward` and post-sigmoid in `_preprocess` (`SSM/modules/mamba3.py:166`, `SSM/modules/mamba3.py:257`). The `forward` kernels apply the sigmoid (`TRI/mamba3_siso_fwd.py:279`, `TIL/mamba3_mimo_fwd.py:193`). So "in [0, 1]" does not describe the argument `Trap`.
Proof: SSM/modules/mamba3.py:227: Trap=trap, with no sigmoid before it. TRI/mamba3_siso_fwd.py:279: trap = sigmoid_approx(trap). SSM/ops/tilelang/mamba3/mamba3_mimo.py:256: "Trapezoidal mixing factor, pre-sigmoid".

Cell: Our code follows
Old: ?
New: All sources, with one form of the value. `lambda_raw` is pre-sigmoid in `forward` and in `step`. `helpers::trapezoidal_coefficients` applies the sigmoid and forms `TrapezoidCoeffs::nu` and `TrapezoidCoeffs::gamma`. Differences: `Mamba3::lambda_channels_total` gives one `λ` for each head and micro-step. `TapPattern::None` projects no `λ`, so `λ ≡ 1`. A closed tap sets `λ = 1` at its position.
Proof: src/mamba3/helpers.rs:627, :658, :699, :704, :708, :709, :728. src/mamba3/mamba3.rs:497. src/mamba3/tap_pattern.rs:108.

### L140

Cell: Official code
Old: `shifted_gamma = dt_shifted * (1 - trap_shifted)`, with no decay. `scale = shifted_gamma + gamma` (`TRI/mamba3_siso_fwd.py:285`, `TRI/mamba3_siso_fwd.py:287`).
New: Forward: `shifted_gamma = dt_shifted * (1 - trap_shifted)`, with no decay. `scale = shifted_gamma + gamma` (`TRI/mamba3_siso_fwd.py:285`, `TRI/mamba3_siso_fwd.py:287`, `TIL/mamba3_mimo_fwd.py:182`, `TIL/mamba3_mimo_fwd.py:196`). The decay of the next position then applies to this term, so the state at a call boundary is the same. `step`: `beta_val = (1.0 - trap_val) * dt_val * alpha_val`, with the decay (`SSM/ops/cute/mamba3/mamba3_step_fn.py:400`).
Proof: SSM/ops/cute/mamba3/mamba3_step_fn.py:400. The module calls that function at SSM/modules/mamba3.py:357. /tmp/s3_coeff_check.py gives max |h - (s - nu_shift*bx)| = 2.2e-16 and s[-1] - h[-1] = 0. The mid-sequence difference is 1.2. That is a numpy model of the two recurrences, not a run of the kernels.

Cell: Our code follows
Old: ?
New: Each form, by pathway. `TrapezoidCoeffs::nu` is `(1−λ)Δ`, with no decay. `Mamba3::forward_double_ssd` follows the paper and mamba3-minimal: it multiplies `ν` by `α` to form `β`. `Mamba3::forward_single_ssd` follows the official forward: `scaleₜ = γₜ + νₜ₊ₗₐ₉`, with no decay. `Mamba3::step_double_ssd` forms `β` with the decay for both pathways, as the official `step` does. Differences: a tap at lag > 1 also gets `helpers::interior_gap_decay`, and no source has this lag. `Mamba3::forward_single_ssd` adds a boundary-β seed from the cache. The official SISO forward has no input state (`SSM/modules/mamba3.py:234`).
Proof: src/mamba3/helpers.rs:489, :708. src/mamba3/double_ssd/double_ssd/mod.rs:324, :326, :907, :1057, :1058. src/mamba3/single_ssd/single_ssd/mod.rs:295, :301, :425, :430, :656.

### L141

Cell: Our code follows
Old: ?
New: mamba3-minimal, with the adds in another place. `Layer::forward` runs the Pre-LN and the block, and returns the delta `h₁ + h₂`. `Layers` adds the input under `Residuals::Standard`, so the output is `(x + h₁) + h₂`. The official `Add -> LN -> Mixer` order gives the same value. Differences: `Layer` returns no residual, and it wraps `Mamba3` as any other block. `Residuals::MultiGate` has no skip. `Layers::ignore_first_residual` and `Layers::ignore_last_residual` remove the outer add.
Proof: ../burn-stack/src/modules/layer.rs:13, :25, :243. ../burn-stack/src/modules/layers.rs:26, :28, :567, :582, :587. /tmp/s3_residual_check.py: official against minimal 0.0, ours against minimal 3.3e-15. That is a numpy model of the three orders, with stand-in functions for the mixer and the MLP.

### L142

Cell: Our code follows
Old: ?
New: The official code: the part is optional. `Layer::mlp` is `Option<GatedMlp>`, and `LayersBuilder::mlp` is `None` by default. `GatedMlpConfig::d_intermediate` rounds up to `GatedMlpConfig::multiple_of`, 128 by default, as in the official `GatedMLP`. Differences: `None` selects no feed-forward, not `d_intermediate == 0`. `GatedMlpConfig::from_hidden_ratio` with ratio 4 gives the width of mamba3-minimal. `GatedMlp` holds one `fc1`, as the official code does. mamba3-minimal holds `w_gate` and `w_up` (`MM/mamba3.py:648`, `MM/mamba3.py:649`).
Proof: ../burn-stack/src/modules/layer.rs:64. ../burn-stack/src/modules/layers.rs:1186, :1207. ../burn-stack/src/modules/mlp.rs:40, :42, :59, :60, :65, :73. SSM/modules/mlp.py:14, :24, :25. MM/mamba3.py:80.

## Extra findings

- L139: `TRI/mamba3_siso_step.py:187` applies a sigmoid to its `trap`. Inside `mamba_ssm/`, `mamba3_siso_step` occurs only in its own file (:27, :233, :336). `Mamba3.step` calls `mamba3_step_fn` with a post-sigmoid `trap` (SSM/modules/mamba3.py:327, :357). A port from that kernel reads a different contract. The same kernel has `beta = alpha * dt * (1 - trap)` at :190.
- L139: `sigmoid_approx` returns `tl.sigmoid(x)` (TRI/utils.py:114). So the kernels have no approximate sigmoid to port.
- L141 negative claim: the claim holds. A search of the whole official checkout for `Block\(|create_block\(` finds only `mamba_ssm/models/mixer_seq_simple.py:29`, `:73`, `:154` and `mamba_ssm/modules/block.py:10`. `create_block` raises an error for any other layer name at `SSM/models/mixer_seq_simple.py:55`.
- L142 uncited claim: "Always present" holds. mamba3-minimal has no option: `mlp` occurs at MM/mamba3.py:137, :138 and :181, with no condition. The official default is `d_intermediate: int = 0` (`SSM/models/config_mamba.py:8`), so the default official model has no feed-forward.
- This batch has no "No citation" cell.

## For the user

No difference in this batch looks like a bug. Each case below follows no Mamba source, and each one is a selectable option of our code.

- L140, tap at lag > 1. `helpers::interior_gap_decay` and `νₜ₊ₗₐ₉` (src/mamba3/helpers.rs:489, src/mamba3/double_ssd/double_ssd/mod.rs:326, src/mamba3/single_ssd/single_ssd/mod.rs:295) have no source form. The paper, the official code and mamba3-minimal read only `t−1` (M3/structure/methods_trap.tex:129, TRI/mamba3_siso_fwd.py:285, MM/mamba3.py:375). I read the code at lag 1 and it matches those forms, but I did not run our code. Confidence that no source has this lag: high, from the cited lines and the searches of this batch.
- L138, `Gain::Kalman`. The gain sets `TrapezoidCoeffs::da` (src/mamba3/helpers.rs:686, :691), so `da` is not `Δ·A` under this member. The cited sources have no such gain, but I did not search them for one. The default is `Gain::Projected` (src/mamba3/mamba3.rs:986). Confidence: medium.
- L141, `Residuals::MultiGate`. This scheme has no outer skip (../burn-stack/src/modules/layers.rs:271, :587). It does not come from a Mamba source. Its words are in `../burn-stack/docs/source-terms.md`, per the header of our doc. The default is `Residuals::Standard`. Confidence: high.