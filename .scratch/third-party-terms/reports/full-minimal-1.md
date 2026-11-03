Batch `minimal-1`: all 67 citations are done. 67 confirmed, 0 wrong, 0 not found. Rows with a correction: L42, L44, L73. Not finished: none.

Delivery and process notes:
- The report is in this message. No report file exists, because the harness refused the Write call.
- My check command created three dot-named scratch files in `/tmp/source-terms-check/`. This broke limit 1 of the instructions. I moved them to `/tmp/minimal1-scratch/` and deleted nothing. They hold no findings.
- `dcg` blocked a heredoc check, so I did not retry it in that form. I compared the row and citation pairs by eye with a print of the batch file, and they match.
- `ste-lint` did not check the prose. Its only run had empty input.

```text
L38 | `MM/mamba3.py:136` | confirmed | 136: mixer=Mamba3(args, device=device),
L38 | `MM/mamba3.py:177` | confirmed | 177: y, h[i] = layer.mixer(layer.mixer_norm(x), h[i])
L38 | `MM/mamba3.py:241` | confirmed | 241: # Mamba-3 SSM Block (the core mixer)
L38 | `MM/mamba3.py:246` | confirmed | 246: Mamba-3 SSM block implementing:
L39 | `MM/mamba3.py:94` | confirmed | 94: ssm_state: Tensor # ... SSM hidden state h_t
L39 | `MM/mamba3.py:154` | confirmed | 154: the model `forward` takes `h: list[InferenceCache]`, also `list[None]` or `None`.
L39 | `MM/mamba3.py:159` | confirmed | 159: h: hidden states for inference step.
L39 | `MM/mamba3.py:177` | confirmed | 177: y, h[i] = layer.mixer(layer.mixer_norm(x), h[i])
L39 | `MM/mamba3.py:317` | confirmed | 317: h: hidden state for inference. Line 313 types it as `InferenceCache`.
L39 | `MM/mamba3.py:501` | confirmed | 501: h: current inference cache (ssm_state, prev_Bx, cum_angle)
L40 | `MM/mamba3.py:94` | confirmed | 94: ssm_state: Tensor # ... SSM hidden state h_t
L42 | `MM/mamba3.py:63` | confirmed | 63: Number of Mamba-3 layers (each = SSM mixer + SwiGLU MLP)
L42 | `MM/mamba3.py:130` | confirmed | 130: layers=nn.ModuleList( . The literal `layers[i]` is absent: see Row corrections.
L42 | `MM/mamba3.py:175` | confirmed | 175: for i, layer in enumerate(self.backbone.layers): . The literal `layers[i]` is absent: see Row corrections.
L42 | `MM/mamba3.py:181` | confirmed | 181: x = x + layer.mlp(layer.mlp_norm(x))
L43 | `MM/mamba3.py:21` | confirmed | 21-22: alternating Mamba-3 SSM blocks and SwiGLU MLP / blocks
L43 | `MM/mamba3.py:855` | confirmed | 855: Pre-normalization before each block
L44 | `MM/mamba3.py:127` | confirmed | 127: self.backbone = nn.ModuleDict( . The row text is incomplete: see Row corrections.
L44 | `MM/mamba3.py:175` | confirmed | 175: for i, layer in enumerate(self.backbone.layers):
L46 | `MM/mamba3.py:15` | confirmed | 15: Complex-valued SSM / Data-Dependent RoPE
L46 | `MM/mamba3.py:358` | confirmed | 358: Data-Dependent RoPE angles (shared between SISO and MIMO)
L46 | `MM/mamba3.py:661` | confirmed | 661: def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
L46 | `MM/mamba3.py:662` | confirmed | 662: Apply rotary position embedding with data-dependent angles.
L46 | `MM/mamba3.py:666` | confirmed | 666-667: unlike standard RoPE which uses fixed / frequency schedules
L47 | `MM/README.md:21` | confirmed | 21: **Complex SSM / RoPE**
L49 | `MM/mamba3.py:369` | confirmed | 369: Trapezoidal coefficients (Proposition 1, Eq. 4)
L49 | `MM/mamba3.py:375` | confirmed | 375: beta = (1 - lam) * dt * alpha
L49 | `MM/mamba3.py:376` | confirmed | 376: gamma = lam * dt
L49 | `MM/mamba3.py:547` | confirmed | 547: beta = (1 - lam) * dt * alpha # left-endpoint coefficient
L49 | `MM/mamba3.py:548` | confirmed | 548: gamma = lam * dt # right-endpoint coefficient
L51 | `MM/mamba3.py:349` | confirmed | 349: dt controls both forget-gate (α) and input-gate (γ)
L52 | `MM/mamba3.py:159` | confirmed | 159-161: If present, the constant-time ... inference path is taken. input_ids should have shape (batch, 1).
L52 | `MM/mamba3.py:323` | confirmed | 323-324: if h is not None: return self.step(u, h)
L54 | `MM/mamba3.py:70` | confirmed | 70: use_mimo: bool = False # Toggle MIMO variant
L54 | `MM/mamba3.py:444` | confirmed | 444: # ── SISO path ── . Line 443 is the `else:` of line 381.
L54 | `MM/mamba3.py:719` | confirmed | 719: def ssd(x, A, B, C, chunk_size, ...
L54 | `MM/mamba3.py:785` | confirmed | 785: def ssd_mimo(x, A, B, C, chunk_size, ...
L59 | `MM/mamba3.py:150` | confirmed | 150: # Weight tying (standard practice)
L59 | `MM/mamba3.py:151` | confirmed | 151: self.lm_head.weight = self.backbone.embedding.weight
L72 | `MM/mamba3.py:270` | confirmed | 270: trapezoidal interpolation parameter (one per head)
L72 | `MM/mamba3.py:375` | confirmed | 375: beta = (1 - lam) * dt * alpha . Line 376 gives the current share: gamma = lam * dt.
L73 | `MM/mamba3.py:271` | confirmed | 271: θ: d_state // 2 — rotation angles. The row misses `theta`: see Row corrections.
L78 | `MM/mamba3.py:471` | confirmed | 471: Skip connection: y = y + D * x. Line 283: skip connection coefficient (one per head).
L97 | `MM/mamba3.py:95` | confirmed | 95: prev_Bx: Tensor # ... B̄_{t-1} ⊗ x_{t-1} for β term
L97 | `MM/mamba3.py:371` | confirmed | 371: β_t = ... — left-endpoint (previous input)
L97 | `MM/mamba3.py:411` | confirmed | 411: B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
L97 | `MM/mamba3.py:459` | confirmed | 459: B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0)) # shifted right. Line 460 holds `x_prev`.
L97 | `MM/README.md:70` | confirmed | 70: **β SSD** (Previous-timestep): `ssd(x_prev * β, dA, B_prev, C)`
L105 | `MM/README.md:61` | confirmed | 61: ## Two-SSD Decomposition
L105 | `MM/README.md:69` | confirmed | 69: **γ SSD** (Current-timestep)
L105 | `MM/README.md:70` | confirmed | 70: **β SSD** (Previous-timestep)
L105 | `MM/mamba3.py:404` | confirmed | 404: Trapezoidal SSD (two-SSD decomposition, MIMO variant)
L105 | `MM/mamba3.py:453` | confirmed | 453: Trapezoidal SSD (two-SSD decomposition)
L105 | `MM/mamba3.py:723` | confirmed | 723: handled externally via a two-SSD decomposition
L106 | `MM/mamba3.py:693` | confirmed | 693: Structured State Space Duality (SSD) — Core Algorithm
L106 | `MM/mamba3.py:719` | confirmed | 719: def ssd(x, A, B, C, chunk_size, ...
L106 | `MM/mamba3.py:720` | confirmed | 720: Structured State Space Duality (SSD) — the chunked parallel algorithm.
L111 | `MM/mamba3.py:14` | confirmed | 14: Trapezoidal Discretization (Proposition 1, Eq. 3-5)
L111 | `MM/README.md:20` | confirmed | 20: **Trapezoidal Discretization**
L113 | `MM/mamba3.py:269` | confirmed | 269: dt: nheads — step size Δ (one per head)
L114 | `MM/mamba3.py:370` | confirmed | 370: α_t = exp(Δ_t * A_t) — decay
L114 | `MM/mamba3.py:373` | confirmed | 373: dA = dt * rearrange(A, "h -> 1 1 h"). Line 374: alpha = torch.exp(dA).
L117 | `MM/mamba3.py:199` | confirmed | 199: # ── Process prompt ──
L117 | `MM/mamba3.py:203` | confirmed | 203: n_chunked = (prefix.shape[0] // chunk_size) * chunk_size. Lines 200-202 state the bulk and the tail.
L117 | `MM/mamba3.py:211` | confirmed | 211: for i in range(n_chunked, prefix.shape[0]): . Line 212 passes one token.
L159 | `MM/mamba3.py:313` | confirmed | 313: def forward(self, u: Tensor, ... Line 316: u: (batch, seqlen, d_model) — input.
L161 | `MM/mamba3.py:154` | confirmed | 154: the model `forward` takes `h: list[InferenceCache]`, also `list[None]` or `None`.
```

## Row corrections

### L42

The literal `layers[i]` is in no source. A search of mamba3-minimal, the official checkout and the papers found no match. The row also gives "Mamba-3 layer" only the paper meaning (our block). In mamba3-minimal the same word is one mixer and one MLP (our layer).

Cell: Source word
Old: `layers[i]`
New: `self.backbone.layers`
Proof: `MM/mamba3.py:175`: "for i, layer in enumerate(self.backbone.layers):"

Cell: Its meaning in the source
Old: mamba3-minimal: `layers[i]` is "SSM mixer + SwiGLU MLP".
New: mamba3-minimal: one entry of `self.backbone.layers` is "SSM mixer + SwiGLU MLP". "Mamba-3 layers" names these entries, not our block.
Proof: `MM/mamba3.py:63`: "Number of Mamba-3 layers (each = SSM mixer + SwiGLU MLP)"

Cell: Our term
Old: **Layer** for `self.layers` and `layers[i]`.
New: **Layer** for `self.layers`, for `self.backbone.layers` and for "Mamba-3 layers" of mamba3-minimal.
Proof: `MM/mamba3.py:63`, `MM/mamba3.py:175`.

### L44

`backbone` also holds the embedding and the final norm. The current text hides that the wrong port of this row (a second embedding) also applies to mamba3-minimal.

Cell: Its meaning in the source
Old: mamba3-minimal: `backbone` holds the layers.
New: mamba3-minimal: `backbone` holds the embedding, the layers and a final norm.
Proof: `MM/mamba3.py:129`: "embedding=nn.Embedding(". `MM/mamba3.py:130`: "layers=nn.ModuleList(". `MM/mamba3.py:144`: "norm_f=RMSNorm(".

Cell: Citation
Old: `MM/mamba3.py:127`, `MM/mamba3.py:175`
New: `MM/mamba3.py:127`, `MM/mamba3.py:129`, `MM/mamba3.py:144`, `MM/mamba3.py:175`
Proof: the lines above.

### L73

The row lists `angles` as the projected rate and cites mamba3-minimal. In that source the identifier of the rate is `theta`, which the row does not list. The only `angles` identifier there is the argument of `apply_rope`, and it is the cumulative angle. A reader who ports from this source gets a wrong fact.

Cell: Source word
Old: `θ_t`, `angles`, `Angles`, `angle_proj`, "Rotary angle rates"
New: `θ_t`, `theta`, "rotation angles", `angles`, `Angles`, `angle_proj`, "Rotary angle rates"
Proof: `MM/mamba3.py:334`: "z, x, B, C, dt, lam, theta = torch.split(". `MM/mamba3.py:343`: "θ: rotation angles".

Cell: Its meaning in the source
Old: The projected rate of the rotation.
New: The projected rate of the rotation. mamba3-minimal: `theta` and its comment "rotation angles" name this rate. The argument `angles` of `apply_rope` is the cumulative angle, not this rate.
Proof: `MM/mamba3.py:661`: "def apply_rope(x: Tensor, angles: Tensor)". `MM/mamba3.py:671`: "angles: (..., d_state // 2) — cumulative rotation angles". `MM/mamba3.py:450`: "B = apply_rope(B, cum_angles)".

Cell: Citation
Old: `MM/mamba3.py:271`
New: `MM/mamba3.py:271`, `MM/mamba3.py:334`, `MM/mamba3.py:671`
Proof: the lines above.

## Extra findings

These items show no fault. I checked them by reading the source. I did not run the code.

- L40, "`ssm_state` is the state". The citation is only a comment, and the code agrees. `MM/mamba3.py:469`: "ssm_state = state_gamma + state_beta", and this sum holds the γ term. `MM/mamba3.py:621` reads the output directly from `new_ssm_state`. `MM/mamba3.py:761` decays each sample to the end of its chunk, so the forward sum expands to the recurrence at `MM/mamba3.py:615`.
- L54, `use_mimo=False`. The literal form is only at `MM/tests/test_mimo.py:55`. The citation `MM/mamba3.py:70` holds `use_mimo: bool = False`, so a reader finds the field. The step has the same split at `MM/mamba3.py:553` and `MM/mamba3.py:607`.
- L72, "trapezoidal parameter". The row shows it through `M3/structure/methods_trap.tex:146`. mamba3-minimal also has it at `MM/mamba3.py:342`.
- L97, "Each source has one tap only, at `t−1`". True for mamba3-minimal. The inputs shift by one sample at `MM/mamba3.py:411`, `:412`, `:459` and `:460`. The cache holds one product at `MM/mamba3.py:95`.
- L105, "The only form of mamba3-minimal" and "one β SSD". True. The search found SSD calls only at `MM/mamba3.py:405`, `:414`, `:454` and `:462`, which is one γ SSD and one β SSD in each forward branch. The step runs the recurrence with no SSD, at `MM/mamba3.py:577` and `:615`.
- L106, "in the form of our `Minimal`". True. `MM/mamba3.py:754` to `:780` run the same four steps as the header of `/shared/claude/burn-mamba/src/mamba2/ssd/minimal.rs`. Both use `segsum` for the decay inside a chunk (`MM/mamba3.py:756`). I read that header and a search of the file, not its whole body.
- L159, `u_t`. The row shows it through the paper. mamba3-minimal also has it at `MM/mamba3.py:351`: "λ_t = σ(u_t)".
- L161, "`h` is the whole cache in mamba3-minimal". True for the argument `h`. It is a list at the model (`MM/mamba3.py:154`) and one `InferenceCache` at the block (`MM/mamba3.py:313`). The letter has two more uses in this source. `h_t` is the state (`MM/mamba3.py:94`, `:496`). `h` in an einops pattern is `nheads` (`MM/mamba3.py:373`, `:379`). Optional text to append: "`h_t` is the state there, and `h` in an einops pattern is `nheads`."