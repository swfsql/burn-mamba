Batch `modules-1` is finished. 81 citations: 78 confirmed, 3 wrong, 0 not found. Rows with a correction: L39, L40, L42, L46, L79, L88. Rows not finished: none.

I did not write `/tmp/source-terms-check/report-modules-1.md`, because the harness refused the file. The report is below.

```text
L38 | `SSM/modules/block.py:10` | confirmed | 10: class Block(nn.Module):
L38 | `SSM/modules/block.py:15` | confirmed | 15: Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection
L38 | `SSM/modules/block.py:29` | confirmed | 29: self.norm = norm_cls(dim)
L38 | `SSM/modules/block.py:30` | confirmed | 30: self.mixer = mixer_cls(dim)
L38 | `SSM/modules/block.py:32` | confirmed | 32: self.norm2 = norm_cls(dim)
L38 | `SSM/modules/block.py:33` | confirmed | 33: self.mlp = mlp_cls(dim)
L38 | `SSM/modules/block.py:67` | confirmed | 67: hidden_states = self.mixer(hidden_states, ...)
L38 | `SSM/models/mixer_seq_simple.py:29` | confirmed | 29: def create_block(
L38 | `SSM/models/mixer_seq_simple.py:56` | confirmed | 56: mixer_cls = partial( and 57: Mamba2 if ssm_layer == "Mamba2" else Mamba,
L38 | `SSM/models/mixer_seq_simple.py:63` | confirmed | 63: mixer_cls = partial(MHA, ...)
L38 | `SSM/models/mixer_seq_simple.py:73` | confirmed | 73: block = Block(
L38 | `SSR:55` | confirmed | 55: ### Mamba Block
L38 | `SSR:57` | confirmed | 57: The main module of this repository is the Mamba architecture block wrapping the selective SSM.
L38 | `SSR:59` | confirmed | 59: Source: [modules/mamba_simple.py](mamba_ssm/modules/mamba_simple.py).
L39 | `SSM/modules/block.py:43` | confirmed | 43: self, hidden_states: Tensor, residual: Optional[Tensor] = None, ...
L39 | `SSM/modules/block.py:67` | confirmed | 67: hidden_states = self.mixer(hidden_states, ...)
L39 | `SSM/modules/mamba_simple.py:119` | confirmed | 119: def forward(self, hidden_states, inference_params=None):
L39 | `SSM/modules/mamba_simple.py:121` | confirmed | 121: hidden_states: (B, L, D)
L39 | `SSM/modules/mamba2.py:156` | wrong | 156: u: (batch, seqlen, hidden_dim) if seqlen=None. The citation can stay. The row misreads the word: `hidden_dim` names the last axis of `u`, not a tensor.
L40 | `SSM/modules/mamba3.py:241` | confirmed | 241: ssm_state.copy_(last_state), in the SISO branch of forward.
L40 | `SSM/modules/mamba3.py:286` | confirmed | 286: def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs): 289: Also modify the state vars in-place for the next step.
L40 | `SSM/modules/mamba3.py:426` | confirmed | 426: # SSM State
L40 | `SSM/modules/mamba3.py:427` | confirmed | 427: ssm_state = torch.zeros( and 428: (batch_size, self.nheads, self.headdim, self.d_state),
L40 | `SSM/modules/mamba2.py:169` | wrong | 169: conv_state, ssm_state = None, None. The line has no `last_state`. The proof is at SSM/modules/mamba2.py:262: y, last_state, *rest = y
L42 | `SSR:51` | confirmed | 51: Mamba is based on a selective SSM layer, which is the focus of the paper
L42 | `SSM/models/mixer_seq_simple.py:53` | confirmed | 53: ssm_layer = ssm_cfg.pop("layer", "Mamba1")
L42 | `SSM/models/mixer_seq_simple.py:54` | confirmed | 54: if ssm_layer not in ["Mamba1", "Mamba2"]:
L42 | `SSM/models/mixer_seq_simple.py:57` | confirmed | 57: Mamba2 if ssm_layer == "Mamba2" else Mamba,
L42 | `SSM/models/mixer_seq_simple.py:152` | confirmed | 152: self.layers = nn.ModuleList( and 154: create_block(
L42 | `SSM/models/mixer_seq_simple.py:193` | confirmed | 193: for layer in self.layers:
L44 | `SSM/models/mixer_seq_simple.py:118` | confirmed | 118: class MixerModel(nn.Module):
L44 | `SSM/models/mixer_seq_simple.py:140` | confirmed | 140: self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
L44 | `SSM/models/mixer_seq_simple.py:152` | confirmed | 152: self.layers = nn.ModuleList(
L44 | `SSM/models/mixer_seq_simple.py:171` | confirmed | 171: self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
L44 | `SSM/models/mixer_seq_simple.py:241` | confirmed | 241: self.backbone = MixerModel(
L46 | `SSM/modules/mamba3.py:76` | confirmed | 76: # RoPE flags. 77 has `rope_fraction`. 78 has `self.rotary_dim_divisor`.
L46 | `SSM/modules/mamba3.py:418` | confirmed | 418: # RoPE State
L48 | `SSM/modules/mamba3.py:166` | confirmed | 166: trap = rearrange(trap, "b l h -> b h l"). forward has no sigmoid. 191 and 227: Trap=trap.
L48 | `SSM/modules/mamba3.py:257` | confirmed | 257: trap = torch.sigmoid(trap_proj), in `_preprocess` (253).
L50 | `SSM/modules/ssd_minimal.py:34` | confirmed | 34: def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
L50 | `SSM/modules/ssd_minimal.py:47` | confirmed | 47: # Rearrange into blocks/chunks
L52 | `SSM/modules/mamba3.py:146` | confirmed | 146: if inference_params.seqlen_offset > 0:
L52 | `SSM/modules/mamba3.py:147` | confirmed | 147: out, _, _, _, _ = self.step(u, angle_dt_state, ssm_state, k_state, v_state)
L52 | `SSM/modules/mamba2.py:173` | confirmed | 173: if inference_params.seqlen_offset > 0:
L52 | `SSM/modules/mamba2.py:175` | confirmed | 175: out, _, _ = self.step(u, conv_state, ssm_state)
L52 | `SSM/modules/mamba_simple.py:129` | confirmed | 129: if inference_params.seqlen_offset > 0:
L52 | `SSM/modules/mamba_simple.py:131` | confirmed | 131: out, _, _ = self.step(hidden_states, conv_state, ssm_state)
L54 | `SSM/modules/mamba3.py:66` | confirmed | 66: if not self.is_mimo:
L54 | `SSM/modules/mamba3.py:67` | confirmed | 67: self.mimo_rank = 1
L54 | `SSM/modules/mamba3.py:184` | confirmed | 184: if self.is_mimo: and 185: y = mamba3_mimo_combined(
L54 | `SSM/modules/mamba3.py:221` | confirmed | 221: y = mamba3_siso_combined(
L55 | `SSM/modules/mamba2_simple.py:87` | confirmed | 87: if self.learnable_init_states:
L55 | `SSM/modules/mamba2_simple.py:88` | confirmed | 88: self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
L56 | `SSM/modules/ssd_minimal.py:14` | confirmed | 14: def segsum_unstable(x): and 15: Naive segment sum calculation.
L56 | `SSM/modules/ssd_minimal.py:23` | confirmed | 23: def segsum(x): and 54: L = torch.exp(segsum(A))
L57 | `SSM/ops/utils/generation.py:29` | confirmed | 29: def reset(self, max_seqlen, max_batch_size): 32 sets seqlen_offset to 0. 34 zeroes lengths_per_sample.
L58 | `SSM/modules/block.py:12` | confirmed | 12: self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
L58 | `SSM/modules/block.py:23` | confirmed | 23: This is purely for performance reasons, as we can fuse add and LayerNorm.
L58 | `SSM/modules/mamba2.py:58` | confirmed | 58: # Fused kernel and sharding options
L58 | `SSM/modules/mamba_simple.py:46` | confirmed | 46: use_fast_path=True,  # Fused kernel options
L59 | `SSM/models/config_mamba.py:18` | confirmed | 18: tie_embeddings: bool = True
L59 | `SSM/models/mixer_seq_simple.py:267` | confirmed | 267: def tie_weights(self):
L59 | `SSM/models/mixer_seq_simple.py:269` | confirmed | 269: self.lm_head.weight = self.backbone.embedding.weight
L63 | `SSM/models/config_mamba.py:17` | confirmed | 17: pad_vocab_size_multiple: int = 8
L63 | `SSM/models/mixer_seq_simple.py:235` | confirmed | 235: pad_vocab_size_multiple = config.pad_vocab_size_multiple
L63 | `SSM/models/mixer_seq_simple.py:239` | confirmed | 239: if vocab_size % pad_vocab_size_multiple != 0:
L76 | `SSM/modules/mamba3.py:105` | confirmed | 105: self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs). `C_norm` is at 106.
L77 | `SSM/modules/mamba3.py:100` | confirmed | 100: self.B_bias = nn.Parameter(1+torch.zeros((self.nheads, self.mimo_rank, self.d_state), ...). `C_bias` is at 101.
L78 | `SSM/modules/mamba3.py:118` | confirmed | 118: # D "skip" parameter and 119: self.D = nn.Parameter(torch.ones(self.nheads, device=device))
L79 | `SSM/modules/mamba3.py:114` | confirmed | 114: self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
L79 | `SSM/modules/mamba3.py:116` | confirmed | 116: self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)
L83 | `SSM/modules/ssd_minimal.py:53` | confirmed | 53: # 1. Compute the output for each intra-chunk (diagonal blocks)
L83 | `SSM/modules/ssd_minimal.py:62` | confirmed | 62: # 3. Compute the inter-chunk SSM recurrence
L85 | `SSM/ops/utils/generation.py:24` | confirmed | 24: seqlen_offset: int = 0
L85 | `SSM/modules/mamba3.py:146` | confirmed | 146: if inference_params.seqlen_offset > 0:
L86 | `SSM/modules/mamba2.py:46` | confirmed | 46: d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
L86 | `SSM/modules/mamba2.py:272` | confirmed | 272: y = torch.cat([F.silu(z0) * x0, y], dim=-1)
L87 | `SSM/modules/mamba_simple.py:38` | confirmed | 38: dt_rank="auto",
L87 | `SSM/modules/mamba_simple.py:58` | confirmed | 58: self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
L88 | `SSM/modules/mamba2.py:51` | confirmed | 51: norm_before_gate=False,
L88 | `SSM/modules/mamba3.py:41` | wrong | 41: is_outproj_norm=False, The citation can stay. The row misreads the word: the flag enables the norm and does not select a place. The proof is at SSM/modules/mamba3.py:122: if self.is_outproj_norm: and SSM/modules/mamba3.py:126: norm_before_gate=True,
```

## Row corrections

### L39

Cell: Its meaning in the source
Old: Official code: `hidden_states` and `hidden_dim` name the `d_model` tensor that passes through the layers.
New: Official code: `hidden_states` names a `d_model` tensor. At a layer boundary it is the output of the last mixer or MLP. The class `Block` passes a second tensor, `residual`, with it. The next class `Block` adds them. `hidden_dim` names the last axis of that tensor.
Proof: SSM/modules/block.py:22 "the hidden_states (output of the mixer) and the residual". SSM/modules/block.py:52 "residual = (hidden_states + residual)". SSM/modules/block.py:88 "return hidden_states, residual". SSM/modules/mamba2.py:156 "u: (batch, seqlen, hidden_dim)".

Cell: Our term
Old: **Residual stream** for `hidden_states`.
New: **Residual stream** for the sum of `hidden_states` and `residual` at a layer boundary.
Proof: The same lines prove the old mapping false after the first layer. SSM/models/mixer_seq_simple.py:144 "the residual branch (output of Add)". A port that carries only `hidden_states` loses the residual.

Cell: Citation
Old: `SSM/modules/block.py:43`
New: `SSM/modules/block.py:22`, `SSM/modules/block.py:43`, `SSM/modules/block.py:52`
Proof: The two new lines hold the proof above.

### L40

Cell: Its meaning in the source
Old: Official code: `ssm_state` is the full `[batch, nheads, headdim, d_state]` tensor. A forward writes the accumulator to it, and `step` writes the state.
New: Official code: `ssm_state` of Mamba-2 and Mamba-3 is the full `[batch, nheads, headdim, d_state]` tensor. `ssm_state` of Mamba-1 is `[batch, d_inner, d_state]`. A Mamba-3 forward writes the accumulator to it, and `step` writes the state.
Proof: SSM/modules/mamba_simple.py:264 "batch_size, self.d_model * self.expand, self.d_state". SSM/modules/mamba2.py:353 "batch_size, self.nheads, self.headdim, self.d_state". SSM/modules/mamba2.py:264 "ssm_state.copy_(last_state)" is in a Mamba-2 forward, and this row calls that value the state.

Cell: Its meaning in the source
Old: `last_state` of Mamba-2 is the state.
New: `last_state` of Mamba-1 and Mamba-2 is the state. `last_state` of Mamba-3 is the accumulator.
Proof: SSM/ops/selective_scan_interface.py:49 "last_state = x[:, :, -1, 1::2]". SSM/modules/mamba_simple.py:202 "y, last_state = y". SSM/modules/mamba3.py:239 "y, last_angle, last_state, last_k, last_v, *rest = y". TRI/mamba3_siso_combined.py:149 returns `Final_SSM_State`. TRI/mamba3_siso_fwd.py:431 stores `acc_ssm_states`. For the Mamba-1 line, I read only the grep hits of the kernel, not its loop.

Cell: Our term
Old: **Accumulator** for the kernel values and for what an official forward writes.
New: **Accumulator** for the kernel values and for what an official Mamba-3 forward writes.
Proof: SSM/modules/mamba2.py:264 and SSM/modules/mamba_simple.py:203 are forwards that write `last_state`.

Cell: Citation
Old: `SSM/modules/mamba2.py:169`
New: `SSM/modules/mamba2.py:262`, `SSM/modules/mamba2.py:264`, `SSM/modules/mamba_simple.py:202`, `SSM/modules/mamba_simple.py:264`
Proof: SSM/modules/mamba2.py:169 has no `last_state`. SSM/modules/mamba2.py:262 "y, last_state, *rest = y".

### L42

Cell: Citation
Old: `SSM/models/mixer_seq_simple.py:152`
New: `SSM/models/mixer_seq_simple.py:152`, `SSM/models/mixer_seq_simple.py:164`, `SSM/models/mixer_seq_simple.py:167`
Proof: No citation of the row shows `layer_idx` or `n_layer`. 164 "layer_idx=i,". 167 "for i in range(n_layer)".

### L46

Cell: Its meaning in the source
Old: Official code: "rotary position embeddings", "Rotary embeddings", `rope` and `rotary` name our rotation.
New: Official code, in `Mamba3` and its kernels: "rotary position embeddings", "Rotary embeddings", `rope` and `rotary` name our rotation. In the class `MHA`, `rotary` names the fixed-frequency encoding.
Proof: SSM/modules/mha.py:44 "class MHA(nn.Module):". SSM/modules/mha.py:61 "rotary_emb_base=10000.0,". SSM/modules/mha.py:95 "self.rotary_emb = RotaryEmbedding(". I checked only `SSM/modules/mamba3.py` in this batch. The words in the kernels are for the kernel batch.

Cell: Citation
Old: `SSM/modules/mamba3.py:418`
New: `SSM/modules/mamba3.py:418`, `SSM/modules/mha.py:61`, `SSM/modules/mha.py:95`
Proof: The proof above.

### L79

Cell: Citation
Old: `SSM/modules/mamba3.py:114`, `SSM/modules/mamba3.py:116`
New: `SSM/modules/mamba3.py:114`, `SSM/modules/mamba3.py:115`, `SSM/modules/mamba3.py:116`
Proof: No SSM citation shows `mimo_z`. 115 "self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)".

### L88

Cell: Its meaning in the source
Old: The place of the extra RMSNorm around the gate.
New: "pre-gate", "post-gate" and `norm_before_gate` give the place of the extra RMSNorm around the gate. `is_outproj_norm` enables that norm in Mamba-3, always before the gate.
Proof: SSM/modules/mamba3.py:122 "if self.is_outproj_norm:". SSM/modules/mamba3.py:126 "norm_before_gate=True,". SSM/modules/mamba2.py:144 "norm_before_gate=self.norm_before_gate". M3/appendix/results.tex:15 "placement (pre- vs post-gate)".

Cell: Citation
Old: `SSM/modules/mamba3.py:41`
New: `SSM/modules/mamba3.py:41`, `SSM/modules/mamba3.py:122`, `SSM/modules/mamba3.py:126`
Proof: The proof above.

## Extra findings

- L38: `SSR` has a second use of "Mamba block". SSR:159 "two Mamba blocks are needed for each "layer" (MHA block + MLP block) of a Transformer". The sentence counts layers, so the phrase can mean the class `Block` with no MLP, not the class `Mamba`. This is my reading, and the source does not state it. No correction is proposed.
- L42: `ssm_cfg["layer"]` is not literal source text. SSM/models/mixer_seq_simple.py:53 reads the key with `ssm_cfg.pop("layer", "Mamba1")`. A reader still finds the key at the citation, so no correction is proposed.
- L52: the Mamba-3 one-token branch may fail. SSM/modules/mamba3.py:140 "batch, seqlen, dim = u.shape", and 147 passes that 3-axis `u` to `step`. SSM/modules/mamba3.py:295 "u: (batch, d_model)". SSM/modules/mamba3.py:260 has the pattern "b (r g s) -> b r g s", with two input axes. Mamba-2 removes the axis at SSM/modules/mamba2.py:281 "hidden_states.squeeze(1)". I did not run this code. The row is correct on which branch runs, but "also serves one-token calls" can be false for Mamba-3.
- L55: the row lists `init_state`, and its meaning cell does not name it. TRI/angle_dt.py:128 "init_state: Optional[Tensor] = None,". TRI/angle_dt.py:138 "init_state: Initial state (num_sequences, nheads, dim) or None". This word is outside the source group of this batch.
- L85: the cited lines do not show the count. The proof is at SSM/ops/utils/generation.py:225 "inference_params.seqlen_offset += sequences[-1].shape[1]".
- L87: the cited lines give the value of `dt_rank`, not its use. The `Δ` projection is at SSM/modules/mamba_simple.py:80 "self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)".