Batch `modules-2`: finished. No row is left unfinished.

Counts: 50 confirmed, 1 wrong, 0 not found (51 citations).
Rows with a correction: L98, L99, L106, L160, L164, L166.
No report file exists. The harness refused my one Write attempt, and I used no other tool to write it. The report is below.

L97 | `SSM/modules/mamba3.py:433` | confirmed | 433: # K (=B) State
L97 | `SSM/modules/mamba3.py:447` | confirmed | 447: # V (=x) State
L98 | `SSM/modules/mamba3.py:85` | confirmed | 85: # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
L98 | `SSM/modules/mamba3.py:87` | confirmed | 87: self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
L98 | `SSM/modules/mamba3.py:151` | confirmed | 151: zxBCdtAtrap = self.in_proj(u)
L98 | `SSM/modules/mamba2.py:95` | confirmed | 95: # Order: [z, x, B, C, dt]
L98 | `SSM/modules/mamba2.py:178` | confirmed | 178: zxbcdt = self.in_proj(u)
L99 | `SSM/modules/mamba3.py:152` | confirmed | 152: z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
L99 | `SSM/modules/mamba3.py:161` | wrong | 161: dim=-1) is the last line of the split. It holds no part name. The proof is at SSM/modules/mamba3.py:152: z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
L99 | `SSM/modules/mamba2.py:211` | confirmed | 211: z0, x0, z, xBC, dt = torch.split( on zxbcdt. The part names differ from the row list. See the correction.
L101 | `SSM/ops/utils/generation.py:18` | confirmed | 18: class InferenceParams:
L101 | `SSM/ops/utils/generation.py:26` | confirmed | 26: key_value_memory_dict: dict = field(default_factory=dict)
L101 | `SSM/modules/block.py:90` | confirmed | 90: def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
L101 | `SSM/modules/mamba3.py:489` | confirmed | 489: inference_params.key_value_memory_dict[self.layer_idx] = (angle_dt_state, ssm_state, k_state, v_state)
L102 | `SSM/modules/mamba3.py:36` | confirmed | 36: rope_fraction=0.5,
L102 | `SSM/modules/mamba3.py:77` | confirmed | 77: assert rope_fraction in [0.5, 1.0]
L102 | `SSM/modules/mamba3.py:78` | confirmed | 78: self.rotary_dim_divisor = int(2/rope_fraction)
L102 | `SSM/modules/mamba3.py:79` | confirmed | 79: self.split_tensor_size = int(d_state * rope_fraction)
L104 | `SSM/modules/mamba3.py:425` | confirmed | 425: # Mamba-3 Combined Kernel States
L106 | `SSM/modules/ssd_minimal.py:34` | confirmed | 34: def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
L106 | `SSM/modules/mamba2.py:60` | confirmed | 60: use_mem_eff_path=True,
L106 | `SSM/modules/mamba2.py:184` | confirmed | 184: if self.use_mem_eff_path and inference_params is None: The line adds a condition. See the correction.
L107 | `SSM/modules/mamba2.py:269` | confirmed | 269: if self.rmsnorm:
L107 | `SSM/modules/mamba2.py:270` | confirmed | 270: y = self.norm(y, z)
L108 | `SSM/ops/utils/generation.py:133` | confirmed | 133: cg=False,
L108 | `SSM/ops/utils/generation.py:310` | confirmed | 310: assert message "CUDA graph decoding requires that the model has a method allocate_inference_cache"
L108 | `SSM/ops/utils/generation.py:342` | confirmed | 342: def capture_graph(
L108 | `SSM/ops/utils/generation.py:372` | confirmed | 372: graph = torch.cuda.CUDAGraph()
L108 | `SSM/ops/utils/generation.py:385` | confirmed | 385: graph.replay()
L114 | `SSM/modules/mamba3.py:172` | confirmed | 172: ADT = _A * DT
L115 | `SSM/models/mixer_seq_simple.py:67` | confirmed | 67: if d_intermediate == 0:
L115 | `SSM/models/mixer_seq_simple.py:71` | confirmed | 71: GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
L116 | `SSM/modules/mamba3.py:74` | confirmed | 74: self.num_bc_heads = ngroups
L158 | `SSM/modules/mamba3.py:260` | confirmed | 260: B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)
L159 | `SSM/modules/mamba3.py:135` | confirmed | 135: def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
L159 | `SSM/modules/mamba2.py:154` | confirmed | 154: def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
L160 | `SSM/modules/mamba_simple.py:137` | confirmed | 137: "d (b l) -> b d l", on xz. Its d is d_inner * 2 (62: nn.Linear(self.d_model, self.d_inner * 2).
L160 | `SSM/modules/mamba_simple.py:182` | confirmed | 182: x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
L162 | `SSM/modules/mamba3.py:164` | confirmed | 164: "b l (r g n) -> b l r g n", r=self.mimo_rank
L164 | `SSM/modules/mamba3.py:164` | confirmed | 164: "b l (r g n) -> b l r g n". The size at 156 makes n equal d_state. The row generalizes too far. See the correction.
L166 | `SSM/modules/mamba3.py:162` | confirmed | 162: z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
L166 | `SSM/modules/ssd_minimal.py:48` | confirmed | 48: rearrange(x, "b (c l) ... -> b c l ...", l=block_len). The literal b c l h p is at 77. See the correction.
L167 | `SSM/modules/mamba3.py:188` | confirmed | 188: V=x,
L167 | `SSM/modules/mamba3.py:299` | confirmed | 299: v_state: (batch, nheads, headdim). The line holds the word and the shape of x.
L168 | `SSM/modules/mamba3.py:187` | confirmed | 187: K=B,
L168 | `SSM/modules/mamba3.py:298` | confirmed | 298: k_state: (batch, R, nheads, d_state). The line holds the word and the shape of B.
L169 | `SSM/modules/mamba3.py:186` | confirmed | 186: Q=C,
L170 | `SSM/modules/ssd_minimal.py:48` | confirmed | 48: "b (c l) ... -> b c l ...". Here c is the chunk axis.
L180 | `SSM/modules/ssd_minimal.py:86` | confirmed | 86: # Denoted (B, T, Q, D, P) in the paper
L183 | `SSM/modules/mamba3.py:73` | confirmed | 73: self.nheads = self.d_inner // self.headdim
L184 | `SSM/modules/mamba2.py:45` | confirmed | 45: headdim=64,

## Row corrections

### L98

Cell: Meaning
Old: One weight. Its name lists its parts: `z`, `x`, `B`, `C`, `dt`, `A`, `trap`.
New: One weight. The name `zxBCdtAtrap` lists seven parts: `z`, `x`, `B`, `C`, `dt`, `A`, `trap`. The name omits the eighth part, `angle`. The name `zxbcdt` lists the five parts of Mamba-2.
Proof: SSM/modules/mamba3.py:85 "# Order: [z, x, B, C, dd_dt, dd_A, trap, angle]". SSM/modules/mamba2.py:95 "# Order: [z, x, B, C, dt]".

### L99

Cell: Source word
Old: the parts of `torch.split`: `z`, `x`, `B`, `C`, `dd_dt`, `dd_A`, `trap`, `angles`
New: the parts of `torch.split`: `z`, `x`, `B`, `C`, `dd_dt`, `dd_A`, `trap`, `angles` in Mamba-3, `z0`, `x0`, `z`, `xBC`, `dt` in Mamba-2
Proof: SSM/modules/mamba2.py:211 "z0, x0, z, xBC, dt = torch.split(".

Cell: Citation
Old: `SSM/modules/mamba3.py:152`, `SSM/modules/mamba3.py:161`, `SSM/modules/mamba2.py:211`
New: `SSM/modules/mamba3.py:152`, `SSM/modules/mamba2.py:211`
Proof: SSM/modules/mamba3.py:161 holds only "dim=-1)".

### L106

Cell: Meaning
Old: `use_mem_eff_path` selects the split-conv combined kernel.
New: `use_mem_eff_path` selects the split-conv combined kernel only if `inference_params` is `None`.
Proof: SSM/modules/mamba2.py:184 "if self.use_mem_eff_path and inference_params is None:".

### L160

Cell: Source use
Old: `d` is an inner width in the Mamba-1 einops.
New: `d` is `d_model` at the input of `in_proj` in the Mamba-1 einops. After `in_proj`, `d` is an inner width.
Proof: SSM/modules/mamba_simple.py:136 "self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)")". SSM/modules/mamba_simple.py:62 "nn.Linear(self.d_model, self.d_inner * 2".

Cell: Citation
Old: `SSM/modules/mamba_simple.py:137`, `SSM/modules/mamba_simple.py:182`, `TRI/mamba3_mimo_rotary_step.py:240`
New: `SSM/modules/mamba_simple.py:136`, `SSM/modules/mamba_simple.py:137`, `SSM/modules/mamba_simple.py:182`, `TRI/mamba3_mimo_rotary_step.py:240`
Proof: SSM/modules/mamba_simple.py:136 "b l d -> d (b l)".

### L164

Cell: Source use
Old: `n` is the state dimension in einops. `N` is the state in the papers and in Tilelang, and the count of heads in `mamba3_mimo_rotary_step.py`.
New: `n` is the state dimension in most einops. In `"b l n -> b n l"`, `n` is the count of heads. `N` is the state in the papers and in Tilelang. `N` is the count of heads in the comments of `mamba3.py` and in `mamba3_mimo_rotary_step.py`.
Proof: SSM/modules/mamba3.py:173 "DT = rearrange(DT, "b l n -> b n l")". SSM/modules/mamba3.py:171 "# (B, L, N)". SSM/modules/mamba3.py:158 gives the size of dd_dt as self.nheads.

I did not check the claims of this row about the papers and Tilelang. They are from the old text.

Cell: Citation
Old: `SSM/modules/mamba3.py:164`, `TIL/mamba3_mimo_fwd.py:43`, `TRI/mamba3_mimo_rotary_step.py:240`
New: `SSM/modules/mamba3.py:164`, `SSM/modules/mamba3.py:171`, `SSM/modules/mamba3.py:173`, `TIL/mamba3_mimo_fwd.py:43`, `TRI/mamba3_mimo_rotary_step.py:240`
Proof: SSM/modules/mamba3.py:171 and :173, as above.

### L166

Cell: Citation
Old: `SSM/modules/mamba3.py:162`, `SSM/modules/ssd_minimal.py:48`, `M2/glossary.tex:32`, `M1/method.tex:87`
New: `SSM/modules/mamba3.py:162`, `SSM/modules/ssd_minimal.py:48`, `SSM/modules/ssd_minimal.py:77`, `M2/glossary.tex:32`, `M1/method.tex:87`
Proof: The row gives `b c l h p` as code. SSM/modules/ssd_minimal.py:48 holds "b c l ...". SSM/modules/ssd_minimal.py:77 holds "b c l h p -> b (c l) h p".

## Extra findings

Each finding supports a row or adds a fact. None needs a correction.

- L97, "k_state holds B, v_state holds x": SSM/modules/mamba3.py:342 "nxt_v_state = x", :343 "nxt_k_state = B". The call at :336 rebinds `B` to the output of `apply_rotary_qk_inference_fwd`, with `bias_k` as an argument at :338. So in the step, `k_state` holds the `B` after that call.
- L97, "one tap only", official code: `k_state` and `v_state` have no time axis (SSM/modules/mamba3.py:441-452). SSM/modules/mamba3.py:409-410 copies the new values into them.
- L97, source words: `B_prev`, `x_prev` and `prev_Bx` are not in `mamba_ssm/` or README.md. A word-bounded `rg -w` finds no match. They belong to the other source groups of the row.
- L98, Mamba-1: `in_proj` gives only `xz` (SSM/modules/mamba_simple.py:62). `dt`, `B` and `C` come from `x_proj` (SSM/modules/mamba_simple.py:182-183). The row does not name Mamba-1, so its part list does not apply to `mamba_simple.py`.
- L98, other spelling: the Mamba-3 step names the same output `zxBCdt` (SSM/modules/mamba3.py:311). The split at :312 has the same eight parts.
- L99, negative claim "no noun for one part": `rg -i "sub.?proj|component|\bparts?\b|pieces?|portion|\bslice|\bhead of"` on mamba3.py, mamba2.py, mamba_simple.py and README.md finds only README.md:235 "Some parts of the model". That line is not about a projection. The claim holds for these files and this word list.
- L99, Mamba-2: `x`, `B` and `C` are not direct parts of the output of `in_proj`. SSM/modules/mamba2.py:243 splits them from `xBC` after the convolution.
- L101, "holds no learned parameter": SSM/ops/utils/generation.py:22-27 holds ints, the dict and an optional tensor. The claim holds.
- L101, "cache of all layers, keyed by layer_idx": SSM/models/mixer_seq_simple.py:186 "i: layer.allocate_inference_cache(". SSM/ops/utils/generation.py:317 "key_value_memory_dict=inf_cache,". The claim holds.
- L102, "divisor 2 means the whole state": SSM/modules/mamba3.py:78 gives divisor 2 for `rope_fraction` 1.0. SSM/modules/mamba3.py:79 then gives the whole `d_state`. TIL/mamba3_mimo.py:299 agrees: "rotary embedding on entire or half of headdim_qk". The claim holds.
- L102, source word: `rotary_dim` is not in `mamba_ssm/modules/` or `mamba_ssm/models/`. In `mamba_ssm/`, only `TRI/mamba3_mimo_rotary_step.py` holds it, for example at line 31.
- L104, negative claim "no word for a pathway": `rg -i "pathway|single.?ssd|double.?ssd|two.?ssd"` on the checkout, without `*.ipynb` files, finds no match. The claim holds.
- L104, "the only form": the forward calls `mamba3_mimo_combined` at SSM/modules/mamba3.py:185 and `mamba3_siso_combined` at :221. The step calls `mamba3_step_fn` at :357 and :380.
- L107, source words: `rmsnorm=True` is at SSM/modules/mamba2.py:50. SSM/modules/mamba2.py:26 imports `RMSNorm as RMSNormGated`, and :144 uses it. SSM/modules/mamba2.py:252 gives the scan `z` only if `rmsnorm` is false. This supports the row.
- L108, uncited claim "records decode calls only": SSM/ops/utils/generation.py:184 "if not cg or not decoding:" sends a non-decoding call to the model directly. SSM/ops/utils/generation.py:192 calls `model._decoding_cache.run(` in the other case. `capture_graph` has one call site at :323, and `update_graph_cache` has one at :161. The claim holds.
- L114, source word: `dd_A` is at SSM/modules/mamba3.py:152 and :169 "_A = -F.softplus(dd_A.to(torch.float32))".
- L115, uncited claim "optional part after its block": SSM/modules/block.py:31-35 sets `self.mlp` to `None` for `nn.Identity`. SSM/modules/block.py:86 runs `self.mlp` after the mixer at :67. The claim holds.
- L115, source word: the official code also has `class SwiGLU` at SSM/ops/triton/k_activations.py:158. `GatedMLP` is at SSM/modules/mlp.py:6. I did not read the forward of `GatedMLP`, so I do not state that it uses `SwiGLU`.
- L158, meanings of `S` that the row does not list: the comments of `mamba3.py` use `S` for the angles at SSM/modules/mamba3.py:177 "# (B, L, N, S)" and for the state at :266 "# (B, R, N, S)".
- L160, Mamba-2, which the row does not name: `d` is `d_in_proj` at SSM/modules/mamba2.py:180 and the width of `xBC` at :220.
- L167 and L168, stronger proof of "holds": SSM/modules/mamba3.py:342 and :343. The cited lines :299 and :298 give only the word and the shape.