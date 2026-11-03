Batch `mixed-1`: the report file does NOT exist. The Write tool refused `/tmp/source-terms-check/report-mixed-1.md` with "Subagents should return findings as text, not write report files". I did not work around this through Bash. Your script will not find the file. To use the script, save the text between the two markers below to that path.

Counts: 17 citations. 17 confirmed, 0 wrong, 0 not found.
Rows with a correction: L186, L187.
Rows not finished: none.
The linter did not run on this text, because it had no file to read.

--- report start ---
L185 | `SSM/modules/mamba3.py:30` | confirmed | 30: d_state=128,
L185 | `SSM/ops/triton/ssd_combined.py:99` | confirmed | 99: chunk_size, hdim, dstate,
L185 | `TRI/mamba3_siso_combined.py:318` | confirmed | 318: Q: Query tensor (batch, seqlen, nheads_qk, headdim_qk). SSM/modules/mamba3.py:164 and :222 pass a tensor with last axis d_state as Q.
L187 | `SSM/modules/mamba3.py:33` | confirmed | 33: ngroups=1,
L187 | `SSM/modules/mamba3.py:74` | confirmed | 74: self.num_bc_heads = ngroups
L187 | `TRI/mamba3_siso_combined.py:318` | confirmed | 318: Q: Query tensor (batch, seqlen, nheads_qk, headdim_qk). SSM/modules/mamba3.py:164 gives g=self.num_bc_heads, and :222 passes Q=C.squeeze(2).
L188 | `SSM/modules/mamba3.py:82` | confirmed | 82: self.num_rope_angles = self.split_tensor_size // 2
L188 | `TRI/mamba3_siso_combined.py:327` | confirmed | 327: Angles: Rotary angle rates (batch, seqlen, nheads, headdim_angles)
L188 | `TIL/mamba3_mimo.py:262` | confirmed | 262: Angles: Rotary position embeddings (batch, seqlen, nheads, headangles)
L189 | `SSM/modules/mamba2.py:81` | confirmed | 81: self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
L189 | `SSM/modules/mamba2.py:104` | confirmed | 104: conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
L189 | `SSM/modules/mamba2_simple.py:73` | confirmed | 73: conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
L162 | `MM/mamba3.py:71` | confirmed | 71: mimo_rank: int = 4      # MIMO rank (r) when use_mimo=True
L162 | `MM/mamba3.py:385` | confirmed | 385: B = rearrange(B, "b l (n r) -> b l n r", r=R)
L182 | `MM/mamba3.py:65` | confirmed | 65: expand: int = 2       # Expansion factor (E). d_inner = expand * d_model
L186 | `TRI/mamba3_mimo_rotary_step.py:29` | confirmed | 29: stride_out_q,           # (batch, mimo_dim, nheads, headdim)
L186 | `TIL/mamba3_mimo_fwd.py:45` | confirmed | 45: R, (a kernel argument). TIL/mamba3_mimo.py:283 gives the meaning: "where R is the MIMO rank".

## Row corrections

### L186

Cell: Citation
Old: `M3/appendix/mimo.tex:5`, `TRI/mamba3_mimo_rotary_step.py:29`
New: `M3/appendix/mimo.tex:5`, `SSM/modules/mamba3.py:43`, `TRI/mamba3_mimo_rotary_step.py:29`
Proof: SSM/modules/mamba3.py:43: `mimo_rank=4,`. The row claims "Official module: `mimo_rank`" and no citation of the row shows it.

### L187

Cell: Spelling in each source
Old: Kernels: `nheads_qk`.
New: Kernels: `nheads_qk`. The Tilelang forward and backward files: `G`.
Proof: TIL/mamba3_mimo_fwd.py:42: `G,`. TIL/mamba3_mimo_fwd.py:71: `Q: T.Tensor([B, S, R, G, N], dtype)`. `rg -c nheads_qk` finds no match in the four files `mamba3_mimo_fwd.py`, `mamba3_mimo_fwd_varlen.py`, `mamba3_mimo_bwd.py` and `mamba3_mimo_bwd_varlen.py`. In the Tilelang folder, the search that lists files finds `nheads_qk` only in the wrapper TIL/mamba3_mimo.py.

Cell: Citation
Old: `TRI/mamba3_siso_combined.py:318`
New: `TRI/mamba3_siso_combined.py:318`, `TIL/mamba3_mimo_fwd.py:42`
Proof: TIL/mamba3_mimo_fwd.py:42: `G,`. L165 gives `G` only for Mamba-2, so L165 does not cover this use.

## Extra findings

- L185, kernels: the Tilelang forward and backward files spell the state rank `N`. TIL/mamba3_mimo_fwd.py:43: `N,`. TIL/mamba3_mimo_fwd.py:71: `Q: T.Tensor([B, S, R, G, N], dtype)`. The row omits this spelling. L164 gives it, so a reader does not get a wrong fact. Optional text: "Kernels: `dstate` and `headdim_qk`. The Tilelang forward and backward files: `N`."
- L185, look-alike word: `dstates` (plural) names a tensor, not the dimension. TIL/mamba3_mimo_bwd.py:603: `dstates_shared = T.alloc_shared([N, P], dtype)`. SSM/ops/triton/ssd_combined.py:96, three lines above the citation of the row, holds `dstates_ptr`.
- L185, plural claim "Official modules": the test passes. `d_state` is at SSM/modules/mamba_simple.py:35, SSM/modules/mamba2_simple.py:28, SSM/modules/mamba2.py:41 and SSM/modules/mamba3.py:30.
- L186, kernels: two more spellings exist. TIL/mamba3_mimo.py:293: `batch, seqlen, mimo_rank, nheads_qk, headdim_qk = Q.shape`. SSM/ops/cute/mamba3/mamba3_step_fn.py:87: `# B: batch size, H: num heads, D: head dim, N: dstate, R: mimo`. A reader does not get a wrong fact from this omission.
- L187, Mamba-2 kernels: these use our word `ngroups`. SSM/ops/triton/ssd_combined.py:265: `_, _, ngroups, dstate = B.shape`. This is not a fault if "Kernels" in this row means the Mamba-3 kernels.
- L188, kernels: `headangles` has one match in the official code, the docstring at TIL/mamba3_mimo.py:262. The kernels write this axis in other forms. TIL/mamba3_mimo_fwd.py:82: `ANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], T.float32)`. TRI/mamba3_mimo_rotary_step.py:31: `# (batch, nheads, rotary_dim // 2)`. L102 lists `rotary_dim_divisor` and `rotary_dim`, but L102 and L188 do not link them to `a`. Optional text: "Kernels: `headdim_angles` and `headangles`. `TIL/mamba3_mimo_fwd.py`: `N//rotary_dim_divisor`. `TRI/mamba3_mimo_rotary_step.py`: `rotary_dim // 2`."
- L188, link from the module to the kernel: the test passes. SSM/modules/mamba3.py:159 splits `self.num_rope_angles` values. SSM/modules/mamba3.py:230 passes them as `Angles=angles`.
- L162, negative test: the test passes. In MM/mamba3.py, 21 einops strings hold the letter `r`, at lines 385 to 399, 439, 557 to 569, 575, 584, 821, 825, 839 and 842. All of them are in a MIMO path: the `use_mimo` branches at MM/mamba3.py:381 and :553, or `ssd_mimo` at MM/mamba3.py:785.
- L182, negative test: `E` as an expansion factor has one match in mamba3-minimal, at MM/mamba3.py:65. The other matches of `\bE\b` are "Appendix E" in MM/tests/test_parity.py:16, :27 and :68.
- Outside this batch, L160 and L184: the Cute kernel uses `D` and `hdim` for `per_head_dim`. SSM/ops/cute/mamba3/mamba3_step_fn.py:87: `D: head dim`. SSM/ops/cute/mamba3/mamba3_step_fn.py:590: `batch, nheads, hdim, dstate = state.shape`. SSM/ops/triton/ssd_combined.py:99 also has `hdim`. L184 does not list these spellings. L160 gives only the `D` of `mamba3_mimo_rotary_step.py`. I did not check these two rows.
--- report end ---