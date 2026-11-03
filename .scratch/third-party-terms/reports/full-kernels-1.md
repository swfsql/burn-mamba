Batch `kernels-1`: 74 confirmed, 0 wrong, 0 not found. All rows are finished.
Rows with a correction: L40, L49, L55, L73, L74, L79, L97, L100, L107, L184.
I wrote no report file, as you instructed. The report follows.

Two verdicts need your judgement:
- L107 `SSM/ops/triton/layernorm_gated.py:343` is confirmed only if "inside the norm" means "inside the norm call". The correction removes this ambiguity.
- L74 `TRI/mamba3_mimo_rotary_step.py:46` holds the proof, but in upper case.

```text
L40 | `TRI/mamba3_siso_fwd.py:336` | confirmed | 336: acc_ssm_states = tl.load(
L40 | `TRI/mamba3_siso_fwd.py:352` | confirmed | 352: acc_ssm_states += input_v_state[:, None] * input_k_state[None, :] * dt_scalar * (1 - trap_scalar)
L40 | `TRI/mamba3_siso_fwd.py:423` | confirmed | 423: acc_ssm_states = acc_ssm_states * tl.math.exp2(da_cs_last) + tl.dot(
L40 | `TRI/mamba3_siso_fwd.py:431` | confirmed | 431: acc_ssm_states, the value that 429 stores to final_ssm_state_ptr.
L46 | `TRI/mamba3_siso_combined.py:311` | confirmed | 311: "Supports GQA (grouped-query attention), rotary". 312 continues: "position embeddings".
L46 | `TRI/mamba3_siso_fwd.py:267` | confirmed | 267: # Load rotary angles
L46 | `TRI/mamba3_siso_fwd.py:312` | confirmed | 312: # Apply rotary embeddings to K and scale
L46 | `TRI/mamba3_siso_fwd.py:470` | confirmed | 470: Angles: Rotary angles (batch, seqlen, nheads, headdim_angles)
L46 | `TRI/mamba3_mimo_rotary_step.py:46` | confirmed | 46: "rotate the first half and second half separately (like in the original RoPE paper)"
L46 | `TIL/mamba3_mimo.py:262` | confirmed | 262: Angles: Rotary position embeddings (batch, seqlen, nheads, headangles)
L48 | `TRI/mamba3_siso_combined.py:323` | confirmed | 323: Trap: Trapezoidal factor (batch, nheads, seqlen)
L48 | `TIL/mamba3_mimo.py:256` | confirmed | 256: Trap: Trapezoidal mixing factor, pre-sigmoid (batch, nheads, seqlen)
L48 | `TIL/mamba3_mimo_fwd.py:193` | confirmed | 193: gamma_frag[cs] = dt_frag[cs] * T.sigmoid(trap_frag[cs])
L49 | `TRI/mamba3_siso_fwd.py:285` | confirmed | 285: shifted_gamma = dt_shifted * (1 - trap_shifted). 275-277 load dt_shifted at offs_seqlen + 1.
L49 | `TRI/mamba3_siso_fwd.py:286` | confirmed | 286: gamma = dt * trap
L49 | `TRI/mamba3_siso_fwd.py:287` | confirmed | 287: scale = shifted_gamma + gamma. See the correction for the second `scale`.
L49 | `TIL/mamba3_mimo_fwd.py:179` | confirmed | 179: shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype). The value is at 181-182: dt_shifted_frag[cs] * T.sigmoid(-trap_shifted_frag[cs]).
L49 | `TIL/mamba3_mimo_fwd.py:193` | confirmed | 193: gamma_frag[cs] = dt_frag[cs] * T.sigmoid(trap_frag[cs])
L50 | `SSM/ops/triton/ssd_combined.py:80` | confirmed | 80: triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, ...
L50 | `TRI/angle_dt.py:166` | confirmed | 166: BLOCK_D = triton.next_power_of_2(dim)
L50 | `TRI/mamba3_siso_fwd.py:264` | confirmed | 264: q_pre_block = q_desc.load([chunk_start, 0])
L50 | `TRI/mamba3_siso_fwd.py:268` | confirmed | 268: angle_block = tl.load(
L53 | `TRI/mamba3_siso_combined.py:318` | confirmed | 318: Q: Query tensor (batch, seqlen, nheads_qk, headdim_qk)
L53 | `TIL/mamba3_mimo_fwd.py:40` | confirmed | 40: S, a kernel argument. 59: nchunks = tilelang.cdiv(S, chunk_size).
L55 | `SSM/ops/triton/ssd_combined.py:636` | confirmed | 636: initial_states: (batch, nheads, headdim, dstate)
L55 | `TRI/mamba3_siso_combined.py:334` | confirmed | 334: Input_States: Optional initial state tuple for recurrent inference.
L55 | `TRI/mamba3_siso_combined.py:335` | confirmed | 335: Angle State: (num_sequences, nheads, headdim_angles)
L60 | `TIL/mamba3_mimo_bwd.py:140` | confirmed | 140: # --- Output Accumulators ---. 142: dPhi_shared = T.alloc_shared([R, P], accum_dtype).
L60 | `TIL/mamba3_mimo_fwd.py:57` | confirmed | 57: accum_dtype = 'float32'
L72 | `TRI/mamba3_siso_combined.py:323` | confirmed | 323: Trap: Trapezoidal factor (batch, nheads, seqlen)
L72 | `TRI/mamba3_siso_combined.py:324` | confirmed | 324: Mixing factor in [0, 1] for trapezoidal discretization.
L72 | `TIL/mamba3_mimo.py:256` | confirmed | 256: Trap: Trapezoidal mixing factor, pre-sigmoid
L72 | `TIL/mamba3_mimo_fwd.py:114` | confirmed | 114: Trap: convex-combination modulator used in exponential-trapezoidal discretization.
L73 | `TRI/mamba3_siso_combined.py:327` | confirmed | 327: Angles: Rotary angle rates (batch, seqlen, nheads, headdim_angles)
L73 | `TIL/mamba3_mimo.py:262` | confirmed | 262: Angles: Rotary position embeddings. 71-72 pass Angles and DT to angle_dt_fwd, so Angles is a rate.
L74 | `TRI/mamba3_mimo_rotary_step.py:46` | confirmed | 46: ROTATE_PAIRWISE: tl.constexpr, "If true, rotate every pair of dimensions together." The identifier is upper case. See the correction.
L74 | `TRI/mamba3_mimo_rotary_step.py:279` | confirmed | 279: # Interleaved rotary: pairs are (x0,x1), (x2,x3), ...
L74 | `TRI/mamba3_mimo_rotary_step.py:301` | confirmed | 301: # Halved rotary: split full headdim in half, pairs are (dim_i, dim_{i+D/2})
L79 | `TIL/mamba3_mimo_fwd.py:111` | confirmed | 111: Psi: MIMO X projection.
L79 | `TIL/mamba3_mimo_fwd.py:113` | confirmed | 113: Zeta: MIMO Z projection.
L97 | `TRI/mamba3_siso_fwd.py:351` | confirmed | 351: # Step on the SSM states with input K/V states to account for trapezoidal discretization
L97 | `TRI/mamba3_siso_fwd.py:352` | confirmed | 352: acc_ssm_states += input_v_state[:, None] * input_k_state[None, :] * dt_scalar * (1 - trap_scalar)
L100 | `TRI/mamba3_siso_combined.py:341` | confirmed | 341: cu_seqlens: Cumulative sequence lengths for variable-length support.
L100 | `TRI/mamba3_siso_combined.py:342` | confirmed | 342: Shape: (num_sequences + 1,), dtype: torch.int32.
L100 | `TRI/mamba3_siso_combined.py:344` | confirmed | 344: When using cu_seqlens, batch must be 1 and the seqlen dimension
L100 | `TRI/mamba3_siso_combined.py:362` | confirmed | 362: num_sequences = batch for batched mode, len(cu_seqlens)-1 for varlen mode.
L100 | `TRI/mamba3_siso_combined.py:379` | confirmed | 379: if batch != 1: and 380 raises ValueError.
L100 | `SSM/ops/triton/ssd_combined.py:155` | confirmed | 155: seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, ...
L100 | `SSM/ops/triton/ssd_combined.py:158` | confirmed | 158: scale = tl.where(seq_idx_m == seq_idx_last, ..., 0.0)
L100 | `SSM/ops/triton/ssd_combined.py:637` | confirmed | 637: seq_idx: (batch, seqlen)
L100 | `TRI/angle_dt.py:57` | confirmed | 57: cu_seqlen_start = tl.load(CU_SEQLENS + pid_seq * stride_cu_seqlen).to(tl.int32)
L100 | `TRI/angle_dt.py:58` | confirmed | 58: cu_seqlen_end = tl.load(CU_SEQLENS + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
L102 | `TIL/mamba3_mimo.py:266` | confirmed | 266: rotary_dim_divisor: Divisor for rotary embedding dimensions (default: 4, meaning angles have 1/4 of headdim_qk)
L102 | `TIL/mamba3_mimo.py:299` | confirmed | 299: assert rotary_dim_divisor in [2, 4], "currently only supports rotary embedding on entire or half of headdim_qk"
L102 | `TRI/mamba3_mimo_rotary_step.py:183` | confirmed | 183: rotary_dim = angle_state.shape[-1] * 2
L103 | `TRI/mamba3_siso_fwd.py:256` | confirmed | 256: # Phase 1: Preprocessing - Apply bias, rotary embeddings, compute QK dots.
L103 | `TRI/mamba3_siso_fwd.py:312` | confirmed | 312: # Apply rotary embeddings to K and scale. Phase 2 starts at 334.
L104 | `TRI/mamba3_siso_fwd.py:285` | confirmed | 285: shifted_gamma = dt_shifted * (1 - trap_shifted)
L104 | `TRI/mamba3_siso_fwd.py:287` | confirmed | 287: scale = shifted_gamma + gamma. 324 applies it to K in one pass.
L106 | `SSM/ops/triton/ssd_combined.py:624` | confirmed | 624: def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, ...
L106 | `SSM/ops/triton/ssd_combined.py:646` | confirmed | 646: def mamba_chunk_scan(x, dt, A, B, C, chunk_size, ...
L106 | `SSM/ops/triton/ssd_combined.py:977` | confirmed | 977: def mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, ...
L107 | `TRI/mamba3_siso_combined.py:332` | confirmed | 332: Z: Gating tensor (batch, seqlen, nheads, headdim_v)
L107 | `SSM/ops/triton/layernorm_gated.py:343` | confirmed | 343: "If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))". The order depends on the flag. See the correction.
L113 | `TIL/mamba3_mimo.py:255` | confirmed | 255: DT: Time delta tensor dt (batch, nheads, seqlen)
L116 | `TRI/mamba3_siso_combined.py:318` | confirmed | 318: Q: Query tensor (batch, seqlen, nheads_qk, headdim_qk)
L158 | `TIL/mamba3_mimo_fwd.py:40` | confirmed | 40: S. 71: Q: T.Tensor([B, S, R, G, N], dtype). 59 divides S into chunks.
L158 | `TRI/mamba3_mimo_rotary_step.py:242` | confirmed | 242: angle_state: torch.Tensor, # (B, N, S) S: num_rope_angles
L160 | `TRI/mamba3_mimo_rotary_step.py:240` | confirmed | 240: q: torch.Tensor, # (B, R, N, D). 251: batch, mimo_dim, nheads, headdim = q.shape.
L164 | `TIL/mamba3_mimo_fwd.py:43` | confirmed | 43: N. 89: FINAL_STATE: T.Tensor([B, H, N, P], T.float32).
L164 | `TRI/mamba3_mimo_rotary_step.py:240` | confirmed | 240: # (B, R, N, D). 251 binds N to nheads.
L184 | `TRI/mamba3_siso_combined.py:320` | confirmed | 320: V: Value tensor (batch, seqlen, nheads, headdim_v)
L184 | `TRI/mamba3_mimo_rotary_step.py:29` | confirmed | 29: stride_out_q, # (batch, mimo_dim, nheads, headdim)
L184 | `TRI/mamba3_mimo_rotary_step.py:168` | confirmed | 168: q: (batch, mimo_dim, nheads, headdim)
```

## Row corrections

### L40

Cell: Citation
Old: `TRI/mamba3_siso_fwd.py:431`,
New: `TRI/mamba3_siso_fwd.py:200`, `TRI/mamba3_siso_fwd.py:431`,
Proof: No citation shows `Final_SSM_State`. TRI/mamba3_siso_fwd.py:200: final_ssm_state_ptr = Final_SSM_State + seq_idx * stride_final_ssm_state_seq + ...

### L49

Cell: Its meaning in the source
Old: `scale` is `shifted_gamma + gamma`.
New: In Phase 1 of `mamba3_siso_fwd.py`, `scale` is `shifted_gamma + gamma`. In Phase 2 of that kernel and in the Mamba-2 kernels, `scale` holds the decay.
Proof: TRI/mamba3_siso_fwd.py:421: scale = tl.math.exp2(da_cs_rev). SSM/ops/triton/ssd_combined.py:153: scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)).

Cell: Citation
Old: `TRI/mamba3_siso_fwd.py:287`,
New: `TRI/mamba3_siso_fwd.py:287`, `TRI/mamba3_siso_fwd.py:421`, `SSM/ops/triton/ssd_combined.py:153`,
Proof: As above.

### L55

Cell: Its meaning in the source
Old: `Input_States` has one entry for each sequence.
New: `Input_States` has one entry for each sequence. `init_state` of `angle_dt_fwd` takes the angle entry of `Input_States`.
Proof: The row lists `init_state`, but it gives no meaning or citation. TRI/angle_dt.py:138: init_state: Initial state (num_sequences, nheads, dim) or None. TRI/mamba3_siso_combined.py:92: init_state=Input_Angle_State. TRI/mamba3_siso_combined.py:382 unpacks Input_Angle_State from Input_States.

Cell: Citation
Old: `TRI/mamba3_siso_combined.py:335`,
New: `TRI/mamba3_siso_combined.py:335`, `TRI/mamba3_siso_combined.py:92`, `TRI/angle_dt.py:138`,
Proof: As above.

### L73

Cell: Citation
Old: `TRI/mamba3_siso_combined.py:327`,
New: `TRI/mamba3_siso_combined.py:327`, `TRI/mamba3_mimo_rotary_step.py:171`, `SSM/modules/mamba3.py:272`,
Proof: The two kernel citations and MM/mamba3.py:271 do not show `angle_proj`. I did not read M3/structure/methods_complex.tex:23. TRI/mamba3_mimo_rotary_step.py:171: angle_proj: (batch, nheads, rotary_dim / 2). SSM/modules/mamba3.py:272: angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1).

### L74

Cell: Source word
Old: `rotate_pairwise`, "Interleaved rotary"
New: `rotate_pairwise`, `ROTATE_PAIRWISE`, "Interleaved rotary"
Proof: TRI/mamba3_mimo_rotary_step.py:46 holds only the upper-case kernel constant.

Cell: Citation
Old: `TRI/mamba3_mimo_rotary_step.py:46`,
New: `TRI/mamba3_mimo_rotary_step.py:46`, `TRI/mamba3_mimo_rotary_step.py:161`,
Proof: TRI/mamba3_mimo_rotary_step.py:161: rotate_pairwise=True,

### L79

Cell: Citation
Old: `TIL/mamba3_mimo_fwd.py:111`, `TIL/mamba3_mimo_fwd.py:113`,
New: `TIL/mamba3_mimo_fwd.py:111`, `TIL/mamba3_mimo_fwd.py:112`, `TIL/mamba3_mimo_fwd.py:113`,
Proof: No citation shows Phi. TIL/mamba3_mimo_fwd.py:112: Phi: MIMO O projection.

### L97

Cell: Meaning
Old: Official code: `k_state` holds `B` and `v_state` holds `x`.
New: Official code: `k_state` holds `B` after its bias and its rotation. `v_state` holds `x`.
Proof:
- TRI/mamba3_siso_fwd.py:297 adds the bias, 316 rotates K, and 321 stores k_pre_block to the final K state. 324 applies `scale` after this store.
- SSM/modules/mamba3.py:336 rebinds B to the rotated output, with bias_k at 338. 343: nxt_k_state = B.
- TIL/mamba3_mimo_fwd.py:272 rotates K, and 279 copies k_shared to FINAL_K. I did not check the bias order in Tilelang.

Cell: Citation
Old: `TRI/mamba3_siso_fwd.py:352`,
New: `TRI/mamba3_siso_fwd.py:352`, `TRI/mamba3_siso_fwd.py:321`, `SSM/modules/mamba3.py:343`,
Proof: As above.

### L100

Cell: Meaning
Old: An entry of `cu_seqlens` or a change of `seq_idx` is the start of a sequence.
New: Each entry of `cu_seqlens` except the last is the start of a sequence. A change of `seq_idx` is also the start of a sequence.
Proof: TRI/mamba3_siso_combined.py:343: "[0, 128, 256, 512] for 3 sequences". TRI/angle_dt.py:58 reads the next entry as the end of a sequence.

Cell: Citation
Old: `TRI/mamba3_siso_combined.py:344`,
New: `TRI/mamba3_siso_combined.py:343`, `TRI/mamba3_siso_combined.py:344`,
Proof: As above.

### L107

Cell: Meaning
Old: In Mamba-2 with `rmsnorm=True`, the gate runs inside the norm.
New: In Mamba-2 with `rmsnorm=True`, the norm takes `z`. The default `norm_before_gate=False` gives `norm(x * silu(z))`. `norm_before_gate=True` gives `norm(x) * silu(z)`.
Proof: SSM/modules/mamba2.py:51: norm_before_gate=False. SSM/modules/mamba2.py:270: y = self.norm(y, z). SSM/ops/triton/layernorm_gated.py:343 gives both orders.

Cell: Citation
Old: `SSM/modules/mamba2.py:269`,
New: `SSM/modules/mamba2.py:26`, `SSM/modules/mamba2.py:51`, `SSM/modules/mamba2.py:269`,
Proof: No citation shows `RMSNormGated`. SSM/modules/mamba2.py:26: from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated. The class name in the kernel file is `RMSNorm`, at SSM/ops/triton/layernorm_gated.py:415.

### L184

Cell: Spelling in each source
Old: Official modules: `headdim`. Triton kernels: `headdim_v`.
New: Official modules and Mamba-2 Triton kernels: `headdim`. Mamba-3 SISO Triton kernels: `headdim_v`.
Proof: SSM/ops/triton/ssd_combined.py:263: batch, seqlen, nheads, headdim = x.shape. The only Triton files that hold `headdim_v` are mamba3_siso_{fwd,combined,bwd,step}.py.

Cell: Citation
Old: `SSM/modules/mamba2.py:45`,
New: `SSM/modules/mamba2.py:45`, `SSM/ops/triton/ssd_combined.py:263`,
Proof: As above.

## Extra findings

These findings are optional additions. The row text is not false in any of them.

- L40: In Tilelang, `states_frag` and `FINAL_STATE` are the spellings of the kernel's final state, and the doc has neither word.
  - TIL/mamba3_mimo_fwd.py:157 calls it "Running State", 158 allocates `states_frag`, 406 applies the decay, and 411 copies it to FINAL_STATE.
  - I did not read its additive update.
  - The comment at 157 uses the word "State" for this value. This is the confusion that L40 warns about.
  - `Final_SSM_State` also occurs at TIL/mamba3_mimo.py:94.
- L40: `last_state` also occurs in Mamba-3. SSM/modules/mamba3.py:207 unpacks it from the kernel output, and 209 copies it to ssm_state. The row names only Mamba-2.
- L48, L72: TRI/mamba3_siso_combined.py:324 gives the range of `Trap` as [0, 1], but the kernel applies the sigmoid to this input at TRI/mamba3_siso_fwd.py:278-279. A reader of the L72 citation can skip the sigmoid.
  - Text for L48: The docstring at `TRI/mamba3_siso_combined.py:324` gives the range [0, 1], but the kernel applies the sigmoid to `Trap`.
- L60: `accum_dtype` also types `states_frag` at TIL/mamba3_mimo_fwd.py:158.
  - Text for L60: `accum_dtype` also types `states_frag`.
  - TRI/mamba3_siso_bwd.py:917 "# Zero accumulators for this qk_head" is a gradient buffer in a Triton kernel (`dq_acc`, `dk_acc` at 918-919). The row names only Tilelang.
- L73: TRI/mamba3_siso_combined.py:315 and :328 describe cumsum(Angles * DT), but the kernel applies tanh and π. TRI/angle_dt.py:94: angle_vals = tanh_approx(angle_vals) * PI. TRI/mamba3_mimo_rotary_step.py:75: "tanh(angle_proj) * dt * pi". So `Angles` is pre-tanh.
- L79: The kernel names of these tensors have no hit in the doc: `MIMO_V`, `MIMO_O`, `MIMO_Z`, `MIMO_Out`.
  - TIL/mamba3_mimo_fwd.py:100: "MIMO_V (Psi), MIMO_O (Phi), optional MIMO_Z (Zeta)".
  - TIL/mamba3_mimo.py:261: MIMO_Out.
  - The row spells the first one `mimo_x`, so a text search for `MIMO_V` does not find this row.
- L100: In the Mamba-3 kernels, `seq_idx` is the id of one sequence, not a tensor along the sequence axis. TRI/angle_dt.py:56: seq_idx = pid_seq. TRI/mamba3_siso_fwd.py:143 is the same. So "a change of `seq_idx`" applies to the Mamba-2 kernel only.
- L100: The Mamba-2 kernel limits batch 1 to `cu_seqlens`, and only for the varlen states.
  - SSM/ops/triton/ssd_combined.py:390: "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1".
  - SSM/ops/triton/ssd_combined.py:637 gives `seq_idx` a batch axis. So "Our packed row keeps the batch axis" does not separate us from `seq_idx`.
  - Text for L100: `seq_idx` of the Mamba-2 kernel keeps the batch axis.
- L100 negative claim "an index or an offset, not a mark": it holds. `rg -i -w 'reset|is_first|is_start|mask_start'` has no hit in ops/triton or ops/tilelang.
- L103 negative claim "does not use the word trick": it holds. `rg -i trick` has no hit in mamba_ssm, and `rg -i -w trick` has no hit in the checkout. "Phase 1" occurs only at TRI/mamba3_siso_fwd.py:256.
- L104 negative claim "no word for a pathway": it holds. `rg -i pathway` has no hit in the checkout. `rg -i 'single.?ssd|double.?ssd|one.?pass'` has no hit in mamba_ssm or the README. The phrase occurs only at SSM/modules/mamba3.py:425, as "Mamba-3 Combined Kernel States".