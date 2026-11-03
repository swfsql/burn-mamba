Batch `s3-paths`: all rows finished. Citations: 42. confirmed 41, wrong 1 (L133 `SSM/modules/mamba3.py:425`), not found 0. Rows with a correction: L132, L133, L134, L135, L136, L144. I wrote no report file, as you instructed. One throwaway script stays at `/tmp/source-terms-check/work-s3-paths/acc.py`.

```text
L132 | `SSM/modules/mamba3.py:433` | confirmed | 433: # K (=B) State. 435 allocates k_state.
L132 | `SSM/modules/mamba3.py:447` | confirmed | 447: # V (=x) State. 448 allocates v_state.
L132 | `MM/mamba3.py:95` | confirmed | 95: prev_Bx: Tensor ... B̄_{t-1} ⊗ x_{t-1} for β term
L133 | `M3/structure/methods_trap.tex:171` | confirmed | 171: a mask L whose structure is a 1-semiseparable matrix composed with a 2-band matrix
L133 | `SSM/modules/mamba3.py:425` | wrong | 425: # Mamba-3 Combined Kernel States, a comment in allocate_inference_cache. The proof is at SSM/modules/mamba3.py:185: y = mamba3_mimo_combined( and SSM/modules/mamba3.py:221: y = mamba3_siso_combined(.
L133 | `TRI/mamba3_siso_fwd.py:285` | confirmed | 285: shifted_gamma = dt_shifted * (1 - trap_shifted)
L133 | `TRI/mamba3_siso_fwd.py:287` | confirmed | 287: scale = shifted_gamma + gamma
L133 | `MM/README.md:61` | confirmed | 61: ## Two-SSD Decomposition
L133 | `MM/README.md:69` | confirmed | 69: **γ SSD** (Current-timestep): ssd(x * γ, dA, B, C)
L133 | `MM/README.md:70` | confirmed | 70: **β SSD** (Previous-timestep): ssd(x_prev * β, dA, B_prev, C)
L133 | `MM/mamba3.py:404` | confirmed | 404: Trapezoidal SSD (two-SSD decomposition, MIMO variant)
L133 | `MM/mamba3.py:453` | confirmed | 453: Trapezoidal SSD (two-SSD decomposition)
L133 | `MM/mamba3.py:723` | confirmed | 723: discretization is handled externally via a two-SSD decomposition
L134 | `TRI/mamba3_siso_fwd.py:423` | confirmed | 423: acc_ssm_states = acc_ssm_states * tl.math.exp2(da_cs_last) + tl.dot(
L134 | `TRI/mamba3_siso_fwd.py:431` | confirmed | 431: acc_ssm_states is the value that the store to final_ssm_state_ptr writes.
L134 | `SSM/modules/mamba3.py:241` | confirmed | 241: ssm_state.copy_(last_state)
L134 | `SSM/modules/mamba3.py:286` | confirmed | 286: def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs). 289: Also modify the state vars in-place. The write is at SSM/ops/cute/mamba3/mamba3_step_fn.py:467.
L134 | `MM/mamba3.py:94` | confirmed | 94: ssm_state: Tensor ... SSM hidden state h_t
L135 | `M3/structure/methods_arch.tex:15` | confirmed | 15: Mamba-3 employs the SISO SSM by default
L135 | `M3/structure/methods_mimo.tex:21` | confirmed | 21: stacking the SISO recurrence
L135 | `M3/appendix/mimo.tex:32` | confirmed | 32: The MIMO variant of Mamba-3 incurs additional parameters compared to its SISO counterpart.
L135 | `SSM/modules/mamba3.py:66` | confirmed | 66: if not self.is_mimo:
L135 | `SSM/modules/mamba3.py:67` | confirmed | 67: self.mimo_rank = 1
L135 | `SSM/modules/mamba3.py:184` | confirmed | 184: if self.is_mimo:
L135 | `SSM/modules/mamba3.py:221` | confirmed | 221: y = mamba3_siso_combined(
L135 | `MM/mamba3.py:70` | confirmed | 70: use_mimo: bool = False  # Toggle MIMO variant
L135 | `MM/mamba3.py:444` | confirmed | 444: SISO path
L135 | `MM/mamba3.py:719` | confirmed | 719: def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
L135 | `MM/mamba3.py:785` | confirmed | 785: def ssd_mimo(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
L136 | `M2/efficient.tex:40` | confirmed | 40: for some block size Q.
L136 | `M2/efficient.tex:105` | confirmed | 105: for small chunk lengths Q
L136 | `M3/structure/methods_mimo.tex:53` | confirmed | 53: divided into chunks of length C
L136 | `M3/structure/methods_mimo.tex:56` | confirmed | 56: (2C^2 N + 2C^2 P) FLOPs per chunk
L136 | `SSM/modules/ssd_minimal.py:34` | confirmed | 34: def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
L136 | `SSM/modules/ssd_minimal.py:47` | confirmed | 47: # Rearrange into blocks/chunks. 48 uses l=block_len.
L136 | `MM/mamba3.py:67` | confirmed | 67: chunk_size: int = 64  # Matrix partition size for chunked SSD (Q)
L136 | `MM/mamba3.py:732` | confirmed | 732: chunk_size: int — partition size Q
L144 | `M2/architecture.tex:134` | confirmed | 134: The number of heads is defined as H = D / P. 133 names D the model dimension.
L144 | `SSM/modules/mamba2.py:81` | confirmed | 81: self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
L144 | `SSM/modules/mamba2.py:85` | confirmed | 85: self.nheads = self.d_ssm // self.headdim
L144 | `SSM/modules/mamba3.py:73` | confirmed | 73: self.nheads = self.d_inner // self.headdim
L144 | `MM/mamba3.py:76` | confirmed | 76: self.nheads = self.d_inner // self.headdim
```

## Row corrections

### L132

Cell: Our code follows
Old: ?
New: The official code. `Mamba3DoubleSsdCache` and `Mamba3SingleSsdCache` keep two tensors, `k_state_bumhr` and `v_state_buhp`, and not the product. Differences: each tensor has a slot axis of `TapPattern::tap_slots` positions, oldest first. Both tensors are `None` under `TapPattern::None`. Above lag 1, `helpers::tail_decay` scales each slot of `v_state_buhp`. At lag 1, the slot is the plain `x`.
Proof: src/mamba3/double_ssd/cache.rs:108, :119. src/mamba3/single_ssd/cache.rs:108, :115. src/mamba3/tap_pattern.rs:332. src/mamba3/helpers.rs:523.

### L133

Cell: Official code
Old: `SSM/modules/mamba3.py:425`
New: `SSM/modules/mamba3.py:185`, `SSM/modules/mamba3.py:221`
Proof: SSM/modules/mamba3.py:221: y = mamba3_siso_combined(

Cell: Our code follows
Old: ?
New: Both, by pathway. `Mamba3::forward_double_ssd` follows mamba3-minimal, with a γ SSD and one β SSD for each tap. `Mamba3::forward_single_ssd` follows the official kernels, with one SSD call. The variant of `Mamba3Cache` selects the pathway, and no cache selects single-SSD. Differences: double-SSD makes two calls by default, three for a two-tap `TapPattern` and one for `TapPattern::None`. A forward with a cache reads its first β inputs from the tap slots. A mamba3-minimal forward pads zeros there, and with a cache it runs only `step`. `Mamba3::step` makes no SSD call on either pathway.
Proof: src/mamba3/double_ssd/double_ssd/mod.rs:19-21, :54, :334, :449, :474. src/mamba3/single_ssd/single_ssd/mod.rs:77, :541. src/mamba3/mamba3.rs:1721. MM/mamba3.py:323, :459.

### L134

Cell: Papers
Old: No citation.
New: Mamba-2: a chunk passes its "final state" `h_{k-1}` (`M2/scan.tex:236`, `M2/scan.tex:261`). Its listing returns `final_state` (`M2/code.py:46`, `M2/code.py:55`). Mamba-3: a run passes "the hidden state" between segments (`M3/appendix/results.tex:183`). No paper has the name `ssm_state`.
Proof: M2/scan.tex:261: directly compute the final state h_{k-1} ... then "pass" the state to the next chunk. rg `ssm.state` in M2 and M3 finds no identifier.

Cell: Official code
Old: `SSM/modules/mamba3.py:286`)
New: `SSM/modules/mamba3.py:286`, `SSM/modules/mamba3.py:370`, `SSM/ops/cute/mamba3/mamba3_step_fn.py:467`)
Proof: mamba3_step_fn.py:467: tSrS_f32[v] = tSrS_f32[v] * alpha_val + new_state[v]. mamba3.py:370: state_out=None. mamba3_step_fn.py:568: updated in place if state_out is None.

Cell: Our code follows
Old: ?
New: Both, by cache type. Our name is `ssm_bhpr`. `Mamba3DoubleSsdCache::ssm_bhpr` holds the state, as in mamba3-minimal. `Mamba3SingleSsdCache::ssm_bhpr` holds the accumulator, as an official forward does. `Mamba3::step` runs `Mamba3::step_double_ssd` for both types, so `step` writes the state, as the official code does. Difference: the official code has one tensor for both values. Our two types stop a mix inside a chunked pass, and `From` converts them at a call boundary.
Proof: src/mamba3/double_ssd/cache.rs:93. src/mamba3/single_ssd/cache.rs:94. src/mamba3/cache.rs:189, :202. src/mamba3/single_ssd/single_ssd/mod.rs:656. src/mamba3/double_ssd/double_ssd/mod.rs:907.

### L135

Cell: Official code
Old: Separate kernels `mamba3_siso_*`, with `mimo_rank = 1` (`SSM/modules/mamba3.py:66`, `SSM/modules/mamba3.py:67`, `SSM/modules/mamba3.py:184`, `SSM/modules/mamba3.py:221`).
New: A forward calls a separate kernel, `mamba3_siso_combined`, with `mimo_rank = 1` (`SSM/modules/mamba3.py:66`, `SSM/modules/mamba3.py:67`, `SSM/modules/mamba3.py:184`, `SSM/modules/mamba3.py:221`). `step` calls `mamba3_step_fn` in both forms (`SSM/modules/mamba3.py:357`, `SSM/modules/mamba3.py:380`). Only the tests call `mamba3_siso_step` (`TRI/mamba3_siso_step.py:233`).
Proof: the two `mamba3_step_fn` calls branch on is_outproj_norm, not on is_mimo. SSM/modules/mamba3.py:22 imports it. `rg -l mamba3_siso_step` finds only tests/ops/triton/test_mamba3_siso.py and the definition file.

Cell: Our code follows
Old: ?
New: No source for the structure (ADR-0004). SISO is `Mamba3Config::mimo_rank` = 1 of the one implementation, and 1 is the default. At rank 1, `Mamba3::mimo_x_hmp`, `Mamba3::mimo_z_hmp` and `Mamba3::mimo_o_hmp` are `None`, and `rotate_bc_forward` uses interleaved pairs, as the official SISO form does. `Mamba3Config::siso_specialization` and `Mamba3Config::siso_specialization_decode` change only the speed. Difference: official `is_mimo=True` with `mimo_rank=1` and mamba3-minimal `use_mimo=True` with `mimo_rank=1` use the MIMO path at rank 1. The official form also uses halved pairs. Neither has an equal here.
Proof: src/mamba3/mamba3.rs:304, :931, :1133, :1153, :1538. src/mamba3/rotation/mod.rs:1050: let rotate_pairwise = mimo_rank == 1. SSM/modules/mamba3.py:335: rotate_pairwise = not self.is_mimo.

### L136

Cell: Papers
Old: Mamba-2: `Q` (`M2/efficient.tex:40`, `M2/efficient.tex:105`).
New: Mamba-2: `Q` (`M2/efficient.tex:40`, `M2/efficient.tex:105`), and `block_len` in its listing (`M2/code.py:11`).
Proof: M2/code.py:11: def ssd(X, A, B, C, block_len=64, initial_states=None):. M2/efficient.tex:152 inputs structure/code.py.

Cell: Official code
Old: `chunk_size`. `block_len` in the Listing 1 form (`SSM/modules/ssd_minimal.py:34`, `SSM/modules/ssd_minimal.py:47`).
New: `chunk_size` (`SSM/modules/mamba2.py:59`, `SSM/modules/mamba3.py:46`). `block_len` in the Listing 1 form (`SSM/modules/ssd_minimal.py:4`, `SSM/modules/ssd_minimal.py:34`, `SSM/modules/ssd_minimal.py:47`).
Proof: SSM/modules/mamba2.py:59: chunk_size=256. SSM/modules/mamba3.py:46: chunk_size=64, # Recommended: 64 for SISO, 64/mimo_rank for MIMO. SSM/modules/ssd_minimal.py:4: This is the same as Listing 1 from the paper.

Cell: Our code follows
Old: ?
New: No source for the name or the default. `Mamba2SsdPath::chunk_len` and `Mamba3SsdPath::chunk_len` give the length. `None` selects `optimal_chunk_len`, about `√(state_rank·per_head_dim)`, rounded up to a multiple of 32. The source defaults are fixed values: 256 for official Mamba-2, and 64 for official Mamba-3 and mamba3-minimal. Differences: under MambaProduct, `chunk_len` counts write positions, and `Mamba3SsdPath::chunk_tokens` gives the tokens. A sequence can have any length, but the Listing 1 form asserts that `block_len` divides it.
Proof: src/mamba2/ssd/ssd_path.rs:123, :131, :180. src/mamba3/ssd_path.rs:88, :115, :126, :194. Padding: src/mamba2/mamba2.rs:944, src/mamba3/double_ssd/double_ssd/mod.rs:362. Assert: SSM/modules/ssd_minimal.py:45.

### L144

Cell: Our code follows
Old: ?
New: Official Mamba-3, mamba3-minimal and the default of official Mamba-2. `Mamba2Config::nheads` and `Mamba3Config::nheads` return `d_inner / per_head_dim`. Both configs assert that `per_head_dim` divides `d_inner`. Differences: our code has no `d_ssm`, so the whole `d_inner` goes through the plant. The Mamba-2 paper divides `D`, not `d_inner`, so at the same `P` its count is `1/expand` of ours.
Proof: src/mamba2/mamba2.rs:436, :504, :903. src/mamba3/mamba3.rs:1172, :1405.

## Extra findings

- L132, Papers "No citation": the cell stays. I searched M3 (`structure`, `appendix`, `tables`) for `cache`, `k_state`, `v_state`, `prev`, `B_{t-1}`, `store`, `memory`, `decod`, `keep` and `retain`. The only cache statement is the "KV cache" of Transformers (`M3/structure/intro.tex:6`). `M3/structure/methods_trap.tex:134` writes the term `β_t B_{t-1} x_{t-1}` and says nothing about a cache. The input list of `M3/tables/mimo_arith_intensity.tex:24` has `H_t, x_t, a_t, b_t, c_t` and no tap.
- L133, optional citation: `M3/appendix/discretization.tex:70` reads the 2-band factor as "a convolution of size two applied on the state-input" before the `L_1` scan.
- L134, accumulator: `TRI/mamba3_siso_fwd.py:352` seeds the accumulator from `k_state` and `v_state`. A scalar copy of the kernel arithmetic (`/tmp/source-terms-check/work-s3-paths/acc.py`, 12 steps, chunk 4) gives final accumulator minus state = 0.0. At the two inner chunk ends the difference is -0.393 and 0.259. This supports "equal only at call boundaries" for the SISO form. The script does not run the real kernel, because torch is not available.
- L134, MIMO: `SSM/modules/mamba3.py:209` has the same `ssm_state.copy_(last_state)`. I did not read the TIL kernel, so the row's accumulator claim is checked only for SISO.
- L135, negative test: rg `R\s*=\s*1|rank.?1|R=1` in M3 finds nothing. So the paper does not call SISO the rank-1 form, and "its own default form" holds. `M3/appendix/results.tex:160` and `:164` list the same CuTe decode kernel for SISO and MIMO, which agrees with the L135 correction.
- L136: M3 also uses "chunk size" (`M3/structure/methods_mimo.tex:55`, `:60`). `:60` gives `C_MIMO = C_SISO / R`, which matches the `mimo_rank` divisor in `Mamba3SsdPath::optimal_chunk_len`.
- L144, negative test: "Mamba-3: no symbol" holds. In M3, `H` is the state only (`M3/appendix/discretization.tex:67`, `M3/tables/mimo_arith_intensity.tex:24`). `docs/source-terms.md` lines 161 and 183 already say this.

## For the user

No difference in this batch looks like a bug. These cases follow no source. Confidence that each is intended is high, from the ADR or module docs that I cite.

- L135: one implementation for SISO and MIMO follows no source. ADR-0004 records this choice. A source configuration with the MIMO flag on and rank 1 cannot port directly, because our rank 1 is always the SISO form. Sides: src/mamba3/rotation/mod.rs:1050, SSM/modules/mamba3.py:335.
- L136: our default chunk length follows no source default. At `state_rank` 128 and `per_head_dim` 64, our value is 96 and the official Mamba-3 value is 64. The SSD paths are documented as exact reformulations, so I expect this to change only speed and memory. I did not run a comparison. Sides: src/mamba3/ssd_path.rs:88, SSM/modules/mamba3.py:46.
- L132: no source has a tap slot axis or a scale on the `x` slots, because no source has a lag above 1. At the default lag 1, the fields match the official `k_state` and `v_state` plus one slot axis. Sides: src/mamba3/double_ssd/cache.rs:108, :119, SSM/modules/mamba3.py:433, :447.