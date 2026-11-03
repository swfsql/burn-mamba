Batch `s3-rotation`: 31 citations. 30 confirmed, 1 wrong (L143), 0 not found. Corrections on L128, L129, L130, L131, L143. All rows finished.

As the coordinator told me, the report is in this message and I wrote no report file. Throwaway files: `/tmp/source-terms-check/l128_sign.py` (float64 check) and `/tmp/source-terms-check/k2.log`. I did not change the doc or `src/`.

```text
L128 | `M3/structure/methods_complex.tex:64` | confirmed | 64: \left( \prod_{i=0}^t \bR^\top_i \right) \Delta_t\bB_tx_t. Line 65 applies the same product to C.
L128 | `SSM/modules/mamba3.py:419` | confirmed | 419: angle_dt_state = torch.zeros(
L128 | `TRI/mamba3_siso_combined.py:315` | confirmed | 315: Internally computes cumulative angles: Angles_Cumsum = cumsum(Angles * DT) mod 2π
L128 | `TRI/angle_dt.py:104` | confirmed | 104: chunk_cumsum = tl.cumsum(vals, axis=0). No negation: vals = angle_vals * dt_vals (101), angle_vals = tanh_approx(angle_vals) * PI (94).
L128 | `SSM/modules/mamba3.py:339` | confirmed | 339: conjugate=False, inplace=False. This is the only call in the module (336).
L128 | `MM/mamba3.py:96` | confirmed | 96: cum_angle: Tensor # cumulative RoPE angle Σ Δ_i * θ_i
L128 | `MM/mamba3.py:367` | confirmed | 367: cum_angles = -torch.cumsum(raw_angles, dim=1)
L129 | `M3/structure/methods_complex.tex:23` | confirmed | 23: Diag(A(t) + iθ(t)), with no bound.
L129 | `M3/structure/methods_complex.tex:44` | confirmed | 44: Block({R(Δ_t θ_t[i])}), with no bound.
L129 | `TRI/angle_dt.py:94` | confirmed | 94: angle_vals = tanh_approx(angle_vals) * PI
L129 | `TRI/angle_dt.py:101` | confirmed | 101: vals = angle_vals * dt_vals[:, None]
L129 | `TRI/mamba3_mimo_rotary_step.py:75` | confirmed | 75: # Match angle_dt: tanh(angle_proj) * dt * pi
L129 | `TRI/mamba3_mimo_rotary_step.py:77` | confirmed | 77: angle = angle_state + angle_proj * dt * 3.141592653589793
L129 | `MM/mamba3.py:363` | confirmed | 363: raw_angles = ( dt.unsqueeze(-1) * rearrange(theta, ...) ), a multi-line statement to 366.
L130 | `M3/structure/methods_complex.tex:29` | confirmed | 29: h(t) ∈ C^{N/2}, θ(t) ∈ R^{N/2}
L130 | `SSM/modules/mamba3.py:36` | confirmed | 36: rope_fraction=0.5,
L130 | `SSM/modules/mamba3.py:77` | confirmed | 77: assert rope_fraction in [0.5, 1.0]
L130 | `MM/mamba3.py:276` | confirmed | 276: + args.d_state // 2  # θ
L130 | `MM/mamba3.py:343` | confirmed | 343: args.d_state // 2,   # θ: rotation angles
L131 | `M3/structure/methods_complex.tex:58` | confirmed | 58: a scalar decayed block-diagonal matrix of 2×2 data-dependent rotation matrices
L131 | `M3/structure/methods_complex.tex:91` | confirmed | 91: R_t is the block-diagonal rotation matrix
L131 | `TRI/mamba3_mimo_rotary_step.py:46` | confirmed | 46: ROTATE_PAIRWISE: tl.constexpr, # If true, rotate every pair of dimensions together.
L131 | `TRI/mamba3_mimo_rotary_step.py:279` | confirmed | 279: # Interleaved rotary: pairs are (x0,x1), (x2,x3), ...
L131 | `TRI/mamba3_mimo_rotary_step.py:301` | confirmed | 301: # Halved rotary: split full headdim in half, pairs are (dim_i, dim_{i+D/2})
L131 | `MM/mamba3.py:676` | confirmed | 676: # Split into pairs: even dims (2j) and odd dims (2j+1)
L131 | `MM/mamba3.py:677` | confirmed | 677: x1 = x[..., 0::2]
L131 | `MM/mamba3.py:678` | confirmed | 678: x2 = x[..., 1::2]
L143 | `M3/structure/methods_complex.tex:30` | wrong | 30: this system is equivalent to a real-valued SSM. That SSM keeps R_t in its transition (33), so it is not a form with no rotation. The proof is at M3/structure/methods_trap.tex:133: \alpha_t \vh_{t-1}, with no R.
L143 | `M3/structure/experiments.tex:54` | confirmed | 54: Mamba-3 without RoPE, Mamba-3 with standard RoPE, and Mamba-2 fail to learn these tasks.
L143 | `SSM/modules/mamba3.py:83` | confirmed | 83: assert self.num_rope_angles > 0
L143 | `MM/mamba3.py:271` | confirmed | 271: # θ: d_state // 2 — rotation angles for data-dependent RoPE. The code is at 276 and 343.
```

## Row corrections

### L128

Cell: Papers
Old: Mamba-3: the product of `R^⊤` multiplies `B` (`M3/structure/methods_complex.tex:64`).
New: Mamba-3: the product of `R^⊤` multiplies `B` and `C` (`M3/structure/methods_complex.tex:64`, `M3/structure/methods_complex.tex:65`). `R` has the angle `+Δθ` (`M3/structure/methods_complex.tex:44`), so the net turn is `R(−ΣΔθ)`.
Proof: M3/structure/methods_complex.tex:48-52 defines R(θ) = [[cos, −sin], [sin, cos]]. M3/appendix/complex.tex:97 and :101 define the rotated C and B with R^⊤.

Cell: Official code
Old: The kernel accumulates `+Δθ` (`TRI/angle_dt.py:104`). `mamba3_mimo_rotary_step.py` has a `conjugate` flag, and the module passes `False` (`SSM/modules/mamba3.py:339`).
New: The kernel accumulates `+Δ·π·tanh(θ)` (`TRI/angle_dt.py:94`, `TRI/angle_dt.py:101`, `TRI/angle_dt.py:104`). Forward and step turn `B` and `C` by `R(+angle)` (`TRI/mamba3_siso_fwd.py:314`, `TRI/mamba3_siso_fwd.py:329`, `TIL/mamba3_mimo_fwd.py:256`, `TIL/mamba3_mimo_fwd.py:272`, `TRI/mamba3_mimo_rotary_step.py:104`, `TRI/mamba3_mimo_rotary_step.py:135`). The step has a `conjugate` flag that negates only `sin` (`TRI/mamba3_mimo_rotary_step.py:85`), and the module passes `False` (`SSM/modules/mamba3.py:339`). The forward kernels have no such flag.
Proof: the flag is declared at TRI/mamba3_mimo_rotary_step.py:41 and :160. The angle is stored at :80, before the flip at :85. `rg conj` has no hit in TRI/mamba3_siso_fwd.py or TIL/mamba3_mimo_fwd.py.

Cell: mamba3-minimal
Old: It accumulates `−Δθ` (`MM/mamba3.py:367`).
New: It accumulates `−Δθ` in forward and in step (`MM/mamba3.py:367`, `MM/mamba3.py:542`). `apply_rope` turns `B` and `C` by `R(angle)` (`MM/mamba3.py:684`, `MM/mamba3.py:685`), so the net turn is `R(−ΣΔθ)`, the same as the paper.
Proof: 684: x_rot_even = cos_a * x1 - sin_a * x2. 685: x_rot_odd = sin_a * x1 + cos_a * x2. Numeric check A2: difference 0 against the paper form.

Cell: Our code follows
Old: ?
New: The official code. `angle_increment` gives `+Δ·ϱ·π·tanh(ϑ)`, `helpers::prefix_sum` accumulates it, and `rotate_bc_forward` turns `B` and `C` by `R(+θ)` through `apply_rope_partial`, in forward and in step. The paper and mamba3-minimal turn by `R(−θ)`. With the same weights the outputs differ, and a sign flip of the weights and bias of the angle channels of the in-projection makes them equal. `Quaternion4D` and `Rotor4D` turn by the inverse (`quat_conj`), because a non-abelian rotation has no such flip.
Proof: src/mamba3/rotation/mod.rs:883-884, :1021, :1051, :1062, :1145, :1157. src/mamba3/rotation/rope.rs:83-84, :97-98, :175-176. Forward: src/mamba3/double_ssd/double_ssd/mod.rs:256, src/mamba3/single_ssd/single_ssd/mod.rs:334. Step: src/mamba3/double_ssd/double_ssd/mod.rs:983. Numeric checks A1, A3, G3.

### L129

Cell: Our code follows
Old: ?
New: The official form, with a setting. `angle_increment` computes `Δ·ϱ·π·tanh(ϑ)`, and `ϱ` is `Mamba3Config::rotation_range`. Its default is `2.0` and the official value is `1.0`, so a port of an official model must set `1.0`. The quaternion kinds bound the magnitude of the rotation vector (`bound_rotation_vector`), which no source has.
Proof: src/mamba3/rotation/mod.rs:883, :584, :930. src/mamba3/mamba3.rs:1113-1114.

### L130

Cell: Our code follows
Old: ?
New: The official setting, with another default. `Mamba3Config::rotation_fraction` takes `0.5` or `1.0`, and `Mamba3Config::rotation_dim` rounds down to an even count as the official code does. Our default is `1.0`, the whole state, as in the paper and mamba3-minimal. The official default is `0.5`, so a port of an official model must set `0.5`.
Proof: src/mamba3/mamba3.rs:1052-1053, :1184-1187, :1413. Official rounding: SSM/modules/mamba3.py:79-81.

### L131

Cell: Official code
Old: (`TRI/mamba3_mimo_rotary_step.py:46`, `TRI/mamba3_mimo_rotary_step.py:279`, `TRI/mamba3_mimo_rotary_step.py:301`).
New: (`TRI/mamba3_mimo_rotary_step.py:46`, `TRI/mamba3_mimo_rotary_step.py:279`, `TRI/mamba3_mimo_rotary_step.py:301`). The module selects interleaved pairs for SISO and halved pairs for MIMO (`SSM/modules/mamba3.py:335`).
Proof: SSM/modules/mamba3.py:335: rotate_pairwise = not self.is_mimo. The forward kernels match: TRI/mamba3_siso_fwd.py:313, TIL/mamba3_mimo_fwd.py:249.

Cell: Our code follows
Old: ?
New: The official code. `rotate_bc_forward` sets `rotate_pairwise` to `mimo_rank == 1`, and `apply_rope` and `apply_rope_partial` use it. mamba3-minimal uses interleaved pairs also for MIMO, and ours uses half-and-half pairs there. An official block with `is_mimo=True` and `mimo_rank=1` uses halved pairs, and ours uses interleaved pairs. A fixed permutation of the `B` and `C` channels and biases maps one layout to the other.
Proof: src/mamba3/rotation/mod.rs:1050. src/mamba3/rotation/rope.rs:77, :91, :151-154. The official flag and rank are separate: SSM/modules/mamba3.py:64-67. Numeric checks F1 (difference 3.6e-15) and F2 (21.2 with no permutation).

### L143

Cell: Papers
Old: Mamba-3: a real-valued form, and an ablation "without RoPE" (`M3/structure/methods_complex.tex:30`, `M3/structure/experiments.tex:54`).
New: Mamba-3: a recurrence with no `R` (`M3/structure/methods_trap.tex:133`), and an ablation "without RoPE" (`M3/structure/experiments.tex:54`, `M3/tables/ablations.tex:39`). The "real-valued SSM" of `M3/structure/methods_complex.tex:30` is not this form, because it keeps `R_t` in its transition (`M3/structure/methods_complex.tex:33`).
Proof: M3/structure/methods_complex.tex:33: e^{Δ_t A_t} R_t h_{t-1}. M3/tables/ablations.tex:39: Mamba-3 (w/o RoPE).

Cell: Our code follows
Old: ?
New: The ablation of the paper, as a setting that neither code source has. `RotationKind::Real1D` projects no rotation channels, holds no accumulator (`RotationState::Real`), and passes `B` and `C` unchanged. `Mamba3Config::rotation_fraction` rejects `0.0`. `Real1D` also accepts an odd `state_rank`.
Proof: src/mamba3/rotation/mod.rs:106-115, :1007. src/mamba3/mamba3.rs:1181-1183, :1413-1414.

## Extra findings

- L129, "no bound" in the paper: holds. `rg "tanh|bounded"` over the M3 tex hits only appendix/discretization.tex:76 and :82, on `A`, `B`, `x` and derivatives, not on `θ`. The `\pi` hit at methods_complex.tex:9 is a parity example.
- L129, "no word and no setting" in the official code: holds. `rg -i "bound|rotation_range|angle_range|max_angle"` over the module and the two kernels has no hit. The only related text is the `tanh` comment at TRI/mamba3_mimo_rotary_step.py:75.
- L129, mamba3-minimal: `rg "tanh|clamp|bound"` hits only MM/mamba3.py:765, "Propagate accumulated states across chunk boundaries". This hit is unrelated.
- L131, the paper selects no layout. It writes `B = [B; B̂]` (methods_complex.tex:38), which reads as halves, and `Block(...)` of 2×2 matrices (:44), which reads as adjacent pairs. The proof only says "by grouping pairs of coordinates" (appendix/complex.tex:6). The row cell stays correct.
- L131, "no selector" in mamba3-minimal: holds. `rg -i "pairwise|selector|half"` has no hit.
- L143, "not possible" in the official code: holds. SSM/modules/mamba3.py:77 allows only 0.5 and 1.0, and :83 rejects zero angles.

## For the user

### L128: the three questions

**1. Our signs.** The accumulated angle is `+`, and the rotation of `B` and `C` is `R(+θ)`.
- `angle_increment` gives `+Δ·ϱ·π·tanh(ϑ)` (src/mamba3/rotation/mod.rs:883-884).
- `helpers::prefix_sum` sums it with no negation (mod.rs:1021).
- `rotate_bc_forward` calls `apply_rope_partial` on `B` and `C` (mod.rs:1051, :1062).
- `apply_rope` computes `cos·x0 − sin·x1` and `sin·x0 + cos·x1` (rope.rs:83-84, :97-98, :175-176).
- Forward and step share `rotate_bc_forward`, so they cannot differ.
- I ran `k2_quaternion_matches_production_rotation` (src/mamba3/rotation/tests.rs:698), and it passed. It pins `apply_rope(+Φ)` to left multiplication by `Q`, so it pins the direction of `apply_rope` only. The signs of `angle_increment` and the sum come from reading.

**2. The full convention of each source.** I confirmed all three statements of the ticket.

| Source | Accumulated angle | Matrix applied to `B` and `C` | Net turn | Forward | Step |
|---|---|---|---|---|---|
| Paper | `∏` of `R(+Δθ)` | `R^⊤` | `R(−ΣΔθ)` | methods_complex.tex:64-65, :86-89 | the paper has only the recurrence |
| Official | `+Δ·π·tanh(θ)` | `R(+angle)` | `R(+Σ)` | TRI/angle_dt.py:94-104; TRI/mamba3_siso_fwd.py:314-315, :329-330; TIL/mamba3_mimo_fwd.py:256-257, :272-273 | TRI/mamba3_mimo_rotary_step.py:76-77, :104-105, :135-136; `conjugate=False` at SSM/modules/mamba3.py:339 |
| mamba3-minimal | `−Δθ` | `R(+cum_angle)` | `R(−ΣΔθ)` | MM/mamba3.py:367, :396-397, :450-451, :684-685 | MM/mamba3.py:542, :566-567, :610-611 |
| Ours | `+Δ·ϱ·π·tanh(ϑ)` | `R(+θ)` | `R(+Σ)` | `rotate_bc_forward` | `rotate_bc_forward` |

- The two groups are: official and ours, against the paper and mamba3-minimal.
- Each source uses one direction for `B` and `C`, and one direction for forward and step.
- `conjugate=True` would give the direction of the paper in step only. It negates `sin`, and the stored angle keeps its sign. The forward kernels have no such flag, so that setting would break the official forward and step.
- The wrap differs: official forward to `[0, 2π)`, ours to `[−π, π]`, and none in the official step or in mamba3-minimal. This does not change the output (D3 1.1e-14, D4 3.6e-15).

**3. The effect on outputs.** With the same weights, the outputs differ (A1: 21.8), so a checkpoint cannot load unchanged across the two groups. With a sign flip of the angle channels, the outputs are equal (A3, A4: 0). In this sense the difference is only the sign of the learned angles. The flip does not fix the bound (see the last point below).

The flip absorbs the sign difference under these conditions:

1. The bound is odd. `π·tanh` and the identity are odd, so all our sources meet this. A non-odd bound breaks the flip (B1: 0, B2: 28.4).
2. The flip covers every term of the angle channel. A projection bias must flip with the weights (A5: 0, A6: 18.9). The official and mamba3-minimal `in_proj` have `bias=False` (SSM/modules/mamba3.py:87, MM/mamba3.py:278). Ours has `has_proj_bias`, and the bias flips with the same row. `rg` shows that our rotation channels reach only `rotate_bc_forward`.
3. `B` and `C` turn in the same direction. A mixed direction is not absorbed (C1: 31.6, C2: 21.3 after the flip).
4. Forward and step turn in the same direction. A mixed direction is not absorbed (C3: 25.5, C4: 17.0 after the flip). A consistent one gives 0 (C5, C6).
5. The group is abelian. For a quaternion, `conj(Q(g))` is not `Q(−g)`, because the factors come in the reversed order (G3: 13.1; with one fixed axis, G4: 0). The direction is also not free there. `conj(Q)` telescopes to the direct form (G1: 1.2e-14), and `Q` does not (G2: 18.2). So this flip applies only to `Complex2D`.
6. For trained models to match, the start law and the optimizer must be sign-symmetric. SGD with momentum from a flipped start stays exactly mirrored (H1, H2: 0, with a real loss drop in H3: 6.2). Our `uniform_init` is symmetric (src/mamba3/mamba3.rs:1483-1489). The sources use `nn.Linear`, and I did not check for a later re-init. For Adam and Muon I only reasoned, and did not run them.

The result holds also for both pair layouts and for a partial rotation (D2: 0 for all four cases).

A second route needs no odd bound. Negate the second element of each pair in `B`, `C` and their biases, and keep `θ` (B3: 0). This also maps `R(−θ)` to `R(+θ)`. No source needs this route.

The flip does not make mamba3-minimal equal to the official code, because the bound also differs (B4: 16.8). That is L129.

Scope and confidence: high. The numeric program models the conventions in float64. It is not our Rust block.

### Doc versus code: the header of `src/mamba3/mamba3.rs`, section 2

This is a sign fault in a doc. It does not change any output or capacity.

- Lines 60-61 give the direct form `ρₜ = R(+Δₜ·ϱ·π·tanh(ϑₜ))` and `hₜ = αₜ ρₜ hₜ₋₁ + …`.
- Lines 83-85 give `Rₜ = ρₜ⋯ρ₁` and `B̃ₜ = Rₜ Bₜ`, and say that this form absorbs that direct form.
- The two do not match. A direct form with `ρ = R(+a)` equals `R(−θ)` on `B` and `C`, the form of the paper (E1: 5.8e-15). It differs from our `R(+θ)` (E3: 21.8).
- Our code equals a direct form with `ρ = R(−a)` (E4: 8.9e-15).
- Line 99, `Cᵢᵀ R(θⱼ − θᵢ) Bⱼ`, is correct for the code. It shows the opposite turn for a read after a write.
- The header of `src/mamba3/rotation/mod.rs` (lines 32-36) has the correct form for the quaternion kinds, `B̄ᵢ = Pᵢᵀ Bᵢ`.

Confidence: high. The code matches the official kernels, so the fault looks like it is in the prose only. The user decides which side to change.

### `Complex2D` and the quaternion kinds turn in opposite directions

`Complex2D` applies `R(+θ)`. `Quaternion4D` and `Rotor4D` apply `conj(Q)` (src/mamba3/rotation/mod.rs:1145, :1157). So a raw projection of the same sign turns them in opposite directions. The ladder `Complex2D ⊂ Rotor4D` stays true up to a flip of the generators (A3, G4). I do not think this is a bug. A comparison of weights across these kinds must know it.

### Differences that follow no source

- The default `rotation_range = 2.0` follows no source. Its docs give the reason (src/mamba3/mamba3.rs:1103-1108). This is a choice, not a bug.
- At `mimo_rank = 1`, our pair layout depends on the rank. The official layout depends on `is_mimo`. This affects only an official block with `is_mimo=True` and `mimo_rank=1`.