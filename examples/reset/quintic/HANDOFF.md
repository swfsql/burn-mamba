# reset-quintic + reflections for Mamba-3 — handoff

Self-contained state for a fresh session. Replaces the earlier reset-quintic handoff.
It is a work log: delete it before the final commit (CLAUDE.md keeps changelogs out
of the docs proper).

Two threads:

- **A. `reset-quintic`** (committed in `6a4e6ef`): A₅ in one Mamba-3 block, S₅ in two.
- **B. Reflections** (this session, all under the gitignored `tmp/reflections/`).
  Nothing in `src/` changed. It has three parts:
  - *why* S₅ needs two layers,
  - a way for **one** layer to hold it (an SO(5) block),
  - a PyTorch probe in which one layer **learned** S₅ exactly.

---

## A. reset-quintic (state unchanged since `6a4e6ef`)

### The task

The reset stream over five items: two turns and a reset `R`. The label at every
position is the running arrangement.

- `--group s5` (default): `s = (0 1)`, `c = (0 1 2 3 4)`, 120 classes.
- `--group a5`: `d = (0 1)(2 3)`, `t = (0 2 4)`, 60 classes.

The families are `random` / `shuffle` / `runs`. The words are 96 symbols long. The
numbers below are in the order random/shuffle/runs.

### Results

- **A₅, one layer.**
  - Hand-built: exact at the config of `reset-swap` (`d_model 2`, 2 heads,
    `state_rank 4`, `TapPattern::None`, no final norm, 60-way head), with 308 params.
    Conjugation `v ↦ q v q̄` in a `Rotor4D` 4-block is SO(3), the icosahedron.
  - Trained: exact at `d_model 4`, 2 heads, `per_head_dim 2`, `mimo_rank 2`, len 96
    (608 params). 96 → 100/100/100. 288 → 100/99.5/99.3 (seed 1: 100/91.4/91.3).
  - The quaternion/complex ablations get ≈ 66/14/43 @96.
- **S₅, one layer: impossible** (any size, kind, `mimo_rank`, `micro_steps`). See B§1.
- **S₅, two layers.**
  - Hand-built: exact (`d_model 3`, 3 heads, 902 params) at 96 and 288. Layer 1 holds
    the sign. Layer 2 steps the even part by `c` or `s∘c∘s` (Krohn–Rhodes,
    `S₅ = A₅ ⋊ C₂`).
  - **Training never found it.** The best run is `s5-d4h2m2-l2-len96` (1200 params):
    81.6/36.1/54.0 @96, 76.3/12.7/23.4 @288. One layer (908 params) gets
    78.0/14.0/33.6.
  - Tried: seeds, `max_lr` 1e-2, 240 epochs, 3 layers, `d_model 8`, four one-channel
    heads, len 32 or 96. All stall. Layer 1 learns `c` as a ~72° turn and never the
    sign.

### Findings worth keeping

1. Training lands the group on **one head**. That head has one scalar readout ⇒ not
   exact. `mimo_rank 2` gives it two readouts. That is why the trained width is wider
   than the hand-built floor.
2. Training often finds **A₅'s 4-D irrep** (left/right quaternion factors in the two
   Galois-twin 2I's, axes 69.1° vs 20.9°), not conjugation. This is exactly `std`
   restricted to A₅: the embedding that extends to S₅ in SO(5) (B§3).
3. **Length leak**: trained on 32 symbols, exact angles but decay ᾱ ≈ 0.99 ⇒ 56% on
   `shuffle` @288. Training at 96 pays for it. (The probe in B hit the same thing.)
4. Hand-built needs a scale-free head (direction decoder, no bias, `REF_POINT`
   `(5,−5,2)`). S₅ embeds `c` as the zero vector, so the decaying sign of layer 1 is
   a pure direction after the pre-norm.

### Files and surface

- Files:
  - `examples/reset/quintic/{main,dataset,model,training,inference,tests}.rs`,
  - `Cargo.toml` `[[example]] reset-quintic`,
  - `examples/reset/README.md` `## reset-quintic` (task, "Why this task", "Measured",
    "Notes"),
  - the `examples/README.md` entry.
- CLI after `--`: `--group a5|s5`, `--rotation complex|quaternion|rotor`, `--layers`,
  `--d-model`, `--heads`, `--mimo-rank`, `--expand`, `--train-length` (default 96).
  `model::DEFAULT_WIDTH = (4, 2, 2)`, `model::floor_width` (hand-built floors).
- Tests: `cargo test --release --example reset-quintic -- --test-threads=1`. They cover
  hand-built A₅/S₅ at 96+288, the left-isoclinic twin,
  `s5_needs_a_reflection_one_layer_does_not_have`, the ceilings and the labels.
  Ignored: `learned_rotations` (`QUINTIC_ARTIFACTS=<run dir>`).
- Run artifacts: `examples/reset/quintic/tmp/<dir>` (gitignored, each with `log.txt`
  and `training_config.json`). The git history of this file has the table of all runs
  (`git show 6a4e6ef:examples/reset/quintic/HANDOFF.md`).

### Open (A)

- Only one run uses `--expand`. Keep it or drop it.
- README polish:
  - the place of `reset-quintic` in the ladder intro,
  - possibly the B§1 sharpenings (−1 eigenvalues ≠ reflections, `I_h ≇ S₅`, the SO(5)
    route).
- S₅ two-layer trainability. B deprioritises it: one SO(5) layer may be the better
  road than coaxing the cascade.

---

## B. Reflections for Mamba-3

Files: `tmp/reflections/notes.md` (fuller notes, § numbers match `check.py`),
`tmp/reflections/check.py` (numpy, float64, 24 checks, a few minutes),
`tmp/reflections/train_s5.py` (PyTorch probe), `tmp/reflections/runs/` (artifacts).

### §1 Why one layer cannot hold S₅: orientation, not negative eigenvalues

- **Negative eigenvalues** (Grazzi et al., arXiv:2411.12537v5, `main.tex` :395–411).
  For real-diagonal (Mamba) and one generalized Householder (DeltaNet), widening
  [0,1] → [−1,1] lets one eigenvalue hit −1. For those classes, that is a reflection
  (det −1). Parity needs no more.
- **Mamba-3 already has −1 eigenvalues**: a half-turn has (−1, −1), det +1 (how
  Complex2D does parity). Every transition is `α·R`, `α > 0`, `R ∈ ∏SO(2|4)` ⇒ det > 0.
- **S₅ ⊂ O(4), S₅ ⊄ SO(4).** No 2-/3-D irreps (dims 1,1,4,4,5,5,6), so the only
  faithful 4-D reps are `std` and `std⊗sign`. A transposition has det −1 in both
  (check §1). A claim seen in an earlier *web* session that "S₅ is solvable by SO(4)"
  is **wrong**. It is O(4): the 4 Householders of DeltaProduct multiply into O(4).

  Picture: S₅ is the symmetry group of the regular 5-cell. Its rotations are A₅, and
  swaps are mirrors.
- Not even as a quotient of a subgroup of ∏SO(4). The argument is in `tests.rs` (the
  doc of `s5_needs_a_reflection_…`) and in README "Why this task". The finite
  single-block case, re-derived:
  1. Lift to SU(2)×SU(2).
  2. An A₅ image needs a whole 2I in one factor.
  3. 2I is self-normalising in SU(2), so everything else commutes with it and maps
     into `C_{S₅}(A₅) = 1`.
- **Parity of dimension** (`⊗sign` multiplies det by (−1)^dim): S₃ ⊂ SO(3) (the
  flip-the-triangle of reset-swap), S₄ ⊂ SO(3), **S₅ ⊂ SO(5)**, S₆ ⊂ SO(5).
  Rotor4D = SO(4) ⊇ O(3): everything through S₄. S₅ is the first that needs a fifth
  axis.
- **Two independent obstructions**:
  - Non-solvability (A₅ is the smallest non-solvable group, order 60) excludes
    abelian/diagonal transitions and stacks of them. Rotor4D clears it.
  - Orientation (the *action* of the sign on A₅, the outer automorphism that swaps
    the 72°/144° five-cycles) excludes one SO(4) layer.
- **Grazzi's Mamba [−1,1]** (`2s−1`, scalar) gives −I. It is central ⇒ no outer
  automorphism. It is also already in SO on even blocks. Their Mamba change needed
  CUDA scan edits and breaks log-space cumprods (:722, :1486–1488).
- **DeltaProduct** (`../papers/deltanet/deltaproduct/neurips_2024.tex`):
  - Two reflections = a rotation (:322–328).
  - S₄/A₅ at `n_h=2` via SO(3). S₅ needs `n_h=4` (:465–468).
  - With `n_h=1`, even 10 layers are not enough to learn S₅ (:603).
  - The 2-layer dihedral theorem (:1029) is the mirror image of the cascade of
    quintic. It emulates rotations with reflections. Quintic emulates the effect of
    the reflection with rotations.
  - **Error** at :1281: the full dodecahedral group is `I_h ≅ A₅×C₂`, not S₅. The
    inversion fixes all five cubes, and S₅ has a trivial centre.
- In the terms of `info/mamba-3/rotation-as-optimization.md`, a reflection is **real
  step × rank-one curvature** (`α·H = (1−α)I + 2α·kkᵀ`). Isotropic curvature has no
  non-central orientation reversal under any step algebra (the table of §8). Hence
  "not a one-line change".

### §2–3 The way out: O(4) ↪ SO(5)

- The move parity already made: O(1) ↪ SO(2) *is* Complex2D (−1 reached around the
  circle, not through 0). For S₅: `T ↦ diag(T, det T)`. A transposition becomes a
  half-turn (eigenvalues −1,−1,1,1,1).
- **Dilation**: a rotation by θ in the plane (k, e₅), compressed to ℝ⁴, is exactly
  DeltaNet's `I − (1−cosθ)kkᵀ`, β ∈ [0,2] (check §3). The rotation parks the erased
  component in e₅ and does not destroy it. At β ∈ {0,2} they coincide, and e₅ carries
  the sign: layer 1 of the cascade, folded into the block. A rank-one factor becomes
  an isometry one dimension up, which the RoPE trick absorbs.

### §4–6 A concrete kind: "Rotor4D + tilt"

- Spin(5) ≅ Sp(2) (2×2 unit quaternion matrices), acting on ℝ⁵ by `X ↦ U X U*`,
  `X = [[a, b], [b̄, −a]]`, a ∈ ℝ, b ∈ ℍ. `diag(q,p)` acts as Rotor4D on b and fixes a.
  `tilt(θ) = [[cosθ, −sinθ],[sinθ, cosθ]]` rotates (a, b₀) by 2θ (check §4).
- Per step, `diag(q,p)·tilt(θ)` = the channels of Rotor4D + **one angle** per block.
  It reaches all of O(4) ↪ SO(5) with θ ∈ {0, π/2} (check §5), and its products
  generate SO(5). A hand-built one-block Sp(2) scan tracks S₅ exactly over
  1000-symbol words (check §6).
- **Naming caveat**: *not* the 5-D analogue of Rotor4D. One step of Rotor4D is all of
  SO(4) (6 = dim). For Quaternion4D, it is SU(2) (3). One step of Rotor4D+tilt is
  a 7-dim subset of SO(5) (dim 10). It generates the same group with a smaller
  per-step set, and that set is enough for the generators of S₅. Full SO(5) per step
  = Cartan form `diag(q,p)·tilt(θ)·diag(q',p')` (13 params, redundant) or `exp` on
  sp(2).
- **The "8-D ceiling" in the `src/mamba3/rotation/mod.rs` header** (SO(4) is the
  ceiling for k=4, and at k=8 "the scan itself breaks") is about the
  *representation*. Complex2D/Quaternion4D store one number per block, which needs an
  associative normed division algebra (dims 1,2,4: the octonions are
  non-associative). SO(8) itself scans fine as matrices. 5-D has no division algebra
  at all. Sp(2) is a *matrix over ℍ*, so it is associative. That sentence could say
  so.
- Crate cost (if built):
  - blocks of 5 (`state_rank` multiple of 5),
  - +1 in-proj channel per (head, block),
  - a scan over 2×2 quaternion matrices (8 `quat_mul` per combine, recompute-backward
    like `quat_scan/`),
  - unitarity renormalisation (Gram–Schmidt on two quaternionic columns),
  - a new `RotationState` accumulator,
  - a `muon_projections` sub-projection.

### §7 The cheap alternative (worse)

Signed decay picks orientation: `M = |a|·T·κ^{[a<0]}` (κ = quaternion conjugation).
|a| stays in the log-space segsum. The bit rides the quaternion scan as a conditional
swap `(ε₁⊕ε₂, q₁·(ε₁ ? p₂ : q₂), p₁·(ε₁ ? q₂ : p₂))` (check §7, associative). But the
two components of O(4) meet only at the zero matrix ⇒ every flip crosses a full reset
of the head. In the probe (`o4sign`), the sign never crossed zero.

### Experiments: `tmp/reflections/train_s5.py`

**Probe, not the crate.**

- Model: one linear-recurrent layer, 4 heads, `h ← M[tok]·h + b[tok]` with
  `M = decay × rotation`. The transitions/writes are per-token lookups (this isolates
  the group from the in-projection).
- Readout: per-head normalised states → concat → one linear layer.
- Decay: `sigmoid(a_raw)` (init 0.95).
- Optimiser: Adam, lr 3e-3, warmup 200, cosine to 0.1×, clip 1.0, 6000 steps.
- Eval: train distribution @64 (with resets), reset-free words @256 and @1024
  ("tail" = last quarter).

| kind | per-step transition | n |
|---|---|---|
| `rotor4` | `L_q R_p̄` (Rotor4D group) | 4 |
| `so4` | `matrix_exp(W−Wᵀ)` | 4 |
| `rotor5` | `(L_q R_p̄ ⊕ 1)·G(2·tilt)` on (a, b₀) — Rotor4D + tilt | 5 |
| `so5` | `matrix_exp(W−Wᵀ)` — full SO(5) per step | 5 |
| `o4sign` | `|a|·T·K^{[a<0]}`, `a = tanh σ` (§7) | 4 |
| `o4hh` | `α·T·diag(tanh s, 1, 1, 1)` — non-orthogonal reference | 4 |

Control: A₅ on `rotor4`. 3 seeds each. ~22 min/run at len 192 on one CPU thread.
21 runs with `--procs 10` ≈ 45 min (12 cores).

- **sweep-1** (`--length 64 --p-reset 0.05`): nothing length-generalises, control
  included (A₅ 100% @64, ~45% @256). Cause: the leak. The A₅ head had exact
  180°/120°, writes ~2e-4 and decay 0.92, and the training segments averaged 15.5
  symbols. The transitions were still informative. `so5` s0/s2 put `std⊕sign` on one
  head (s ≈ half-turn + 2.5° residual, c = 72°/144°). The SO(4) kinds only learned c
  as a lone 72° turn.
- **sweep-2-len192** (`--length 192 --p-reset 0.005 --batch 96 --procs 10`), @256 (all):

  | kind | s0 | s1 | s2 |
  |---|---|---|---|
  | `rotor4` | 6.8 | 7.9 | 6.7 |
  | `so4` | 6.7 | 6.4 | 9.6 |
  | **`rotor5`** | 7.6 | **98.0** (tail 99.5) | 6.8 |
  | `so5` | 7.0 | 9.3 | 12.5 |
  | `o4sign` | 6.9 | 6.7 | 7.0 |
  | `o4hh` | 7.0 | 7.9 | 7.9 |
  | A₅ control `rotor4` | 99.9 | 99.8 | 100.0 |

  All fall at @1024 (A₅ control 33–45%, `rotor5` s1 27.5%).
- **`s5-rotor5-s1`, head 0** is exactly `std⊕sign`: s eigen-angles (0,0,0,180,180),
  c (0,72,72,144,144), decays 0.974 / 0.968. Found late: @256 3% at step 2000 → 16% at
  2500 → 60% at 4000 → 96% at 5000.
- **Head-0 decode** (inline script, not saved, recreate it as below): nearest centroid
  on the normalised state of head 0. The centroids are fit on 256 reset-free
  48-symbol words (min gap 0.618).
  - As trained: 99.97% @256, 27.8% @1024, 7.6% @4096.
  - With **decay 1 and zero writes on s/c** (`a_raw[1:] = 30`, `b[1:] = 0`):
    **100.00% @256, @1024 and @4096**.

  So the learned rotations *are* S₅. The long-length failure is the leak of the probe
  plus the mixing of the three non-group heads by the linear head.

Loading a probe model:

```python
import sys, json, torch, numpy as np
sys.path.insert(0, 'tmp/reflections'); import train_s5 as T
d = 'tmp/reflections/runs/sweep-2-len192/s5-rotor5-s1'
cfg = json.load(open(d + '/result.json'))['cfg']        # also 'final', 'history', 'diagnose'
task = T.Task(cfg['group'], cfg['p_reset'])
m = T.Layer(cfg['kind'], cfg['heads'], task.n_classes, cfg['init_scale'])
m.load_state_dict(torch.load(d + '/model.pt'))
tok, lab = task.sample(np.random.default_rng(7), 64, 1024, resets=False)
```

**Caveats.** 1/3 seeds (`rotor5`) vs 0/3 (`so5`) does not rank the two kinds. They also
differ in parametrisation. Per-token lookups are easier than the input projections of
the crate. Exactness at length needed a hand edit that stops the leak.

### Next steps (B), in order

1. **Harden the probe** (cheap):
   - more seeds for `rotor5` vs `so5` (+ a full-Sp(2) per-step variant),
   - train longer / len 384–512 or a curriculum, so that training pays for the leak,
   - a readout per head (or MIMO-style several readouts of the state of one head), so
     that one group head suffices.

   Target: exact @1024 without surgery.
2. **Crate prototype** of Rotor4D+tilt as a `RotationKind` (see the costs in B§4–6,
   and the module headers of `src/mamba3/rotation/` and `src/mamba3/quat_scan/`).
   Forward/step/gradient parity tests like the other kinds.
3. **reset-quintic with it**: a one-layer S₅ run (would replace the two-layer story in
   the README if it trains).
4. Docs, only if wanted:
   - an `info/` note (reflections as dilation), with `check.py` moved to `scripts/`,
   - the 8-D sentence of the `rotation/mod.rs` header,
   - README sharpenings (A open items).

---

## Process reminders

- Edit crate files only with Edit/Write (no `sed -i`, heredocs, python writes).
- Never `rm` artifacts. Use a fresh run directory per run (`train_s5.py --out` refuses
  an existing dir).
- CPU for these probes. GPU runs: `default,backend-cuda`, VRAM ≤ 3.6 GB, one big run at
  a time. The user runs benches.
- No `src/` change yet ⇒ no File Map update pending. If step B2 adds, deletes or renames
  a file, update its File Map line in the same change (CLAUDE.md rule).
- Commit messages: write the text only (title + short body + `Co-Authored-By:`
  trailer). Do not run git commit.
