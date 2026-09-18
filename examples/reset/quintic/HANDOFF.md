# reset-quintic + reflections for Mamba-3 — handoff

Self-contained state for a fresh session. Replaces the earlier reset-quintic handoff.
It is a work log: delete it before the final commit (CLAUDE.md keeps changelogs out
of the docs proper).

Two threads:

- **A. `reset-quintic`** (committed in `6a4e6ef`): A₅ in one Mamba-3 block, S₅ in two.
- **B. Reflections** (this session, all under the gitignored `tmp/reflections/`):
  *why* S₅ needs two layers, a way for **one** layer to hold it (an SO(5) block), and
  a PyTorch probe in which one layer **learned** S₅ exactly. Nothing in `src/` changed.

---

## A. reset-quintic (state unchanged since `6a4e6ef`)

### The task

The reset stream over five items: two turns and a reset `R`; the label at every
position is the running arrangement. `--group s5` (default): `s = (0 1)`,
`c = (0 1 2 3 4)`, 120 classes. `--group a5`: `d = (0 1)(2 3)`, `t = (0 2 4)`, 60
classes. Families `random` / `shuffle` / `runs`; words 96 symbols (numbers below are
random/shuffle/runs).

### Results

- **A₅, one layer.** Hand-built exact at `reset-swap`'s config (`d_model 2`, 2 heads,
  `state_rank 4`, `Trapezoid::None`, no final norm, 60-way head) — 308 params:
  conjugation `v ↦ q v q̄` in a `Rotor4D` 4-block is SO(3), the icosahedron.
  Trains exact at `d_model 4`, 2 heads, `per_head_dim 2`, `mimo_rank 2`, len 96
  (608 params): 96 → 100/100/100; 288 → 100/99.5/99.3 (seed 1: 100/91.4/91.3).
  Quaternion/complex ablations ≈ 66/14/43 @96.
- **S₅, one layer: impossible** (any size, kind, `mimo_rank`, `micro_steps`) — see B§1.
- **S₅, two layers.** Hand-built exact (`d_model 3`, 3 heads, 902 params) at 96 and
  288: layer 1 holds the sign, layer 2 steps the even part by `c` or `s∘c∘s`
  (Krohn–Rhodes, `S₅ = A₅ ⋊ C₂`). **Training never found it**: best
  `s5-d4h2m2-l2-len96` (1200 params) 81.6/36.1/54.0 @96, 76.3/12.7/23.4 @288;
  one layer (908 params) 78.0/14.0/33.6. Tried: seeds, `max_lr` 1e-2, 240 epochs,
  3 layers, `d_model 8`, four one-channel heads, len 32 or 96 — all stall. Layer 1
  learns `c` as a ~72° turn and never the sign.

### Findings worth keeping

1. Training lands the group on **one head**; that head has one scalar readout ⇒ not
   exact. `mimo_rank 2` gives it two readouts — why the trained width is wider than
   the hand-built floor.
2. Training often finds **A₅'s 4-D irrep** (left/right quaternion factors in the two
   Galois-twin 2I's, axes 69.1° vs 20.9°), not conjugation. This is exactly `std`
   restricted to A₅ — the embedding that extends to S₅ in SO(5) (B§3).
3. **Length leak**: trained on 32 symbols, exact angles but decay ᾱ ≈ 0.99 ⇒ 56% on
   `shuffle` @288. Training at 96 prices it in. (The probe in B hit the same thing.)
4. Hand-built needs a scale-free head (direction decoder, no bias; `REF_POINT`
   `(5,−5,2)`); S₅ embeds `c` as the zero vector so layer 1's decaying sign is a pure
   direction after the pre-norm.

### Files and surface

- `examples/reset/quintic/{main,dataset,model,training,inference,tests}.rs`;
  `Cargo.toml` `[[example]] reset-quintic`; `examples/reset/README.md` `## reset-quintic`
  (task, "Why this task", "Measured", "Notes"); `examples/README.md` entry.
- CLI after `--`: `--group a5|s5`, `--rotation complex|quaternion|rotor`, `--layers`,
  `--d-model`, `--heads`, `--mimo-rank`, `--expand`, `--train-length` (default 96).
  `model::DEFAULT_WIDTH = (4, 2, 2)`, `model::floor_width` (hand-built floors).
- Tests: `cargo test --release --example reset-quintic -- --test-threads=1` — hand-built
  A₅/S₅ at 96+288, left-isoclinic twin, `s5_needs_a_reflection_one_layer_does_not_have`,
  ceilings, labels; ignored `learned_rotations` (`QUINTIC_ARTIFACTS=<run dir>`).
- Run artifacts: `examples/reset/quintic/tmp/<dir>` (gitignored; each has `log.txt`,
  `training_config.json`). Table of all runs is in git history of this file
  (`git show 6a4e6ef:examples/reset/quintic/HANDOFF.md`).

### Open (A)

- `--expand` is used by one run only — keep or drop.
- README polish: `reset-quintic`'s place in the ladder intro; possibly the B§1
  sharpenings (−1 eigenvalues ≠ reflections; `I_h ≇ S₅`; the SO(5) route).
- S₅ two-layer trainability — deprioritised by B: one SO(5) layer may be the better
  road than coaxing the cascade.

---

## B. Reflections for Mamba-3

Files: `tmp/reflections/notes.md` (fuller notes, § numbers match `check.py`),
`tmp/reflections/check.py` (numpy, float64, 24 checks, a few minutes),
`tmp/reflections/train_s5.py` (PyTorch probe), `tmp/reflections/runs/` (artifacts).

### §1 Why one layer cannot hold S₅: orientation, not negative eigenvalues

- **Negative eigenvalues** (Grazzi et al., `/shared/claude/papers/deltanet/state-tracking/main.tex`
  :395–411): for real-diagonal (Mamba) and one generalized Householder (DeltaNet),
  widening [0,1] → [−1,1] lets one eigenvalue hit −1 — for those classes a
  reflection (det −1). Parity needs no more.
- **Mamba-3 already has −1 eigenvalues**: a half-turn has (−1, −1), det +1 (how
  Complex2D does parity). Every transition is `α·R`, `α > 0`, `R ∈ ∏SO(2|4)` ⇒ det > 0.
- **S₅ ⊂ O(4), S₅ ⊄ SO(4).** No 2-/3-D irreps (dims 1,1,4,4,5,5,6), so the only faithful
  4-D reps are `std` and `std⊗sign`; a transposition has det −1 in both (check §1).
  A claim seen in an earlier *web* session that "S₅ is solvable by SO(4)" is **wrong**
  (it is O(4); DeltaProduct's 4 Householders multiply into O(4)). Picture: S₅ is the
  symmetry group of the regular 5-cell; its rotations are A₅; swaps are mirrors.
- Not even as a quotient of a subgroup of ∏SO(4) — the argument in `tests.rs`
  (`s5_needs_a_reflection_…` doc) and README "Why this task". Finite single-block case
  re-derived: lift to SU(2)×SU(2); an A₅ image needs a whole 2I in one factor; 2I is
  self-normalising in SU(2), so everything else commutes with it and maps into
  `C_{S₅}(A₅) = 1`.
- **Parity of dimension** (`⊗sign` multiplies det by (−1)^dim): S₃ ⊂ SO(3) (reset-swap's
  flip-the-triangle), S₄ ⊂ SO(3), **S₅ ⊂ SO(5)**, S₆ ⊂ SO(5). Rotor4D = SO(4) ⊇ O(3):
  everything through S₄; S₅ is the first that needs a fifth axis.
- **Two independent obstructions**: non-solvability (A₅ is the smallest non-solvable
  group, order 60) rules out abelian/diagonal transitions and stacks of them — Rotor4D
  clears it; orientation (the sign's *action* on A₅, the outer automorphism swapping the
  72°/144° five-cycles) rules out one SO(4) layer.
- **Grazzi's Mamba [−1,1]** (`2s−1`, scalar) gives −I: central ⇒ no outer automorphism;
  already in SO on even blocks. Their Mamba change needed CUDA scan edits and breaks
  log-space cumprods (:722, :1486–1488).
- **DeltaProduct** (`/shared/claude/papers/deltanet/deltaproduct/neurips_2024.tex`):
  two reflections = a rotation (:322–328); S₄/A₅ at `n_h=2` via SO(3), S₅ needs `n_h=4`
  (:465–468); S₅ with `n_h=1` not learned even at 10 layers (:603); the 2-layer dihedral
  theorem (:1029) is the mirror image of quintic's cascade (it emulates rotations with
  reflections; quintic emulates the reflection's effect with rotations). **Error** at
  :1281: the full dodecahedral group is `I_h ≅ A₅×C₂`, not S₅ (the inversion fixes all
  five cubes; S₅ has trivial centre).
- In `info/mamba-3/rotation-as-optimization.md` terms: a reflection is **real step × rank-one
  curvature** (`α·H = (1−α)I + 2α·kkᵀ`); isotropic curvature has no non-central
  orientation reversal under any step algebra (§8's table). Hence "not a one-line change".

### §2–3 The way out: O(4) ↪ SO(5)

- The move parity already made: O(1) ↪ SO(2) *is* Complex2D (−1 reached around the
  circle, not through 0). For S₅: `T ↦ diag(T, det T)`; a transposition becomes a
  half-turn (eigenvalues −1,−1,1,1,1).
- **Dilation**: a rotation by θ in the plane (k, e₅), compressed to ℝ⁴, is exactly
  DeltaNet's `I − (1−cosθ)kkᵀ`, β ∈ [0,2] (check §3). The erased component is parked in
  e₅, not destroyed; at β ∈ {0,2} they coincide and e₅ carries the sign — layer 1 of
  the cascade folded into the block. A rank-one factor becomes an isometry one
  dimension up, which the RoPE trick absorbs.

### §4–6 A concrete kind: "Rotor4D + tilt"

- Spin(5) ≅ Sp(2) (2×2 unit quaternion matrices), acting on ℝ⁵ by `X ↦ U X U*`,
  `X = [[a, b], [b̄, −a]]`, a ∈ ℝ, b ∈ ℍ. `diag(q,p)` acts as Rotor4D on b, fixes a;
  `tilt(θ) = [[cosθ, −sinθ],[sinθ, cosθ]]` rotates (a, b₀) by 2θ (check §4).
- Per step `diag(q,p)·tilt(θ)` = Rotor4D's channels + **one angle** per block; reaches
  all of O(4) ↪ SO(5) with θ ∈ {0, π/2} (check §5); products generate SO(5). A
  hand-built one-block Sp(2) scan tracks S₅ exactly over 1000-symbol words (check §6).
- **Naming caveat**: *not* the 5-D analogue of Rotor4D. Rotor4D's one step is all of
  SO(4) (6 = dim); Quaternion4D's is SU(2) (3). Rotor4D+tilt's one step is a 7-dim
  subset of SO(5) (dim 10): same generated group, smaller per-step set; enough for S₅'s
  generators. Full SO(5) per step = Cartan form `diag(q,p)·tilt(θ)·diag(q',p')`
  (13 params, redundant) or `exp` on sp(2).
- **The "8-D ceiling" in CLAUDE.md** ("SO(4) is the ceiling for k=4; k=8 would break
  the scan") is about the *representation*: Complex2D/Quaternion4D store one number per
  block, which needs an associative normed division algebra (dims 1,2,4; octonions
  are non-associative). SO(8) itself scans fine as matrices. 5-D has no division algebra
  at all; Sp(2) is a *matrix over ℍ*, associative. The CLAUDE.md sentence could say so.
- Crate cost (if built): blocks of 5 (`state_rank` multiple of 5); +1 in-proj channel per
  (head, block); scan over 2×2 quaternion matrices (8 `quat_mul` per combine;
  recompute-backward like `quat_scan/`); unitarity renormalisation (Gram–Schmidt on two
  quaternionic columns); new `RotationState` accumulator; `muon_projections` segment.

### §7 The cheap alternative (worse)

Signed decay picks orientation: `M = |a|·T·κ^{[a<0]}` (κ = quaternion conjugation).
|a| stays in the log-space segsum; the bit rides the quaternion scan as a conditional
swap `(ε₁⊕ε₂, q₁·(ε₁ ? p₂ : q₂), p₁·(ε₁ ? q₂ : p₂))` (check §7, associative). But O(4)'s
two components meet only at the zero matrix ⇒ every flip crosses a full reset of the
head. In the probe (`o4sign`) the sign never crossed zero.

### Experiments: `tmp/reflections/train_s5.py`

**Probe, not the crate.** One linear-recurrent layer, 4 heads,
`h ← M[tok]·h + b[tok]` with `M = decay × rotation`; transitions/writes are per-token
lookups (isolates the group from the in-projection). Readout: per-head normalised
states → concat → one linear layer. Decay `sigmoid(a_raw)` (init 0.95). Adam, lr 3e-3,
warmup 200, cosine to 0.1×, clip 1.0, 6000 steps. Eval: train distribution @64 (with
resets), reset-free words @256 and @1024 ("tail" = last quarter).

| kind | per-step transition | n |
|---|---|---|
| `rotor4` | `L_q R_p̄` (Rotor4D group) | 4 |
| `so4` | `matrix_exp(W−Wᵀ)` | 4 |
| `rotor5` | `(L_q R_p̄ ⊕ 1)·G(2·tilt)` on (a, b₀) — Rotor4D + tilt | 5 |
| `so5` | `matrix_exp(W−Wᵀ)` — full SO(5) per step | 5 |
| `o4sign` | `|a|·T·K^{[a<0]}`, `a = tanh σ` (§7) | 4 |
| `o4hh` | `α·T·diag(tanh s, 1, 1, 1)` — non-orthogonal reference | 4 |

Control: A₅ on `rotor4`. 3 seeds each. ~22 min/run at len 192 on one CPU thread;
21 runs with `--procs 10` ≈ 45 min (12 cores).

- **sweep-1** (`--length 64 --p-reset 0.05`): nothing length-generalises, control
  included (A₅ 100% @64, ~45% @256). Cause: the leak (A₅ head had exact 180°/120°,
  writes ~2e-4, decay 0.92; training segments averaged 15.5 symbols). Transitions were
  still telling: `so5` s0/s2 put `std⊕sign` on one head (s ≈ half-turn + 2.5° residual,
  c = 72°/144°); SO(4) kinds only learned c as a lone 72° turn.
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
  c (0,72,72,144,144); decays 0.974 / 0.968. Found late: @256 3% at step 2000 → 16% at
  2500 → 60% at 4000 → 96% at 5000.
- **Head-0 decode** (inline script, not saved; recreate as below): nearest centroid on
  head 0's normalised state, centroids fit on 256 reset-free 48-symbol words (min gap
  0.618). As trained: 99.97% @256, 27.8% @1024, 7.6% @4096. With **decay 1 and zero
  writes on s/c** (`a_raw[1:] = 30`, `b[1:] = 0`): **100.00% @256, @1024 and @4096**.
  So the learned rotations *are* S₅; the long-length failure is the probe's leak plus the
  linear head mixing the three non-group heads.

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

**Caveats.** 1/3 seeds (`rotor5`) vs 0/3 (`so5`) does not rank the two kinds — they also
differ in parametrisation. Per-token lookups are easier than the crate's input
projections. Exactness at length needed the leak removed by hand.

### Next steps (B), in order

1. **Harden the probe** (cheap): more seeds for `rotor5` vs `so5` (+ a full-Sp(2)
   per-step variant); train longer / len 384–512 or a curriculum so the leak is priced
   in; readout per head (or MIMO-style several readouts of one head's state) so one
   group head suffices. Target: exact @1024 without surgery.
2. **Crate prototype** of Rotor4D+tilt as a `RotationKind` (see B§4–6 costs; CLAUDE.md
   "Mamba-3: rotation" and File Map for `src/mamba3/rotation/`, `src/mamba3/quat_scan/`);
   forward/step/gradient parity tests like the other kinds.
3. **reset-quintic with it**: a one-layer S₅ run (would replace the two-layer story in
   the README if it trains).
4. Docs, only if wanted: an `info/` note (reflections as dilation) with `check.py` moved
   to `scripts/`; the CLAUDE.md 8-D sentence; README sharpenings (A open items).

---

## Process reminders

- Edit crate files only with Edit/Write (no `sed -i`, heredocs, python writes).
- Never `rm` artifacts; fresh run directory per run (`train_s5.py --out` refuses an
  existing dir).
- CPU for these probes; GPU runs: `default,backend-cuda`, VRAM ≤ 3.6 GB, one big run at a
  time. The user runs benches.
- No `src/` change yet ⇒ no File Map / `files.md` update pending. If step B2 lands,
  prepare `tmp.md` at the end (CLAUDE.md rule), don't edit those files mid-work.
- Commit messages: write the text only (title + short body + `Co-Authored-By:` trailer);
  don't run git commit.
