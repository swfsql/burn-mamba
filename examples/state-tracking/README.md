# State tracking — the `A₅` word problem in the grokking protocol

The successor substrate to the [`grokking`](../grokking/README.md) example.
Its §6 retired mod-p addition for Mamba-3 (the substrate walks through the
starved-data wall under weight decay alone, leaving the structural priors
nothing to do); this example moves the same experimental machinery onto a
task where the state has to do something a sum cannot: **compose a
non-abelian group**.

## The task, in plain words

The grokking task was a *combination table over categories*: tokens like
`grape grape` in, category `banana` out, because `(2+2) mod 3 = 1`. This
task is the same game with one change — the things being combined are not
numbers on a clock but **shuffles of 5 cards**.

Picture 5 cards in a row and an alphabet of two moves: **`a`** slides every
card one place over (the last wraps around — a 5-cycle), **`b`** rotates
just the first three cards among themselves (a 3-cycle). An input like
`a b b a …` (12 moves) means "perform these shuffles in order", and the one
supervised question, at the last position, is: **what does the row look
like now?** Exactly 60 arrangements are reachable; each is a category, so
the answer is a 60-way classification (chance ≈ 1.7%).

The essential difference from mod-p: clock steps **commute** — advance-2
then advance-3 lands where advance-3 then advance-2 does — so a bag of
tokens (a sum, a phase) answers it, and that is precisely the shortcut the
grokked mod-p circuits used. Card shuffles **don't commute** (`ab ≠ ba`):
no sum, count, or accumulated angle can answer. The model must *carry the
current arrangement through the sequence*, updating it move by move —
state tracking in the literal sense.

Formally this is the **word problem of `A₅`** (the alternating group on 5
letters, the rotation group of the icosahedron, the smallest *non-solvable*
group), and by Barrington's theorem it is `NC¹`-complete — non-solvability
is what closes every layered-counter shortcut: you compose, or you fail.
The reason it is *this* group: an abelian rotation state (`Complex2D`,
`SO(2)`) is a set of clock hands and can only add angles, while the
non-abelian `SU(2)` quaternion rotation (`--quat`) can represent the
**binary icosahedral group `2I = SL(2,5)`** — a double cover of `A₅` — so a
quaternion state can *store the current arrangement as a 3D rotation* and
update it by multiplication. That contrast, at otherwise identical
configuration, is what this example exists to measure, with the grokking
example's full instrument attached.

## The protocol (transposed from `grokking`)

Everything that made the grokking study controlled carries over — shared in
`examples/common/` (`diagnostics.rs`: state PR on both axes, `PR_ℂ(M_phys)`
for Mamba-3, weight spectra, the differentiable rank/norm/noise penalties;
`protocol.rs`: the full-batch plumbing):

- **Enumerate, then starve.** All `2^seq_len` generator words (default
  `seq_len = 12` → 4096), deterministically split by fraction
  (`--train-fraction`, seeded `ChaCha8Rng`). Generalization = accuracy on
  held-out words of the same length.
- **Final-position supervision only.** No per-position CE: the running
  product is never supervised, so intermediate composition must be *learned*,
  not fitted. Full-batch AdamW, plain decoupled weight decay, constant lr
  (grokking-literature setup; resume-friendly).
- **Per-position accuracy as an eval-only depth probe** (`positions.csv`,
  printed each eval). Because the model is causal, its output at position
  `t` equals its final-position output on the length-`t` truncated word — so
  this curve is exactly **accuracy on shorter (never-supervised) words**,
  i.e. length generalization. Early positions reading ~0% is normal while
  the head is calibrated to the trained length only; a genuine
  rotation-conveyor circuit (`yₜ = Cₜᴴ Pₜ B₀ x₀`) is length-independent by
  construction, so the curve snapping up **at all depths at once** is the
  circuit-formation signature. (Supervising every position instead would
  train all prefix lengths simultaneously — an easier, denser task that
  hands early positions to memorization; deliberately not done here.)
- **PR diagnostics + penalties** at eval points: state PR (`PR_ℂ(M_phys)` —
  quaternionic blocks under `--quat`), the p-axis probe (via `--inference`),
  weight spectra, `--pr-lambda` / `--l2-lambda` / `--noise-lambda` /
  `--state-pr-lambda`, sine/gating schedules — same semantics as the
  [grokking knob table](../grokking/README.md#knobs-extra-args-after-the-second---).
  Task knobs here: `--seq-len`, `--quat`, `--rope-fraction`, `--stepwise`
  (chunkwise is the default at 13 tokens; the state-PR penalty needs it).

Model: the grokking `--mamba3` twin (Mamba-3 only) — 1 head
(`per_head_dim = d_inner`), SISO, `ngroups = 1`, 1 layer, untied head,
`d_model = 64`, `state_rank = 32`, `rope_fraction = 1.0` default. One shared
vocabulary, symbols first (`[a, b, anchor | 60 element classes]`), so the
weight diagnostics read the input-alphabet rows directly; the loss is CE
over the full vocabulary with mass on the class region.

Two design points that make the comparison valid:

- **The anchor token.** The rotation acts on both `B` and `C`, so the SSD
  readout only ever sees *relative* rotations `Pₜ Pᵢ⁻¹`. A fixed anchor at
  position 0 (rotation learned to identity) makes the anchor's contribution
  `Cₜᵀ Pₜ B₀ x₀` carry the *absolute* product.
- **The depth metric** (the truncation-equivalence probe above): the scalar
  test accuracy only sees the trained length; the curve is what separates a
  length-bound fit from the length-independent conveyor.

## Run

```bash
# the two arms (fresh dirs under tmp/):
cargo run --release --example state-tracking --features backend-cuda,fusion -- \
    --training -a examples/state-tracking/tmp/<run> -- --wd 1.0 --steps 20000
cargo run --release --example state-tracking --features backend-cuda,fusion -- \
    --training -a examples/state-tracking/tmp/<run> -- --wd 1.0 --steps 20000 --quat

# full panel (accuracies, depth curve, both state-PR axes, sample words):
cargo run --release --example state-tracking --features backend-cuda,fusion -- \
    --inference -a examples/state-tracking/tmp/<run>
```

Resume mechanics are the grokking example's: relaunch with the same `-a`
plus `--steps N --step-offset <done>`
([details](../grokking/README.md#resume-mechanics-multi-phase-runs)).

## Hypotheses (pre-registered)

1. **Does it grok at all?** Unlike mod-p addition, the final product of a
   non-abelian group is not a sum of per-token contributions — no
   phase-coded embedding-sum shortcut exists (the grokking §6 escape hatch).
   Expected: a real plateau for both arms, with the depth probe flat at ~0
   off the trained length until (if) composition is found.
2. **The quaternion arm can represent the circuit; the complex arm cannot.**
   `2I ⊂ SU(2)`: a single quaternionic conveyor whose data-dependent
   rotation *is* the group element would solve the task with quaternionic
   `PR_ℂ(M_phys) ≈ 1` and a flat depth curve — the exact non-abelian
   analogue of the grokking §6 endpoint. The abelian arm's rotations
   commute, so its in-state compositions collapse; if it generalizes at all
   it must do so through gating nonlinearity across positions (visible as a
   shallow depth curve and/or p-axis rank).
3. **Rank pressure has a barrier to act on here.** If the plateau is real,
   `--state-pr-lambda` (differentiable through the rotation — quaternion
   included) is predicted to help the quaternion arm settle into the
   single-block conveyor, and to be unable to rescue the complex arm — the
   causal test that mod-p addition could no longer host.

## Status (runs in progress, 2026-07-10)

Interim, single seed, `seq_len 12`, fraction 0.5, ≈26k params. Hypothesis 1
already holds: a real plateau exists (train ≈ 0.14, both arms), escaped only
with enough weight-decay heat — wd 1.0 chokes the circuit, wd 0.03 never
escapes, **wd 0.1 escapes at ~13k**. The working quaternion recipe is a
three-phase schedule on one lineage (`tmp/a5-l12-f0.5-wd0.1-quat`):
wd 0.1 heat (0–20k) → wd 0.03 anneal (20–40k) → lr 3e-4 cool (40–60k),
reaching **train 0.43 / test 0.38** and — hypothesis 2's signature, with no
PR pressure applied — the state consolidating into a **single quaternion
block** (`PR_ℂ` 2.9 → 1.2 while the state magnitude grows 14×; head rank
≈ 2, a phase-plane read-out). The complex twin under the identical phase-1
heat is **flat at train 0.14 / test 0.11 through 20k+** (extension to 40k
running). Numbers will move; the CSVs in `tmp/` are the record.

## See also

- `src/mamba3/rotation/` — the quaternion rotation (algebra, cumulative
  scan, `RotationKind` / `RotationState`); the module header derives the
  relative-rotation readout that motivates the anchor token.
- The grokking README's [Mamba-3
  read-out](../grokking/README.md#the-mamba-3-read-out-pr-over-a-complex-state)
  section — the `PR_ℂ(M_phys)` observable this example logs and penalizes —
  and its §6 for why this task replaced mod-p addition.
- The Mamba-3 paper's "Complex-Valued SSMs" / state-tracking sections.
