# State tracking — the `A₅` word problem in the grokking protocol

The successor substrate to the [`grokking`](../grokking/README.md) example.
Its §6 retired mod-p addition for Mamba-3 (the substrate walks through the
starved-data wall under weight decay alone, leaving the structural priors
nothing to do); this example moves the same experimental machinery onto a
task where the state has to do something a sum cannot: **compose a
non-abelian group**.

## The task

The **word problem of `A₅`** (the alternating group on 5 letters — the
rotation group of the icosahedron, the smallest *non-solvable* group): read
an anchor token then a word of `A₅` generators (`a` = a 5-cycle, `b` = a
3-cycle), and output the **final running product** — a 60-way classification
read at the last position only (chance ≈ 1.7%).

By Barrington's theorem this is `NC¹`-complete. An abelian rotation
(`Complex2D`, `SO(2)`) composes only commutatively; the non-abelian `SU(2)`
quaternion rotation (`--quat`) can represent the **binary icosahedral group
`2I = SL(2,5)`** — a double cover of `A₅` — so it can compose the task
natively *inside the state*. That contrast is what this example exists to
measure, with the grokking example's full instrument attached.

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
  printed each eval): position `t` has ≤ `2^t` reachable prefixes, so early
  positions are solvable by memorization; genuine composition shows as the
  curve holding up at depth. This replaces the old harness's per-position
  *training* signal, which blurred exactly this distinction.
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

Two design points inherited from the old harness (they make the comparison
valid):

- **The anchor token.** The rotation acts on both `B` and `C`, so the SSD
  readout only ever sees *relative* rotations `Pₜ Pᵢ⁻¹`. A fixed anchor at
  position 0 (rotation learned to identity) makes the anchor's contribution
  `Cₜᵀ Pₜ B₀ x₀` carry the *absolute* product.
- **The depth metric**, since averages hide the memorization frontier
  (≈ `log₂(train words)` ≈ 11 at fraction 0.5 — deliberately close to
  `seq_len`: the deepest positions are exactly where memorization runs out).

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

## Hypotheses (runs pending)

1. **Does it grok at all?** Unlike mod-p addition, the final product of a
   non-abelian group is not a sum of per-token contributions — no
   phase-coded embedding-sum shortcut exists (the grokking §6 escape hatch).
   Expected: a real memorization plateau for both arms, with the depth probe
   pinned at the memorization frontier until (if) composition is found.
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

## Status

Harness rebuilt and smoke-tested on CUDA (both arms, penalty path,
diagnostics, CSVs — `tmp/smoke-complex`, `tmp/smoke-quat`); experiment runs
pending.

## See also

- `src/mamba3/rotation/` — the quaternion rotation (algebra, cumulative
  scan, `RotationKind` / `RotationState`); the module header derives the
  relative-rotation readout that motivates the anchor token.
- The grokking README's [Mamba-3
  read-out](../grokking/README.md#the-mamba-3-read-out-pr-over-a-complex-state)
  section — the `PR_ℂ(M_phys)` observable this example logs and penalizes —
  and its §6 for why this task replaced mod-p addition.
- The Mamba-3 paper's "Complex-Valued SSMs" / state-tracking sections.
