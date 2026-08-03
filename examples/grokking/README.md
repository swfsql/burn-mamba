# Grokking — modular addition (a study in rank pressure, annealing, and the two barriers)

`(a₁ + … + a_k) mod p` with a small Mamba-2 LM: the classic grokking task
(Power et al. 2022 — train accuracy saturates early, test accuracy jumps much
later), instrumented as an experimentation/ablation platform. The example
began as a diagnostic study ("does the effective rank of the recurrent state
track the memorize→generalize transition?") and grew a family of
interventions: differentiable rank/norm/noise loss terms, penalty schedules,
and an SGD probe path. This README is the standalone report: setup, knobs,
findings, and reproduction commands for every claim.

## TL;DR — conclusions

1. **The classic grokking plateau is largely an Adam(W) artifact.** After
   memorization the CE gradients die, Adam's moments collapse, and the
   optimizer freezes while decoupled weight decay keeps contracting. Adding
   *any* live auxiliary loss term — a rank penalty, a plain L2 term, even a
   pure-noise gradient carrying zero information — restores motion and
   collapses the plateau to ~0 (test ≈ 100% by step 2k where weight decay
   alone needs 10k+). Plain tuned SGD (whose gradients are never
   moment-normalized) shows **no plateau at all**.
2. **Adam self-normalizes any live auxiliary gradient to lr-scale**
   (`update ≈ lr·g/√E[g²]`), which explains why the catalysis is flat across
   a 30× coefficient range, why noise magnitude is not a lever, and why an
   injected noise term *drowns* a directed penalty sharing the same
   normalizer (map–heat interference).
3. **A second, genuine barrier exists and is optimizer-independent: data
   starvation.** As the train fraction shrinks, a real search plateau
   returns (f = 0.25: ~4k steps under both AdamW+noise and plain SGD) and
   then blocks undirected exploration entirely (f = 0.15: chance for ≥20k
   steps under both).
4. **Directed PR (rank) compression crosses the starved-data wall where
   matched undirected exploration provably fails** — under both optimizers.
   It is a *scale-invariant, information-free structural prior*: not
   effective in isolation (it is a slow driver and needs a live exploration
   channel beside it: decoupled wd under AdamW, or SGD's native dynamics),
   and it must be dosed gradually (λ ≈ 0.03; a gated hard crush compresses
   the spectra 20× faster than wd and stays at chance — the *wrong* low-rank
   subspace). At f = 0.15 it takes AdamW from chance-forever to 98.8%, and
   SGD through the wall ~2× faster (with an unstable endgame). Prior work
   has shifted/crossed the wall by other means (LRD's spectral decay region
   expansion; knowledge distillation below the critical threshold) — the
   deltas here are the matched-exploration controls, the wrong-subspace
   dose result, and the heat–map interference (below).
5. **The original state-PR diagnostic works as a progress measure** (an
   inflection leads/tracks every transition, 4-for-4 arms), but the strong
   hypothesis is rejected: the final generalizing circuit *re-compresses*
   the state to a near-rank-1 conveyor; the transient rank expansion is
   search scaffolding, not structure. The loud channel is the weight
   spectra (embedding / head / B/C slices).

## Setup

- All `pᵏ` sequences (mixed-radix enumeration, capped at 2M), deterministically
  split train/test **by sequence** (`ChaCha8Rng(split_seed)`). Full batch;
  cross-entropy on the final position only. Default `p = 97, k = 2`
  (9409 pairs), the k-arm uses `p = 11, k = 4` (14641 sequences).
- Model (see `model.rs`): Mamba-2 `MambaVocabNet`, `d_model 64`, `expand 1`,
  1 head, `state_rank 32`, `conv_kernel 1` (all pair interaction flows through
  the recurrent state), untied LM head — 29k params. `k = 4` needs 2 layers
  (`--n-layers 2`, 35k params): 1 layer is capacity-blocked at any width
  (d128/61k params stalls < 40% train), composition beats width. `--mamba3`
  swaps in a Mamba-3 block under the same constraints (SISO, 1 head, no conv
  at all; 18.8k params) — the complex-state arm (see the Mamba-3 read-out
  section).
- Optimizer: AdamW with **plain (non-cautious) decoupled decay** (cautious
  decay masks exactly the pressure grokking relies on), grad-clip 1.0,
  lr 1e-3 constant. `--sgd <momentum>` switches to plain SGD (see below).
- Training runs token-by-token (`step()`) by default — mathematically
  identical to the chunkwise `forward()` (the library's parity contract),
  ~7× faster at these tiny sequence lengths, and it exposes the per-step
  state caches the diagnostics read. `--chunked` selects the
  recompute-backward chunkwise path (less memory) for capacity probes.
- Artifacts directory (`-a <dir>`): configs, `model.bpk`/`optim.bpk`
  (checkpointed every `save_every = 2000` steps), and three CSVs —
  `metrics.csv` (`step,lr,train_loss,train_acc,test_acc,emb_pr,head_pr,emb_freq_pr`),
  `pr.csv` (per layer/head state PR), `weights.csv` (per-layer weight PRs).
  `train_loss` is always CE-only (comparable across arms); penalty values are
  printed to stdout at eval points.
- Cost: k2 runs ≈ 1 GB VRAM, k4 ≈ 2.6 GB. On the dev GPU (CUDA + fusion) a
  k2 step costs ~0.03 s (~30–90 s per 1k steps depending on diagnostics and
  contention).

```bash
# the memorization control arm (wd = 0 is the default)
cargo run --release --example grokking --features "backend-cuda,fusion" \
    -- --training -a artifacts/grok-wd0

# a classic grokking arm
cargo run --release --example grokking --features "backend-cuda,fusion" \
    -- --training -a artifacts/grok-wd1 -- --wd 1.0 --steps 20000

# evaluate + diagnostics panel + sample predictions on any checkpoint
cargo run --release --example grokking --features "backend-cuda,fusion" \
    -- --inference -a artifacts/grok-wd1
```

All commands below abbreviate the prefix to `grokking --training -a <dir> --`.

### Knobs (extra args after the second `--`)

| knob | meaning |
|---|---|
| `--wd <f32>` | AdamW decoupled decay; also fills `sgd_wd` (coupled) for the SGD path |
| `--lr <f64>`, `--steps`, `--train-fraction`, `--p`, `--k` | schedule / task |
| `--d-model --expand --state-rank --n-layers` | model size (fresh configs only) |
| `--mamba3` | build a Mamba-3 block instead of Mamba-2 (fresh configs only); diagnostics/penalty switch to `PR_ℂ(M_phys)` automatically |
| `--quat`, `--rope-fraction <0\|0.5\|1.0>` | with `--mamba3`: `Quaternion4D` rotation / rotated fraction of `state_rank` (default `Complex2D`, 0.5; fresh configs only) |
| `--chunked` | chunkwise `forward()` instead of stepwise |
| `--no-diag` / `--no-state-pr` | skip all PR diagnostics / only the (costly) state-PR pass |
| `--pr-lambda <f64>` | differentiable spectral-PR penalty coefficient; **negative = expansion reward** (spell `--pr-lambda=-0.01`) |
| `--pr-target <emb\|emb-head\|bc\|all>` | which weight matrices the penalties target |
| `--pr-sine-period <steps>` | "breathing": λ_eff = `pr_lambda·sin(2π·step/period)` |
| `--pr-start-step <step>` | keep the PR penalty off until this step (gate) |
| `--l2-lambda <f64>` | plain `Σ‖W‖²_F` loss term on the same targets (norm control) |
| `--noise-lambda <f64>` | `Σ⟨W, detach(ε)⟩`, fresh ε/step: pure-noise gradient of this RMS (information-free control) |
| `--state-pr-lambda <f64>` | penalize the **recurrent state's** PR directly (Σ over layers/heads, batch-pooled; `PR_ℂ(M_phys)` on a Mamba-3 net); requires `--chunked` |
| `--sgd <momentum>` | replace AdamW with plain SGD (coupled `--wd`, grad-clip 1.0 hardcoded, fresh optimizer each launch) |
| `--step-offset <n>` | added to logged/CSV step numbers on resumed runs |

### Diagnostics (the measurement side)

Everything is the participation ratio `PR(Σ) = (tr Σ)²/tr(Σ²)` — the effective
rank of a covariance/spectrum from two traces only; rotation- and
scale-invariant, range 1…N (`diagnostics.rs`):

- **State PR** (the original question): per layer/head, the recurrent states
  `ssm_bhpr` collected over (batch, step, channel) — "how many distinct write
  directions does the state use". On a Mamba-3 net this automatically becomes
  the **Hermitian `PR_ℂ` of the physical-frame state** (see [the Mamba-3
  read-out](#the-mamba-3-read-out-pr-over-a-complex-state) below).
- **Weight spectral PRs**: embedding, LM head, each `in_proj` slice
  (`z|x|B|C`), `out_proj`, and the token-centered **B-alphabet**
  (`PR(emb·W_B)`, DC removed).
- **Embedding frequency PR**: exact p-point DFT energy spectrum of the
  embedding (non-DC bins), `(Σe)²/Σe²` = effective number of active
  frequencies (the Fourier-circuit detector; `rfft` is unusable — it needs
  power-of-two lengths).
- The penalties are the differentiable twins (`pr_tensor` on `WᵀW`), so the
  penalized quantity is exactly the logged one.

### The Mamba-3 read-out: PR over a complex state

*(Instrument built and tested in the library; the model arm is `--mamba3`
below — wiring smoke-tested, no experiment runs yet: see Open threads.)*

Mamba-3 breaks the plain covariance read-out above, because its state is
genuinely **complex**: the data-dependent RoPE realises `h ∈ ℂ^{p×r/2}` as a
real `[p, r]` tensor in which each rotation plane is one complex coordinate's
`(Re, Im)` pair, and the implementation carries a **cache-frame** state with
the cumulative rotations absorbed into B̃/C̃ (the paper's "RoPE trick"). Two
naive readings then fail, in dual ways:

- the realified **real PR double-counts**: a single rotating complex
  direction — precisely the conveyor §1's circuits end on — reads PR ≈ 2
  (≈ 4 for the quaternionic rotation), a representation artifact, not
  memory;
- the **cache-frame moment charges frame drift**: a *static* physical state
  under ongoing rotation spins in the cache frame, decohering its pooled
  covariance toward the number of active planes — phantom rank with zero
  change in what the model remembers.

The shipped observable (and penalty) is therefore the **Hermitian PR of the
physical-frame moment**, `PR_ℂ(M_phys)` with `M_phys = Σₜ cₜᴴcₜ`, where `cₜ`
is the cache state **de-rotated per token** — the frame that raw, un-rotated
C reads (`yₜ = C̃ₜᴴh̃ₜ = Cₜᴴcₜ`). Design points, each pinned by a library
test:

- **Rank-honest by construction.** Within-plane rotation is free (the
  rotating conveyor reads `PR_ℂ ≡ 1`); rank *created* by rotating retained
  writes apart is charged (a constant writer spread over k planes at
  different rates → `PR_ℂ → k`) — that is genuine memory occupancy which no
  per-token gauge can collapse; and `θ ≡ 0` degenerates **exactly** to the
  Mamba-2 moment, so the Mamba-3 penalty is a strict generalization of the
  one §4 proved out.
- **No closed form exists.** The per-token phases couple the token index to
  the moment's matrix entry, so the Mamba-2 trick (three chunk-level GEMMs,
  no state materialisation) has no analogue; the library instead
  materialises the states **one chunk at a time** (the same
  recompute-and-discard discipline as the `SerialRecalculated` SSD path) and
  de-rotates them, with a custom recompute backward so training-scale memory
  stays flat. Gradients flow to the **rotation angles** too — the penalty
  can shape the rotation itself, not just the write directions.
- **One source of truth for the pairing.** Which realified coordinates form
  a complex (or quaternionic) pair depends on the block's rotation config
  (interleaved SISO vs half-and-half MIMO, partial `rope_fraction`,
  `Quaternion4D`); the block exports it (`Mamba3::state_pairing()`) and the
  PR recombines the same real moment sums `Σhhᵀ` under it —
  `pr_complex(Real)` is byte-identical to `pr()`, so every Mamba-2 number in
  this README is untouched by the extension.

The example's diagnostics and the `--state-pr-lambda` penalty are
pairing-aware (`diagnostics::state_pairing_of`): pointed at a Mamba-3 net
they log and penalize `PR_ℂ(M_phys)` with no further wiring, on both the
stepwise path (the cache state de-rotated by the cumulative rotation) and
the chunkwise path (the library's per-token moments) — forward/step
agreement is part of the library's parity test suite. The weight-side
diagnostics/penalties carry over too (both families lead `in_proj` with the
same `[z|x|B|C|…]` column layout). The model arm is `--mamba3`
(`--quat` / `--rope-fraction` select the rotation; same size constraints as
the Mamba-2 model, SISO, no conv — 18.8k params at the p = 97 defaults):

```bash
# a Mamba-3 arm with the state-PR penalty (artifacts under tmp/mamba-3/)
grokking --training -a tmp/mamba-3/<run> -- --mamba3 --rope-fraction 1.0 --wd 1.0 \
    --state-pr-lambda 0.03 --chunked --steps 20000
```

### Resume mechanics (multi-phase runs)

Relaunching with the same `-a` dir resumes model + optimizer. Notes:
- `examples/common/cli.rs` works around a burn bug (persisted `ParamId`s are
  dropped by `load_record`, which silently resets Adam moments and grows the
  optim record each relaunch — see `info/optim-load.md`): ids are re-stamped
  on load and orphaned optimizer entries pruned. Verified: CE continues
  seamlessly across a relaunch.
- Every launch **re-saves `training_config.json` with the CLI overrides
  applied** — a resume with different knobs silently overwrites the config
  provenance. Back up the dir first (`cp -r`) if you care about it.
- `--step-offset <n>` keeps console/CSV step numbers continuous (the loop,
  save cadence, and sine phase run on the raw step).
- There is no `--seed` CLI override; to change the seed, pre-write
  `training_config.json` (copy one, edit `seed`) into a fresh dir — configs
  load from the artifacts dir.

---

## Findings

Numbers below are from seed 0 (the default) on a CUDA backend; exact values
wobble slightly across hardware/nondeterminism but every qualitative claim
reproduced on re-runs (and the key one across seeds). Chance = 1/p ≈ 1.03%
for p = 97.

### 1. State PR is a leading indicator; the strong hypothesis is rejected

Diagnostics-on arms (state PR logged to `pr.csv`):

```bash
grokking --training -a tmp/wd1   -- --wd 1.0 --steps 50000    # groks ~10k
grokking --training -a tmp/wd01  -- --wd 0.1 --steps 100000   # liftoff ~32k, 97.6% @100k
grokking --training -a tmp/wd0   -- --steps 20000             # control: chance forever
grokking --training -a tmp/k4wd1 -- --p 11 --k 4 --n-layers 2 --wd 1.0 --steps 12000
```

- A state-PR inflection led or tracked the transition in **4 of 4** grokking
  arms; wd-0 controls stay flat. wd 1.0: dip to the global minimum (1.36 @2k)
  → spike to 3.15 exactly through the test-acc jump → decay. wd 0.1: shallow
  dip then re-expansion from ~26k, leading liftoff by ~6k steps. k4 (2-layer):
  layer 0 stays a flat conveyor (~1.4) while layer 1 — the accumulator — rises
  1.34→2.19 through the transition, then decays.
- **Rejected**: the final generalizing circuit re-compresses the state PR to
  near-conveyor levels (k2: ~1.3; k4 L1 peak 2.2 ≪ 2×#frequencies ≈ 10). The
  rank expansion is transient search scaffolding, not final structure.
- The loud, persistent channel is the **weight spectra**: emb PR 41→7–15,
  B/C slices → ~2, B-alphabet → ~1.1, and the emb-frequency PR concentrates
  (47 → 22–30 ≈ 2×#frequencies, the Fourier circuit). Decay strength sets the
  timescale (~10× between wd 1.0 and 0.1); the destination is approximately
  shared. Wrinkle: wd 1.0's emb PR *re-expands* 7.7→14.9 during the final
  96%→100% consolidation while emb-freq concentrates — frequency-adding for
  error correction.

### 2. PR compression is a causal (but slow) grokking driver at wd 0

The penalty: `loss += λ · Σ PR(W)` over `--pr-target` matrices — pure rank
pressure, zero norm pressure (PR is scale-invariant).

```bash
# the causal arm (60k) + continuation (40k): 37% @60k → stall ~80% @100k
grokking --training -a tmp/pr001 -- --pr-lambda 0.01 --pr-target all --no-state-pr --steps 60000
grokking --training -a tmp/pr001 -- --steps 40000 --step-offset 60000 --pr-lambda 0.01 --pr-target all --no-state-pr
# matched control (same seed/init, λ=0): chance through 60k+
grokking --training -a tmp/ctrl  -- --no-state-pr --steps 60000
```

- λ = 0.01 at wd 0 **groks** (liftoff ~30k, 80% @100k) where the identical
  wd-0 control never leaves chance — compression is a driver, not a decay
  side effect. λ = 0.1 blocks the fit entirely (the penalty's compression
  gradient fights memorization — unlike weight decay, which doesn't slow the
  fit at all).
- The arm **stalls at ~71–80%** with every weight PR frozen at its floor
  (emb 1.2) and CE ≈ 1e-4: a rank-pressure/CE equilibrium.
- **Release test** (resume the stalled checkpoint with `--pr-lambda 0`): no
  unlock — the climb *slows ~10×* (≈0.12 %/k) and CE collapses to 2e-6.
  The pressure was the motor, not the barrier.
- **Sign check** (`--pr-lambda=-0.01`, expansion reward): anti-grokking —
  spectra slam to their ceiling (emb PR 62.9/64), test at chance.
- **Zero-mean breathing from scratch** (`--pr-lambda 0.01 --pr-sine-period
  8000`): fails (≤0.8% by 33k) — oscillation without net compression bias is
  not a driver.
- **Breathing from the stall** (resume the 80% checkpoint with
  `--pr-lambda=-0.01 --pr-sine-period 8000` — negative flips the phase to
  expand-first): un-sticks it, +6% in two breaths (gains land in the
  compression half-cycles), then its own ceiling ~86% with diminishing
  returns per breath.
- **Tempo**: no PR-penalty configuration (λ 0.001–0.1, gated via
  `--pr-start-step`, fast breathing) approaches wd 1.0's speed. A gated
  λ = 0.1 crushes the spectra 20× faster than wd 1.0 does — and gains
  nothing: compression speed ≠ grokking speed; finding the *right* low-rank
  subspace is the slow part.

### 3. The plateau is an optimizer freeze; any live loss term collapses it

The decisive battery (all 2k–3k steps, `--no-state-pr`; test acc @1k/@2k):

| arm | @1k | @2k | command suffix |
|---|---|---|---|
| wd 1.0 alone | 1.1% | 3.7% | `--wd 1.0 --steps 2000` |
| PR λ0.01 alone (wd 0) | ~1% | ~1% | `--pr-lambda 0.01 --steps 2000` |
| L2 alone (wd 0) | 6.4% | 20.5% | `--l2-lambda 0.00033 --steps 2000` |
| noise alone (wd 0) | 0.2% | 0.9% | `--noise-lambda 0.0003 --steps 2000` |
| **wd 1.0 + PR λ0.01** | **93.3%** | **99.98%** | `--wd 1.0 --pr-lambda 0.01 --steps 2000` |
| wd 1.0 + PR, seed 1 | 38.7% | 94.5% | (seed via config file) |
| wd 1.0 + PR λ∈{0.001,0.003,0.03} | 94.5/97.9/49.1% | 97.2/99.98/**100.0%** | `--pr-lambda <λ>` |
| wd 1.0 + matched L2 | 99.3% | 99.5% | `--wd 1.0 --l2-lambda 0.00033 --steps 2000` |
| wd 1.0 + sign-flipping PR | 97.2% | 100.0% | `--wd 1.0 --pr-lambda 0.01 --pr-sine-period 500 --steps 2000` |
| wd 1.0 + pure noise | 99.3% | 99.9% | `--wd 1.0 --noise-lambda 0.0003 --steps 2000` |

(The L2 coefficient is matched so its initial loss contribution equals the PR
term's: `Σ‖W‖²_init ≈ 6283`, PR contribution ≈ 2.1 ⇒ μ = 3.3e-4.)

- The combination is ~10× superadditive, **train and test rise together** —
  grokking with no plateau. Robust across seeds and a 30× λ range.
- It is **not rank-specific** (matched L2 works), **not norm-specific**
  (sign-flipping PR has zero net rank bias *and* zero norm pressure — PR is
  scale-invariant — and works), **not even weight-dependent** (a pure-noise
  gradient works).
- Two-factor grid: contraction only (wd 1.0) 3.7% @2k · heat only (noise at
  wd 0) 0.9% · both **99.9%**. Simulated annealing, decomposed.
- Mechanism: post-memorization CE gradients →0, Adam's m,v collapse, updates
  die; decoupled wd bypasses the moments (contraction without search). Any
  auxiliary gradient g gives `v ≈ E[g²]` ⇒ `update ≈ lr·g/√v` — **Adam
  self-normalizes any live term to lr-scale**, whatever its coefficient.
  Hence the dose-flatness; hence also (see §4) noise *drowning* a directed
  term that shares the normalizer.
- Noise magnitude is not a lever: λ_n ∈ {1e-3, 3e-3, 1e-2} all behave like
  3e-4 (only the CE floor rises with λ_n — v-inflation damping the CE
  signal).

### 4. The second barrier — data starvation — and compression as the only key

Generality probes of the wd+noise recipe:

```bash
# k4 composition: no boundary — test 95.5% @1k (baseline wd-alone: 99% @5.5k)
grokking --training -a tmp/k4n -- --p 11 --k 4 --n-layers 2 --wd 1.0 --noise-lambda 0.0003 --steps 3000 --no-state-pr

# f=0.25: the plateau RETURNS (~4k at chance), then 79% @10k
grokking --training -a tmp/f25n -- --train-fraction 0.25 --wd 1.0 --noise-lambda 0.0003 --steps 10000 --no-state-pr
# wd 1.0 alone at f=0.25: chance through 10k
grokking --training -a tmp/f25w -- --train-fraction 0.25 --wd 1.0 --steps 10000 --no-state-pr

# f=0.15: heat fails entirely (chance through 20k)
grokking --training -a tmp/f15n -- --train-fraction 0.15 --wd 1.0 --noise-lambda 0.0003 --steps 20000 --no-state-pr
```

At f = 0.15 (all wd 1.0, memorized by ≤2k, `--train-fraction 0.15`):

| auxiliary term | @10k | @20k | outcome |
|---|---|---|---|
| noise 3e-4 (any λ_n) | 0.7% | 0.7% | flat chance |
| PR λ0.01 | 0.8% | — | chance (dose too weak) |
| PR λ0.1 gated @2k | 0.6% | — | chance (crush → wrong subspace) |
| **PR λ0.03** | 2.3% | 5.9% | → 18% @30k → 65% @40k → **98.8% @48k** |
| PR λ0.03 **+ noise** | — | 1.0% | interference kills the climb |

```bash
# the wall-crosser (extend in 10k blocks with --step-offset):
grokking --training -a tmp/f15pr -- --train-fraction 0.15 --wd 1.0 --pr-lambda 0.03 --pr-target all --steps 10000 --no-state-pr
grokking --training -a tmp/f15pr -- --steps 10000 --step-offset 10000 --no-state-pr   # …repeat to 50k
```

- The grokking delay has **two components**: the Adam freeze (cured by heat;
  the whole plateau at generous data) and a genuine, data-dependent basin
  search (f = 0.25: ~4k even with heat; f = 0.15: blocks heat entirely).
  Annealing fixes the optimizer, not the statistics.
- **Directed λ0.03 compression crosses the wall** — the dose window is
  narrow and must let compression co-evolve with the fit (0.01 too weak,
   0.1-crush lands wrong).
- **Map + heat interfere under Adam**: adding noise to λ0.03 kills it
  (shared normalizer: `(m_PR+m_noise)/√(v_PR+v_noise)` buries the small
  persistent PR component under `v_noise`).
- **A second directed key — state-rank compression — and the two stack.**
  `--state-pr-lambda` penalizes the recurrent *state* PR directly (needs
  `--chunked`; §1's re-compression target, not the weights). Alone at λ0.03 it
  also crosses the wall (~90% @45k, comparable to weight-PR-alone); **combined**
  with the weight-PR penalty (both λ0.03) it is superadditive on *speed* — ~90%
  by ~30k and ~98% final, vs ~42–45k for either alone — while pinning the state
  to rank-1 (state PR ≡ 1.00) throughout. Squeezing the state toward the
  conveyor it re-compresses to anyway (§1) is a useful directed prior, not a
  fight with the circuit.

  ```bash
  # weight-PR + state-PR stacked (both --chunked; state-PR requires it):
  grokking --training -a tmp/f15both -- --train-fraction 0.15 --wd 1.0 \
      --pr-lambda 0.03 --pr-target all --state-pr-lambda 0.03 --chunked --steps 50000
  ```

### 5. SGD probes: no native plateau; the search wall reproduces

`--sgd <momentum>` switches to plain SGD (coupled decay from `--wd`,
grad-clip 1.0 — unclipped full-batch SGD+momentum NaNs within ~2k at any
workable lr; lr search: 0.1 and 1.0 diverge, 0.03 too slow, **0.05 + m 0.9**
works). Coupled decay must be small: `--wd 0.02` (shrink-matched to AdamW's
per-step 1e-3) *blocks the fit* — raw CE gradients (~1e-4) drown under
`0.02·w` without Adam's rescaling. Use `--wd 0.002`.

```bash
# NO PLATEAU: test 86.9% @1k, 100.00% @2k — no auxiliary term at all
grokking --training -a tmp/sgd -- --sgd 0.9 --lr 0.05 --wd 0.002 --steps 3000 --no-state-pr
# decay still required: wd 0 memorizes (98.9% train @3k), test at chance
grokking --training -a tmp/sgd0 -- --sgd 0.9 --lr 0.05 --wd 0 --steps 3000 --no-state-pr

# the search wall is optimizer-independent:
grokking --training -a tmp/sgdf25 -- --sgd 0.9 --lr 0.05 --wd 0.002 --train-fraction 0.25 --steps 10000 --no-state-pr   # plateau ~4k → 99.8% @10k
grokking --training -a tmp/sgdf15 -- --sgd 0.9 --lr 0.05 --wd 0.002 --train-fraction 0.15 --steps 20000 --no-state-pr   # chance flat

# directed compression crosses under SGD too — faster, but unstable endgame
grokking --training -a tmp/sgdf15pr -- --sgd 0.9 --lr 0.05 --wd 0.002 --train-fraction 0.15 \
    --pr-lambda 0.03 --pr-target all --steps 20000 --no-state-pr   # 49% @16k, peak 90.4% @24k, then limit-cycles
```

- SGD's exploration never freezes (no moment normalization: residual CE
  gradients + momentum + a large lr keep it moving), so contraction + native
  heat grok immediately — the f = 0.5 plateau simply never forms.
- f = 0.25 reproduces the ~4k search plateau near-quantitatively; f = 0.15
  blocks SGD exactly as it blocks Adam+noise. The wall is a property of the
  data, not the noise source.
- SGD + λ0.03 transits the wall ~2× faster than AdamW + λ0.03 (native heat
  does not share a normalizer with the map — no interference), but
  oscillates at the compressed endgame (test 90→60→77→43%, loss spiking
  10×) where AdamW consolidates cleanly to 98.8%: Adam's normalization is a
  liability mid-search and an asset at convergence.
- Caveat: SGD ran at 50× AdamW's lr; these are mechanism claims, not a
  tuned-fairness comparison.

## Related work & positioning

A literature pass (July 2026) places the findings as follows.

- **Rank pressure causality (§2)** — *rescoped as an isolation, not a
  first*: DeMoss et al., "The Complexity Dynamics of Grokking"
  ([2412.09810](https://arxiv.org/abs/2412.09810)) already show a
  scale-invariant spectral-entropy penalty causing grokking — but inside a
  stack with wd = 1 + weight noise, never isolated. The wd = 0 isolation
  (control never leaves chance), the release test (pressure-as-motor, ~10×
  slowdown on removal), and the sign-flip anti-generalization control appear
  new. "Low-Rank Decay" ([2606.04405](https://arxiv.org/abs/2606.04405)) is
  the nearest 2026 neighbor (spectral regularizer accelerating grokking and
  expanding the data-fraction region in scale-invariant transformers) but
  its nuclear-norm-like term carries *norm* pressure — PR's scale-invariance
  is what dissociates rank from norm here; LRD lists exactly this causal
  isolation as future work. Correlational base: Yunis et al. spectral
  dynamics; spectral-entropy collapse as a leading indicator with a blocking
  (necessity-direction) intervention
  ([2604.13123](https://arxiv.org/abs/2604.13123)) — ours is the inducing
  (sufficiency-direction) twin.
- **Optimizer-freeze / noise catalysis (§3)** — the decomposition ("any live
  auxiliary gradient un-freezes AdamW because Adam self-normalizes it to
  lr-scale; hence dose-flat, information-free catalysis") appears
  unarticulated; the pieces exist separately: softmax collapse
  ([2501.04697](https://arxiv.org/abs/2501.04697)) — gradient death by
  numerical absorption (a *different* freeze: it predicts stalls under any
  optimizer, whereas our SGD arms show no plateau; still, a float64/StableMax
  control is owed); variance-limited phase transition
  ([2603.15492](https://arxiv.org/abs/2603.15492)) — Adam rectifies noise
  anisotropically (their isotropic-noise-fails matches our heat-only cell);
  Arrhenius/metastable-escape accounts
  ([2606.17120](https://arxiv.org/abs/2606.17120)) — our dose-flatness and
  the SGD no-plateau *dispute* the thermal-barrier picture for the f=0.5
  plateau; informative auxiliary terms collapsing the plateau
  ([2605.15787](https://arxiv.org/abs/2605.15787), KL-to-oracle) — our
  detached-noise control shows the information content is unnecessary;
  Grokfast ([2405.20233](https://arxiv.org/abs/2405.20233)), Slingshot, Muon
  — adjacent acceleration results by other mechanisms.
- **Heat–map interference (§4)** — new in grokking; the closest articulation
  of "Adam's shared denominator distorts auxiliary-gradient composition" is
  in continual learning ([2604.22407](https://arxiv.org/abs/2604.22407),
  attenuate-then-adapt conflict — the mirror image of ours).
- **Data wall (§4)** — well established as critical dataset size (Varma et
  al. circuit efficiency,
  [2309.02390](https://arxiv.org/abs/2309.02390); ungrokking/semi-grokking;
  KD crossing below threshold,
  [2511.04760](https://arxiv.org/abs/2511.04760)). The contribution is the
  two-barrier *decomposition* (optimizer artifact vs optimizer-independent
  search wall, verified across two optimizers).
- **Diagnostics (§1)** — transient complexity rise-and-fall, effective-dim
  tracking, and embedding DFT concentration are transformer-standard (Nanda
  et al.; Liu et al.; DeMoss). New mainly via the substrate: recurrent-state
  covariance PR in an SSM, and the re-compression to a ~rank-1.3 conveyor.
- **SSM substrate** — no formal grokking-on-Mamba paper found; one informal
  LessWrong write-up (Oct 2024) observed grokking in a minimal SSM. "First
  systematic grokking study on an SSM" is defensible citing it.

Suggested write-up spine (from the same review): lead with the two-barrier
decomposition and noise-catalysis mechanism; present rank compression as the
directed instantiation that crosses the second barrier.

## Open threads

Toward a shareable write-up, in rough blocking order: **3–5 seeds per
headline cell**; **a transformer replication** of §3/§4 (the mechanism is
claimed at the optimizer level — substrate-independence is a prediction);
**direct m/v moment traces** through plateau and revival (currently
inferred); **softmax-collapse controls** (float64 / StableMax); an **ε/β₂
sweep** (the freeze should depend on Adam's ε floor); **fairness-matched
SGD** (current SGD lr is 50× AdamW's); more task families (modular
multiplication, k = 3, one non-modular). Then the earlier list:
plateau-length vs train-fraction curve; AdamW at higher lr; hybrid schedules
(λ taper / lr decay) for the SGD endgame; frequency-resolved embedding
diagnostics (which Fourier bins get selected, when); endpoint
frequency/state ablations.

**Mamba-3 runs** — the measurement/penalty instrument and the model arm are
both in place (see [the Mamba-3
read-out](#the-mamba-3-read-out-pr-over-a-complex-state); `--mamba3`,
smoke-tested); what's missing are the experiment runs. The question sharpens
nicely on this substrate: mod-p addition *is* rotation composition, and Mamba-3's
data-dependent angles can represent it natively — the grokked Fourier
circuit may migrate from the write directions into θ. Concrete predictions
to test: the Mamba-2 endpoint needs ≈ 2×#frequencies realified write
directions, while a circuit living in the rotations could hold
`PR_ℂ(M_phys)` between 1 (single-plane conveyor, all structure in θ) and
#frequencies (one plane per frequency — rotation-*created* rank is charged,
so multi-frequency memory cannot hide); and since the penalty is
differentiable through the angles, pressing on `PR_ℂ` tests whether rank
pressure actively pushes the circuit *into* the rotation — a cleaner
low-rank prior than Mamba-2 can express.
