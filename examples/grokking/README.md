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
4. **Directed PR (rank) compression is the only intervention found that
   crosses the starved-data wall** — under both optimizers. It is not
   effective in isolation (it is a *slow* driver and needs a live
   exploration channel beside it: decoupled wd under AdamW, or SGD's native
   dynamics), it must be dosed gradually (λ ≈ 0.03; a hard crush finds the
   wrong low-rank subspace), but at f = 0.15 it takes AdamW from
   chance-forever to 98.8%, and SGD through the wall ~2× faster (with an
   unstable endgame).
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
  (d128/61k params stalls < 40% train), composition beats width.
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
| `--chunked` | chunkwise `forward()` instead of stepwise |
| `--no-diag` / `--no-state-pr` | skip all PR diagnostics / only the (costly) state-PR pass |
| `--pr-lambda <f64>` | differentiable spectral-PR penalty coefficient; **negative = expansion reward** (spell `--pr-lambda=-0.01`) |
| `--pr-target <emb\|emb-head\|bc\|all>` | which weight matrices the penalties target |
| `--pr-sine-period <steps>` | "breathing": λ_eff = `pr_lambda·sin(2π·step/period)` |
| `--pr-start-step <step>` | keep the PR penalty off until this step (gate) |
| `--l2-lambda <f64>` | plain `Σ‖W‖²_F` loss term on the same targets (norm control) |
| `--noise-lambda <f64>` | `Σ⟨W, detach(ε)⟩`, fresh ε/step: pure-noise gradient of this RMS (information-free control) |
| `--sgd <momentum>` | replace AdamW with plain SGD (coupled `--wd`, grad-clip 1.0 hardcoded, fresh optimizer each launch) |
| `--step-offset <n>` | added to logged/CSV step numbers on resumed runs |

### Diagnostics (the measurement side)

Everything is the participation ratio `PR(Σ) = (tr Σ)²/tr(Σ²)` — the effective
rank of a covariance/spectrum from two traces only; rotation- and
scale-invariant, range 1…N (`diagnostics.rs`):

- **State PR** (the original question): per layer/head, the recurrent states
  `ssm_bhpr` collected over (batch, step, channel) — "how many distinct write
  directions does the state use".
- **Weight spectral PRs**: embedding, LM head, each `in_proj` slice
  (`z|x|B|C`), `out_proj`, and the token-centered **B-alphabet**
  (`PR(emb·W_B)`, DC removed).
- **Embedding frequency PR**: exact p-point DFT energy spectrum of the
  embedding (non-DC bins), `(Σe)²/Σe²` = effective number of active
  frequencies (the Fourier-circuit detector; `rfft` is unusable — it needs
  power-of-two lengths).
- The penalties are the differentiable twins (`pr_tensor` on `WᵀW`), so the
  penalized quantity is exactly the logged one.

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

## Open threads

Plateau-length vs train-fraction curve (where does the ~4k search wall
diverge?); AdamW at higher lr (does a hotter Adam shrink the plateau
without auxiliary terms?); hybrid schedules (λ taper / lr decay) for the SGD
endgame; frequency-resolved embedding diagnostics over training (which
Fourier bins get selected, when — the DFT machinery is already in
`diagnostics.rs`); mechanistic checks on saved endpoints (frequency/state
ablations); a Mamba-3 arm (does the circuit move into the rotation angles?).
