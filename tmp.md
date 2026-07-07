# Prepared doc updates (apply after context reset, per CLAUDE.md rules)

No `src/` files changed — the CLAUDE.md File Map needs **no** update.
`examples/README.md` already updated in-place (grokking entry + `MambaVocabNet` mention).

## If files.md covers the touched files, candidate entries

- `examples/grokking/` — new example: modular addition `(a+b) mod p` grokking with
  state-PR diagnostics.
  - `main.rs` launch + extra-args overrides (`--wd --lr --steps --train-fraction --chunked`)
  - `dataset.rs` all `p²` pairs, `ChaCha8Rng` split disjoint by pair; `full(p)` diagnostic set
  - `model.rs` `model_config(p)`: Mamba-2 `MambaVocabNetConfig`, d_model 64, expand 1,
    1 head, state_rank 32, conv_kernel 1, untied head (rationale in doc comment)
  - `training.rs` `GrokkingConfig` + full-batch AdamW loop (plain non-cautious decay),
    CE on final position; `final_logits()` stepwise (default, ~7× faster at T=2,
    parity-checked) vs chunked; `metrics.csv`/`pr.csv` logging
  - `diagnostics.rs` participation ratio `PR(Σ)=(trΣ)²/tr(Σ²)` from the two traces:
    N-side state PR per layer/head (pooled/final × centered/uncentered) read from
    `ssm_bhpr` via step-mode caches; weight-side spectral PR (embedding, LM head);
    exact p-point DFT embedding frequency-energy PR (`rfft` unusable: needs pow-2 length)
- `examples/common/model/mod.rs` — added `impl ModelConfigExt for MambaVocabNetConfig`
  (was `MambaLatentNetConfig`-only).
- `Cargo.toml` — added dev-dependency `rand_chacha = "0.9.0"` (deterministic dataset splits).

## Experiment status (not for files.md — session context)

- Memorization milestone done: d64/expand1/N32 (~29k params) hits train acc 1.0 in ~4k
  steps (wd 0), test at chance. d32 (12k params) memorizes far too slowly.
- Key early finding: memorizing solution has LOW N-side state PR (~4.4/32; init ~1.8,
  which the conv-bias/SiLU DC component of B explains — the multiplicative `x(a)[p]`
  weighting makes it un-centerable). Hypothesis sign may flip: generalization may RAISE
  state PR. emb-freq PR ~47/48 (flat) at memorization = no Fourier structure — the
  circuit detector works.
- Scout arms running (detached, scratchpad): grok-wd1.0-f0.5 (50k steps),
  grok-wd0.1-f0.5 (100k steps).
- **wd 1.0 arm grokked** (test 80% @ 10.5k, climbing). State PR through it:
  memorization hump (→3.9 @250) → fall to global min **1.36 @2000** (train acc
  saturates, test at chance) → **rise to 3.15 exactly during test-acc liftoff**
  (2k→4.3k: test 4%→35%) → slow drift down ~2.0 during consolidation. PR min
  LEADS the jump by ~400–900 steps. LM-head spectral PR halves (32→17.6)
  smoothly through generalization; emb-freq PR only 47→40 (no strong Fourier
  concentration yet).
- **wd 0.1 arm (not grokked @11k)**: NO dip — high-PR memorized plateau ~4.4–5.4;
  emb PR rising 39→46, head PR flat ~41, emb-freq pinned 47.3. In-run
  prediction: its PR plateau should break downward before test acc leaves
  chance; if it groks without the dip, the V is a wd-1.0 artifact.
- Step-2 design implication: signature is dip→re-expansion (V), not one-way
  collapse — penalty/reward scheduling should target the V, and the handoff's
  "PR collapse" criterion is satisfied in the literal leading-indicator sense.
- **wd 0.1 prediction CONFIRMED**: PR plateau ~5 → shallow dip (2.7–4 @14–24k)
  → steep re-expansion from ~26k (→7.45 @32k) leading test-acc liftoff @~32k
  by ~6k steps. Two-for-two on PR inflection preceding the jump.
- **Generalized solution panel (wd1.0 @ 96% test)**: massive compression
  everywhere — emb PR 41→7.7, emb-freq 47→29 (concentrating), B/C slices
  ~2, B-alphabet 1.2, state PR back down to 1.29 (the transition spike was a
  transient). At T=2 even the generalizing circuit uses the state as a ~rank-1
  conveyor; the Fourier machinery lives in embedding × gate. Full trajectory
  shape: hump → dip → spike (leads jump) → decay.
- Added since: weight panel (weights.csv: per-slice in_proj z/x/B/C, out_proj,
  token-centered B-alphabet) + k-summand arm (`--p --k`; dataset generalized,
  mixed-radix enumeration, diagnostic_set caps at 10k samples). Old artifact
  training_config.json files needed a `"k": 2` patch (burn Config load has no
  serde defaults for missing fields).
- k-arm scout grok-p11k4-wd1.0 (wd 1.0): NO grokking plateau — test tracked
  train from step ~250 (both →26%/20% @1k), dipped, then ground at ~24%/18%
  until killed @9.25k (VRAM budget). wd 1.0 confiscates capacity (all weight
  PRs crushed: z 29→11, B 6.7→2.6) faster than the k=4 task fits. Data kept.
- k=4 runs cost ~2.6 GB VRAM each (vs ~1 GB k=2; keep total < 4.6 GB — k4 arms
  serialize). Two early k4 launches under VRAM pressure hit cubecl memory-pool
  corruption (21k+ assertion failures — grok-p11k4-wd0/-wd0.1 dirs are
  INVALID; wd0 control relaunched as grok-p11k4-wd0-2).
- Running: grok-wd0.1-f0.5 to 100k (endpoint-convergence check: does weak
  decay reach the same compressed circuit as wd 1.0?); grok-p11k4-wd0-2
  (20k, memorization control — decides if d64/N32 can fit k=4 at all, else
  capacity bump before k-arm grokking is meaningful).
- **k=2 Step 1 CLOSED — both arms complete, convergent endpoints.**
  Final panels: wd1.0 (50k, test 1.0000): state PR 1.32, emb 14.9, head 11.8,
  emb-freq 22.4, z/x 10.6/11.6, B/C 1.2/2.6, B-alpha 1.1. wd0.1 (100k, test
  0.976): state PR 3.06, emb 7.4, head 13.8, emb-freq 30.0, z/x 25.9/22.3,
  B/C 3.5/5.8, B-alpha 1.3. Decay sets timescale (~10×), destination shared
  at the architecture level (rank-1 write alphabet, compressed block).
  Wrinkle: wd1.0's emb PR re-expanded 7.7→14.9 during 96%→100% while
  emb-freq concentrated 29→22.4 — consistent with ADDING Fourier frequencies
  (≈2×#freqs ⇒ ~7–8) for error correction during final consolidation.
- k=4 capacity: 1-layer blocked regardless of width (d64: gap ≈ 0, creeps;
  d128 61k params: stuck <40% train @12k). **2-layer d64 (35k params)
  memorizes k=4 at k2 pace** (train 95% @2.2k) — composition beats width.
  At wd 0, test acc creeps to ~36% during memorization (k4 leaks partial
  generalization, unlike k2's chance-pinned control).
- Probe conveniences added: `--chunked` (recompute backward, less memory) +
  `--no-diag` (skip PR passes; metrics.csv gets nan columns) + model-size
  overrides `--d-model --expand --state-rank --n-layers` (fresh configs only).
- k4 arms: wd0.3 too slow (killed @7k, test 40%; data kept). **wd1.0 GROKKED**
  (grok-p11k4-d64L2-wd1.0): memorize ~3k, test 34%→99% over 3–5.5k.
  **Layer-resolved signature**: L0 = flat conveyor (~1.4); L1 (accumulator)
  state PR rose 1.34→2.19 exactly through the transition, tracking test acc,
  then DECAYED in consolidation (1.66 @5.75k, falling). L1 also holds richer
  write machinery (B/C slices 9–11 vs L0 3–5; B-alpha 4.9 vs 2.9).
- **STEP-1 VERDICT (4-for-4 arms)**: (1) state PR works as a *progress
  measure* — an inflection leads/tracks every transition; wd-0 controls flat.
  (2) The strong hypothesis is REJECTED: the final generalizing circuit always
  re-compresses state PR to near-conveyor levels (k4 L1 peak 2.2 ≪ 2×#freqs
  ≈ 10) — rank expansion is transient search scaffolding, not structure.
- Run finishing to 30k for the k4 endpoint panel (compare vs k2 endpoints).
- Ops: kill runs by PID, never `pkill -f <name>` (pattern matches own shell).

## Proposed next steps (brainstormed at wrap-up, not yet decided)

1. **Weight-PR penalty replacing wd (k2, wd 0)** — most causal, cheapest (1 GB
   runs). At wd 0 the model memorizes and never generalizes; add a
   differentiable spectral-PR penalty (two-trace, on WᵀW) on chosen weights.
   If it groks, compression is the *driver*, not a decay side effect.
   Targets to map: emb-only / B+C slices / head / all (panel says emb+head
   compressed most, B/C → rank 2). Controls: matched plain norm penalty
   (magnitude vs rank-specificity), PR reward (sign check), random-subset
   target (specificity).
2. **State-PR reward on the k4 accumulator (L1)** — the reshaped Step-2: the
   transition signature is *transient expansion*, so test inducing it early
   (reward from step 0 vs gated on train-acc ≈ 1) at low/zero wd; prediction
   from Step-1 data: original penalty should *delay* grokking (sign check).
   Control: ‖h‖² at matched strength. Stepwise training already exposes
   tracked cache states; needs a `final_logits` variant returning states.
3. **Grokking-before-memorization** (strong early push): weight-PR penalty
   from step 0 at wd 0 — does compression pressure route directly to the
   circuit, omnigrok-style? (k4-wd0's 36% test leak says a direct path exists.)
4. **Frequency-resolved diagnostics**: log per-frequency embedding energies
   (top-k bins) over training at p=97 — which frequencies get selected and
   when (Nanda-style). DFT matmul already in diagnostics.rs. (p=11 has only
   5 bins — no resolution; use p=97/k2 or p=13+/k arms.)
5. **No-training mechanistic checks on saved endpoints**: frequency ablations
   on the k2-wd1.0 final embedding (project out bins → test acc); verify the
   L1-accumulator role at k4 by state ablations; check logits for
   cos((a+b)θ) structure.
6. **Verdict-table grid** (Step-1 report deliverable): (fraction × wd) k2
   runs, PR-inflection step vs test-acc-crossing step, lead-time table.
   Serialized 1 GB runs, can interleave with the above.
7. **Deferred/optional**: Mamba3+Complex2D arm (circuit in rotation angles?);
   `[a,b,=]` T=3 variant; larger-p k-arm (needs minibatching).

Suggested order: 1 → (its controls) → 2 → 6, with 4 added to whichever runs
next (cheap). Penalty infra is shared: one config (target enum, λ, schedule
gate) serves 1–3.
- Run artifacts now under examples/grokking/tmp/ (durable, in-repo); VRAM hard
  budget 4.6GB (silent UB above!); one big run at a time; no watcher loops —
  user reports run status.

## Weight-PR penalty arm (proposal 1) — implemented & running

- Code: `diagnostics.rs` gained `PrPenaltyTarget` enum (Emb|EmbHead|Bc|All),
  differentiable `pr_tensor(w)` (spectral PR via two traces, Gram on smaller
  side, clamp_min 1e-12) and `weight_pr_penalty(model, target)` (sum over
  selected matrices; slicing mirrors `weight_pr`). `training.rs`:
  `pr_lambda: f64 = 0`, `pr_target = All`, `state_diagnostics: bool = true`
  config fields; loss = CE + λ·penalty (csv train_loss stays CE-only; penalty
  printed at eval); state-PR pass gated on `state_diagnostics`. `main.rs`
  overrides: `--pr-lambda --pr-target <emb|emb-head|bc|all> --no-state-pr`.
  NOTE: old artifact training_config.json files not patched for the 3 new
  fields — loading them (e.g. `--inference` panels) will panic until patched.
- λ calibration probes (400 steps, k2/wd0, target all, dirs
  grok-k2-prprobe-l0.1 / -l0.01 + grok-k2-prsmoke): init penalty ≈ 211
  (7 matrices). λ=0.1 too strong: penalty →21.7 @400, train acc 2.3% (fit
  blocked). λ=0.01 right regime: penalty →38.9 decelerating (equilibrium),
  train acc 23% @400 (~2× slower than wd0 baseline's ~50%). Both probes: test
  acc → 0.0000, BELOW 1% chance, during early compression.
- MAIN RUN launched: grok-k2-wd0-prall-l0.01 (60k steps, λ=0.01, target all,
  wd 0, --no-state-pr, log at ….log). Question: does pure rank pressure
  (scale-invariant — zero norm shrinkage) induce grokking at wd 0?
- **IT GROKS (in progress)**: test acc 2.1%@20k → 8.6%@32.5k → 13.7%@40k →
  37.2%@60k, train 1.0 throughout — wd0 control never left chance, so
  compression is a *driver*. Geometry differs from wd arms: emb PR 1.38,
  head 1.18, all block PRs ~1.0 (near rank-1 spectra; wd endpoints were
  7–15) yet test climbs; emb-freq PR 28.8 (slow concentration). The circuit
  lives in the small-singular-value tail under one dominant component.
- Old-config patching: jq `{defaults} + .` applied to all
  examples/grokking/tmp/*/training_config.json (k/stepwise/diagnostics/
  state_diagnostics/pr_lambda/pr_target, later + pr_sine_period/step_offset).
- **Optim-load workaround ported from ../midi-gen/src/cli/mod.rs** (burn bug,
  see info/optim-load.md: load_record discards persisted ParamIds → Adam
  moments silently reset + optim record accretes a dead 2×-model cohort per
  relaunch). `examples/common/cli.rs`: `restore_param_ids`/`read_param_ids`/
  `ParamIdStamper` (stamp record ids onto loaded module by dotted path) after
  `load_model`'s load_record; `load_optim` now takes `model: &impl Module` and
  prunes orphaned entries (scalar keys `"{param_id}.{field}"`) before
  `from_bytes`. All 5 examples' `load_or_save_optim(&…, &model)` call sites
  updated. Cargo.toml: burn-pack dev-dep (same rev, std). VERIFIED on resume:
  CE continuous 1.6755e-4→1.6753e-4, no warnings (all ids matched), optim.bpk
  239,616 B = exactly 2× model.bpk (check it stays that size after saves).
- New knobs: `--pr-lambda` may be negative = expansion *reward* (spell
  `--pr-lambda=-0.01`; gate is `!= 0`); `--pr-sine-period <steps>` "breathing"
  λ_eff = pr_lambda·sin(2π·step/period) (raw loop step drives phase);
  `--step-offset` adds to logged/csv step numbers only (loop, cadence, sine
  phase stay raw). `GrokkingConfig::pr_lambda_at(step)`; eval prints λ_eff.
- RESUMED grok-k2-wd0-prall-l0.01 +40k steps (logs 60001–100000); 55% @81k.
- **Control gap (user)**: longest k2/wd0 plain run was only 5k steps (test
  0.45%) — can't claim "wd0 never groks" at the 100k horizon. Launched
  grok-k2-wd0-control-100k (100k steps, wd0, λ=0, seed 0 = same init as the
  penalty arm ⇒ exact counterfactual pair; --no-state-pr), concurrent with
  the resume (2×~1GB VRAM, precedented).
- Penalty arm STALLED ~71–75% from ~88k: weight PRs frozen at floor (emb
  1.20, head 1.10, embfreq 28.0) — CE/rank-pressure equilibrium. Hypothesis
  (from wd1.0's final emb RE-expansion 7.7→14.9 during 96→100%): finishing
  needs spectral room the constant λ forbids. Planned decisive follow-up:
  resume from 100k with λ released (0 or 0.003) — if test resumes climbing,
  the floor was binding ("compression drives the transition, expansion
  finishes it"). User: let it finish, set aside for now.
- LAUNCHED (concurrent, ~3.7GB total with resume+control):
  grok-k2-wd0-prall-lneg0.01 (60k, λ=-0.01 expansion REWARD, sign check;
  penalty rising 218@1 vs 211 init — working) and grok-k2-wd0-prall-sine8k
  (100k, λ_eff=0.01·sin(2π·step/8000) breathing; peaks +0.01 @2k, crosses to
  expansion @4k). Both k2/wd0 seed 0 --no-state-pr.
- Penalty arm FINISHED @100k: test 80%. Backed up to
  grok-k2-wd0-prall-l0.01-bak-100k/ (+run.log) BEFORE resuming. NOTE: each
  launch re-saves training_config.json with overrides applied — resumes
  silently overwrite config provenance (live file now shows the λ=0 phase);
  history lives in backups/log/tmp.md. Phases: 0–100k λ=0.01; 100k+ λ=0.
- **RELEASE TEST (λ=0 from 80%): hypothesis REJECTED** — no unlock; climb
  SLOWED ~10× (80.8→82.3% over 15k ≈0.12%/k vs ~1%/k under λ=0.01); PRs
  frozen (emb 1.18, embfreq 28.0); CE→2e-6 (vanishing gradients). Rank
  pressure was the MOTOR, not the barrier. Killed @~115k logged.
- **REWARD ARM: sign check passed** — train 1.0, test chance, spectra at
  CEILING (emb 62.9/head 63.7 of max 64). Expansion = anti-grokking. Killed
  @15k. Control @36k: still chance (penalty arm was ~10% at that step).
  Sine8k @12k: emb PR breathing 24.9→39.8→24.6 per half-cycle, test chance
  (too early). Synthesis: progress needs ACTIVE directed spectral pressure;
  passive room does nothing.
- LAUNCHED grok-k2-wd0-prall-sineresume100k (PID cohort; 40k steps, offset
  100000): the stalled-80% checkpoint (from -bak-100k, CSVs continue) under
  EXPANSION-FIRST breathing λ_eff=-0.01·sin(2π·step/8000) (negative λ flips
  phase: expand 0–4k, recompress 4–8k, …). Decisive test: does breathing
  re-energize the stalled climb where both constant-λ and release stalled?

## Final results wave (all k2, seed 0, wd0 unless noted)

- **Control (λ=0, 60k, killed)**: test 0.32% flat, train 1.0, spectra broad
  (emb 42). Penalty arm lifted @~30k from same init ⇒ causal claim SOLID.
- **Scratch sine8k (zero-mean breathing, 33.5k, killed)**: FAILED — test
  ≤0.8%, non-monotone; emb PR oscillated 24↔48. Oscillation without net
  compression bias doesn't grok. Directed pressure needed, not just motion.
- **Sine-resume from 80% stall: FINISHED 86.4% @140k.** Broke the stall
  (+6.4%): jumps to ~86% in 2 breaths (gains land in COMPRESSION halves;
  climb rate recovered to ~1.5%/k vs 0.12%/k release-crawl), then its own
  ceiling — last 3 cycles oscillate 79↔86 (expansion dents, recompression
  recovers). Breathing un-sticks; diminishing returns per breath.
- **Gated-penalty machinery**: `pr_start_step` config field +
  `--pr-start-step` (λ 0 before it; sine phase counts from gate). Configs
  re-patched (+pr_start_step:0). Reusable memorized base checkpoint:
  grok-k2-wd0-base2k (+-bak backup): 2k steps λ=0, train 94.6%, test chance
  (fit @1250 = 87.0% matches user's watched run — same seed ⇒ identical).
  Sweep arms = cp base2k → dir, resume --step-offset 2000.
- **wd1.0-tempo sweep: NO PR-penalty config matches wd1.0** (bar: 11%@3k,
  27%@4k, 50%@5k, 80%@10k total):
  · λ0.02 from scratch, 3k: test 0.6%, train 84% (penalty FIGHTS the fit —
    λ0.01 needed ~8k to memorize vs wd0's 4k, wd1.0's 1k; wd doesn't slow
    fit at all).
  · gated λ0.1 from base2k, +4k: chance. Compression brutal+instant
    (penalty 211→58 in 250 steps, emb 32→3.5) with train pinned 1.0 — but
    WRONG subspace: compression speed ≠ grokking speed; circuit-finding is
    the slow search process.
  · λ0.001 from base2k, +3k: chance; compression selective in wd1.0's ORDER
    (B 13→4 first, emb untouched) but ~10× too slow.
  · sine λ0.1 period 2k from base2k, +8k (4 breaths): chance.
- **SYNTHESIS**: PR pressure is a genuine grokking driver but inherently a
  SLOW one (≈wd0.1 tempo: liftoff ~30k). wd1.0's speed needs the norm
  channel — decay under Adam continuously shrinks norms ⇒ rising effective
  step size ⇒ built-in annealing motor that performs the search while rank
  falls. Scale-invariant PR pressure has no motor once at its floor
  (release-freeze, reward-ceiling, breathing-unsticks all consistent).
- Old gate2k-l0.1 dir: user-launched, interrupted @1250 pre-save — only
  init-state .bpk inside; superseded by base2k flow. Data kept.

## NORM-MOTOR TEST: MASSIVE SYNERGY (headline result)

- **wd1.0 + PR λ0.01 (all), from scratch: test 93.3% @1k, 99.98% @2k**
  (dir grok-k2-wd1.0-prall-l0.01-3k; SIGTERMed after 2k, answer in).
  Controls: wd1.0 ALONE re-baselined on current binary = 1.1% @1k, 3.7% @2k
  (grok-k2-wd1.0-rebase-2k — matches historical run exactly). PR alone:
  liftoff ~30k. **~10× superadditive**; train and test rise TOGETHER
  (train 1.0 @1k) — grokking with (almost) no plateau.
- **Seed-1 replication** (grok-k2-wd1.0-prall-l0.01-seed1, config-file seed
  trick — no --seed CLI): 38.7% @1k, 94.5% @2k. Robust.
- Combo spectra @1k: B/C 1.1, B-alpha 1.1, emb 7.4, emb-freq 39→29.7@2k —
  circuit found near-immediately.
- Note: no --seed/--split-seed CLI overrides exist; workaround = write
  training_config.json into a fresh dir before launch (config loads from
  artifacts dir).

## Dose-response + specificity controls (CORRECTED interpretation)

- λ dose at wd1.0 (2k runs, test @1k/@2k): 0.001 → 94.5/97.2; 0.003 →
  97.9/99.98; 0.01 → 93.3/99.98; 0.03 → 49.1 (train 78% @1k)/**100.00**.
  Flat plateau over 30× of λ — synergy is not a tuning artifact; even trace
  rank bias catalyzes fully.
- **Rank-specificity control FAILED (informative!)**: `--l2-lambda` added
  (plain Σ‖W‖²_F loss term, same `pr_target` matrices; Σ‖W‖²_init ≈ 6283,
  matched μ=3.3e-4 to PR's ≈2.1 init contribution). wd1.0 + matched-L2 =
  99.3% @1k — indistinguishable from wd1.0+PR. The accelerant is NOT
  rank-specific.
- Decomposition (all 2k, @1k/@2k): wd1.0 alone 1.1/3.7; L2-loss alone (wd0)
  6.4/20.5; PR alone chance (liftoff ~30k); wd1.0+L2 99.3/99.5; wd1.0+PR
  93–98/97–100.
- **REVISED synthesis: instant grokking = decoupled weight decay × ANY
  loss-coupled shrinkage penalty.** The loss-coupled term passes through
  Adam's moment statistics (unlike decoupled wd) — plausibly keeping
  gradients/second-moments alive so the optimizer never freezes while wd
  anneals. Rank pressure works as a loss-coupled term but its
  rank-specificity is not what buys the speed. PR-specific claims that
  SURVIVE: PR-alone causally groks where wd0 control never does (slow);
  breathing un-sticks stalls; state/weight PR as diagnostics (Step 1).
- l2_lambda config field + `--l2-lambda` CLI; configs re-patched
  (+l2_lambda:0). Dirs: grok-k2-wd1.0-prall-l{0.001,0.003,0.03},
  grok-k2-wd1.0-l2match, grok-k2-wd0-l2only, grok-k2-l2probe,
  grok-k2-wd1.0-rebase-2k.
- **NORM-NEUTRAL PROBE CONFIRMS "LIVE-TERM" STORY**
  (grok-k2-wd1.0-prsine500): wd1.0 + sign-flipping PR (λ0.01, sine period
  500 = 4 full cycles, zero net rank bias; PR is norm-neutral by
  construction) → 17.3% @500, 97.2% @1k, **100.00% @2k**. A loss-coupled
  term with NO net norm pressure and NO net rank bias still fully catalyzes
  wd1.0 (vs 3.7% @2k wd-alone). ⇒ The accelerant is the *presence of a live
  auxiliary gradient through Adam* (keeping moments/steps alive while
  decoupled wd anneals), not any specific pressure direction. Open q for
  later: does a weight-independent-gradient term (e.g. Σ w·detach(noise)) or
  even LR/noise injection reproduce it, or must the term depend on W?
- **NOISE PROBE: FULL CATALYSIS** (grok-k2-wd1.0-noise3e-4): wd1.0 +
  `noise_lambda` 3e-4 (new `Σ⟨W, detach(ε)⟩` term, fresh ε/step ⇒ gradient =
  pure noise, zero W-information; `--noise-lambda`, configs +noise_lambda:0)
  → 55.7% @500, 99.3% @1k, 99.9% @2k. The auxiliary term needs NO structure
  at all.
- **MECHANISM (closes the arc)**: post-memorization CE grad→0 ⇒ Adam's m,v
  collapse ⇒ frozen (decoupled wd bypasses moments — contraction only, no
  search). ANY auxiliary gradient g: v≈E[g²] ⇒ update ≈ lr·g/√v = lr·ĝ —
  **Adam self-normalizes any live gradient to lr-scale**, explaining the
  flat dose-response (30× λ range, and why 3e-4 noise isn't "small").
  Result: lr-scale exploration walk + wd contraction = literal simulated
  annealing ⇒ wide/structured (Fourier) basin found in ~1k steps. Classic
  grokking delay = waiting for tiny residual CE noise to do this walk
  unaided. Predictions (untested): dose-flatness down to λn 1e-6; noise-only
  at wd0 ⇒ NO grokking (exploration without contraction); plain SGD (no
  moment normalization) should lose the effect/need tuned scale.
- **NOISE-ONLY wd0 CONTROL CONFIRMS** (grok-k2-wd0-noiseonly, 2k): train
  97.0%, test 0.89% ≈ chance. Heat without contraction = memorization only.
  **Two-factor grid complete @2k: contraction only (wd1.0) 3.7% · heat only
  (noise) 0.9% · both 99.9%.** Remaining sharpest probe: SGD (no moment
  normalization) should lose the catalysis — needs optimizer swap (examples
  hardcode AdamW via common::training::optimizer_config).

## Boundary hunt: where is wd+noise NOT sufficient?

- **k4/p11/d64L2 (composition): NO boundary** —
  grok-p11k4-wd1.0-noise3e-4: test 95.5% @1k (train 99.75%; test TRACKS
  train from step 500). Baseline wd1.0-alone: 34% @2k, 99% @5.5k. Plateau
  collapsed on the compositional task too.
- **Data-starvation axis (k2, wd1.0+noise3e-4)**: f=0.5 instant (plateau
  ≈0); **f=0.25: plateau RETURNS** (~4k at chance, then 5.7%@6k → 26.8%@8k
  → 79.2%@10k; dir grok-k2-f0.25-wd1.0-noise, extended via resume) while
  wd1.0-ALONE at f=0.25 is chance through 10k (grok-k2-f0.25-wd1.0-alone) —
  acceleration persists, plateau-collapse doesn't; **f=0.15: chance through
  10k even with the combo** (grok-k2-f0.15-wd1.0-noise) — noise insufficient
  (train 1.0 @2k).
- **Refined mechanism: the grokking delay has TWO components.** (1)
  Adam-freeze (CE grads die post-memorization) — fully cured by any live
  auxiliary term; dominates at generous data (f=0.5, k4), where curing it
  collapses the plateau to ~0. (2) Genuine basin search — re-emerges as data
  shrinks the generalizing basin (f=0.25: ~4k; f=0.15: >10k). Annealing
  fixes the optimizer, not the statistics.

## f=0.15 head-to-head: compression BEATS noise at the starved boundary

- All arms f=0.15, wd1.0, train 1.0 by ≤2k. Test @10k/@20k:
  · +noise 3e-4: 0.68% / **0.65%** — flat chance through 20k.
  · +PR λ0.01: 0.79% @10k — chance (dose too weak).
  · +PR λ0.1 gated@2k: 0.61% @10k — chance (instant crush → wrong subspace,
    same failure as ungated-λ0.1 sweep earlier).
  · +PR λ0.03 constant: 2.3% @10k → **5.9% @20k, accelerating**
    (2.3→2.7→3.0→4.5→5.9 per 2k) — genuine liftoff foot.
- **⇒ Where search is hard (scarce data), the DIRECTED rank bias carries
  real information pure heat lacks — the "map" earns back its role.** The
  effective dose is narrow (0.01 nothing, 0.03 works, 0.1-crush wrong):
  compression must co-evolve with the fit, not outrun it.
- **Noise-scale sweep (f=0.15, wd1.0, 4k each): NULL at every dose**
  (λn 1e-3/3e-3/1e-2 all chance, train 1.0). CE floor rises with λn
  (0.033/0.077/0.21) = v-inflation drowning the CE signal — scale is not a
  lever, mildly counterproductive. (Caveat: 4k horizon; but 3e-4 flat
  through 20k.)
- **COMBINED map+heat (λ0.03 + noise 3e-4, f=0.15, 20k): INTERFERENCE, not
  stacking** — chance through 20k (1.04%) vs λ0.03-alone's 5.9%; fit also
  slowed (96.5% @4k vs 1.0 @2k). Mechanism-consistent: the noise term
  inflates v on every weight, and Adam's shared normalization
  (m_PR+m_noise)/√(v_PR+v_noise) crushes the small persistent PR component
  under v_noise ≈ λn² — the map drowns in the heat. Heat rescales all other
  signals down; use one or the other, matched to the regime.
- **λ0.03 FULL GROK at f=0.15** (extended 10k-by-10k to 50k): test 2.3%@10k
  → 5.9@20k → 18.2@30k → 64.8@40k → **98.8%@48–50k** (train 1.0 throughout;
  classic S-curve, sharp knee ~38–42k). At 15% data, where wd+noise is
  chance-flat and wd-alone hopeless, directed compression completes the
  grok. The regime split is now fully demonstrated end-to-end.
- Remaining queued: plateau-vs-fraction curve.

## SGD PROBE: THE PLATEAU IS AN ADAM(W) ARTIFACT (major reframe)

- Machinery: `sgd_momentum` config (`--sgd <momentum>`; <0 = AdamW) +
  `sgd_wd` (coupled decay; `--wd` fills both) + hardcoded clip Value(1.0) in
  the SGD path (unclipped full-batch SGD+momentum NaNs by ~1–2k at any
  workable lr; lr search: 0.1/1.0 NaN, 0.03 slow, **0.05+m0.9 works**).
  Configs patched (+sgd_momentum:-1,+sgd_wd:0). SGD path always fresh-inits
  the optimizer (no resume of momentum).
- Coupled wd 0.02 (shrink-matched to AdamW) BLOCKS the SGD fit — raw CE
  grads (~1e-4) drown under 0.02·w; SGD lacks Adam's rescaling. Same
  self-normalization asymmetry, opposite side. wd 0.002 is right.
- **SGD lr0.05/m0.9/wd0.002/clip1.0, f=0.5, NO auxiliary term: test 86.9%
  @1k, 100.00% @2k** (grok-k2-sgdbase) — train/test rise together, NO
  PLATEAU, matching the best AdamW combos. **SGD wd0 control**
  (grok-k2-sgd-wd0-3k): memorizes (train 98.9% @3k), test chance — decay
  still required.
- **UNIFIED STORY**: grokking delay = time for exploration×contraction to
  find the circuit. SGD keeps exploration natively (no moment normalization
  ⇒ residual CE grads + momentum + big lr keep moving); + coupled decay ⇒
  immediate grok. AdamW kills exploration post-memorization (normalization
  freeze) while decoupled wd contracts ⇒ the classic plateau — cured by ANY
  live loss term (heat). At scarce data the search is genuinely hard and
  only directed compression helps (regime 2 unchanged).
- Caveat: SGD lr 0.05 = 50× AdamW's lr (different effective temperature);
  the claim is mechanistic, not a tuned-fairness comparison. Untested: SGD
  at f=0.25/0.15 (does the search-limited regime reproduce under SGD?);
  AdamW at higher lr.
