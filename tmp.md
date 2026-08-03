# tmp.md — pending files.md / File-Map updates

## From the grokking study (cross-example remainders)

- `examples/common/cli.rs` — optim-load workaround (burn bug,
  `info/optim-load.md`): `load_model` re-stamps persisted `ParamId`s onto the
  loaded module (`restore_param_ids`/`ParamIdStamper`, matched by dotted
  path); `load_optim` now takes `model: &impl Module` and prunes orphaned
  optimizer-state entries before `from_bytes`. All 5 examples'
  `load_or_save_optim(&cfg, &model)` call sites updated.
- `examples/common/model/mod.rs` — added
  `impl ModelConfigExt for MambaVocabNetConfig`.
- `Cargo.toml` — dev-deps added: `rand_chacha = "0.9.0"` (deterministic
  splits); `burn-pack` (same git rev, `std`) for the ParamId workaround.
- `examples/README.md` — grokking entry added (done in-place).

The grokking study itself lives in `examples/grokking/README.md`
(standalone report). CLAUDE.md deliberately NOT updated for it.
Run artifacts under `examples/grokking/tmp/` (kept).

## State moments for `forward()` (new src/ feature — implemented & tested)

**File Map additions:**

```text
├─ modules/
│  └─ state_moments.rs  StateMoments: pooled per-token SSM-state moments (Σhhᵀ, Σh, count) + merge/pool_batch/pr
├─ mamba2/ssd/
│  └─ moments.rs        closed-form per-token state moments from the chunkwise tensors (no state materialisation)
```

**files.md entry drafts:**

- `src/modules/state_moments.rs` — family-agnostic `StateMoments
  { m2_bhrr, m1_bhr, count }` (raw sums ⇒ composable via `merge`;
  `pool_batch` folds batch into samples; `pr(center)` = differentiable
  `(trΣ)²/tr(Σ²)` per `(batch, head)`). Sample convention = one
  `(token, p)` row in `ℝ^r`, matching a `step`-loop cache read.
- `src/mamba2/ssd/moments.rs` — `Mamba2SsdInput::state_moments(valid_len)`:
  intra-chunk states decompose as `hₜ = dₜh₋ + sₜ`; `Σₜhₜᵀhₜ` reduces to
  three chunk-level GEMMs off tensors SSD Steps 1–4 already build (carry
  `(Σdₜ²)h₋ᵀh₋`, cross `h₋ᵀXᵀdiag(w)B̄ + ᵀ`, input `B̄ᵀ(LᵀL ∘ XXᵀ)B̄`);
  boundary states recomputed via Steps 2–3 ⇒ **pathway-agnostic** and plain
  autodiff. Validity mask excludes zero-pad `t`. Also
  `Mamba2SsdInput::detached()`.
- `src/mamba2/mamba2.rs` — `forward` → thin wrapper over private
  `forward_impl(.., with_moments: Option<bool /*detach*/>)`; new
  `forward_with_state_moments` (detached, diagnostics) and
  `forward_with_state_moments_grad` (attached — penalty use; the moments
  subgraph is independent of the chosen ssd_path, so it composes with
  SerialRecalculated's custom backward untouched).
- `src/modules/mod.rs` — `MambaBlock::block_forward_with_state_moments`
  (default panics; Mamba-2 only), `pub mod state_moments`.
- `src/modules/cache.rs` — Mamba-2 `MambaBlock` impl overrides the new
  trait method.
- `src/modules/layer.rs` — `Layer::forward_with_state_moments` +
  crate-private `forward_maybe_moments` (single code path for `Layers`).
- `src/modules/layers.rs` — `forward` → wrapper over private
  `forward_impl(.., with_moments)`; `forward_with_state_moments` returns
  `Vec<StateMoments>` (one per **virtual** layer, cache-slot order; both
  residual modes covered).
- `src/modules/network.rs` — `forward_with_state_moments` on
  `LatentNetwork`/`VocabNetwork` + both runtime enums (Mamba-1/3 panic via
  the trait default); new `network/tests.rs`.
- `src/lib.rs` — prelude re-exports `StateMoments`.
- `examples/grokking/diagnostics.rs` — `state_pr_forward(model, inputs,
  ssd_path)`: pooled PRs from per-layer moments (`pool_batch().pr(center)`),
  final-step PRs from the returned caches; fills the same `StatePr`.
- `examples/grokking/training.rs` — eval picks `state_pr` (stepwise) vs
  `state_pr_forward` (chunked) by `config.stepwise`.

**Tests added (all green, 181 total):** `modules/state_moments/tests.rs`
(PR identities, centering algebra, merge/pool); `mamba2/ssd/moments/tests.rs`
(closed form vs brute-force recurrence — values **and grads** wrt
x/dt/a_decay/B/init, incl. padded + learnable-init; plus end-to-end
`pr_matches_brute_force_states`: `pool_batch().pr(center)` vs a brute-force
PR over explicitly collected states — the `state_pr_forward` ↔ `state_pr`
equivalence);
`mamba2/mamba2/tests.rs::forward_state_moments_match_step` (vs real `step()`
loop: values, padding, random cache, streamed-merge; plain outputs
unperturbed) and `::forward_state_moments_grads_match_step` (moments-loss
grads vs step-loop grads through in_proj/conv/dt_bias/a_log, on
SerialRecalculated; y-only params confirmed untouched);
`modules/network/tests.rs::vocab_forward_state_moments_match_step`
(full cascade, per-layer, the grokking consumption pattern).

**Deferred:** cascading the `_grad` variant above block level (Layer/Layers/
networks) — do together with the actual state-PR penalty wiring; Mamba-3
moments (same decomposition applies; note RoPE puts states in
per-timestep-rotated frames, so pooled PR is frame-mixed there).
