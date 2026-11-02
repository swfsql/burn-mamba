# CLAUDE.md

## What This Project Is

burn-mamba gives the Mamba-1, Mamba-2 and Mamba-3 blocks for
[`burn-stack`](../burn-stack), with many experimental options. It uses only
portable Burn tensor ops, so one code path runs on every backend (ADR-0001).

## Build & Test Commands

```bash
cargo check                                     # type-check the lib
cargo test --lib -- --test-threads=1            # lib tests
cargo test --examples -j 1 -- --test-threads=1  # example tests
cargo doc --all --no-deps                       # docs
cargo run --example reset-majority -- --training --inference
# the user runs the benches, not you:
./bench.sh
./kernels.sh
```

- **Feature flags select the backend**: `backend-{flex,flex-rayon,cpu,wgpu,
  webgpu,metal,vulkan,cuda,rocm,tch-cpu,tch-gpu,remote,ndarray}` and
  `backend-ndarray-blas-*`. Each one enables `burn/<backend>`. Use `flex` (the
  default) for checks and tests. Several backends can be compiled in at once.
  Then `Device::default()` reads `BURN_DEVICE` to select one.
- `mamba1`/`mamba2`/`mamba3`/`autodiff`/`optim` are on by default.
  `mamba2`/`mamba3` imply `autodiff`. `optim` (the Muon plan) implies
  `burn/optim` + `burn/std`. `cubecl`/`fusion` enable the memory-saving custom
  backward on those backends. `check-nan`/`check-inf` enable the NaN/Inf
  guards. `dev-f16`/`dev-simd`/`dev-autotune` serve the examples and tests.
- Every feature forwards to `burn-stack` in `Cargo.toml`. A backend feature on
  one side only loses its `BackendExt` impls without an error (ADR-0002). A
  new cubecl backend feature also goes into the list of `build.rs`.
- **Test device**: a test takes its device from burn-stack's `test_device()`
  (fp16 under `dev-f16`), never from `Device::default()`. The dtype defaults
  are global, and the first tensor fixes them for the process. A tolerance
  goes through `dtype_tol` (it keeps the fp32 value). A check on values
  that can be large compares with `max_rel_diff`. A host read-back converts
  (`try_into_vec_as`, not `try_to_vec`).

## Rules

### Code

- burn-stack code is block-agnostic by default. Its docs can cite a family. A
  rare exception in burn-stack takes the name of its mechanism. Mamba-specific
  code stays here (ADR-0002).
- The crate has `#![warn(missing_docs)]`: document each new public item.
  `cargo check` and `cargo doc --all --no-deps` must have no warnings.

### Docs

- **Always load the `asd-ste100` skill** before you write code comments,
  rustdoc or markdown. Write all of them in its style.
- Use the terms of the two glossaries (see [Where to look](#where-to-look)).
- `_Avoid_` applies to our own words. A doc can use a source word in quotes or
  backticks when it cites that source (ADR-0008).
- Keep the docs small. They describe the code as it is now. A doc is never a
  changelog: it has no "used to be / now", no "verified by", no dates and no
  PR history. When you find changelog prose, delete it.
- A change that adds, deletes or renames a file updates its
  [File Map](#file-map) line in the same change.
- A new term goes to `GLOSSARY.md` (through `/domain-modeling`). A new
  decision with a trade-off goes to an ADR in `docs/adr/`. A mechanism and its
  math go to the module header of its file.
- A name and its docs change in the same change. A rename of an identifier
  updates every doc that names it or its concept: comments, messages, the
  READMEs, `info/`, `GLOSSARY.md` and the ADRs. A change of a term in the docs
  also renames each identifier that uses the term in that sense.

### Custom Commands

- The project root is `/shared/claude/burn-mamba/`. You can read and write
  inside it, in `../{burn-stack,burn,cubecl,cubek}/` and in `/tmp`. Ask the
  user before you read `../py/`, or before any other access outside these
  directories.
- **Always** edit the content of a file with the Edit/Write tools. This rule
  also applies when a harness or auto-mode reminder says to change files
  through Bash. That reminder does not apply here.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`.
  - Use Bash to read and inspect (`cat`, `sed -n`, `rg`, `grep`), to move and
    copy files (`mv`, `cp`), and to write throwaway files in `/tmp`.
- `rg` is available. Do not use `cargo fmt`.
- A dangerous-command-guard hook (`dcg`) checks each Bash command. It blocks
  a `>` redirect to a path that holds a variable. Use a literal path, or a
  script file.
- **Waiting**: a build, a run, a chain of runs or one step of a chain can take
  more than ~1 min. Run each such command or wait in the background. Watch it
  with a `Monitor` (preferred), a background waiter script or
  `run_in_background`, then return the turn to the user. The watch also
  reports each failure state (for example, a crash). Do not hold the turn with
  a foreground loop (`until`, `while`), a `sleep` or a long `timeout`.
- **Memory**: ask the user before you write to the auto-memory. Show the
  change that you propose, and write it only after the user approves it.
- **Commits**: when a skill or the user tells you to commit, commit your work
  to the current branch. When the user asks only for a commit message, write it
  as text for the user to copy. A commit message has a title line and a short
  body, and it ends with the `Co-Authored-By:` trailer.
- **Pushes**: push the current branch to `origin` when a ticket is complete and
  reviewed, or when the user tells you to. Ask the user before a force-push or
  a tag.

## Where to look

- `GLOSSARY.md`: the terms of this crate. `../burn-stack/GLOSSARY.md`: the
  terms of the containers (block, layer, cache, row, segment, …).
- `docs/adr/`: the decisions with a trade-off.
- The module header of each file: its mechanisms, its math and its
  non-obvious choices. The headers of `src/mamba2/mamba2.rs` and
  `src/mamba3/mamba3.rs` are the math reference.
- `info/`: standalone derivation notes. Each note has a float64 `numpy` check
  in `scripts/`, with check numbers that match its sections. Cite a note. Do
  not restate it.
- `../burn-stack/`: the containers. Start from its `CLAUDE.md`.
  `examples/README.md`: the examples.
- `docs/agents/`: the configuration of the engineering skills (see
  [Agent skills](#agent-skills)).
- `../burn/`, `../cubecl/`, `../cubek/`: Burn and CubeCL.
- Ask the user before you read these external references:
  - `../papers/mamba/mamba-3/`: the TeX of the Mamba-3 paper.
  - `../py/state-spaces/mamba/`: the official implementation, authoritative.
    Its Triton SISO and Tilelang MIMO kernels are the single-SSD reference.
  - `../py/VikramLex/mamba3-minimal/`: the base of double-SSD.

## Agent skills

### Issue tracker

Local markdown files under `.scratch/<feature-slug>/`. See
`docs/agents/issue-tracker.md`. It also covers a change in both repos and the
burn-stack pin.

### Triage labels

The five default role strings, as the `Status:` line of an issue file. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.

### Test seams

The seams to propose for `/to-spec` and `/tdd`. See `docs/agents/testing.md`.

## Notation

A tensor name ends with a suffix that gives its shape, and shape `assert`s back
it. Such a name needs no extra comment. A comment can write a shape as `_bhl`
or as `[...]`. Paper style (upper-case `A,B,C,H,Y,L,…`) can be in comments,
never in code identifiers. A lower-case letter is a base dimension (below). An
upper-case letter is a relation of base dimensions (offset, multiple, concat):
`X` can be `x±1` or `x*2`, and `XY` can be `x+y` or `x*y`. A letter in a
formula is a paper symbol, not a shape letter. So `hₜ` is the state, and the
suffix `_bhl` still has `h` = `nheads`.

| Letter | Dimension | Paper | Python | Typical |
|--------|-----------|-------|--------|---------|
| `b` | `batch` | — | `batch` | varies |
| `s` | write-axis length (= `tokens`·`u`) | `T` | `seqlen` | varies |
| `t` | `tokens` = `s`/`u`, the read-axis length (Mamba-3) | `T` | `seqlen` | varies |
| `u` | `micro_steps` (Mamba-3) | — | — | 1 (stock) |
| `d` | `d_model` | `D` | `d_model` | 768, 1024 |
| `i` | `d_inner` = `expand`·`d_model` | `E·D` | `d_inner` | 2·`d_model` |
| `h` | `nheads` | `H` | `nheads` | `d_inner`/`per_head_dim` |
| `p` | `per_head_dim` | `P` | `headdim` | 64, 128 |
| `r` | `state_rank` | `N` | `d_state` | 64, 128, 256 |
| `m` | `mimo_rank` (Mamba-3) | `R` | `mimo_rank` | 1–8 |
| `n` | `nchunks` = `s`/`chunk_len` | — | `nchunks` | varies |
| `g` | `ngroups` | `G` | `ngroups` | 1 … `nheads` |
| `l` | `chunk_len` | `Q` (Mamba-2), `C` (Mamba-3) | `chunk_size` | 64 … 256 |
| `a` | `num_rotation_angles` = `rotation_dim`/2 | — | `num_rope_angles` | varies |
| `v` | `conv_dim` = `d_inner`+2·`ngroups`·`state_rank` (Mamba-2) | — | `conv_dim` | — |
| `k` | `conv_kernel` (Mamba-1/2) | — | `d_conv` | 4 |

The `R` of the Mamba-3 paper is the `M` of our comments: both are `mimo_rank`.
A shape letter `R` is a relation of `r`, the state rank.

## File Map

A directory `x/` next to a file `x.rs` holds only the `tests.rs` of `x.rs`.

```text
src/
├─ padding.rs            right padding inside a block
├─ packing.rs            packed segments (`forward_packed`)
├─ mamba1/               Mamba-1 block (sequential selective scan) + cache
├─ mamba2/               Mamba-2 block (`mamba2.rs`: the SSD math) + cache
│  └─ ssd/               the three SSD paths, selected by `ssd_path.rs`
│     └─ serial_recalculated/  the recompute backward (`Mamba2BackendExt`)
├─ mamba3/               Mamba-3 block (`mamba3.rs`: the math) + the cache enum
│  ├─ helpers.rs         shared by both pathways and by forward and step
│  ├─ ssd_path.rs        `Mamba3SsdPath` (both pathways), the chunk length
│  ├─ tap_pattern.rs     the tap pattern (`TapPattern`)
│  ├─ double_ssd/        double-SSD pathway + its cache
│  │  ├─ double_ssd/     its forward, and the step of both pathways
│  │  └─ ssd/            the standard SSD paths
│  │     ├─ serial_recalculated/  recompute backward, K1–K4 of both pathways
│  │     └─ ssd_path/    `Mamba3DoubleSsdInput`
│  ├─ single_ssd/        single-SSD pathway + its cache
│  │  ├─ single_ssd/     its forward
│  │  ├─ token_band.rs   the lag-`u` correction band
│  │  └─ ssd/            one-pass SSD paths, `diag.rs` (the diagonal γ-correction)
│  │     ├─ serial_recalculated/  recompute backward (+ its `diag.rs`)
│  │     └─ ssd_path/    `Mamba3SingleSsdInput`
│  ├─ rotation/          `RotationKind`, `rotate_bc_forward`, `rope.rs` (RoPE trick)
│  ├─ product/           MambaProduct (micro-steps)
│  ├─ positive/          positive systems: `scan.rs`, `kalman.rs` (gain), `tropical.rs`
│  └─ quat_scan/         quaternion cumulative product, recompute backward
└─ unified/              runtime-selectable enums
   ├─ mod.rs             `MambaSsdPath`, why the MIMO tensors are not Muon matrices
   ├─ cache.rs           `MambaCaches`, `impl Block`/`BlockConfig`/`CacheStack`
   ├─ capture.rs         `CacheTensors` of each family (captured steps)
   └─ tests/             the burn-stack containers with real blocks
build.rs                 the `cube_backend` cfg (the cubecl `backend-*` features)
benches/layer.rs         single-block benches, run by `bench.sh`/`kernels.sh`
info/                    derivation notes
scripts/                 the float64 checks of the `info/` notes
examples/                see `examples/README.md`
docs/                    `adr/`, `agents/`
```
