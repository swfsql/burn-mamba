# Handoff — tiny-stories ≤1M-param search (resume from a cleared context)

Read this first, then `ORIENTATION.md` (goal, rules, tools, protocol) in this
directory (`examples/tiny-stories/tmp/1m-b/`). `notes.md` is the full log
(long: search it, do not read it whole; "Part c" is the latest session).

## Where the search stands

Best: **0.7281 valid bits/char**, `h16` @14400 (`runs/p12k-h16-lr1-d`).

**Ceilings (user):** training steps, params (< 1M) and VRAM (the process
< 3.6 GB, `--max-vram 3480`). Wall clock is not a ceiling, unless a candidate
is much slower.

### The regime (every candidate)

- **Packed training:** `-- --pack 4224 --pack-rows 2 --ssd-path recalc:128`
  (`recalc:256` for a u2 model: the chunk is 128 tokens). Whole stories per
  row. Chosen by chars per second (`speeds.txt`).
- **Weight decay 0.1** for Muon and AdamW (`gen.py … wd=0.1`). User: every
  candidate has it. It ties with 1e-4 at every horizon.
- **LR:** 4e-3 constant until step 4630, then 1e-3 (`--max-lr 1e-3` on the
  extension). A halving to 5e-4 at 8000 ties (0.7283 vs 0.7281).
- **Milestones in fresh steps:** stable 2630 / 4630 / 8000 / 12000, each with
  a decay branch of M/5 (600 / 1000 / 1600 / 2400): `screen.sh` for the
  first, `extend.sh` for the rest. Every pool member has a fresh packed
  history, so `agg.py` compares them bucket by bucket.
- The validation is unchanged (bs 8, seq 1024, runlen 1: 190,070 chars).

### The pool (model configs; decayed valid, far use @9600)

| id | cfg | params | ms/step | reserved | @3230 | @5630 | @9600 | @14400 | far use | latest stable run |
|---|---|---|---|---|---|---|---|---|---|---|
| **h16** | `s1k-u1-wd1-h16` (d160 L5 p20, 16 heads) | 969,160 | 748 | 3443 MB | 0.8579 | 0.8014 | 0.7493 | **0.7281** | 0.270 | `p12k-h16-lr1` (12000, 1e-3) |
| **h16 × g2** | `s1k-u1-wd1-h16g2` (d152 p19, 16 heads, 2 groups) | 982,747 | 777 | 3491 MB | 0.8649 | 0.8083 | 0.7552 | – | **0.284** | `p12k-h16g2-lr1` (12000, 1e-3) |
| r128 | `s1k-u1-wd1-d152r128` (d152 p76, r128) | 975,372 | 491 | 2521 MB | 0.8680 | 0.8113 | 0.7590 | – | 0.267 | `p8k-r128-lr1` (8000, 1e-3) |

All are u1 (HorizontalCarryOver), MultiGate with 4 streams, Complex2D, 4
`Start` class latents. The 4th slot is open. `p12k-h16-lr05` (12000, 5e-4)
is the halving twin of h16. Evicted or parked (each can return): B
(`s1k-u1-wd1`, 0.7607 @9600), C (u2 lean), D, A, Kalman, the quaternion
L3 × V6 (`s1k-q4d-g2-d164L3V6`), MIMO m2 (over VRAM), h8, L6, L4 × r256,
r192 (VRAM margin), MultiGate n3/n6, standard residual.

### Findings that shape the next choices (details in `notes.md`)

- **Heads lower the loss** (h16: −0.01 at every milestone, 80 decay
  timescales). h8 did not help (non-monotone). h16 sits at the VRAM cap.
- **Groups raise the context use** (h16 × g2: the highest far use at every
  milestone, +0.015 over h16 @9600) at +0.006 of loss (d152 to fit 1M).
- **The LR step-down** at 4630 gives −0.011…−0.014 (paired, on two
  architectures). Further halvings saturate. The points fit a single-run
  cosine: run it from zero only when the search ends (user).
- **Without wd**, the step-down recovers less at the decay (the Kalman run:
  0.7791 @9600). Keep wd on.
- A larger state (r128) learns faster early and uses more context, but its
  loss converges to B's late. u2 lost. d160 L5 is the depth/width sweet spot.
  MultiGate n4 is locally optimal.
- **Context use** grows slowly with training (far use ~+0.03 per doubling of
  the steps). The old 256-char windows had only ~0.05. The user's two
  prompts (`infer-*` directories) still give fluent but off-plot text: rerun
  them on each new best (`--prompt … --continuations 3 [--temperature T]`).

### Known issues

- **A memory leak in the untied-parameter path** (virtual layers + untied
  tensors). A no-autodiff forward or step grows memory (~150 MB/s on flex, on
  the GPU too); training stays flat. `burn_stack::utils::untied::view`
  clones and re-wraps the layer on every application of every call (also ~3×
  slower). Isolated by elimination (`notes.md`, "leakA / leakB"). A
  burn-bug session investigates it:
  `/shared/claude/burn-bug/handoff/handoff-untied-leak.md` (the cause can
  also be in burn-stack or burn-mamba).
- The quaternion L3 × V6 model has untied tensors: measure its context use
  with `--valid-stories 16` only.

## Next steps

1. **Finish M = 12000:** the decay of h16 × g2 (`p12k-h16g2-lr1`, stable
   12000, done; its `-d` was stopped at 12078 and is unsaved: run a fresh
   decay branch of 2400 from it with `cfg/dec-p12k-h16g2-lr1`, for example
   `branch.sh p12k-h16g2-lr1 p12k-h16g2-lr1-d2 dec-p12k-h16g2-lr1 2400 --
   <packing flags>`), and r128 (`extend.sh p8k-r128-lr1 p12k-r128-lr1 4000
   2400 --max-lr 1e-3 -- <packing flags>`). Measure the context use after
   each (flex, between GPU runs, `ctx5/`).
2. **The 4th slot:** build on the two levers (heads → loss, groups →
   context use). h16 is at the VRAM cap. Its extra memory is the per-head
   intra-chunk tensors (∝ heads × chunk). A 64-token chunk would halve them,
   but the example sets the story alignment to the chunk, so the fill (the
   chars per step) would change. A `--pack-align` decoupled from the chunk
   (a multiple of it) would keep the packing fixed.
3. **LR:** 1e-3 holds to 12000. Test the next point later (for example 5e-4
   from 12000).
4. **The next milestone:** M = 16000 or 20000 for the pool, each extension
   bounded (~1–1.5 h per run at h16's speed).

## Code and repo state

- burn-mamba `main` at `4b35cb3` (packed training) + **uncommitted** example
  work: `examples/tiny-stories/{cli.rs, inference.rs, main.rs, README.md}`
  (`--ssd-path …:L`, `--prompt`, `--continuations`, `--temperature`,
  `--context-use`, `--context-batch`), new `context.rs`, `context/tests.rs`,
  `inference/tests.rs`, `test_net.rs`, and the waiting rule in `CLAUDE.md`.
  `cargo test --example tiny-stories` (flex): 2 pass, 1 ignored
  (`TS_CHECKPOINT=<run> … -- --ignored` checks a checkpoint's decode). The
  overview for `files.md` / `examples/README.md` is in `/shared/claude/burn-mamba/tmp.md`.
- `bin/ts-cuda` = `bin/ts-cuda-v11-ctxb` (all the flags above).
