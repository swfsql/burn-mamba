# ORIENTATION: hnet-stories ≤1M-param search

Workspace: `examples/hnet-stories/tmp/1m-a/` (the paths below are relative
to it). Read `handoff.md` first. This search replaces the flat search of
`../../../tiny-stories/tmp/1m-cased/` (the cased world, the same metric).

## Goal and ceilings (user)

- The lowest valid bits/char of the cased text (`bits` = `primary` +
  `case`) with an H-Net whose stacks are all Mamba-3. The number of stages,
  and the layer and block settings of each stage, are hyperparameters.
- The ceilings: training steps, params (< 1M), VRAM (the process under
  3.6 GiB preferred, under 3.8 GiB for the search). Wall clock is not a
  ceiling. Log ms/step anyway.
- **A long run is a sequence of extensions of < ~1 h each** (user), so that
  a partial result can stop it.
- Library changes (burn-stack, burn-mamba) are allowed. Claude can commit
  both (the `[patch]` of burn-stack stays in the commits of burn-mamba).

## Protocol

- **Unpacked** (an H-Net takes no packed rows): seq 1024, `runlen=1`,
  frontier Always, **bs 10**: ~7450 scored chars/step, as the packed
  2×4224 rows of tiny-stories (~7.5k). The fill is 0.73. 12% of the stories
  are longer than 1024 and lose their tail.
- The metric: the valid split of tiny-stories (256 stories, their first
  1024 chars, 190,070 chars). Keep `seq × runlen = 1024`.
- The LR and the milestones are those of `../../../tiny-stories/tmp/1m-b/
  ORIENTATION.md` (the trunk at constant LR, a decay leaf per milestone,
  rank by the decayed valid `dv`). Each stage trains at `lr · λˢ`
  (`lrm`). Short screens (600 + 200) predicted the extended order in 1m-b.
- The flat reference: `tiny-stories/tmp/1m-cased/runs/h16-s2` (h16, packed).
- **`--grad-checkpointing` everywhere** (user): `args.sh` adds it to every
  training script unless `--no-gc` (the same losses in tiny-stories, −22%
  VRAM, +2–5% time). The screens before 2026-10-09 04:10 ran without it.
- **The budget table of the knobs**: `knobs.sh` (one change at a time from
  the base, the params and the probe lines) and `knobs.py` (Δparams,
  Δreserved MB, MB per kparam). The VRAM-heavy, param-cheap knobs fill the
  VRAM that a shape leaves.
- **Virtual layers** (user): repeat the middle real layers more than the
  edge layers (`msched=0,1,1,1,1,2`), mostly when not every real layer
  repeats. They always take the untied tensors of mnist-class, in the
  block config and in the layer (`gen.py muntie=auto`, the default, does
  it).
- **Compare at matched budgets** (user). A change can be cheap in params
  and heavy in VRAM (the state rank, MIMO, virtual layers, less
  compression), or the reverse. Log the params and the reserved MB of each
  variant against its base. A variant that frees budget (Real1D stages:
  −8k params, −566 MB) has not had its fair test until it spends that
  budget. **Every shape should use the most of both ceilings** (user,
  again 2026-10-09). The recipe of a full-budget shape (`F-*`):
  1. Dose the VRAM knob so the steady reserved is ≤ ~3500 MB (the process
     < 3.6 GiB with the ~100 MiB CUDA context). With gc, the steady
     reserved ≈ 1.06 × the step-10 probe (L-N2: 2086 against 1962).
  2. Then fill the params with `fit.py` (the main MLP width `mmlp` at
     `mlpmul=8`, VRAM-free, ~10k per step) to ≥ ~990k. The main width
     `md` steps by ~48k and must stay ≥ the stage width.
  The reference is `F-base` (b + `mmlp` 32, 998k).
- **Rank by the recall too**: `ctx.sh -g -n 2048` (`ctxg.txt`). The valid
  bits hide it. The re-mention first letter (the choice of the entity)
  moves against the bits with the compression (`notes.md`).

## Tools

The scripts of `../../../tiny-stories/tmp/1m-cased/` (see
`../../../tiny-stories/tmp/1m-b/ORIENTATION.md`), with these changes:

- The binary is `bin/hn-cuda` (`./build.sh`). Match it with
  `pgrep -x hn-cuda`.
- `gen.py <cfg> k=v …`: an H-Net config (the header lists the keys).
  Per-stage keys take `/` lists: `S=2 d=64/128 enc=1 dec=1`.
- The example flags: `--ssd-path`, `--main-ssd-path`, `--profile N`,
  `--max-vram`, `--grad-checkpointing`, `--chunk-log N` (the chunk ratio of
  each stage, against its target), `--continuations`, `--show-chunks`.
- `ctx.sh` / `ctxstat.py`: the context probe (`--context-use`), as in
  tiny-stories.
- **Kaggle** (a second runner, T4 × 2, the same VRAM accounting):
  `kaggle/kqueue.txt` holds the lines of a batch (each line a job, the
  steps of one job joined with `&&`; the two GPUs take the lines in
  parallel). `kaggle/kpush.sh [hours]` pushes it (the binary, the scripts,
  `cfg/`, the source runs, as `payload.tgz.bin` in a new private dataset
  `hnet-pay-<ts>`: Kaggle's unpacking of a `.tar.gz` can stall for hours)
  and starts the kernel; `kaggle/kpull.sh` gives
  the status and, when the kernel ends, merges its runs into `runs/` (each
  with a `host.txt`). One kernel at a time. The token stays in the
  gitignored `.kaggle/access_token` of the repo (`kaggle/kg.sh`).
