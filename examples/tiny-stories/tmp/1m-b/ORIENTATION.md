# ORIENTATION: tiny-stories ≤1M-param search, part b

Workspace: `examples/tiny-stories/tmp/1m-b/` (the paths below are relative
to it). Read `handoff.md` first.

## Files

- `handoff.md`: the state, the pool, the running jobs, the next steps.
- `ORIENTATION.md` (this file): the goal, the rules, the protocol, the
  metric, the tools.
- `notes.md`: the results and the findings.
- `far-use.md`: the plan for auxiliary objectives that target the far use.
- Git tracks the `*.md`, `*.sh` and `*.py` files of this directory
  (`examples/tiny-stories/.gitignore`). It does not track the other files:
  `runs/`, `cfg/`, `bin/`, `snap/` (snapshots), the `ctx*/` and `ceval*/`
  copies, the `.txt` logs and tables.
- `tmp.md`: the notes of the running session. At the end of a session they
  go to the files above, and `tmp.md` is empty again.
- The old chronological logs (`archive/`) are in git only. Read one with
  `git show 8a67461:examples/tiny-stories/tmp/1m-b/archive/notes.md` (also
  `handoff.md`, `ORIENTATION.md`).

## Goal and ceilings (user)

- The lowest held-out **valid bits/char** of the `tiny-stories` example,
  **and long-context coherence** (the **far use** of `--context-use`, see
  Metric). Measure the far use of every screen and every milestone
  (`ctx.sh -g -n 2048`, between GPU runs), and rank the candidates by both.
- Use the most of both ceilings (params and VRAM). Where a shape has VRAM
  left, add cheap capacity (expand, virtual layers, …). Virtual layers
  (V > L) always take the untied tensors of mnist-class (`gen.py untie=1`).
- The ceilings are **training steps, params (< 1M) and VRAM**. Wall clock is
  not a ceiling, unless a candidate is much slower. Log ms/step anyway.
- VRAM (best effort, it can change): **the process (host excluded) stays
  under 3.6 GiB (3686 MiB). For the search, under 3.8 GiB (3891 MiB) is
  fine.** Whether the old NaN ceiling still applies on burn `main` is not
  known.
- The process = the pools' `reserved` (the `prof` line) + ~140 MiB of CUDA
  context. That holds only with `--max-vram`, which also flushes the pages
  that cubecl queues. The scripts add `-- --max-vram 3750` (3.8 GiB). Pools
  above 3545 MiB put the process over 3.6 GiB: note it. nvidia-smi cannot
  give the memory of one process here, and the host's share varies.
- **Ask the user before a single run of several hours.**
- The architecture can change a lot. The example docs need no update during
  the search.

## Working rules (user)

- **A pool of 4 candidates** (model configs). Extend them in bounded steps
  and compare them at matched steps. Replace a clearly worse one with a
  promoted one, even if it lags. Never push one candidate alone.
- **Claude decides** evictions and fresh candidates. Do not ask the user
  questions that block progress: decide, log the reason in `notes.md`, and
  continue.
- New ideas: breadth-first, one run at a time, no blind sweeps. After each
  run, pick the single most useful next run. Some features pay only late
  (`agg.py` shows the rank per bucket).
- **Models before seeds**: test a new variant on one seed first. Two
  different models are worth more than two seeds of one model. Add a seed
  only when a decision needs it.
- Read the results with `summ.py`, `agg.py`, `tail`, `grep ^prof`. Never dump
  big files.
- **Waiting**: no foreground loop or `sleep`. Wait in the background
  (`wait.sh`, a chain script that prints a summary, or a `Monitor`), then
  return the turn. Match processes with `pgrep -x ts-cuda`. Kill by exact PID
  or `pkill -x` only (`-f` also matches the waiters and the shell).
- Never `rm` artifacts. Every run gets a fresh `runs/<name>`.
- **Every new directory gets a `build.txt`** (the burn-mamba version, commit
  and diff of its binary). The scripts do it. For a directory made by hand,
  run `./stamp.sh <dir>`.
- Change the hand-written files (`notes.md`, this file, crate sources) only
  with Edit/Write. Scripted edits are allowed for scripts and configs. Do not
  edit a script while a run uses it.
- One GPU job at a time. **One host job at a time** (16 GB of RAM: flex
  evaluations, dry runs, builds). A flex job slows a GPU training, so run
  the flex evaluations between GPU runs.
- The CPU governor is dynamic: compare speeds only inside one sitting.

## Protocol (every candidate)

- **Packed training**: `-- --pack 4224 --pack-rows 2 --ssd-path recalc:128`
  (`recalc:256` at u2: the chunk is 128 tokens). Whole stories per row,
  ~7.5k scored chars per step.
- **wd 0.1** for Muon and AdamW (`gen.py … wd=0.1`).
- **LR**: 4e-3 constant to step 4630, then 1e-3 (`--max-lr 1e-3` on the
  extension).
- **Milestones** in fresh steps: stable 2630 / 4630 / 8000 / 12000, each with
  a decay branch of M/5 (600 / 1000 / 1600 / 2400). `screen.sh` runs the
  first, `extend.sh` the others. Each member has a fresh packed history, so
  `agg.py` compares the members bucket by bucket.
- **Rank by the decayed (`-d`) valid.** A constant-LR reading misleads (the
  decay recovers 0.04–0.12). The noise is ±0.005–0.008.
- The *train* loss in the logs is a per-window average (for curves only).
  Compare train buckets only between runs with the same wd and LR schedule.
- The single-run LR schedule (a cosine fit to the step-down points) runs from
  zero only when the search ends (user).

## Metric

- Valid bits per **character** over the fixed 256-story valid split (bs 8,
  seq 1024, runlen 1: 190,070 chars). The state carries across the windows
  of a story. **Keep `seq × runlen = 1024`**, or the metric changes. The
  validation stays unpacked.
- **Context use** (`--context-use`, `examples/tiny-stories/context.rs`): on
  the valid stories of ≥ 512 chars, the cost of chars 256..512 after their
  own first 256 chars against after the chars of another story. **Far use**
  (chars 64..256 of the continuation) is the plot-level signal.
  - Use 2048 stories on the GPU: `ctx.sh -g -n 2048 -r ctxg <run>` (~78 s,
    1968 stories; append the line to `ctxg.txt`). The default (244 stories,
    flex, `ctx5/`) is too noisy to compare models (s.e. of the gain 0.006
    against 0.002). A GPU job: run it between GPU runs or in the queue.
  - `ctx.sh` works on a copy of the run dir (the app rewrites its configs
    at startup). It also takes a snapshot dir (`snap/<run>/<step>`).
  - `context-stories.jsonl` has the scores of each story.
    `ctxstat.py <dir>…` gives the mean ± s.e. and the paired difference to
    the first dir (a paired s.e. is ~0.001).
- **Gain** = `bits(C | no prompt) − bits(C | own)` of the far part, and the
  penalty = other − none (use = gain + penalty). **The stable metric is the
  own far bits, paired with the control.** The gain and the use carry the
  noise of their baselines (~0.01 between seeds), and a model can raise the
  use by the penalty alone (`notes.md` §4). The log also has the coherence
  boosting table.
- Samples: `--inference` on a copy of the run dir, with the example flags
  `--prompt <TEXT>` (once per prompt), `--continuations 3`, `--temperature
  T`. Run the user's two prompts (`infer-*` dirs, see their `README.txt`)
  again on each new best.

## Tools

- `bin/ts-cuda`: the example (`default,backend-cuda`, flex included, so
  `BURN_DEVICE=flex bin/ts-cuda …` runs on the host). `bin/ts-cuda.txt`
  names its build. `./build.sh` (a host job) builds the checkout, keeps it as
  `bin/ts-cuda-v<version>-<commit>[-dirty-<hash>]` + `.txt` (the commit, the
  burn revs, the full diff), and installs it. The scripts copy that `.txt`
  into each new directory (`build.txt`), and their `==` log lines name the
  tag.
- The builds before `v0.64.0` (`ts-cuda-v0…v11`, `ts-flex`) leak host memory
  when they evaluate a model with untied tensors. The builds from `main` do
  not.
- **Run noise.** With 2 dataloader workers, two same-seed runs train on
  different batch orders from step 2, and dv moves by ~0.01. With 1 worker
  (the `*-w1` cfgs: `num_workers` 1, the same speed) the order is fixed:
  the per-step losses agree to ~1e-7 for ~50 steps, then float noise grows
  to a mean gap of ~6e-3 per step by step 250. The noise of a final dv on
  1 worker is not measured yet. Compare a treatment only with a control on
  the same harness and launcher.
- CLI: burn-stack's `AppArgs` flags first (`-a -c -m -b -i --resume --seed
  --max-lr --valid-batches …`), then after `--` the example flags (corpus,
  packing, `--pack-align N` (a multiple of the chunk: chunk 64 can train on
  the rows of chunk 128), `--ssd-path recalc|serial|minimal[:L]`,
  `--profile N`,
  `--max-vram`, prompts, `--context-use [--context-batch N]
  [--context-stories N]`, the contrast `--contrast* …` and
  `--contrast-eval N`, see `far-use.md` and `notes.md` §5).
  `bin/ts-cuda --help` and `bin/ts-cuda -- --help` list them.
- `gen.py <cfg> k=v …` → `cfg/<cfg>/` (the keys are `DEFAULTS` in the file).
  A key with a string default keeps `None` as a string.
- `params.sh <cfg>` → the param count (a 2 s flex dry run). `probe.sh <cfg> [args [-- ex args]]` →
  reserved, total and ms/step of 10 steps (`probes.txt`). Probe before a run
  that grows the activations. `speed.sh` → a packed throughput line
  (`speeds.txt`).
- `screen.sh <run> <cfg> <T1> <T2> [args]`: `runs/<run>` at constant LR for
  T1 steps, then `runs/<run>-d`, a cosine decay to max/100 over 0.9·T2.
  Launch with `nohup ./screen.sh … > /dev/null 2>&1 &`.
- `snapscreen.sh <run> <cfg> <T1> <T2> <E> [args]`: `screen.sh` with a
  checkpoint every E steps, each copied whole to `snap/<run>/<step>/` (the
  decay branch too). b01 and c03 used it with the `*-w1` cfg and E = 250.
- `ceval.sh [-g] [-n N] [-r root] <run | snap/<run>/<step>> [-- flags]`:
  `--contrast-eval` on a copy (`ceval*/`): the push and the `Δ` of each cut
  × negative × J, with no backward. A GPU job (~1 min).
- `extend.sh <src> <new> <N1> <N2> [args]`: from a stable checkpoint (never a
  `-d`), +N1 steps (`--resume`, and `--max-lr` sets the new constant LR),
  then a decay branch `runs/<new>-d` of N2.
- `branch.sh <src> <new> <cfg|-> <nb> [args]` continues a run.
  `branch.sh <src> <new> - 0 --valid-batches 32` validates a checkpoint
  again (the full valid set).
  `decay.py <src> <cfg> <D> [min_lr]` writes a decay config (WSD by config:
  `CosineAnnealing` with `min_lr = max_lr` is warmup + constant).
- `wait.sh <run> [runs…]`: the silent waiter (one summary, an OOM guard at
  5400 MiB total).
- `queue.sh` (launch with `nohup … &`) runs the lines of `queue.txt` in
  order, one at a time, and logs `start`/`done`/`fail` into `queue.log`. To
  add, drop or reorder the pending runs, edit `queue.txt` (not the script).
  `qwait.sh` (in the background) prints once when the next line ends: its
  summary and the errors, with the OOM guard of `wait.sh`.
- `agg.py [-b N] [runs]`: one row per run, ranked by dv (buckets, slope,
  ranks). `summ.py [-c N] <run>…`: steps/s, buckets, every validation.
- `args.sh` (sourced by the scripts) adds `--max-vram 3750` unless given.
- Data: `stories=262144` (210M chars, cached). A `--resume` shuffles the rest
  of the epoch again. `step` in `metrics.jsonl` counts schedule steps, and
  `-b` counts optimizer steps. The first ~35 s of a process are the corpus
  load and the JIT.
