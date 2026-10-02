# ORIENTATION — tiny-stories ≤1M-param search, part b (read this first after a reset)

Workspace: `examples/tiny-stories/tmp/1m-b/` (paths below are relative to it).
**Resuming? Read `handoff.md` first** (state, candidate pool, next steps).
`notes.md` is the log of this part (search it, do not read it whole). `../1m/`
(`handoff.md`, `notes.md`, `runs/`) is older history: read it only for a
detail.

## Goal and ceilings (from the user)

- Lowest held-out **valid bits/char** of the `tiny-stories` example.
- The ceilings are **training steps, params (< 1M) and VRAM** only. Wall
  clock is not a ceiling, unless a candidate is much slower (then it counts
  again). Log `ms/step` anyway.
- VRAM: **the process itself (host excluded) stays under 3.6 GB.** The
  process enforces it: `-- --max-vram 3480` (MiB of cubecl pools, + ~100 MiB
  CUDA context), which every script adds (`args.sh`). Read memory from the
  `reserved` of the `prof` line (cubecl's `bytes_reserved`), not nvidia-smi.
- **Ask the user before a single run of several hours.**
- The architecture can change a lot. A `model.rs` can live here if useful.
- The docs of the example need no update during the search.
- The CPU governor is dynamic (schedutil or powersave), so wall-clock numbers
  drift. Compare speeds only inside one sitting.

## Working rules (from the user)

- **A pool of 4 extending candidates** (see `handoff.md`). Extend them in
  steps and compare them at matched steps. Replace a candidate that is clearly
  worse by a promoted one, even if it lags in extensions. Do not push one
  candidate alone to its loss floor. M has no limit, but each extension
  stays bounded (no multi-hour run).
- **Claude decides** evictions (an evicted candidate can return) and fresh
  candidates. **Do not ask the user questions that block progress**: decide,
  log the reason in `notes.md`, and continue.
- New ideas: breadth-first, never blind sweeps. One run at a time. After each
  run, extract what it had to answer, then pick the single most useful next
  run. Screens (packed): fresh 2630 + 600 (`screen.sh`, ~24 min).
- Some features pay only late: weigh that before a cut on an early reading
  (`agg.py` shows the rank per bucket).
- Append every run's stdout (`>>`) to a log (the scripts do it).
- **Context**: read results with `summ.py`, `agg.py`, `tail`, `grep ^prof`.
  Diff configs (`diff cfg/a/args.txt cfg/b/args.txt`). Never dump big files.
- **Waiting**: never keep the turn alive with a foreground loop or `sleep`
  (see `CLAUDE.md`). Wait in the background (`wait.sh`, a chain command that
  prints only a summary, or a `Monitor`), then return the turn. Match
  processes with `pgrep -x ts-cuda` (never `pgrep -f`).
- Never `rm` artifacts. Every run gets a fresh `runs/<name>`.
- Hand-written files (crate sources, `notes.md`, this file) change only
  through the Edit/Write tools. Scripted edits are allowed for mechanical
  migrations of scripts and configs. The scripts write their own outputs.
- Do not edit a script while a run uses it (bash reads a script as it runs).
- One GPU job at a time.
- **The host has 16 GB of RAM: one host job at a time** (flex evaluations,
  dry runs, builds). A flex job also slows a running GPU training (its host
  side needs the CPU): run flex evaluations between GPU runs. Kill by exact
  PID or `pkill -x` only: `pkill -f` matches the waiters too.

## Tools

- `bin/ts-cuda` (`default,backend-cuda`, so flex is compiled in too): burn-mamba
  `4b35cb3` + the uncommitted example flags of part c (`bin/ts-cuda-v11-ctxb`;
  burn `8f08fdad` + the swfsql fork patches). `BURN_DEVICE=flex bin/ts-cuda …`
  runs it on the host. `bin/ts-flex` is the old `b58a0ab` build (only
  `params.sh` uses it). Rebuild after a crate change:
  `cargo build --release --example tiny-stories --features default,backend-cuda`,
  then copy with `cp` to a new name and `mv` it over `bin/ts-cuda` (a running
  binary cannot be overwritten).
- CLI: burn-stack's `AppArgs` flags first (`-a -c -m -b --max-seconds --resume
  --seed --valid-every --valid-batches --checkpoint-every --max-lr
  --batch-size --adamw/--sgd/--muon --no-graph`), then after `--` the
  example's own flags: corpus knobs (`--seq-len --run-len --frontier-bits
  --no-frontier --train-stories --valid-stories`), packing (`--pack <W>
  --pack-rows <N> [--pack-open <K>] | --no-pack`), `--ssd-path
  recalc|serial|minimal[:L]` (`L` = the chunk length in folded positions),
  `--profile N [--profile-sync]` (the `prof` line also prints the pools'
  `reserved`), `--max-vram <MiB>` (exits with code 3 above it, checked after
  each backward only), inference (`--prompt <TEXT>` repeatable,
  `--continuations <N>`, `--temperature <T>`), and `--context-use
  [--context-batch <N>]` (see Metric). `bin/ts-cuda --help` and
  `bin/ts-cuda -- --help` list them.
- `--max-lr <LR>` on a `--resume` sets the constant LR of the continuation
  (the step-down): `extend.sh <src> <new> <N1> <N2> --max-lr 1e-3 -- …`.
  `decay.py` then decays from that LR.
- `gen.py <cfg> k=v …` → `cfg/<cfg>/{model,training}_config.json` + `args.txt`.
  Keys: `DEFAULTS` in the file (also `trop`, `untie`, `mgrpv`, `sched`).
  Defaults reproduce the shipped example exactly (`cfg/_check2`). `V=0` ⇒ no
  virtual cycling, `mgr=0` ⇒ standard residual, `mlp=0` ⇒ no MLP, `muon=0` ⇒
  AdamW only. A key with a string default keeps `None` as a string.
- `params.sh <cfg>` → the param count (a flex dry run under `params/`).
- `probe.sh <cfg> [args [-- example args]]` → one line: pools reserved,
  total, ms/step of steps 6..10 (~1 min, killed above 5400 MiB total). Probe
  before a run that grows the activations. Results in `probes.txt`.
- `speed.sh <run> <cfg> <steps> [args]` → a packed throughput run: fill,
  chars/step, ms/step after step 20, kchars/s, reserved (`speeds.txt`).
- `screen.sh <run> <cfg> <T1> <T2> [args]` — a fresh screen. Phase 1
  (`runs/<run>`): constant LR for T1. Phase 2 (`runs/<run>-d`): a decay branch
  for T2, the cosine to max/100 over 0.9·T2. T = optimizer steps, or minutes
  with an `m` suffix. Launch: `nohup ./screen.sh … > /dev/null 2>&1 &`.
- `extend.sh <src> <new> <N1> <N2> [args]` — copy a stable checkpoint
  `runs/<src>` (never a `-d` branch), +N1 steps at its constant LR
  (`--resume`), then a decay branch `runs/<new>-d` of N2.
- `wait.sh <run> [runs to summarize…]` — the silent waiter. It waits for
  ts-cuda and `runs/<run>/screen.pid`, logs the peak total VRAM, prints one
  summary. Its total-VRAM guard (5400 MiB) only protects the card from OOM.
- `args.sh` (sourced): `split_args` splits the args at `--` and adds
  `--max-vram 3480` (`VRAM_MAX_MIB`) unless given. `run.sh`, `branch.sh`,
  `screen.sh`, `extend.sh` use it.
- `agg.py [-b N] [runs]` → one row per run ranked by the decayed valid:
  params, ms, reserved, train-bits buckets, the late slope, sv/dv, the rank
  per bucket. Compare train buckets only inside one regime (same seq/bs).
- `summ.py [-c N] <run>…` → steps/s, train-bits buckets, every validation.
- `run.sh <run> <cfg> <max-batches> [args]`, `decay.py <src> <cfg> <D>
  [min_lr]`, `branch.sh <src> <new> <cfg|-> <nb> [args]`: manual runs, decay
  configs and continuations. `branch.sh … - 0 --valid-batches 8` validates a
  checkpoint again.

## Metric and protocol

- Valid bits are **per character**, over the fixed 256-story valid split,
  with the state carried across the windows of a story. The example caps the
  valid set at `seq × runlen` characters per story: **keep `seq × runlen =
  1024`** (190,070 chars), or the metric changes.
- The *train* loss in the logs is a per-window average (good for curves only).
  Compare train buckets only between runs with the same wd and LR schedule
  (wd raises the stable-phase loss).
- **Rank by the decayed (`-d`) valid.** A constant-LR reading misleads: the
  decay recovers ~0.04–0.12 bits.
- Noise: ±0.005–0.008 on the decayed valid of a screen (one seed each).
- **Training is packed** (`--pack 4224 --pack-rows 2`, a 128-token chunk):
  whole stories per row, ~7.5k scored chars per step. The validation stays
  unpacked (the same metric). A fresh packed run matches a mixed history at
  equal chars.
- **Context use** (`--context-use`, `examples/tiny-stories/context.rs`): on
  the 244 validation stories of ≥ 512 chars, the cost of chars 256..512 after
  their own first 256 chars vs after another story's (two pairings: their
  spread, ≤ 0.003, is the noise). "Far use" (chars 64..256 of the
  continuation) is the plot-level signal. ~2 min per checkpoint on flex
  (`BURN_DEVICE=flex`, on a copy of the run dir: the app rewrites its configs
  at startup). A model with untied tensors leaks memory there (see
  `handoff.md`): use `--valid-stories 16`.
- Data: `stories=262144` (210M chars, cached). A `--resume` draws the rest of
  the epoch from a new shuffle (a few repeats are possible).
- LR is WSD by config: `CosineAnnealing` with `min_lr = max_lr` is warmup +
  constant. `decay.py` writes the decay branch (`warmup_steps = S`,
  `total_steps = S + D`).
- `step` in `metrics.jsonl` counts **schedule** steps. `-b` counts
  **optimizer** steps (one per window).
- The first ~35 s of a process are corpus load + cubecl JIT.
