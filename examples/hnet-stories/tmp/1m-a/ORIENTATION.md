# ORIENTATION: hnet-stories ≤1M-param search (1m-a)

Workspace: `examples/hnet-stories/tmp/1m-a/` (the paths below are relative
to it). Read `handoff.md` (the state, the next steps), then this file, then
`notes.md` (the findings). `tmp.md` collects the results of the running
session. At the session end they go into `notes.md`, compressed, by topic.

## Goal and ceilings (user)

- The lowest bits/char of the cased text (`bits` = `primary` + `case`) on
  the whole valid (`wv`), with an H-Net whose stacks are all Mamba-3. The
  stages, the layers and the blocks of each stack are hyperparameters.
- The ceilings:
  - the training steps (matched with the flat reference),
  - the params (< 1M),
  - the VRAM (the process under 3.6 GiB preferred, under 3.8 GiB for the
    search).

  Wall clock is not a ceiling. Log ms/step anyway.
- **Every shape uses the most of both ceilings** (user).
- **A long run is a sequence of extensions of < ~1 h each** (user), so that
  a partial result can stop it.
- Library changes (burn-stack, burn-mamba) are allowed.

## Working rules (user)

- Claude decides: no forms, no questions that block progress. Questions
  and suggestions are welcome. Log each decision and its reason in
  `tmp.md`.
- Models before seeds. Rank by long runs. Add a seed only where a decision
  needs it.
- Name a variant by its change from its base (`R-N15 × main d152 × 4`).
- **Context efficiency**: every run appends (`>>`) its stdout to a log.
  Read results with `summ.py` (`| grep valid@`), `ctxstat.py`,
  `longstat.py`, `tail`, `grep ^prof`. Compare configs with `diff
  cfg/a/args.txt cfg/b/args.txt`. Never dump big files (logs,
  `metrics.jsonl`, the output of `kpull.sh`).
- **Waiting**: no foreground loop or `sleep`. Wait in the background (a
  waiter script or a `Monitor`), then return the turn. Match the binary
  with `pgrep -x hn-cuda`. Kill by exact PID.
  - **Never wait on `pgrep -f <pattern>`**: the command line of the
    waiter holds the pattern, so `pgrep -f` always finds the waiter
    itself, and the wait never ends. Wait on a line in a file
    (`queue.log`, a log, `probes-B.txt`), on `pgrep -x hn-cuda`, or on
    a PID (`kill -0 <pid>`).
- One GPU job at a time here. A ~3 GB VRAM probe does not fit beside a
  training on the 6 GB card, but a `long.sh` or `ctx.sh` probe does. One
  host job at a time (builds).
- Never `rm`. `cp` and `mv` are allowed. Every run gets a fresh
  `runs/<name>`, and every new directory a `build.txt` (`stamp.sh`).
- Edit the hand-written files (notes, crate sources) only with Edit/Write.
- **A new config field** (user): add it to every saved config that loads
  again. Change `gen.py`, then regenerate `cfg/` from each `args.txt`. Edit
  the `dec-*` configs and the run dirs by hand. Never make the field an
  `Option` for compatibility: a Burn config default does not apply when a
  config loads. Stop the runners first: the next phase of a running run
  loads its config with the new binary.
- Commits (both repos, at good points): `git -c user.name="Thiago Machado"
  -c user.email=swfsql@gmail.com commit -F /tmp/<msg>`, the message ends
  with the `Co-Authored-By` trailer. The dcg hook blocks three things in a
  command line: text like `<owner>/x`, `sleep N; …` chains, and a `>`
  redirect to a variable path. Use a literal path or a script file.
- No auto-memory writes (user): durable notes go into these files.

## Protocol

- **Whole stories, full backprop** (user, 2026-10-09): the length buckets
  1024,1536,2048,3072,4160 (`gen.py` default). Each story is one row of the
  shortest slot that holds it. A batch scores ~7450 chars (`bchars`), as
  the packed 2 × 4224 rows of the flat model do. **The H-Net runs before
  2026-10-09 20:40 are invalid** (the first 1024 chars only, `notes.md`
  §2).
- **The LR** (the flat's plan): a ramp to 4e-3 over 128 steps, 4e-3 to step
  4630, then 1e-3, then a cosine 1e-3 → 5e-4 over 8000–12000. Each stage
  trains at `lr · λˢ` (`lrm=auto`).
- **The milestones**: stable 2630 / 4630 / 8000 / 12000, each with a decay
  leaf of 600 / 1000 / 1600 / 2400. `screen.sh` runs the first,
  `extend.sh` the others. The stable runs form a trunk. Nothing continues
  from a decay leaf (`-d`).
- **Rank by the `wv` of the decayed run**, with the recall (`ctx.sh`)
  beside it. From binary `4e42bba3` on, the dv of a bucketed run is `wv`
  (every valid story whole). An older run needs `long.sh -g <run>-d`. A long run has ±0.005 of seed noise.
  A short screen has ±0.009, so it cannot rank strong shapes: use it only
  to drop clear losers.
- Every run: `--grad-checkpointing` and `--max-vram 3750` (`args.sh` adds
  them), `--ssd-path recalc:32` (pins the SSD chunk, which doubles at a
  state rank ≥ 80).
- The flat reference: h16 of `../../../tiny-stories/tmp/1m-cased/runs/`
  (`h16-s2`, `h16-x4630`, `h16-x8000`, `h16-x12000`). Its `wv` is in
  `notes.md` §1.

## Shapes and budgets

- **A full-budget shape** (user): a candidate is near **both** ceilings,
  not only under them. A result far from the VRAM ceiling is useless.
  Many probes are fine.
  1. The VRAM: `probe.sh <cfg>` runs 2 batches of each slot and reports the
     reserved MB after each slot and their maximum (~1.5 min). A run holds
     more, by a factor that depends on the shape: × 1.04 (M-s64r256-L4),
     × 1.17 (R-N15-B5), × 1.32 (S2a-B5, two routers). Use the probe for
     the coarse search. Then `dose.sh <cfg>` (200 steps of the real run,
     ~4 min) gives the steady reserved MB of the run: dose it to
     **~3450 MB**, so that the process stays < 3.6 GiB.
  2. The params: `fit.py` fills the budget to ≥ ~990k (the main MLP `mmlp`
     at `mlpmul=8` costs no VRAM, ~10k per step). The main width `md` must
     stay ≥ the stage width.
- **Keep the trapezoid and the B/C bias on** (user): this Mamba-3 has no
  short conv, and they take its place.
- **Power-of-two state ranks**: another rank costs the VRAM of the next
  power of two, or more. A Complex2D stage above r 64 does not fit.
- **Virtual layers** (user): repeat the middle real layers more than the
  edge layers (`msched=0,1,1,1,1,2`), with the untied tensors (`gen.py
  muntie=auto`). They lost so far.
- The budget table of the knobs: `knobs.sh` / `knobs.py` (`notes.md` §7).

## Tools

- **Binaries**: `bin/hn-cuda` (`./build.sh`, a host job: it keeps each
  build as `bin/hn-cuda-<tag>` + `.txt`). The flat model's `ts-cuda` is in
  `../../../tiny-stories/tmp/1m-cased/bin/` (its own `build.sh`).
  `bin/hn-cuda -- --help` lists the example flags.
- **Configs**: `gen.py <cfg> k=v …` → `cfg/<cfg>/`. The keys and their
  defaults are at the top of the file. Per-stage keys take `/` lists.
  `fit.py <cfg> <key> [from=V] [step=S] k=v …` fits one key to the param
  cap (`python3 fit.py`). `params.sh <cfg>` counts the params (~2 s, flex).
- **Runs**:
  - `screen.sh <run> <cfg> <T1> <T2> [args]`: a stable run of T1 steps,
    then a cosine decay leaf `<run>-d` of T2 steps.
  - `extend.sh <src> <new> <N1> <N2> [--lr LR [--span A-B]] [args]`: from
    a stable checkpoint, +N1 steps (with a step-down or an anneal), then a
    leaf of N2. With `N2 = 0` it is a trunk piece.
  - `branch.sh`, `decay.py`, `stage.py`, `lrseq.py`: the LR sequence
    tools.
- **The queue**: `queue.sh` runs the lines of `queue.txt` one at a time
  (`bash -c`) and logs `start` / `done` / `fail` into `queue.log`. It
  exits when the queue is empty. Check it with `ps -p $(cat queue.pid) -o
  args=`. Start it with `setsid nohup ./queue.sh > /dev/null 2>&1 <
  /dev/null &`. Waiters: `qwait.sh`, `qall.sh`, `after.sh`, `wait.sh`.
- **Reading**: `summ.py <run>…` (speed, train buckets, every validation),
  `agg.py`, `cool.py`.
- **Probes** (after a run, on the GPU beside a training):
  - `long.sh -g [-t] <run>` → `long.txt` (~50 s): the whole valid (`wv`:
    all, chars 0..1024, past 1024) and the long stories (0..1024,
    1024..2048, past 2048). `-t` takes a tiny-stories run directory.
    `longstat.py [-w] long/A long/B …`: the paired differences against A,
    and the profile per 256 chars (`-w`: the whole valid).
  - `ctx.sh -g -n 2048 -r ctxg <run>` → `ctxg.txt`: the recall (far use,
    re-mentions). `ctxstat.py` pairs the runs.
  - `probe.sh <cfg> -- --ssd-path recalc:32 --chunk-log 0` →
    `probes-B.txt` (the bucket protocol).
  - `dose.sh <cfg> [steps] -- --ssd-path recalc:32 --chunk-log 0` →
    `doses.txt`: the steady reserved MB of the run (a GPU job, not beside
    a training). `probes.txt`, `fills.txt` and
    `knobs.txt` are from the old protocol.
  - `infer.sh <run> <dir>`: samples on flex.
- **Kaggle** (a second runner, two T4s, the same VRAM accounting):
  - `kaggle/kqueue.txt` holds the lines of a batch: one job per line, its
    steps joined with `&&`. The two GPUs take the lines in parallel.
  - `kaggle/kwait.sh <hours> <log>` (in the background) pushes the batch
    (`kpush.sh`: the binary, the scripts, `cfg/`, the source runs). Then it
    waits for the kernel, merges its runs (`kpull.sh`), and runs `long.sh`
    on each pulled `-d` run.
  - One kernel at a time. `kpush.sh` empties `kqueue.txt`, so stage the
    next batch after a push ends.
  - The token stays in the gitignored `.kaggle/access_token`
    (`kaggle/kg.sh` reads it). Never print it. `owner.txt`, `kqueue.txt`
    and `out/` stay out of git.
