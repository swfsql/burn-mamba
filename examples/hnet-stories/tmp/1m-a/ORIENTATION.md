# ORIENTATION: hnet-stories ≤1M-param search (1m-a)

Workspace: `examples/hnet-stories/tmp/1m-a/` (the paths below are relative
to it). Read `handoff.md` (the state, the next steps), then this file, then
`notes.md` (the findings). `tmp.md` collects the results of the running
session. At the session end they go into `notes.md`, compressed, by topic.

## Strict rules (user): never drop, never soften

These rules come from the user. Earlier rewrites of this file lost some of
them, so:

- **Copy this section verbatim** into every rewrite, compression or handoff
  of this file. Never shorten, merge, soften or move a rule out of it.
- A new strict rule from the user goes here, with its date.

1. **Both ceilings, or the run is useless** (2026-10-10). A candidate is
   near **both** ceilings, not only under them:
   - the params: ≥ ~990k and < 1M,
   - the VRAM of the process: near the ceiling (< 3.6 GiB preferred,
     < 3.8 GiB for the search).

   A result far from either ceiling is useless: do not run it, do not rank
   by it. Many probes and doses are fine.
2. **Never decide by `nvidia-smi`** (2026-10-10). It counts the host too,
   and the host usage moves. Measure the VRAM from inside the process (see
   [VRAM](#vram)).
3. **Read logs only through the scripts or by minimal reads**:
   `summ.py <run> | grep valid@`, `ctxstat.py`, `longstat.py`, `tail -n`,
   `grep ^prof`, `probes-B.txt`, `doses.txt`, `diff cfg/a/args.txt
   cfg/b/args.txt`. **Never dump a big file in full** (a log,
   `metrics.jsonl`, the output of `kpull.sh`). Every run appends (`>>`) its
   stdout to a log.
4. **Never wait on `pgrep -f <pattern>`**: the command line of the waiter
   holds the pattern, so `pgrep -f` always finds the waiter itself, and the
   wait never ends. Wait on a line in a file (`queue.log`, a log,
   `probes-B.txt`), on `pgrep -x hn-cuda`, or on a PID (`kill -0 <pid>`).
   Kill by exact PID.
5. **No foreground waits**: no foreground loop or `sleep`. Wait in the
   background (a waiter script or a `Monitor`), then return the turn.
6. **A long run is a sequence of pieces of < ~1 h each**, so that a partial
   result can stop it.
7. **Never `rm`.** `cp` and `mv` are allowed. Every run gets a fresh
   `runs/<name>`, and every new directory a `build.txt` (`stamp.sh`).
8. **Edit the hand-written files only with Edit/Write** (notes, scripts,
   crate sources): no `sed -i`, no heredoc into a file, no python write.
9. **A new config field**: add it to every saved config that loads again.
   Change `gen.py`, then regenerate `cfg/` from each `args.txt`. Edit the
   `dec-*` configs and the run dirs by hand. Never make the field an
   `Option` for compatibility: a Burn config default does not apply when a
   config loads. Stop the runners first: the next phase of a running run
   loads its config with the new binary.
10. **No auto-memory writes.** Durable notes go into these files.
11. **Semi-supervised**: no forms, no questions that block progress. Claude
    decides. Questions and suggestions are welcome. Log each decision and
    its reason in `tmp.md`.
12. **Keep the trapezoid and the B/C bias on**: this Mamba-3 has no short
    conv, and they take its place.
13. **Commits** (both repos, at good points): `git -c user.name="Thiago
    Machado" -c user.email=swfsql@gmail.com commit -F /tmp/<msg>`, the
    message ends with the `Co-Authored-By` trailer.
14. **No auto-fill of a free budget** (2026-10-10):
    - The params: fill the rest with the main MLP (`mmlp`, `fit.py`) only
      when the shape is already close to the param ceiling. A larger gap
      gets a deliberate design change.
    - The VRAM: **never** fill the free VRAM automatically. There are too
      many ways to spend it. Each candidate spends it by an explicit
      design choice, logged with its reason.

## Goal

- The lowest bits/char of the cased text (`bits` = `primary` + `case`) on
  the whole valid (`wv`), with an H-Net whose stacks are all Mamba-3. The
  stages, the layers and the blocks of each stack are hyperparameters.
- The ceilings: the training steps (matched with the flat), the params,
  the VRAM (strict rule 1). Wall clock is not a ceiling; log ms/step.
- **The reference is the flat h16-V7u** (`notes.md` §1). Library changes
  (burn-stack, burn-mamba) are allowed.

## Working notes

- Models before seeds. Rank by long runs. Add a seed where a decision needs
  it. **One change per screen** against its base.
- Name a variant by its change from its base (`R-N15 × main d152 × 4`).
  `catalog.py` prints the shape, the params, the dose and the role of each
  model (`python3 catalog.py > catalog.txt`; the roles are in the script).
- One GPU job at a time here (a probe or a dose runs alone), one host job
  at a time (builds).
- The long H-Net runs go to Kaggle. Here: the flat runs, the probes, the
  doses, `long.sh`, `ctx.sh`, and a long run when this GPU would idle.
- The dcg hook blocks in a command line: text like `<owner>/x`,
  `sleep N; …` chains, a `>` redirect to a variable path. Use a literal
  path or a script file.
- `queue.sh` changes `queue.txt`: read it again before a Write, and start
  `queue.sh` only after the Write. `pgrep -x queue.sh` does not match: use
  `ps -p $(cat queue.pid) -o args=`.

## Protocol

- **Whole stories, full backprop** (user, 2026-10-09): the length buckets
  1024,1536,2048,3072,4160 (`gen.py` default), one story per row of the
  shortest slot that holds it, ~7450 scored chars per batch (as the flat's
  packed 2 × 4224 rows). The H-Net runs before 2026-10-09 20:40 are
  invalid.
- **The LR** (the flat's plan): a ramp to 4e-3 over 128 steps, 4e-3 to
  4630, then 1e-3, then a cosine 1e-3 → 5e-4 over 8000–12000. Each stage
  trains at `lr · λˢ` (`lrm=auto`).
- **The trunk schedule is in the config**, as in the flat: `gen.py`
  writes the protocol (`stages`), and every trunk holds it. `extend.sh`
  follows it with **no** `--lr`; `--lr` is only for a deliberate,
  logged step-down. `lrprof.py` guards `extend.sh` and `kpush.sh`, and
  `lrprof.py <run>…` prints where a run leaves the protocol.
- **The milestones**: stable 2630 / 4630 / 8000 / 12000, each with a decay
  leaf of 600 / 1000 / 1600 / 2400 (3230 / 5630 / 9600 / 14400).
  `screen.sh` runs the first, `extend.sh` the others. Nothing continues
  from a leaf (`-d`).
- **Rank by the `wv` of the leaf**, with the long stories (`long.sh`) and
  the recall (`ctx.sh`) beside it. The flat's `wv` comes from `long.sh -t`.
  The seed noise is ~0.008: one run ranks only a gap ≳ 0.01.
- Every H-Net run: `--grad-checkpointing`, `--max-vram 3750`,
  `--vram-cleanup-every 1` (`args.sh` adds them), `--ssd-path recalc:32`.
- **The flat h16-V7u** lives in `../../../tiny-stories/tmp/1m-cased` (its
  own `gen.py`, `screen.sh`, `extend.sh`, `probe.sh`, `lrprof.py`,
  `bin/ts-cuda`), runs `h16-V7u-s2` (3230; seed 1: `h16-V7u-s1-s2`),
  `-x4630`, `-sd2630`, `-x8000`, `-x10000`, `-x12000` (14400). Its flags:
  `--pack 4224 --pack-rows 2 --ssd-path recalc:128 --profile 100
  --grad-checkpointing --vram-cleanup-every 1` (`extend.sh` needs no
  `--pack`). Run it from this queue: `cd ../../../tiny-stories/tmp/1m-cased
  && ./extend.sh …`. For the flat, a probe is a dose (× 1.0).

## VRAM

- `bytes_reserved` (the `prof` lines, `--max-vram`, `probe.sh`,
  `dose.sh`) is the sum of the pool pages (~168 MB each). The `prof` line
  prints their high-water mark as `peak X MB`. A dropped page waits for an
  explicit cleanup; `--vram-cleanup-every 1` cleans up before every window.
- The CUDA context and the kernels are outside the pools: **the process ≈
  `peak` + ~140 MiB**.
- **The dose target**: a `peak` of **~3400–3550 MB** (the process then
  stays < 3.6 GiB = 3865 MB). One page over (3592, 3663) is still < 3.6 GiB.

## Shapes and budgets

- **The dose** (strict rule 1): `probe.sh <cfg>` (~1 min) is for the coarse
  search only (a run holds × 1.0 to × 1.34 of it). `dose.sh <cfg>` runs the
  first 520 steps of the real run (~10 min): dose every shape before it
  trains. The params: design near the cap; `fit.py` fills only a small rest
  with the main MLP (strict rule 14). The widths must not shrink inward.
- **The VRAM knobs** (`notes.md` §7): the stages set the peak (the heads ×
  the state rank). **Use a power of 2 for a state rank** (user): another
  rank can cost more VRAM than the next power of 2.
- **Virtual layers** (in a stage: `eV`/`dV`; in the main: `mV`):
  - Untie every virtual application, in every stack (user, 2026-10-11):
    cheap in params, usually a better score. A real layer is fully untied;
    a virtual application holds its own copy of the `untied` tensors only.
    `gen.py` does it by default (`suntie=auto`, `muntie=auto`).
  - The schedule is a test variable: Cyclic beat the middle placements in
    the flat (two seeds, `notes.md` §1). The H-Net main schedule is open
    (R-N15-r64V7 against its Cyclic twin R-N15-r64V7c).

## Tools

- **Binaries**: `bin/hn-cuda` (`./build.sh`, a host job; it keeps each
  build as `bin/hn-cuda-<tag>` + `.txt`). The flat's `bin/ts-cuda` (its
  own `build.sh`, ~3 min). `bin/hn-cuda -- --help` lists the flags.
- **Configs**: `gen.py <cfg> k=v …` → `cfg/<cfg>/` (the keys and defaults
  are at the top; per-stage keys take `/` lists). `fit.py <cfg> <key>
  [from=V] [step=S] k=v …` fits one key to the param cap. `params.sh <cfg>`
  counts the params (~2 s, flex).
- **Runs**: `screen.sh <run> <cfg> <T1> <T2> [args]` (a trunk of T1, a
  leaf of T2). `extend.sh <src> <new> <N1> <N2> [--lr LR [--span A-B]]
  [args]` (+N1 from a trunk, then a leaf of N2; `N2 = 0`: a trunk piece).
  The LR tools: `lrseq.py`, `stage.py`, `decay.py`, `branch.sh`,
  `lrprof.py` (the protocol check; `--fill`, `--guard`, `--batch`).
- **The queue**: `queue.sh` runs `queue.txt` one line at a time and logs
  `start` / `done` / `fail` / `empty` into `queue.log`. Start it with
  `setsid nohup ./queue.sh > /dev/null 2>&1 < /dev/null &`. Waiters:
  `qwait.sh`, `qall.sh`, `after.sh`, `wait.sh`. `gate.sh <cfg> [lo] [hi]`
  exits 0 when the last dose of `<cfg>` is in 3380–3670 MB: `if ./gate.sh
  A; then ./screen.sh …; fi`.
- **Reading**: `summ.py <run>…`, `agg.py`, `cool.py`.
- **Probes**: `long.sh -g [-t] <run>` → `long.txt` (`longstat.py [-w]
  long/A long/B …`: paired against A). `ctx.sh -g -n 2048 -r ctxg <run>` →
  `ctxg.txt` (`ctxstat.py ctxg/A ctxg/B …`). `probe.sh` → `probes-B.txt`,
  `dose.sh` → `doses.txt` (both with `-- --ssd-path recalc:32 --chunk-log
  0`). `infer.sh <run> <dir>`: samples on flex.
- **Kaggle** (two T4s): `kaggle/kqueue.txt` holds a batch (one job per
  line, its steps joined with `&&`; the GPUs take the lines in parallel).
  `kaggle/kwait.sh <hours> <log>` (in the background) pushes it
  (`kpush.sh`: the LR check, then the binary, the scripts, `cfg/`, the
  source runs), waits, merges (`kpull.sh`) and queues `long.sh` for the
  pulled `-d` runs (it misses some: check each). One kernel at a time;
  `kpush.sh` empties `kqueue.txt`. The CLI can vanish after an environment
  reset (the user installs it). The token is in the gitignored
  `.kaggle/access_token` (`kaggle/kg.sh`): never print it. `owner.txt`,
  `kqueue.txt` and `out/` stay out of git.
