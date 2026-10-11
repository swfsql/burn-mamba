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
- The ceilings: the training steps (matched with the flat reference), the
  params, the VRAM (strict rule 1). Wall clock is not a ceiling. Log
  ms/step anyway.
- **The reference is the flat h16-V7u** (at both ceilings, `notes.md` §1).
  The older flat h16 is under both ceilings: do not rank against it.
- Library changes (burn-stack, burn-mamba) are allowed.

## Working notes

- Models before seeds. Rank by long runs. Add a seed only where a decision
  needs it.
- Name a variant by its change from its base (`R-N15 × main d152 × 4`).
  `catalog.py` prints the shape, the params and the dose of each model
  (`python3 catalog.py > catalog.txt`). Its role notes are in the script.
- One GPU job at a time here. A probe or a dose runs alone. One host job at
  a time (builds).
- The long runs go to Kaggle (a T4: ~1.17 s/step for the R shapes, this
  card: ~1.05). Here: the probes, the doses, the short diagnostics,
  `long.sh` and `ctx.sh`. The flat runs go here too. A long run goes here
  when Kaggle is busy and this GPU would idle.
- The dcg hook blocks three things in a command line: text like
  `<owner>/x`, `sleep N; …` chains, and a `>` redirect to a variable path.
  Use a literal path or a script file.
- An edit string must not end with a space: the Edit tool drops it.
- `queue.sh` changes `queue.txt` (`sed -i`): read the file again before a
  Write. `pgrep -x queue.sh` does not match (the process is `bash`): use
  `ps -p $(cat queue.pid) -o args=`.

## Protocol

- **Whole stories, full backprop** (user, 2026-10-09): the length buckets
  1024,1536,2048,3072,4160 (`gen.py` default). Each story is one row of the
  shortest slot that holds it. A batch scores ~7450 chars (`bchars`), as
  the packed 2 × 4224 rows of the flat model do. **The H-Net runs before
  2026-10-09 20:40 are invalid** (the first 1024 chars only).
- **The LR** (the flat's plan): a ramp to 4e-3 over 128 steps, 4e-3 to step
  4630, then 1e-3, then a cosine 1e-3 → 5e-4 over 8000–12000. Each stage
  trains at `lr · λˢ` (`lrm=auto`). The best step-down depends on the
  budget (`notes.md` §2): compare two models on the same schedule.
- **The trunk schedule is in the config**, as in the flat: `gen.py`
  writes the protocol (`stages` default), and every trunk holds it. So
  `extend.sh` follows the protocol with **no** `--lr`. `--lr` is only for
  a deliberate step-down off the protocol (`sd2630`, `h6315`), logged.
  Two guards (`lrprof.py`): `extend.sh` refuses an extension that leaves
  the protocol with no `--lr`, and `kpush.sh` checks a batch before the
  push. `lrprof.py <run>…` prints where a run leaves the protocol.
- **The milestones**: stable 2630 / 4630 / 8000 / 12000, each with a decay
  leaf of 600 / 1000 / 1600 / 2400. `screen.sh` runs the first,
  `extend.sh` the others. The stable runs form a trunk. Nothing continues
  from a decay leaf (`-d`).
- **Rank by the `wv` of the decayed run**, with the long stories
  (`long.sh`) and the recall (`ctx.sh`) beside it. From binary `4e42bba3`
  on, the dv of a bucketed run is `wv`. The in-loop valid of the flat
  scores the first 1024 chars only: its `wv` comes from `long.sh -t`.
- **The seed noise** of a long run: 0.008 between two seeds (R-N15-B5).
  One run per shape ranks only a gap ≳ 0.01. Below that, add a seed.
- Every H-Net run: `--grad-checkpointing`, `--max-vram 3750` and
  `--vram-cleanup-every 1` (`args.sh` adds them), `--ssd-path recalc:32`
  (pins the SSD chunk, which doubles at a state rank ≥ 80).
- **The flat reference h16-V7u** lives in `../../../tiny-stories/tmp/1m-cased`
  (its own `gen.py`, `screen.sh`, `extend.sh`, `probe.sh`, `bin/ts-cuda`):
  runs `h16-V7u-s2` (3230), `-x4630`, `-sd2630`, `-x8000`. Its flags:
  `--pack 4224 --pack-rows 2 --ssd-path recalc:128 --profile 100
  --grad-checkpointing --vram-cleanup-every 1`. A run saves its pack
  settings, so `extend.sh` needs only the other flags. Run it from this
  queue: `cd ../../../tiny-stories/tmp/1m-cased && ./extend.sh …`. The
  old reference h16 has the runs `h16-s2`, `-x4630`, `-x8000`, `-x12000`,
  `-sd2630`.

## VRAM

How cubecl (48301b5, the `Adaptive` layout, the `Sync` allocator) spends
the device memory (burn-stack `vram.rs` has the detail):

- `bytes_reserved` (the `prof` lines, `--max-vram`, `probe.sh`,
  `dose.sh`) is the sum of the pages that the pools hold. A page is ~168
  MB, so a peak moves in steps of that size.
- **A dropped page waits** in the storage queue until an explicit cleanup
  (`Device::memory_cleanup`). The report does not count the queue.
  **`--vram-cleanup-every 1`** (`args.sh`) cleans up before every window,
  so the queue never outlives a window.
- `vram::check` reads the pools after every backward. The `prof` line
  prints their high-water mark as `peak X MB`.
- The CUDA context and the loaded kernels are outside the pools (~140
  MiB). So **the process ≈ `peak` + ~140 MiB**.
- **The dose target**: a `peak` of **~3400–3550 MB** (the process then
  stays < 3.6 GiB = 3865 MB). R-N15-B5 is at 3386, h16-V7u at 3455. One
  page over (3592, 3663) still keeps the process < 3.6 GiB.

## Shapes and budgets

- **The dose** (strict rule 1):
  1. The VRAM: `probe.sh <cfg>` (2 batches of each slot, ~1 min) is for
     the coarse search only: a run holds × 1.0 to × 1.34 of the probe, by
     shape (`notes.md` §7). `dose.sh <cfg> [steps]` runs the first 520
     steps of the real run (~10 min): dose every shape before it trains.
  2. The params: design the shape near the cap, then `fit.py` fills only
     a small rest with the main MLP (strict rule 14). `mmlp` costs no
     VRAM. The widths must not shrink inward (stage ≤ inner stage ≤ main).
- **The VRAM knobs** (`notes.md` §7): the stages set the peak (the 1024
  slot). The heads × the state rank of a stage drive it. **Use a power
  of 2 for a state rank** (user): a rank that is not a power of 2 jumps
  in the pages, and it can cost more VRAM than the next power of 2 (the
  flat, `../../../tiny-stories/tmp/1m-b/notes.md`: h16 × r 96 3995 MB,
  h16 × r 128 3776 MB). It can also help the score. A head dim
  below 16 doubles the heads (over the band at d64 and d112). N moves the
  peak little. A main at expand 1 and the main MLP are cheap.
- **Virtual layers** (user): repeat the middle real layers more than the
  edge layers (`msched=0,1,1,1,1,2`).
  - A real layer is fully untied. A virtual application is partly untied:
    it holds its own copy of the `untied` tensors only (the block tail, the
    small per-head tensors, the pre-norms).
  - Untie every virtual application, in every stack (user, 2026-10-11).
    The copies are cheap in params and usually give a better score.
    `gen.py` does this by default: `muntie=auto` in the main, `suntie=auto`
    in a stage with `eV`/`dV`.
  - The results: in the H-Net main, the score is a draw (R-N15-r64V7 =
    R-N15-B5 at 3230). In the flat, they are the fill of h16-V7u (Cyclic).

## Tools

- **Binaries**: `bin/hn-cuda` (`./build.sh`, a host job: it keeps each
  build as `bin/hn-cuda-<tag>` + `.txt`). The flat's `ts-cuda` is in
  `../../../tiny-stories/tmp/1m-cased/bin/` (its own `build.sh`, ~3 min).
  `bin/hn-cuda -- --help` lists the example flags.
- **Configs**: `gen.py <cfg> k=v …` → `cfg/<cfg>/`. The keys and their
  defaults are at the top of the file. Per-stage keys take `/` lists.
  `fit.py <cfg> <key> [from=V] [step=S] k=v …` fits one key to the param
  cap (`python3 fit.py`). `params.sh <cfg>` counts the params (~2 s, flex).
  `catalog.py` prints the models.
- **Runs**:
  - `screen.sh <run> <cfg> <T1> <T2> [args]`: a stable run of T1 steps,
    then a cosine decay leaf `<run>-d` of T2 steps.
  - `extend.sh <src> <new> <N1> <N2> [--lr LR [--span A-B]] [args]`: from
    a stable checkpoint, +N1 steps (with a step-down or an anneal), then a
    leaf of N2. With `N2 = 0` it is a trunk piece. `--lr` removes the
    later segments of the schedule: add them again with a later `--lr`.
  - `branch.sh`, `decay.py`, `stage.py`, `lrseq.py`: the LR sequence
    tools. `lrprof.py`: the protocol check of a run (`--guard`, `--batch`),
    and `--fill` (gives a trunk the rest of the protocol).
- **The queue**: `queue.sh` runs the lines of `queue.txt` one at a time
  (`bash -c`) and logs `start` / `done` / `fail` into `queue.log`. It
  exits when the queue is empty. Start it with `setsid nohup ./queue.sh >
  /dev/null 2>&1 < /dev/null &`. Waiters: `qwait.sh`, `qall.sh` (silent
  until the next `empty`), `after.sh`, `wait.sh`. Never edit a script
  while a queue line runs it (bash reads a script as it runs).
- **Reading**: `summ.py <run>…` (speed, train buckets, every validation),
  `agg.py`, `cool.py`.
- **Probes**:
  - `long.sh -g [-t] <run>` → `long.txt` (~50 s): the whole valid (`wv`:
    all, chars 0..1024, past 1024) and the long stories (0..1024,
    1024..2048, past 2048). `-t` takes a tiny-stories run directory.
    `longstat.py [-w] long/A long/B …`: the paired differences against A,
    and the profile per 256 chars (`-w`: the whole valid).
  - `ctx.sh -g -n 2048 -r ctxg <run>` → `ctxg.txt`: the recall (far use,
    re-mentions). `ctxstat.py ctxg/A ctxg/B …` pairs the runs.
  - `probe.sh <cfg> -- --ssd-path recalc:32 --chunk-log 0` →
    `probes-B.txt` (the bucket protocol).
  - `dose.sh <cfg> [steps] -- --ssd-path recalc:32 --chunk-log 0` →
    `doses.txt`.
  - `gate.sh <cfg> [lo] [hi]`: exit 0 when the last dose of `<cfg>` is in
    the band (default 3380–3670 MB). A queue line starts a screen only
    after a good dose: `if ./gate.sh A; then ./screen.sh …; fi`.
  - `infer.sh <run> <dir>`: samples on flex.
- **Kaggle** (a second runner, two T4s, the same pool accounting):
  - `kaggle/kqueue.txt` holds the lines of a batch: one job per line, its
    steps joined with `&&`. The two GPUs take the lines in parallel.
  - `kaggle/kwait.sh <hours> <log>` (in the background) pushes the batch
    (`kpush.sh`: the binary, the scripts, `cfg/`, the source runs). Then it
    waits for the kernel, merges its runs (`kpull.sh`), and queues
    `long.sh` for each pulled `-d` run in the local queue. It can miss a
    `-d` run (`notes.md` §8): check each one.
  - Only `probes.txt`, `knobs.txt`, `ctxg.txt` and `speeds.txt` come back
    as text. A dose on Kaggle: read the peaks in the pulled `runs/dose-*`.
  - One kernel at a time. `kpush.sh` empties `kqueue.txt`, so stage the
    next batch after a push ends. The flat cannot run on Kaggle (the
    payload holds this workspace only).
  - The CLI (`/usr/local/bin/kaggle`) can be gone after an environment
    reset: the user installs it. The token stays in the gitignored
    `.kaggle/access_token` (`kaggle/kg.sh` reads it). Never print it.
    `owner.txt`, `kqueue.txt` and `out/` stay out of git.
