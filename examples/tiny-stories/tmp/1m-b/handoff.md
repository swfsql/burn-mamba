# Handoff: tiny-stories ≤1M-param search

Read this file, then `ORIENTATION.md` (goal, rules, protocol, tools, and
the map of the files). `notes.md` has the results and the findings (§3 = the
`v0.64.0` pool build, §4 = the far-use diagnostics and the 2048-story
context use, §5 = the contrast, closed, §6 = h16 and h16 × g2 past 14400).
`far-use.md` is the plan for auxiliary objectives that target the far use.
The paths are relative to `examples/tiny-stories/tmp/1m-b/`.

## Goal (user)

The lowest valid bits/char **and** long-context coherence: the **far use**
of `--context-use`. Measure the context use of every screen and every
milestone (`ctx.sh -g -n 2048`), and rank by both. Judge the far use by the
own far bits paired with a control. Use the most of both ceilings (1M
params, VRAM). Virtual layers (V > L) always take the untied tensors of
mnist-class (`gen.py untie=1`).

## Guidelines (user)

- **Plan before long runs.** First run the checks of a few minutes that
  show whether a run of 40 minutes or more can be informative: evals on
  the existing checkpoints and snapshots, short probes, tiny smoke runs.
  Do not spend a long run that can be a null.
- **Models before seeds** (`ORIENTATION.md`, Working rules).
- The plain cross-entropy does better in the long run than the auxiliary
  losses tried so far. h16 and h16 × g2 are not saturated: find their
  limit.

## State

Best loss: **0.7163**, h16 @19200 (`runs/p16k-h16-lr1-d`). Best far use:
**0.295**, h16 × g2 @19200 (`runs/p16k-h16g2-lr1-d`), but h16 has the better
own far bits (0.7264 against 0.7349, `notes.md` §6).

The pool is a 2 × 2 on h16: {base, g2} × {L5, L5 × V6 untied}. Each cell:
decayed valid / far use (pairing i+1). Up to @14400: 244 stories, from
@19200: 2048 stories. Process memory on the current build.

| id | cfg | params | ms/step | process | @3230 | @5630 | @9600 | @14400 | @19200 | latest stable run |
|---|---|---|---|---|---|---|---|---|---|---|
| **h16** | `s1k-u1-wd1-h16` | 969,160 | ~680 | 3134 MiB | 0.8579 / 0.224 | 0.8014 / 0.253 | 0.7493 / 0.269 | 0.7281 / 0.284 | **0.7163** / 0.291 | `p16k-h16-lr1` (16000, 1e-3) |
| h16 × g2 | `s1k-u1-wd1-h16g2` | 982,747 | ~715 | 3284 MiB | 0.8649 / 0.241 | 0.8083 / 0.261 | 0.7552 / 0.285 | 0.7327 / 0.289 | 0.7222 / **0.295** | `p16k-h16g2-lr1` (16000, 1e-3) |
| h16 × V6 | `n-h16-V6` | 982,380 | 871 | ~3580 MiB | **0.8533** / 0.225 | – | – | – | – | `n03-h16-V6` (2630) |
| h16 × g2 × V6 | `n-h16g2-V6` | 995,318 | 880 | ~3844 MiB | 0.8589 / 0.222 | – | – | – | – | `n06-h16g2V6` (2630) |

- The pool runs used 2 workers: their dv has a data-order noise of ~0.01
  (`ORIENTATION.md`, Run noise). Parked: r128 (0.7590 / 0.267 @9600).
- The contrast is closed (`notes.md` §5, `far-use.md`).

## Queue

- **Running** (`queue.sh`, from 2026-10-06 17:52 UTC), each from the
  stable 16000 (`p16k-*-lr1`): +4000, then a decay of 4000 → **@24000**,
  each followed by its 2048-story context use (→ `ctxg.txt`):
  1. h16 at 5e-4 (`p20k-h16-lr05`), stable 20000: 0.7392,
  2. the 1e-3 control (`p20k-h16-lr1`),
  3. h16 × g2 at 5e-4 (`p20k-h16g2-lr05`).
  ~1.6 h per run. `qwait.sh` (in the background) prints when a line ends.
  Check the queue with `pgrep -f queue.sh` (not `-x`).
- **Pending, not to run before the user says so** (`queue.pending.txt`,
  lines 1–6): the full-epoch leaves from the stable 20000 at 5e-4, all →
  @28021 (one epoch): h16 (a) +2421, then a cooldown of 5600, h16 (b) a
  cooldown of 8021 at once (the cooldown-length pair), h16 × g2 (a). Then
  the parked pool work: n03 and n06 → 4630, the probe + screen of n07 (h16
  + `trop=MaxPlus`), each with its context use. To run lines: move them to
  `queue.txt`.

## Binary

- `bin/ts-cuda` = `v0.64.0-eef06da-dirty-0779d0ea`: `eef06da` +
  `--contrast-profile` and `--contrast-one-sided`. The queue uses it. Do
  not replace it while the queue runs: each decay phase starts a fresh
  `bin/ts-cuda`.
- burn-mamba pins burn-stack `d865deeb` (the `Lr::Sequence` and
  `Lr::Linear` schedules). After the commit of this session, `./build.sh`
  gives a clean tag. Build the production binary only from committed
  burn-stack and burn-mamba (user).
- Keep `--max-vram` on every training run (the scripts add 3750).

## Next steps

1. When the queue ends: the dv and the far use of the three runs
   @24000. h16 at 5e-4 against the 1e-3 control (the matched pair) decides
   the step-down. Against @19200: `ctxstat.py ctxg/p16k-h16-lr1-d
   ctxg/p20k-h16-lr05-d ctxg/p20k-h16-lr1-d`. The 1e-3 trend guess for
   h16 @24000: ~0.709.
2. With the GPU idle: move the scripts to the LR sequence (`ORIENTATION.md`,
   The LR in the config): `decay.py`, a new `stage.py`, `gen.py`, and the
   history of the `p20k-*` trunk configs as sequences. Then a clean
   `./build.sh`.
3. The full-epoch leaves and the cooldown pair (Queue, pending), when the
   user says so.
4. The parked pool work (Queue). More ideas: a clean quaternion test at B's
   shape (h4 + Q4D rf 0.5), h32 r32, a longer `dt` init range, the
   distance curve (`far-use.md` step 1.3).
5. Far use, if it comes back: gapped rows (`far-use.md` step 4) is the
   best founded idea, with a cheap prototype (`$` as the blank).
