# Handoff: tiny-stories ≤1M-param search

Read this file, then `ORIENTATION.md` (goal, rules, protocol, tools, and
the map of the files). `notes.md` has the results and the findings (§3 = the
`v0.64.0` pool build, §4 = the far-use diagnostics and the 2048-story
context use, §5 = the contrast, closed, §6 = h16 and h16 × g2 past 14400,
the LR step-down, the re-mention probe). `far-use.md` is the plan for
auxiliary objectives that target the far use. The paths are relative to
`examples/tiny-stories/tmp/1m-b/`.

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
  losses tried so far.
- **Next priority: larger state sizes**, probably from scratch, for a long
  run, at h16 g1 (user, 2026-10-06). A long run is one config extended in
  many steps, with a measurement at each milestone, not one uninterrupted
  run.
- Prefer cosine anneals between LR rates to jumps (user).

## State

Best loss: **0.7010**, h16 @24000 (`runs/p20k-h16-lr05-d`). Best far use:
**0.2987**, h16 × g2 @24000 (`runs/p20k-h16g2-lr05-d`), but h16 has the
better own far bits (0.7112 against 0.7200) and recalls the entities
better (`notes.md` §6).

The pool. Each cell: decayed valid / far use (pairing i+1). Up to @14400:
244 stories, from @19200: 2048 stories.

| id | cfg | params | ms/step | @3230 | @9600 | @14400 | @19200 | @24000 | latest stable run |
|---|---|---|---|---|---|---|---|---|---|
| **h16** | `s1k-u1-wd1-h16` | 969,160 | ~670 | 0.8579 / 0.224 | 0.7493 / 0.269 | 0.7281 / 0.284 | 0.7163 / 0.291 | **0.7010** / 0.295 | `p20k-h16-lr05` (20000, 5e-4) |
| h16 × g2 | `s1k-u1-wd1-h16g2` | 982,747 | ~715 | 0.8649 / 0.241 | 0.7552 / 0.285 | 0.7327 / 0.289 | 0.7222 / 0.295 | 0.7065 / **0.299** | `p20k-h16g2-lr05` (20000, 5e-4) |
| h16 × V6 | `n-h16-V6` | 982,380 | 871 | **0.8533** / 0.225 | – | – | – | – | `n03-h16-V6` (2630) |
| h16 × g2 × V6 | `n-h16g2-V6` | 995,318 | 880 | 0.8589 / 0.222 | – | – | – | – | `n06-h16g2V6` (2630) |

- LR: 4e-3 to 4630, 1e-3 to 16000, then 5e-4 (it beat a 1e-3 control
  @24000 by 0.006 dv). **Not tested: how low the trunk LR should go** (2.5e-4
  from 20000, user: not a priority).
- The re-mention probe: the first letter of a re-mentioned entity costs 1.8
  bits after the own prompt (h16 @24000). r128 recalls better at B's
  shape. A quaternion and g2 do not. So the binding is a capacity problem.
- The pool runs used 2 workers: their dv has a data-order noise of ~0.01
  (`ORIENTATION.md`, Run noise). Parked: r128 (0.7590 / 0.267 @9600).
- The contrast is closed (`notes.md` §5, `far-use.md`).

## Queue

- Empty. Nothing runs.
- **Pending, not to run before the user says so** (`queue.pending.txt`,
  lines 1–6): the full-epoch leaves from the stable 20000 at 5e-4, all →
  @28021 (one epoch): h16 (a) +2421, then a cooldown of 5600, h16 (b) a
  cooldown of 8021 at once (the cooldown-length pair), h16 × g2 (a). Then
  the parked pool work: n03 and n06 → 4630, the probe + screen of n07 (h16
  + `trop=MaxPlus`), each with its context use. To run lines: move them to
  `queue.txt`, and launch `queue.sh` with `setsid nohup` (ORIENTATION).

## Binary

- `bin/ts-cuda` = `v0.65.0-b571dfd-dirty-3e6ee768`: `b571dfd` + the
  re-mention probe + the scripts of the LR sequence, burn-stack `d865deeb`
  (`Lr::Sequence`, `Lr::Linear`). After the commit of this session,
  `./build.sh` gives a clean tag of the same code.
- Smoke tests of this build (GPU): `probe.sh s1k-u1-wd1-h16` (a converted
  cfg): the logged rates of steps 1–10 are 4e-3·k/128 exactly, reserved
  3149 MB, 671 ms/step (as before). `extend.sh p20k-h16-lr05 smoke-ext 2 2
  --lr 2.5e-4`: 2.5e-4 at 20001–20002 (the stage from 20000), then the
  decay segment from 20002, as `lrseq.rate` gives it. `ctx.sh -g -n 256 -r
  ctxsmoke`: the six rows (with `letters` and `first`). The test artifacts
  are deleted, but for `probe-s1k-u1-wd1-h16-1791327549` (a test only).
- The trunks and the pool cfgs hold LR sequences (`seq.py`). The older
  runs keep the old form: `seq.py` rewrites a run before it extends again.
- Keep `--max-vram` on every training run (the scripts add 3750).

## Next steps

1. A clean `./build.sh` (the committed code is the installed dirty build).
2. **The seed noise of the re-mention rows** (~80 s, before any run on the
   state size): `ctx.sh -g -n 2048 -r ctxrg n04-h16-s1-d`, then `ctxstat.py
   ctxrg/k16-Bwd-h16-d ctxrg/n04-h16-s1-d` (h16 @3230, seeds 0 and 1). The
   r128 result (first letters −0.037 ± 0.010 against B) carries only the
   error of the stories, and float noise alone moved the far use by 0.008
   (`notes.md` §5). If the seed spread of the first-letter own bits is well
   below 0.037, the direction of the state size is solid. If it is close,
   r128 needs a second seed first.
3. **Larger state sizes at h16 g1** (user: the priority). The re-mention
   probe says that the state size limits the binding of entities.
   - `params.sh` and `probe.sh` (10 steps) for h16 × r96 / r128 at
     `d_model` ~150 (B/C cost ~+100k params at r128).
   - A screen (2630 + 600) against h16 @3230 (first letters 2.284): a
     go/no-go in under an hour. The r128 effect showed at @3230 already.
   - Then the long run of one config, from scratch, but as the trunk and
     its leaves of the protocol: many extensions, each with a measurement
     (a decayed leaf and `ctx.sh`) at its milestone. Never one
     uninterrupted run of many hours (user).
4. **Rank also by the first-letter re-mention cost** (the `first` row of
   `ctx.sh`), beside the dv and the own far bits. For r128 it moved 4–5×
   more than the far loss, and it tracks the binding failure of the
   samples.
5. **The LR of the new config: step down earlier and smoothly** (user:
   agreed). At least `stages=4630:1e-3,12000:5e-4` (5e-4 tied 1e-3 from
   8000, the 1e-3 readout flattened by 16000, the excess was 0.052 at
   12000). Better: cosine anneals between the rates, not jumps (user), e.g.
   1e-3 → 5e-4 over 8000–12000. A segment `{from: 8000, CosineAnnealing {
   max_lr 1e-3, min_lr 5e-4, warmup_steps 0, total_steps 4000 }}` holds 5e-4
   after its end, so the config can hold it now, and `decay.py` reads the
   rate of a leaf in the middle of an anneal. But `gen.py stages=` and
   `stage.py` write only jumps: give them an anneal form first (e.g.
   `A-B:LR`, a cosine from the rate at A to LR at B).
6. The full-epoch leaves and the cooldown pair (Queue, pending), when the
   user says so.
7. The parked pool work (Queue). More ideas: a longer `dt` init range, the
   distance curve (`far-use.md` step 1.3).
8. Far use, if it comes back: gapped rows (`far-use.md` step 4), with a
   cheap prototype (`$` as the blank).
