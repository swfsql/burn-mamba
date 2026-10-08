# Handoff: tiny-stories ≤1M-param search, the cased world

Read this file, then `ORIENTATION.md` (the cased world and its metric) and
`../1m-b/ORIENTATION.md` (the protocol and the tools, the same here).
`notes.md` has the results.

## Goal (user)

The lowest valid bits/char of the cased text **and** long-context coherence
(the far use of `--context-use`). The hard problem: the binding of names
("jack and jack": three names of a prompt collapse into one, `../1m-b`
`infer-h16-lr05-24000-prompts/inference/sample-prompted-1-1.txt`). The case
channel gives the model a mark at each name.

## State

- The code of the cased world is ready and tested (`notes.md` §1).
  `bin/ts-cuda` = `v0.65.0-76b714f-dirty-2fdd53d8` (its `build.txt` holds
  the burn-stack diff). After the commits, `./build.sh` gives a clean tag.
- Nothing runs. The queue is empty. No baseline yet: `runs/h16-s` is a
  partial run (235 steps).
- The pool is empty. The first member is the baseline h16 (`cfg/h16`, the
  model of `../1m-b` h16, with the LR plan `stages=4630:1e-3,8000-12000:5e-4`).

## Next steps

1. A clean `./build.sh` after the commits.
2. The baseline screen: `screen.sh h16-s2 h16 2630 600 -- --pack 4224
   --pack-rows 2 --ssd-path recalc:128 --profile 100`, then `ctx.sh -g -n
   2048 -r ctx h16-s2-d >> ctx.txt` and `infer.sh h16-s2-d <dir>`. Its
   stable run is the trunk of the long run.
3. Does the case mark reduce "jack and jack"? The samples of the two
   prompts, and the re-mention rows.
4. A binding probe (proposed, not built): templated prompts at points where
   the name is fixed (coordination "X and ___", the speaker, attributes,
   common against rare names). It scores the failure of the samples with a
   standard error.
5. The pool: `--grad-checkpointing` fits h16 × V6 and h16 × g2 × V6 under
   the preferred VRAM limit (an estimate, probe first).
