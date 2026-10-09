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
- No `ctx.sh` (no `--context-use` here).
