# Handoff: tiny-stories ≤1M-param search (the case-folded world)

**The search continues in `../1m-cased/`** (user, 2026-10-08: the corpus
keeps its case, a new world). No number here compares with a number
there. This directory is the record of the case-folded world.

Read `ORIENTATION.md` for the goal, the rules, the protocol and the tools
(`../1m-cased/` uses them too). `notes.md` has the results and the findings
(§3 = the `v0.64.0` pool build, §4 = the far-use diagnostics, §5 = the
contrast, closed, §6 = h16 and h16 × g2 past 14400, §7 = a larger state at
h16, which fails). `far-use.md` is the plan for the auxiliary objectives.

## State at the end

Best loss: **0.7010**, h16 @24000 (`runs/p20k-h16-lr05-d`). Best far use:
**0.2987**, h16 × g2 @24000 (`runs/p20k-h16g2-lr05-d`), but h16 has the
better own far bits (0.7112 against 0.7200) and recalls the entities
better (`notes.md` §6).

| id | cfg | params | ms/step | @3230 | @9600 | @14400 | @19200 | @24000 |
|---|---|---|---|---|---|---|---|---|
| **h16** | `s1k-u1-wd1-h16` | 969,160 | ~670 | 0.8579 / 0.224 | 0.7493 / 0.269 | 0.7281 / 0.284 | 0.7163 / 0.291 | **0.7010** / 0.295 |
| h16 × g2 | `s1k-u1-wd1-h16g2` | 982,747 | ~715 | 0.8649 / 0.241 | 0.7552 / 0.285 | 0.7327 / 0.289 | 0.7222 / 0.295 | 0.7065 / **0.299** |
| h16 × V6 | `n-h16-V6` | 982,380 | 871 | **0.8533** / 0.225 | – | – | – | – |
| h16 × g2 × V6 | `n-h16g2-V6` | 995,318 | 880 | 0.8589 / 0.222 | – | – | – | – |

Each cell: decayed valid / far use (pairing i+1).

- h16 × r128 × rf 0.5 (d152) is paused at @9600 (dv 0.7544 against h16
  0.7493, no recall gain). Its trunk `runs/p8k-h16r128rf05` is saved at
  8000, with the anneal 8000–12000 in its config (`notes.md` §7).
- Not run (`queue.pending.txt`): the full-epoch leaves and the cooldown
  pair of h16, the parked V6 pool, n07 (h16 + `trop=MaxPlus`).
- `--grad-checkpointing` now fits h16 × V6 and h16 × g2 × V6 under the
  preferred VRAM limit (an estimate: h16 saves ~700 MB).

## Binary

`bin/ts-cuda` = `v0.65.0-76b714f-dirty-762d39fd` (`76b714f` + the
checkpointing flag). The builds of the case-folded world read the
case-folded caches (`<split>-<n>.txt`). A build from the cased code reads
cased stories, so it cannot evaluate the checkpoints here.
