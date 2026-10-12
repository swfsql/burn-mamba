# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-12 session 2 (end)

- 03:20 **The residuals** (user question): every stack of every H-Net
  run of the new protocol is `Standard` (`gen.py` `mgr=0`, `mmgr=0`). The
  flat h16-V7u is **Multi-Gate 4** (the flat's `gen.py` default `mgr=4`,
  one module for all applications). In the flat (`1m-b`), the standard
  residual lost 0.046 against Multi-Gate 4 (Multi-Gate 3 / 6: +0.011 /
  +0.017), for ~1.6k params and +24 MB. The H-Net's one test
  (`s128-mg4`, old protocol: 600 steps, lr 8e-3, the first 1024 chars)
  tied. So Multi-Gate is untested on the H-Net in the new protocol.
- 03:22 **Decision** (doses only, user: no new runs): R-N125-r128s11V2-m5
  × Multi-Gate 4 in every stack (`-mg4`, 993,485) and in the main only
  (`-mmg4`, 992,965). Their doses run here after the screen (~04:10).
- 03:15 **User**: (1) the param trim of a main application is `start=2`
  (fewer start latents). (2) Compare the main schedules on the H-Net
  itself: `R-N15-r64V7c` = R-N15-r64V7 with a Cyclic main (0,1,2,0,1,2,0
  against 0,1,1,1,1,1,2), all else equal (995,713). Reason: another arch
  or stack can prefer another schedule than the flat. Its dose runs
  after the Multi-Gate doses (~04:45). Rank it against R-N15-r64V7 on
  the same milestones (3230: 0.8644 / 0.8675; 5630, 9600 on the
  protocol), two seeds if the gap is < 0.01.
