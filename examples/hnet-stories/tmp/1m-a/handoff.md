# Handoff: hnet-stories ≤1M-param search (1m-a)

Read this file, then `ORIENTATION.md` (the goal, the protocol, the tools).
`tmp.md` holds the results of the running session (they go to `notes.md`
at its end).

## State (2026-10-09 05:15)

- **The base "b"**: stage d128 2+2 (r32, p16, Complex2D), main d136 × 3
  (r64), N 2, untied head, Burn init, lr 8e-3 (5e-3 ties), λ auto, α 0.3,
  bs 10 × 1024, gc. 958,776 params. `cfg/s128-N2` (= `k-base`).
- **600 + 200 screens** (dv bits/char): b 1.1286, b × N 1.5 1.1175, b × N 3
  1.1404, the flat h16 1.1314. Lost: Real1D stages, 2 stages, Multi-Gate,
  Cyclic main virtual layers, a deep stage, the reference init, the tied
  head.
- **The recall** (`ctxg.txt`, the re-mention first letter, 800 steps):
  N 2 recalls best (gain 0.697), N 1.5 worst (0.467), the flat 0.631. The
  user's hypothesis: compression pays later. The long screens test it.
- **The budget table** (`knobs.py`): what each knob costs in params and
  VRAM (tmp.md).

## Running

- Local (`queue.txt`, `queue.sh`): L-N2 (b, 2630 + 600), then its ctx
  probe, then the VRAM fills f-sr48, f-sr64, f-mr96, f-mm2, f-mv6m
  (600 + 200).
- Kaggle (batch 20261009-051418, kernel v4, T4 × 2): L-N15 and L-N3
  (2630 + 600) + their ctx probes, one per GPU. `kaggle/kpull.sh` merges
  them when the kernel ends.
- The heartbeat (a session cron, every 30 min) checks both runners.

## Next steps

1. Compare L-N2 / L-N15 / L-N3 (dv and the recall rows of `ctxg.txt`).
   If the gap of N 2 / N 3 to N 1.5 shrinks, extend them in < 1 h steps
   (`extend.sh`, the LR plan of 1m-b) and measure the recall at each
   milestone.
2. Read the VRAM fills against b; spend the idle VRAM (b uses ~2.1 GB of
   3.75 GB with gc) on the param-cheap knobs of the budget table.
3. Give Real1D stages a budget-matched test (it frees ~600 MB).
