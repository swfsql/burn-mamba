# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-12 19:45: step 1, the one-change screens

- The four candidates pass `gate.sh` (doses: `-mg4`, `-mmg4`, `-st2`
  3482 MB; `R-N15-r64V7c` 3592 MB). New configs `-s1` (seed 1) of each:
  `gen.py` from `args.txt` + `seed=1`, only the seed differs.
- Decision: two seeds of `-mg4` and `-st2` at once. Reason: the base
  R-N125-r128s11V2-m5 has a seed spread of 0.022 at 3230, so one seed of
  a candidate ranks only a gap ≳ 0.02. One seed first for `-mmg4` and
  `R-N15-r64V7c` (the base R-N15-r64V7 spreads 0.003).
- Local queue: `-mg4` (the strongest prior), then `R-N15-r64V7c`.
- Kaggle batch (`kqueue.txt`): `-mg4-s1` then `-mmg4`; `-st2` then
  `-st2-s1`. The CLI is gone after an environment reset: the push waits
  for the user to install it. No `gate.sh` in a Kaggle line (`doses.txt`
  does not go up): the gate passed here.
- 20:32: the user installed the CLI (pip, Kaggle API 1.7.4.5). Batch
  `20261012-203200` pushed (kernel version 34, budget 4.5 h). Waiter:
  `kwait.sh`, its log `/tmp/kwait-20261012-b1.log`.

### `-mg4` (Multi-Gate 4 in every stack), seed 0, local (21:03)

- wv at 3230: **0.8625**. The base: 0.8533 / 0.8756 (mean 0.8645). The
  flat: mean 0.8602. Schedule: `lrprof.py` clean. Peak 3482 MB, 1340
  ms/step (the base 1283 at the dose).
- Paired against base s0: wv +0.0085 ± 0.0015, the long stories +0.0155,
  far own +0.051, re-mention own +0.072 (base s1: +0.0215 / +0.114). So
  it is between the two base seeds on every row.
- No verdict yet: `-mg4-s1` is on Kaggle. A tie with the base mean
  (−0.002) is far from the gain of 0.046 in the flat.

### Kaggle batch 1 (pulled 23:18): step 1 + step 2 at 3230

| run | wv s0 / s1 (mean) | Δ base mean |
|---|---|---|
| base R-N125-r128s11V2-m5 | 0.8533 / 0.8756 (0.8645) | 0 |
| `-mg4` | 0.8625 / 0.8552 (0.8589) | −0.0056 |
| `-mmg4` | 0.8586 | −0.0059 |
| `-st2` | 0.8566 / 0.8657 (0.8612) | −0.0033 |
| the flat h16-V7u | 0.8644 / 0.8559 (0.8602) | −0.0043 |

- `lrprof.py` clean on all four. `kwait.sh` missed the `long.sh` of
  `-mg4-s1-d` again: added by hand.
- Every one ties at 3230. Base s1 (0.8756) is the outlier: every other
  seed of this family is 0.8533–0.8657. The recall rows (`ctxstat.py`)
  are inside the two base seeds too.
- **Multi-Gate**: −0.0056 (two-seed means), not significant, and no sign
  of the 0.046 of the flat at 3230. It goes long (the user's prior).
- The long stories (`longstat.py -w`, paired against base s0) follow wv
  on every run: `-mg4` +0.0085 / +0.0022, `-mmg4` +0.0047, `-st2`
  +0.0036 / +0.0120, base s1 +0.0215.
- **Step 2, the start rows**: `-st2` ties (−0.0033). So 2 rows cost
  nothing: no `-st8`, no per-stack start rows.
- **The verdict "stage + main reuse lost 0.019" is weaker than it
  read.** That one-seed screen (`R-N15-r128s11V2-m5V6c-st2`, 0.8733) was
  compared with base s0 only. Against the base mean it is +0.009;
  against base s1, −0.002. It is still the worst seed but one of the
  family. So it is not a clear loss, and main reuse paid only after 3230
  on R-N15-r64V7 (§4). So step 3 tests the composition again.

### `-mg4` s0 at 5630 (local, 23:45)

- wv **0.8040**. The base: 0.7997 / 0.8126 (mean 0.8062); the flat
  0.8037. `lrprof.py` clean. So −0.0022 against the base mean (3230:
  −0.0020), +0.0043 against base s0 (3230: +0.0092). No late payoff yet;
  the 9600 chain runs.
- **9600 (01:57): wv 0.7455.** The base: 0.7448 / 0.7533 (mean 0.7491);
  the flat 0.7487. `lrprof.py` clean. Paired against base s0: wv
  +0.0010 ± 0.0015, the long stories +0.0012 past 1024, far own +0.002.
- The trend of `-mg4` s0 against the flat: +0.0023 / +0.0003 / −0.0032
  (3230 / 5630 / 9600); against the base mean: −0.0020 / −0.0022 /
  −0.0036. A small drift toward a gain, inside the noise. `-mg4-s1` at
  9600 (batch 2) tells more.

### Kaggle batch 2 (pulled 02:47)

- Line B stopped at its first gate: `R-N125-r128s11V3-m5-sp32` dosed
  **3096 MB** (T4), under the band. So 8 stage heads × 3+3 applications
  still sit ~3 pages under the 16-head 2+2 leader (3482). The `&&` chain
  then skipped the combo, and the GPU idled from 23:40. Lesson: one
  gated candidate per Kaggle line, or a fallback after it.
- Line A, `-mg4-s1`: **0.7978 at 5630, 0.7426 at 9600**. `lrprof.py`
  clean. `kwait.sh` missed the `long.sh` of `-x4630-d`: added by hand.

| wv (two-seed mean) | 3230 | 5630 | 9600 |
|---|---|---|---|
| `-mg4` | 0.8589 | 0.8009 | **0.7441** |
| base R-N125-r128s11V2-m5 | 0.8645 | 0.8062 | 0.7491 |
| the flat h16-V7u (one seed past 3230) | 0.8602 | 0.8037 | 0.7487 |
| `-mg4` − flat | −0.0013 | −0.0028 | **−0.0047** |

- **The first H-Net that beats the flat at 9600**: both seeds (−0.0032 /
  −0.0061), and the lead grows with the steps (the user's prior: Multi-
  Gate pays late). The seed spread shrinks: 0.003 at 9600 (base 0.0085).
  The far recall at 9600 (`ctxstat.py`): own 0.7530 / 0.7467, between
  the base seeds (0.7510 / 0.7618).

### Step 4 + the next shapes (03:00)

- Kaggle batch 3: both `-mg4` seeds → `-t10000` → `-x12000` (14400; the
  flat 0.7220). No `--lr` (`lrprof.py --batch` ok).
- Every new candidate builds on `-mg4` (`mgr=4 mmgr=4`: 0 MB, +1.8k).
  Local doses after the screen of `-st2-s1`: the 8-head reuse everywhere
  with Multi-Gate, `R-N125-r128s11V3-m5V6c-st2-sp32-mg4` (3+3, 992,365)
  and `…V4…` (4+4, 999,421). Reason: 3+3 at 8 heads dosed 3096, the
  main V6 costs ~290 MB at N 1.25: so 3+3 is near the low edge, and
  4+4 can fill the band. The one in the band (the higher if both) gets
  a screen.

### Local, after the `-mg4` 9600 chain (decided 00:45)

- `R-N15-r128s11V2-m5V6c-st2-s1`: seed 1 of the old stage + main reuse
  screen (16 heads, N 1.5; the gate of its seed 0 dose, 3482 MB). Reason:
  its "lost 0.019" verdict came from one seed against base s0 only, and
  the local GPU would idle until the batch 2 pull (~02:30). A second seed
  settles the verdict and gives the 8-head combo of batch 2 a reference.

### Kaggle batch 2 (pushed ~23:25)

- Line A: `-mg4-s1` → 5630 → 9600 (the user: Multi-Gate goes long).
- Line B (dose + `gate.sh` + screen on Kaggle, in sequence):
  - `R-N125-r128s11V3-m5-sp32`: the leader × 8 stage heads × stage 3+3
    × `mmlp=8` (998,241). The fill: the main MLP (rule 14: VRAM-free,
    neutral; the rest was 8k from the floor, a small rest).
  - `R-N125-r128s11V3-m5V6c-st2-sp32`: the same stage × main V6 Cyclic ×
    `start=2` (990,545, no fill): "reuse everywhere", step 3 of the
    plan, paid by the VRAM of the 8 heads.
- The local sp32V3 dose: dropped (the Kaggle line doses it).
- Local queue: the four `long.sh`, then the `-mg4` s0 9600 chain.

### `R-N15-r64V7c` (Cyclic main), seed 0, local (22:16)

- wv at 3230: **0.8636**. The base R-N15-r64V7 (0,1,1,1,1,1,2): 0.8644 /
  0.8675 (mean 0.8660). `lrprof.py` clean. Peak 3592 MB, ~1270 ms/step.
- Paired against base s0: wv −0.0010 ± 0.0019, the long stories +0.0084
  ± 0.0068 past 1024, the far recall rows within ±0.0025.
- So at 3230 the schedule of the main does not matter: a tie. (In the
  flat, Cyclic beat `m2` by 0.003 and `m13` by 0.010.) Decision: no seed
  1 now. A tie
  at 3230 gives no reason to spend a GPU slot on it. Cyclic is the
  default for a new main schedule (the flat's best; it ties here).

### Step 3 prep: 8 stage heads to give the page back

- `R-N125-r128s11V2-m5V6c-st2-sp32`: the N 1.25 stage + main reuse shape
  (dose 3771, over) with stage `p=32` (8 heads, the same state size).
  999,969 → 983,489 params (the per-head tensors, held per untied
  application). Queued: a dose after `R-N15-r64V7c`. Reason: the dose
  says whether the page comes back before any param design. The 16k
  gap then needs a design choice (not an automatic `mmlp` fill).
- **Dose (22:27): 3144 MB** (16 heads: 3771), 1182 ms/step (1523). So 8
  stage heads free ~627 MB (~4 pages) at the same state size. The likely
  cause: B/C after the per-head bias are `[tokens, heads, r]`, so their
  cost is heads × r, not heads × p × r. A new VRAM knob: the stage heads.
- Under the band now: the freed VRAM needs a deliberate spend. Param
  counts (`params.sh`): the leader × p 32 975,185; × r 256 1,046,353
  (over); × stage 3+3 applications 982,241.
- Decision: dose `R-N125-r128s11V3-m5-sp32` (the leader × 8 stage heads
  × 3+3 stage applications). Reason: stage reuse pays at r 128 (2+2: the
  leader), and a third application at r 128 was over the band with 16
  heads (probe 3678). The confound (8 heads alone) is accepted for now:
  8 heads alone would sit under the band (strict rule 1). The ~8k param
  rest: a design choice after the dose. Inserted before the `-mg4` 9600
  chain (+10 min to it).

### `-mg4` goes long (user, 21:30)

- The user: in other problems, Multi-Gate is slow to pay, but it can pay
  later. So `-mg4` extends whatever 3230 says.
- Local queue (after the dose): `-mg4` s0 → `-x4630` (5630), then
  `-t6315` → `-x8000` (9600), the chain of the base. `-mg4-s1`: the same
  chain on Kaggle after it comes back. Rank each milestone against the
  base seed pair and the flat.
