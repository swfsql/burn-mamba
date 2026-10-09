# tmp: the results of the running session

The results and the decisions of the running session, in time order. At
the session end they go into `notes.md` (compressed, by topic), and this
file starts empty again.

## 2026-10-09 22:40 restart

- Here: the decay leaf of L-R-N15-B5 (the failed partial leaf is
  `runs/fail2-L-R-N15-B5-d`), then the probes. Their order: the three
  extremes of the M family first (M-s64r64-L4, M-s96r64-L5,
  M-s96r64-L4r256), then Nm4-s96r128-B5, the other M shapes, W-m64r-B5.
  The probe of R-N15-B5 is done (2903 MB), so it does not run again.
- Kaggle batch 19: L-R-B5 (the N 2 reference) and **L-R-N15-B5-s1** (the
  baseline at `seed=1`: the seed noise of the new protocol, user
  suggestion 2). L-Nm4-s96r128-B5 moves out of this batch: it has no
  probe yet, and it belongs with the M family (the next batch, on the new
  dv binary).
- L-R-N15-B5-d at step 100: 1.32 s/step, reserved 3384.8 MB (the phase 1
  3386).

## The validation scores whole stories (user suggestion 1)

- burn-stack `lm.rs` `dataloaders_for`: a config with `bucket` validates
  on the same slots and `chars`: the same 256 stories, each whole, in one
  window (`BucketedStoriesDataset` of the valid split, the order of
  `training.seed`). No new config field: the `bucket` field selects it.
  An unbucketed config validates as before (windows of `seq_len`).
- So dv of a bucketed run is `wv`. A run of an older binary still has the
  1024-char dv: read the binary tag in `build.txt`.
- `gen.py`: only the comment changes (`seq`/`runlen` do nothing under the
  buckets). The configs do not change.
- **Check** (binary `4e42bba3`, `runs/vchk-L-R-N15-B5-d`: `-b 0` on a copy
  of the leaf): 28 batches, 207,187 chars, bits 0.86880287. `long.sh`:
  wv 0.86880286. Equal. So from this binary on, dv = wv, and `long.sh` is
  needed only for the long stories and the per-position profile.

## The baseline (new protocol)

- **L-R-N15-B5-d**: **wv 0.8688** (chars 0..1024 0.8612, past 1024
  0.9529), the old dv (first 1024) 0.861. The flat h16 at 3230: 0.8868, so
  the H-Net leads by **0.018**.
- Against old-protocol R-N15 at 3230 (wv 0.8659, dv 0.8534): chars
  0..1024 +0.008, past 1024 about −0.07 (computed from the 7.2% share),
  all +0.003. So the 5 slots cost the story heads about as much as the
  grouping did before (B-c1024 +0.008).
- The long stories: 0..1024 0.9938, 1024..2048 1.0105, 2048.. 0.9779,
  all 0.9981. The recall (`ctxg.txt`): gain 0.1426 ± 0.0017, penalty
  0.0908.
- The leaf: 600 steps in 13 min here (1.32 s/step).
