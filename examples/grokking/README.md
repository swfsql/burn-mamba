# Grokking — modular addition

`(a + b) mod p` (`p = 97`) with a small single-layer Mamba-2 LM: the classic
grokking task, and the substrate for the state-participation-ratio diagnostic
(does the effective rank of the recurrent state collapse at the
memorize→generalize transition?).

- All `p²` pairs, deterministically split train/test **by pair**
  (`ChaCha8Rng(split_seed)`); full-batch AdamW with plain (non-cautious)
  decoupled weight decay; cross-entropy on the final position only.
- The model is constrained on purpose: `conv_kernel = 1` (all pair interaction
  flows through the recurrent state), 1 head, untied LM head, no oscillatory
  channel (Mamba-2, not -3). See `model.rs` for the rationale.
- `metrics.csv` in the artifacts directory holds
  `step,lr,train_loss,train_acc,test_acc` (power-of-two steps early, then every
  `eval_every`).

```bash
# memorization control arm (wd = 0, default)
cargo run --release --example grokking -- --training -a artifacts/grok-wd0

# a grokking arm
cargo run --release --example grokking -- --training -a artifacts/grok-wd01 \
    -- --wd 0.1 --steps 100000

# evaluate + sample predictions
cargo run --release --example grokking -- --inference -a artifacts/grok-wd01
```
