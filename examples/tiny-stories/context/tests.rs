//! [`score`] reads the right rows. Its bits equal those of a reference that
//! scores each character of each continuation with its own `forward` (over the
//! prompt and the continuation up to that character, the last row), across
//! more than one batch.

use super::{BATCH, score};
use crate::dataset::VOCAB_SIZE;
use crate::test_net::tiny_net;
use crate::training::Run;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn_mamba::prelude::*;
use burn_stack::utils::test_helpers::test_device;

#[test]
fn score_reads_the_rows_of_the_continuation() {
    let device = test_device();
    device.seed(0);
    let net = tiny_net(&device);
    let run = Run {
        ssd_path: MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None)),
        graphs: false,
        max_vram_mib: None,
    };
    // Two batches, the second one partial.
    let (n, p, c, join) = (BATCH + 1, 5, 6, 2);
    let ids = |i: usize, len: usize, salt: usize| -> Vec<u8> {
        (0..len).map(|k| ((k * 7 + i * 11 + salt) % VOCAB_SIZE) as u8).collect()
    };
    let prompts: Vec<Vec<u8>> = (0..n).map(|i| ids(i, p, 1)).collect();
    let continuations: Vec<Vec<u8>> = (0..n).map(|i| ids(i, c, 5)).collect();
    let pair = |i: usize| (i + 1) % n;
    let got = score(&net, &run, &device, &prompts, &continuations, join, BATCH, pair);

    // The reference: one `forward` per scored character.
    let (mut near, mut far) = (0.0, 0.0);
    for i in 0..n {
        for k in 0..c {
            let tokens: Vec<i32> = prompts[pair(i)]
                .iter()
                .chain(&continuations[i][..k])
                .map(|&t| t as i32)
                .collect();
            let x = Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).reshape([1, tokens.len()]);
            let (logits, _) = net.forward(x, None, run.ssd_path.clone(), None, None);
            let rows = logits.dims()[1];
            let log_p = log_softmax(logits.narrow(1, rows - 1, 1), 2);
            let target = continuations[i][k] as usize;
            let nll = -(log_p.narrow(2, target, 1).into_scalar::<f32>() as f64);
            if k < join {
                near += nll;
            } else {
                far += nll;
            }
        }
    }
    let ln2 = std::f64::consts::LN_2;
    let want_join = near / (n * join) as f64 / ln2;
    let want_far = far / (n * (c - join)) as f64 / ln2;
    assert!((got.join - want_join).abs() < 1e-4, "join: {} vs {want_join}", got.join);
    assert!((got.far - want_far).abs() < 1e-4, "far: {} vs {want_far}", got.far);
}
