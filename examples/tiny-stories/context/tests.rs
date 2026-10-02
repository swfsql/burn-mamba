//! [`score`] reads the right rows. Its bits equal those of a reference that
//! scores each character of each continuation with its own `forward` (over the
//! prompt and the continuation up to that character, the last row), across
//! more than one batch. With no prompt, the reference reads the first
//! character from `prime`. The reference boosts each character by itself. The
//! check is per story.

use super::{BATCH, Bits, score};
use crate::dataset::VOCAB_SIZE;
use crate::test_net::tiny_net;
use crate::training::Run;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn_mamba::prelude::*;
use burn_stack::utils::ClassCursors;
use burn_stack::utils::test_helpers::{dtype_tol, test_device};

#[test]
fn score_reads_the_rows_of_the_continuation() {
    let device = test_device();
    device.seed(0);
    let net = tiny_net(&device);
    let run = Run {
        ssd_path: MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None)),
        pack_align: None,
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
    let own = |i: usize| i;
    let other = |i: usize| (i + 1) % n;
    let pairings: [&dyn Fn(usize) -> usize; 2] = [&own, &other];
    let alphas = [0.0, 0.5];
    let got = score(&net, &run, &device, &prompts, &continuations, join, BATCH, &pairings, &alphas);

    // The reference: the log-probabilities of the next character after
    // `tokens`, with the class latents in front.
    let next_log_p = |tokens: &[u8]| -> Tensor<1> {
        let logits = if tokens.is_empty() {
            let (logits, _) = net.prime(1, None, Some(&mut ClassCursors::stream()));
            logits.expect("the tiny network has class latents")
        } else {
            let tokens: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let x = Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).reshape([1, tokens.len()]);
            let (logits, _) = net.forward(x, None, run.ssd_path.clone(), None, None);
            let rows = logits.dims()[1];
            logits.narrow(1, rows - 1, 1).flatten::<2>(0, 1)
        };
        log_softmax(logits.flatten::<1>(0, 1), 0)
    };
    let nll = |log_p: Tensor<1>, target: u8| -(log_p.narrow(0, target as usize, 1).into_scalar::<f32>() as f64);
    // The summed negative log-likelihood of the join and of the far part of
    // each story: no prompt, then each pairing at each α.
    let mut none = vec![(0.0, 0.0); n];
    let mut paired = vec![vec![vec![(0.0, 0.0); n]; alphas.len()]; pairings.len()];
    let add = |sums: &mut (f64, f64), k: usize, x: f64| {
        if k < join {
            sums.0 += x;
        } else {
            sums.1 += x;
        }
    };
    for i in 0..n {
        for k in 0..c {
            let target = continuations[i][k];
            let log_p_none = next_log_p(&continuations[i][..k]);
            add(&mut none[i], k, nll(log_p_none.clone(), target));
            for (j, pair) in pairings.iter().enumerate() {
                let tokens: Vec<u8> = prompts[pair(i)].iter().chain(&continuations[i][..k]).copied().collect();
                let log_p = next_log_p(&tokens);
                for (a, &alpha) in alphas.iter().enumerate() {
                    let boosted = log_softmax(
                        log_p.clone().mul_scalar(1.0 + alpha) - log_p_none.clone().mul_scalar(alpha),
                        0,
                    );
                    add(&mut paired[j][a][i], k, nll(boosted, target));
                }
            }
        }
    }
    let ln2 = std::f64::consts::LN_2;
    let bits = |&(near, far): &(f64, f64)| Bits {
        join: near / join as f64 / ln2,
        far: far / (c - join) as f64 / ln2,
    };
    let tol = f64::from(dtype_tol(1e-4));
    let check = |name: &str, got: &[Bits], want: &[(f64, f64)]| {
        assert_eq!(got.len(), n, "{name}: one score per story");
        for (i, (got, want)) in got.iter().zip(want.iter().map(bits)).enumerate() {
            assert!((got.join - want.join).abs() < tol, "{name}, story {i}, join: {} vs {}", got.join, want.join);
            assert!((got.far - want.far).abs() < tol, "{name}, story {i}, far: {} vs {}", got.far, want.far);
        }
    };
    check("no prompt", &got.none, &none);
    for (j, sums) in paired.iter().enumerate() {
        for (a, sums) in sums.iter().enumerate() {
            check(&format!("pairing {j}, α {}", alphas[a]), &got.pairings[j][a], sums);
        }
    }
    // The boost changes the score, so the check above can tell α apart.
    let far = |a: usize| Bits::mean(&got.pairings[0][a]).far;
    assert!((far(0) - far(1)).abs() > 10.0 * tol);
}
