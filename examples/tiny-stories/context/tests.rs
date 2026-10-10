//! [`score`] reads the right rows. Its bits equal those of a reference that
//! scores each character of each continuation with its own `forward` (over the
//! prompt and the continuation up to that character, the last row), across
//! more than one batch. With no prompt, the reference reads the first
//! character from `prime`. The reference boosts each character by itself, the
//! id and the case apart. The check is per story, on cased text.

use super::{BATCH, Bits, remention_marks, score};
use crate::dataset::{FIRST_LETTER, UPPER, VOCAB, VOCAB_SIZE, pair};
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
        vram_cleanup_every: None,
    };
    // Two batches, the second one partial.
    let (n, p, c, join) = (BATCH + 1, 5, 6, 2);
    // Cased tokens: every other letter is upper case.
    let ids = |i: usize, len: usize, salt: usize| -> Vec<u8> {
        (0..len)
            .map(|k| {
                let id = ((k * 7 + i * 11 + salt) % VOCAB_SIZE) as u8;
                if (k + i) % 2 == 0 && id as usize >= FIRST_LETTER { id | UPPER } else { id }
            })
            .collect()
    };
    let prompts: Vec<Vec<u8>> = (0..n).map(|i| ids(i, p, 1)).collect();
    let continuations: Vec<Vec<u8>> = (0..n).map(|i| ids(i, c, 5)).collect();
    let own = |i: usize| i;
    let other = |i: usize| (i + 1) % n;
    let pairings: [&dyn Fn(usize) -> usize; 2] = [&own, &other];
    let alphas = [0.0, 0.5];
    // Two mark sets with no pattern of the join or the far part.
    let marks: Vec<Vec<Vec<bool>>> = (0..2)
        .map(|s| (0..n).map(|i| (0..c).map(|k| (i + k + s) % 3 == 0).collect()).collect())
        .collect();
    let logits_of = |x: Tensor<3, Int>| net.forward(x, None, run.ssd_path.clone(), None, None).0;
    let got = score(&logits_of, &device, &prompts, &continuations, join, BATCH, &pairings, &alphas, &marks);

    // The reference: the log-probabilities of the next character after
    // `tokens`, with the class latents in front: the ids `[VOCAB_SIZE]`, and
    // the case `[2]` (lower, upper).
    let next_log_p = |tokens: &[u8]| -> (Tensor<1>, Tensor<1>) {
        let logits = if tokens.is_empty() {
            let (logits, _) = net.prime(1, None, Some(&mut ClassCursors::stream()));
            logits.expect("the tiny network has class latents")
        } else {
            let pairs: Vec<i32> = tokens.iter().flat_map(|&t| pair(t)).collect();
            let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), &device).reshape([1, tokens.len(), 2]);
            let (logits, _) = net.forward(x, None, run.ssd_path.clone(), None, None);
            let rows = logits.dims()[1];
            logits.narrow(1, rows - 1, 1).flatten::<2>(0, 1)
        };
        let logits = logits.flatten::<1>(0, 1);
        let z = logits.clone().narrow(0, VOCAB_SIZE, 1);
        let case = Tensor::cat(vec![z.zeros_like(), z], 0);
        (log_softmax(logits.narrow(0, 0, VOCAB_SIZE), 0), log_softmax(case, 0))
    };
    let at = |log_p: &Tensor<1>, k: usize| log_p.clone().narrow(0, k, 1).into_scalar::<f32>() as f64;
    let nll = |(ids, case): (Tensor<1>, Tensor<1>), target: u8| {
        let [id, flag] = pair(target);
        let case_bits = if id as usize >= FIRST_LETTER { at(&case, flag as usize) } else { 0.0 };
        -(at(&ids, id as usize) + case_bits)
    };
    // The summed negative log-likelihood of the join and of the far part of
    // each story: no prompt, then each pairing at each α.
    let mut none = vec![(0.0, 0.0); n];
    let mut paired = vec![vec![vec![(0.0, 0.0); n]; alphas.len()]; pairings.len()];
    // `marked[k][s][i]`: no prompt (k 0), then each pairing at `alphas[0]`.
    let mut marked = vec![vec![vec![0.0; n]; marks.len()]; 1 + pairings.len()];
    let mut mark = |k: usize, i: usize, ch: usize, x: f64| {
        for (s, set) in marks.iter().enumerate() {
            if set[i][ch] {
                marked[k][s][i] += x;
            }
        }
    };
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
            let x_none = nll(log_p_none.clone(), target);
            add(&mut none[i], k, x_none);
            mark(0, i, k, x_none);
            for (j, pair) in pairings.iter().enumerate() {
                let tokens: Vec<u8> = prompts[pair(i)].iter().chain(&continuations[i][..k]).copied().collect();
                let log_p = next_log_p(&tokens);
                for (a, &alpha) in alphas.iter().enumerate() {
                    let mix = |full: &Tensor<1>, none: &Tensor<1>| {
                        log_softmax(full.clone().mul_scalar(1.0 + alpha) - none.clone().mul_scalar(alpha), 0)
                    };
                    let boosted = (mix(&log_p.0, &log_p_none.0), mix(&log_p.1, &log_p_none.1));
                    let x = nll(boosted, target);
                    add(&mut paired[j][a][i], k, x);
                    if a == 0 {
                        mark(1 + j, i, k, x);
                    }
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
    // The marked sums, in bits (a sum over up to `c` characters).
    assert_eq!(got.marked.len(), 1 + pairings.len(), "no prompt, then one score per pairing");
    for (k, want) in marked.iter().enumerate() {
        for (s, want) in want.iter().enumerate() {
            for (i, want) in want.iter().enumerate() {
                let (got, want) = (got.marked[k][s][i], want / ln2);
                assert!((got - want).abs() < c as f64 * tol, "marked, score {k}, set {s}, story {i}: {got} vs {want}");
            }
        }
    }
}

#[test]
fn remention_marks_keep_the_rare_words_of_the_own_prompt() {
    let enc = |t: &str| VOCAB.encode_cased(t);
    // "the" is in every story, "ball" and "tom" in one story each. The 10
    // stories put the share of a word in one story at 0.1 (below
    // `REMENTION_DF`), and of "the" at 1. The case does not matter: "Tom" of
    // the prompt is "tom" of the continuation.
    let mut prompts = vec![enc("Tom has the ball. ")];
    let mut continuations = vec![enc("so The ball and tom and the cat. Tomato")];
    for i in 1..10 {
        prompts.push(enc(&format!("The story {i}. ")));
        continuations.push(enc(&format!("the end of it {i}. The end.")));
    }
    let join = 2;
    let [letters, first] = remention_marks(&prompts, &continuations, join);
    let on = |set: &[bool]| -> Vec<usize> { (0..set.len()).filter(|&k| set[k]).collect() };
    // "ball" (at 7) and "tom" (at 16) are rare and in the own prompt. "the"
    // is common, "cat" is not in the prompt, "tomato" (at 33) is another
    // word, and "so" is short.
    assert_eq!(VOCAB.decode(&continuations[0][7..11]), "ball");
    assert_eq!(on(&letters[0]), vec![7, 8, 9, 10, 16, 17, 18]);
    assert_eq!(on(&first[0]), vec![7, 16]);
    assert!(letters[1..].iter().all(|row| !row.contains(&true)), "the other stories re-mention nothing rare");
    // A word that starts before `join` does not count.
    let [late, _] = remention_marks(&prompts, &continuations, 10);
    assert_eq!(on(&late[0]), vec![16, 17, 18]);
}
