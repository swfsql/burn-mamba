//! The bits of whole long stories (`--long-stories`).
//!
//! The validation loss scores the first characters of each story (the
//! training window). Most stories end inside that window, so the loss tells
//! little about the end of a long story. There, the model must keep the names,
//! the objects and the events of the start in its state, and it reads
//! positions past the training window.
//!
//! The measure does not depend on the model. As [`super::context`], it reads
//! the model through a [`LogitsFn`], with the lead rows in front of the tokens.
//!
//! It reads the validation stories that are longer than [`MIN_CHARS`]
//! characters, from the whole validation split (not only the stories of the
//! validation loss). About 1.5% of the stories are that long. It scores each
//! story over its whole length in one `forward`, from a fresh state, with the
//! lead rows in front (the last lead row predicts the first character). It
//! reports the bits per character of the segments that [`BOUNDS`] cut, and of
//! each [`BIN`] characters (the position profile). The bits are those of the
//! cased text, as the validation scores it ([`cased_log_probs`]).
//!
//! **The whole valid.** The validation loss cuts each story at its window
//! (the first 1024 characters in the 1M searches). The measure also scores
//! the stories of the validation loss in full, each in one `forward`. That
//! gives the bits per character of their whole text, for any model and any
//! training protocol (windows, packed rows, length buckets).
//!
//! **Padding.** A batch holds stories of similar length. Each story is
//! right-padded (with its last character) to a multiple of [`BIN`]. The models
//! are causal, so the padding does not change the logits of the real
//! characters, and the measure does not score it.

use super::context::{LogitsFn, cased_log_probs, mean_se};
use super::tiny_stories::dataset::{FIRST_LETTER, Split, VOCAB, VOCAB_SIZE, pair, stories};
use burn::prelude::*;
use std::path::Path;

/// A story is scored if it has more characters than this.
pub const MIN_CHARS: usize = 2048;

/// The segment bounds. The segments are `0..1024`, `1024..2048` and `2048..`
/// (the tail).
pub const BOUNDS: [usize; 2] = [1024, 2048];

/// The characters of one bin of the position profile. The bounds are
/// multiples of it, and so is the padded length of a batch.
pub const BIN: usize = 256;
const _: () = assert!(BOUNDS[0] % BIN == 0 && BOUNDS[1] % BIN == 0);

/// The summed bits of each [`BIN`] of each story: `bits[i][k]` for the story
/// `tokens[i]` and bin `k` (the last bin can be partial). One call of
/// `logits_of` takes `batch` stories.
pub fn score(logits_of: &LogitsFn, device: &Device, tokens: &[Vec<u8>], batch: usize) -> Vec<Vec<f64>> {
    let n = tokens.len();
    // Similar lengths share a batch, so the padding stays small.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| tokens[i].len());
    let mut bits = vec![Vec::new(); n];
    for rows in order.chunks(batch) {
        let b = rows.len();
        let len = rows.iter().map(|&i| tokens[i].len()).max().unwrap().div_ceil(BIN) * BIN;
        // Per position: the input pair, the column of its id, the column of its
        // case, whether it is real, and whether its case counts (a letter).
        let (mut pairs, mut id_cols, mut case_cols, mut real, mut letter) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &i in rows {
            let story = &tokens[i];
            for k in 0..len {
                let [id, flag] = pair(story[k.min(story.len() - 1)]);
                pairs.extend([id, flag]);
                id_cols.push(id);
                case_cols.push(VOCAB_SIZE as i32 + flag);
                let is_real = k < story.len();
                real.push(f32::from(u8::from(is_real)));
                letter.push(f32::from(u8::from(is_real && id as usize >= FIRST_LETTER)));
            }
        }
        let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), device).reshape([b, len, 2]);
        let logits = logits_of(x);
        // The lead rows come first, then one row per token. The row of token
        // `k − 1` predicts token `k`.
        let lead = logits.dims()[1] - len;
        assert!(lead >= 1, "no row predicts the first character: the model has no lead rows");
        let log_p = cased_log_probs(logits.narrow(1, lead - 1, len));
        let column = |cols: Vec<i32>| Tensor::<1, Int>::from_ints(cols.as_slice(), device).reshape([b, len, 1]);
        let weight = |w: Vec<f32>| Tensor::<1>::from_floats(w.as_slice(), device).reshape([b, len]);
        let id_bl = log_p.clone().gather(2, column(id_cols)).reshape([b, len]);
        let case_bl = log_p.gather(2, column(case_cols)).reshape([b, len]);
        let nll_bl = (id_bl * weight(real) + case_bl * weight(letter)).neg();
        let bins = len / BIN;
        // One read-back per batch.
        let nll: Vec<f64> = nll_bl.reshape([b, bins, BIN]).sum_dim(2).into_data().try_into_vec_as().unwrap();
        for (&i, row) in rows.iter().zip(nll.chunks(bins)) {
            let real_bins = tokens[i].len().div_ceil(BIN);
            bits[i] = row[..real_bins].iter().map(|nats| nats / std::f64::consts::LN_2).collect();
        }
    }
    bits
}

/// One line per story: `story` is its index in the validation split (so two
/// models pair up on it), `bins` the summed bits of each [`BIN`].
fn story_lines(indices: &[usize], lens: &[usize], bits: &[Vec<f64>]) -> String {
    (0..indices.len())
        .map(|i| {
            let bins: Vec<String> = bits[i].iter().map(f64::to_string).collect();
            format!("{{\"story\":{},\"len\":{},\"bins\":[{}]}}\n", indices[i], lens[i], bins.join(","))
        })
        .collect()
}

/// Score the long stories of the validation split (see the module header),
/// `batch` stories per `forward`. Print the bits per character of each
/// segment and of each bin. Then score the stories of the validation loss
/// (the first `valid_stories`) in full: the bits per character of the whole
/// text, of its chars `0..1024` and of the rest. Save the summary as
/// `long-stories.json` in `artifacts` (the artifacts directory), and the bits
/// of each bin of each story in `long-stories.jsonl` and `whole-valid.jsonl`.
pub fn long_stories(logits_of: &LogitsFn, device: &Device, artifacts: &Path, valid_stories: usize, batch: usize) {
    let split = Split::Valid.capacity();
    let all: Vec<Vec<u8>> = stories(Split::Valid, split).iter().map(|story| VOCAB.encode_cased(story)).collect();
    // `indices`: the index of each scored story in the validation split.
    let (indices, tokens): (Vec<usize>, Vec<Vec<u8>>) = all
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, tokens)| tokens.len() > MIN_CHARS)
        .unzip();
    let n = tokens.len();
    assert!(n >= 2, "no two validation stories are longer than {MIN_CHARS} characters");
    let bits = score(logits_of, device, &tokens, batch);
    let lens: Vec<usize> = tokens.iter().map(Vec::len).collect();

    // The bits and the characters of story `i` in `from..to`. The bounds are
    // multiples of [`BIN`].
    let range = |i: usize, from: usize, to: usize| {
        let to = to.min(lens[i]);
        let bits: f64 = bits[i][from / BIN..to.div_ceil(BIN)].iter().sum();
        (bits, to.saturating_sub(from))
    };
    // Per segment: the bits per character of all its characters, and the mean
    // over the stories of the bits per character of each (± its s.e.).
    let segment = |from: usize, to: usize| {
        let per_story: Vec<(f64, usize)> = (0..n).map(|i| range(i, from, to)).collect();
        let (bits, chars) = per_story.iter().fold((0.0, 0), |(b, c), &(x, k)| (b + x, c + k));
        let means: Vec<f64> = per_story.iter().map(|&(x, k)| x / k as f64).collect();
        (bits / chars as f64, mean_se(&means), chars)
    };
    let max_len = *lens.iter().max().unwrap();
    let segments = [
        (0, BOUNDS[0]),
        (BOUNDS[0], BOUNDS[1]),
        (BOUNDS[1], max_len),
        (0, max_len),
    ];
    let names = segments.map(|(from, to)| {
        if (from, to) == (0, max_len) {
            "all".to_owned()
        } else if to == max_len {
            format!("{from}..")
        } else {
            format!("{from}..{to}")
        }
    });
    let results = segments.map(|(from, to)| segment(from, to));
    // The later segments against the first one, story by story.
    let diff = |s: usize| {
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                let ((a, ka), (b, kb)) = (range(i, segments[s].0, segments[s].1), range(i, 0, BOUNDS[0]));
                a / ka as f64 - b / kb as f64
            })
            .collect();
        mean_se(&xs)
    };

    println!(
        "long stories: {n} of {split} validation stories with more than {MIN_CHARS} characters \
         ({}..{max_len} characters, mean {:.0})",
        lens.iter().min().unwrap(),
        lens.iter().sum::<usize>() as f64 / n as f64
    );
    println!("  bits/char  {}", names.iter().map(|s| format!("{s:>16}")).collect::<String>());
    println!(
        "  per char   {}",
        results.iter().map(|r| format!("{:>16.4}", r.0)).collect::<String>()
    );
    println!(
        "  per story  {}",
        results.iter().map(|r| format!("{:>8.4} ± {:.4}", r.1.0, r.1.1)).collect::<String>()
    );
    let (d1, d2) = (diff(1), diff(2));
    println!(
        "  later − first (per story): {} {:+.4} ± {:.4}, {} {:+.4} ± {:.4}",
        names[1], d1.0, d1.1, names[2], d2.0, d2.1
    );
    // The position profile: the bits per character of each bin, over the
    // stories that reach it.
    let profile: Vec<(usize, usize, f64)> = (0..max_len.div_ceil(BIN))
        .map(|k| {
            let (bits, chars, stories) = (0..n)
                .filter_map(|i| bits[i].get(k).map(|&x| (x, range(i, k * BIN, (k + 1) * BIN).1)))
                .fold((0.0, 0, 0), |(b, c, s), (x, ch)| (b + x, c + ch, s + 1));
            (k * BIN, stories, bits / chars as f64)
        })
        .collect();
    println!(
        "  profile (from: bits/char, stories): {}",
        profile
            .iter()
            .map(|(from, stories, b)| format!("{from}: {b:.4} {stories}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // The whole valid: the stories of the validation loss, each in full.
    let m = valid_stories.min(split);
    let whole = &all[..m];
    let whole_bits = score(logits_of, device, whole, batch);
    let whole_lens: Vec<usize> = whole.iter().map(Vec::len).collect();
    let (mut first, mut past, mut first_chars, mut past_chars) = (0.0, 0.0, 0, 0);
    for (bins, &len) in whole_bits.iter().zip(&whole_lens) {
        first += bins.iter().take(BOUNDS[0] / BIN).sum::<f64>();
        past += bins.iter().skip(BOUNDS[0] / BIN).sum::<f64>();
        first_chars += len.min(BOUNDS[0]);
        past_chars += len.saturating_sub(BOUNDS[0]);
    }
    let longer = whole_lens.iter().filter(|&&len| len > BOUNDS[0]).count();
    let whole_chars = first_chars + past_chars;
    let whole_all = (first + past) / whole_chars as f64;
    let (whole_first, whole_past) = (first / first_chars as f64, past / past_chars as f64);
    println!(
        "whole valid: the first {m} validation stories in full ({whole_chars} characters, {longer} stories \
         longer than {cut}): bits/char {whole_all:.4}; chars 0..{cut} {whole_first:.4}, past {cut} {whole_past:.4}",
        cut = BOUNDS[0]
    );

    let segments_json: Vec<String> = (0..segments.len())
        .map(|s| {
            let (per_char, (mean, se), chars) = results[s];
            format!(
                "{{\"name\":\"{}\",\"from\":{},\"to\":{},\"chars\":{chars},\"bits\":{per_char},\
                 \"story_mean\":[{mean},{se}]}}",
                names[s], segments[s].0, segments[s].1
            )
        })
        .collect();
    let profile_json: Vec<String> = profile
        .iter()
        .map(|(from, stories, b)| format!("{{\"from\":{from},\"stories\":{stories},\"bits\":{b}}}"))
        .collect();
    let json = format!(
        "{{\"stories\":{n},\"valid_stories\":{split},\"min_chars\":{MIN_CHARS},\"bin\":{BIN},\
         \"segments\":[{}],\"later_minus_first\":[[{},{}],[{},{}]],\"profile\":[{}],\
         \"whole_valid\":{{\"stories\":{m},\"chars\":{whole_chars},\"longer\":{longer},\"bits\":{whole_all},\
         \"first\":{whole_first},\"past\":{whole_past}}}}}\n",
        segments_json.join(","),
        d1.0,
        d1.1,
        d2.0,
        d2.1,
        profile_json.join(",")
    );
    let path = artifacts.join("long-stories.json");
    std::fs::write(&path, json).expect("failed to write long-stories.json");
    println!("saved to {path:?}");
    for (name, lines) in [
        ("long-stories.jsonl", story_lines(&indices, &lens, &bits)),
        ("whole-valid.jsonl", story_lines(&(0..m).collect::<Vec<_>>(), &whole_lens, &whole_bits)),
    ] {
        let path = artifacts.join(name);
        std::fs::write(&path, lines).unwrap_or_else(|e| panic!("failed to write {name}: {e}"));
        println!("saved to {path:?}");
    }
}
