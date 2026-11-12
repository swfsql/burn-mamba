//! [`RowEnd`] against its direct form. For each row, the direct form takes
//! `[start | the real samples of the last segment]` on the host and keeps its
//! last `width` entries. `start` is the origin field for a row with a reset,
//! else the incoming field.
//!
//! The values are small integers, so each read is exact in `f16` too.

use super::*;
use burn::tensor::TensorData;
use burn_stack::utils::test_helpers::test_device;

/// The entries of each sample.
const FEATURES: usize = 2;

/// One row of a test call, as text. Each `.` is a real position and each `p`
/// is a padded position. A `|` comes before the first position of a segment.
struct Row {
    pad: Vec<bool>,
    starts: Vec<usize>,
}

fn parse(row: &str) -> Row {
    let mut out = Row { pad: Vec::new(), starts: Vec::new() };
    for c in row.chars() {
        match c {
            '|' => out.starts.push(out.pad.len()),
            '.' => out.pad.push(false),
            'p' => out.pad.push(true),
            _ => panic!("unknown position {c:?} in {row:?}"),
        }
    }
    out
}

fn sample(b: usize, p: usize, f: usize) -> f32 {
    (200 * b + 8 * p + f + 1) as f32
}

fn incoming(b: usize, j: usize, f: usize) -> f32 {
    -sample(b, j, f)
}

fn origin(b: usize, j: usize, f: usize) -> f32 {
    1000.0 + sample(b, j, f)
}

/// `[batch, len, FEATURES]` from `value(b, p, f)`.
fn tensor(batch: usize, len: usize, value: impl Fn(usize, usize, usize) -> f32, device: &Device) -> Tensor<3> {
    let mut data = Vec::with_capacity(batch * len * FEATURES);
    for b in 0..batch {
        for p in 0..len {
            for f in 0..FEATURES {
                data.push(value(b, p, f));
            }
        }
    }
    Tensor::from_data(TensorData::new(data, [batch, len, FEATURES]), device)
}

/// The direct form of one read, `[batch, width, FEATURES]` on the host. Each
/// row starts from zeros when `zeros` is true. Else a row with a reset starts
/// from the origin field, and another row from the incoming field.
fn direct(rows: &[Row], width: usize, zeros: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for (b, row) in rows.iter().enumerate() {
        let first = row.starts.last().copied();
        let start = |j, f| match (zeros, first) {
            (true, _) => 0.0,
            (false, None) => incoming(b, j, f),
            (false, Some(_)) => origin(b, j, f),
        };
        let mut entries: Vec<Vec<f32>> = (0..width).map(|j| (0..FEATURES).map(|f| start(j, f)).collect()).collect();
        for p in first.unwrap_or(0)..row.pad.len() {
            if !row.pad[p] {
                entries.push((0..FEATURES).map(|f| sample(b, p, f)).collect());
            }
        }
        out.extend(entries[entries.len() - width..].iter().flatten());
    }
    out
}

/// Builds the [`RowEnd`] of `rows`, reads along `axis` (1 or 2), and compares
/// each read with its direct form. A call with no `p` has no padding, and a
/// call with no `|` has no resets.
fn check(rows: &[&str], chunk_len: usize, micro_steps: usize, width: usize, axis: usize) {
    let device = test_device();
    let rows: Vec<Row> = rows.iter().map(|r| parse(r)).collect();
    let batch = rows.len();
    let len = rows[0].pad.len();
    assert!(rows.iter().all(|r| r.pad.len() == len), "each row has the length of the call");

    let any_pad = rows.iter().any(|r| r.pad.contains(&true));
    let pad_bs = any_pad.then(|| {
        let flags: Vec<bool> = rows.iter().flat_map(|r| r.pad.clone()).collect();
        Tensor::<2, Bool>::from_data(TensorData::new(flags, [batch, len]), &device)
    });
    let any_reset = rows.iter().any(|r| !r.starts.is_empty());
    let nchunks = len.div_ceil(chunk_len);
    let reset_bn = any_reset.then(|| {
        let mut flags = vec![false; batch * nchunks];
        for (b, row) in rows.iter().enumerate() {
            for &s in &row.starts {
                assert_eq!(s % chunk_len, 0, "a segment starts at a chunk start");
                flags[b * nchunks + s / chunk_len] = true;
            }
        }
        Tensor::<2, Bool>::from_data(TensorData::new(flags, [batch, nchunks]), &device)
    });
    let row_end = RowEnd::new(pad_bs.as_ref(), reset_bn.as_ref().map(|r| (r, chunk_len)), micro_steps);

    // Axis 2 reads the transposed tensors, as a conv window does.
    let on_axis = |t: Tensor<3>| if axis == 2 { t.swap_dims(1, 2) } else { t };
    let samples = on_axis(tensor(batch, len, sample, &device));
    let host = |t: Tensor<3>| -> Vec<f32> { on_axis(t).into_data().try_into_vec_as::<f32>().unwrap() };

    let got = row_end.read(
        samples.clone(),
        axis,
        width,
        on_axis(tensor(batch, width, incoming, &device)),
        Some(on_axis(tensor(batch, width, origin, &device))),
    );
    assert_eq!(host(got), direct(&rows, width, false), "the read of a field");

    let got = row_end.read_after_zeros(samples, axis, width);
    assert_eq!(host(got), direct(&rows, width, true), "the read after zeros");
}

#[test]
fn one_narrow_without_padding_or_resets() {
    check(&["......", "......"], 2, 1, 3, 2);
    check(&["......", "......"], 2, 1, 1, 1);
}

/// A full row, a short row, a row with no real position and a row with one.
#[test]
fn padding() {
    check(&["......", "...ppp", "pppppp", ".ppppp"], 2, 1, 3, 2);
    check(&["......", "...ppp", "pppppp", ".ppppp"], 2, 1, 1, 1);
}

/// The last segment of the first row is shorter than `width`. The second row
/// has no reset in a call with resets, and the third row resets at its first
/// position. The second call has the same segments, with padding.
#[test]
fn reset_with_a_last_segment_shorter_than_width() {
    check(&["....|..", "......", "|......", "..|...."], 2, 1, 3, 2);
    check(&["...p|.p", ".....p", "|..pppp", "..|.ppp"], 2, 1, 3, 2);
}

/// The last segment of the first two rows holds only padding, so their read
/// is the origin field. The row end of the second row is in an earlier
/// segment.
#[test]
fn last_segment_of_padding_only() {
    check(&["..|..|pp", "..|.p|pp", "|pp|..|.."], 2, 1, 3, 2);
    check(&["..|..|pp", "..|.p|pp", "|pp|..|.."], 2, 1, 1, 1);
}

/// With no padding, a read of at most one token cannot cross the last reset:
/// it is one `narrow`.
#[test]
fn one_narrow_with_resets_within_one_token() {
    check(&["....|..", "|......", "......"], 4, 2, 2, 1);
    check(&["....|..", "|......", "......"], 4, 2, 1, 1);
}

#[test]
#[should_panic(expected = "needs the origin field")]
fn resets_need_the_origin_field() {
    let device = test_device();
    let reset_bn = Tensor::<2, Bool>::from_data(TensorData::new(vec![false, true], [1, 2]), &device);
    let row_end = RowEnd::new(None, Some((&reset_bn, 2)), 2);
    let samples = tensor(1, 4, sample, &device);
    row_end.read(samples, 1, 1, tensor(1, 1, incoming, &device), None);
}

/// The read takes entries of `start`, with padding, with no padding and with
/// resets.
#[test]
fn call_shorter_than_width() {
    check(&["..", ".p", "pp"], 3, 1, 4, 2);
    check(&["..", ".."], 3, 1, 4, 2);
    check(&["|..", "..", "|.p"], 3, 1, 4, 2);
}
