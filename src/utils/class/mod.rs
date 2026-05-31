use burn::config::Config;
use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

// ===========================================================================
// Class tokens / latents (learnable sequence-inserted tokens)
// ===========================================================================
//
// A *class token* / *class latent* is a learnable embedding spliced into the
// sequence — a transformer-`[CLS]`-style register the model can read/write
// through. They are inserted at the input boundary of a container (a network's
// input for [`ClassToken`], width = the input feature width; a layer's working
// sequence for [`ClassLatent`], width = `d_model`), permanently lengthening the
// sequence for everything downstream. A container can carry any number; the
// markers below say *where* each one lands, while a single `Param<Tensor<2>>`
// of shape `[num_markers, width]` holds the embeddings (row `i` ↔ marker `i`).
//
// Insertion order (all relative to the *original* length `L`): every `Start`
// first (index 0), then `Middle` (index `L/2`, splitting the original
// sequence), then `End` (index `L`), then `Custom(index)` (explicit index,
// inserted last). Markers sharing an index keep their `Vec` order. Because
// `Middle`/`End` materialise positions that a single-token `step()` cannot
// reproduce, their presence makes `step()` panic; `Start`/`Custom` are a
// forward-time concern and are simply not re-inserted during `step()`.

/// Position marker for a learnable class **token** inserted into a *network's*
/// input sequence (embedding width = the network input width / "d_input").
#[derive(Config, Debug)]
pub enum ClassToken {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert at the middle of the original sequence (index `L/2`).  
    /// Incompatible with `step()` calls.
    Middle,
    /// Append after the whole sequence (index `L`).  
    /// Incompatible with `step()` calls.
    End,
    /// Insert at an explicit index into the original sequence.
    Custom(usize),
}

/// Position marker for a learnable class **latent** inserted into a *layer's*
/// working sequence (embedding width = `d_model`).
#[derive(Config, Debug)]
pub enum ClassLatent {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert at the middle of the original sequence (index `L/2`).
    /// Incompatible with `step()` calls.  
    Middle,
    /// Append after the whole sequence (index `L`).
    /// Incompatible with `step()` calls.  
    End,
    /// Insert at an explicit index into the original sequence.
    Custom(usize),
}

/// Shared behaviour of the [`ClassToken`] / [`ClassLatent`] position markers,
/// letting one generic helper place either kind.
pub trait ClassMarker: Clone {
    /// Insertion index measured against the *original* sequence length `orig_len`.
    fn insert_pos(&self, orig_len: usize) -> usize;
    /// Tie-break rank among markers sharing an index (`Start`<`Middle`<`End`<`Custom`).
    fn group_rank(&self) -> usize;
    /// Whether this marker is incompatible with single-token `step()`
    /// (`Middle`/`End` create positions a per-token recurrence cannot reproduce).
    fn forbids_step(&self) -> bool;
}

macro_rules! impl_class_marker {
    ($ty:ty) => {
        impl ClassMarker for $ty {
            fn insert_pos(&self, orig_len: usize) -> usize {
                match self {
                    Self::Start => 0,
                    Self::Middle => orig_len / 2,
                    Self::End => orig_len,
                    Self::Custom(index) => *index,
                }
            }
            fn group_rank(&self) -> usize {
                match self {
                    Self::Start => 0,
                    Self::Middle => 1,
                    Self::End => 2,
                    Self::Custom(_) => 3,
                }
            }
            fn forbids_step(&self) -> bool {
                matches!(self, Self::Middle | Self::End)
            }
        }
    };
}
impl_class_marker!(ClassToken);
impl_class_marker!(ClassLatent);

/// Insert the learnable class tokens `emb` (`[k, width]`, row `i` ↔ `markers[i]`)
/// into `x` (`[batch, orig_len, width]`) per the `markers`, returning the
/// lengthened sequence (`[batch, orig_len + k, width]`) and, for each marker in
/// `Vec` order, its position in the output sequence.
///
/// `markers` empty ⇒ `x` is returned unchanged with an empty index vector.
pub(crate) fn insert_class_markers<M: ClassMarker>(
    x: Tensor<3>,
    markers: &[M],
    emb: Option<&Param<Tensor<2>>>,
) -> (Tensor<3>, Vec<usize>) {
    let [batch, orig_len, width] = x.dims();
    let k = markers.len();
    if k == 0 {
        return (x, Vec::new());
    }
    let emb = emb
        .expect("class-token markers present but no embedding param")
        .val();
    assert_eq!(emb.dims(), [k, width], "one embedding row per class marker");

    // Emit in (insert_pos, group_rank, vec order) order.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&i| (markers[i].insert_pos(orig_len), markers[i].group_rank(), i));

    let mut segments: Vec<Tensor<3>> = Vec::new();
    let mut cursor = 0usize; // consumed prefix of the original sequence
    let mut out_len = 0usize; // length emitted so far
    let mut out_index = vec![0usize; k];
    for &i in &order {
        let p = markers[i].insert_pos(orig_len);
        assert!(
            p <= orig_len,
            "class-token insert index {p} > sequence length {orig_len}"
        );
        if p > cursor {
            segments.push(x.clone().narrow(1, cursor, p - cursor));
            out_len += p - cursor;
            cursor = p;
        }
        let row = emb
            .clone()
            .narrow(0, i, 1) // [1, width]
            .unsqueeze_dim::<3>(0) // [1, 1, width]
            .expand([batch, 1, width]);
        segments.push(row);
        out_index[i] = out_len;
        out_len += 1;
    }
    if cursor < orig_len {
        segments.push(x.narrow(1, cursor, orig_len - cursor));
    }
    let out = Tensor::cat(segments, 1);
    assert_eq!(out.dims(), [batch, orig_len + k, width]);
    (out, out_index)
}

/// The output-sequence position of each marker (in `Vec` order) for an input of
/// length `orig_len`, without materialising any tensor. Mirrors the placement in
/// [`insert_class_markers`] — useful for reading a class token back out.
pub(crate) fn class_marker_output_indices<M: ClassMarker>(
    markers: &[M],
    orig_len: usize,
) -> Vec<usize> {
    let k = markers.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&i| (markers[i].insert_pos(orig_len), markers[i].group_rank(), i));
    let mut cursor = 0usize;
    let mut out_len = 0usize;
    let mut out_index = vec![0usize; k];
    for &i in &order {
        let p = markers[i].insert_pos(orig_len).min(orig_len);
        if p > cursor {
            out_len += p - cursor;
            cursor = p;
        }
        out_index[i] = out_len;
        out_len += 1;
    }
    out_index
}

/// Build the embedding param for `n` class markers of the given `width`
/// (`None` when there are none — Burn has no zero-width tensors).
pub(crate) fn init_class_emb(n: usize, width: usize, device: &Device) -> Option<Param<Tensor<2>>> {
    (n > 0).then(|| {
        Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        }
        .init([n, width], device)
    })
}

/// Panic if any marker is incompatible with single-token `step()`.
pub(crate) fn assert_step_compatible<M: ClassMarker>(markers: &[M], who: &str) {
    assert!(
        !markers.iter().any(|m| m.forbids_step()),
        "{who}: Middle/End class tokens are not compatible with step()"
    );
}

/// The output-sequence position of each step-injectable marker (in `Vec` order),
/// for use by `step`'s cursor. Asserts no `Middle`/`End` (those need the full
/// length — `forward` only). `Start`/`Custom` positions are length-independent,
/// so an unbounded `orig_len` resolves them exactly.
pub(crate) fn class_step_injections<M: ClassMarker>(markers: &[M], who: &str) -> Vec<usize> {
    assert_step_compatible(markers, who);
    class_marker_output_indices(markers, usize::MAX)
}
