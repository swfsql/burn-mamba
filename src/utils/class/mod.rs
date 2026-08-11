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
// inserted last). Markers sharing an index keep their `Vec` order.
//
// One rule places three of the four kinds: the marker names an index into the
// *original* sequence and is emitted immediately **before** the user token
// sitting there — `Start` (0) opens the sequence, `Middle` (`L/2`) splits it,
// `Custom(k)` precedes token `k`. `End` is the sole exception, having no token
// to precede: it **closes** the sequence, trailing the last user token.
//
// `Custom` is uniform in `k`, so a `Custom(k ≥ L)` simply never lands — there is
// no token `k` to precede. Should the caller keep feeding tokens past the
// announced `L`, it lands then, still *before* the next token; it never trails.
// An open-ended stream (no hint) is likewise never closed — the same reason
// `End` needs the length hint below.
//
// `step` therefore returns the output of the **last** token it emitted (and
// leaves the state after it): the user token, unless `End` follows it — `End`
// being then the sequence's latest token. Markers emitted before the user token
// only leave their mark on the state.
//
// `prime` is the call that reads *those* back. It emits the markers waiting for
// the next user token — without one, so it needs no input data — and returns the
// last of them (`None` when none were waiting). Being exactly `step`'s opening
// half, `prime` followed by `step` emits what that `step` alone would have, in
// the same order: seedless generation is `prime` → sample → `step` → sample → …
// `End` is never primed — it *closes* the sequence, so it belongs to the call
// carrying the last user token, which is why it is the one marker `step` (and
// `forward`) already hands back.
//
// Placement is **streamed**, identically for `forward` (a chunk of the
// sequence) and `step` (a single token): [`ClassCursors`] carries one
// `full_len` hint — the length of the whole sequence the call is part of — plus
// one cursor per level, each recording how much of *that* level's output
// sequence earlier calls already emitted. From those:
//
//   * a marker whose output position is behind the cursor was emitted by an
//     earlier call and is skipped — so `Start` fires only while the cursor is
//     still at 0, and a resumed stream does not re-insert it;
//   * `Middle`/`End` resolve only against the whole sequence, so they **panic**
//     without a `full_len` hint (`Start`/`Custom` are length-independent and
//     work on an open-ended stream);
//   * a marker landing exactly at a chunk's end is emitted by that chunk only
//     if the chunk *closes* the sequence — otherwise it opens the next one, so
//     splitting a sequence anywhere leaves the placement unchanged.
//
// No cursors (`None`) keeps the two calls' historical defaults: `forward` treats
// its argument as the whole sequence (`full_len` = its length, cursors at 0),
// `step` injects nothing at all.

/// Position marker for a learnable class **token** inserted into a *network's*
/// input sequence (embedding width = the network input width / "d_input").
#[derive(Config, Debug)]
pub enum ClassToken {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert before the middle token of the original sequence (index `L/2`).
    /// Needs a [`ClassCursors::full_len`] hint.
    Middle,
    /// **Close** the sequence: appended after its last token (index `L`) — the
    /// only marker that trails one instead of preceding it, and so the token a
    /// closing `step` returns. Needs a [`ClassCursors::full_len`] hint.
    End,
    /// Insert before the original sequence's token `index` — for any `index`,
    /// so one at or past the end never lands (no such token), unless the caller
    /// feeds tokens past the announced length and it precedes the next one.
    Custom(usize),
}

/// Position marker for a learnable class **latent** inserted into a *layer's*
/// working sequence (embedding width = `d_model`).
#[derive(Config, Debug)]
pub enum ClassLatent {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert before the middle token of the original sequence (index `L/2`).
    /// Needs a [`ClassCursors::full_len`] hint.
    Middle,
    /// **Close** the sequence: appended after its last token (index `L`) — the
    /// only marker that trails one instead of preceding it, and so the token a
    /// closing `step` returns. Needs a [`ClassCursors::full_len`] hint.
    End,
    /// Insert before the original sequence's token `index` — for any `index`,
    /// so one at or past the end never lands (no such token), unless the caller
    /// feeds tokens past the announced length and it precedes the next one.
    Custom(usize),
}

/// Shared behaviour of the [`ClassToken`] / [`ClassLatent`] position markers,
/// letting one generic helper place either kind.
pub trait ClassMarker: Clone {
    /// Insertion index measured against the *original* sequence length `orig_len`.
    fn insert_pos(&self, orig_len: usize) -> usize;
    /// Tie-break rank among markers sharing an index (`Start`<`Middle`<`End`<`Custom`).
    fn group_rank(&self) -> usize;
    /// Whether this marker's position is only defined against the whole sequence
    /// (`Middle`/`End`), so placing it requires a [`ClassCursor::full_len`] hint.
    fn needs_full_len(&self) -> bool;
    /// Whether this marker *closes* the sequence — trailing its last token
    /// rather than preceding one. `End` alone does.
    fn closes_sequence(&self) -> bool;
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
            fn needs_full_len(&self) -> bool {
                matches!(self, Self::Middle | Self::End)
            }
            fn closes_sequence(&self) -> bool {
                matches!(self, Self::End)
            }
        }
    };
}
impl_class_marker!(ClassToken);
impl_class_marker!(ClassLatent);

/// Placement state of **one** class-marker level (one container's own markers):
/// how far into that level's *output* sequence the previous calls got, and the
/// length that positions are measured against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassCursor {
    /// Output-sequence position reached so far — user tokens *and* already
    /// emitted class markers (0 ⇒ the sequence has not started).
    pub offset: usize,
    /// Length of the whole sequence this level receives, its own markers
    /// excluded. `None` ⇒ an open-ended stream: `Start`/`Custom` still place
    /// exactly, `Middle`/`End` panic.
    pub full_len: Option<usize>,
}

impl ClassCursor {
    /// A cursor at `offset` measured against `full_len`.
    pub fn at(offset: usize, full_len: Option<usize>) -> Self {
        Self { offset, full_len }
    }

    /// A fresh cursor for a call that covers the entire sequence of length
    /// `len` — what `forward` assumes when given no cursors.
    pub fn whole(len: usize) -> Self {
        Self {
            offset: 0,
            full_len: Some(len),
        }
    }
}

/// Everything a `forward` (chunk) or `step` (single token) call needs in order
/// to place the class tokens / class latents of a whole network: one
/// full-length hint plus one cursor per class-marker level. Pass the **same**
/// value to every call of a sequence — each call advances the cursors it uses,
/// so the next one resumes exactly where this one stopped.
///
/// The levels nest, and each cursor counts the sequence *its* level sees (which
/// already includes whatever the levels below it spliced in):
///
/// ```text
/// network     LatentNetwork's own ClassTokens  (before `in_proj`)
/// stack       Layers'/BidiLayers' own ClassLatents
/// per_layer   one cursor per virtual layer, for that Layer's ClassLatents
/// ```
///
/// [`Self::full_len`] is the length of the user sequence handed to the
/// outermost call; the inner levels' lengths are derived from it.
///
/// To read a marker back out of a *chunked* `forward`: its position in the whole
/// output is the container's `class_*_output_indices(full_len)`; subtracting the
/// level's cursor as it was *before* that call gives its index inside the
/// chunk's own output. A `step` returns one token — the last it emitted, which
/// is the user token unless an `End` marker trails it, `End` being then the
/// sequence's true last token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassCursors {
    /// Total length of the user sequence all the calls form together, or `None`
    /// for an open-ended stream (then `Middle`/`End` markers panic).
    pub full_len: Option<usize>,
    /// Cursor of a network's own [`ClassToken`]s (unused by a bare layer stack).
    pub network: usize,
    /// Cursor of a layer container's own [`ClassLatent`]s.
    pub stack: usize,
    /// One cursor per **virtual** layer, for the per-layer [`ClassLatent`]s.
    /// Left empty it is sized (with zeros) on the first call.
    pub per_layer: Vec<usize>,
}

impl ClassCursors {
    /// Cursors at the start of a sequence of known total length — the form that
    /// enables `Middle`/`End` markers, and what a whole-sequence `forward` uses
    /// when given none.
    pub fn new(full_len: usize) -> Self {
        Self {
            full_len: Some(full_len),
            ..Default::default()
        }
    }

    /// Cursors at the start of an open-ended stream (unknown total length):
    /// `Start`/`Custom` markers place exactly, `Middle`/`End` panic.
    pub fn stream() -> Self {
        Self::default()
    }

    /// Size [`Self::per_layer`] for a stack of `n` virtual layers (idempotent).
    pub(crate) fn fit(&mut self, n: usize) {
        if self.per_layer.is_empty() {
            self.per_layer = vec![0; n];
        }
        assert_eq!(
            self.per_layer.len(),
            n,
            "one class-latent cursor per virtual layer"
        );
    }

    /// Enter the inner level, whose sequence is longer by the `extra` markers
    /// this level splices in. Returns the previous hint, for [`Self::leave`].
    pub(crate) fn enter(&mut self, extra: usize) -> Option<usize> {
        let saved = self.full_len;
        self.full_len = saved.map(|l| l + extra);
        saved
    }

    /// Leave the inner level, restoring the hint [`Self::enter`] returned.
    pub(crate) fn leave(&mut self, saved: Option<usize>) {
        self.full_len = saved;
    }
}

/// Panic if any marker's position needs the whole sequence length while none is
/// known — `Middle`/`End` cannot be placed from a chunk (or a single token).
pub(crate) fn assert_full_len_known<M: ClassMarker>(
    markers: &[M],
    full_len: Option<usize>,
    who: &str,
) {
    assert!(
        full_len.is_some() || !markers.iter().any(|m| m.needs_full_len()),
        "{who}: Middle/End class markers need a full-length hint (ClassCursors::new)"
    );
}

/// Which of `markers` fall inside the next `chunk_len` user tokens, and where.
///
/// Returns `(at, marker)` pairs in output order: insert `markers[marker]`
/// *before* the chunk's `at`-th token (`at == chunk_len` ⇒ after the last one).
/// `cursor` is advanced past the whole chunk, its insertions included.
pub(crate) fn class_chunk_plan<M: ClassMarker>(
    markers: &[M],
    chunk_len: usize,
    cursor: &mut ClassCursor,
    who: &str,
) -> Vec<(usize, usize)> {
    class_plan(markers, chunk_len, cursor, false, who)
}

/// [`class_chunk_plan`] for a **prime**: the chunk carries no user token of its
/// own past the `chunk_len` tokens a lower level handed up (`0` at the level the
/// call enters), and one more user token is still to come.
///
/// It therefore differs on the chunk's trailing edge only: the markers waiting
/// there for that next token are emitted now (they precede it either way), while
/// `End` — which trails the last token rather than preceding one — is left to
/// the call that carries it. A cursor already at the announced end has no next
/// token to emit anything for, so the plan is then empty.
pub(crate) fn class_prime_plan<M: ClassMarker>(
    markers: &[M],
    chunk_len: usize,
    cursor: &mut ClassCursor,
    who: &str,
) -> Vec<(usize, usize)> {
    class_plan(markers, chunk_len, cursor, true, who)
}

/// The shared placement loop of [`class_chunk_plan`] / [`class_prime_plan`].
fn class_plan<M: ClassMarker>(
    markers: &[M],
    chunk_len: usize,
    cursor: &mut ClassCursor,
    prime: bool,
    who: &str,
) -> Vec<(usize, usize)> {
    if markers.is_empty() {
        cursor.offset += chunk_len;
        return Vec::new();
    }
    assert_full_len_known(markers, cursor.full_len, who);
    let positions = class_marker_output_indices(markers, cursor.full_len.unwrap_or(usize::MAX));

    // User tokens consumed once this chunk is done: the output positions behind
    // the cursor, minus the markers among them, plus this chunk. Feeding more
    // than an announced `full_len` is allowed — everything is already placed by
    // then, so the extra tokens simply stream on.
    let start = cursor.offset;
    let consumed = start - positions.iter().filter(|&&p| p < start).count() + chunk_len;
    // Whether this chunk reaches the announced end, i.e. carries the last user
    // token — the one and only place an `End` can go. A prime carries no token
    // of its own, so it never closes anything.
    let closes = !prime && cursor.full_len == Some(consumed);
    // Whether a prime may flush the markers waiting at the chunk's end: only
    // while a further user token is announced (or the stream is open-ended) is
    // there one for them to precede.
    let flush = prime && cursor.full_len != Some(consumed);

    let mut order: Vec<usize> = (0..markers.len()).collect();
    order.sort_by_key(|&i| positions[i]);

    let mut out = start; // running output position
    let mut at = 0usize; // chunk tokens placed before it
    let mut plan = Vec::new();
    for i in order {
        let p = positions[i];
        if p < start {
            continue; // emitted by an earlier call
        }
        let need = p - out; // user tokens preceding this marker
        if at + need > chunk_len {
            break; // the token it precedes is in a later chunk, as is it
        }
        if at + need == chunk_len {
            // Nothing left in this chunk to precede: only a closing `End` on the
            // chunk that ends the sequence belongs here — or, on a prime, the
            // markers the next token is due to be preceded by. A `Custom` waits
            // for its token — for one at/past the end, forever.
            let closing = closes && markers[i].closes_sequence();
            let pending = flush && !markers[i].closes_sequence();
            if !closing && !pending {
                break;
            }
        }
        at += need;
        out = p + 1;
        plan.push((at, i));
    }
    cursor.offset = out + (chunk_len - at);
    plan
}

/// Splice the learnable class markers `emb` (`[k, width]`, row `i` ↔
/// `markers[i]`) that fall inside the chunk `x` (`[batch, chunk_len, width]`),
/// returning the lengthened chunk and advancing `cursor` past it.
///
/// `markers` empty (or none of them landing in this chunk) ⇒ `x` unchanged.
pub(crate) fn insert_class_markers<M: ClassMarker>(
    x: Tensor<3>,
    markers: &[M],
    emb: Option<&Param<Tensor<2>>>,
    cursor: &mut ClassCursor,
    who: &str,
) -> Tensor<3> {
    let [batch, chunk_len, width] = x.dims();
    let plan = class_chunk_plan(markers, chunk_len, cursor, who);
    if plan.is_empty() {
        return x;
    }
    let emb = emb
        .expect("class-token markers present but no embedding param")
        .val();
    assert_eq!(
        emb.dims(),
        [markers.len(), width],
        "one embedding row per class marker"
    );

    let mut segments: Vec<Tensor<3>> = Vec::with_capacity(2 * plan.len() + 1);
    let mut taken = 0usize; // chunk tokens emitted so far
    for (at, i) in plan {
        if at > taken {
            segments.push(x.clone().narrow(1, taken, at - taken));
            taken = at;
        }
        segments.push(
            emb.clone()
                .narrow(0, i, 1) // [1, width]
                .unsqueeze_dim::<3>(0) // [1, 1, width]
                .expand([batch, 1, width]),
        );
    }
    if taken < chunk_len {
        segments.push(x.narrow(1, taken, chunk_len - taken));
    }
    Tensor::cat(segments, 1)
}

/// Width of the class embeddings (`[num_markers, width]`) — what a `prime`
/// sizes its rows by, having no token to read the width from. Only ever called
/// where a marker is about to be emitted, so the param is present.
pub(crate) fn class_emb_width(emb: Option<&Param<Tensor<2>>>) -> usize {
    emb.expect("class-token markers present but no embedding param")
        .val()
        .dims()[1]
}

/// The embedding row of marker `i` as one broadcast token (`[batch, width]`) —
/// the `step` counterpart of a slice of [`insert_class_markers`].
pub(crate) fn class_row(
    emb: Option<&Param<Tensor<2>>>,
    i: usize,
    batch: usize,
    width: usize,
) -> Tensor<2> {
    emb.expect("class-token markers present but no embedding param")
        .val()
        .narrow(0, i, 1)
        .expand([batch, width])
}

/// The output-sequence position of each marker (in `Vec` order) for an input of
/// length `orig_len`, without materialising any tensor. Mirrors the placement in
/// [`insert_class_markers`] — useful for reading a class token back out.
///
/// A marker that never lands (a `Custom` at or past the end — it has no token to
/// precede) reports the position it *would* take, which is then `>= orig_len +
/// (number of markers that do land)`, i.e. past the emitted sequence.
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
