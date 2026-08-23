//! Character-level [TinyStories-GPT4-clean] corpus: a stream of single-character
//! tokens over a **case-folded ASCII** alphabet, windowed into fixed-length
//! next-character training sequences.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! # Alphabet
//!
//! The dataset is documented (and verified by its cleaning pipeline) to contain
//! exactly 74 distinct ASCII characters — the 52 cased letters plus
//! ``\n !"$',-.0123456789:;?``. Case-folding the letters leaves [`ALPHABET`]:
//! 48 tokens, every one of which actually occurs. That is the whole vocabulary;
//! there is no `<unk>`, no `<bos>`, and no padding class
//! (`pad_vocab_size_multiple = 1`), so every logit the model emits is a
//! character it can legitimately produce.
//!
//! # Download
//!
//! The dataset ships as one 673 MB parquet file, which is absurd for an example
//! this size, so instead of the [`HuggingfaceDatasetLoader`](burn_dataset) path
//! (python + `datasets` + a full sqlite import) the corpus is paged out of the
//! public [datasets-server] `/rows` endpoint, 100 stories per request — the
//! endpoint's hard maximum. The normalized text is cached under
//! `~/.cache/burn-dataset/tinystories-gpt4-clean/`, one file per
//! `(split, story count)`, so the download happens once.
//!
//! [datasets-server]: https://huggingface.co/docs/datasets-server
//!
//! That endpoint is rate limited (CloudFront answers `429` with an HTML body
//! once the budget — about 28 requests per two minutes — runs out), so the pager
//! paces itself with [`THROTTLE`] and retries a failed page with exponential
//! backoff.
//!
//! Splits follow the dataset card's suggested row ranges (the rows are
//! pre-shuffled, so a contiguous range is already a random sample): rows
//! `0..10k` are test, `10k..20k` validation, and `20k..` training.
//!
//! # Windows
//!
//! Stories are joined with `"\n\n"` — a sequence that never occurs *inside* a
//! story (single `\n` separates its paragraphs), so the blank line is an
//! unambiguous document boundary the model can learn — and the resulting token
//! stream is cut into non-overlapping windows of `seq_len + 1`. Each window is
//! one training item: the first `seq_len` tokens are the input, the last
//! `seq_len` (shifted by one) are the next-character targets.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn_dataset::{Dataset, DatasetError, network::downloader::download_file_as_bytes};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

// ===========================================================================
// Vocabulary
// ===========================================================================

/// Every character the corpus contains, after case-folding: 48 tokens, in ASCII
/// order. Token ids are indices into this string.
pub const ALPHABET: &str = "\n !\"$',-.0123456789:;?abcdefghijklmnopqrstuvwxyz";

/// Number of character classes (= the model's `vocab_size`).
pub const VOCAB_SIZE: usize = ALPHABET.len();

/// Token id reserved by [`Vocab`] for "not in the alphabet".
const NO_TOKEN: u8 = u8::MAX;

/// The document boundary: a blank line, which never occurs inside a story.
pub const STORY_SEPARATOR: &str = "\n\n";

/// Byte ↔ token-id tables for [`ALPHABET`], with `A-Z` folded onto `a-z`.
pub struct Vocab {
    /// `byte → token id`, [`NO_TOKEN`] for bytes outside the alphabet.
    to_id: [u8; 256],
    /// `token id → byte`.
    to_byte: [u8; VOCAB_SIZE],
}

/// The one vocabulary, built at compile time.
pub const VOCAB: Vocab = Vocab::new();

impl Vocab {
    /// Build the tables from [`ALPHABET`].
    pub const fn new() -> Self {
        let alphabet = ALPHABET.as_bytes();
        let mut to_id = [NO_TOKEN; 256];
        let mut to_byte = [0u8; VOCAB_SIZE];
        let mut i = 0;
        while i < alphabet.len() {
            let byte = alphabet[i];
            to_id[byte as usize] = i as u8;
            to_byte[i] = byte;
            // The upper-case half of the corpus folds onto the same token.
            if byte.is_ascii_lowercase() {
                to_id[byte.to_ascii_uppercase() as usize] = i as u8;
            }
            i += 1;
        }
        Self { to_id, to_byte }
    }

    /// Token id of `byte` (case-folded), or `None` when it is outside the
    /// alphabet.
    pub const fn token(&self, byte: u8) -> Option<u8> {
        match self.to_id[byte as usize] {
            NO_TOKEN => None,
            id => Some(id),
        }
    }

    /// The character a token id stands for.
    pub const fn character(&self, token: u8) -> char {
        self.to_byte[token as usize] as char
    }

    /// Encode `text`, silently dropping anything outside the alphabet (the
    /// cached corpus is normalized first, so this only bites on user prompts).
    pub fn encode(&self, text: &str) -> Vec<u8> {
        text.bytes().filter_map(|byte| self.token(byte)).collect()
    }

    /// Decode token ids back to text.
    pub fn decode(&self, tokens: &[u8]) -> String {
        tokens.iter().map(|&t| self.character(t)).collect()
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Download + cache
// ===========================================================================

/// The Hugging Face dataset id.
const DATASET: &str = "karpathy/tinystories-gpt4-clean";

/// The rows endpoint's hard maximum page size.
const PAGE: usize = 100;

/// Pause between two page requests. Measured budget: ~28 requests per ~2min,
/// i.e. a sustained page every ~4s, after which `429`s appear for ~15s at a
/// time. Pacing to the budget beats sawtoothing through it.
const THROTTLE: std::time::Duration = std::time::Duration::from_secs(4);

/// How many times one page is attempted before giving up.
const MAX_ATTEMPTS: usize = 6;

/// Backoff after the first failed attempt; doubles on each further one. Long
/// enough to sit out a rate-limit window rather than spend attempts inside it
/// (a retry during the window counts against the budget too).
const BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);

/// Which of the dataset card's suggested row ranges to read.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Split {
    /// Rows `20_000..` — the training stories.
    Train,
    /// Rows `10_000..20_000` — the validation stories.
    Valid,
    /// Rows `0..10_000` — the held-out test stories.
    Test,
}

impl Split {
    /// First row index of this split, per the dataset card.
    pub const fn offset(self) -> usize {
        match self {
            Split::Test => 0,
            Split::Valid => 10_000,
            Split::Train => 20_000,
        }
    }

    /// Number of rows the split has (unbounded for `Train`).
    pub const fn capacity(self) -> usize {
        match self {
            Split::Test | Split::Valid => 10_000,
            Split::Train => 2_712_634,
        }
    }

    /// Cache-file stem.
    pub const fn name(self) -> &'static str {
        match self {
            Split::Test => "test",
            Split::Valid => "valid",
            Split::Train => "train",
        }
    }
}

/// One `/rows` response (or the endpoint's error object).
#[derive(Deserialize)]
struct RowsResponse {
    #[serde(default)]
    rows: Vec<RowEntry>,
    #[serde(default)]
    error: Option<String>,
}

/// One row: `{"row_idx": .., "row": {"text": ..}, ..}`.
#[derive(Deserialize)]
struct RowEntry {
    row: Story,
}

/// The dataset's single column.
#[derive(Deserialize)]
struct Story {
    text: String,
}

/// Case-fold `story` and drop the (vanishingly rare, ~5 per million) characters
/// outside [`ALPHABET`] — after which the text is exactly the token stream.
fn normalize(story: &str) -> String {
    story
        .bytes()
        .filter(|&byte| VOCAB.token(byte).is_some())
        .map(|byte| byte.to_ascii_lowercase() as char)
        .collect()
}

/// `~/.cache/burn-dataset/tinystories-gpt4-clean/<split>-<n_stories>.txt`.
fn cache_path(split: Split, n_stories: usize) -> PathBuf {
    let dir = dirs::home_dir()
        .expect("Could not get home directory")
        .join(".cache")
        .join("burn-dataset")
        .join("tinystories-gpt4-clean");
    std::fs::create_dir_all(&dir).expect("Failed to create the cache directory");
    dir.join(format!("{}-{n_stories}.txt", split.name()))
}

/// Request one page, retrying with exponential backoff. The downloader hands
/// back whatever body it got without looking at the status code, so a rate-limit
/// (`429`, an HTML body) and a real error object both surface here as "not the
/// JSON we asked for" — and both are worth another try.
fn fetch_page(url: &str, message: &str, offset: usize) -> Vec<RowEntry> {
    let mut backoff = BACKOFF;
    let mut last = String::new();
    for attempt in 1..=MAX_ATTEMPTS {
        let bytes = download_file_as_bytes(url, message);
        match serde_json::from_slice::<RowsResponse>(&bytes) {
            Ok(response) if !response.rows.is_empty() => return response.rows,
            Ok(response) => {
                last = response
                    .error
                    .unwrap_or_else(|| "the response carried no rows".to_owned());
            }
            Err(e) => {
                let body = String::from_utf8_lossy(&bytes);
                let head: String = body.chars().take(120).collect();
                last = format!("unparseable JSON ({e}): {head}");
            }
        }
        if attempt < MAX_ATTEMPTS {
            println!("  retrying rows at offset {offset} in {backoff:?} ({last})");
            std::thread::sleep(backoff);
            backoff *= 2;
        }
    }
    panic!("datasets-server gave up at offset {offset} after {MAX_ATTEMPTS} attempts: {last}");
}

/// Page `n_stories` rows of `split` out of the datasets-server, normalize them,
/// and join them with [`STORY_SEPARATOR`].
fn download(split: Split, n_stories: usize) -> String {
    assert!(
        n_stories <= split.capacity(),
        "{n_stories} stories exceed the {} split ({} rows)",
        split.name(),
        split.capacity(),
    );
    let pages = n_stories.div_ceil(PAGE);
    println!(
        "downloading {n_stories} {} stories from {DATASET} ({pages} requests, cached afterwards)",
        split.name(),
    );

    let mut stories = Vec::with_capacity(n_stories);
    while stories.len() < n_stories {
        let offset = split.offset() + stories.len();
        let length = PAGE.min(n_stories - stories.len());
        let url = format!(
            "https://datasets-server.huggingface.co/rows\
             ?dataset={DATASET}&config=default&split=train&offset={offset}&length={length}"
        );
        let message = format!(
            "{} rows {offset}..{} ({}/{pages})",
            split.name(),
            offset + length,
            stories.len() / PAGE + 1,
        );
        // The progress bar hides itself when stdout is not a terminal, so say
        // where we are: a long download is otherwise minutes of silence.
        println!("  {message}");
        std::thread::sleep(THROTTLE);
        let rows = fetch_page(&url, &message, offset);
        stories.extend(rows.into_iter().map(|e| normalize(&e.row.text)));
    }
    stories.join(STORY_SEPARATOR)
}

/// The normalized text of `n_stories` stories from `split`, downloading it on
/// the first call and reading the cache afterwards.
pub fn text(split: Split, n_stories: usize) -> String {
    let path = cache_path(split, n_stories);
    if let Ok(cached) = std::fs::read_to_string(&path) {
        return cached;
    }
    let text = download(split, n_stories);
    std::fs::write(&path, &text).expect("Failed to write the corpus cache");
    println!("cached {} characters into {path:?}", text.len());
    text
}

// ===========================================================================
// Dataset + batcher
// ===========================================================================

/// One training window: `seq_len + 1` token ids (input and shifted target
/// overlap by `seq_len - 1`, so they share one buffer).
#[derive(Clone, Debug)]
pub struct TinyStoriesItem {
    /// Token ids, `[seq_len + 1]`.
    pub tokens: Vec<u8>,
}

/// A character stream cut into non-overlapping `seq_len + 1` windows.
pub struct TinyStoriesDataset {
    /// The whole split as token ids (shared, so cloning the dataset is free).
    tokens: Arc<Vec<u8>>,
    /// Window length; each item is `seq_len + 1` tokens.
    seq_len: usize,
}

impl TinyStoriesDataset {
    /// Load (downloading once) `n_stories` of `split` and window it.
    pub fn new(split: Split, n_stories: usize, seq_len: usize) -> Self {
        let tokens = VOCAB.encode(&text(split, n_stories));
        assert!(
            tokens.len() > seq_len,
            "the {} corpus ({} tokens) is shorter than one window",
            split.name(),
            tokens.len(),
        );
        Self {
            tokens: Arc::new(tokens),
            seq_len,
        }
    }

    /// Total number of characters in the split.
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }
}

impl Dataset<TinyStoriesItem> for TinyStoriesDataset {
    fn get(&self, index: usize) -> Result<TinyStoriesItem, DatasetError> {
        let start = index * self.seq_len;
        Ok(TinyStoriesItem {
            tokens: self.tokens[start..start + self.seq_len + 1].to_vec(),
        })
    }

    fn len(&self) -> usize {
        // The final window needs one extra token for its last target.
        (self.tokens.len() - 1) / self.seq_len
    }
}

/// A batch of next-character windows.
#[derive(Clone, Debug)]
pub struct TinyStoriesBatch {
    /// Input token ids, `[batch_size, seq_len]`.
    pub inputs: Tensor<2, Int>,
    /// Next-character targets (the inputs shifted by one), `[batch_size, seq_len]`.
    pub targets: Tensor<2, Int>,
}

/// Stacks [`TinyStoriesItem`]s into a [`TinyStoriesBatch`].
#[derive(Clone, Default)]
pub struct TinyStoriesBatcher {}

impl Batcher<TinyStoriesItem, TinyStoriesBatch> for TinyStoriesBatcher {
    fn batch(&self, items: Vec<TinyStoriesItem>, device: &Device) -> TinyStoriesBatch {
        let batch_size = items.len();
        let seq_len = items[0].tokens.len() - 1;
        let mut inputs = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);
        for item in &items {
            assert_eq!(item.tokens.len(), seq_len + 1);
            inputs.extend(item.tokens[..seq_len].iter().map(|&t| t as i32));
            targets.extend(item.tokens[1..].iter().map(|&t| t as i32));
        }
        let shape = [batch_size, seq_len];
        TinyStoriesBatch {
            inputs: Tensor::<1, Int>::from_ints(inputs.as_slice(), device).reshape(shape),
            targets: Tensor::<1, Int>::from_ints(targets.as_slice(), device).reshape(shape),
        }
    }
}
