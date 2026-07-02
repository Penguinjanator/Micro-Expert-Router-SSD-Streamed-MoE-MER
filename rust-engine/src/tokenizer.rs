//! Tokenizer abstraction (Phase 4).
//!
//! When the `tokenizer` feature is enabled, this module loads a real
//! HuggingFace tokenizer (`tokenizer.json`) via the [`tokenizers`] crate.
//! When disabled (the default), it falls back to a deterministic
//! byte-level tokenizer that maps every input byte to its u8 value as a
//! token id (vocab_size = 256). The fallback exists so the rest of the
//! server (HTTP API, request scheduling, generation loop) can be built
//! and tested without pulling in a heavy native-code dep.
//!
//! Both implementations expose the same minimal interface used by the
//! generation loop:
//! - [`Tokenizer::encode`]
//! - [`Tokenizer::decode`]
//! - [`Tokenizer::vocab_size`]

use std::path::Path;

/// Errors a tokenizer can produce.
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Backend(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "tokenizer io error: {e}"),
            TokenizerError::Backend(m) => write!(f, "tokenizer backend error: {m}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

/// Type-erased tokenizer the server uses.
pub enum Tokenizer {
    /// Deterministic byte-level fallback. Vocab is 0..=255; every byte of
    /// the input is one token. Decode is just `String::from_utf8_lossy`.
    Bytes,
    #[cfg(feature = "tokenizer")]
    Hf(tokenizers::Tokenizer),
}

impl Tokenizer {
    /// Always succeeds. Use when no `tokenizer.json` is available.
    pub fn bytes() -> Self {
        Tokenizer::Bytes
    }

    /// Try to load a HuggingFace `tokenizer.json` from disk. Falls back
    /// to the byte tokenizer when the `tokenizer` feature is disabled
    /// or the file isn't there.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        #[cfg(feature = "tokenizer")]
        {
            let inner = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| TokenizerError::Backend(e.to_string()))?;
            return Ok(Tokenizer::Hf(inner));
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            // Behave deterministically when the user asks for a tokenizer
            // file but the backend is not compiled in: surface the
            // missing-feature condition rather than silently downgrading.
            let _ = path;
            Err(TokenizerError::Backend(
                "tokenizer feature is disabled at compile time; rebuild with \
                 `--features tokenizer` to load tokenizer.json".to_string(),
            ))
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Bytes => 256,
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => t.get_vocab_size(true),
        }
    }

    /// Largest token id this tokenizer can emit, including added and
    /// special tokens. For the byte fallback this is always 255. For a
    /// HuggingFace tokenizer it is the maximum id over the full vocabulary
    /// (`with_added_tokens = true`), which covers reserved/special ids that
    /// may sit above the base-vocabulary count.
    pub fn max_token_id(&self) -> u32 {
        match self {
            Tokenizer::Bytes => 255,
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => t.get_vocab(true).values().copied().max().unwrap_or(0),
        }
    }

    /// Validate that every token id this tokenizer can emit is addressable
    /// by a model whose output/embedding vocabulary is `model_vocab_size`.
    ///
    /// The invariant is `max_token_id < model_vocab_size` (ids are
    /// zero-based). This deliberately checks the maximum *emittable* id —
    /// including added and special tokens — rather than requiring the raw
    /// base-vocabulary count to equal `model_vocab_size`, because real
    /// checkpoints routinely pad the embedding table beyond the tokenizer's
    /// base vocab and reserve high ids for special tokens.
    pub fn validate_vocab_compat(&self, model_vocab_size: usize) -> Result<(), TokenizerError> {
        let max_id = self.max_token_id() as usize;
        if max_id >= model_vocab_size {
            return Err(TokenizerError::Backend(format!(
                "tokenizer can emit token id {max_id} but model vocab_size is \
                 {model_vocab_size}; every token id must be < model vocab_size"
            )));
        }
        Ok(())
    }

    pub fn encode(&self, input: &str) -> Result<Vec<u32>, TokenizerError> {
        match self {
            Tokenizer::Bytes => Ok(input.bytes().map(|b| b as u32).collect()),
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => {
                let enc = t
                    .encode(input, false)
                    .map_err(|e| TokenizerError::Backend(e.to_string()))?;
                Ok(enc.get_ids().to_vec())
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        match self {
            Tokenizer::Bytes => {
                let bytes: Vec<u8> = ids
                    .iter()
                    .map(|&id| (id & 0xFF) as u8)
                    .collect();
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => t
                .decode(ids, true)
                .map_err(|e| TokenizerError::Backend(e.to_string())),
        }
    }
}

/// Incremental streaming decoder (hardening pass, F2).
///
/// The streaming path previously re-decoded the *entire* cumulative
/// completion after every token and diffed against a cloned cumulative
/// string — `O(tokens²)` total decode work plus one full-string clone
/// per token. `StreamDecoder` instead keeps a **bounded look-behind
/// window** of recent token ids, just large enough for UTF-8 /
/// byte-fallback / BPE boundary correctness:
///
/// * Each pushed token decodes only the window (a handful of ids), not
///   the whole completion.
/// * A trailing run of U+FFFD replacement characters — the signature
///   of a UTF-8 sequence still split across byte-fallback tokens — is
///   **held back** until a later token completes (or disproves) the
///   sequence, so multi-byte characters are never emitted torn.
/// * Once the window's decode has been fully emitted, the window is
///   trimmed to its final token (kept as decoder context for
///   BPE/metaspace joining) and the bookkeeping prefix is re-derived,
///   so the window never grows with the stream.
/// * A hard cap ([`Self::MAX_LOOKBEHIND_TOKENS`]) force-flushes
///   pathological streams that never resolve (e.g. an endless run of
///   invalid bytes), bounding both memory and per-token decode cost.
///
/// [`Self::finish`] flushes any held-back text at end of stream.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    /// Bounded look-behind window of the most recent token ids.
    ids: Vec<u32>,
    /// Prefix of `decode(&ids)` that has already been emitted.
    emitted: String,
}

impl StreamDecoder {
    /// Hard cap on the look-behind window. A well-formed UTF-8
    /// scalar spans at most 4 bytes (≤ 4 byte-fallback tokens), so 64
    /// is generous margin for merged BPE pieces while keeping the
    /// worst-case per-token decode cost small and constant.
    pub const MAX_LOOKBEHIND_TOKENS: usize = 64;

    pub fn new() -> Self {
        Self::default()
    }

    /// Number of ids currently held in the look-behind window
    /// (bounded by [`Self::MAX_LOOKBEHIND_TOKENS`]).
    pub fn lookbehind_len(&self) -> usize {
        self.ids.len()
    }

    /// Feed one generated token id; returns the newly stable decoded
    /// text (possibly empty while a multi-byte sequence is pending).
    ///
    /// `emitted` bookkeeping is **monotonic**: text that has been
    /// delivered is never re-emitted and never retracted. Byte-fallback
    /// decoders transiently *shrink* the window decode (adding one byte
    /// to a pending run re-invalidates previously valid characters
    /// until the run completes); those shrinks are recognised as
    /// prefixes of the delivered text and produce empty deltas instead
    /// of regressions.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<String, TokenizerError> {
        self.ids.push(id);
        let s = tokenizer.decode(&self.ids)?;
        // Hold back a trailing replacement-character run: it may be an
        // incomplete UTF-8 sequence that the next byte-fallback token
        // completes. Force-flush at the window cap so an endless run
        // of genuinely invalid bytes cannot grow the window forever.
        let force = self.ids.len() >= Self::MAX_LOOKBEHIND_TOKENS;
        let hold = if force {
            0
        } else {
            trailing_replacement_len(&s)
        };
        let safe_end = s.len() - hold;
        let delta = if safe_end >= self.emitted.len() && s.starts_with(self.emitted.as_str()) {
            // Extension: emit the newly stable suffix.
            let d = s[self.emitted.len()..safe_end].to_string();
            self.emitted = s[..safe_end].to_string();
            d
        } else if self.emitted.starts_with(&s[..safe_end]) {
            // Transient shrink (byte-fallback run re-invalidated, or a
            // skipped special token): everything stable was already
            // delivered — emit nothing and keep the delivered prefix.
            String::new()
        } else {
            // The decoder revised earlier characters (rare; possible
            // only with exotic non-prefix-stable decoders): text that
            // has already been delivered can never be retracted, so
            // emit only the revised window text *after* the longest
            // stable prefix and resynchronise. The serving decoders
            // (byte-level BPE, metaspace, byte-fallback) never reach
            // this branch because uncertain trailing text is held back
            // as U+FFFD runs above; for exotic decoders this fallback
            // guarantees no text is lost or stalled (exact
            // stream/one-shot equality is not achievable without
            // retracting delivered text).
            let lcp = common_prefix_boundary(&s, &self.emitted).min(safe_end);
            let d = s[lcp..safe_end].to_string();
            self.emitted = s[..safe_end].to_string();
            d
        };
        if force {
            // Pathological stream that never resolved: everything has
            // been flushed; drop the window entirely so it cannot grow
            // past the cap (joining context is lost, which is the
            // documented force-flush divergence).
            self.ids.clear();
            self.emitted.clear();
        } else if hold == 0 && self.ids.len() > 1 {
            // Fully emitted: trim the window to its last token (kept
            // as decode context for BPE joining) and re-derive the
            // emitted prefix so the invariant "emitted is delivered
            // text for this window" holds for the trimmed window too.
            // A token that does not decode cleanly on its own (e.g. a
            // byte-fallback fragment like `<0xAC>` that only completed
            // a character in context) is *not* a usable context anchor
            // — re-deriving the prefix from it would misalign later
            // byte runs — so the window is kept whole until a
            // cleanly-decoding token arrives (the cap still bounds it).
            let last = *self.ids.last().expect("just pushed");
            let last_alone = tokenizer.decode(std::slice::from_ref(&last))?;
            if !last_alone.contains(char::REPLACEMENT_CHARACTER) {
                self.ids.clear();
                self.ids.push(last);
                self.emitted = tokenizer.decode(&self.ids)?;
            }
        }
        Ok(delta)
    }

    /// Flush any held-back (possibly incomplete) text at end of
    /// stream and reset the decoder.
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String, TokenizerError> {
        if self.ids.is_empty() {
            self.emitted.clear();
            return Ok(String::new());
        }
        let s = tokenizer.decode(&self.ids)?;
        let delta = if s.len() >= self.emitted.len() && s.starts_with(self.emitted.as_str()) {
            s[self.emitted.len()..].to_string()
        } else if self.emitted.starts_with(s.as_str()) {
            // The final window decode is a prefix of the delivered
            // text (transiently re-invalidated byte run at EOS whose
            // stable part was already emitted): nothing left to flush.
            String::new()
        } else {
            let lcp = common_prefix_boundary(&s, &self.emitted);
            s[lcp..].to_string()
        };
        self.ids.clear();
        self.emitted.clear();
        Ok(delta)
    }
}

/// Byte length of the trailing run of U+FFFD replacement characters in
/// `s` (each is 3 bytes in UTF-8). Used to hold back potentially
/// incomplete UTF-8 sequences during incremental decoding.
fn trailing_replacement_len(s: &str) -> usize {
    let mut hold = 0usize;
    for c in s.chars().rev() {
        if c == char::REPLACEMENT_CHARACTER {
            hold += c.len_utf8();
        } else {
            break;
        }
    }
    hold
}

/// Byte length of the longest common prefix of `a` and `b`, backed off
/// to a character boundary of `a`. Used by [`StreamDecoder::push`]'s
/// resynchronisation fallback so a revising decoder never re-emits the
/// stable prefix that was already delivered.
fn common_prefix_boundary(a: &str, b: &str) -> usize {
    let mut n = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count();
    while n > 0 && !a.is_char_boundary(n) {
        n -= 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trips_ascii() {
        let t = Tokenizer::bytes();
        let ids = t.encode("hello").unwrap();
        assert_eq!(ids, vec![104, 101, 108, 108, 111]);
        let s = t.decode(&ids).unwrap();
        assert_eq!(s, "hello");
        assert_eq!(t.vocab_size(), 256);
    }

    #[test]
    fn byte_tokenizer_handles_utf8_lossily() {
        let t = Tokenizer::bytes();
        let ids = t.encode("héllo").unwrap();
        // "h" + 2 bytes for é + "llo"
        assert_eq!(ids.len(), 6);
        // Round-trip succeeds (the two bytes for é form a valid UTF-8 sequence).
        let s = t.decode(&ids).unwrap();
        assert_eq!(s, "héllo");
    }

    #[cfg(not(feature = "tokenizer"))]
    #[test]
    fn missing_tokenizer_feature_returns_error_for_file_load() {
        let path = std::path::PathBuf::from("/nonexistent/tokenizer.json");
        match Tokenizer::from_file(&path) {
            Err(TokenizerError::Backend(msg)) => assert!(msg.contains("tokenizer feature")),
            other => panic!("expected Backend error about disabled feature, got {}",
                match other { Ok(_) => "Ok(_)".to_string(), Err(e) => format!("Err({e})") }),
        }
    }

    #[test]
    fn byte_tokenizer_max_token_id_is_255() {
        assert_eq!(Tokenizer::bytes().max_token_id(), 255);
    }

    #[test]
    fn vocab_compat_accepts_model_larger_than_max_token_id() {
        // Byte tokenizer emits ids 0..=255; a model with vocab_size 256
        // addresses all of them.
        assert!(Tokenizer::bytes().validate_vocab_compat(256).is_ok());
        assert!(Tokenizer::bytes().validate_vocab_compat(100_000).is_ok());
    }

    #[test]
    fn vocab_compat_rejects_max_token_id_at_or_above_vocab_size() {
        // vocab_size == max_id fails (ids are zero-based, so 255 needs
        // vocab_size >= 256); anything smaller also fails.
        let err = Tokenizer::bytes().validate_vocab_compat(255).unwrap_err();
        match err {
            TokenizerError::Backend(m) => assert!(m.contains("255") && m.contains("vocab_size")),
            other => panic!("expected Backend error, got {other}"),
        }
        assert!(Tokenizer::bytes().validate_vocab_compat(10).is_err());
    }

    /// F2: multi-byte UTF-8 characters split across byte-fallback
    /// tokens are held back until complete and then emitted whole —
    /// never as torn replacement characters.
    #[test]
    fn stream_decoder_holds_back_split_multibyte_utf8() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        // "é" = 0xC3 0xA9; "€" = 0xE2 0x82 0xAC.
        assert_eq!(d.push(&t, 0xC3).unwrap(), "");
        assert_eq!(d.push(&t, 0xA9).unwrap(), "é");
        assert_eq!(d.push(&t, b'x' as u32).unwrap(), "x");
        assert_eq!(d.push(&t, 0xE2).unwrap(), "");
        assert_eq!(d.push(&t, 0x82).unwrap(), "");
        assert_eq!(d.push(&t, 0xAC).unwrap(), "€");
        assert_eq!(d.finish(&t).unwrap(), "");
    }

    /// F2: a genuinely invalid intermediate byte sequence is
    /// eventually emitted as replacement characters once later valid
    /// text resolves it — the stream neither stalls nor drops text.
    #[test]
    fn stream_decoder_resolves_invalid_intermediate_bytes() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&t, b'a' as u32).unwrap(), "a");
        // Stray continuation byte: held back (could be a prefix of a
        // longer sequence from the decoder's perspective).
        assert_eq!(d.push(&t, 0xA9).unwrap(), "");
        // A following ASCII byte proves it invalid; both are emitted.
        let out = d.push(&t, b'b' as u32).unwrap();
        assert_eq!(out, "\u{FFFD}b");
        assert_eq!(d.finish(&t).unwrap(), "");
    }

    /// F2: a trailing incomplete sequence at end of stream is flushed
    /// by `finish` (as a replacement character) rather than dropped.
    #[test]
    fn stream_decoder_finish_flushes_trailing_incomplete_sequence() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&t, b'h' as u32).unwrap(), "h");
        assert_eq!(d.push(&t, 0xE2).unwrap(), "");
        assert_eq!(d.push(&t, 0x82).unwrap(), "");
        assert_eq!(d.finish(&t).unwrap(), "\u{FFFD}");
        // Decoder is reusable after finish.
        assert_eq!(d.push(&t, b'i' as u32).unwrap(), "i");
    }

    /// F2: long streams — the concatenated deltas equal the one-shot
    /// decode of the full id sequence, and the look-behind window
    /// stays bounded (no O(tokens²) re-decode, no unbounded state).
    #[test]
    fn stream_decoder_long_stream_matches_full_decode_with_bounded_window() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        // Mixed ASCII + multi-byte content, repeated well past any
        // window size.
        let text = "héllo wörld €42 ✓ ".repeat(200);
        let ids = t.encode(&text).unwrap();
        assert!(ids.len() > 4 * StreamDecoder::MAX_LOOKBEHIND_TOKENS);
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&d.push(&t, id).unwrap());
            assert!(
                d.lookbehind_len() <= StreamDecoder::MAX_LOOKBEHIND_TOKENS,
                "look-behind window must stay bounded"
            );
        }
        out.push_str(&d.finish(&t).unwrap());
        assert_eq!(out, t.decode(&ids).unwrap());
        assert_eq!(out, text);
    }

    /// F2: an adversarial endless run of invalid bytes cannot grow the
    /// window past the hard cap; the force-flush emits the pending
    /// replacement characters and keeps streaming.
    #[test]
    fn stream_decoder_force_flushes_endless_invalid_run_at_cap() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        let mut emitted = String::new();
        for _ in 0..(3 * StreamDecoder::MAX_LOOKBEHIND_TOKENS) {
            emitted.push_str(&d.push(&t, 0xC3).unwrap());
            assert!(d.lookbehind_len() <= StreamDecoder::MAX_LOOKBEHIND_TOKENS);
        }
        emitted.push_str(&d.finish(&t).unwrap());
        assert!(
            emitted.contains('\u{FFFD}'),
            "invalid bytes must eventually surface as replacement characters"
        );
    }

    /// F2 (merged BPE pieces): with the HF tokenizer backend, pieces
    /// that decode differently in context than in isolation still
    /// stream correctly — the concatenated deltas equal the one-shot
    /// decode of the full sequence.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_for_hf_bpe() {
        // Minimal in-memory BPE tokenizer with a metaspace-style
        // decoder: "▁" marks word boundaries and decodes to a space in
        // context but is stripped at sequence start.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": true},
            "post_processor": null,
            "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {"\u2581": 0, "h": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7,
                           "he": 8, "ll": 9, "hell": 10, "hello": 11, "\u2581w": 12, "or": 13,
                           "orl": 14, "\u2581world": 15, "\u2581hello": 16, "\u2581worl": 17},
                "merges": ["h e", "l l", "he ll", "hell o", "\u2581 w", "o r", "or l",
                            "\u2581w orl", "\u2581worl d", "\u2581 hello"]
            }
        }"#;
        let inner = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("valid tokenizer");
        let t = Tokenizer::Hf(inner);
        let ids = t.encode("hello world").unwrap();
        assert!(ids.len() >= 2, "expected merged BPE pieces");
        let mut d = StreamDecoder::new();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&d.push(&t, id).unwrap());
        }
        out.push_str(&d.finish(&t).unwrap());
        assert_eq!(out, t.decode(&ids).unwrap());
    }

    // ===== Incremental-decoder equality matrix (validation closure, item 2) =====
    //
    // Contract asserted by every case:
    //
    //   concatenate(all streamed text deltas) == tokenizer.decode(all ids)
    //
    // while the look-behind window stays bounded.

    /// Stream every id through a fresh [`StreamDecoder`] and return the
    /// concatenated deltas (including the final `finish` flush),
    /// asserting the bounded-window invariant on every step.
    fn stream_all(t: &Tokenizer, ids: &[u32]) -> String {
        let mut d = StreamDecoder::new();
        let mut out = String::new();
        for &id in ids {
            out.push_str(&d.push(t, id).unwrap());
            assert!(
                d.lookbehind_len() <= StreamDecoder::MAX_LOOKBEHIND_TOKENS,
                "look-behind window must stay bounded"
            );
        }
        out.push_str(&d.finish(t).unwrap());
        out
    }

    fn assert_stream_equals_full_decode(t: &Tokenizer, ids: &[u32]) {
        let streamed = stream_all(t, ids);
        let full = t.decode(ids).unwrap();
        assert_eq!(
            streamed, full,
            "concatenated streamed deltas must equal the one-shot decode"
        );
    }

    /// Item 2: a 10,000-token stream (mixed ASCII / multi-byte /
    /// stray-byte content) matches the one-shot decode exactly.
    #[test]
    fn stream_decoder_ten_thousand_token_stream_matches_full_decode() {
        let t = Tokenizer::bytes();
        let mut ids: Vec<u32> = Vec::with_capacity(10_000);
        let pattern = t.encode("héllo wörld €42 ✓ plain ascii ").unwrap();
        while ids.len() < 10_000 {
            ids.extend_from_slice(&pattern);
        }
        ids.truncate(10_000);
        // Keep the truncation from ending mid-character irrelevant:
        // equality must hold regardless (finish flushes the tail).
        assert_eq!(ids.len(), 10_000);
        assert_stream_equals_full_decode(&t, &ids);
    }

    /// Item 2: a stream ending mid-UTF-8-sequence has a non-empty
    /// held-back final tail; including the `finish` flush, streamed
    /// text still equals the one-shot decode.
    #[test]
    fn stream_decoder_incomplete_final_sequence_equality_including_tail() {
        let t = Tokenizer::bytes();
        // "hi" + first two bytes of '€' (0xE2 0x82 0xAC).
        let ids = vec![b'h' as u32, b'i' as u32, 0xE2, 0x82];
        let mut d = StreamDecoder::new();
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&d.push(&t, id).unwrap());
        }
        let tail = d.finish(&t).unwrap();
        assert!(!tail.is_empty(), "the held-back tail must be non-empty");
        streamed.push_str(&tail);
        assert_eq!(streamed, t.decode(&ids).unwrap());
    }

    /// Build an in-memory HF tokenizer from a `tokenizer.json` string.
    #[cfg(feature = "tokenizer")]
    fn hf(json: &str) -> Tokenizer {
        Tokenizer::Hf(tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("valid tokenizer"))
    }

    /// Look up a token id by its vocabulary string.
    #[cfg(feature = "tokenizer")]
    fn tid(t: &Tokenizer, piece: &str) -> u32 {
        match t {
            Tokenizer::Hf(inner) => inner
                .get_vocab(true)
                .get(piece)
                .copied()
                .unwrap_or_else(|| panic!("token {piece:?} not in vocab")),
            _ => unreachable!(),
        }
    }

    /// Llama-style byte-fallback BPE `tokenizer.json`: `<0xNN>` byte
    /// tokens plus a `ByteFallback` + `Metaspace` decoder sequence.
    #[cfg(feature = "tokenizer")]
    fn byte_fallback_tokenizer_json() -> String {
        r##"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 10, "content": "</s>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 11, "content": "<tool>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": false}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": true},
            "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "ByteFallback"},
                {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": true}
            ]},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": {"\u2581": 0, "h": 1, "i": 2, "\u2581hi": 3,
                           "<0xE2>": 4, "<0x82>": 5, "<0xAC>": 6,
                           "<0xC3>": 7, "<0xA9>": 8, "x": 9, "hi": 12},
                "merges": ["\u2581 hi", "h i"]
            }
        }"##
        .to_string()
    }

    /// Item 2 (byte-fallback tokens): multi-byte characters split
    /// across `<0xNN>` byte-fallback tokens stream without tearing and
    /// match the one-shot decode. This is also the canonical
    /// "decoding revises recent text" case — the raw window decode
    /// goes "\u{FFFD}" → "\u{FFFD}\u{FFFD}" → "€" as bytes arrive, and
    /// equality holds because uncertain text is held back rather than
    /// emitted and retracted.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_for_byte_fallback_tokens() {
        let t = hf(&byte_fallback_tokenizer_json());
        // "hi" + '€' as three byte-fallback tokens + 'é' as two.
        let ids = vec![
            tid(&t, "\u{2581}hi"),
            tid(&t, "<0xE2>"),
            tid(&t, "<0x82>"),
            tid(&t, "<0xAC>"),
            tid(&t, "<0xC3>"),
            tid(&t, "<0xA9>"),
        ];
        // Sanity: the one-shot decode resolves the byte runs.
        let full = t.decode(&ids).unwrap();
        assert!(full.contains('€') && full.contains('é'), "full decode: {full:?}");
        // The revision is invisible to the client: no delta ever
        // carries a torn replacement character for these byte runs.
        let mut d = StreamDecoder::new();
        let mut streamed = String::new();
        for &id in &ids {
            let delta = d.push(&t, id).unwrap();
            assert!(
                !delta.contains(char::REPLACEMENT_CHARACTER),
                "byte-fallback runs must be held back until complete; got {delta:?}"
            );
            streamed.push_str(&delta);
        }
        streamed.push_str(&d.finish(&t).unwrap());
        assert_eq!(streamed, full);
    }

    /// Item 2 (added + special tokens): special tokens are skipped by
    /// decode and added tokens surface verbatim; streamed deltas match
    /// the one-shot decode either way.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_with_added_and_special_tokens() {
        let t = hf(&byte_fallback_tokenizer_json());
        let ids = vec![
            tid(&t, "\u{2581}hi"),
            tid(&t, "<tool>"), // added (non-special): surfaces in decode
            tid(&t, "h"),
            tid(&t, "i"),
            tid(&t, "</s>"), // special: skipped by decode
        ];
        assert_stream_equals_full_decode(&t, &ids);
        let full = t.decode(&ids).unwrap();
        assert!(full.contains("<tool>"), "added token must surface: {full:?}");
        assert!(!full.contains("</s>"), "special token must be skipped: {full:?}");
    }

    /// Item 2 (byte-level BPE cleanup): a GPT-2-style ByteLevel
    /// pre-tokenizer/decoder pair ("Ġ" space marker, byte-to-char
    /// remapping) streams to exactly the one-shot decode.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_for_byte_level_bpe() {
        let json = r##"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6,
                           "he": 7, "ll": 8, "hell": 9, "hello": 10,
                           "\u0120": 11, "\u0120w": 12, "or": 13, "orl": 14,
                           "\u0120worl": 15, "\u0120world": 16},
                "merges": ["h e", "l l", "he ll", "hell o", "\u0120 w", "o r", "or l",
                            "\u0120w orl", "\u0120worl d"]
            }
        }"##;
        let t = hf(json);
        let ids = t.encode("hello world").unwrap();
        assert!(ids.len() >= 2, "expected merged byte-level BPE pieces");
        assert_stream_equals_full_decode(&t, &ids);
        assert_eq!(t.decode(&ids).unwrap(), "hello world");
    }

    /// Item 2 (metaspace / SentencePiece-like joining): word-boundary
    /// markers decode to spaces in context; streamed deltas match the
    /// one-shot decode. (Complements the merged-BPE variant above.)
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_for_metaspace_joining() {
        let t = hf(&byte_fallback_tokenizer_json());
        // "hi hi" via explicit metaspace-marked pieces.
        let ids = vec![
            tid(&t, "\u{2581}hi"),
            tid(&t, "\u{2581}"),
            tid(&t, "h"),
            tid(&t, "i"),
        ];
        assert_stream_equals_full_decode(&t, &ids);
    }

    /// Item 2 (real `tokenizer.json` from disk): the same matrix holds
    /// for a tokenizer loaded through the production
    /// [`Tokenizer::from_file`] path.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_for_tokenizer_json_file() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mer-tokenizer-test-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, byte_fallback_tokenizer_json()).unwrap();
        let t = Tokenizer::from_file(&path).expect("load tokenizer.json from disk");
        let _ = std::fs::remove_file(&path);
        let ids = vec![
            tid(&t, "\u{2581}hi"),
            tid(&t, "<0xC3>"),
            tid(&t, "<0xA9>"),
            tid(&t, "h"),
            tid(&t, "i"),
        ];
        assert_stream_equals_full_decode(&t, &ids);
        // 10k-token soak on the real-backend path too.
        let mut long: Vec<u32> = Vec::with_capacity(10_000);
        while long.len() < 10_000 {
            long.extend_from_slice(&ids);
        }
        long.truncate(10_000);
        assert_stream_equals_full_decode(&t, &long);
    }

    /// The resynchronisation fallback for non-prefix-stable decoders:
    /// when a revision does occur, the stable common prefix already
    /// delivered is never re-emitted (no duplication) and no text is
    /// lost or stalled.
    #[test]
    fn common_prefix_boundary_respects_char_boundaries() {
        assert_eq!(common_prefix_boundary("hello", "help"), 3);
        assert_eq!(common_prefix_boundary("abc", "abc"), 3);
        assert_eq!(common_prefix_boundary("abc", "xyz"), 0);
        // "é" is 2 bytes; a partial byte match must back off to the
        // char boundary.
        assert_eq!(common_prefix_boundary("é", "\u{c3}\u{28}"), 0);
        assert_eq!(common_prefix_boundary("aé", "aè"), 1);
    }
}
