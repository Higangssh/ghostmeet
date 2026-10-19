//! Hallucination blocklist.
//!
//! Decoded on its own, a span of about a second at the edge of speech (a click, a cough,
//! the tail of a sentence) makes Whisper invent a subtitle credit: "한글자막 by …",
//! "시청해주셔서 감사합니다." Its no-speech probability stays low, so the usual
//! no_speech/logprob filter does not catch them (`bench/README.md`). What does is the
//! combination: a short span *and* a known credit phrase. Either one alone is ordinary
//! speech — "감사합니다." on a short span is a real reply and is kept.

/// Spans shorter than this (padding included) are where credits appear.
pub const SHORT_SPAN_SEC: f32 = 2.0;

/// Credit phrases, compared with spaces and punctuation removed and case folded. Each is
/// specific enough that a meeting reply does not contain it.
pub const PHRASES: &[&str] = &[
    "한글자막",
    "자막제공",
    "시청해주셔서감사합니다",
    "구독과좋아요",
    "뉴스스토리였습니다",
    "부에서계속됩니다",
    "thankyouforwatching",
    "thanksforwatching",
];

/// True when `text`, decoded from a span `span_sec` long, is a credit to drop.
pub fn is_hallucination(text: &str, span_sec: f32) -> bool {
    if span_sec >= SHORT_SPAN_SEC {
        return false;
    }
    let key = normalize(text);
    PHRASES.iter().any(|p| key.contains(p))
}

/// Text with nothing but punctuation and spaces in it.
pub fn is_blank(text: &str) -> bool {
    !text.chars().any(char::is_alphanumeric)
}

fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}
