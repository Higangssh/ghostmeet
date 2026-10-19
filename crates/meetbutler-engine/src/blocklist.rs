//! Hallucination blocklist.
//!
//! Decoded on its own, a span of about a second at the edge of speech (a click, a cough,
//! the tail of a sentence) makes Whisper invent a subtitle credit: "한글자막 by …",
//! "시청해주셔서 감사합니다." Its no-speech probability stays low, so the usual
//! no_speech/logprob filter does not catch them (`bench/README.md`). What does is the
//! combination: a short span *and* a known credit phrase. Either one alone is ordinary
//! speech — "감사합니다." on a short span is a real reply and is kept.
//!
//! Given the user's vocabulary as its prompt (STT-7), Whisper sometimes writes the prompt
//! back out on a span instead of what was said: "지호, 지호, 지호, 지호." Text made of
//! nothing but vocabulary terms, three or more of them, is that echo; a name said once or
//! twice, or a term inside a sentence, is speech.

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

/// Terms in a row it takes for text made only of vocabulary terms to be an echo: someone
/// may call a name twice ("지호, 지호."), and a real repeated reply ("네, 네, 네.") is not
/// in the vocabulary.
pub const ECHO_TERMS: usize = 3;

/// True when `text` is nothing but `ECHO_TERMS` or more vocabulary terms in a row.
pub fn is_vocabulary_echo(text: &str, vocabulary: &[String]) -> bool {
    let text: Vec<char> = normalize(text).chars().collect();
    let terms: Vec<Vec<char>> = vocabulary
        .iter()
        .map(|t| normalize(t).chars().collect::<Vec<_>>())
        .filter(|t| !t.is_empty())
        .collect();
    if text.is_empty() || terms.is_empty() {
        return false;
    }
    // most[i]: the most terms that make up text[..i] exactly, if any do
    let mut most: Vec<Option<usize>> = vec![None; text.len() + 1];
    most[0] = Some(0);
    for i in 0..text.len() {
        let Some(n) = most[i] else { continue };
        for t in &terms {
            if text[i..].starts_with(t) {
                let j = i + t.len();
                most[j] = Some(most[j].map_or(n + 1, |m| m.max(n + 1)));
            }
        }
    }
    most[text.len()].is_some_and(|n| n >= ECHO_TERMS)
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
