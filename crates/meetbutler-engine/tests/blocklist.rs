//! The hallucination blocklist: a short span whose text is a known subtitle credit is
//! dropped, and nothing else is — above all not a real short reply.
//!
//! The phrases are the ones Whisper produced on the benchmark's short spans (FLEURS and
//! synthetic replies, `bench/README.md`); none comes from a meeting.

use meetbutler_engine::blocklist::{is_blank, is_hallucination, SHORT_SPAN_SEC};

const SHORT: f32 = 1.2;

#[test]
fn subtitle_credits_on_a_short_span_are_dropped() {
    for text in [
        "한글자막 by 한효정",
        "시청해주셔서 감사합니다.",
        "시청해 주셔서 감사합니다!",
        "지금까지 뉴스 스토리였습니다.",
        "2부에서 계속 됩니다.",
        "한글자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.",
        "구독과 좋아요 부탁드립니다",
        "Thank you for watching!",
        "Thanks for watching.",
    ] {
        assert!(is_hallucination(text, SHORT), "{text}");
    }
}

#[test]
fn real_short_replies_are_kept() {
    for text in [
        "네.",
        "네",
        "아니요.",
        "맞아요.",
        "알겠습니다.",
        "좋습니다.",
        "잠깐만요.",
        "글쎄요.",
        "그렇죠",
        "감사합니다.",
        "감사합니다",
        "고맙습니다.",
        "Thank you.",
        "Yes.",
        "Okay.",
    ] {
        assert!(!is_hallucination(text, SHORT), "{text}");
    }
}

#[test]
fn a_credit_phrase_in_a_long_span_is_kept() {
    // someone can say it out loud; only the short spans at clip edges invent it
    assert!(!is_hallucination(
        "시청해주셔서 감사합니다.",
        SHORT_SPAN_SEC
    ));
    assert!(!is_hallucination("시청해주셔서 감사합니다.", 4.0));
}

#[test]
fn the_cut_off_is_two_seconds() {
    assert_eq!(SHORT_SPAN_SEC, 2.0);
    assert!(is_hallucination("시청해주셔서 감사합니다.", 1.99));
}

#[test]
fn punctuation_only_text_is_blank() {
    assert!(is_blank("."));
    assert!(is_blank("  ... !"));
    assert!(is_blank(""));
    assert!(!is_blank("네."));
    assert!(!is_blank("3."));
}
