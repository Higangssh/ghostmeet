//! The user's vocabulary (STT-7): attendee names and terms, given to whisper as its
//! initial prompt so it spells them the way the meeting does.
//!
//! The prompt only biases the decoder; it is not a word list whisper can match against.
//! A term in it can surface in a span where nobody said it, so keep it to what the user
//! wrote, and with no vocabulary the decoder runs exactly as before.

/// The initial prompt for these terms, or `None` when there is nothing to give.
///
/// Terms are trimmed, empty ones dropped and repeats kept once, in the user's order.
/// whisper keeps only the end of a prompt longer than its budget, so the terms past
/// `MAX_PROMPT_CHARS` are dropped here instead, keeping the ones the user put first.
pub fn prompt(terms: &[String]) -> Option<String> {
    let kept = kept(terms);
    if kept.is_empty() {
        None
    } else {
        Some(format!("{}.", kept.join(", ")))
    }
}

/// The terms `prompt` gives whisper, in order.
pub fn kept(terms: &[String]) -> Vec<String> {
    let mut kept: Vec<&str> = Vec::new();
    let mut len = 0;
    for term in terms.iter().map(|t| t.trim()) {
        if term.is_empty() || kept.contains(&term) {
            continue;
        }
        let add = term.chars().count() + if kept.is_empty() { 0 } else { 2 };
        if len + add > MAX_PROMPT_CHARS {
            break;
        }
        len += add;
        kept.push(term);
    }
    kept.into_iter().map(String::from).collect()
}

/// whisper conditions on at most half its text context (224 tokens); Hangul takes one to
/// three tokens a syllable, so this many characters stay inside it.
pub const MAX_PROMPT_CHARS: usize = 100;

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_terms_give_no_prompt() {
        assert_eq!(prompt(&[]), None);
        assert_eq!(prompt(&terms(&["", "  "])), None);
    }

    #[test]
    fn terms_are_trimmed_and_kept_once_in_order() {
        assert_eq!(
            prompt(&terms(&[
                " 주소록 동기화",
                "하늘",
                "주소록 동기화 ",
                "누리온"
            ]))
            .as_deref(),
            Some("주소록 동기화, 하늘, 누리온.")
        );
    }

    #[test]
    fn a_long_vocabulary_keeps_the_first_terms_within_the_budget() {
        let many: Vec<String> = (0..100).map(|i| format!("용어{i:02}")).collect();
        let p = prompt(&many).unwrap();
        assert!(p.chars().count() <= MAX_PROMPT_CHARS + 1, "{p}");
        assert!(p.starts_with("용어00, 용어01"), "{p}");
        assert!(!p.contains("용어99"), "{p}");
    }
}
