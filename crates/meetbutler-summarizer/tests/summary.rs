//! The meeting summary the app asks the sidecar for, without a model: a scripted
//! generator stands in for llama.cpp and records what it was asked.

use std::sync::atomic::{AtomicBool, Ordering};

use meetbutler_summarizer::client::ClientError;
use meetbutler_summarizer::protocol::{ErrorKind, Event, GenerateRequest, Phase, Stats};
use meetbutler_summarizer::summary::{
    chatml, chunks, n_ctx_for, parse, summarise, Generate, Options, Summary, CHUNK_SYLLABLES,
    GRAMMAR, INSTRUCTIONS,
};

/// Answers each chunk with the next scripted reply and the overall summary with "전체".
struct Scripted {
    replies: Vec<&'static str>,
    asked: Vec<GenerateRequest>,
    /// Refuse the first request this many tokens long, as the sidecar does when a prompt
    /// does not fit.
    too_small: Option<u32>,
}

impl Scripted {
    fn new(replies: &[&'static str]) -> Self {
        Self {
            replies: replies.to_vec(),
            asked: Vec::new(),
            too_small: None,
        }
    }
}

impl Generate for Scripted {
    fn generate(
        &mut self,
        req: GenerateRequest,
        on_event: &mut dyn FnMut(&Event) -> bool,
    ) -> Result<(String, Stats), ClientError> {
        self.asked.push(req.clone());
        if let Some(n) = self.too_small.take() {
            return Err(ClientError::Failed {
                kind: ErrorKind::Generate,
                message: format!(
                    "a prompt of {n} tokens does not fit a context of {}",
                    req.n_ctx
                ),
            });
        }
        let _ = on_event(&Event::Progress {
            id: req.id.clone(),
            phase: Phase::Generate,
            done: 1,
            total: 2,
        });
        let text = if req.grammar.is_some() {
            self.replies.remove(0).to_string()
        } else {
            "전체".to_string()
        };
        Ok((
            text,
            Stats {
                prompt_tokens: 10,
                gen_tokens: 5,
                ..Stats::default()
            },
        ))
    }
}

const ONE: &str = r#"{"summary":"앞","decisions":["결제 화면 배포를 11월 3일로 확정"],"action_items":[{"owner":"지호","task":"점검표 정리","due":"11월 3일"}],"open_questions":[]}"#;
const TWO: &str = r#"{"summary":"뒤","decisions":["결제 화면 배포를 11월 3일로 확정","예산 300만 원 확정"],"action_items":[],"open_questions":["보관 기간"]}"#;

fn lines(n: usize, syllables_each: usize) -> Vec<String> {
    (0..n).map(|_| "가".repeat(syllables_each)).collect()
}

fn opts() -> Options {
    Options {
        model: "/models/qwen.gguf".into(),
        ..Options::default()
    }
}

#[test]
fn the_prompt_is_qwens_chat_format_with_thinking_off() {
    assert_eq!(
        chatml("안녕"),
        "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn a_transcript_is_cut_into_chunks_of_about_ten_minutes_on_line_boundaries() {
    let text = lines(10, CHUNK_SYLLABLES / 4);
    let cut = chunks(&text);
    assert_eq!(cut, vec![0..4, 4..8, 8..10]);
    assert!(chunks(&[]).is_empty());
}

#[test]
fn context_is_sized_to_the_prompt_in_steps_of_2048() {
    let n = n_ctx_for(&"가".repeat(3000), 2048);
    assert_eq!(n % 2048, 0);
    assert!(n as f64 >= 3000.0 * 1.3 + 2048.0, "{n}");
    assert!(n <= 131072);
}

#[test]
fn every_chunk_is_summarised_in_the_fixed_format_and_the_lists_are_joined_once() {
    let mut g = Scripted::new(&[ONE, TWO]);
    let text = lines(8, CHUNK_SYLLABLES / 4);

    let s = summarise(&mut g, &text, &opts(), &AtomicBool::new(false), |_| {}).unwrap();

    assert_eq!(g.asked.len(), 3, "two chunks and the overall summary");
    for chunk in &g.asked[..2] {
        assert_eq!(chunk.grammar.as_deref(), Some(GRAMMAR));
        assert!(chunk.prompt.contains(INSTRUCTIONS));
        assert!(chunk.prompt.starts_with("<|im_start|>user\n"));
        assert_eq!(chunk.repeat_penalty, Some(1.1));
    }
    assert!(g.asked[2].grammar.is_none());
    assert!(g.asked[2].prompt.contains("앞\n뒤"));
    assert_eq!(s.summary, "전체");
    assert_eq!(
        s.decisions,
        ["결제 화면 배포를 11월 3일로 확정", "예산 300만 원 확정"]
    );
    assert_eq!(s.action_items.len(), 1);
    assert_eq!(s.action_items[0].owner, "지호");
    assert_eq!(s.open_questions, ["보관 기간"]);
    assert_eq!((s.chunks, s.chunks_broken), (2, 0));
    assert_eq!(s.input_tokens, 30);
}

#[test]
fn a_chunk_that_breaks_the_format_loses_its_items_but_not_the_summary() {
    let mut g = Scripted::new(&["{\"summary\": 끊김", TWO]);
    let text = lines(8, CHUNK_SYLLABLES / 4);

    let s = summarise(&mut g, &text, &opts(), &AtomicBool::new(false), |_| {}).unwrap();

    assert_eq!((s.chunks, s.chunks_broken), (2, 1));
    assert_eq!(s.decisions.len(), 2);
}

#[test]
fn a_prompt_that_does_not_fit_is_retried_once_with_the_size_the_sidecar_reports() {
    let mut g = Scripted::new(&[ONE]);
    g.too_small = Some(9000);
    let text = lines(2, 100);

    summarise(&mut g, &text, &opts(), &AtomicBool::new(false), |_| {}).unwrap();

    assert_eq!(g.asked.len(), 3);
    assert!(g.asked[1].n_ctx >= 9000 + 2048, "{}", g.asked[1].n_ctx);
    assert_eq!(g.asked[1].n_ctx % 1024, 0);
}

#[test]
fn progress_runs_from_zero_to_one_across_the_chunks() {
    let mut g = Scripted::new(&[ONE, TWO]);
    let text = lines(8, CHUNK_SYLLABLES / 4);
    let mut seen = Vec::new();

    summarise(&mut g, &text, &opts(), &AtomicBool::new(false), |p| {
        seen.push(p)
    })
    .unwrap();

    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    assert_eq!(seen.first(), Some(&0.0));
    assert_eq!(seen.last(), Some(&1.0));
    assert!(seen.iter().any(|&p| p > 0.0 && p < 0.5), "{seen:?}");
}

#[test]
fn a_cancelled_summary_stops_before_the_next_chunk() {
    let mut g = Scripted::new(&[ONE, TWO]);
    let text = lines(8, CHUNK_SYLLABLES / 4);
    let cancel = AtomicBool::new(false);

    let err = summarise(&mut g, &text, &opts(), &cancel, |p| {
        if p > 0.0 {
            cancel.store(true, Ordering::SeqCst);
        }
    })
    .unwrap_err();

    assert!(matches!(
        err,
        ClientError::Failed {
            kind: ErrorKind::Cancelled,
            ..
        }
    ));
    assert_eq!(g.asked.len(), 1);
}

#[test]
fn an_empty_transcript_asks_for_nothing() {
    let mut g = Scripted::new(&[]);
    let s = summarise(&mut g, &[], &opts(), &AtomicBool::new(false), |_| {}).unwrap();
    assert!(g.asked.is_empty());
    assert_eq!(s, Summary::default());
}

#[test]
fn parse_takes_the_first_object_in_the_fixed_format() {
    let s = parse(&format!("```json\n{ONE}\n```")).unwrap();
    assert_eq!(s.decisions.len(), 1);
    assert!(parse("{\"summary\":\"x\"}").is_none());
    assert!(parse("no json").is_none());
}
