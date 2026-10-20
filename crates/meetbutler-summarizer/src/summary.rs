//! The meeting summary (SUM-1): the transcript cut into chunks of about ten minutes, each
//! summarised by the local model in the fixed format, the lists joined (`merge`), and
//! one short last prompt for the overall summary. The configuration the benchmark chose
//! (`bench/results/RESULTS.md` §8: qwen3.5-4b, chunked, prompt p2); the prompt texts and
//! grammar are the files in `prompts/`, which `bench/run_summary.py` reads too.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::client::{ClientError, Sidecar};
use crate::merge::{merge, Part};
use crate::protocol::{ErrorKind, Event, GenerateRequest, Phase, Stats};

pub const INSTRUCTIONS: &str = include_str!("../prompts/instructions.txt");
pub const MERGE: &str = include_str!("../prompts/merge.txt");
pub const GRAMMAR: &str = include_str!("../prompts/summary.gbnf");

/// About ten minutes of Korean speech.
pub const CHUNK_SYLLABLES: usize = 3000;
pub const MAX_TOKENS_CHUNK: u32 = 2048;
pub const MAX_TOKENS_OVERALL: u32 = 400;
/// llama.cpp's usual repeat penalty: without it small models loop on one list item.
pub const REPEAT_PENALTY: f32 = 1.1;
/// Rough tokens per Hangul syllable, to size the context.
const TOKENS_PER_SYLLABLE: f64 = 1.3;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionItem {
    pub owner: String,
    pub task: String,
    pub due: String,
}

/// The fixed summary format, plus how it was made.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub summary: String,
    pub decisions: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub open_questions: Vec<String>,
    #[serde(default)]
    pub chunks: usize,
    /// Chunks whose answer was not the format; their items are lost.
    #[serde(default)]
    pub chunks_broken: usize,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    /// Where each listed item was said (SUM-3), from the merge step.
    #[serde(default)]
    pub evidence: crate::merge::Evidence,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The GGUF file.
    pub model: PathBuf,
}

/// Something that runs one generation: the sidecar, or a stand-in in tests. `on_event`
/// sees `loaded` and `progress`; returning false from it asks for the request to stop.
pub trait Generate {
    fn generate(
        &mut self,
        req: GenerateRequest,
        on_event: &mut dyn FnMut(&Event) -> bool,
    ) -> Result<(String, Stats), ClientError>;
}

impl Generate for Sidecar {
    fn generate(
        &mut self,
        req: GenerateRequest,
        on_event: &mut dyn FnMut(&Event) -> bool,
    ) -> Result<(String, Stats), ClientError> {
        let canceller = self.canceller();
        let mut asked = false;
        Sidecar::generate(self, req, |event| {
            if !on_event(event) && !asked {
                if let Event::Progress { id, .. } | Event::Loaded { id, .. } = event {
                    canceller.cancel(id);
                    asked = true;
                }
            }
        })
    }
}

/// Qwen3.5's chat template for one user turn, thinking off (what its jinja template
/// renders with `enable_thinking=False`; BOS is added by the tokenizer).
pub fn chatml(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn syllables(text: &str) -> usize {
    text.chars().filter(|c| ('가'..='힣').contains(c)).count()
}

/// Line ranges of about `CHUNK_SYLLABLES` each; a chunk ends at the line that reaches it.
pub fn chunks(lines: &[String]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut count) = (0, 0);
    for (i, line) in lines.iter().enumerate() {
        count += syllables(line);
        if count >= CHUNK_SYLLABLES {
            out.push(start..i + 1);
            start = i + 1;
            count = 0;
        }
    }
    if start < lines.len() {
        out.push(start..lines.len());
    }
    out
}

/// Context for a prompt and its answer, rounded up to 2048.
pub fn n_ctx_for(prompt: &str, max_tokens: u32) -> u32 {
    let need = (syllables(prompt) as f64 * TOKENS_PER_SYLLABLE
        + prompt.chars().count() as f64 * 0.1) as u32
        + max_tokens
        + 512;
    need.div_ceil(2048).saturating_mul(2048).min(131072)
}

/// The first JSON object in `output` that has every field of the format.
pub fn parse(output: &str) -> Option<Summary> {
    let mut from = 0;
    while let Some(i) = output[from..].find('{') {
        let start = from + i;
        let mut stream =
            serde_json::Deserializer::from_str(&output[start..]).into_iter::<serde_json::Value>();
        if let Some(Ok(value)) = stream.next() {
            let complete = ["summary", "decisions", "action_items", "open_questions"]
                .iter()
                .all(|f| value.get(f).is_some());
            return if complete {
                serde_json::from_value(value).ok()
            } else {
                None
            };
        }
        from = start + 1;
    }
    None
}

fn cancelled() -> ClientError {
    ClientError::Failed {
        kind: ErrorKind::Cancelled,
        message: "the summary was cancelled".into(),
    }
}

/// One generation, retried once with a larger context when the sidecar says the prompt
/// does not fit ("a prompt of N tokens ...").
fn run(
    gen: &mut dyn Generate,
    opts: &Options,
    prompt: String,
    grammar: Option<&str>,
    max_tokens: u32,
    on_event: &mut dyn FnMut(&Event) -> bool,
) -> Result<(String, Stats), ClientError> {
    let mut n_ctx = n_ctx_for(&prompt, max_tokens);
    for attempt in 0..2 {
        let req = GenerateRequest {
            id: String::new(),
            model: opts.model.clone(),
            prompt: prompt.clone(),
            max_tokens,
            n_ctx,
            grammar: grammar.map(String::from),
            repeat_penalty: Some(REPEAT_PENALTY),
        };
        match gen.generate(req, on_event) {
            Err(ClientError::Failed { message, .. })
                if attempt == 0 && reported_tokens(&message).is_some() =>
            {
                let tokens = reported_tokens(&message).unwrap();
                n_ctx = (tokens + max_tokens + 256).div_ceil(1024) * 1024;
            }
            other => return other,
        }
    }
    unreachable!("the second attempt returns")
}

fn reported_tokens(message: &str) -> Option<u32> {
    let rest = &message[message.find("prompt of ")? + "prompt of ".len()..];
    rest.split_whitespace().next()?.parse().ok()
}

/// Summarise a transcript, one line per segment. `progress` gets 0.0 first and 1.0
/// last; `cancel` is checked before every generation and while one runs.
pub fn summarise(
    gen: &mut dyn Generate,
    lines: &[String],
    opts: &Options,
    cancel: &AtomicBool,
    mut progress: impl FnMut(f64),
) -> Result<Summary, ClientError> {
    let ranges = chunks(lines);
    let mut out = Summary::default();
    if ranges.is_empty() {
        return Ok(out);
    }
    progress(0.0);
    // the chunks are the work; the overall summary is short and counts as one more step
    let steps = ranges.len() as f64 + 1.0;
    let mut parts = Vec::new();
    let mut summaries = Vec::new();
    for (i, range) in ranges.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err(cancelled());
        }
        let prompt = chatml(&format!(
            "{INSTRUCTIONS}\n\n전사:\n{}",
            lines[range.clone()].join("\n")
        ));
        let mut on_event = |e: &Event| {
            if let Event::Progress {
                phase, done, total, ..
            } = e
            {
                let within = match phase {
                    Phase::Prompt => 0.5 * *done as f64 / (*total).max(1) as f64,
                    Phase::Generate => 0.5 + 0.5 * *done as f64 / (*total).max(1) as f64,
                };
                progress((i as f64 + within.min(1.0)) / steps);
            }
            !cancel.load(Ordering::SeqCst)
        };
        let (text, stats) = run(
            gen,
            opts,
            prompt,
            Some(GRAMMAR),
            MAX_TOKENS_CHUNK,
            &mut on_event,
        )?;
        out.input_tokens += stats.prompt_tokens as u64;
        out.output_tokens += stats.gen_tokens as u64;
        progress((i + 1) as f64 / steps);
        let Some(part) = parse(&text) else {
            out.chunks_broken += 1;
            continue;
        };
        summaries.push(part.summary.clone());
        parts.push(Part {
            lines: range.clone(),
            summary: part,
        });
    }
    let merged = merge(lines, parts);
    (
        out.decisions,
        out.action_items,
        out.open_questions,
        out.evidence,
    ) = (
        merged.decisions,
        merged.action_items,
        merged.open_questions,
        merged.evidence,
    );
    out.chunks = ranges.len();
    if cancel.load(Ordering::SeqCst) {
        return Err(cancelled());
    }
    let mut on_event = |_: &Event| !cancel.load(Ordering::SeqCst);
    let (text, stats) = run(
        gen,
        opts,
        chatml(&format!("{MERGE}\n\n{}", summaries.join("\n"))),
        None,
        MAX_TOKENS_OVERALL,
        &mut on_event,
    )?;
    out.input_tokens += stats.prompt_tokens as u64;
    out.output_tokens += stats.gen_tokens as u64;
    out.summary = text.trim().to_string();
    progress(1.0);
    Ok(out)
}
