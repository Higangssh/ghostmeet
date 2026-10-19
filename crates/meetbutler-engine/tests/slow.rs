//! Slow tier: the real models on synthetic speech.
//!
//!     bench/make_synth.sh && bench/make_replies.sh      # synthetic audio, macOS `say`
//!     cargo test -p meetbutler-engine --features whisper,silero --test slow -- --ignored --nocapture
//!
//! Models are read from `bench/data/models` (or `MEETBUTLER_MODELS`), already downloaded;
//! nothing here touches the network. The audio is made-up lines in the Yuna voice, never
//! a meeting.
#![cfg(all(feature = "whisper", feature = "silero"))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use meetbutler_engine::incremental::{IncrementalTranscriber, Segment, WindowConfig};
use meetbutler_engine::silero::SileroOnnx;
use meetbutler_engine::span_decoder::{final_transcript, SpanTranscriber};
use meetbutler_engine::spans::BENCH;
use meetbutler_engine::whisper::{WhisperDecoder, WhisperModel};
use meetbutler_engine::{AudioSource, SAMPLE_RATE};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn models() -> PathBuf {
    std::env::var_os("MEETBUTLER_MODELS")
        .map_or_else(|| repo().join("bench/data/models"), PathBuf::from)
}

fn synth() -> PathBuf {
    repo().join("bench/data/synth")
}

/// One model on the GPU at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Loaded per test and dropped with it: a Metal model still alive when the process exits
/// trips ggml's device teardown (`ggml_metal_device_free`), so nothing is kept in a static.
fn model(name: &str) -> Arc<WhisperModel> {
    Arc::new(WhisperModel::load(models().join(format!("ggml-{name}.bin")), true).unwrap())
}

fn vad() -> SileroOnnx {
    SileroOnnx::load(models().join("silero_vad.onnx")).unwrap()
}

fn decoder(name: &str) -> WhisperDecoder {
    WhisperDecoder::new(model(name), Some("ko".into())).unwrap()
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("{}: {e} — run bench/make_synth.sh", path.display()));
    let spec = r.spec();
    assert_eq!(
        (spec.sample_rate, spec.channels, spec.bits_per_sample),
        (16_000, 1, 16)
    );
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

struct Memory(Vec<f32>);

impl AudioSource for Memory {
    fn total_samples(&self) -> usize {
        self.0.len()
    }
    fn read(&self, start: usize, end: Option<usize>) -> std::io::Result<Vec<f32>> {
        let end = end.unwrap_or(self.0.len()).min(self.0.len());
        Ok(self.0[start.min(end)..end].to_vec())
    }
}

/// Grows by `tick` seconds at a time, like capture does.
struct Growing {
    audio: Vec<f32>,
    visible: std::sync::atomic::AtomicUsize,
}

impl AudioSource for Growing {
    fn total_samples(&self) -> usize {
        self.visible.load(std::sync::atomic::Ordering::SeqCst)
    }
    fn read(&self, start: usize, end: Option<usize>) -> std::io::Result<Vec<f32>> {
        let total = self.total_samples();
        let end = end.unwrap_or(total).min(total);
        Ok(self.audio[start.min(end)..end].to_vec())
    }
}

/// Letters and digits only, so CER ignores spacing and punctuation.
fn chars(text: &str) -> Vec<char> {
    text.chars().filter(|c| c.is_alphanumeric()).collect()
}

fn cer(reference: &str, hypothesis: &str) -> f64 {
    let (r, h) = (chars(reference), chars(hypothesis));
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hc) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + (rc != hc) as usize)
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

/// The lines `make_synth.sh` voices, pauses removed: one source of truth for the script.
fn script() -> String {
    let sh = std::fs::read_to_string(repo().join("bench/make_synth.sh")).unwrap();
    let body = sh
        .split("<<'TEXT'\n")
        .nth(1)
        .unwrap()
        .split("\nTEXT")
        .next()
        .unwrap();
    let mut out = String::new();
    let mut rest = body;
    while let Some(i) = rest.find("[[") {
        out.push_str(&rest[..i]);
        rest = &rest[rest[i..].find("]]").unwrap() + i + 2..];
    }
    out.push_str(rest);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

const SHORT_REPLIES: &[&str] = &["네", "감사합니다", "아니요", "맞아요", "잠깐만요", "글쎄요"];

/// Replies the benchmark's VAD itself never puts in a span, so no decoder or blocklist can
/// keep them: the "네" at 9.3-10.1 s, 0.7 s after a sentence (absent from
/// `fixtures/silero_synth.json` too). VAD recall is measured in the benchmark; this test
/// is about what happens to speech once it is in a span.
const MISSED_BY_THE_VAD: &[&str] = &["네"];

fn text_of(segs: &[Segment]) -> String {
    segs.iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs bench/data models and synthetic audio"]
fn the_final_pass_keeps_every_short_reply_in_the_synthetic_session() {
    let _g = serial();
    let session = Memory(read_wav(&synth().join("session.wav")));
    let started = std::time::Instant::now();

    let out = final_transcript(
        &session,
        &mut vad(),
        &mut decoder("large-v3-turbo"),
        &BENCH,
        600.0,
        |_| {},
    )
    .unwrap();

    let text = text_of(&out);
    let score = cer(&script(), &text);
    eprintln!(
        "final (turbo): CER {:.1}% in {:.1}s\n  {text}",
        score * 100.0,
        started.elapsed().as_secs_f32()
    );
    let found: Vec<&str> = out.iter().map(|s| s.text.as_str()).collect();
    for reply in SHORT_REPLIES
        .iter()
        .filter(|r| !MISSED_BY_THE_VAD.contains(r))
    {
        assert!(
            found
                .iter()
                .any(|t| chars(t).starts_with(&chars(reply)[..])),
            "lost \"{reply}\": {found:?}"
        );
    }
    assert!(
        score <= 0.03,
        "final CER {:.1}% is over the 3% bar",
        score * 100.0
    );
}

#[test]
#[ignore = "needs bench/data models and synthetic audio"]
fn each_benchmark_reply_survives_on_its_own_between_pauses() {
    // the replies the benchmark's recall check used, each between a second of silence on
    // either side as in a meeting
    let _g = serial();
    let dir = repo().join("bench/data/replies");
    let tsv = std::fs::read_to_string(dir.join("replies.tsv")).expect("run bench/make_replies.sh");
    let mut dec = decoder("large-v3-turbo");
    let mut lost = Vec::new();
    for line in tsv.lines() {
        let (file, reply) = line.split_once('\t').unwrap();
        let silence = vec![0.0; SAMPLE_RATE];
        let audio = [&silence[..], &read_wav(&dir.join(file)), &silence[..]].concat();
        let out =
            final_transcript(&Memory(audio), &mut vad(), &mut dec, &BENCH, 600.0, |_| {}).unwrap();
        let text = text_of(&out);
        eprintln!("  {reply:6} -> {text:?}");
        if cer(reply, &text) > 0.0 {
            lost.push(reply.to_string());
        }
    }
    assert!(lost.is_empty(), "lost or garbled: {lost:?}");
}

#[test]
#[ignore = "needs bench/data models and synthetic audio"]
fn live_captions_with_small_stay_under_the_live_bar_without_duplicates() {
    let _g = serial();
    let audio = read_wav(&synth().join("session.wav"));
    let total = audio.len();
    let source = Arc::new(Growing {
        audio,
        visible: 0.into(),
    });
    let transcribe = SpanTranscriber::new(vad(), decoder("small"), BENCH);
    let mut t = IncrementalTranscriber::new(source.clone(), transcribe, WindowConfig::default());

    let tick = 3 * SAMPLE_RATE;
    let mut longest = std::time::Duration::ZERO;
    let mut shown = 0;
    while shown < total {
        shown = (shown + tick).min(total);
        source
            .visible
            .store(shown, std::sync::atomic::Ordering::SeqCst);
        let started = std::time::Instant::now();
        t.step(false);
        longest = longest.max(started.elapsed());
    }
    while t.pending_samples() > 0 {
        t.step(true);
    }

    let text = t.full_text();
    let score = cer(&script(), &text);
    eprintln!(
        "live (small, 3 s ticks): CER {:.1}%, slowest pass {:.2}s\n  {text}",
        score * 100.0,
        longest.as_secs_f32()
    );
    let mut lines: Vec<&str> = t.transcript().iter().map(|s| s.text.as_str()).collect();
    let n = lines.len();
    lines.sort();
    lines.dedup();
    assert!(
        lines.len() == n || n < 3,
        "duplicated captions: {:?}",
        t.transcript()
    );
    assert!(
        score <= 0.10,
        "live CER {:.1}% is over the 10% bar",
        score * 100.0
    );
}
