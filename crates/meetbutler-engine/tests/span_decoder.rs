//! Decoding by our own spans, live and after the meeting, without either model.
//!
//! A fake VAD reports speech wherever the audio is non-zero; a fake decoder reports what
//! it was handed. That is enough to check the wiring the benchmark measured: spans cut by
//! the shared rule, one decode per span, times offset into the session, credits on short
//! spans dropped and nothing else.

use std::io;
use std::sync::{Arc, Mutex};

use meetbutler_engine::incremental::RawSegment;
use meetbutler_engine::span_decoder::{final_transcript, SpanDecode, SpanTranscriber, SpeechProbs};
use meetbutler_engine::spans::{BENCH, FRAME};
use meetbutler_engine::{AudioSource, Transcribe, TranscribeError, SAMPLE_RATE};

const SR: usize = SAMPLE_RATE;

/// Speech wherever a frame is louder than silence.
#[derive(Default)]
struct LoudnessVad {
    calls: Arc<Mutex<Vec<usize>>>,
}

impl SpeechProbs for LoudnessVad {
    fn probs(&mut self, audio: &[f32]) -> Result<Vec<f32>, TranscribeError> {
        self.calls.lock().unwrap().push(audio.len());
        Ok(audio
            .chunks_exact(FRAME)
            .map(|f| {
                if f.iter().any(|&s| s != 0.0) {
                    0.9
                } else {
                    0.0
                }
            })
            .collect())
    }
}

/// Reports one segment per span, labelled by what the span sounds like: the fake audio
/// carries a "word id" as its amplitude.
#[derive(Default, Clone)]
struct EchoDecoder {
    spans: Arc<Mutex<Vec<usize>>>,
    words: Arc<Mutex<Vec<(f32, &'static str)>>>,
    terms: Vec<String>,
}

impl EchoDecoder {
    fn says(pairs: &[(f32, &'static str)]) -> Self {
        let d = Self::default();
        *d.words.lock().unwrap() = pairs.to_vec();
        d
    }
}

impl SpanDecode for EchoDecoder {
    fn vocabulary(&self) -> &[String] {
        &self.terms
    }

    fn decode(&mut self, audio: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
        self.spans.lock().unwrap().push(audio.len());
        let level = audio.iter().cloned().fold(0.0f32, f32::max);
        let words = self.words.lock().unwrap();
        let text = words
            .iter()
            .find(|(l, _)| (l - level).abs() < 1e-3)
            .map_or("speech", |w| w.1);
        let dur = audio.len() as f64 / SR as f64;
        Ok(vec![RawSegment {
            start: 0.1,
            end: dur - 0.1,
            text: text.to_string(),
        }])
    }
}

/// Builds audio: (seconds, level) runs; level 0 is silence.
fn audio(runs: &[(f32, f32)]) -> Vec<f32> {
    runs.iter()
        .flat_map(|&(secs, level)| vec![level; (secs * SR as f32) as usize])
        .collect()
}

struct Memory(Vec<f32>, Arc<Mutex<Vec<(usize, usize)>>>);

impl AudioSource for Memory {
    fn total_samples(&self) -> usize {
        self.0.len()
    }
    fn read(&self, start: usize, end: Option<usize>) -> io::Result<Vec<f32>> {
        let end = end.unwrap_or(self.0.len()).min(self.0.len());
        let start = start.min(end);
        self.1.lock().unwrap().push((start, end));
        Ok(self.0[start..end].to_vec())
    }
}

fn texts(raw: &[RawSegment]) -> Vec<&str> {
    raw.iter().map(|s| s.text.as_str()).collect()
}

#[test]
fn each_span_is_decoded_on_its_own_with_times_in_the_window() {
    let window = audio(&[(1.0, 0.0), (3.0, 0.5), (2.0, 0.0), (4.0, 0.6), (1.0, 0.0)]);
    let decoder = EchoDecoder::says(&[(0.5, "first"), (0.6, "second")]);
    let mut t = SpanTranscriber::new(LoudnessVad::default(), decoder.clone(), BENCH);

    let out = t.transcribe(&window).unwrap();

    assert_eq!(texts(&out), ["first", "second"]);
    assert_eq!(decoder.spans.lock().unwrap().len(), 2);
    // the first span starts near 1 s (minus padding and sherpa's look-back)
    assert!(out[0].start > 0.5 && out[0].start < 1.2, "{:?}", out[0]);
    assert!(out[1].start > 5.5 && out[1].start < 6.2, "{:?}", out[1]);
    assert!(out[1].end <= window.len() as f64 / SR as f64);
}

#[test]
fn a_segment_never_ends_past_its_span() {
    struct Overrun;
    impl SpanDecode for Overrun {
        fn decode(&mut self, _: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
            Ok(vec![RawSegment {
                start: 0.0,
                end: 99.0,
                text: "long".into(),
            }])
        }
    }
    let window = audio(&[(1.0, 0.0), (2.0, 0.5), (3.0, 0.0)]);
    let mut t = SpanTranscriber::new(LoudnessVad::default(), Overrun, BENCH);

    let out = t.transcribe(&window).unwrap();

    assert!(out[0].end < 4.0, "{:?}", out[0]);
}

#[test]
fn silence_is_never_sent_to_the_decoder() {
    let decoder = EchoDecoder::default();
    let mut t = SpanTranscriber::new(LoudnessVad::default(), decoder.clone(), BENCH);

    assert!(t.transcribe(&audio(&[(10.0, 0.0)])).unwrap().is_empty());
    assert!(decoder.spans.lock().unwrap().is_empty());
}

#[test]
fn a_credit_on_a_short_span_is_dropped_but_a_short_reply_is_kept() {
    let window = audio(&[
        (1.0, 0.0),
        (0.6, 0.1), // a click at a clip edge
        (2.0, 0.0),
        (0.6, 0.2), // a real "감사합니다."
        (2.0, 0.0),
        (4.0, 0.3), // a long span someone reads the phrase out in
        (1.0, 0.0),
    ]);
    let decoder = EchoDecoder::says(&[
        (0.1, "시청해주셔서 감사합니다."),
        (0.2, "감사합니다."),
        (0.3, "시청해주셔서 감사합니다."),
    ]);
    let mut t = SpanTranscriber::new(LoudnessVad::default(), decoder, BENCH);

    let out = t.transcribe(&window).unwrap();

    assert_eq!(texts(&out), ["감사합니다.", "시청해주셔서 감사합니다."]);
    assert_eq!(t.dropped(), 1);
}

#[test]
fn blank_and_punctuation_only_segments_are_dropped() {
    struct Dots;
    impl SpanDecode for Dots {
        fn decode(&mut self, _: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
            Ok(vec![
                RawSegment {
                    start: 0.0,
                    end: 0.5,
                    text: " . ".into(),
                },
                RawSegment {
                    start: 0.5,
                    end: 1.0,
                    text: "".into(),
                },
                RawSegment {
                    start: 1.0,
                    end: 1.5,
                    text: "네.".into(),
                },
            ])
        }
    }
    let mut t = SpanTranscriber::new(LoudnessVad::default(), Dots, BENCH);

    let out = t
        .transcribe(&audio(&[(1.0, 0.0), (2.0, 0.5), (1.0, 0.0)]))
        .unwrap();

    assert_eq!(texts(&out), ["네."]);
}

#[test]
fn a_failing_decoder_fails_the_pass() {
    // the transcriber logs it and consumes the window; silently returning nothing would
    // look like silence
    struct Broken;
    impl SpanDecode for Broken {
        fn decode(&mut self, _: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
            Err("out of memory".into())
        }
    }
    let mut t = SpanTranscriber::new(LoudnessVad::default(), Broken, BENCH);
    assert!(t
        .transcribe(&audio(&[(1.0, 0.0), (2.0, 0.5), (1.0, 0.0)]))
        .is_err());
}

// --- after the meeting ---------------------------------------------------------------

#[test]
fn the_final_pass_cuts_the_whole_session_with_absolute_times() {
    let session = audio(&[(5.0, 0.0), (3.0, 0.5), (100.0, 0.0), (4.0, 0.6), (2.0, 0.0)]);
    let reads = Arc::default();
    let source = Memory(session, reads);
    let decoder = EchoDecoder::says(&[(0.5, "early"), (0.6, "late")]);

    let out = final_transcript(
        &source,
        &mut LoudnessVad::default(),
        &mut decoder.clone(),
        &BENCH,
        30.0,
        |_| {},
    )
    .unwrap();

    let got: Vec<(&str, bool)> = out
        .iter()
        .map(|s| (s.text.as_str(), s.start > 100.0))
        .collect();
    assert_eq!(got, [("early", false), ("late", true)]);
    assert!(out[1].start > 107.0 && out[1].start < 108.5, "{:?}", out[1]);
}

#[test]
fn the_final_pass_reads_the_session_a_bounded_chunk_at_a_time() {
    // a 4-hour session must not be read into memory to be cut into spans
    let session = audio(&[(50.0, 0.0), (3.0, 0.5), (50.0, 0.0)]);
    let reads: Arc<Mutex<Vec<(usize, usize)>>> = Arc::default();
    let source = Memory(session, reads.clone());
    let vad = LoudnessVad::default();
    let vad_calls = vad.calls.clone();

    final_transcript(
        &source,
        &mut { vad },
        &mut EchoDecoder::default(),
        &BENCH,
        20.0,
        |_| {},
    )
    .unwrap();

    assert!(reads
        .lock()
        .unwrap()
        .iter()
        .all(|(a, b)| b - a <= 20 * SR + FRAME));
    assert!(vad_calls.lock().unwrap().len() >= 5);
}

#[test]
fn a_span_across_a_chunk_boundary_is_decoded_once_whole() {
    let session = audio(&[(18.0, 0.0), (6.0, 0.5), (20.0, 0.0)]); // speech crosses 20 s
    let source = Memory(session, Arc::default());
    let decoder = EchoDecoder::says(&[(0.5, "across")]);

    let out = final_transcript(
        &source,
        &mut LoudnessVad::default(),
        &mut decoder.clone(),
        &BENCH,
        20.0,
        |_| {},
    )
    .unwrap();

    assert_eq!(
        out.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        ["across"]
    );
    let decoded = decoder.spans.lock().unwrap();
    assert_eq!(decoded.len(), 1);
    assert!(
        decoded[0] >= 6 * SR,
        "the whole span, not a piece: {}",
        decoded[0]
    );
}

#[test]
fn the_final_pass_reports_progress_to_the_end() {
    let session = audio(&[(5.0, 0.0), (3.0, 0.5), (60.0, 0.0)]);
    let source = Memory(session, Arc::default());
    let mut seen = Vec::new();

    final_transcript(
        &source,
        &mut LoudnessVad::default(),
        &mut EchoDecoder::default(),
        &BENCH,
        20.0,
        |p| seen.push(p),
    )
    .unwrap();

    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    assert_eq!(*seen.last().unwrap(), 1.0);
}

#[test]
fn the_final_pass_stops_on_a_decoder_failure() {
    // a final transcript with holes must not replace a complete live one
    struct Broken;
    impl SpanDecode for Broken {
        fn decode(&mut self, _: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
            Err("out of memory".into())
        }
    }
    let source = Memory(audio(&[(1.0, 0.0), (3.0, 0.5), (1.0, 0.0)]), Arc::default());
    assert!(final_transcript(
        &source,
        &mut LoudnessVad::default(),
        &mut Broken,
        &BENCH,
        30.0,
        |_| {}
    )
    .is_err());
}

#[test]
fn the_final_pass_drops_credits_like_the_live_one() {
    let session = audio(&[(1.0, 0.0), (0.6, 0.1), (3.0, 0.0), (0.6, 0.2), (2.0, 0.0)]);
    let source = Memory(session, Arc::default());
    let mut decoder = EchoDecoder::says(&[(0.1, "한글자막 by 누군가"), (0.2, "네.")]);

    let out = final_transcript(
        &source,
        &mut LoudnessVad::default(),
        &mut decoder,
        &BENCH,
        30.0,
        |_| {},
    )
    .unwrap();

    assert_eq!(
        out.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        ["네."]
    );
}

#[test]
fn the_final_pass_drops_the_vocabulary_echoed_back_and_keeps_a_repeated_reply() {
    let session = audio(&[
        (1.0, 0.0),
        (1.5, 0.1),
        (3.0, 0.0),
        (0.6, 0.2),
        (2.0, 0.0),
        (3.0, 0.3),
        (1.0, 0.0),
    ]);
    let source = Memory(session, Arc::default());
    let mut decoder = EchoDecoder::says(&[
        (0.1, "지호, 지호, 지호, 지호."),
        (0.2, "네, 네."),
        (0.3, "지호님이 점검표를 정리해 주세요."),
    ]);
    decoder.terms = vec!["지호".into(), "주소록 동기화".into()];

    let out = final_transcript(
        &source,
        &mut LoudnessVad::default(),
        &mut decoder,
        &BENCH,
        30.0,
        |_| {},
    )
    .unwrap();

    assert_eq!(
        out.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        ["네, 네.", "지호님이 점검표를 정리해 주세요."]
    );
}
