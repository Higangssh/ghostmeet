//! Speech spans from Silero frame probabilities.
//!
//! We cut speech ourselves instead of leaving it to the decoder's built-in VAD, and live
//! captions and the after-meeting pass cut it with the same function and the same values.
//! The rule is sherpa-onnx's `VoiceActivityDetector` (the one the benchmark measured
//! with), reimplemented over probabilities so it does not care which runtime produced
//! them: one probability per 512-sample frame, in order.
//!
//! On top of sherpa, one rule of our own: no span is longer than a Whisper window. sherpa
//! tightens its threshold past `max_speech_sec` but does not cut speech that never dips,
//! and a span longer than 30 s would send the decoder into its own long-form mode.

use crate::SAMPLE_RATE;

/// Samples per Silero frame at 16 kHz.
pub const FRAME: usize = 512;

/// The longest span handed to the decoder: one Whisper window.
pub const MAX_SPAN_SAMPLES: usize = 30 * SAMPLE_RATE;

/// Past `max_speech_sec`, sherpa ends a span on a much shorter, much clearer dip.
const LONG_THRESHOLD: f32 = 0.90;
const LONG_MIN_SILENCE_SEC: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpanConfig {
    /// A frame above this is speech.
    pub threshold: f32,
    /// A pause this long ends a span.
    pub min_silence_sec: f32,
    /// Shorter blips are dropped.
    pub min_speech_sec: f32,
    /// Past this, a brief dip is enough to end a span.
    pub max_speech_sec: f32,
    /// Context kept on each side of a span.
    pub pad_sec: f32,
}

/// The values the benchmark numbers were measured with (`bench/engines.py`, `VAD`).
pub const BENCH: SpanConfig = SpanConfig {
    threshold: 0.3,
    min_silence_sec: 0.5,
    min_speech_sec: 0.25,
    max_speech_sec: 25.0,
    pad_sec: 0.2,
};

impl SpanConfig {
    pub fn pad_samples(&self) -> usize {
        secs_to_samples(self.pad_sec)
    }
}

fn secs_to_samples(secs: f32) -> usize {
    (secs * SAMPLE_RATE as f32) as usize
}

/// `[start, end)` in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    pub fn start_sec(&self) -> f32 {
        self.start as f32 / SAMPLE_RATE as f32
    }

    pub fn end_sec(&self) -> f32 {
        self.end as f32 / SAMPLE_RATE as f32
    }

    pub fn duration_sec(&self) -> f32 {
        self.len() as f32 / SAMPLE_RATE as f32
    }

    /// The same span moved `by` samples later.
    pub fn offset(&self, by: usize) -> Self {
        Self::new(self.start + by, self.end + by)
    }
}

/// sherpa's span state machine, fed one frame probability at a time.
///
/// Emits unpadded spans in frame order. Streaming, so a whole session can be cut a chunk
/// of probabilities at a time without holding its audio in memory.
#[derive(Debug, Clone)]
pub struct SpanCutter {
    cfg: SpanConfig,
    min_speech: usize,
    max_speech: usize,
    // the model's own state (sherpa's SileroVadModel::IsSpeech)
    current: usize,
    temp_start: usize,
    temp_end: usize,
    triggered: bool,
    // the detector's buffer (sherpa's VoiceActivityDetector): samples [head, tail)
    head: usize,
    tail: usize,
    start: Option<usize>,
}

impl SpanCutter {
    pub fn new(cfg: SpanConfig) -> Self {
        Self {
            cfg,
            min_speech: secs_to_samples(cfg.min_speech_sec),
            max_speech: secs_to_samples(cfg.max_speech_sec),
            current: 0,
            temp_start: 0,
            temp_end: 0,
            triggered: false,
            head: 0,
            tail: 0,
            start: None,
        }
    }

    /// Feed the next frame's probability. Returns a span when this frame ended one.
    pub fn push(&mut self, prob: f32) -> Option<Span> {
        let long = self.tail - self.head > self.max_speech;
        let (threshold, min_silence) = if long {
            (LONG_THRESHOLD, secs_to_samples(LONG_MIN_SILENCE_SEC))
        } else {
            (self.cfg.threshold, secs_to_samples(self.cfg.min_silence_sec))
        };

        self.tail += FRAME;
        let speech = self.is_speech(prob, threshold, min_silence);
        let lookback = 2 * FRAME + self.min_speech;

        if speech {
            if self.start.is_none() {
                self.start = Some(self.tail.saturating_sub(lookback).max(self.head));
            }
            return None;
        }

        let mut out = None;
        match self.start.take() {
            Some(start) => {
                let end = self.tail.saturating_sub(min_silence).max(start);
                if end > start {
                    out = Some(Span::new(start, end));
                }
                self.head = end;
            }
            None => self.head = self.head.max(self.tail.saturating_sub(lookback)),
        }
        out
    }

    /// No more frames: speech still running becomes a span ending at the last frame.
    pub fn finish(&mut self) -> Option<Span> {
        let start = self.start.take()?;
        let end = self.tail;
        self.head = end;
        (end > start).then(|| Span::new(start, end))
    }

    fn is_speech(&mut self, prob: f32, threshold: f32, min_silence: usize) -> bool {
        self.current += FRAME;
        let above = prob > threshold;
        let below = prob < threshold;

        if above && self.temp_end != 0 {
            self.temp_end = 0;
        }
        if above && self.temp_start == 0 {
            // speech starts, but has to last min_speech before it counts
            self.temp_start = self.current;
            return false;
        }
        if above && !self.triggered {
            if self.current - self.temp_start < self.min_speech {
                return false;
            }
            self.triggered = true;
            return true;
        }
        if below && !self.triggered {
            self.temp_start = 0;
            self.temp_end = 0;
            return false;
        }
        let neg_threshold = (threshold - 0.15).max(0.01);
        if prob > neg_threshold && self.triggered {
            return true;
        }
        if below && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current;
            }
            if self.current - self.temp_end < min_silence {
                return true;
            }
            self.temp_start = 0;
            self.temp_end = 0;
            self.triggered = false;
        }
        false
    }
}

/// Speech spans of `total_samples` of audio, padded and clipped to it, none longer than
/// [`MAX_SPAN_SAMPLES`].
pub fn speech_spans(probs: &[f32], total_samples: usize, cfg: &SpanConfig) -> Vec<Span> {
    let mut cutter = SpanCutter::new(*cfg);
    let mut raw: Vec<Span> = probs.iter().filter_map(|&p| cutter.push(p)).collect();
    raw.extend(cutter.finish());
    raw.into_iter().flat_map(|s| finish_span(s, total_samples, cfg)).collect()
}

/// Pad a raw span, clip it to the audio, and split it if it is longer than a window.
pub fn finish_span(raw: Span, total_samples: usize, cfg: &SpanConfig) -> Vec<Span> {
    let pad = cfg.pad_samples();
    let padded = Span::new(
        raw.start.saturating_sub(pad),
        (raw.end + pad).min(total_samples),
    );
    if padded.is_empty() {
        return Vec::new();
    }
    split(padded, MAX_SPAN_SAMPLES)
}

/// Cut `span` into equal pieces no longer than `max`.
fn split(span: Span, max: usize) -> Vec<Span> {
    let pieces = span.len().div_ceil(max);
    let step = span.len().div_ceil(pieces);
    (0..pieces)
        .map(|i| Span::new(span.start + i * step, (span.start + (i + 1) * step).min(span.end)))
        .collect()
}
