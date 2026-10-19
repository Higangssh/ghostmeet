//! Decoding by our own speech spans — the live window and the after-meeting pass.
//!
//! Both cut speech with the same rule (`spans::BENCH`) and decode each span on its own,
//! which is the configuration the benchmark measured (`*-filtered`): segment times are
//! the decoder's within the span, offset into the input and clipped to the span; blank
//! text, credits on short spans and the vocabulary echoed back (`blocklist`) are
//! dropped. The decoder's own VAD and long-form mode are never used.
//!
//! The models sit behind two small traits so this wiring is tested without them.

use crate::blocklist::{is_blank, is_hallucination, is_vocabulary_echo};
use crate::incremental::{RawSegment, Segment};
use crate::spans::{finish_span, speech_spans, Span, SpanConfig, SpanCutter, FRAME};
use crate::{AudioSource, Transcribe, TranscribeError, SAMPLE_RATE};

/// Silero: one speech probability per 512-sample frame.
pub trait SpeechProbs {
    fn probs(&mut self, audio: &[f32]) -> Result<Vec<f32>, TranscribeError>;
}

/// Whisper over one span, times relative to the span.
pub trait SpanDecode {
    /// The user's vocabulary the decoder was primed with (STT-7), if any: text that only
    /// echoes it back is dropped like a credit.
    fn vocabulary(&self) -> &[String] {
        &[]
    }

    fn decode(&mut self, audio: &[f32]) -> Result<Vec<RawSegment>, TranscribeError>;
}

/// The live path: a `Transcribe` for the tail window that decodes span by span.
pub struct SpanTranscriber<V, D> {
    vad: V,
    decoder: D,
    cfg: SpanConfig,
    dropped: usize,
}

impl<V: SpeechProbs, D: SpanDecode> SpanTranscriber<V, D> {
    pub fn new(vad: V, decoder: D, cfg: SpanConfig) -> Self {
        Self {
            vad,
            decoder,
            cfg,
            dropped: 0,
        }
    }

    /// Segments the blocklist has dropped so far.
    pub fn dropped(&self) -> usize {
        self.dropped
    }
}

impl<V: SpeechProbs, D: SpanDecode> Transcribe for SpanTranscriber<V, D> {
    fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
        let probs = whole_frames(self.vad.probs(audio)?, audio.len());
        let mut out = Vec::new();
        for span in speech_spans(&probs, audio.len(), &self.cfg) {
            let piece = &audio[span.start..span.end];
            out.extend(decode_span(
                &mut self.decoder,
                piece,
                span,
                &mut self.dropped,
            )?);
        }
        Ok(out)
    }
}

/// The after-meeting pass over a whole recorded channel.
///
/// The session is cut a `chunk_sec` chunk at a time, with one span state machine running
/// across chunks, so a span crossing a chunk edge is decoded once, whole, and memory stays
/// bounded however long the meeting was. Any decoder failure fails the pass: a final
/// transcript with holes must not replace a complete live one. `progress` gets the
/// fraction of audio cut so far, ending at 1.0.
pub fn final_transcript(
    source: &dyn AudioSource,
    vad: &mut impl SpeechProbs,
    decoder: &mut impl SpanDecode,
    cfg: &SpanConfig,
    chunk_sec: f64,
    mut progress: impl FnMut(f64),
) -> Result<Vec<Segment>, TranscribeError> {
    let total = source.total_samples();
    let chunk = ((chunk_sec * SAMPLE_RATE as f64) as usize / FRAME).max(1) * FRAME;
    let mut cutter = SpanCutter::new(*cfg);
    let mut dropped = 0;
    let mut out = Vec::new();

    let mut decode = |raw: Span, out: &mut Vec<Segment>| -> Result<(), TranscribeError> {
        for span in finish_span(raw, total, cfg) {
            let audio = source.read(span.start, Some(span.end))?;
            for r in decode_span(decoder, &audio, span, &mut dropped)? {
                out.push(Segment::new(r.text, r.start, r.end));
            }
        }
        Ok(())
    };

    let mut pos = 0;
    while pos < total {
        let end = (pos + chunk).min(total);
        let audio = source.read(pos, Some(end))?;
        for p in whole_frames(vad.probs(&audio)?, audio.len()) {
            if let Some(raw) = cutter.push(p) {
                decode(raw, &mut out)?;
            }
        }
        pos = end;
        progress(pos as f64 / total as f64);
    }
    if let Some(raw) = cutter.finish() {
        decode(raw, &mut out)?;
    }
    progress(1.0);
    if dropped > 0 {
        log::info!("final pass dropped {dropped} credit or vocabulary-echo segment(s)");
    }
    Ok(out)
}

/// Decode one span; segment times come back in the span's own frame of reference
/// (`span.start` samples into whatever `span` was cut from).
fn decode_span(
    decoder: &mut impl SpanDecode,
    audio: &[f32],
    span: Span,
    dropped: &mut usize,
) -> Result<Vec<RawSegment>, TranscribeError> {
    let (offset, end, duration) = (
        span.start_sec() as f64,
        span.end_sec() as f64,
        span.duration_sec(),
    );
    let mut out = Vec::new();
    for s in decoder.decode(audio)? {
        let text = s.text.trim();
        if is_blank(text) {
            continue;
        }
        if is_hallucination(text, duration) || is_vocabulary_echo(text, decoder.vocabulary()) {
            *dropped += 1;
            continue;
        }
        let start = (offset + s.start).min(end);
        out.push(RawSegment {
            start,
            end: (offset + s.end).clamp(start, end),
            text: text.to_string(),
        });
    }
    Ok(out)
}

/// sherpa scores whole frames only; a VAD that pads the trailing partial frame reports
/// one probability more.
fn whole_frames(mut probs: Vec<f32>, samples: usize) -> Vec<f32> {
    probs.truncate(samples / FRAME);
    probs
}
