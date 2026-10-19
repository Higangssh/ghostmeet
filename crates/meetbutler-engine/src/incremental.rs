//! Windowed incremental transcription.
//!
//! Each pass transcribes only the audio that has arrived since the last one. The window
//! has a hard upper bound, which is what keeps a 4-hour meeting costing the same per pass
//! as the first minute.
//!
//! Window edges are where duplicates come from. A segment that ends near the edge may be
//! a sentence cut off mid-word, so it is held back rather than committed, and the next
//! pass starts exactly where the last committed segment ended. That boundary is a real
//! pause, so nothing is re-read and nothing can be emitted twice. Only when the cursor had
//! to be cut blindly (silence, or speech filling a whole window) does the next pass rewind
//! by `overlap_sec` to catch words straddling the cut.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{AudioSource, Transcribe, SAMPLE_RATE};

/// One transcribed span, timed from the start of the session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    /// Seconds from the start of the session, derived from the sample cursor.
    pub start: f64,
    pub end: f64,
    /// The channel ("me" / "them") until diarization lands, then the speaker.
    pub speaker: Option<String>,
    /// Wall-clock seconds since the Unix epoch when the segment was recognised.
    pub timestamp: f64,
}

impl Segment {
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            speaker: None,
            timestamp: now(),
        }
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// What the decoder returns for one window: times relative to the audio passed in.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowConfig {
    /// The most audio one pass ever reads.
    pub window_sec: f64,
    /// How far a pass rewinds after a blind cut.
    pub overlap_sec: f64,
    /// A normal pass waits for at least this much new audio.
    pub min_new_sec: f64,
    /// Segments ending this close to the window edge wait for the next pass.
    pub holdback_sec: f64,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_sec: 30.0,
            overlap_sec: 2.0,
            min_new_sec: 1.0,
            holdback_sec: 1.0,
        }
    }
}

/// (absolute start, absolute end, text)
type Spoken = (f64, f64, String);

/// Transcribes a growing audio source one bounded window at a time.
pub struct IncrementalTranscriber<S, T> {
    source: S,
    transcribe: T,
    window: usize,
    overlap: usize,
    min_new: usize,
    holdback: f64,
    speaker: Option<String>,
    cursor: usize,      // samples consumed
    blind_cut: bool,    // cursor sits at a window edge, not at the end of speech
    emitted_until: f64, // absolute seconds covered by emitted segments
    transcript: Vec<Segment>,
}

impl<S: AudioSource, T: Transcribe> IncrementalTranscriber<S, T> {
    pub fn new(source: S, transcribe: T, cfg: WindowConfig) -> Self {
        let samples = |secs: f64| (secs * SAMPLE_RATE as f64) as usize;
        Self {
            source,
            transcribe,
            window: samples(cfg.window_sec),
            overlap: samples(cfg.overlap_sec),
            min_new: samples(cfg.min_new_sec),
            holdback: cfg.holdback_sec,
            speaker: None,
            cursor: 0,
            blind_cut: false,
            emitted_until: 0.0,
            transcript: Vec::new(),
        }
    }

    /// Label every segment from this source, e.g. with its channel.
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// Audio that has arrived but not been consumed yet.
    pub fn pending_samples(&self) -> usize {
        self.source.total_samples().saturating_sub(self.cursor)
    }

    pub fn transcript(&self) -> &[Segment] {
        &self.transcript
    }

    pub fn full_text(&self) -> String {
        self.transcript
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Transcribe the oldest audio not yet consumed.
    ///
    /// Set `force` once no more audio is coming: the threshold is skipped and speech
    /// running into the end of the session is committed instead of held back.
    pub fn step(&mut self, force: bool) -> Vec<Segment> {
        let total = self.source.total_samples();
        let pending = total.saturating_sub(self.cursor);
        if pending == 0 || (!force && pending < self.min_new) {
            return Vec::new();
        }

        // Oldest-first, so a backlog is caught up rather than skipped over. Bounding the
        // END (not the start) keeps cost per pass flat without ever dropping audio.
        let rewind = if self.blind_cut { self.overlap } else { 0 };
        let start = self.cursor.saturating_sub(rewind);
        let end = total.min(start + self.window);
        let offset = start as f64 / SAMPLE_RATE as f64;

        let decoded = self
            .source
            .read(start, Some(end))
            .map_err(Into::into)
            .and_then(|audio| {
                if audio.is_empty() {
                    Ok(None)
                } else {
                    self.transcribe.transcribe(&audio).map(Some)
                }
            });
        let raw = match decoded {
            Ok(Some(raw)) => raw,
            Ok(None) => return Vec::new(),
            Err(err) => {
                // one bad pass must not end the meeting
                log::error!("transcription pass failed at {offset:.1}s: {err}");
                self.cursor = end;
                self.blind_cut = true;
                return Vec::new();
            }
        };

        let spoken: Vec<Spoken> = raw
            .into_iter()
            .filter_map(|r| {
                let text = r.text.trim();
                (!text.is_empty()).then(|| (offset + r.start, offset + r.end, text.to_string()))
            })
            .collect();
        let final_pass = force && end == total; // no audio after this window will ever be read
        let mut ready = self.committable(&spoken, end as f64 / SAMPLE_RATE as f64, final_pass);
        let mut blind = final_pass;
        if ready == 0 && !spoken.is_empty() && end - start >= self.window {
            // speech fills the whole window: there is no pause to wait for, so cut blindly
            ready = spoken.len();
            blind = true;
        }
        let held = ready < spoken.len();

        let previous = self.cursor;
        let new = self.accept(spoken.into_iter().take(ready));
        if blind {
            self.cursor = end;
            self.blind_cut = true;
        } else if !new.is_empty() {
            // resume exactly at the end of committed speech: a real pause, nothing re-read
            let resume = (self.emitted_until * SAMPLE_RATE as f64) as usize;
            self.cursor = end.min(previous.max(resume));
            self.blind_cut = false;
        } else if !held {
            // nothing recognised — consume the window rather than rescan it
            self.cursor = end;
            self.blind_cut = true;
        }
        // else: only speech still running into the edge; wait for more audio. The next
        // window starts at the same cursor and is still capped at window_sec.
        new
    }

    /// How many leading segments are safe to commit: all of them on the final pass,
    /// otherwise those ending before the hold-back margin. Order is kept, so the first
    /// one held holds the rest.
    fn committable(&self, spoken: &[Spoken], window_end: f64, final_pass: bool) -> usize {
        if final_pass {
            return spoken.len();
        }
        let edge = window_end - self.holdback;
        spoken.iter().take_while(|s| s.1 <= edge).count()
    }

    /// Keep segments whose midpoint falls after everything already emitted.
    fn accept(&mut self, spoken: impl Iterator<Item = Spoken>) -> Vec<Segment> {
        let mut new = Vec::new();
        for (start, end, text) in spoken {
            if (start + end) / 2.0 <= self.emitted_until {
                continue;
            }
            let mut segment = Segment::new(text, start, end);
            segment.speaker = self.speaker.clone();
            self.transcript.push(segment.clone());
            new.push(segment);
            self.emitted_until = self.emitted_until.max(end);
        }
        new
    }
}
