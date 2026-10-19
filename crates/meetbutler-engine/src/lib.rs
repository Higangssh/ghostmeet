//! MeetButler's transcription engine.
//!
//! Everything here runs on the user's machine. Audio lives on disk (`pcm_store`),
//! Whisper only ever sees a bounded tail window (`incremental`), and speech is cut into
//! spans by our own Silero rule (`spans`) before any decoding, the same way live and
//! after the meeting.

use std::io;

pub mod blocklist;
pub mod incremental;
pub mod model_cache;
pub mod pcm_store;
pub mod pipeline;
pub mod recorder;
#[cfg(feature = "silero")]
pub mod silero;
pub mod span_decoder;
pub mod spans;
pub mod store;
pub mod vocabulary;
#[cfg(feature = "whisper")]
pub mod whisper;

/// Every sample in the engine is 16 kHz mono.
pub const SAMPLE_RATE: usize = 16_000;

/// A growing recording that can be read back a window at a time.
pub trait AudioSource {
    /// Whole samples available so far.
    fn total_samples(&self) -> usize;
    /// Samples `[start, end)` as f32, clamped to what exists; `None` reads to the end.
    fn read(&self, start: usize, end: Option<usize>) -> io::Result<Vec<f32>>;
}

impl<T: AudioSource + ?Sized> AudioSource for std::sync::Arc<T> {
    fn total_samples(&self) -> usize {
        (**self).total_samples()
    }
    fn read(&self, start: usize, end: Option<usize>) -> io::Result<Vec<f32>> {
        (**self).read(start, end)
    }
}

/// Why a decoding pass failed. A failed pass is logged and skipped, never fatal.
pub type TranscribeError = Box<dyn std::error::Error + Send + Sync>;

/// Turns one bounded window of audio into segments timed relative to that window.
pub trait Transcribe {
    fn transcribe(
        &mut self,
        audio: &[f32],
    ) -> Result<Vec<incremental::RawSegment>, TranscribeError>;
}

impl<F> Transcribe for F
where
    F: FnMut(&[f32]) -> Result<Vec<incremental::RawSegment>, TranscribeError>,
{
    fn transcribe(
        &mut self,
        audio: &[f32],
    ) -> Result<Vec<incremental::RawSegment>, TranscribeError> {
        self(audio)
    }
}
