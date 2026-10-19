//! MeetButler's transcription engine.
//!
//! Everything here runs on the user's machine. Audio lives on disk (`pcm_store`),
//! Whisper only ever sees a bounded tail window (`incremental`), and speech is cut into
//! spans by our own Silero rule (`spans`) before any decoding, the same way live and
//! after the meeting.

pub mod spans;

/// Every sample in the engine is 16 kHz mono.
pub const SAMPLE_RATE: usize = 16_000;
pub mod blocklist;
