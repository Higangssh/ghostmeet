//! Keep a channel on the clock when its source stops delivering.
//!
//! Segment times come from each channel's sample count, so a channel must grow with
//! wall-clock time whatever its source does. WASAPI loopback delivers nothing while
//! nothing plays (`WINDOWS.md`), and a device change or a stall leaves a hole too. Each
//! packet carries its capture time; when it arrives later than the samples so far
//! account for by more than `FILL_AFTER_SEC`, the difference is filled with silence.
//!
//! Positions come from counts, not accumulated durations, so hours do not drift. A
//! packet stamped early is kept whole: audio is never thrown away to match a clock. A
//! device clock that runs a little fast leaves the channel slightly ahead until the next
//! gap, which it then shortens.

use crate::TARGET_RATE;

/// Holes shorter than this are jitter, not silence.
pub const FILL_AFTER_SEC: f64 = 0.1;

#[derive(Debug, Clone, Default)]
pub struct GapFiller {
    /// Capture time of the first packet, in seconds on the source's clock.
    origin: Option<f64>,
    /// Samples handed on so far, filled silence included.
    emitted: u64,
}

impl GapFiller {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hand on `samples` (16 kHz mono, captured at `time` seconds on the source's clock),
    /// after the silence a hole before them stands for.
    pub fn push(&mut self, time: f64, samples: &[f32], out: &mut Vec<f32>) {
        let origin = *self.origin.get_or_insert(time);
        let rate = TARGET_RATE as f64;
        let gap = time - (origin + self.emitted as f64 / rate);
        if gap > FILL_AFTER_SEC {
            let n = (gap * rate).round() as usize;
            out.resize(out.len() + n, 0.0);
            self.emitted += n as u64;
        }
        out.extend_from_slice(samples);
        self.emitted += samples.len() as u64;
    }
}
