//! Device audio to 16 kHz mono, chunk by chunk.
//!
//! Runs inside capture callbacks on whatever chunk sizes the device hands over, so the
//! result does not depend on where chunks are cut, and output positions are computed
//! from a sample count rather than accumulated, so hours of audio do not drift.

use std::f64::consts::PI;

use crate::TARGET_RATE;

/// Windowed-sinc resampler with a low-pass at 90% of the lower Nyquist frequency.
#[derive(Debug, Clone)]
pub struct Resampler {
    /// Input samples per output sample.
    ratio: f64,
    /// Low-pass cutoff in cycles per input sample.
    cutoff: f64,
    /// Kernel half-width in input samples.
    half: i64,
    /// Input samples from absolute index `start` on; earlier ones are no longer needed.
    buf: Vec<f32>,
    start: i64,
    /// Input samples seen so far.
    seen: i64,
    /// Next output sample to produce.
    next: u64,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        let ratio = from as f64 / to as f64;
        Self {
            ratio,
            cutoff: 0.5 * ratio.recip().min(1.0) * 0.9,
            half: (8.0 * ratio.max(1.0)).ceil() as i64,
            buf: Vec::new(),
            start: 0,
            seen: 0,
            next: 0,
        }
    }

    fn passthrough(&self) -> bool {
        self.ratio == 1.0
    }

    /// Resample `input`, appending what can be produced so far to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough() {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        self.seen += input.len() as i64;
        self.produce(self.seen, out);
        self.trim();
    }

    /// No more input: produce the tail, treating audio past the end as silence.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if self.passthrough() {
            return;
        }
        let total = (self.seen as f64 / self.ratio).ceil() as u64;
        let pad = self.half as usize + 1;
        self.buf.extend(std::iter::repeat_n(0.0, pad));
        let end = self.seen + pad as i64;
        while self.next < total {
            if !self.produce_one(end, out) {
                break;
            }
        }
        self.buf.truncate(self.buf.len() - pad);
        self.trim();
    }

    /// Produce every output whose kernel lies within the first `end` input samples.
    fn produce(&mut self, end: i64, out: &mut Vec<f32>) {
        while self.produce_one(end, out) {}
    }

    fn produce_one(&mut self, end: i64, out: &mut Vec<f32>) -> bool {
        let x = self.next as f64 * self.ratio;
        let center = x.floor() as i64;
        if center + self.half >= end {
            return false;
        }
        let mut sum = 0.0f64;
        for k in (center - self.half + 1)..=(center + self.half) {
            if k < 0 {
                continue; // before the first sample: silence
            }
            let s = self.buf[(k - self.start) as usize] as f64;
            sum += s * self.kernel(x - k as f64);
        }
        out.push(sum as f32);
        self.next += 1;
        true
    }

    fn kernel(&self, t: f64) -> f64 {
        let half = self.half as f64;
        if t.abs() >= half {
            return 0.0;
        }
        let fc2 = 2.0 * self.cutoff;
        let arg = PI * fc2 * t;
        let sinc = if arg == 0.0 { 1.0 } else { arg.sin() / arg };
        let window = 0.5 * (1.0 + (PI * t / half).cos()); // Hann
        fc2 * sinc * window
    }

    /// Drop input no future output can reach.
    fn trim(&mut self) {
        let needed = (self.next as f64 * self.ratio).floor() as i64 - self.half + 1;
        let drop = (needed - self.start).clamp(0, self.buf.len() as i64);
        if drop > 0 {
            self.buf.drain(..drop as usize);
            self.start += drop;
        }
    }
}

/// Interleaved device frames at any rate and channel count, to 16 kHz mono.
#[derive(Debug, Clone)]
pub struct Conformer {
    channels: usize,
    partial: Vec<f32>,
    mono: Vec<f32>,
    resampler: Resampler,
}

impl Conformer {
    pub fn new(rate: u32, channels: u16) -> Self {
        Self {
            channels: channels.max(1) as usize,
            partial: Vec::new(),
            mono: Vec::new(),
            resampler: Resampler::new(rate, TARGET_RATE),
        }
    }

    /// Convert `interleaved`, appending 16 kHz mono to `out`. A frame cut off at the end
    /// of a chunk is completed by the next one.
    pub fn process(&mut self, interleaved: &[f32], out: &mut Vec<f32>) {
        self.mono.clear();
        let mut rest = interleaved;
        if !self.partial.is_empty() {
            let need = (self.channels - self.partial.len()).min(rest.len());
            self.partial.extend_from_slice(&rest[..need]);
            rest = &rest[need..];
            if self.partial.len() == self.channels {
                self.mono
                    .push(self.partial.iter().sum::<f32>() / self.channels as f32);
                self.partial.clear();
            }
        }
        let whole = rest.len() / self.channels * self.channels;
        self.mono.extend(
            rest[..whole]
                .chunks_exact(self.channels)
                .map(|f| f.iter().sum::<f32>() / self.channels as f32),
        );
        self.partial.extend_from_slice(&rest[whole..]);
        self.resampler.process(&self.mono, out);
    }

    pub fn flush(&mut self, out: &mut Vec<f32>) {
        self.resampler.flush(out);
    }
}
