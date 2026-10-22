//! Kaldi fbank as kaldi-native-fbank computes it with sherpa-onnx's defaults: 25 ms / 10 ms
//! frames, no snip edges (reflected), DC removed, pre-emphasis 0.97, Povey window, 512-point
//! power spectrum, 80 mel bins from 20 Hz to Nyquist - 400 Hz, natural log, no dither.
use rustfft::{num_complex::Complex32, FftPlanner};

const RATE: f32 = 16000.0;
const LEN: usize = 400;
const SHIFT: usize = 160;
const NFFT: usize = 512;
pub const BINS: usize = 80;

fn mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

pub struct Fbank {
    window: Vec<f32>,
    banks: Vec<(usize, Vec<f32>)>, // first fft bin, weights
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Fbank {
    pub fn new() -> Self {
        let a = 2.0 * std::f64::consts::PI / (LEN as f64 - 1.0);
        let window = (0..LEN)
            .map(|i| (0.5 - 0.5 * (a * i as f64).cos()).powf(0.85) as f32)
            .collect();
        let (low, high) = (20.0f32, RATE / 2.0 - 400.0);
        let (ml, mh) = (mel(low), mel(high));
        let delta = (mh - ml) / (BINS as f32 + 1.0);
        let width = RATE / NFFT as f32;
        let banks = (0..BINS)
            .map(|b| {
                let (left, center, right) = (ml + b as f32 * delta, ml + (b as f32 + 1.0) * delta, ml + (b as f32 + 2.0) * delta);
                let w: Vec<f32> = (0..NFFT / 2)
                    .map(|k| {
                        let m = mel(k as f32 * width);
                        if m > left && m < right {
                            if m <= center { (m - left) / (center - left) } else { (right - m) / (right - center) }
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let first = w.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = w.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                (first, w[first..=last].to_vec())
            })
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(NFFT);
        Fbank { window, banks, fft }
    }

    /// `samples` in [-1, 1]; one row of BINS per frame.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let n = samples.len();
        let frames = (n + SHIFT / 2) / SHIFT;
        let mut out = Vec::with_capacity(frames * BINS);
        let mut frame = vec![0f32; LEN];
        let mut buf = vec![Complex32::new(0.0, 0.0); NFFT];
        for i in 0..frames {
            let start = (i * SHIFT + SHIFT / 2) as i64 - (LEN / 2) as i64;
            for (j, f) in frame.iter_mut().enumerate() {
                let mut s = start + j as i64;
                if s < 0 {
                    s = -s - 1;
                }
                if s >= n as i64 {
                    s = 2 * n as i64 - s - 1;
                }
                *f = samples[s as usize];
            }
            let mean = frame.iter().sum::<f32>() / LEN as f32;
            frame.iter_mut().for_each(|v| *v -= mean);
            for j in (1..LEN).rev() {
                frame[j] -= 0.97 * frame[j - 1];
            }
            frame[0] -= 0.97 * frame[0];
            for (j, c) in buf.iter_mut().enumerate() {
                *c = Complex32::new(if j < LEN { frame[j] * self.window[j] } else { 0.0 }, 0.0);
            }
            self.fft.process(&mut buf);
            let power: Vec<f32> = buf[..NFFT / 2].iter().map(|c| c.norm_sqr()).collect();
            for (first, w) in &self.banks {
                let e: f32 = w.iter().zip(&power[*first..]).map(|(a, b)| a * b).sum();
                out.push(e.max(f32::EPSILON).ln());
            }
        }
        out
    }
}
