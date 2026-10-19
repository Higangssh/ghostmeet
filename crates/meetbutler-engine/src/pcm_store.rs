//! Append-only PCM buffer backed by a file.
//!
//! Audio accumulates on disk so a session's memory footprint stays flat regardless of
//! length — transcription only ever reads a short tail window back into RAM.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::{AudioSource, SAMPLE_RATE};

const BYTES_PER_SAMPLE: u64 = 2; // s16le
const INT16_FULL_SCALE: f32 = 32768.0;

/// 16 kHz mono s16le PCM on disk, appended incrementally and read by window.
///
/// Shared between threads: capture appends while the transcription worker reads.
#[derive(Debug)]
pub struct PcmStore {
    path: PathBuf,
    write: Mutex<Option<File>>,
    bytes_written: AtomicU64,
}

impl PcmStore {
    /// Open `path` for appending, creating it and its directory if needed. Audio already
    /// in the file is kept.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let size = file.metadata()?.len();
        Ok(Self {
            path,
            write: Mutex::new(Some(file)),
            bytes_written: AtomicU64::new(size),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn duration(&self) -> f64 {
        self.total_samples() as f64 / SAMPLE_RATE as f64
    }

    /// Append raw s16le bytes. Bytes arriving after `close` are dropped: a capture thread
    /// can still be draining when the session tears down, and crashing on that is wrong.
    pub fn append(&self, pcm: &[u8]) -> io::Result<()> {
        if pcm.is_empty() {
            return Ok(());
        }
        let mut guard = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let Some(file) = guard.as_mut() else {
            return Ok(());
        };
        file.write_all(pcm)?;
        file.flush()?;
        // published after the write, so a reader never asks for bytes not yet on disk
        self.bytes_written
            .fetch_add(pcm.len() as u64, Ordering::Release);
        Ok(())
    }

    /// Append float samples in [-1, 1], clipped and stored as s16le.
    pub fn append_f32(&self, samples: &[f32]) -> io::Result<()> {
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|&s| ((s * INT16_FULL_SCALE).clamp(-32768.0, 32767.0) as i16).to_le_bytes())
            .collect();
        self.append(&bytes)
    }

    pub fn close(&self) {
        let mut guard = self.write.lock().unwrap_or_else(|e| e.into_inner());
        guard.take();
    }
}

impl AudioSource for PcmStore {
    /// Whole samples available. A dangling odd byte is not yet a sample.
    fn total_samples(&self) -> usize {
        (self.bytes_written.load(Ordering::Acquire) / BYTES_PER_SAMPLE) as usize
    }

    /// Samples [start, end) as f32 in [-1, 1), clamped to what exists.
    fn read(&self, start: usize, end: Option<usize>) -> io::Result<Vec<f32>> {
        let total = self.total_samples();
        let start = start.min(total);
        let end = end.map_or(total, |e| e.clamp(start, total));
        if end <= start {
            return Ok(Vec::new());
        }

        let mut raw = vec![0u8; (end - start) * BYTES_PER_SAMPLE as usize];
        let mut fh = File::open(&self.path)?;
        fh.seek(SeekFrom::Start(start as u64 * BYTES_PER_SAMPLE))?;
        fh.read_exact(&mut raw)?;

        Ok(raw
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / INT16_FULL_SCALE)
            .collect())
    }
}
