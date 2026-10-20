//! System audio: what the other side of a call sounds like.
//!
//! macOS: ScreenCaptureKit (13+), which needs the screen-recording permission; without
//! it opening fails with [`CaptureError::PermissionDenied`] and the app records the mic
//! alone. The app's own sound is excluded. No virtual audio device is involved.
//!
//! Windows: WASAPI loopback of the default output device, through cpal (an input stream
//! on an output device), with a render stream of silence beside it so loopback keeps
//! delivering while nothing plays, and the gap filler for what it still misses. A change
//! of default device ends both; they are built again on the new one (`supervise`).
//! Untested on a real Windows machine yet; the plan and the open questions are in
//! `WINDOWS.md`.

use crate::{AudioSink, Capture, CaptureError};

#[cfg(target_os = "macos")]
pub fn open(sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    crate::on_own_thread("system", move || macos::build(sink))
}

#[cfg(target_os = "windows")]
pub fn open(sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    windows::open(sink)
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn open(_sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    Err(CaptureError::Unsupported("system audio on this platform"))
}

#[cfg(target_os = "macos")]
mod macos {
    use std::sync::Mutex;

    use screencapturekit::prelude::*;

    use crate::conform::Conformer;
    use crate::gap::GapFiller;
    use crate::{AudioSink, CaptureError};

    /// ScreenCaptureKit's own rate; we conform it like any device.
    const RATE: i32 = 48_000;

    struct State {
        sink: AudioSink,
        conformer: Conformer,
        gaps: GapFiller,
        mono: Vec<f32>,
        conformed: Vec<f32>,
        out: Vec<f32>,
    }

    struct Output {
        state: Mutex<State>,
    }

    impl SCStreamOutputTrait for Output {
        fn did_output_sample_buffer(&self, sample: CMSampleBuffer, kind: SCStreamOutputType) {
            if !matches!(kind, SCStreamOutputType::Audio) {
                return;
            }
            let Ok(list) = sample.audio_buffer_list() else {
                return;
            };
            // whether ScreenCaptureKit sends silence or nothing while nothing plays is not
            // known yet (app/README manual checks); the gap filler covers the latter
            let at = sample.presentation_timestamp().as_seconds();
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let State {
                sink,
                conformer,
                gaps,
                mono,
                conformed,
                out,
            } = &mut *state;
            // Float32, one buffer per channel (non-interleaved); average them
            mono.clear();
            let n = list.num_buffers();
            for i in 0..n {
                let Some(buffer) = list.get(i) else { continue };
                let samples = buffer
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) / n as f32);
                if i == 0 {
                    mono.extend(samples);
                } else {
                    mono.iter_mut().zip(samples).for_each(|(m, s)| *m += s);
                }
            }
            conformed.clear();
            conformer.process(mono, conformed);
            out.clear();
            match at {
                Some(at) => gaps.push(at, conformed, out),
                None => out.extend_from_slice(conformed), // no time: pass it on as it is
            }
            if !out.is_empty() {
                sink(out);
            }
        }
    }

    pub(super) struct Running(SCStream);

    impl Drop for Running {
        fn drop(&mut self) {
            if let Err(err) = self.0.stop_capture() {
                log::warn!("stopping system audio: {err}");
            }
        }
    }

    pub(super) fn build(sink: AudioSink) -> Result<Running, CaptureError> {
        let content = SCShareableContent::get().map_err(|e| {
            CaptureError::PermissionDenied(format!("screen recording, for system audio ({e})"))
        })?;
        let display = content
            .displays()
            .into_iter()
            .next()
            .ok_or_else(|| CaptureError::Backend("no display".into()))?;
        let filter = SCContentFilter::create()
            .with_display(&display)
            .with_excluding_windows(&[])
            .build()
            .map_err(|e| CaptureError::Backend(e.to_string()))?;
        // the video is required by the API and thrown away, so keep it tiny
        let config = SCStreamConfiguration::new()
            .with_width(2)
            .with_height(2)
            .with_captures_audio(true)
            .with_sample_rate(RATE)
            .with_channel_count(1)
            .with_excludes_current_process_audio(true);

        let mut stream =
            SCStream::new(&filter, &config).map_err(|e| CaptureError::Backend(e.to_string()))?;
        let output = Output {
            state: Mutex::new(State {
                sink,
                conformer: Conformer::new(RATE as u32, 1),
                gaps: GapFiller::named("system"),
                mono: Vec::new(),
                conformed: Vec::new(),
                out: Vec::new(),
            }),
        };
        stream
            .add_output_handler(output, SCStreamOutputType::Audio)
            .map_err(|e| CaptureError::Backend(e.to_string()))?;
        stream
            .start_capture()
            .map_err(|e| CaptureError::PermissionDenied(format!("screen recording ({e})")))?;
        Ok(Running(stream))
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{SampleFormat, SizedSample};

    use crate::gap::GapFiller;
    use crate::mic::{backend, conformed_input, ends_stream};
    use crate::supervise::{supervised, Rebuild};
    use crate::{AudioSink, Capture, CaptureError};

    /// Between tries to open the new default device after a change.
    const RETRY: Duration = Duration::from_secs(1);

    /// What outlives a rebuild: the channel's clock and where its audio goes.
    struct Shared {
        sink: AudioSink,
        gaps: GapFiller,
        out: Vec<f32>,
    }

    struct Running {
        _loopback: cpal::Stream,
        _keepalive: Option<cpal::Stream>,
    }

    pub(super) fn open(sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
        let shared = Arc::new(Mutex::new(Shared {
            sink,
            gaps: GapFiller::named("system"),
            out: Vec::new(),
        }));
        supervised("system", RETRY, move |rebuild| build(&shared, rebuild))
    }

    fn on_error(rebuild: &Rebuild, what: &'static str) -> impl FnMut(cpal::Error) + Send + 'static {
        let rebuild = rebuild.clone();
        move |err| {
            if ends_stream(err.kind()) {
                rebuild.request();
            } else {
                log::warn!("system audio {what}: {err}");
            }
        }
    }

    fn build(shared: &Arc<Mutex<Shared>>, rebuild: &Rebuild) -> Result<Running, CaptureError> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or(CaptureError::NoDevice)?;
        // loopback takes the output side's format; default_input_config fails on it
        let supported = device.default_output_config().map_err(backend)?;
        log::info!(
            "system: loopback at {} Hz, {} ch, {:?}",
            supported.sample_rate(),
            supported.channels(),
            supported.sample_format()
        );

        // silence played into the same endpoint keeps loopback delivering while nothing
        // else plays; without it the gap filler still keeps the clock
        let keepalive = match silence(&device, supported, on_error(rebuild, "keep-alive")) {
            Ok(s) => Some(s),
            Err(err) => {
                log::warn!("system: no keep-alive stream ({err}); gaps will be filled");
                None
            }
        };

        let shared = shared.clone();
        let loopback = conformed_input(
            &device,
            supported,
            move |at, audio| {
                let mut s = shared.lock().unwrap_or_else(|e| e.into_inner());
                let Shared { sink, gaps, out } = &mut *s;
                out.clear();
                gaps.push(at, audio, out);
                if !out.is_empty() {
                    sink(out);
                }
            },
            on_error(rebuild, "loopback"),
        )?;
        loopback.play().map_err(backend)?;
        Ok(Running {
            _loopback: loopback,
            _keepalive: keepalive,
        })
    }

    fn silence(
        device: &cpal::Device,
        supported: cpal::SupportedStreamConfig,
        on_error: impl FnMut(cpal::Error) + Send + 'static,
    ) -> Result<cpal::Stream, CaptureError> {
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let stream = match format {
            SampleFormat::F32 => zeros::<f32>(device, config, on_error),
            SampleFormat::I16 => zeros::<i16>(device, config, on_error),
            SampleFormat::I32 => zeros::<i32>(device, config, on_error),
            SampleFormat::U16 => zeros::<u16>(device, config, on_error),
            other => {
                return Err(CaptureError::Backend(format!(
                    "unsupported sample format {other:?}"
                )))
            }
        }?;
        stream.play().map_err(backend)?;
        Ok(stream)
    }

    fn zeros<T: SizedSample>(
        device: &cpal::Device,
        config: cpal::StreamConfig,
        on_error: impl FnMut(cpal::Error) + Send + 'static,
    ) -> Result<cpal::Stream, CaptureError> {
        device
            .build_output_stream::<T, _, _>(
                config,
                |data: &mut [T], _| data.fill(T::EQUILIBRIUM),
                on_error,
                None,
            )
            .map_err(backend)
    }
}
