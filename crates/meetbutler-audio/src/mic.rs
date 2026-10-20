//! The default microphone, through cpal. Conformed to 16 kHz mono, kept on the clock by
//! the capture times cpal gives each packet (`gap`), and built again when the default
//! microphone changes or goes away (`supervise`).
//!
//! [`conformed_input`] is the cpal input stream every cpal source uses: the mic here, and
//! on Windows the loopback of the default output device (`system`).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::conform::Conformer;
use crate::gap::GapFiller;
use crate::supervise::{supervised_watching, Rebuild};
use crate::{AudioSink, Capture, CaptureError};

/// Between tries to open the new default microphone after a change.
const RETRY: Duration = Duration::from_secs(1);

/// What outlives a rebuild: the channel's clock and where its audio goes.
struct Shared {
    sink: AudioSink,
    gaps: GapFiller,
    out: Vec<f32>,
}

/// A stream and the device it was opened on.
struct Running {
    _stream: cpal::Stream,
    device: Option<cpal::DeviceId>,
}

/// Open the default input device, and follow it: when the default microphone changes (a
/// headset plugged in) or goes away, the stream is built again on the new one, and the
/// time in between is silence on the channel. Capture stops when the returned handle is
/// stopped or dropped.
pub fn open(sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    let shared = Arc::new(Mutex::new(Shared {
        sink,
        gaps: GapFiller::named("mic"),
        out: Vec::new(),
    }));
    supervised_watching(
        "mic",
        RETRY,
        move |rebuild| build(&shared, rebuild),
        // an input stream stays on its device when the default changes, and says nothing
        |running: &Running| running.device.is_some() && default_input_id() != running.device,
    )
}

fn default_input_id() -> Option<cpal::DeviceId> {
    cpal::default_host()
        .default_input_device()
        .and_then(|d| d.id().ok())
}

fn build(shared: &Arc<Mutex<Shared>>, rebuild: &Rebuild) -> Result<Running, CaptureError> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or(CaptureError::NoDevice)?;
    let supported = device.default_input_config().map_err(backend)?;
    log::info!(
        "mic: {} Hz, {} ch, {:?}",
        supported.sample_rate(),
        supported.channels(),
        supported.sample_format()
    );
    let shared = shared.clone();
    let rebuild = rebuild.clone();
    let stream = conformed_input(
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
        move |err| {
            if ends_stream(err.kind()) {
                rebuild.request();
            } else {
                log::warn!("mic stream: {err}");
            }
        },
    )?;
    stream.play().map_err(backend)?;
    Ok(Running {
        _stream: stream,
        device: device.id().ok(),
    })
}

/// An input stream on `device` in `supported`'s format, handing `on_audio` each packet as
/// 16 kHz mono with its capture time in seconds (the backend's clock; only differences
/// mean anything). On a Windows output device this is loopback.
pub(crate) fn conformed_input(
    device: &cpal::Device,
    supported: cpal::SupportedStreamConfig,
    on_audio: impl FnMut(f64, &[f32]) + Send + 'static,
    on_error: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<cpal::Stream, CaptureError> {
    let (rate, channels) = (supported.sample_rate(), supported.channels());
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    match format {
        SampleFormat::F32 => stream::<f32>(device, config, rate, channels, on_audio, on_error),
        SampleFormat::I16 => stream::<i16>(device, config, rate, channels, on_audio, on_error),
        SampleFormat::I32 => stream::<i32>(device, config, rate, channels, on_audio, on_error),
        SampleFormat::U16 => stream::<u16>(device, config, rate, channels, on_audio, on_error),
        other => Err(CaptureError::Backend(format!(
            "unsupported sample format {other:?}"
        ))),
    }
}

fn stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    rate: u32,
    channels: u16,
    mut on_audio: impl FnMut(f64, &[f32]) + Send + 'static,
    on_error: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<cpal::Stream, CaptureError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let mut conformer = Conformer::new(rate, channels);
    let (mut floats, mut conformed) = (Vec::new(), Vec::new());
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data: &[T], info: &cpal::InputCallbackInfo| {
                floats.clear();
                floats.extend(data.iter().map(|&s| s.to_sample::<f32>()));
                conformed.clear();
                conformer.process(&floats, &mut conformed);
                on_audio(
                    info.timestamp().capture.as_nanos() as f64 * 1e-9,
                    &conformed,
                );
            },
            on_error,
            None,
        )
        .map_err(backend)
}

pub(crate) fn backend(err: cpal::Error) -> CaptureError {
    let text = err.to_string();
    if format!("{err:?}").contains("PermissionDenied") {
        CaptureError::PermissionDenied(format!("microphone ({text})"))
    } else {
        CaptureError::Backend(text)
    }
}

/// Errors after which a stream does not come back by itself: it has to be built again
/// (the default device changed under it, or went away). Others pass with a warning.
pub fn ends_stream(kind: cpal::ErrorKind) -> bool {
    matches!(
        kind,
        cpal::ErrorKind::StreamInvalidated | cpal::ErrorKind::DeviceNotAvailable
    )
}
