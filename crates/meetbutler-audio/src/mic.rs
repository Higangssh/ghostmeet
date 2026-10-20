//! The default microphone, through cpal. Conformed to 16 kHz mono, then kept on the clock
//! by the capture times cpal gives each packet (`gap`).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use crate::conform::Conformer;
use crate::gap::GapFiller;
use crate::{on_own_thread, AudioSink, Capture, CaptureError};

/// Open the default input device. Capture stops when the returned handle is stopped or
/// dropped.
pub fn open(sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    on_own_thread("mic", move || build(sink))
}

fn build(sink: AudioSink) -> Result<cpal::Stream, CaptureError> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or(CaptureError::NoDevice)?;
    let supported = device.default_input_config().map_err(backend)?;
    let (rate, channels, format) = (
        supported.sample_rate(),
        supported.channels(),
        supported.sample_format(),
    );
    log::info!("mic: {rate} Hz, {channels} ch, {format:?}");
    let config: cpal::StreamConfig = supported.into();

    let stream = match format {
        SampleFormat::F32 => stream::<f32>(&device, config, rate, channels, sink),
        SampleFormat::I16 => stream::<i16>(&device, config, rate, channels, sink),
        SampleFormat::I32 => stream::<i32>(&device, config, rate, channels, sink),
        SampleFormat::U16 => stream::<u16>(&device, config, rate, channels, sink),
        other => {
            return Err(CaptureError::Backend(format!(
                "unsupported sample format {other:?}"
            )))
        }
    }?;
    stream.play().map_err(backend)?;
    Ok(stream)
}

fn stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    rate: u32,
    channels: u16,
    mut sink: AudioSink,
) -> Result<cpal::Stream, CaptureError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let mut conformer = Conformer::new(rate, channels);
    let mut gaps = GapFiller::new();
    let (mut floats, mut conformed, mut out) = (Vec::new(), Vec::new(), Vec::new());
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data: &[T], info: &cpal::InputCallbackInfo| {
                floats.clear();
                floats.extend(data.iter().map(|&s| s.to_sample::<f32>()));
                conformed.clear();
                conformer.process(&floats, &mut conformed);
                out.clear();
                gaps.push(
                    // the backend's clock in seconds; the filler only takes differences
                    info.timestamp().capture.as_nanos() as f64 * 1e-9,
                    &conformed,
                    &mut out,
                );
                if !out.is_empty() {
                    sink(&out);
                }
            },
            |err| log::warn!("mic stream: {err}"),
            None,
        )
        .map_err(backend)
}

fn backend(err: cpal::Error) -> CaptureError {
    let text = err.to_string();
    if format!("{err:?}").contains("PermissionDenied") {
        CaptureError::PermissionDenied(format!("microphone ({text})"))
    } else {
        CaptureError::Backend(text)
    }
}
