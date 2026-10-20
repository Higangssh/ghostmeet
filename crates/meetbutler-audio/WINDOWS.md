# Windows system audio (WASAPI loopback): plan

Status (2026-10-20): implemented as below, linted for the Windows target from macOS (CI
builds it on `windows-latest` once pushed), **never run on a Windows machine**.
`system::open` opens loopback on the default output device
with a silent keep-alive stream, under `supervise` (rebuilt when the device changes),
through `gap::GapFiller`. What is tested without Windows: the gap filler, the supervisor,
and which cpal errors end a stream.

## What we need

The "them" channel on Windows: what the other side of a call sounds like, as 16 kHz mono
f32 into the same `AudioSink` the macOS path feeds, with the same guarantees the engine
relies on:

- **The channel's sample count tracks wall-clock time.** Segment times come from the
  sample cursor (CLAUDE.md, "absolute timestamps"), and "me" and "them" must line up for
  the transcript order and, later, diarization. A channel that skips silence drifts.
- No virtual audio driver, no extra install, no port (the local-only promise).
- Starts only on Start; a failure leaves the meeting recording from the mic.

## How: cpal's loopback

cpal 0.18.2 (already a dependency, Apache-2.0) does loopback: building an *input* stream
on an *output* device sets `AUDCLNT_STREAMFLAGS_LOOPBACK` (`host/wasapi/device.rs`, in
`build_input_stream_raw_inner`). So:

```rust
let device = host.default_output_device()?;          // the render endpoint
let config = device.default_output_config()?;        // default_input_config() fails on it
device.build_input_stream::<T, _, _>(config.into(), on_data, on_error, None)?
```

then the same `Conformer` as the mic (mix format is usually 48 kHz float, 2 channels,
sometimes 6 or 8; the conformer averages any channel count).

The default output device is cpal's `DeviceHandle::DefaultOutput`, opened through
`ActivateAudioInterfaceAsync` on the default-render interface, with a
`DefaultDeviceMonitor` that reports a default-device change to the error callback.

## What cpal does not do for us

1. **Silence is no data.** In loopback, WASAPI delivers packets only while something is
   playing on the endpoint; with nothing playing, `GetNextPacketSize` stays 0 and the
   callback does not run. Left alone, the "them" channel would be shorter than the
   meeting and every segment after a quiet stretch would be stamped early.
   Two remedies, both planned:
   - **Keep the endpoint running:** open a render stream on the same default device that
     plays zeros for as long as the capture runs (what OBS and others do). Zeros mixed in
     change nothing audible, and loopback then delivers continuously.
   - **Fill gaps from timestamps anyway:** cpal gives each packet its capture time
     (`InputCallbackInfo::timestamp().capture`, from WASAPI's QPC position). A
     `GapFiller` in front of the conformer compares that with the samples delivered so far
     and inserts zeros for a gap longer than a threshold (say 20 ms). This is pure logic,
     the same on every platform, and covers the keep-alive failing or a device glitch.
2. **`AUDCLNT_BUFFERFLAGS_SILENT` is ignored.** WASAPI marks some packets as silence and
   says to ignore their contents; cpal passes the buffer through as samples. Usually they
   are zeros, but that is not promised. Our callback cannot see the flag. If real
   machines show noise there, the options are an upstream cpal fix (expose or honour the
   flag) or a WASAPI path of our own (below).
3. **Device changes end the stream.** Plugging a headset in, or Bluetooth switching to
   its call profile, changes the default device; cpal reports it through the error
   callback (`StreamInvalidated` for a new default, `DeviceNotAvailable` when none is
   left). We rebuild the stream on the new default, and the gap filler covers the time
   in between.
4. **No exclusion of our own sound.** Loopback captures everything the endpoint plays.
   MeetButler plays nothing, so this is fine today; it matters if the app ever plays
   audio (a "listen back" button). Process loopback (`AUDIOCLIENT_ACTIVATION_TYPE_
   PROCESS_LOOPBACK`, Windows 10 2004+) can include or exclude a process tree, but cpal
   does not expose it.

If 2 or 4 become real problems, the fallback is the `wasapi` crate (MIT; it wraps the
same API with packet flags and process loopback). To be checked then: its licence file,
its maintenance, and that it brings no network crate (the security test covers the
Windows targets already).

## What a Windows meeting will look like that a Mac one does not

- **No permission prompt** for loopback (Windows has none for output capture). The
  microphone is governed by Settings > Privacy > Microphone > "Let desktop apps access
  your microphone"; a denied mic surfaces from cpal as an error we map to
  `PermissionDenied`, so the window can say where to allow it.
- **Speakers, no headset:** the mic hears the other side too, so "them" appears in both
  channels. Echo handling is 0.3 (requirements roadmap); until then the manual check notes
  how often it happens.
- **Protected content** (some players use DRM paths) records as silence. Calls do not.
- **Exclusive-mode apps** take the endpoint away from the shared mix; loopback then gets
  nothing. Rare for call apps.

## Tests

Default tests stay model-free and device-free (CLAUDE.md), and run on macOS and in CI:

- `GapFiller` (pure): packets with capture times in, a continuous stream out. Cases: no
  gap; a gap of 3 s becomes 3 s of zeros; jitter under the threshold adds nothing; a
  packet stamped earlier than expected (overlap) is trimmed, not doubled; hours of
  packets do not drift (positions from counts, like the resampler); the first packet sets
  the origin.
- Loopback setup as a function over a small trait (`default_output_config`,
  `build_input_stream`), with a fake device: it asks for the *output* config, picks the
  sample format branch, and maps errors (no output device → `NoDevice`; a device-change
  error → rebuild requested).
- The existing `fallback.rs` already checks that a failed system source leaves the mic
  recording.
- Windows-only code compiles in CI: `cargo check --target x86_64-pc-windows-msvc
  -p meetbutler-audio` (and clippy) on every push, so it cannot rot unseen.

On a Windows machine, manual (to add to `app/README.md` when it lands): a call with a
headset (both channels, no echo); a quiet minute then speech (the "them" channel's
length matches the mic's to within a frame, and later segments are not early); headset
plugged in mid-meeting (recording continues, a short gap of zeros); speakers without a
headset (note the echo); a denied microphone (system audio alone, the window says why).

## What can be done without a Windows machine

- All of the default tests above, including the gap filler and the fake-device setup.
- `cargo check`/`clippy` for `x86_64-pc-windows-msvc` from this Mac. That needs the Rust
  target's standard library (`rustup target add x86_64-pc-windows-msvc`, a download of a
  Rust toolchain component; nothing is linked, so no Windows SDK is needed). A full build
  of the app for Windows from macOS would need the MSVC libraries (`cargo xwin` fetches
  them under Microsoft's licence); not proposed.
- A GitHub Actions `windows-latest` job: build and run the default tests on real Windows.
  Its runners have no audio endpoint, so loopback itself cannot be tested there without
  a third-party virtual audio driver, which we do not want in the pipeline.
- A check that is useful now, on the Mac: the same silence question for ScreenCaptureKit.
  Record a minute with nothing playing and compare `system.pcm` with `mic.pcm` in a test
  data folder; if ScreenCaptureKit also skips silence, the gap filler is needed on macOS
  too, and sooner.

What only a Windows machine can tell: whether silent packets carry noise, how device
changes behave on real Bluetooth headsets, the mix formats seen in the wild, and the
caption delay on a 6-core Windows laptop (requirements §7).

## Order of work

1. Done: `GapFiller` with its tests, in front of the mic and macOS system audio too; it
   logs every fill of a second or more and the total at the end.
2. Done: Windows `system::open` (loopback, the silent keep-alive render stream, rebuilt
   on a device change through `supervise`).
3. Done: CI builds and tests on `windows-latest` and lints the audio crate there.
4. Still to do, on a Windows machine: the manual checks in `app/README.md`, then decide
   on the `wasapi` crate only if silent-packet noise or self-capture shows up.

Not done: the fake-device test of the loopback setup itself. cpal's `Device` is a
concrete type, so testing that glue without Windows would mean wrapping cpal in a trait
for a dozen lines of calls; CI compiling it for Windows plus the manual checks were
judged enough. The microphone follows its device too, on every platform: rebuilt on the
errors above, and also when the default input changes without one (on macOS an input
stream stays on the device it opened, so the supervisor asks about once a second).
