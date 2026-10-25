# REC-4 echo cancellation research (0.3 design memo)

Documents only. No code experiments. Anything not confirmed is marked **estimated**.

## Why it is needed

When a call plays through the speakers, the far side's voice leaks back into the microphone. The app takes a call's mic channel whole as "me" (SPK-1). So the leaked far-side voice becomes "my" speech, and it shows up twice in the transcript.

In the bench's chunked test (`bench/results/RESULTS.md` §9, 6 synthetic calls), a −25 dB leak alone raised call DER sharply.

| Condition | call DER |
|---|---:|
| No leak (CAM++ / ERes2NetV2) | 0.120 / 0.111 |
| −25 dB leak | 0.471 / 0.463 |
| Leak + `drop_echo` | 0.158 / 0.150 |

## What the app already has

The app stores mic and system on disk per channel (`PcmStore`). Both channels' times come from the sample cursor. So **the reference signal (system) is already there, aligned.** That opens an option other than OS echo cancellation.

## macOS: VoiceProcessingIO (VPIO)

- VPIO is an Audio Unit that ties input and output into one unit. It knows what goes out through its own output path and removes that from the mic.
- I found no property for feeding in an outside reference signal. My reading is that the system audio ScreenCaptureKit captures cannot be given to VPIO as its reference.
- WWDC23 "What's new in voice processing" says:
  - Every sound except the app's voice stream (other apps included) is "other audio".
  - It is mixed with voice on output, and by default **its volume is lowered (ducking).**
  - `kAUVoiceIOProperty_OtherAudioDuckingConfiguration` (`voiceProcessingOtherAudioDuckingConfiguration` in AVAudioEngine) picks how much, from min to max.
  - By this description, other apps' sound is also in the output VPIO sees. So the far-side voice from the meeting app may be removed (**estimated**; the docs do not say so).
- **Coexistence problems:**
  - The mic is opened with cpal today (HAL-based input). With VPIO, VPIO holds the same device separately.
  - There is a report on the Apple forums that an app with VPIO on **cut another app's (Zoom etc.) input stream**. I did not find an answer.
  - A recording app must not cut the user's meeting-app mic or turn down the far side's voice. These two are VPIO's biggest risks.
- **Device limits:** an old PJSIP ticket and forum posts report that "VPIO only works on the default or built-in input and output". There are also reports of gain dropping sharply on Bluetooth headsets.
- Conclusion: VPIO is likely to clash with the app's premise, "record quietly next to the meeting app". Using it needs these three checked on a real machine first.
  1. Does input stay up while Zoom or Meet is running at the same time?
  2. With ducking at min, does what the user hears stay unchanged?
  3. Is other apps' sound used as the reference?

## Windows: communications-mode AEC

There are two ways.

1. **A communications stream + the device's AEC effect (APO).**
   - Opening the capture stream in the communications category (`AudioCategory_Communications`) turns on the device driver's communications processing chain. Only when the device has an AEC APO.
   - The reference is the default render device. From Windows build 22621, `IAcousticEchoCancellationControl::SetEchoCancellationRenderEndpoint` picks the reference render device.
   - An endpoint may support AEC but not reference control. Then `GetService` returns `E_NOINTERFACE` (Microsoft docs).
   - Limit 1: whether there is AEC **depends on the device and driver.**
   - Limit 2: once a communications stream opens, Windows' "reduce other sounds when communication activity is detected" (default 80%) may turn down the user's other sounds (**estimated**, needs checking).
2. **Voice Capture DSP (`CLSID_CWMAudioAEC`, since Vista).**
   - A DMO bundling AEC, mic array, noise suppression, AGC and VAD. Each feature can be turned on and off.
   - In source mode the DMO holds the device itself.
   - **In filter mode the app feeds in mic (stream 0) and speaker (stream 1) samples.** The app's WASAPI loopback (the system channel) can be given as the reference.
   - AEC supports a single channel only. Output is 16-bit, 8/11.025/16/22.05 kHz. That fits the app's 16 kHz mono.
   - It is an old API (DMO), so whether it will be kept up is uncertain.

## Alternative: software echo cancellation after the recording (recommended)

Both channels are on disk with their times aligned, so after the meeting, before diarization, the system channel can be removed from the mic channel. It is the same code on every OS. The capture path during the meeting (cpal, ScreenCaptureKit, WASAPI) is not touched. It neither turns down other apps' sound nor cuts other apps' input.

- **Candidate crates:**
  - `webrtc-audio-processing`: WebRTC APM bindings. The original C++ is BSD-3-Clause. The crate's own licence is not checked yet.
  - `sonora`: AEC3 ported to pure Rust, BSD-3-Clause, at 0.1.0, early stage.
  - Both are permissive, but check the licence and dependencies in Cargo.toml before adopting.
- **To solve:**
  - Delay: speaker → air → mic, plus the buffer difference between the two capture paths. AEC3 has a delay estimator.
  - Clock drift: the two devices' sample clocks differ. Over 4 hours it adds up. Silence that GapFiller filled is mixed into the times aligned by the sample cursor.
  - Double talk: when I talk over the far side, my voice must not be removed. The bench test "talking over the far side is still me" guards this.
- **Not applied to captions during the meeting.** Live captions are provisional. In 0.3 we accept that leaked far-side voice shows briefly in "me" captions. Cleaning the mic channel before the final pass also takes the duplicates out of the final transcript.
- It fits the two passes. Duplicates left in the live captions go away when the final pass overwrites the whole meeting's transcript in one transaction. If echo cancellation fails or is cancelled, the final pass runs on the uncleaned mic. That is still no worse than today.
- **`drop_echo` as the floor:**
  - A crude stand-in in the bench. For each 0.1 s frame, if the zero-delay correlation of mic and system is over 0.5, that frame is taken out of "me".
  - On synthetic calls it gave DER 0.46→0.15. It only removes leaks with no delay and no room reverb, so with real speakers it does worse than this.
  - As the fallback when software AEC fails, it can be used in the same shape by adding delay estimation (the peak of the cross-correlation).
  - This fallback does not fix the audio; it only trims "me" spans. Duplicates in the transcript remain.

## To check (experiment order)

1. Add software AEC to the bench. Add delay (10–200 ms) and room reverb to the synthetic leak, and measure call DER, double-talk preservation and 4-hour drift. Public speech only.
2. On a real Mac, record a call over speakers and measure the mic leak level and delay. The audio does not go into the repo; only numbers are recorded.
3. VPIO and Windows communications mode only if 1 fails: check the coexistence problems (other apps' input cut, ducking) on a real machine first.

## Sources

- Apple WWDC23 "What's new in voice processing": https://developer.apple.com/videos/play/wwdc2023/10235/
- Apple forums, VPIO cuts another app's input: https://developer.apple.com/forums/thread/751100
- Apple forums, macOS VPIO device limits and gain: https://developer.apple.com/forums/thread/66953
- Microsoft `IAcousticEchoCancellationControl`: https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nn-audioclient-iacousticechocancellationcontrol
- Microsoft Voice Capture DSP: https://learn.microsoft.com/en-us/windows/win32/medfound/voicecapturedmo
- `sonora` (AEC3 Rust port): https://docs.rs/crate/sonora/0.1.0

## Check experiment: software echo cancellation after the recording (2026-10-22)

**Crates:**
- **`sonora` 0.2.0**: BSD-3-Clause. WebRTC audio processing (AEC3) ported to pure Rust. Its dependencies are all MIT, Apache-2.0, Zlib or BSD-3; nothing is downloaded at build time, and there is no C/C++ build. MSRV 1.91 (our toolchain is 1.98).
- `webrtc-audio-processing` 2.1.0: crates.io shows its licence as "non-standard"; the repo's `COPYING` is BSD-3-Clause. It needs a system library linked dynamically (or, with `bundled`, a C++ build with meson and ninja), and Windows is not documented. It does not fit the app.

**Bench:**
- The run is `bench/run_aec.py`, the leak model `bench/aec_leak.py` (6 tests), echo cancellation `bench/aec/` (outside the app workspace). Results are in `bench/results/aec/`.
- Meetings: the five 30-minute calls of the chunked test. The mic is me alone, system is the far side.
- Leak: system is added to the mic at −25 dB. On top of that: delay (20/80/200 ms), room (RT60 0.3 s, synthetic impulse response), clock drift (50 ppm, 90 ms over 30 minutes), and "real", all three together (80 ms + room + 50 ppm).
- The far side is split as preregistered (ERes2NetV2, chunked); only "me" changes.
- Methods compared:
  - no processing
  - `drop_echo`
  - sonora with the reference as recorded
  - sonora with the reference moved 20 ms earlier (`lead20`)

Mean call DER (collar 0.25, 5 meetings). With no leak it is **0.101**. In brackets: the share of time taken as "me" that is not my speech (0.010 with no leak).

| Leak | No processing | `drop_echo` | sonora | sonora + lead20 |
|---|---:|---:|---:|---:|
| −25 dB, no delay | 0.439 (0.50) | 0.143 (0.15) | 0.198 (0.24) | **0.104** (0.011) |
| Delay 20 ms | 0.438 | 0.428 | 0.103 | **0.104** |
| Delay 80 ms | 0.438 | 0.437 | 0.104 | **0.104** |
| Delay 200 ms | 0.435 | 0.435 | 0.104 | **0.104** |
| Room (no delay) | 0.397 | 0.180 | 0.181 (0.20) | **0.103** (0.011) |
| Drift 50 ppm | 0.437 | 0.431 | 0.105 | **0.104** |
| real (80 ms + room + 50 ppm) | 0.391 (0.47) | 0.390 | 0.103 (0.011) | **0.103** (0.011) |

- **sonora brings every leak with a delay back to the no-leak level** (0.103–0.105, floor 0.101). The same holds with delay, room and drift combined. Per meeting it is within 0.004 of the floor too (a 2-person call: 0.044 against 0.040).
- **With zero delay it fails** (0.198, room 0.181). When the echo arrives at the same moment as its reference or earlier, AEC3 cannot remove it (causality). A real speaker leak always has a delay from buffers and air, but the app stores the two tracks aligned by the sample cursor, so depending on the capture paths' buffer difference it can be close to 0. **Moving the reference 20 ms earlier brings every condition back to the floor.** The tracks are on disk, so this can always be done. The app does this by default.
- **Keeping my speech (double talk):** my speech still kept as "me" went from 0.945 with no leak to 0.939 with sonora, 0.6 points less.
- `drop_echo` only works with zero delay (0.143). Even 20 ms late makes it useless (0.428). As a fallback it needs delay estimation. As long as sonora works like this, the need for a fallback is small.
- **Speed:** sonora runs at RTF about 0.021 on one thread. About 5 minutes for a 4-hour mic. Run before the final pass, it adds only about 5 minutes to the after-meeting processing.
- **Left:**
  - 4-hour drift: at 50 ppm it is 720 ms by the end. Delays up to 200 ms were fine, but 720 ms was not measured.
  - Nonlinear speaker distortion and real room recordings. A synthetic RIR is simpler than a real room.
  - The leak level and delay on a real Mac (experiment order 2).
  - No meeting audio was used.

## Decision (2026-10-22)

- REC-4 in 0.3 runs sonora 0.2.0 after the recording, before the final pass. Moving the reference 20 ms earlier (lead20) is the default. lead20 is a post hoc design, added after seeing the zero-delay failure in the first run.
- VoiceProcessingIO and Windows communications-mode AEC are demoted to fallbacks.
- Order: sonora is integrated after the ort skeleton. When it enters the app workspace, check `security.rs` and the dependency licences together.
- The 4-hour drift (720 ms at the end) runs after the 0.2 GUI check. If it is bad, consider per-span delay estimation. Nonlinear distortion gets no separate synthetic experiment; the real Mac measurement below stands in for it.

## Real Mac leak measurement procedure (during the GUI check)

A measurement in which the user has to play sound through the speakers. **No real meeting audio and no human voices.** Play synthetic speech only, and nobody talks meanwhile. Only numbers go into the repo.

1. **Sound to play.** Put 60 seconds of synthetic speech made with `say` at `bench/data/aec/mac/far.wav` (like `bench/make_synth.sh`, outside the repo). Several voices take turns, with silence in between.
2. **Recording.**
   - Launch the MeetButler 0.2 build **on the test data folder**, the same way as for the GUI check.

     ```bash
     open -n --env MEETBUTLER_DATA_DIR=$HOME/MeetButler-test \
             --env MEETBUTLER_MODELS=$HOME/MeetButler-test/models .../MeetButler.app
     ```

   - The real data folder (`~/Library/Application Support/dev.sshlab.meetbutler/`) is not used.
   - Turn on both the microphone and system audio and press Start.
   - Close other meeting apps. Unplug earphones and headsets; use the built-in speakers and built-in mic.
3. **Playback.** Play `afplay bench/data/aec/mac/far.wav` once at 50% volume and once at 100%, with a 10-second pause in between. Keep the room quiet meanwhile. If there is an external speaker, do it once more.
4. **Stop.** Press Stop. The final pass and summary may be cancelled.
5. **Analysis.**
   - Measure `mic.pcm` and `system.pcm` (16 kHz mono s16le) in `$HOME/MeetButler-test/sessions/<id>/` with a script that only reads them. Do not copy the files into the repo or anywhere else.
   - The script reads only sessions in the test folder. It does not take the real data folder even as a path argument. If a path is outside the test folder or under the real data folder, it stops before opening it.
   - Measure:
     - leak level (dB relative to system)
     - delay (peak of the cross-correlation, ms)
     - tail length (time until the correlation envelope reaches −30 dB)
     - the drop in leak energy before and after sonora (lead20) (ERLE, dB)
   - Measure each volume separately. The script is written in `bench/` at measurement time (it prints numbers only).
6. **Cleanup.** Delete the measurement sessions from the library of the app launched on the test folder. Only sessions inside `$HOME/MeetButler-test` are deleted.
7. **Record.** Write only the numbers above into this document. If they differ from the bench's synthetic conditions (−25 dB, 20–200 ms, RT60 0.3 s), add one more bench condition with those values and measure call DER again.
