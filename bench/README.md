# bench — Korean STT engine comparison

Measurement scripts for MeetButler 0.2's engine decisions (requirements.md §8, §13).
Results are in [`results/RESULTS.md`](results/RESULTS.md).

## Data and licences

| What | Source | Licence |
|---|---|---|
| Speech | [FLEURS](https://huggingface.co/datasets/google/fleurs) `ko_kr` test (382 sentences, 80 min, read speech) | CC BY 4.0 |
| faster-whisper models | Systran/faster-whisper-small, mobiuslabsgmbh/faster-whisper-large-v3-turbo | MIT |
| whisper.cpp models | ggerganov/whisper.cpp `ggml-small.bin`, `ggml-large-v3-turbo.bin` | MIT |
| whisper.cpp VAD | ggml-org/whisper-vad `ggml-silero-v5.1.2.bin` | MIT |
| SenseVoice | k2-fsa/sherpa-onnx `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` (int8) | FunASR Model License (requires crediting the model; check before shipping it in the app) |
| sherpa-onnx VAD | k2-fsa/sherpa-onnx `silero_vad.onnx` (Silero v4 export, [release asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx), sha256 `9e2449e1…cbb1fd6`). The Rust engine runs the same file | MIT |

No real meeting recordings are used. Downloaded speech and models live only in `bench/data/`
(gitignored), and the raw engine output stays in `bench/results/raw/` (gitignored). Only
numbers are committed.

## Setup (macOS)

```bash
brew install python@3.12 whisper.cpp ffmpeg
python3.12 -m venv bench/.venv
bench/.venv/bin/pip install faster-whisper sherpa-onnx jiwer soundfile numpy requests

mkdir -p bench/data/downloads/fleurs bench/data/models bench/data/fleurs_ko
# FLEURS ko_kr test
curl -L -o bench/data/downloads/fleurs/test.tsv \
  https://huggingface.co/datasets/google/fleurs/resolve/main/data/ko_kr/test.tsv
curl -L -o bench/data/downloads/fleurs/test.tar.gz \
  https://huggingface.co/datasets/google/fleurs/resolve/main/data/ko_kr/audio/test.tar.gz
bench/.venv/bin/python -I -c "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter='data')" \
  bench/data/downloads/fleurs/test.tar.gz bench/data/fleurs_ko
# models: the files in the table above go into bench/data/models/ (unpack SenseVoice into models/sv/)
```

faster-whisper models are downloaded into `bench/data/models/faster-whisper/` on the first run.

## Running

```bash
sh bench/run_all.sh                                        # everything (about 1.5 hours on an M4)
bench/.venv/bin/python -I bench/run_utterances.py wcpp-small --limit 20   # just one
bench/.venv/bin/python -I bench/run_stream.py fw-small --mode live
```

Engine names: `fw-small`, `fw-large-v3-turbo`, `wcpp-small`, `wcpp-large-v3-turbo`,
`wcpp-small-cpu`, `sensevoice`, `sensevoice-coreml`, and the app's Rust engine `rust-<model>`
(`run_longform.py` only; first run `cargo build --release -p meetbutler-engine --features whisper,silero
--example final_pass`).

VAD reference: `silero_reference.py` writes onnxruntime frame probabilities and sherpa spans as
JSON for a WAV (or for a FLEURS session made with `--fleurs <minutes>`). The Rust side,
`tests/slow_vad.rs`, compares against it.

pytest does not collect `bench/` (`norecursedirs` in `pytest.ini`). The existing test suite
still runs without models or network.

## What was kept the same

- Decoding settings match `legacy/backend/transcriber.py` (v0.1): fixed to Korean, Whisper at
  beam 5. Silero VAD at threshold 0.3, minimum silence 500 ms, padding 200 ms. No text context
  is carried between windows. faster-whisper calls the production `WhisperTranscriber` as is.
- Four threads per engine (the number of M4 performance cores).
- whisper.cpp runs Homebrew's `whisper-server` on loopback so the model loads once. Each request
  pays for sending the WAV over HTTP, but that is a few ms for a 30 s window.
- The tail-window measurement uses `IncrementalTranscriber` from `legacy/backend/incremental.py`
  (v0.1) as is.

## How CER is computed

- Normalise with NFKC, lowercase, then remove spaces and punctuation. Then divide the
  character-level edit distance by the number of reference characters (summed over the corpus).
- FLEURS references carry romanisations in parentheses that are not read aloud
  (`비슈케크(Bishkek)`), so the text in parentheses is removed.
- Number forms (`2명` / `두 명`, "2 people" as digits or words) are not normalised. Instead, a
  separate CER is reported on only the sentences whose reference has no digits. Engines are
  compared on that value.

## Known issues and how to read the results

- Live (10 s period) CER came out much worse than per-sentence CER because, at the time,
  `IncrementalTranscriber` committed the same words twice at window boundaries. It is not an
  engine difference. RESULTS.md keeps the before and after results side by side.
- FLEURS is read speech and clean audio. Far-field room microphones, overlapping speech and
  conversational speech raise every engine's CER. Do not assume the ranking holds; in-person
  meeting conditions need their own measurement.
- Memory is macOS phys_footprint (the same as the "Memory" value in Activity Monitor). It
  includes Metal allocations.
- Even after the fix, whisper.cpp misses the live CER bar (per-sentence + 2%p). Most of its
  errors are deletions at sentence ends. Turning VAD off (`wcpp-small-novad`) made it worse, so
  the hypothesis that VAD time remapping is the cause was rejected. Inaccurate segment end times
  remain the candidate (word-time DTW needs `-nfa` and was not measured).
- `run_stream.py` runs only a forced pass after all the audio is in (the same as
  `pipeline._drain`). It used to skip forcing when 1 s or more remained, so when a tail running
  to the end of the window was held back it re-read the same window forever (seen with
  `wcpp-large-v3-turbo-novad`, no result). All committed results ended normally, so none hit
  this loop.
- With VAD on in whisper-server, only segment times are mapped back to the original timeline;
  word times and DTW times (`t_dtw`) stay on the timeline with the silence cut out (in a 30 s
  window test, the DTW end came before the segment start). So DTW was measured only with VAD
  off (`wcpp-small-novad-dtw`). With VAD off, insertion errors (hallucination in silence,
  repetition) go up, so even with DTW fixing end times, live CER misses the bar. Near the end of
  this measurement another build started running on the same machine, so the per-window time in
  that row is for reference only.
- whisper.cpp long-form (the 36-minute session as one request) has a much worse CER. Most errors
  are deletions, with stretches of 20–35 s skipped whole and repeated segments. Re-transcription
  after the meeting must not be left to the engine's long-form; cut it into windows or VAD spans
  (see the tail-window backlog figures).
- With our VAD spans plus per-span decoding (`*-spans`), short spans hallucinate. Of turbo's 38
  batch spans under 2 s, 37 were subtitle credits such as "한글자막 by …" or
  "시청해주셔서 감사합니다." ("thanks for watching"). Rescoring the saved output without spans
  under 1.5 s (padding included) drops CER from 6.9% to 2.4%. That value is a post hoc estimate,
  not a measurement. Dropping short spans also drops short replies in meetings such as "네"
  (yes) or "아니요" (no); FLEURS has no such utterances, so that cost does not show here.
- CER reference preprocessing: at first, all parentheses were removed. But 22 of the 382
  sentences actually read the Korean inside the parentheses (282 characters). Now only
  parentheses with no Hangul are removed. All results were rescored from the saved output with
  `rescore.py` (engines were not rerun and times are unchanged). Most went down by 0.2–0.8%p;
  rankings and bar passes did not change.
- The last round (RESULTS.md §6). The `no_speech_prob`/`avg_logprob` filter dropped no segment
  at all. whisper.cpp gives a low `no_speech_prob` even for hallucinated subtitle credits. Every
  hallucination came from a VAD span of about 1 s running from the end of a sentence clip into
  the next gap (a splice boundary; in a real meeting, a click or a cough). Packing (`packed`)
  removes the hallucinations but swallows 22 of the 53 short replies. Fed separately, turbo
  transcribes all 8 of those replies, so packing is the cause. Per-span (`spans`) keeps 49/53
  replies.
- An estimate from applying, post hoc, the rule "drop a span under 2 s that contains a known
  hallucination phrase (한글자막…, 시청해주셔서…, 지금까지 뉴스 스토리…, …계속 됩니다,
  감사합니다)" to the saved output: turbo batch CER 6.9% → 2.4%, short replies 49/53 kept. turbo
  live (phrases removed only) is estimated at 6.5% → 2.9%. small live had only one hallucination
  and stays at 8.2%. This is an estimate, not a measurement, and it also drops a real
  "감사합니다" (thank you) in a meeting.
- This round's short-reply session hallucinates far less than the original session (2 vs 37).
  It looks like inserting the replies changed the gaps and the noise layout, but this was not
  checked. Compare reply recall only within the same session.
- whisper.cpp's ggml Silero v5.1.2 is not used (RESULTS.md §7). The original model prepends the
  previous frame's last 64 samples as context; the ggml port reflect-pads each frame instead. On
  36 minutes of FLEURS it found only 23.0 minutes of speech (v4 onnx on the same int16 input:
  26.9 minutes), missed whole utterances, and the Rust final pass CER was 6.0%. On a synthetic
  speech session v5 caught more, so the difference did not show. Do not judge a VAD or engine
  change on synthetic speech alone.
- Even the same Silero gives different spans for float and int16 input (36-minute session,
  sherpa spans 205 vs 174). §5 and §6 are float; §7 is int16, which is what the app records.
  Compare only within the same format.

## Summary model measurement (0.2 milestone 3)

Real meeting transcripts cannot be used, so meetings are synthesised. `meeting_gen.py` makes
3 meeting types (weekly meeting, 1:1, customer meeting) × 3 lengths (30 minutes, 1 hour,
2 hours). It plants decisions, action items (owner and due date), numbers, open questions, a
proposal withdrawn midway, short replies and small talk, and outputs what it planted as the
answer key. All people, company and feature names are invented. As in the app, there are no
speaker labels, so owners are called by name in the speech. Script text lives only in
`bench/data/meetings/` (not committed).

`summary_score.py` scores a fixed-format (SUM-1, JSON) summary against the answer key. No judge
model is used. It counts:
- format pass
- recall of decisions, action items and open questions: an item counts as found when all its key words are inside one item of the output. Action items are also counted separately for a correct owner and due date.
- withdrawn proposals written as decisions
- people and numbers that were not in the meeting

Korean fluency is judged by blind human scoring.

```bash
bench/.venv/bin/python -I bench/meeting_gen.py                                 # 9 scripts
bench/.venv/bin/python -I -m unittest discover -s bench/tests -t bench         # all bench tests (generator, scorer, diarization)
# (standard-library unittest, so no pytest needed; bench/.venv is the venv from "Setup" above)
```

### Summary models (GGUF converted here)

The safetensors from each official repository were downloaded pinned to a commit hash, and
Q4_K_M was built with `convert_hf_to_gguf.py` and `llama-quantize` from the same llama.cpp as
the sidecar (`d812350`) (`llm_models.py`, results in `llm_models.json`). The downloaded
originals were deleted after conversion. They can be rebuilt from the commit hashes, and the
file a user downloads is checked against the sha256 below.

| Name | Source repository@commit | Licence | Device tier | Size | sha256 | transformers used to convert |
|---|---|---|---|---:|---|---|
| kanana-1.5-2.1b | `kakaocorp/kanana-1.5-2.1b-instruct-2505@7df4bc35ccd6` | Apache-2.0 | 8GB | 1.42 GiB | `622293385bb6fb46…` | 4.57.6 |
| midm-2.0-mini | `K-intelligence/Midm-2.0-Mini-Instruct@383eb221c52a` | MIT | 8GB | 1.33 GiB | `f4ba7d9ac35b523f…` | 4.57.6 |
| qwen3.5-4b | `Qwen/Qwen3.5-4B@851bf6e806ef` | Apache-2.0 | 8GB | 2.59 GiB | `60cb21ee219f5c13…` | 4.57.6 |
| ax-4.0-light | `skt/A.X-4.0-Light@ba21c20ea1b3` | Apache-2.0 (ships a NOTICE: built with Qwen 2.5) | 16GB | 4.13 GiB | `2115a098ae9c1309…` | 4.57.6 |
| kanana-1.5-8b | `kakaocorp/kanana-1.5-8b-instruct-2505@c963a5f4f649` | Apache-2.0 | 16GB | 4.58 GiB | `ea411a3934a8cbfb…` | 4.57.6 |
| qwen3.5-9b | `Qwen/Qwen3.5-9B@c20223623576` | Apache-2.0 | 16GB | 5.38 GiB | `d5e23001c0128665…` | 4.57.6 |
| midm-2.0-base | `K-intelligence/Midm-2.0-Base-Instruct@35479c5fc9a1` | MIT | 16GB | 6.55 GiB | `e590278bb19ffaf6…` | 4.57.6 |
| gemma-4-e4b | `google/gemma-4-E4B-it@ee0ef6023621` | Apache-2.0 | 8GB | 4.97 GiB | `4b9bd938e2cd55f4…` | 5.20.1 |

- If A.X 4.0 Light is adopted and shipped, Apache-2.0 §4(d) requires shipping the repository's NOTICE ("Built with Qwen 2.5 — Alibaba Cloud") with it.
- Gemma 4 E4B was converted with transformers 5.20.1 (a separate venv), because the 4.57.6 that llama.cpp pins does not know `gemma4`. The rest used 4.57.6.
- Gemma 4 E4B is "effective 4B", but with its per-layer embeddings it is about 8B, so Q4_K_M is 4.97 GiB. That is over the 8GB tier's cap (4GB peak memory).
- The licence text checks and the excluded models (EXAONE 4.0, Kanana-2, HyperCLOVA X SEED, the Llama family) are in the 0.2 milestone 3 report.

## Diarization measurement (requirements §12 risk 1)

`run_diar.py` measures sherpa-onnx's after-the-meeting diarization (pyannote segmentation-3.0 +
speaker embedding + clustering) on Korean mock meetings and on AMI. `diar_pipe.py` redoes the
same process in numpy on saved embeddings, so the clustering method, threshold and small-cluster
floor can be varied without extracting again. Results are in `results/RESULTS.md` §9.

| What | Source | Licence |
|---|---|---|
| Korean speech | [Zeroth-Korean](https://www.openslr.org/40/) (OpenSLR 40) read speech. The 10 test speakers are mixed into the test meetings, and 12 of the train speakers (124 139 151 158 164 169 183 187 194 200 207 208) into the dev meetings (`diar_mix.py`) | CC BY 4.0 |
| English meetings | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) test, 16 meetings, Mix-Headset and Array1-01 | CC BY 4.0 |
| AMI reference | [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) `2509d8933721023fab4def2618aabd5c28eb82e9`, `only_words/rttms/test`, `uems/test` | Apache-2.0 |
| Models | sherpa-onnx GitHub releases. Sources, licences and sha256 in `diar_models.json` | segmentation MIT, 3D-Speaker (CAM++, ERes2Net, ERes2NetV2) Apache-2.0, TitaNet and WeSpeaker CC BY 4.0 (attribution required) |

Downloading and checking are done by `fetch_diar.sh` alone. Downloaded files stay only in
`bench/data/diar/` (gitignored), and the sha256 of all 3542 files is committed in
`diar_data.sha256`. Each download is checked against its Content-Length, and each .wav for
starting with RIFF, with up to five retries. Zeroth is read once as a stream from its 10.3 GB
archive, keeping only the speakers above.

```bash
bench/.venv/bin/pip install sherpa-onnx==1.13.8 onnxruntime==1.31.0 scipy==1.16.2 numpy==2.5.4 soundfile==0.14.0
bench/fetch_diar.sh                       # models ami zeroth, sha256 check at the end
```

| Package | Version | Licence | Used for |
|---|---|---|---|
| sherpa-onnx | 1.13.8 (bundles its own onnxruntime 1.28.2) | Apache-2.0 | all of diarization, embedding extraction |
| onnxruntime | 1.31.0 | MIT | segmentation in `diar_pipe` |
| scipy | 1.16.2 | BSD-3-Clause | hierarchical clustering in `diar_pipe` |
| numpy | 2.5.4 | BSD-3-Clause | |
| soundfile | 0.14.0 | BSD-3-Clause | reading audio |

Every result JSON carries, in `env`, the sherpa-onnx, onnxruntime, scipy and numpy versions and
the sha256 of the segmentation and embedding models.

The order to rerun. Each step is skipped when its results are saved, and it stops before the
next unit when `bench/data/diar/STOP` exists.

```bash
bench/.venv/bin/python -I bench/run_diar.py tune korean ami          # threshold chosen on a 5-minute dev, test meetings (Korean 12, AMI 32)
bench/.venv/bin/python -I bench/run_diar.py online long              # 60 s piece live vs after the meeting, time and memory of a 4-hour meeting
bench/.venv/bin/python -I bench/run_diar.py absorb                   # absorption (diar_post) with threshold × share chosen on a 30-minute dev, exploratory
bench/.venv/bin/python -I bench/run_diar.py extract-dev verify       # save 30-minute dev embeddings, check sherpa reproduction (DER within 0.5%p, same speaker count)
bench/.venv/bin/python -I bench/run_diar.py grid                     # clustering method × threshold × floor, chosen for shared, room and call separately
bench/.venv/bin/python -I bench/run_diar.py preregister            # write the values to use into results/diar/preregistered.json once, before looking at test (never overwritten)
bench/.venv/bin/python -I bench/run_diar.py extract-test pipe        # save test embeddings, apply only the preregistered values as they are
bench/.venv/bin/python -I -m unittest discover -s bench/tests -t bench    # tests that run without models or data
```

From 6aa83d1, the call scene's band limit zero-pads to a power-of-two length for the FFT (one
4-hour call went from 4.9 hours to 48 s). The output differs by up to 0.026 within 1 s of either
end and by about one 16-bit step (3e-5) in the middle (measured on 30 minutes of noise), so the
Korean results in RESULTS.md §9 (the call meetings and the averages that include them) do not
reproduce byte for byte with the current code. To rebuild them exactly, check out 167a9b8, which
committed those results, delete the Korean meetings in `bench/data/diar/`, and run the order
above. Room meetings and AMI are not affected.

Run the tests in bench/.venv (the venv with scipy installed by the pinned pip command above).
The root `.venv` has no scipy, so the diar_pipe tests fail there.
