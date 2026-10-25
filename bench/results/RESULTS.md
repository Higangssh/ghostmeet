# Korean STT engine comparison — results

Machine: Apple M4 (4P+6E), RAM 16GB, macOS 26.6.2, Python 3.12.15. 4 threads for every engine. No other heavy work while measuring.

Data: FLEURS ko_kr test, 382 sentences / 80.1 min (google/fleurs, CC BY 4.0). No real meeting recordings were used.

## 1. Sentence by sentence (engine accuracy and speed)

| Engine | CER (sentences without digits) | CER (all) | RTF | Load | First result | Latency per sentence median/p95 | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (CPU int8) | **2.2%** | 2.5% | 0.373 | 1.9s¹ | 6.6s¹ | 4.61 / 5.24s | 2247MB |
| whisper.cpp large-v3-turbo (Metal) | **2.3%** | 2.5% | 0.165 | 1.0s | 3.4s | 2.07 / 2.26s | 1936MB |
| sherpa-onnx SenseVoice-Small (CPU int8) | **4.7%** | 6.1% | 0.011 | 0.4s | 0.6s | 0.13 / 0.22s | 949MB |
| faster-whisper small (CPU int8) | **5.1%** | 5.4% | 0.118 | 0.8s | 2.4s | 1.45 / 1.96s | 945MB |
| whisper.cpp small (CPU) | **5.2%** | 5.3% | 0.426 | 0.4s | 5.7s | 5.31 / 5.98s | 823MB |
| whisper.cpp small (Metal) | **5.2%** | 5.4% | 0.056 | 0.5s | 1.2s | 0.69 / 0.93s | 866MB |

- CER: letter edit distance after removing spaces and punctuation / letters in the reference (summed over the corpus). Romanised glosses in brackets in the reference (`비슈케크(Bishkek)`) are not read aloud, so they were removed. Brackets that hold Hangul are read, so they were kept.
- CER without digits: leaving out sentences whose reference has a digit, so spelling differences like `2명`/`두 명` (2 people / two people) do not blur the engine comparison. **Compare engines by this column.**
- RTF = processing time / speech length. First result = from starting to build the engine until the first sentence (14 s) has a result (the model already on disk).
- Peak memory: the macOS phys_footprint peak ("Memory" in Activity Monitor), Metal allocations included. For whisper.cpp it is the whisper-server process.
- ¹ In the full run this was the first run, so a model download got mixed into load. The values are from a separate 5-sentence run with the model cached.

## 2. Long session, tail window (legacy/backend/incremental.py's IncrementalTranscriber as is)

A 36.1 min session of FLEURS sentences joined together, 30 s window. `before fix` = main at the time of measuring (with the window boundary duplicate bug), `after fix` = the window boundary hold-back fix.

| Engine | Code | Condition | Per window first 10 min → last 10 min | p95 | Slope (s/hour) | Memory first → last | Caption delay median/p95 | CER |
|---|---|---|---:|---:|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (CPU int8) | before fix | backlog (30 s window) | 5.50s → 5.22s | 9.32s | -1.156 | 1705 → 1674MB | — | 18.7% |
| faster-whisper large-v3-turbo (CPU int8) | after fix | backlog (30 s window) | 5.61s → 5.79s | 9.71s | +0.686 | 1706 → 1693MB | — | 6.7% |
| faster-whisper large-v3-turbo (CPU int8) | before fix | live (10 s period) | 4.78s → 4.73s | 5.21s | +0.011 | 1667 → 1593MB | 9.6 / 18.1s | 21.2% |
| faster-whisper large-v3-turbo (CPU int8) | after fix | live (10 s period) | 4.96s → 4.92s | 5.37s | -0.222 | 1641 → 1580MB | 12.4 / 20.8s | 3.4% |
| faster-whisper small (CPU int8) | before fix | backlog (30 s window) | 2.35s → 2.65s | 3.56s | +0.138 | 1013 → 1229MB | — | 23.9% |
| faster-whisper small (CPU int8) | after fix | backlog (30 s window) | 2.38s → 2.53s | 2.92s | +0.354 | 1147 → 1096MB | — | 9.5% |
| faster-whisper small (CPU int8) | before fix | live (10 s period) | 1.48s → 1.47s | 2.28s | -0.054 | 903 → 839MB | 4.9 / 12.3s | 29.7% |
| faster-whisper small (CPU int8) | after fix | live (10 s period) | 1.62s → 1.62s | 2.19s | -0.044 | 899 → 829MB | 8.1 / 14.7s | 6.6% |
| sherpa-onnx SenseVoice-Small (CPU int8) | before fix | backlog (30 s window) | 0.44s → 0.38s | 0.58s | -0.141 | 1184 → 1133MB | — | 10.0% |
| sherpa-onnx SenseVoice-Small (CPU int8) | after fix | backlog (30 s window) | 0.38s → 0.39s | 0.54s | +0.004 | 1142 → 956MB | — | 6.2% |
| sherpa-onnx SenseVoice-Small (CPU int8) | before fix | live (10 s period) | 0.16s → 0.15s | 0.22s | -0.005 | 1050 → 879MB | 0.9 / 8.2s | 17.0% |
| sherpa-onnx SenseVoice-Small (CPU int8) | after fix | live (10 s period) | 0.24s → 0.21s | 0.42s | -0.086 | 1115 → 1048MB | 6.5 / 11.2s | 6.0% |
| whisper.cpp large-v3-turbo (Metal) | before fix | backlog (30 s window) | 2.35s → 2.36s | 2.55s | +0.038 | 1943 → 1956MB | — | 13.9% |
| whisper.cpp large-v3-turbo (Metal) | after fix | backlog (30 s window) | 2.39s → 2.42s | 2.59s | +0.056 | 1942 → 1949MB | — | 5.8% |
| whisper.cpp large-v3-turbo (Metal) | before fix | live (10 s period) | 2.12s → 2.10s | 2.26s | -0.029 | 1927 → 1929MB | 5.1 / 12.0s | 21.6% |
| whisper.cpp large-v3-turbo (Metal) | after fix | live (10 s period) | 2.15s → 2.13s | 2.34s | -0.022 | 1933 → 1942MB | 9.3 / 16.0s | 5.8% |
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | after fix | live (10 s period) | 4.69s → 4.07s | 5.93s | -1.007 | 1923 → 1924MB | 13.5 / 20.5s | 6.5% |
| whisper.cpp large-v3-turbo (Metal, Silero spans) | after fix | live (10 s period) | 4.71s → 4.49s | 6.27s | -1.027 | 1924 → 1932MB | 13.2 / 21.8s | 5.9% |
| whisper.cpp small (Metal) | before fix | backlog (30 s window) | 0.94s → 0.86s | 1.28s | -0.119 | 862 → 866MB | — | 29.9% |
| whisper.cpp small (Metal) | after fix | backlog (30 s window) | 1.13s → 1.04s | 1.55s | -0.223 | 862 → 864MB | — | 8.6% |
| whisper.cpp small (Metal) | before fix | live (10 s period) | 0.68s → 0.68s | 0.93s | -0.018 | 857 → 836MB | 4.7 / 11.3s | 31.9% |
| whisper.cpp small (Metal) | after fix | live (10 s period) | 0.76s → 0.72s | 0.95s | -0.104 | 856 → 863MB | 6.8 / 12.5s | 8.9% |
| whisper.cpp small (Metal, Silero spans + filter) | after fix | live (10 s period) | 1.03s → 1.16s | 1.69s | +0.300 | 857 → 787MB | 8.2 / 13.1s | 8.3% |
| whisper.cpp small (Metal, no VAD) | after fix | live (10 s period) | 0.89s → 0.77s | 1.19s | -0.176 | 859 → 864MB | 7.4 / 17.3s | 13.2% |
| whisper.cpp small (Metal, no VAD, DTW) | after fix | live (10 s period) | 0.93s → 0.81s | 1.22s | -0.201 | 899 → 926MB | 8.4 / 15.5s | 12.2% |
| whisper.cpp small (Metal, Silero spans) | after fix | live (10 s period) | 1.32s → 1.22s | 1.88s | -0.301 | 858 → 820MB | 7.8 / 13.7s | 8.6% |

- backlog: the speech is all there already, so every pass fills the 30 s window. This condition shows whether the cost per window is independent of the position in the session.
- live: a virtual clock imitates audio arriving in real time. The next pass runs 10 s after the last one ends (as the pipeline worker does). Caption delay = when the pass ended − when the segment ended.
- Slope: a linear fit of the time per window against the position in the session. Near 0 means flat.

## 3. Before and after the window boundary fix (live, 10 s period)

Done when: live CER ≤ sentence-by-sentence CER (all) + 2.0%p.

| Engine | Sentence CER | live CER before → after | Gap (after − sentence) | Bar | Caption delay median/p95 before → after |
|---|---:|---:|---:|---|---:|
| faster-whisper large-v3-turbo (CPU int8) | 2.5% | 21.2% → **3.4%** | +0.9%p | pass | 9.6 / 18.1s → 12.4 / 20.8s |
| faster-whisper small (CPU int8) | 5.4% | 29.7% → **6.6%** | +1.2%p | pass | 4.9 / 12.3s → 8.1 / 14.7s |
| sherpa-onnx SenseVoice-Small (CPU int8) | 6.1% | 17.0% → **6.0%** | -0.1%p | pass | 0.9 / 8.2s → 6.5 / 11.2s |
| whisper.cpp large-v3-turbo (Metal) | 2.5% | 21.6% → **5.8%** | +3.2%p | fail | 5.1 / 12.0s → 9.3 / 16.0s |
| whisper.cpp small (Metal) | 5.4% | 31.9% → **8.9%** | +3.5%p | fail | 4.7 / 11.3s → 6.8 / 12.5s |

- The comparison is against the sentence-by-sentence CER on 'all', because the session also holds sentences with digits.
- Sentence by sentence feeds one whole sentence; the session joins sentences 0.6 s apart and cuts them into windows. So even the same engine sees slightly different context and boundaries.

## 4. Batch re-transcription after the meeting (the whole session at once)

The same 36.1 min session fed to the engine whole. The engine's long-form decoding does the windowing. For comparison, the same engine's tail-window backlog (after fix) CER is beside it.

| Engine | long-form CER | Processing time | RTF | Peak memory | Tail-window backlog CER |
|---|---:|---:|---:|---:|---:|
| whisper.cpp large-v3-turbo (Metal) | **17.3%** | 93s | 0.043 | 2503MB | 5.8% |

## 5. Our VAD spans + decoding span by span

Neither the engine's VAD nor long-form is used. Silero VAD (MIT) cuts the speech spans and each span is decoded on its own. Times come from the spans. live (tail window) and batch (whole session) use the same function (`engines.speech_spans`) with the same values. The SenseVoice row follows the same rules.

Span rules: `threshold=0.3`, `min_silence_sec=0.5`, `min_speech_sec=0.25`, `max_speech_sec=25.0`, `pad_sec=0.2`.

| Engine | Condition | CER | Sentence CER | Per window / processing time | Caption delay median/p95 | Load while measuring (1 min average) |
|---|---|---:|---:|---:|---:|---|
| whisper.cpp large-v3-turbo (Metal, Silero spans) | live (10 s period) | **5.9%** | 2.5% | 4.71s → 4.49s | 13.2 / 21.8s | 1.96 → 1.00 |
| whisper.cpp large-v3-turbo (Metal, Silero spans) | batch | **6.9%** | 2.5% | 430s (RTF 0.199) | — | 1.17 → 3.70 (peak in between 6.2) |
| whisper.cpp small (Metal, Silero spans) | live (10 s period) | **8.6%** | 5.4% | 1.32s → 1.22s | 7.8 / 13.7s | 1.93 → 1.17 |
| sherpa-onnx SenseVoice-Small (CPU int8) (reference) | live (10 s period) | 6.0% | 6.1% | 0.24s → 0.21s | 6.5 / 11.2s | — |

## 6. Last round: filter, packing, short replies

Section 5's span rules plus a Whisper segment filter. The filter drops a segment with `no_speech_prob > 0.6` and `avg_logprob < -1.0` (the same rule as OpenAI/faster-whisper, and it carries over to whisper-rs as is). `packed` joins neighbouring spans up to 25 s and decodes them at once; `spans` decodes each span on its own. The bar is sentence CER + 2.0%p.

Short replies: 8 replies made with macOS `say -v Yuna` (네, 아니요, 맞아요, 좋습니다, 그렇죠, 알겠습니다, 잠깐만요, 글쎄요: yes, no, right, good, sure, got it, one moment, well) were put in after every third sentence with 0.8 s on either side (`make_replies.sh`; the synthetic audio lives only in bench/data). A reply counts as found when more than half of it matches as is in the letter alignment.

| Engine | Condition | CER | Sentence CER | Bar | Short replies | Segments dropped by the filter | Per window / processing time | Caption delay median/p95 | Load (1 min average) |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | live (10 s period) | **6.5%** | 2.5% | fail | — | — | 4.69s → 4.07s | 13.5 / 20.5s | 1.35 → 1.24 |
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | batch | **6.9%** | 2.5% | fail | — | 0 | 418s (RTF 0.193) | — | 1.63 → 0.57 |
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | batch + short replies | **3.2%** | 2.5% | pass | 49/53 (lost to the filter 0) | 0 | 471s (RTF 0.217) | — | 0.97 → 1.09 |
| whisper.cpp large-v3-turbo (Metal, Silero packed + filter) | batch + short replies | **3.8%** | 2.5% | pass | 31/53 (lost to the filter 0) | 0 | 222s (RTF 0.103) | — | 1.60 → 0.97 |
| whisper.cpp small (Metal, Silero spans + filter) | live (10 s period) | **8.3%** | 5.4% | fail | — | — | 1.03s → 1.16s | 8.2 / 13.1s | 0.57 → 1.35 |
| faster-whisper small (CPU int8) | live (10 s period), comparison | **6.6%** | 5.4% | pass | — | — | 1.62s → 1.62s | 8.1 / 14.7s | — |
| faster-whisper large-v3-turbo (CPU int8) | live (10 s period), comparison | **3.4%** | 2.5% | pass | — | — | 4.96s → 4.92s | 12.4 / 20.8s | — |
| sherpa-onnx SenseVoice-Small (CPU int8) | live (10 s period), comparison | **6.0%** | 6.1% | pass | — | — | 0.24s → 0.21s | 6.5 / 11.2s | — |

## 7. The Rust engine's final pass after the meeting (the app's own path)

`crates/meetbutler-engine`'s `final_transcript` run as is (`examples/final_pass`). The same rules as section 6's `*-filtered` (decode our spans one by one, block subtitle credits in spans under 2 s), but the VAD probabilities, span rules, blocklist and whisper.cpp calls are all Rust. The audio is passed in the format the app records (16 kHz s16le). The bar is a final pass CER of 3% or less. The session with short replies is judged by recall: at least 51/53 replies, 0 replies lost to the blocklist, and CER within +1.0%p of the plain session on the same path.

| Engine | Condition | CER | Bar | Short replies | Processing time | RTF | Peak memory | Load (1 min average) |
|---|---|---:|---|---:|---:|---:|---:|---|
| Rust engine large-v3-turbo (Metal, Silero spans + blocklist) | batch | **2.7%** | pass | — | 203s | 0.094 | 1965MB | 1.81 → 0.73 |
| Rust engine large-v3-turbo (Metal, Silero spans + blocklist) | batch + short replies | **3.3%** | pass | 52/53 | 261s | 0.121 | 1961MB | 0.73 → 1.03 |
| Rust engine large-v3-turbo (Metal, whisper.cpp ggml Silero v5.1.2 spans + blocklist) | batch | **6.0%** | fail | — | 234s | 0.108 | 1963MB | 1.89 → 3.60 |
| Rust engine large-v3-turbo (Metal, whisper.cpp ggml Silero v5.1.2 spans + blocklist) | batch + short replies | **6.7%** | — (plain session failed) | 52/53 | 283s | 0.130 | 1963MB | 3.60 → 1.26 |

- whisper.cpp's ggml Silero v5.1.2 (the `-ggmlvad` rows) is not used. The original Silero prepends the last 64 samples of the previous frame to each frame as context; the ggml port reflect-pads each frame instead (`ggml_pad_reflect_1d(…, 64, 64)`). On this session it found only 23.0 min of speech (v4 found 26.9 min on int16 input) and missed whole utterances, for 314 deleted letters. On synthetic speech (TTS) it caught more, so a synthetic comparison did not show it. When changing the VAD or the engine, check against FLEURS too (`tests/slow_vad.rs`).
- The VAD we adopted is the same `silero_vad.onnx` as bench (Silero v4 as exported by k2-fsa, MIT), run on tract (a pure Rust ONNX runtime). Against onnxruntime the frame probabilities differ by 6.2e-6 on the synthetic session and 1.9e-5 on 10 min of FLEURS, and on the same input not a single span differs from sherpa-onnx.
- The same VAD gives different spans for float and int16 input. On this 36 min session sherpa gave 205 spans on float input and 174 on int16 (many do not overlap). Sections 5 and 6 were measured on float; the app records int16, so this section was measured on int16.

## 8. Local summary models (synthetic meeting scripts, answer script as input)

`run_summary.py` ran the app's summary sidecar the way the app does: a new process per summary, the chat template inside the GGUF, a grammar that forces the fixed format (SUM-1), repeat penalty 1.1, greedy. The scripts are `meeting_gen.py`'s weekly, 1:1 and customer meetings × 30, 60 and 120 min, scored against the answer key (`summary_score.py`). Every model got the same prompt. single takes the whole transcript; chunked summarises about 10 min at a time and joins the lists.

Recall is summed over the scripts (a summary with a broken format counts as finding nothing). Wrong value is the number of decisions with the right agenda item but a number or date that differs from the answer (the error that hurts a reader most, so it weighs as much as recall). One-off decisions is the recall over only the decisions whose template does not come back with different numbers. Invented owner counts owners who are not in the meeting or are not people; invented number counts numbers never said in the meeting. Peak memory is the larger of peak footprint and max RSS. The device caps are 4GB for the 8GB tier and 9GB for the 16GB tier; going over is out. single's n_ctx was set to transcript + 8192 (the output cap).

The two tables below were measured on the earlier generator's scripts (v1, 1ac7bf2). v1 planted the same item template several times in one meeting with only the numbers changed ("개발자 N명 채용", hire N developers, four times in one meeting). After the scorer was fixed everything was scored again, so the numbers differ from b76f90b's tables: retracted as decision no longer counts a sentence that states the retraction ("600만 원 취소 후 800만 원으로 확정", 6 million won cancelled, then confirmed at 8 million) and fell to almost 0, and the same answer item is counted once, so the decisions denominator shrank.

### Six 30 and 60 min scripts, every model (earlier generator v1)

| Model | Tier | Mode | Run failed | Broken format | Decisions | One-off decisions | Wrong value | Action items | Owner right | Due date right | Open questions | Retracted as decision | Invented owner | Invented number | Time per meeting | Peak memory | Generation tok/s | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gemma-4-e4b | 8GB | chunked | 0/6 | 0/6 | 58/99 (59%) | 51/70 (73%) | 1 | 53/60 (88%) | 53/53 (100%) | 53/53 (100%) | 14/18 (78%) | 0 | 4 | 0 | 117s | 5.8GB | 17.8 | out (memory) |
| gemma-4-e4b | 8GB | single | 0/6 | 0/6 | 28/99 (28%) | 27/70 (39%) | 0 | 30/60 (50%) | 29/30 (97%) | 30/30 (100%) | 3/18 (17%) | 0 | 0 | 1 | 67s | 7.2GB | 17.0 | out (memory) |
| kanana-1.5-2.1b | 8GB | chunked | 0/6 | 0/6 | 35/99 (35%) | 29/70 (41%) | 1 | 49/60 (82%) | 43/49 (88%) | 48/49 (98%) | 4/18 (22%) | 2 | 16 | 2 | 69s | 2.7GB | 33.1 | — |
| kanana-1.5-2.1b | 8GB | single | 0/6 | 2/6 | 20/99 (20%) | 20/70 (29%) | 0 | 19/60 (32%) | 16/19 (84%) | 17/19 (89%) | 1/18 (6%) | 0 | 1 | 0 | 177s | 5.9GB | 22.6 | out (memory) |
| midm-2.0-mini | 8GB | chunked | 0/6 | 0/6 | 45/99 (45%) | 40/70 (57%) | 1 | 41/60 (68%) | 38/41 (93%) | 41/41 (100%) | 10/18 (56%) | 0 | 9 | 0 | 54s | 3.4GB | 31.1 | — |
| midm-2.0-mini | 8GB | single | 0/6 | 0/6 | 20/99 (20%) | 20/70 (29%) | 0 | 20/60 (33%) | 15/20 (75%) | 20/20 (100%) | 5/18 (28%) | 0 | 3 | 0 | 43s | 8.3GB | 21.4 | out (memory) |
| qwen3.5-4b | 8GB | chunked | 0/6 | 0/6 | 62/99 (63%) | 52/70 (74%) | 2 | 51/60 (85%) | 50/51 (98%) | 51/51 (100%) | 7/18 (39%) | 0 | 4 | 1 | 99s | 3.2GB | 19.4 | — |
| qwen3.5-4b | 8GB | single | 0/6 | 0/6 | 49/99 (49%) | 48/70 (69%) | 0 | 33/60 (55%) | 31/33 (94%) | 33/33 (100%) | 2/18 (11%) | 0 | 0 | 1 | 64s | 4.0GB | 17.7 | out (memory) |
| ax-4.0-light | 16GB | chunked | 0/6 | 0/6 | 39/99 (39%) | 36/70 (51%) | 0 | 47/60 (78%) | 45/47 (96%) | 47/47 (100%) | 4/18 (22%) | 0 | 6 | 0 | 95s | 4.8GB | 16.6 | — |
| ax-4.0-light | 16GB | single | 0/6 | 0/6 | 24/99 (24%) | 22/70 (31%) | 0 | 30/60 (50%) | 22/30 (73%) | 30/30 (100%) | 0/18 (0%) | 0 | 1 | 0 | 62s | 6.2GB | 16.2 | — |
| kanana-1.5-8b | 16GB | chunked | 0/6 | 0/6 | 47/99 (47%) | 38/70 (54%) | 0 | 42/60 (70%) | 40/42 (95%) | 41/42 (98%) | 13/18 (72%) | 0 | 9 | 1 | 167s | 5.9GB | 14.7 | — |
| kanana-1.5-8b | 16GB | single | 0/6 | 0/6 | 23/99 (23%) | 20/70 (29%) | 0 | 16/60 (27%) | 10/16 (62%) | 16/16 (100%) | 5/18 (28%) | 0 | 5 | 8 | 121s | 9.0GB | 12.7 | out (memory) |
| midm-2.0-base | 16GB | chunked | 0/6 | 0/6 | 33/99 (33%) | 28/70 (40%) | 0 | 42/60 (70%) | 40/42 (95%) | 26/42 (62%) | 10/18 (56%) | 0 | 6 | 14 | 179s | 8.7GB | 10.9 | — |
| midm-2.0-base | 16GB | single | 3/6 | 0/3 | 6/33 (18%) | 4/24 (17%) | 0 | 9/20 (45%) | 6/9 (67%) | 7/9 (78%) | 2/7 (29%) | 0 | 0 | 2 | 76s | 11.2GB | 9.6 | out (memory) |
| qwen3.5-9b | 16GB | chunked | 0/6 | 0/6 | 58/99 (59%) | 42/70 (60%) | 9 | 54/60 (90%) | 54/54 (100%) | 54/54 (100%) | 11/18 (61%) | 1 | 9 | 2 | 179s | 5.9GB | 13.0 | — |
| qwen3.5-9b | 16GB | single | 0/6 | 0/6 | 47/99 (47%) | 42/70 (60%) | 2 | 41/60 (68%) | 40/41 (98%) | 41/41 (100%) | 5/18 (28%) | 0 | 0 | 7 | 111s | 6.7GB | 12.2 | — |

### Three 120 min scripts, models chosen by the 30 and 60 min results (earlier generator v1)

For each tier, the two models with the highest decision + action item recall in a mode that fit the cap. single on the llama-architecture models (Kanana, Mi:dm) failed to decode right at the start (`Decode Error -3`, peak memory around 11GB). A failed run leaves no n_ctx, but by the same formula (transcript + 8192) it would be 57344, past the training context of 32768, with the KV cache alone at 10.5GB for Mi:dm Mini and 7GB for Kanana 8B. The three Mi:dm Base single 60 min runs in the 30 and 60 min table are the same failure (n_ctx about 36864, KV 6.75GB + model 6.5GB). The measurement setup is to blame, but even with n_ctx sized to a prompt twice the 60 min one (about 23 thousand tokens for Mi:dm, about 31 thousand for Kanana), the KV alone is about 4.2GB for Mi:dm Mini and about 3.9GB for Kanana 8B. Mi:dm Mini goes over the 8GB tier's cap, and Kanana 8B sits near the 9GB cap and at the end of its training context.

| Model | Tier | Mode | Run failed | Broken format | Decisions | One-off decisions | Wrong value | Action items | Owner right | Due date right | Open questions | Retracted as decision | Invented owner | Invented number | Time per meeting | Peak memory | Generation tok/s | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| midm-2.0-mini | 8GB | chunked | 0/3 | 0/3 | 51/122 (42%) | 43/83 (52%) | 2 | 56/80 (70%) | 51/56 (91%) | 56/56 (100%) | 13/19 (68%) | 0 | 11 | 0 | 140s | 3.4GB | 30.3 | — |
| midm-2.0-mini | 8GB | single | 3/3 | 0/0 | — | — | 0 | — | — | — | — | 0 | 0 | 0 | — | 11.0GB | — | run failed |
| qwen3.5-4b | 8GB | chunked | 0/3 | 0/3 | 90/122 (74%) | 68/83 (82%) | 0 | 74/80 (92%) | 73/74 (99%) | 74/74 (100%) | 8/19 (42%) | 0 | 3 | 0 | 268s | 3.2GB | 19.2 | — |
| qwen3.5-4b | 8GB | single | 0/3 | 0/3 | 49/122 (40%) | 43/83 (52%) | 5 | 47/80 (59%) | 37/47 (79%) | 47/47 (100%) | 2/19 (11%) | 0 | 0 | 0 | 247s | 4.7GB | 13.8 | out (memory) |
| kanana-1.5-8b | 16GB | chunked | 0/3 | 0/3 | 59/122 (48%) | 46/83 (55%) | 0 | 60/80 (75%) | 60/60 (100%) | 60/60 (100%) | 9/19 (47%) | 0 | 16 | 3 | 425s | 5.9GB | 14.7 | — |
| kanana-1.5-8b | 16GB | single | 3/3 | 0/0 | — | — | 0 | — | — | — | — | 0 | 0 | 0 | — | 11.7GB | — | run failed |
| qwen3.5-9b | 16GB | chunked | 0/3 | 0/3 | 80/122 (66%) | 53/83 (64%) | 7 | 72/80 (90%) | 66/72 (92%) | 72/72 (100%) | 10/19 (53%) | 0 | 15 | 2 | 446s | 5.9GB | 13.4 | — |
| qwen3.5-9b | 16GB | single | 0/3 | 0/3 | 53/122 (43%) | 51/83 (61%) | 10 | 56/80 (70%) | 50/56 (89%) | 56/56 (100%) | 2/19 (11%) | 0 | 0 | 0 | 298s | 7.5GB | 10.6 | — |

### How much n_ctx inflated single's measured memory (estimated, not run again)

llama.cpp reserves the KV cache for all of n_ctx up front. n_ctx needed = prompt + the longest output among that model's well-formed summaries + 64, rounded up to a multiple of 256. Estimated peak memory = measured − (n_ctx used − n_ctx needed) × KV/token. Compute buffers may shrink a little more with n_ctx, so this is an upper-side estimate.

| Model | Tier | Script | KV/token | n_ctx used | n_ctx needed | Measured peak memory | KV saved | Estimated peak memory | Cap |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| gemma-4-e4b | 8GB | customer-30m | 16KiB | 22528 | 8192 | 6.6GB | 0.2GB | 6.4GB | over |
| gemma-4-e4b | 8GB | one_on_one-30m | 16KiB | 22528 | 8192 | 6.6GB | 0.2GB | 6.4GB | over |
| gemma-4-e4b | 8GB | weekly-30m | 16KiB | 22528 | 8192 | 6.3GB | 0.2GB | 6.1GB | over |
| gemma-4-e4b | 8GB | customer-60m | 16KiB | 34816 | 14848 | 7.2GB | 0.3GB | 6.9GB | over |
| gemma-4-e4b | 8GB | one_on_one-60m | 16KiB | 34816 | 15104 | 7.2GB | 0.3GB | 6.9GB | over |
| gemma-4-e4b | 8GB | weekly-60m | 16KiB | 34816 | 14848 | 7.2GB | 0.3GB | 6.9GB | over |
| kanana-1.5-2.1b | 8GB | customer-30m | 128KiB | 22528 | 9216 | 4.4GB | 1.6GB | 2.8GB | under |
| kanana-1.5-2.1b | 8GB | one_on_one-30m | 128KiB | 22528 | 9216 | 4.4GB | 1.6GB | 2.8GB | under |
| kanana-1.5-2.1b | 8GB | weekly-30m | 128KiB | 22528 | 9216 | 4.4GB | 1.6GB | 2.8GB | under |
| kanana-1.5-2.1b | 8GB | customer-60m | 128KiB | 34816 | 16640 | 5.9GB | 2.2GB | 3.7GB | under |
| kanana-1.5-2.1b | 8GB | one_on_one-60m | 128KiB | 34816 | 16640 | 5.9GB | 2.2GB | 3.7GB | under |
| kanana-1.5-2.1b | 8GB | weekly-60m | 128KiB | 34816 | 16384 | 5.9GB | 2.2GB | 3.7GB | under |
| midm-2.0-mini | 8GB | customer-30m | 192KiB | 24576 | 6912 | 6.1GB | 3.2GB | 2.8GB | under |
| midm-2.0-mini | 8GB | one_on_one-30m | 192KiB | 24576 | 6656 | 6.1GB | 3.3GB | 2.8GB | under |
| midm-2.0-mini | 8GB | weekly-30m | 192KiB | 24576 | 6656 | 6.0GB | 3.3GB | 2.8GB | under |
| midm-2.0-mini | 8GB | customer-60m | 192KiB | 36864 | 11776 | 8.3GB | 4.6GB | 3.7GB | under |
| midm-2.0-mini | 8GB | one_on_one-60m | 192KiB | 36864 | 12032 | 8.3GB | 4.5GB | 3.8GB | under |
| midm-2.0-mini | 8GB | weekly-60m | 192KiB | 34816 | 11776 | 7.9GB | 4.2GB | 3.7GB | under |
| qwen3.5-4b | 8GB | customer-30m | 32KiB | 22528 | 10240 | 3.6GB | 0.4GB | 3.3GB | under |
| qwen3.5-4b | 8GB | one_on_one-30m | 32KiB | 22528 | 9984 | 3.6GB | 0.4GB | 3.3GB | under |
| qwen3.5-4b | 8GB | weekly-30m | 32KiB | 22528 | 10240 | 3.6GB | 0.4GB | 3.2GB | under |
| qwen3.5-4b | 8GB | customer-60m | 32KiB | 34816 | 16640 | 4.0GB | 0.6GB | 3.5GB | under |
| qwen3.5-4b | 8GB | one_on_one-60m | 32KiB | 34816 | 16896 | 4.0GB | 0.5GB | 3.5GB | under |
| qwen3.5-4b | 8GB | weekly-60m | 32KiB | 34816 | 16384 | 4.0GB | 0.6GB | 3.5GB | under |
| qwen3.5-4b | 8GB | customer-120m | 32KiB | 57344 | 29440 | 4.7GB | 0.9GB | 3.9GB | under |
| qwen3.5-4b | 8GB | one_on_one-120m | 32KiB | 57344 | 29440 | 4.7GB | 0.9GB | 3.9GB | under |
| qwen3.5-4b | 8GB | weekly-120m | 32KiB | 57344 | 29696 | 4.7GB | 0.8GB | 3.9GB | under |
| ax-4.0-light | 16GB | customer-30m | 56KiB | 22528 | 6144 | 5.6GB | 0.9GB | 4.7GB | under |
| ax-4.0-light | 16GB | one_on_one-30m | 56KiB | 22528 | 5888 | 5.6GB | 0.9GB | 4.7GB | under |
| ax-4.0-light | 16GB | weekly-30m | 56KiB | 22528 | 6144 | 5.5GB | 0.9GB | 4.6GB | under |
| ax-4.0-light | 16GB | customer-60m | 56KiB | 34816 | 11008 | 6.2GB | 1.3GB | 5.0GB | under |
| ax-4.0-light | 16GB | one_on_one-60m | 56KiB | 34816 | 11008 | 6.2GB | 1.3GB | 4.9GB | under |
| ax-4.0-light | 16GB | weekly-60m | 56KiB | 34816 | 10752 | 6.2GB | 1.3GB | 4.9GB | under |
| kanana-1.5-8b | 16GB | customer-30m | 128KiB | 22528 | 9216 | 7.6GB | 1.6GB | 5.9GB | under |
| kanana-1.5-8b | 16GB | one_on_one-30m | 128KiB | 22528 | 9216 | 7.6GB | 1.6GB | 5.9GB | under |
| kanana-1.5-8b | 16GB | weekly-30m | 128KiB | 22528 | 9216 | 7.5GB | 1.6GB | 5.9GB | under |
| kanana-1.5-8b | 16GB | customer-60m | 128KiB | 34816 | 16384 | 9.0GB | 2.2GB | 6.8GB | under |
| kanana-1.5-8b | 16GB | one_on_one-60m | 128KiB | 34816 | 16640 | 9.0GB | 2.2GB | 6.8GB | under |
| kanana-1.5-8b | 16GB | weekly-60m | 128KiB | 34816 | 16384 | 9.0GB | 2.2GB | 6.8GB | under |
| midm-2.0-base | 16GB | customer-30m | 192KiB | 24576 | 6656 | 11.2GB | 3.3GB | 7.9GB | under |
| midm-2.0-base | 16GB | one_on_one-30m | 192KiB | 24576 | 6656 | 11.2GB | 3.3GB | 7.9GB | under |
| midm-2.0-base | 16GB | weekly-30m | 192KiB | 24576 | 6656 | 11.1GB | 3.3GB | 7.8GB | under |
| qwen3.5-9b | 16GB | customer-30m | 32KiB | 22528 | 9216 | 6.3GB | 0.4GB | 5.9GB | under |
| qwen3.5-9b | 16GB | one_on_one-30m | 32KiB | 22528 | 9216 | 6.4GB | 0.4GB | 6.0GB | under |
| qwen3.5-9b | 16GB | weekly-30m | 32KiB | 22528 | 9472 | 6.3GB | 0.4GB | 5.9GB | under |
| qwen3.5-9b | 16GB | customer-60m | 32KiB | 34816 | 15872 | 6.7GB | 0.6GB | 6.1GB | under |
| qwen3.5-9b | 16GB | one_on_one-60m | 32KiB | 34816 | 15872 | 6.7GB | 0.6GB | 6.1GB | under |
| qwen3.5-9b | 16GB | weekly-60m | 32KiB | 34816 | 15616 | 6.7GB | 0.6GB | 6.1GB | under |
| qwen3.5-9b | 16GB | customer-120m | 32KiB | 57344 | 28672 | 7.5GB | 0.9GB | 6.6GB | under |
| qwen3.5-9b | 16GB | one_on_one-120m | 32KiB | 57344 | 28672 | 7.5GB | 0.9GB | 6.6GB | under |
| qwen3.5-9b | 16GB | weekly-120m | 32KiB | 57344 | 28928 | 7.4GB | 0.9GB | 6.5GB | under |

### New baseline (v2 scripts), a better prompt, the speech route

The v2 scripts never plant the same item template twice in one meeting. Only qwen3.5-4b chunked was measured. p2 adds three rules to the first prompt: a deciding word (확정, 승인, 유지, 진행, 채택: confirmed, approved, kept, go ahead, adopted) makes a decision; a decision is written down even when an action item was also set; numbers and dates are copied as transcribed, and a changed value is written as its last value. The deciding words are common words, not the script generator's sentences.

The speech route (asr) reads the v2 scripts aloud with six macOS Korean voices in turn (`asr_meetings.py`), transcribes them with the app's after-meeting final pass (`final_pass`, Silero spans + large-v3-turbo), and summarises that transcript. Scoring uses the scripts' answer key as is.

| Model | Tier | Mode | Run failed | Broken format | Decisions | One-off decisions | Wrong value | Action items | Owner right | Due date right | Open questions | Retracted as decision | Invented owner | Invented number | Time per meeting | Peak memory | Generation tok/s | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-4b | 8GB | chunked | 0/6 | 0/6 | 69/100 (69%) | 69/100 (69%) | 1 | 58/66 (88%) | 56/58 (97%) | 58/58 (100%) | 10/20 (50%) | 1 | 2 | 0 | 101s | 3.2GB | 20.0 | — |
| qwen3.5-4b | 8GB | chunked · asr | 0/6 | 0/6 | 48/100 (48%) | 48/100 (48%) | 2 | 41/66 (62%) | 28/41 (68%) | 41/41 (100%) | 5/20 (25%) | 1 | 20 | 5 | 98s | 3.2GB | 19.9 | — |
| qwen3.5-4b | 8GB | chunked · asr · p2 | 0/6 | 0/6 | 48/100 (48%) | 48/100 (48%) | 3 | 44/66 (67%) | 32/44 (73%) | 44/44 (100%) | 4/20 (20%) | 2 | 18 | 4 | 99s | 3.2GB | 19.6 | — |
| qwen3.5-4b | 8GB | chunked · p2 | 0/6 | 0/6 | 74/100 (74%) | 74/100 (74%) | 1 | 59/66 (89%) | 58/59 (98%) | 59/59 (100%) | 12/20 (60%) | 0 | 4 | 0 | 103s | 3.2GB | 19.7 | — |

A candidate "확인 필요" (needs checking) flag: for each span, chunked checks that a decision's numbers (down to a one-letter unit, "11월", "250만") appear as is in that span's transcript, and flags the decision if not (`summary_score.unsupported_numbers`).

| Condition | Decisions flagged | Of which wrong value | Wrong value in all | Share of wrong values caught |
|---|---:|---:|---:|---:|
| asr · p1 | 0 | 0 | 2 | 0% |
| asr · p2 | 0 | 0 | 3 | 0% |
| p1 | 0 | 0 | 1 | 0% |
| p2 | 0 | 0 | 1 | 0% |

The flag never fired. Every number in a wrong value was in that span's transcript: on the answer scripts the summary moved another number from the same agenda item (bugs left, the first offer that was withdrawn) into the decision; on the speech route the transcription changed a number ("28일"→"18일", "20일"→"21일") and the summary copied it. A check against the transcript cannot catch either.

Speech route transcripts: 6 meetings, 355 min of speech, CER 5.3% (meeting mean), processing RTF 0.198. CER is low, but the wrong letters gather in feature names and people's names ("주소록 동기화"→"수소로 전기화", contact sync heard as nonsense; 20 invented owners). Of the 52 decisions p1 missed on the speech route, 18 had their keywords nowhere in the transcript; for the rest the transcript had them right somewhere, but the summary used a misheard name from elsewhere or left them out. Counting names one letter off as right still gives only 48→51.

### Vocabulary (STT-7): the same recordings without and with a vocabulary

The speech route's recordings were kept (`asr_meetings.py --keep-audio`) and the final pass was given a vocabulary as whisper's initial prompt (`--vocabulary`, `WhisperDecoder::with_vocabulary`). The vocabulary is what a user would plausibly type: the people's names in the meeting, 1–2 feature names never said in the meeting (decoys), the company name, then the most frequent feature names, up to 100 characters. For names and terms we counted how many of the times said in the script were heard in the transcript ("heard"), and separately how many times one was heard more often than said ("extra") and how many times a decoy was heard. Summaries are 4B chunked p2.

| Condition | Names and terms heard | Of which in vocabulary | Extra | Decoys heard | CER | Decoding | Decisions | Action items | Owner right | Invented owner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no vocabulary | 209/321 (65%) | 145/232 (62%) | 0 | 0 | 5.3% | 4214s | 48/100 | 44/66 | 32/44 | 18 |
| vocabulary | 238/321 (74%) | 180/232 (78%) | 14 | 0 | 5.1% | 4289s | 57/100 | 39/66 | 36/39 | 10 |

Without a vocabulary the transcript matched the earlier speech route run to the letter (no regression). The vocabulary raised names and terms from 65%→74% and decisions from 48→57, and decoding took 1.8% longer. No decoy was ever heard. The 14 extras are of two kinds: one short span repeated a vocabulary name as "지호, 지호, 지호, 지호." (4 times), and the other 10 heard another vocabulary name where a different name was said ("시온 님이" was "시원님이" without the vocabulary and "지호님이" with it). Getting it wrong with a real attendee's name is more believable than a name that does not exist, so it does more harm, but owner right went up overall.

The echo filter (`blocklist::is_vocabulary_echo`: drop a span made only of three or more vocabulary terms in a row) dropped only that one span in this vocabulary transcript, and nothing in the transcript without a vocabulary or in the script sentences.

The 10 action items lost in the vocabulary run (5 net, after the newly found ones): in 3 the transcript changed ("다음 주 화요일", next Tuesday, became "다음 주와 요일", so the date was not heard). In 7 the keywords were in the transcript as is, but the summary mixed up the date with another action item from the same template ("배포 전 점검표 정리", "견적 다시 요청"). None came from score matching. Greedy decoding gives the same output for the same input, so the difference is a small change in the transcript shaking the summary.

### The app's summary: the merge step and evidence spans (SUM-3)

The same 4B chunked p2 was run through the Rust path the app uses (`examples/summarise`, `app_summary.py`). The prompt is the same and decoding is greedy, so the model's answer for each span is the same as in bench; only the merge differs. The app's merge lets a later value replace an earlier one for the same item, moves deferred decisions to open questions, and in a meeting with chair rulings ("계속 심사", held over for further review; "가결되었음을 선포", declared passed) moves the decisions of a held-over item's spans to open questions. Evidence right is the share of found items whose evidence line (with one line either side) states the item's value (or its keyword when it has no value).

| Script | Merge | Decisions | Wrong value | Retracted as decision | Open questions | Action items | Decision evidence right | Action item evidence right |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2 | bench | 74/100 | 1 | 0 | 12/20 | 59/66 | — | — |
| v2 | app | 74/100 | 1 | 0 | 12/20 | 59/66 | 74/74 (100%) | 59/59 (100%) |
| asr | bench | 48/100 | 3 | 2 | 4/20 | 44/66 | — | — |
| asr | app | 48/100 | 3 | 2 | 4/20 | 44/66 | 45/48 (94%) | 44/44 (100%) |

On the synthetic scripts the app's merge did not change the scores. It changed one thing only: it moved a decision that read "다음 회의 때 다시 보며 오늘 정하지 않습니다" (we will look again at the next meeting and not decide today) to open questions. A retracted value left as a decision was written by the model inside one span, so the merge cannot see it (one on the speech route was the transcription missing the final amount). The scorer was counting "500만 원에서 400만 원으로 확정" (confirmed at 4 million won, down from 5 million) as a retracted value. It now reads "에서 + number" (from + number) as a changed value, and every table was scored again (12 "retracted as decision" fewer across 9 results).

In a meeting with chair rulings (one set of National Assembly subcommittee minutes, run locally only; neither the text nor the summary is committed), the app's merge on the same model answers: moved the 18 decisions in the two held-over items' spans to open questions (the error of writing a held-over bill as "통과", passed, went away), moved only 2 deferred clauses inside passed items to open questions, and put the chair's ruling sentences in the list with their evidence lines. Decisions 75→62, open questions 16→38.

## 9. Diarization (requirements §12 risk 1)

We measured sherpa-onnx's after-meeting diarization (pyannote segmentation-3.0, one speaker embedding, clustering), the app's candidate, for each embedding model. Data, models, versions and the commands to run it again are in the diarization section of `bench/README.md`; the result JSONs are in `bench/results/diar/`. Each result JSON records the sherpa-onnx, onnxruntime, scipy and numpy versions and the sha256 of the model files. Results made before this was recorded (tune, korean and ami `.th`, `.absorb`, `.oracle`, `.online`, long) have no versions. The default score is DER (collar 0.25 s, overlap included); the value in brackets is collar 0. DER is computed per meeting and averaged.

**Data and limits**
- Korean: synthetic meetings made by joining read sentences from Zeroth-Korean (OpenSLR 40, CC BY 4.0). Scene (room: one microphone, reverb and noise floor at a different distance per speaker / call: the far side of a call, 300–3400 Hz band limit, mu-law, light noise) × 2, 4 or 6 speakers, 30 min. 10% of utterances overlap, and 15% of turns are a listener's short reply of 0.3–0.9 s. test is 12 meetings (6 hours) made from Zeroth test speakers. dev is 6 meetings (3 hours) with the same make-up, made from 12 train speakers who do not overlap with test. Sentence clips are reused from the same speaker pool, so the same clip can repeat even within one meeting. Each speaker's clips come from one recording setup, so the channel can give the speaker away (room's per-speaker reverb blurs this but does not remove it). **So the Korean test may be optimistic.** It is read speech, not real conversation, and the short replies are pieces of sentences, not real "네" (yes).
- AMI: 16 real English meetings (test split, the only_words reference and UEM from BUT AMI-diarization-setup), two microphones: Mix-Headset and Array1-01. 3–4 speakers per meeting, 63 in all. If AMI and Korean rank differently, trust AMI (real conversation, real differences in how much people talk).
- No real meeting recordings were used.
- The Korean results in this section come from meetings made with the mixer before 6aa83d1 (results commit 167a9b8). 6aa83d1 changed the call scene's band limit, which changed the end of call meetings' output a little, so check out 167a9b8 to make them again (the diarization section of `bench/README.md`).

**Candidate models** (sherpa's default path `.th`, with only the threshold chosen on a 5 min dev. Clustering is complete, no absorb. Extraction RTF is on the Korean test)

| Model | Licence | threshold | AMI Headset | AMI Array1 | Korean test | Speakers ± (Korean) | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| CAM++ zh_en (3D-Speaker) | Apache-2.0 | 0.7 | 0.329 | 0.475 | 0.401 | +17.2 | 0.033 |
| ERes2NetV2 zh (3D-Speaker) | Apache-2.0 | 0.8 | 0.253 | 0.288 | 0.331 | +10.6 | 0.134 |
| TitaNet small (NeMo) | CC-BY-4.0, attribution required | 0.8 | 0.284 | 0.407 | 0.343 | +13.3 | 0.037 |
| WeSpeaker r34 LM | CC-BY-4.0, attribution required | 0.5 | 0.541 | 0.636 | 0.450 | +3.8 | 0.055 |
| ERes2Net en (3D-Speaker) | Apache-2.0 | 0.9 | 0.280 | 0.395 | 0.383 | +18.0 | 0.071 |

- ERes2Net en was dropped from the next stage. Its DER is worse than ERes2NetV2 from the same family on every data set, and it is twice as slow as CAM++ and TitaNet.
- Gated models (pyannote community-1, Sortformer) were not measured.
- WeSpeaker: our file is the official wespeaker voxceleb_resnet34_LM as re-released in ONNX by sherpa. Its sha256 differs from the hbredin release that pyannote 3.1 uses (e9848563… vs 7bb2f06e…). We checked that both come from the same original, but **that the weights are the same is assumed**. pyannote 3.1's config.yaml is gated, so we could not see the actual clustering settings (linkage, threshold, min_cluster_size) (assumed). From the public source (pyannote.audio 3.1.1) we confirmed three things. AgglomerativeClustering has method, threshold and min_cluster_size (a number of embeddings) as hyperparameters and reassigns small clusters to the large cluster with the nearest centroid. When the embeddings exceed max_num_embeddings it subsamples them at random. The code default of `embedding_exclude_overlap` is False.
- Published reference values (not targets): pyannote 3.1's model card gives AMI Headset 18.8% (FA 3.6 / Miss 9.5 / Conf 5.7) and Array1 22.4%, collar 0, overlap included. We did not check that the evaluation split and reference rules are the same as ours, so we do not compare them directly with our numbers.

**The oracle is not an upper bound.** `.oracle` (CAM++), given the true number of speakers, is worse than absorb in places (AMI Headset 0.229 vs 0.152). With a fixed number of speakers sherpa cuts complete linkage at that count, so short fragment clusters take up the count and two real speakers merge into one. We confirmed this by printing the total speech per cluster at the fixed count, and confirmed that both paths are scored by the same scorer with the same UEM. It is not a bug.

### Absorb (exploratory)

The sherpa path's results inflate the speakers a lot (AMI Headset +64 on average). We wrote a post-process (`diar_post.absorb`) that folds these short fragment clusters into the nearest large cluster. At first it was a floor in seconds, "under N s of total speech". **We switched to a share after seeing the seconds floor fail on test.** The design changed after looking at test, so the numbers below are exploratory, not confirmatory. After that we rebuilt dev with the same length and make-up as test (30 min, room/call × 2, 4 or 6 speakers) and chose the share and threshold on that dev only. The chosen values are threshold 0.9 and share 8%. 8% was at the top end of the grid, so the next grid (B) widened it to 16%.

| Data (CAM++) | `.th` | Speakers ± | + absorb 8% | Speakers ± | `.oracle` | True speakers |
|---|---:|---:|---:|---:|---:|---:|
| AMI Headset | 0.329 | +64.4 | 0.152 | +2.3 | 0.229 | 63 (16 meetings) |
| AMI Array1 | 0.475 | +78.6 | 0.242 | +2.4 | 0.288 | 63 (16 meetings) |
| Korean test | 0.401 | +17.2 | 0.316 | −0.8 | 0.340 | 48 (12 meetings) |
| Korean room | 0.483 | +27.2 | 0.127 | +0.5 | 0.211 | 24 (6 meetings) |
| Korean call | 0.319 | +7.2 | 0.505 | −2.0 | 0.468 | 24 (6 meetings) |

room gets much better and call gets worse. 6-speaker calls folded into 2 (DER 0.64–0.66). In the call band the embedding distance between speakers narrows, so threshold 0.9 seems to put different people in one cluster and absorb folds what is left even further (assumed, not measured separately).

### Reproduction check: sherpa's steps rewritten in numpy

To try other clustering methods we rewrote sherpa's after-meeting path as `bench/diar_pipe.py`. Embeddings are extracted once and stored (`bench/data/`, not committed), and clustering uses scipy (BSD-3-Clause). The bar before moving on to the grid was "within 0.5%p DER of sherpa, same number of speakers". The results are complete 0.7, 2 dev meetings × 4 models.

| Meeting | Model | DER sherpa | DER ours | Speakers sherpa | Speakers ours |
|---|---|---:|---:|---:|---:|
| room 2 speakers | CAM++ | 0.1080 | 0.1081 | 13 | 13 |
| room 2 speakers | ERes2NetV2 | 0.1795 | 0.1792 | 14 | 14 |
| room 2 speakers | TitaNet | 0.2513 | 0.2511 | 15 | 14 |
| room 2 speakers | WeSpeaker | 0.5047 | 0.5047 | 1 | 1 |
| room 4 speakers | CAM++ | 0.3768 | 0.3764 | 28 | 28 |
| room 4 speakers | ERes2NetV2 | 0.3632 | 0.3621 | 31 | 31 |
| room 4 speakers | TitaNet | 0.5077 | 0.5085 | 29 | 28 |
| room 4 speakers | WeSpeaker | 0.7273 | 0.7273 | 1 | 1 |

- The first run missed the bar (CAM++ 4 speakers 0.3768 vs 0.3975, 28 vs 32 speakers). The cause was cluster numbering. sherpa numbers clusters in order of first appearance; scipy fcluster numbers them differently. In the top-k that picks the speakers for each frame, a tie goes to the lower number, so a different numbering alone changes the result. The clustering itself was the same (sherpa FastClustering and scipy complete gave the same partition 8/8). sherpa was deterministic (1 and 4 threads, repeated runs identical). Our segmentation also changed 0 frames with 1 thread and graph optimisation off. The ORT version difference (sherpa's bundled 1.28.2 vs 1.31.0) was not the cause either. The table above is after fixing the numbering to order of first appearance.
- **The only difference left is how libc++ `std::partial_sort` breaks ties when k≥2.** Example: in the TitaNet 2-speaker meeting at 422.04–422.24 s the frame's speaker count was 2 and the votes were {3:3, 1:2, 2:2, 5:2}. We picked 1, sherpa picked 2. "Keep the earlier number" holds only for k=1. Because of this sherpa gets one more 0.30 s tail speaker (15 vs 14, 29 vs 28).
- **Our rule, "lower number first", is explicit and deterministic.**
- We checked three conditions for accepting it. (a) The difference only appears in the over-split state with no floor. (b) The DER difference is 0.02 and 0.08%p. (c) With absorb on, the speaker counts should match, but in the 2-speaker meeting (absorb 8%) sherpa had 4 and we had 3 (DER 0.1366 vs 0.1364). **(c)'s difference of 2 comes from `diar_post.absorb` leaving in place a 0.30 s tail it cannot extract an embedding for** (no centroid, so nowhere to fold it). We judged it a pass.

### Clustering grid (B, 30 min dev, exploratory)

Four models (CAM++, ERes2NetV2, TitaNet, WeSpeaker) × linkage (complete, average, centroid) × threshold (0.10–1.50) × floor (a count of 1–20, or a share of the embeddings of 0.5–16%). Clusters under the floor go to the large cluster with the nearest centroid (the pyannote way). centroid linkage runs on L2-normalised embeddings and the threshold is read on the cosine scale (d_euc = √(2·d_cos)). For the two models whose optimum touched the bottom end of the threshold (0.10) (CAM++, WeSpeaker), we also ran 0.02–0.08 for that model only.

Per-scene values were allowed too. The app knows that sound on the system channel means a call and the microphone alone means an in-person meeting. Per-scene values were chosen only on the 3 dev meetings of the same scene, so **the risk of overfitting is high**.

Two columns are counted alongside.
- **erased** = the number of absorbed clusters in which one real speaker holds over 80% (a pure fragment got folded). Counted even if that speaker has another large cluster.
- **vanished** = the number of real speakers who are not the one who talks most in any final cluster. Counted from the reference speech under the embedding spans.

| Model | Shared value (6 meetings) | dev DER | Speakers ± | vanished | Shared value room / call | room value | call value | sherpa-like |
|---|---|---:|---:|---:|---|---|---|---:|
| CAM++ | average 0.25, 3% | 0.129 | −0.33 | 2/24 | 0.080 / 0.177 | centroid 0.1, 1%: 0.077 | average 0.2, 5%: 0.099 | 0.170 (+2.83) |
| ERes2NetV2 | complete 0.8, 8% | 0.139 | −0.17 | 2/24 | 0.144 / 0.134 | complete 0.9, 8%: 0.078 | complete 0.6, 8%: 0.124 | 0.197 (+8.67) |
| TitaNet | complete 0.8, 8% | 0.144 | −0.17 | 1/24 | 0.142 / 0.147 | complete 0.85, 8%: 0.102 | complete 0.75, 8%: 0.146 | 0.205 (+9.0) |
| WeSpeaker | centroid 0.04, 2% | 0.255 | −1.17 | 7/24 | 0.246 / 0.263 | centroid 0.02, 2%: 0.137 | average 0.15, 12%: 0.253 | 0.408 (+2.0) |

- With a pyannote-style count floor (12), the shared optimum is CAM++ 0.171, ERes2NetV2 0.185, TitaNet 0.190, WeSpeaker 0.296. That is 4–5 points worse than the share floor, with speakers left over. The count grid's optimum also sat at the top end (20). The floor should be a share that scales with the meeting's length.
- erased runs into the thousands when average or centroid is used at a low threshold. Most embeddings then form a cluster of their own, and one person's fragments get folded. Whether a person really vanished is read from vanished.
- Height inversions in centroid linkage (sum over 6 meetings): CAM++ 2086, ERes2NetV2 1759, TitaNet 1648, WeSpeaker 3221. complete and average have 0. In a dendrogram with inversions, cutting at a threshold loses its meaning, so **the centroid values stay in the table but are dropped from the app candidates.**
- The widened grid: CAM++ room's optimum stayed at 0.10 (not an end effect). WeSpeaker went down to 0.04 and 0.02, but at 0.02 the clusters before the floor are 29% (room) to 60% (call) of the embeddings, a degenerate state (below).
- Degeneracy check (dev): at CAM++ average 0.25 there are 321–459 clusters before the floor in room and 164–230 in call (5–15% of about 3,200 embeddings per meeting). That is not close to the number of embeddings, so it is not fully degenerate. But the 3% floor moves 25–52% of embeddings in room and 9–14% in call, and in all 6 meetings "clusters over the floor = final speakers". complete 0.9 (8%) has 4–23 clusters before the floor and moves 1–11%, so the cut does the work. WeSpeaker centroid 0.02 had no cluster over the floor in call dev, so nothing was folded (about 2,000 speakers).

### Preregistration and test results

The values for test and AMI were written before looking at test into `bench/results/diar/preregistered.json` (2026-10-21 06:48 +0900, written once and never overwritten), and that file reached the public repository before the test results. The `pipe` step reads values from this file only. The headline is each model's shared value, the secondary is the per-scene values (the room value for AMI), and the baseline is sherpa-like (complete, no absorb, dev-optimal threshold). The app candidate is one: CAM++ average 0.25, 3% (no inversions). The table below is as registered; no candidate was added or changed after seeing the results.

| Model | Series | Setting | AMI Headset | Speakers ± | vanished | AMI Array1 | Speakers ± | vanished | Korean test | Speakers ± | vanished |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAM++ | headline | average 0.25, 3% | **0.181** (0.246) | −0.3 | 9/63 | 0.335 (0.391) | −0.1 | 17/63 | 0.274 (0.316) | −0.2 | 7/48 |
| CAM++ | per scene | room centroid 0.1, 1% / call average 0.2, 5% | 0.159 (0.225) | +0.0 | 7/63 | 0.234 (0.298) | −0.2 | 13/63 | 0.259 (0.301) | +1.2 | 4/48 |
| CAM++ | sherpa-like | complete 0.9 | 0.214 (0.292) | +27.2 | 0/63 | 0.320 (0.398) | +32.4 | 0/63 | 0.345 (0.382) | +4.2 | 8/48 |
| ERes2NetV2 | headline | complete 0.8, 8% | 0.190 (0.251) | −0.1 | 8/63 | **0.218** (0.283) | −0.1 | 9/63 | **0.261** (0.304) | +0.2 | 2/48 |
| ERes2NetV2 | per scene | room complete 0.9, 8% / call complete 0.6, 8% | 0.181 (0.243) | +0.1 | 6/63 | 0.229 (0.292) | −0.4 | 10/63 | 0.245 (0.288) | +0.2 | 2/48 |
| ERes2NetV2 | sherpa-like | complete 0.8 | 0.253 (0.329) | +33.8 | 0/63 | 0.286 (0.364) | +36.2 | 0/63 | 0.333 (0.375) | +10.8 | 0/48 |
| TitaNet | headline | complete 0.8, 8% | 0.194 (0.256) | −0.4 | 12/63 | 0.313 (0.368) | −1.1 | 22/63 | 0.299 (0.339) | +0.1 | 5/48 |
| TitaNet | per scene | room complete 0.85, 8% / call complete 0.75, 8% | 0.196 (0.259) | −0.1 | 10/63 | 0.285 (0.343) | −0.9 | 20/63 | 0.262 (0.305) | +0.1 | 3/48 |
| TitaNet | sherpa-like | complete 0.8 | 0.284 (0.365) | +49.0 | 0/63 | 0.406 (0.488) | +62.2 | 0/63 | 0.343 (0.383) | +13.3 | 0/48 |
| WeSpeaker | headline | centroid 0.04, 2% | 0.792 (0.827) | +294.1 | 4/63 | 0.498 (0.536) | −0.9 | 33/63 | 0.337 (0.374) | −0.3 | 10/48 |
| WeSpeaker | per scene | room centroid 0.02, 2% / call average 0.15, 12% | 0.996 (1.022) | +574.1 | 0/63 | 0.874 (0.904) | +434.9 | 11/63 | 0.382 (0.417) | −1.2 | 16/48 |
| WeSpeaker | sherpa-like | complete 0.55 | 0.492 (0.567) | +80.1 | 0/63 | 0.611 (0.657) | +21.3 | 5/63 | 0.442 (0.475) | +2.1 | 9/48 |

Korean test by scene (headline / per scene / sherpa-like, DER):

| Model | room | call |
|---|---|---|
| CAM++ | 0.198 / 0.221 / 0.185 | 0.351 / 0.297 / 0.505 |
| ERes2NetV2 | 0.253 / 0.170 / 0.380 | 0.269 / 0.319 / 0.285 |
| TitaNet | 0.210 / 0.147 / 0.279 | 0.388 / 0.378 / 0.407 |
| WeSpeaker | 0.301 / 0.280 / 0.409 | 0.373 / 0.484 / 0.476 |

**Ranking.** The preregistration only said "rank by the headline's AMI result" and did not say which microphone. Choosing one now would be a post hoc decision, so both rankings are written side by side and neither is called the preregistered ranking.
- Headset alone: CAM++ (0.181) ≥ ERes2NetV2 (0.190) ≈ TitaNet (0.194).
- Array1 alone: ERes2NetV2 (0.218) > TitaNet (0.313) ≈ CAM++ (0.335).
- Tied to the app's scenes: an in-person meeting is far-field speech heard by one laptop microphone, close to Array1. A call's remote speakers are clean speech coming in on the system channel, close to Headset. Conclusion: **in in-person meetings ERes2NetV2 leads clearly (0.218 vs 0.335), and on clean speech the three are about even.**
- WeSpeaker is dropped from the candidates. Its degenerate centroid value made up to 1,185 speakers per meeting on AMI, and it is the worst even in the non-degenerate sherpa-like setting.
- Per-scene values do not generalize. On test the per-scene value is worse than the headline in places (CAM++ room 0.198→0.221, ERes2NetV2 call 0.269→0.319). They were chosen on 3 dev meetings.

**This measurement does not settle the final model.** Three reasons:
1. The gap between dev and test is large. The CAM++ headline went from 0.129 on dev to 0.274 on the Korean test. Six 30 min dev meetings are too few to choose with.
2. The share floor erases people. The speaker count is almost right (±0.4), but on AMI 8–22 of the 63 true speakers vanished. sherpa-like has 0 vanished but +27 to +62 speakers. Even in 4-person AMI meetings people talk very different amounts, so quiet people fall under the floor.
3. The app's structure (processing in pieces, separate mic/system channels) is a condition not measured yet.

The chunked prototype gets both CAM++ (speed) and ERes2NetV2 (far-field accuracy).

For reference (exploratory, not in the ranking): the earlier absorb (CAM++ threshold 0.9 + `diar_post` 8%) gives AMI Headset 0.152 and Array1 0.242, better than the headline. It was not preregistered.

### Live vs after the meeting, a 4-hour meeting

**Live (online)**: diarize each 60 s piece on its own and link the piece's speakers to the speakers so far by embedding similarity (link threshold 0.4/0.5/0.6). Labels once given are not changed. relabelled is the share of transcript spans whose speaker changes when the after-meeting result (`.th`) replaces the live one. CAM++; on AMI only Headset was measured.

| Data | `.th` | online 0.4 | online 0.5 | online 0.6 | relabelled |
|---|---:|---:|---:|---:|---:|
| AMI Headset | 0.329 | 0.233 | 0.147 | 0.145 | 41.5–53.1% |
| Korean test | 0.401 | 0.552 | 0.508 | 0.442 | 48.1–56.3% |

The rule set beforehand was "if live is more than 3 DER points worse or relabelled is over 5%, after-meeting diarization is the default". On Korean, live is 4–15 points worse and relabelled is over 40% in every case, so **after-meeting diarization is the default.** If live speaker labels go in, they are shown as provisional. On AMI, though, live is better than `.th`. That is a sign that diarizing each piece and then linking globally resists over-splitting better than cutting the whole meeting with complete at once, and it is the reason for the chunked prototype below. relabelled was measured against an over-split `.th`, so it may have come out larger than it really is.

**A 4-hour meeting** (Korean room with 4 speakers, synthetic, sherpa's default path `.th`, a separate process per model):

| Model | Processing time | RTF | Peak memory | Speakers (true 4) | DER |
|---|---:|---:|---:|---:|---:|
| CAM++ | 9.0 min | 0.038 | 6.0GB | 86 | 0.442 |
| ERes2NetV2 | 35.8 min | 0.149 | 5.5GB | 75 | 0.498 |
| TitaNet | 9.8 min | 0.041 | 6.5GB | 61 | 0.369 |
| WeSpeaker | 14.7 min | 0.061 | 5.8GB | 10 | 0.437 |
| ERes2Net en | 18.9 min | 0.079 | 6.1GB | 83 | 0.544 |

- The memory comes from the clustering's distance matrix. 4 hours gives N ≈ 35,900 embeddings, and the condensed distance matrix (N(N−1)/2 doubles) is 5.16GB, matching the measured peaks of 5.5–6.7GB. average and complete linkage in scipy need the same matrix, so clustering a 4-hour meeting in one go over the whole meeting takes the app over 6GB. Splitting into pieces and clustering only the piece centroids again globally puts a cap on it (the next stage).
- The app's final pass (large-v3-turbo, section 7 RTF 0.094) takes about 22.6 min for 4 hours. Running diarization after it, after-meeting processing for a 4-hour meeting is CAM++ about 9 min + 22.6 min ≈ **32 min** and ERes2NetV2 about 36 min + 22.6 min ≈ **58 min**. The former runs on Metal and the latter on CPU, so there is room to overlap them, but that was not measured.
- Processing time: AMI's `.th` processing times are left out because other runs were going at the same time; only the Korean values in the table above are used. The RTF of the test extraction (`extract-test`, 24.1 hours of embeddings) is CAM++ 0.035, TitaNet 0.037, WeSpeaker 0.054, ERes2NetV2 0.128. A few minutes of it overlapped with the dev degeneracy check script.

### Chunked prototype: dev grid (CAM++, ERes2NetV2)

Cut the meeting into time pieces, cluster each piece, link the pieces' speakers globally (average linkage between centroids), and put a global floor on the linked speakers' total speech (`bench/diar_chunk.py`, `bench/run_chunk.py`). dev is ten 30 min meetings (seeds 300 and 301) with a late joiner, a quiet speaker (1–3% of the speech) and an early leaver, and two 4-hour meetings (room and call). call keeps mic and system apart as the app does: the mic is "me" whole, and only system is diarized. The score is the mean of the 30 min dev mean DER and the 4-hour dev DER (collar 0.25). Results are in `bench/results/diar/chunk/grid/`, the preregistration in `bench/results/diar/chunk/preregistered.json` (first version in 56b700a; 2b660dd fixed the gate wording and one limit line before test was built).

| | CAM++ | ERes2NetV2 |
|---|---:|---:|
| Preregistered setting | 1200 s pieces, 60 s overlap, average 0.15, link 0.25, floor 2% | 1200 s pieces, 60 s overlap, average 0.15, link 0.3, floor 1% |
| dev score (best of the first grid) | 0.116 (0.135) | 0.086 (0.091) |
| 30 min dev DER / vanished (special speakers) | 0.123 / 6(5) | 0.080 / 1(1) |
| 4-hour dev DER / vanished (special speakers) | 0.110 / 2(2) | 0.093 / 0(0) |
| Top 1% score range (62 settings) | 0.116–0.159 | 0.086–0.114 |
| Cross seed (within the widened 480): chosen on s300, applied to s301 (s301's own best) | 0.206 (0.101) | 0.126 (0.086) |
| Cross seed: chosen on s301, applied to s300 (s300's own best) | 0.098 (0.089) | 0.101 (0.057) |

- **Finding the speakers in a call splits the two.** This is written apart from the ranking criterion, which puts in-person first. With its preregistered setting ERes2NetV2 got the speaker count right in all 6 dev calls (DER 0.02–0.12). CAM++ found only 4 in the 6-speaker call (DER 0.30) and 3 in the 4-speaker call with a quiet speaker (DER 0.35). With the first grid's best it found 3 in the 6-speaker call (DER 0.47).
- **ERes2NetV2 leads on the dev score, on finding speakers in calls, and across seeds.** CAM++ drops a lot in one cross-seed direction (the setting chosen on s300 gives 0.206 on s301, where s301's own best is 0.101): it is sensitive to the setting. The drop in the other direction is 1 point for CAM++ and 4–4.4 points for ERes2NetV2, so ERes2NetV2 is also optimistic by over 4 points in both directions. We expect the dev best to be optimistic on test.
- **The 30 min best and the 4-hour best differ.** In the first grid, choosing on the 30 min dev alone lost 3.5–4 points on the 4-hour dev. So the two are mixed with equal weight.
- **The grid was widened once only.** The first grid's best (5,760 settings per model) sat at the lowest values of link, local threshold and piece share, so the axes that were effectively unanimous in the top 1% (average, piece share 0, cannot-link off) were fixed, and link was widened to 0.15–0.3, local average to 0.1–0.2 and pieces to 600–1800 s (480 settings, of which 40 were the same settings as in the first grid and scored exactly the same). We decided beforehand not to chase a new end, and both models' best came out inside.
- **A 1800 s piece on a 30 min meeting is close to whole-meeting clustering.** The 30 min dev meetings are 1800.8–1809.0 s long, so an 1800 s piece is the whole meeting plus a tail piece of the last few seconds. Many 1800 s settings in the top 1% (22 for CAM++, 14 for ERes2NetV2) mean that on a 30 min meeting not splitting is better; the effect of pieces is seen only on the 4-hour meetings.
- **cannot-link did harm.** The cause is assumed: one person over-split within a piece cannot be joined again.
- **Flat axes in the top 1%:** overlap (0/60), the global floor value, piece length. link moves the most. As link goes down, floor 0 disappears from the top 1% (the floor cleans up the over-splitting).
- Low link with floor 0 gives thousands of speakers on the 4-hour dev (4,937). bench ran out of memory in scoring (a0c7827) and in turn reconstruction (7f21b5c) in this case and was fixed; both fixes give the same output as before.

### Chunked prototype: test results (preregistered settings)

test is 12 meetings: five 30 min meetings and one 4-hour meeting, made from 10 speakers who do not overlap with dev, each as room and call (seeds 400 and 401). The quiet speakers' actual shares are 1.97–2.51% at 30 min and 1.26% at 4 hours, all within 1–3%. The shares were matched when the meetings were made (redrawn up to 5 times). Results are in `bench/results/diar/chunk/{test,ami,long}/` (75c9df7), and the whole-meeting baseline (`.whole`) is the whole-meeting setting written in the preregistration (CAM++ average 0.25, floor 3%; ERes2NetV2 complete 0.8, floor 8%).

| | CAM++ | ERes2NetV2 |
|---|---:|---:|
| AMI Array1 DER (mean of 16 meetings) | 0.259 | 0.176 |
| Korean test room DER (mean of 6 meetings) | 0.191 | 0.162 |
| **Primary criterion (mean of the two)** | **0.225** | **0.169** |
| Secondary: Korean test call DER (two channels, no bleed) | 0.120 | 0.111 |
| For reference: AMI Headset DER | 0.147 | 0.123 |
| For reference: room whole-meeting baseline DER | 0.159 | 0.265 |
| Special speakers vanished (late joiner, quiet, early leaver), pieces / baseline | 5 / 9 | 1 / 15 |
| All vanished speakers, pieces / baseline | 5 / 9 | 1 / 15 |
| room speakers found / actual (2p, 4p-l, 4p-q, 6p 4 hours, 6p, 6p) | 7/2, 6/4, 4/4, 7/6, 6/6, 6/6 | 7/2, 6/4, 7/4, 6/6, 10/6, 7/6 |

- **Ranking: ERes2NetV2.** The primary criterion differs by 5.6 points, so the time rule for a difference under 1 point was not used. On the secondary (call) ERes2NetV2 also leads.
- **The gate is passed, by both models.** The baseline ran on every test meeting including the 4-hour one, so no meeting has to be reported apart. CAM++ has no meeting where more vanished than in the baseline, meeting by meeting either (equal 8, of which 4 are 0 on both sides; fewer 4). The person who vanishes is mostly the quiet speaker 149 (see "Moving to the app" below).
- **For CAM++ the baseline is better on room DER (0.159 vs 0.191).** Pieces made fewer people vanish but counted more speakers. For ERes2NetV2 pieces are better on both DER and vanishing.
- **Worse than dev.** ERes2NetV2's dev score is 0.086 and its test room 0.162. As written for dev, both models' dev best was optimistic.
- **Over-counting.** Both models counted the 2-speaker room as 7, and ERes2NetV2 counted one 6-speaker room as 10. In calls (system only) they are almost right (5 of 6 meetings for both models).
- **Bleed (secondary, not used for choosing).** When system sound leaks into the mic at −25 dB, the rule that takes the mic whole as "me" raises call DER from 0.11–0.12 to 0.46–0.47. A crude stand-in that removes only zero-delay leaks (`drop_echo`, correlation over 0.5 in 0.1 s frames) gives 0.150–0.158. Mixing into one track and diarizing gives 0.161–0.166. A real leak has delay and room reverb, so these values are close to a lower bound.
- **Limits.** The Korean meetings reuse the same clips (12 speakers in dev, 10 in test). room and call are the same mix (same seed, same turns, same shares): they are not independent samples of the quiet speaker. Taking a call's mic whole as "me" is wrong in a hybrid meeting (someone else beside me).

**Cap: memory failure.** A 4-hour meeting run from audio to turns in one process (the long step, preregistered settings).

| 4 hours | Processing time | Peak RSS (preregistered measurement) | Peak RSS, long-fixmem (post hoc, not preregistered) |
|---|---:|---:|---:|
| room, CAM++ | 9.7 min | **2112 MB** | 1064 MB |
| room, ERes2NetV2 | 36.6 min | **1810 MB** | 1046 MB |
| call, CAM++ | 9.3 min | 827 MB | 661 MB |
| call, ERes2NetV2 | 25.3 min | 894 MB | 853 MB |

- The CAM++ time cap (15 min) is passed; the memory cap (1.0 GB) fails on the 4-hour room for both models. This is the preregistered result.
- The cause is the number of global-link candidates. At local average 0.15 one piece splits into hundreds, so the 4-hour room had 7,014 candidates for CAM++ and 6,177 for ERes2NetV2 (call 3,632 and 3,058). The float64 square distance matrix of these candidates and the indices of its upper triangle came close to 1 GB (904 MB tracemalloc in the clustering step). It did not show on the 30 min meetings or in the grid.
- long-fixmem is a post hoc engineering fix (86b3c1d). It builds the distances in condensed form directly and keeps the float32 product whole (multiplying in row blocks rounds differently in BLAS, giving up to 2e-7 difference). The labels are bit-identical on 88 test stores (Korean test and AMI, both models; 108 comparisons), and so is DER. room still goes over 1.0 GB (MB is 2^20 bytes; 4.6–6.4% over counting 1,000 MB, 2.1–3.9% counting 1,024 MB). ERes2NetV2 is already at 872 MB when extraction finishes.
- The 4-hour ERes2NetV2 processing time is 25–37 min (recorded only). Run after the final pass (about 22.6 min), after-meeting processing is about 48–60 min; CAM++ adds 9–10 min for about 32 min. Running the two jobs in parallel was not measured.

### Moving to the app

- State and fix the cluster numbering rule (order of first appearance) and the tie rule (lower number first). Keep a determinism test that the same input gives the same output. On over-split results, the tie rule alone moved DER by 2 points.
- Hand short clusters without an embedding (tails under 10 frames, too short to extract an embedding) to the neighbouring speaker in time or the speaker they overlap most, so no tail is left.
- What CAM++ average 0.25, 3% actually does: average 0.25 finds tight cores, the cores with 3% or more of the embeddings become speaker candidates, and the rest (up to half in room) attach to the nearest candidate's centroid. The floor share sets the number of speakers and the cut sets where the cores are. Implement and explain it as these two steps.
- The structural risk of a share floor: when the number of speakers is set by "3% of all speech", someone who drops in briefly on a long meeting vanishes. 3% of 4 hours is about 7 minutes, so a person who attends for 10 minutes and talks for 3 does not come out as a speaker. The floor's unit is embeddings, but the number of embeddings scales with the amount of speech, so it is the same thing. A quiet person in a 6-person meeting has the same risk (AMI vanished 8–22/63).
- A cap on the number of speakers: some settings give thousands of speakers (link 0.15 with floor 0 in the widened chunked grid gave 4,937 on the 4-hour dev). The app needs a speaker cap (for example 20) and a way to handle going over it (raise the global floor, or keep the k largest and attach the rest to the nearest speaker). The bench scorer also ran out of memory in this case and was fixed (a0c7827, totals the same as before).
- The number of global-link candidates: in the chunked test the 4-hour room had 6–7 thousand candidates, and even with condensed distances it went over 1.0 GB. Cap the number of candidates, or link in two stages (clustering within a piece → a few representatives per piece → global). Either way the clustering step's memory must be fixed regardless of the meeting's length.
- Over-counting is visible to the user at once: a 2-person meeting came out with 7 speakers (chunked test, both models). Design the speaker cap (above) together with SPK-3 (merging two speakers into one when naming them). The cap reduces the over-splitting left, and merging is the user's way to fix it at the end.
- The quiet speaker 149 vanishes again and again on test. With pieces, CAM++ lost 149 in all 3 room meetings 149 is in and in the 4-hour call, ERes2NetV2 in the 4-hour room, and the whole-meeting baseline lost 149 in almost every meeting. 149's speech is 1.3–2.5% of the meeting (38–197 s, median turn 4–6 s), and the nearest voice is always 118. In call (phone band) the centroid cosine of 149 and 118 is 0.84–0.89, the same as how close 149's own embeddings sit to their own centroid (0.86–0.88); in room it is apart at 0.54–0.62, but 149's own spread is on the large side (0.76–0.81; other speakers mostly 0.80–0.85).
- The rule that takes a call's mic whole as "me" is weak against bleed: in the chunked test a −25 dB leak alone raised call DER from 0.11 to 0.46. The app's diarization depends on echo cancellation (AEC). In a hybrid meeting the rule itself is wrong.
- The memory was measured assuming diarization runs after the final pass. Running it at the same time as whisper adds to it.
- What to look at in the chunked prototype:
  - Apply the floor per piece (10–15 min), not over the whole meeting, and link the pieces' speakers globally. A late joiner can pass the floor within their own piece.
  - Count vanished on synthetic meetings with late joiners and quiet speakers. The principle of keeping dev and test speakers apart stays.
  - Also set a fallback rule for when no cluster passes the floor (for example the k largest). `diar_pipe` does nothing in that case now, and WeSpeaker 0.02's speaker explosion is the result.
  - Measure the memory cap, the 4-hour processing time and AMI DER. Choose on the 30 min dev only.
  - The memory cap (1.0 GB RSS) applies to the RSS of the one preregistered setting run in a separate process in the long step. It does not apply to the grid workers (which run many settings and meetings in one process). The heaviest setting in the widened grid peaked at 4.5 GB per worker on the 4-hour dev. The long step records, per stage (extraction, clustering, reconstruct, mic), that stage's own peak allocation (tracemalloc) and the process's peak RSS after it, separately.

### Where it differs from the app

- bench mixes mic and system into one track and diarizes that. The app records the two channels separately, so in a call "me" (mic) is already apart and only the remote speakers on the system side need diarizing. This call result is a harder condition than the app's. The chunked prototype is measured with per-channel input.
- The app's in-person meetings use a laptop's built-in microphone. That is not the same distance and noise condition as AMI Array1 (one channel of a circular microphone array on the table).
