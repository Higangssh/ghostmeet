# 한국어 STT 엔진 비교 — 측정 결과

기기: Apple M4 (4P+6E), RAM 16GB, macOS 26.6.2, Python 3.12.15. 모든 엔진 스레드 4개. 측정 중 다른 무거운 작업 없음.

데이터: FLEURS ko_kr test, 382문장 / 80.1분 (google/fleurs, CC BY 4.0). 실제 회의 녹음은 쓰지 않았다.

## 1. 문장 단위 (엔진 정확도·속도)

| 엔진 | CER(숫자 없는 문장) | CER(전체) | RTF | 로드 | 첫 결과 | 문장당 지연 중앙/p95 | 최대 메모리 |
|---|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (CPU int8) | **2.2%** | 2.5% | 0.373 | 1.9s¹ | 6.6s¹ | 4.61 / 5.24s | 2247MB |
| whisper.cpp large-v3-turbo (Metal) | **2.3%** | 2.5% | 0.165 | 1.0s | 3.4s | 2.07 / 2.26s | 1936MB |
| sherpa-onnx SenseVoice-Small (CPU int8) | **4.7%** | 6.1% | 0.011 | 0.4s | 0.6s | 0.13 / 0.22s | 949MB |
| faster-whisper small (CPU int8) | **5.1%** | 5.4% | 0.118 | 0.8s | 2.4s | 1.45 / 1.96s | 945MB |
| whisper.cpp small (CPU) | **5.2%** | 5.3% | 0.426 | 0.4s | 5.7s | 5.31 / 5.98s | 823MB |
| whisper.cpp small (Metal) | **5.2%** | 5.4% | 0.056 | 0.5s | 1.2s | 0.69 / 0.93s | 866MB |

- CER: 공백·문장부호를 지운 뒤 글자 편집거리 / 정답 글자 수(코퍼스 합산). 정답의 괄호 안 로마자 병기(`비슈케크(Bishkek)`)는 읽히지 않으므로 지웠다. 한글이 든 괄호는 읽으므로 남겼다.
- 숫자 없는 문장 CER: 정답에 숫자가 있는 문장을 뺀 값. `2명`/`두 명` 같은 표기 차이가 엔진 비교를 흐리지 않게 한다. **엔진 비교는 이 열로 본다.**
- RTF = 처리 시간 / 음성 길이. 첫 결과 = 엔진을 만들기 시작해서 첫 문장(14초) 결과가 나올 때까지(모델은 이미 디스크에 있음).
- 최대 메모리: macOS phys_footprint 최대치(활성 상태 보기의 '메모리'). Metal 할당 포함. whisper.cpp는 whisper-server 프로세스 기준.
- ¹ 전체 측정 때는 첫 실행이라 모델 다운로드가 로드에 섞였다. 그래서 모델을 캐시한 상태에서 5문장으로 따로 잰 값을 넣었다.

## 2. 긴 세션, 꼬리 창 (legacy/backend/incremental.py의 IncrementalTranscriber 그대로)

FLEURS 문장을 이어 붙인 36.1분 세션, 창 30초. `수정 전` = 측정 당시 main(창 경계 중복 버그 포함), `수정 후` = 창 경계 보류 수정.

| 엔진 | 코드 | 조건 | 창당 처리 처음10분 → 마지막10분 | p95 | 기울기(초/시간) | 메모리 처음 → 마지막 | 자막 지연 중앙/p95 | CER |
|---|---|---|---:|---:|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (CPU int8) | 수정 전 | backlog(30초 창) | 5.50s → 5.22s | 9.32s | -1.156 | 1705 → 1674MB | — | 18.7% |
| faster-whisper large-v3-turbo (CPU int8) | 수정 후 | backlog(30초 창) | 5.61s → 5.79s | 9.71s | +0.686 | 1706 → 1693MB | — | 6.7% |
| faster-whisper large-v3-turbo (CPU int8) | 수정 전 | live(10초 주기) | 4.78s → 4.73s | 5.21s | +0.011 | 1667 → 1593MB | 9.6 / 18.1s | 21.2% |
| faster-whisper large-v3-turbo (CPU int8) | 수정 후 | live(10초 주기) | 4.96s → 4.92s | 5.37s | -0.222 | 1641 → 1580MB | 12.4 / 20.8s | 3.4% |
| faster-whisper small (CPU int8) | 수정 전 | backlog(30초 창) | 2.35s → 2.65s | 3.56s | +0.138 | 1013 → 1229MB | — | 23.9% |
| faster-whisper small (CPU int8) | 수정 후 | backlog(30초 창) | 2.38s → 2.53s | 2.92s | +0.354 | 1147 → 1096MB | — | 9.5% |
| faster-whisper small (CPU int8) | 수정 전 | live(10초 주기) | 1.48s → 1.47s | 2.28s | -0.054 | 903 → 839MB | 4.9 / 12.3s | 29.7% |
| faster-whisper small (CPU int8) | 수정 후 | live(10초 주기) | 1.62s → 1.62s | 2.19s | -0.044 | 899 → 829MB | 8.1 / 14.7s | 6.6% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 전 | backlog(30초 창) | 0.44s → 0.38s | 0.58s | -0.141 | 1184 → 1133MB | — | 10.0% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 후 | backlog(30초 창) | 0.38s → 0.39s | 0.54s | +0.004 | 1142 → 956MB | — | 6.2% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 전 | live(10초 주기) | 0.16s → 0.15s | 0.22s | -0.005 | 1050 → 879MB | 0.9 / 8.2s | 17.0% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 후 | live(10초 주기) | 0.24s → 0.21s | 0.42s | -0.086 | 1115 → 1048MB | 6.5 / 11.2s | 6.0% |
| whisper.cpp large-v3-turbo (Metal) | 수정 전 | backlog(30초 창) | 2.35s → 2.36s | 2.55s | +0.038 | 1943 → 1956MB | — | 13.9% |
| whisper.cpp large-v3-turbo (Metal) | 수정 후 | backlog(30초 창) | 2.39s → 2.42s | 2.59s | +0.056 | 1942 → 1949MB | — | 5.8% |
| whisper.cpp large-v3-turbo (Metal) | 수정 전 | live(10초 주기) | 2.12s → 2.10s | 2.26s | -0.029 | 1927 → 1929MB | 5.1 / 12.0s | 21.6% |
| whisper.cpp large-v3-turbo (Metal) | 수정 후 | live(10초 주기) | 2.15s → 2.13s | 2.34s | -0.022 | 1933 → 1942MB | 9.3 / 16.0s | 5.8% |
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | 수정 후 | live(10초 주기) | 4.69s → 4.07s | 5.93s | -1.007 | 1923 → 1924MB | 13.5 / 20.5s | 6.5% |
| whisper.cpp large-v3-turbo (Metal, Silero spans) | 수정 후 | live(10초 주기) | 4.71s → 4.49s | 6.27s | -1.027 | 1924 → 1932MB | 13.2 / 21.8s | 5.9% |
| whisper.cpp small (Metal) | 수정 전 | backlog(30초 창) | 0.94s → 0.86s | 1.28s | -0.119 | 862 → 866MB | — | 29.9% |
| whisper.cpp small (Metal) | 수정 후 | backlog(30초 창) | 1.13s → 1.04s | 1.55s | -0.223 | 862 → 864MB | — | 8.6% |
| whisper.cpp small (Metal) | 수정 전 | live(10초 주기) | 0.68s → 0.68s | 0.93s | -0.018 | 857 → 836MB | 4.7 / 11.3s | 31.9% |
| whisper.cpp small (Metal) | 수정 후 | live(10초 주기) | 0.76s → 0.72s | 0.95s | -0.104 | 856 → 863MB | 6.8 / 12.5s | 8.9% |
| whisper.cpp small (Metal, Silero spans + filter) | 수정 후 | live(10초 주기) | 1.03s → 1.16s | 1.69s | +0.300 | 857 → 787MB | 8.2 / 13.1s | 8.3% |
| whisper.cpp small (Metal, no VAD) | 수정 후 | live(10초 주기) | 0.89s → 0.77s | 1.19s | -0.176 | 859 → 864MB | 7.4 / 17.3s | 13.2% |
| whisper.cpp small (Metal, no VAD, DTW) | 수정 후 | live(10초 주기) | 0.93s → 0.81s | 1.22s | -0.201 | 899 → 926MB | 8.4 / 15.5s | 12.2% |
| whisper.cpp small (Metal, Silero spans) | 수정 후 | live(10초 주기) | 1.32s → 1.22s | 1.88s | -0.301 | 858 → 820MB | 7.8 / 13.7s | 8.6% |

- backlog: 음성이 이미 다 있어 매 회차가 30초 창을 꽉 채운다. 창당 비용이 세션 위치와 무관한지 보는 조건.
- live: 가상 시계로 실시간 유입을 흉내 낸다. 직전 회차가 끝나고 10초 뒤에 다음 회차를 돈다(파이프라인 워커와 같음). 자막 지연 = 회차가 끝난 시각 − 세그먼트 끝 시각.
- 기울기: 창당 처리 시간을 세션 위치에 대해 1차 회귀한 값. 0에 가까우면 일정하다는 뜻이다.

## 3. 창 경계 수정 전후 (live 10초 주기)

완료 기준: live CER ≤ 문장 단위 CER(전체) + 2.0%p.

| 엔진 | 문장 단위 CER | live CER 전 → 후 | 차이(후 − 문장) | 기준 | 자막 지연 중앙/p95 전 → 후 |
|---|---:|---:|---:|---|---:|
| faster-whisper large-v3-turbo (CPU int8) | 2.5% | 21.2% → **3.4%** | +0.9%p | 통과 | 9.6 / 18.1s → 12.4 / 20.8s |
| faster-whisper small (CPU int8) | 5.4% | 29.7% → **6.6%** | +1.2%p | 통과 | 4.9 / 12.3s → 8.1 / 14.7s |
| sherpa-onnx SenseVoice-Small (CPU int8) | 6.1% | 17.0% → **6.0%** | -0.1%p | 통과 | 0.9 / 8.2s → 6.5 / 11.2s |
| whisper.cpp large-v3-turbo (Metal) | 2.5% | 21.6% → **5.8%** | +3.2%p | 미달 | 5.1 / 12.0s → 9.3 / 16.0s |
| whisper.cpp small (Metal) | 5.4% | 31.9% → **8.9%** | +3.5%p | 미달 | 4.7 / 11.3s → 6.8 / 12.5s |

- 비교 기준은 문장 단위 '전체' CER이다. 세션도 숫자가 든 문장을 포함하기 때문이다.
- 문장 단위는 문장 하나를 통째로 넣고, 세션은 문장들을 0.6초 간격으로 이어 붙여 창으로 자른다. 그래서 같은 엔진이라도 문맥과 경계 조건이 조금 다르다.

## 4. 회의 후 일괄 재전사 (세션 전체를 한 번에)

같은 36.1분 세션을 엔진에 통째로 넣었다. 창을 나누는 일은 엔진의 long-form 디코딩이 한다. 비교용으로 같은 엔진의 꼬리 창 backlog(수정 후) CER을 붙였다.

| 엔진 | long-form CER | 처리 시간 | RTF | 최대 메모리 | 꼬리 창 backlog CER |
|---|---:|---:|---:|---:|---:|
| whisper.cpp large-v3-turbo (Metal) | **17.3%** | 93s | 0.043 | 2503MB | 5.8% |

## 5. 우리 VAD 구간 + 구간별 디코딩

엔진의 VAD와 long-form을 쓰지 않는다. Silero VAD(MIT)로 음성 구간을 자르고 구간마다 따로 디코딩한다. 시각은 구간에서 온다. live(꼬리 창)와 일괄(세션 전체)이 같은 함수(`engines.speech_spans`)와 같은 값을 쓴다. SenseVoice 행도 같은 규칙이다.

구간 규칙: `threshold=0.3`, `min_silence_sec=0.5`, `min_speech_sec=0.25`, `max_speech_sec=25.0`, `pad_sec=0.2`.

| 엔진 | 조건 | CER | 문장 단위 CER | 창당 처리 / 처리 시간 | 자막 지연 중앙/p95 | 측정 중 부하(1분 평균) |
|---|---|---:|---:|---:|---:|---|
| whisper.cpp large-v3-turbo (Metal, Silero spans) | live(10초 주기) | **5.9%** | 2.5% | 4.71s → 4.49s | 13.2 / 21.8s | 1.96 → 1.00 |
| whisper.cpp large-v3-turbo (Metal, Silero spans) | 일괄 | **6.9%** | 2.5% | 430s (RTF 0.199) | — | 1.17 → 3.70 (중간 최대 6.2) |
| whisper.cpp small (Metal, Silero spans) | live(10초 주기) | **8.6%** | 5.4% | 1.32s → 1.22s | 7.8 / 13.7s | 1.93 → 1.17 |
| sherpa-onnx SenseVoice-Small (CPU int8) (기준) | live(10초 주기) | 6.0% | 6.1% | 0.24s → 0.21s | 6.5 / 11.2s | — |

## 6. 마지막 라운드: 필터, 묶기, 짧은 대답

5절의 구간 규칙에 Whisper 세그먼트 필터를 더했다. 필터는 `no_speech_prob > 0.6` 이면서 `avg_logprob < -1.0`인 세그먼트를 버린다(OpenAI/faster-whisper와 같은 규칙, whisper-rs로 그대로 옮길 수 있다). `packed`는 이웃 구간을 25초까지 묶어 한 번에 디코딩하고, `spans`는 구간마다 따로 디코딩한다. 기준은 문장 단위 CER + 2.0%p다.

짧은 대답: macOS `say -v Yuna`로 합성한 8개(네, 아니요, 맞아요, 좋습니다, 그렇죠, 알겠습니다, 잠깐만요, 글쎄요)를 3문장마다 하나씩 앞뒤 0.8초 간격으로 끼웠다(`make_replies.sh`, 합성 음성은 bench/data에만 있다). 글자 정렬에서 절반 넘게 그대로 맞으면 나온 것으로 센다.

| 엔진 | 조건 | CER | 문장 단위 CER | 기준 | 짧은 대답 | 필터가 버린 세그먼트 | 창당 처리 / 처리 시간 | 자막 지연 중앙/p95 | 부하(1분 평균) |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | live(10초 주기) | **6.5%** | 2.5% | 미달 | — | — | 4.69s → 4.07s | 13.5 / 20.5s | 1.35 → 1.24 |
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | 일괄 | **6.9%** | 2.5% | 미달 | — | 0 | 418s (RTF 0.193) | — | 1.63 → 0.57 |
| whisper.cpp large-v3-turbo (Metal, Silero spans + filter) | 일괄 + 짧은 대답 | **3.2%** | 2.5% | 통과 | 49/53 (필터로 잃음 0) | 0 | 471s (RTF 0.217) | — | 0.97 → 1.09 |
| whisper.cpp large-v3-turbo (Metal, Silero packed + filter) | 일괄 + 짧은 대답 | **3.8%** | 2.5% | 통과 | 31/53 (필터로 잃음 0) | 0 | 222s (RTF 0.103) | — | 1.60 → 0.97 |
| whisper.cpp small (Metal, Silero spans + filter) | live(10초 주기) | **8.3%** | 5.4% | 미달 | — | — | 1.03s → 1.16s | 8.2 / 13.1s | 0.57 → 1.35 |
| faster-whisper small (CPU int8) | live(10초 주기), 비교 | **6.6%** | 5.4% | 통과 | — | — | 1.62s → 1.62s | 8.1 / 14.7s | — |
| faster-whisper large-v3-turbo (CPU int8) | live(10초 주기), 비교 | **3.4%** | 2.5% | 통과 | — | — | 4.96s → 4.92s | 12.4 / 20.8s | — |
| sherpa-onnx SenseVoice-Small (CPU int8) | live(10초 주기), 비교 | **6.0%** | 6.1% | 통과 | — | — | 0.24s → 0.21s | 6.5 / 11.2s | — |

## 7. Rust 엔진의 회의 후 최종본 (앱이 쓸 경로 그대로)

`crates/meetbutler-engine`의 `final_transcript`를 그대로 돌렸다(`examples/final_pass`). 6절 `*-filtered`와 같은 규칙(우리 구간을 하나씩 디코딩, 2초 미만 구간의 자막 문구 차단)이지만 VAD 확률, 구간 규칙, 차단 목록, whisper.cpp 호출이 모두 Rust다. 오디오는 앱이 녹음하는 형식(16 kHz s16le)으로 넘긴다. 합격선은 최종본 CER 3% 이하다. 짧은 대답을 끼운 세션은 재현율로 판정한다: 대답 51/53 이상, 차단 규칙으로 잃은 대답 0, CER은 같은 경로의 본 세션 대비 +1.0%p 이내.

| 엔진 | 조건 | CER | 기준 | 짧은 대답 | 처리 시간 | RTF | 최대 메모리 | 부하(1분 평균) |
|---|---|---:|---|---:|---:|---:|---:|---|
| Rust engine large-v3-turbo (Metal, Silero spans + blocklist) | 일괄 | **2.7%** | 통과 | — | 203s | 0.094 | 1965MB | 1.81 → 0.73 |
| Rust engine large-v3-turbo (Metal, Silero spans + blocklist) | 일괄 + 짧은 대답 | **3.3%** | 통과 | 52/53 | 261s | 0.121 | 1961MB | 0.73 → 1.03 |
| Rust engine large-v3-turbo (Metal, whisper.cpp ggml Silero v5.1.2 spans + blocklist) | 일괄 | **6.0%** | 미달 | — | 234s | 0.108 | 1963MB | 1.89 → 3.60 |
| Rust engine large-v3-turbo (Metal, whisper.cpp ggml Silero v5.1.2 spans + blocklist) | 일괄 + 짧은 대답 | **6.7%** | — (본 세션 미달) | 52/53 | 283s | 0.130 | 1963MB | 3.60 → 1.26 |

- whisper.cpp의 ggml Silero v5.1.2(`-ggmlvad` 행)는 쓰지 않는다. 원래 Silero는 프레임마다 이전 프레임의 마지막 64샘플을 문맥으로 붙이는데, ggml 이식판은 프레임마다 반사 패딩(`ggml_pad_reflect_1d(…, 64, 64)`)을 한다. 이 세션에서 말을 23.0분만 찾았고(v4는 int16 입력에서 26.9분), 발화를 통째로 놓쳐 삭제가 314자였다. 합성 음성(TTS)에서는 오히려 더 많이 잡아서 합성 음성 대조로는 드러나지 않았다. VAD나 엔진을 바꿀 때는 FLEURS로도 확인한다(`tests/slow_vad.rs`).
- 채택한 VAD는 bench와 같은 `silero_vad.onnx`(k2-fsa가 내보낸 Silero v4, MIT)를 tract(순수 Rust ONNX 런타임)로 돌린다. onnxruntime 대비 프레임 확률 차이는 합성 세션 6.2e-6, FLEURS 10분 1.9e-5이고, 같은 입력에서 구간은 sherpa-onnx와 하나도 다르지 않다.
- 같은 VAD라도 입력이 float인지 int16인지에 따라 구간이 달라진다. 이 36분 세션에서 sherpa 구간은 float 입력 205개, int16 입력 174개였다(겹치지 않는 구간이 많다). 5·6절은 float으로 쟀고, 앱은 int16으로 녹음하므로 이 절은 int16으로 쟀다.

## 8. 로컬 요약 모델 (합성 회의 대본, 정답 대본 입력)

`run_summary.py`가 앱의 요약 sidecar를 앱과 같은 방식으로 돌렸다. 요약마다 프로세스를 새로 띄우고, GGUF 안의 chat template, 고정 형식(SUM-1)을 강제하는 grammar, repeat penalty 1.1, greedy를 쓴다. 대본은 `meeting_gen.py`의 주간·1:1·고객 × 30·60·120분이고 정답지와 맞춰 채점했다(`summary_score.py`). 모든 모델에 같은 프롬프트를 썼다. single은 전사를 통째로 넣고, chunked는 약 10분씩 요약한 뒤 목록을 잇는다.

재현율은 대본 합계다(형식이 깨진 요약은 하나도 못 찾은 것으로 센다). 지어낸 담당자는 회의에 없는 이름이나 사람이 아닌 담당자의 수이고, 지어낸 숫자는 회의에서 말하지 않은 숫자의 수다. 최대 메모리는 peak footprint와 max RSS 중 큰 값이다. 기기 상한은 8GB 묶음 4GB, 16GB 묶음 9GB이고, 넘으면 탈락이다. single의 n_ctx는 전사 + 8192(출력 상한)로 잡았다.

### 30·60분 대본 6개, 모든 모델

| 모델 | 묶음 | 방식 | 실행 실패 | 형식 깨짐 | 결정 | 할 일 | 담당자 맞음 | 기한 맞음 | 미해결 | 철회를 결정으로 | 지어낸 담당자 | 지어낸 숫자 | 회의당 시간 | 최대 메모리 | 생성 tok/s | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gemma-4-e4b | 8GB | chunked | 0/6 | 0/6 | 59/100 (59%) | 53/60 (88%) | 53/53 (100%) | 53/53 (100%) | 14/27 (52%) | 7 | 4 | 0 | 117s | 5.8GB | 17.8 | 탈락(메모리) |
| gemma-4-e4b | 8GB | single | 0/6 | 0/6 | 28/100 (28%) | 30/60 (50%) | 29/30 (97%) | 30/30 (100%) | 3/27 (11%) | 0 | 0 | 1 | 67s | 7.2GB | 17.0 | 탈락(메모리) |
| kanana-1.5-2.1b | 8GB | chunked | 0/6 | 0/6 | 35/100 (35%) | 49/60 (82%) | 43/49 (88%) | 48/49 (98%) | 4/27 (15%) | 4 | 16 | 2 | 69s | 2.7GB | 33.1 | — |
| kanana-1.5-2.1b | 8GB | single | 0/6 | 2/6 | 20/100 (20%) | 19/60 (32%) | 16/19 (84%) | 17/19 (89%) | 1/27 (4%) | 0 | 1 | 0 | 177s | 5.9GB | 22.6 | 탈락(메모리) |
| midm-2.0-mini | 8GB | chunked | 0/6 | 0/6 | 45/100 (45%) | 41/60 (68%) | 38/41 (93%) | 41/41 (100%) | 10/27 (37%) | 1 | 9 | 0 | 54s | 3.4GB | 31.1 | — |
| midm-2.0-mini | 8GB | single | 0/6 | 0/6 | 20/100 (20%) | 20/60 (33%) | 15/20 (75%) | 20/20 (100%) | 5/27 (19%) | 0 | 3 | 0 | 43s | 8.3GB | 21.4 | 탈락(메모리) |
| qwen3.5-4b | 8GB | chunked | 0/6 | 0/6 | 62/100 (62%) | 51/60 (85%) | 50/51 (98%) | 51/51 (100%) | 9/27 (33%) | 6 | 4 | 1 | 99s | 3.2GB | 19.4 | — |
| qwen3.5-4b | 8GB | single | 0/6 | 0/6 | 49/100 (49%) | 33/60 (55%) | 31/33 (94%) | 33/33 (100%) | 2/27 (7%) | 5 | 0 | 1 | 64s | 4.0GB | 17.7 | 탈락(메모리) |
| ax-4.0-light | 16GB | chunked | 0/6 | 0/6 | 39/100 (39%) | 47/60 (78%) | 45/47 (96%) | 47/47 (100%) | 4/27 (15%) | 0 | 6 | 0 | 95s | 4.8GB | 16.6 | — |
| ax-4.0-light | 16GB | single | 0/6 | 0/6 | 24/100 (24%) | 30/60 (50%) | 22/30 (73%) | 30/30 (100%) | 0/27 (0%) | 5 | 1 | 0 | 62s | 6.2GB | 16.2 | — |
| kanana-1.5-8b | 16GB | chunked | 0/6 | 0/6 | 47/100 (47%) | 42/60 (70%) | 40/42 (95%) | 41/42 (98%) | 16/27 (59%) | 6 | 9 | 1 | 167s | 5.9GB | 14.7 | — |
| kanana-1.5-8b | 16GB | single | 0/6 | 0/6 | 23/100 (23%) | 16/60 (27%) | 10/16 (62%) | 16/16 (100%) | 5/27 (19%) | 2 | 5 | 8 | 121s | 9.0GB | 12.7 | 탈락(메모리) |
| midm-2.0-base | 16GB | chunked | 0/6 | 0/6 | 33/100 (33%) | 42/60 (70%) | 40/42 (95%) | 26/42 (62%) | 10/27 (37%) | 0 | 6 | 14 | 179s | 8.7GB | 10.9 | — |
| midm-2.0-base | 16GB | single | 3/6 | 0/3 | 6/33 (18%) | 9/20 (45%) | 6/9 (67%) | 7/9 (78%) | 2/10 (20%) | 0 | 0 | 2 | 76s | 11.2GB | 9.6 | 탈락(메모리) |
| qwen3.5-9b | 16GB | chunked | 0/6 | 0/6 | 58/100 (58%) | 54/60 (90%) | 54/54 (100%) | 54/54 (100%) | 14/27 (52%) | 10 | 9 | 2 | 179s | 5.9GB | 13.0 | — |
| qwen3.5-9b | 16GB | single | 0/6 | 0/6 | 47/100 (47%) | 41/60 (68%) | 40/41 (98%) | 41/41 (100%) | 5/27 (19%) | 7 | 0 | 7 | 111s | 6.7GB | 12.2 | — |

### 120분 대본 3개, 30·60분 결과로 고른 모델

묶음마다 상한 안에 든 방식의 결정+할 일 재현율이 높은 두 모델이다. llama 구조 모델(Kanana, Mi:dm)의 single은 시작하자마자 디코딩이 실패했다(`Decode Error -3`, 최대 메모리 11GB대). 실패한 실행은 n_ctx를 남기지 못했지만 같은 식(전사 + 8192)이면 57344로, 학습 문맥 32768을 넘고 KV 캐시만 Mi:dm Mini 10.5GB, Kanana 8B 7GB다. 30·60분 표의 Mi:dm Base single 60분 3개도 같은 실패다(n_ctx 약 36864, KV 6.75GB + 모델 6.5GB). 측정 설정 탓이지만, 프롬프트를 60분의 두 배로 잡아(Mi:dm 약 2.3만, Kanana 약 3.1만 토큰) n_ctx를 맞춰도 KV만 Mi:dm Mini 약 4.2GB, Kanana 8B 약 3.9GB다. Mi:dm Mini는 8GB 묶음 상한을 넘고 Kanana 8B는 9GB 상한 근처이며 학습 문맥 끝에 걸린다.

| 모델 | 묶음 | 방식 | 실행 실패 | 형식 깨짐 | 결정 | 할 일 | 담당자 맞음 | 기한 맞음 | 미해결 | 철회를 결정으로 | 지어낸 담당자 | 지어낸 숫자 | 회의당 시간 | 최대 메모리 | 생성 tok/s | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| midm-2.0-mini | 8GB | chunked | 0/3 | 0/3 | 52/133 (39%) | 56/80 (70%) | 51/56 (91%) | 56/56 (100%) | 13/36 (36%) | 14 | 11 | 0 | 140s | 3.4GB | 30.3 | — |
| midm-2.0-mini | 8GB | single | 3/3 | 0/0 | — | — | — | — | — | 0 | 0 | 0 | — | 11.0GB | — | 실행 실패 |
| qwen3.5-4b | 8GB | chunked | 0/3 | 0/3 | 91/133 (68%) | 74/80 (92%) | 73/74 (99%) | 74/74 (100%) | 13/36 (36%) | 12 | 3 | 0 | 268s | 3.2GB | 19.2 | — |
| qwen3.5-4b | 8GB | single | 0/3 | 0/3 | 51/133 (38%) | 47/80 (59%) | 37/47 (79%) | 47/47 (100%) | 2/36 (6%) | 9 | 0 | 0 | 247s | 4.7GB | 13.8 | 탈락(메모리) |
| kanana-1.5-8b | 16GB | chunked | 0/3 | 0/3 | 61/133 (46%) | 60/80 (75%) | 60/60 (100%) | 60/60 (100%) | 13/36 (36%) | 10 | 16 | 3 | 425s | 5.9GB | 14.7 | — |
| kanana-1.5-8b | 16GB | single | 3/3 | 0/0 | — | — | — | — | — | 0 | 0 | 0 | — | 11.7GB | — | 실행 실패 |
| qwen3.5-9b | 16GB | chunked | 0/3 | 0/3 | 87/133 (65%) | 72/80 (90%) | 66/72 (92%) | 72/72 (100%) | 17/36 (47%) | 14 | 15 | 2 | 446s | 5.9GB | 13.4 | — |
| qwen3.5-9b | 16GB | single | 0/3 | 0/3 | 53/133 (40%) | 56/80 (70%) | 50/56 (89%) | 56/56 (100%) | 2/36 (6%) | 0 | 0 | 0 | 298s | 7.5GB | 10.6 | — |

### single의 메모리를 n_ctx 때문에 얼마나 크게 쟀나 (추정, 다시 돌리지 않음)

llama.cpp는 n_ctx만큼의 KV 캐시를 처음에 다 잡는다. 필요한 n_ctx = 프롬프트 + 그 모델이 형식을 지킨 요약 중 가장 긴 출력 + 64를 256 단위로 올린 값이다. 추정 최대 메모리 = 잰 값 − (쓴 n_ctx − 필요한 n_ctx) × KV/토큰. 계산 버퍼는 n_ctx에 따라 조금 더 줄어들 수 있으니 상한 쪽 추정이다.

| 모델 | 묶음 | 대본 | KV/토큰 | 쓴 n_ctx | 필요한 n_ctx | 잰 최대 메모리 | 줄어드는 KV | 추정 최대 메모리 | 상한 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| gemma-4-e4b | 8GB | customer-30m | 16KiB | 22528 | 8192 | 6.6GB | 0.2GB | 6.4GB | 넘음 |
| gemma-4-e4b | 8GB | one_on_one-30m | 16KiB | 22528 | 8192 | 6.6GB | 0.2GB | 6.4GB | 넘음 |
| gemma-4-e4b | 8GB | weekly-30m | 16KiB | 22528 | 8192 | 6.3GB | 0.2GB | 6.1GB | 넘음 |
| gemma-4-e4b | 8GB | customer-60m | 16KiB | 34816 | 14848 | 7.2GB | 0.3GB | 6.9GB | 넘음 |
| gemma-4-e4b | 8GB | one_on_one-60m | 16KiB | 34816 | 15104 | 7.2GB | 0.3GB | 6.9GB | 넘음 |
| gemma-4-e4b | 8GB | weekly-60m | 16KiB | 34816 | 14848 | 7.2GB | 0.3GB | 6.9GB | 넘음 |
| kanana-1.5-2.1b | 8GB | customer-30m | 128KiB | 22528 | 9216 | 4.4GB | 1.6GB | 2.8GB | 안 넘음 |
| kanana-1.5-2.1b | 8GB | one_on_one-30m | 128KiB | 22528 | 9216 | 4.4GB | 1.6GB | 2.8GB | 안 넘음 |
| kanana-1.5-2.1b | 8GB | weekly-30m | 128KiB | 22528 | 9216 | 4.4GB | 1.6GB | 2.8GB | 안 넘음 |
| kanana-1.5-2.1b | 8GB | customer-60m | 128KiB | 34816 | 16640 | 5.9GB | 2.2GB | 3.7GB | 안 넘음 |
| kanana-1.5-2.1b | 8GB | one_on_one-60m | 128KiB | 34816 | 16640 | 5.9GB | 2.2GB | 3.7GB | 안 넘음 |
| kanana-1.5-2.1b | 8GB | weekly-60m | 128KiB | 34816 | 16384 | 5.9GB | 2.2GB | 3.7GB | 안 넘음 |
| midm-2.0-mini | 8GB | customer-30m | 192KiB | 24576 | 6912 | 6.1GB | 3.2GB | 2.8GB | 안 넘음 |
| midm-2.0-mini | 8GB | one_on_one-30m | 192KiB | 24576 | 6656 | 6.1GB | 3.3GB | 2.8GB | 안 넘음 |
| midm-2.0-mini | 8GB | weekly-30m | 192KiB | 24576 | 6656 | 6.0GB | 3.3GB | 2.8GB | 안 넘음 |
| midm-2.0-mini | 8GB | customer-60m | 192KiB | 36864 | 11776 | 8.3GB | 4.6GB | 3.7GB | 안 넘음 |
| midm-2.0-mini | 8GB | one_on_one-60m | 192KiB | 36864 | 12032 | 8.3GB | 4.5GB | 3.8GB | 안 넘음 |
| midm-2.0-mini | 8GB | weekly-60m | 192KiB | 34816 | 11776 | 7.9GB | 4.2GB | 3.7GB | 안 넘음 |
| qwen3.5-4b | 8GB | customer-30m | 32KiB | 22528 | 10240 | 3.6GB | 0.4GB | 3.3GB | 안 넘음 |
| qwen3.5-4b | 8GB | one_on_one-30m | 32KiB | 22528 | 9984 | 3.6GB | 0.4GB | 3.3GB | 안 넘음 |
| qwen3.5-4b | 8GB | weekly-30m | 32KiB | 22528 | 10240 | 3.6GB | 0.4GB | 3.2GB | 안 넘음 |
| qwen3.5-4b | 8GB | customer-60m | 32KiB | 34816 | 16640 | 4.0GB | 0.6GB | 3.5GB | 안 넘음 |
| qwen3.5-4b | 8GB | one_on_one-60m | 32KiB | 34816 | 16896 | 4.0GB | 0.5GB | 3.5GB | 안 넘음 |
| qwen3.5-4b | 8GB | weekly-60m | 32KiB | 34816 | 16384 | 4.0GB | 0.6GB | 3.5GB | 안 넘음 |
| qwen3.5-4b | 8GB | customer-120m | 32KiB | 57344 | 29440 | 4.7GB | 0.9GB | 3.9GB | 안 넘음 |
| qwen3.5-4b | 8GB | one_on_one-120m | 32KiB | 57344 | 29440 | 4.7GB | 0.9GB | 3.9GB | 안 넘음 |
| qwen3.5-4b | 8GB | weekly-120m | 32KiB | 57344 | 29696 | 4.7GB | 0.8GB | 3.9GB | 안 넘음 |
| ax-4.0-light | 16GB | customer-30m | 56KiB | 22528 | 6144 | 5.6GB | 0.9GB | 4.7GB | 안 넘음 |
| ax-4.0-light | 16GB | one_on_one-30m | 56KiB | 22528 | 5888 | 5.6GB | 0.9GB | 4.7GB | 안 넘음 |
| ax-4.0-light | 16GB | weekly-30m | 56KiB | 22528 | 6144 | 5.5GB | 0.9GB | 4.6GB | 안 넘음 |
| ax-4.0-light | 16GB | customer-60m | 56KiB | 34816 | 11008 | 6.2GB | 1.3GB | 5.0GB | 안 넘음 |
| ax-4.0-light | 16GB | one_on_one-60m | 56KiB | 34816 | 11008 | 6.2GB | 1.3GB | 4.9GB | 안 넘음 |
| ax-4.0-light | 16GB | weekly-60m | 56KiB | 34816 | 10752 | 6.2GB | 1.3GB | 4.9GB | 안 넘음 |
| kanana-1.5-8b | 16GB | customer-30m | 128KiB | 22528 | 9216 | 7.6GB | 1.6GB | 5.9GB | 안 넘음 |
| kanana-1.5-8b | 16GB | one_on_one-30m | 128KiB | 22528 | 9216 | 7.6GB | 1.6GB | 5.9GB | 안 넘음 |
| kanana-1.5-8b | 16GB | weekly-30m | 128KiB | 22528 | 9216 | 7.5GB | 1.6GB | 5.9GB | 안 넘음 |
| kanana-1.5-8b | 16GB | customer-60m | 128KiB | 34816 | 16384 | 9.0GB | 2.2GB | 6.8GB | 안 넘음 |
| kanana-1.5-8b | 16GB | one_on_one-60m | 128KiB | 34816 | 16640 | 9.0GB | 2.2GB | 6.8GB | 안 넘음 |
| kanana-1.5-8b | 16GB | weekly-60m | 128KiB | 34816 | 16384 | 9.0GB | 2.2GB | 6.8GB | 안 넘음 |
| midm-2.0-base | 16GB | customer-30m | 192KiB | 24576 | 6656 | 11.2GB | 3.3GB | 7.9GB | 안 넘음 |
| midm-2.0-base | 16GB | one_on_one-30m | 192KiB | 24576 | 6656 | 11.2GB | 3.3GB | 7.9GB | 안 넘음 |
| midm-2.0-base | 16GB | weekly-30m | 192KiB | 24576 | 6656 | 11.1GB | 3.3GB | 7.8GB | 안 넘음 |
| qwen3.5-9b | 16GB | customer-30m | 32KiB | 22528 | 9216 | 6.3GB | 0.4GB | 5.9GB | 안 넘음 |
| qwen3.5-9b | 16GB | one_on_one-30m | 32KiB | 22528 | 9216 | 6.4GB | 0.4GB | 6.0GB | 안 넘음 |
| qwen3.5-9b | 16GB | weekly-30m | 32KiB | 22528 | 9472 | 6.3GB | 0.4GB | 5.9GB | 안 넘음 |
| qwen3.5-9b | 16GB | customer-60m | 32KiB | 34816 | 15872 | 6.7GB | 0.6GB | 6.1GB | 안 넘음 |
| qwen3.5-9b | 16GB | one_on_one-60m | 32KiB | 34816 | 15872 | 6.7GB | 0.6GB | 6.1GB | 안 넘음 |
| qwen3.5-9b | 16GB | weekly-60m | 32KiB | 34816 | 15616 | 6.7GB | 0.6GB | 6.1GB | 안 넘음 |
| qwen3.5-9b | 16GB | customer-120m | 32KiB | 57344 | 28672 | 7.5GB | 0.9GB | 6.6GB | 안 넘음 |
| qwen3.5-9b | 16GB | one_on_one-120m | 32KiB | 57344 | 28672 | 7.5GB | 0.9GB | 6.6GB | 안 넘음 |
| qwen3.5-9b | 16GB | weekly-120m | 32KiB | 57344 | 28928 | 7.4GB | 0.9GB | 6.5GB | 안 넘음 |
