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

## 2. 긴 세션, 꼬리 창 (backend.incremental.IncrementalTranscriber 그대로)

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
| whisper.cpp large-v3-turbo (Metal, Silero spans) | 수정 후 | live(10초 주기) | 4.71s → 4.49s | 6.27s | -1.027 | 1924 → 1932MB | 13.2 / 21.8s | 5.9% |
| whisper.cpp small (Metal) | 수정 전 | backlog(30초 창) | 0.94s → 0.86s | 1.28s | -0.119 | 862 → 866MB | — | 29.9% |
| whisper.cpp small (Metal) | 수정 후 | backlog(30초 창) | 1.13s → 1.04s | 1.55s | -0.223 | 862 → 864MB | — | 8.6% |
| whisper.cpp small (Metal) | 수정 전 | live(10초 주기) | 0.68s → 0.68s | 0.93s | -0.018 | 857 → 836MB | 4.7 / 11.3s | 31.9% |
| whisper.cpp small (Metal) | 수정 후 | live(10초 주기) | 0.76s → 0.72s | 0.95s | -0.104 | 856 → 863MB | 6.8 / 12.5s | 8.9% |
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
