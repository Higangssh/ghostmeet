# 한국어 STT 엔진 비교 — 측정 결과

기기: Apple M4 (4P+6E), RAM 16GB, macOS 26.6.2, Python 3.12.15. 모든 엔진 스레드 4개. 측정 중 다른 무거운 작업 없음.

데이터: FLEURS ko_kr test, 382문장 / 80.1분 (google/fleurs, CC BY 4.0). 실제 회의 녹음은 쓰지 않았다.

## 1. 문장 단위 (엔진 정확도·속도)

| 엔진 | CER(숫자 없는 문장) | CER(전체) | RTF | 로드 | 첫 결과 | 문장당 지연 중앙/p95 | 최대 메모리 |
|---|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (CPU int8) | **2.5%** | 2.7% | 0.373 | 1.9s¹ | 6.6s¹ | 4.61 / 5.24s | 2247MB |
| whisper.cpp large-v3-turbo (Metal) | **2.6%** | 2.8% | 0.165 | 1.0s | 3.4s | 2.07 / 2.26s | 1936MB |
| sherpa-onnx SenseVoice-Small (CPU int8) | **5.0%** | 6.4% | 0.011 | 0.4s | 0.6s | 0.13 / 0.22s | 949MB |
| faster-whisper small (CPU int8) | **5.4%** | 5.7% | 0.118 | 0.8s | 2.4s | 1.45 / 1.96s | 945MB |
| whisper.cpp small (CPU) | **5.5%** | 5.6% | 0.426 | 0.4s | 5.7s | 5.31 / 5.98s | 823MB |
| whisper.cpp small (Metal) | **5.5%** | 5.6% | 0.056 | 0.5s | 1.2s | 0.69 / 0.93s | 866MB |

- CER: 공백·문장부호를 지운 뒤 글자 편집거리 / 정답 글자 수(코퍼스 합산). 정답의 괄호 안 로마자 병기(`비슈케크(Bishkek)`)는 읽히지 않으므로 지웠다.
- 숫자 없는 문장 CER: 정답에 숫자가 있는 문장을 뺀 값. `2명`/`두 명` 같은 표기 차이가 엔진 비교를 흐리지 않게 한다. **엔진 비교는 이 열로 본다.**
- RTF = 처리 시간 / 음성 길이. 첫 결과 = 엔진을 만들기 시작해서 첫 문장(14초) 결과가 나올 때까지(모델은 이미 디스크에 있음).
- 최대 메모리: macOS phys_footprint 최대치(활성 상태 보기의 '메모리'). Metal 할당 포함. whisper.cpp는 whisper-server 프로세스 기준.
- ¹ 전체 측정 때는 첫 실행이라 모델 다운로드가 로드에 섞였다. 그래서 모델을 캐시한 상태에서 5문장으로 따로 잰 값을 넣었다.

## 2. 긴 세션, 꼬리 창 (backend.incremental.IncrementalTranscriber 그대로)

FLEURS 문장을 이어 붙인 36.1분 세션, 창 30초. `수정 전` = 측정 당시 main(창 경계 중복 버그 포함), `수정 후` = 창 경계 보류 수정.

| 엔진 | 코드 | 조건 | 창당 처리 처음10분 → 마지막10분 | p95 | 기울기(초/시간) | 메모리 처음 → 마지막 | 자막 지연 중앙/p95 | CER |
|---|---|---|---:|---:|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (CPU int8) | 수정 전 | backlog(30초 창) | 5.50s → 5.22s | 9.32s | -1.156 | 1705 → 1674MB | — | 18.4% |
| faster-whisper large-v3-turbo (CPU int8) | 수정 후 | backlog(30초 창) | 5.61s → 5.79s | 9.71s | +0.686 | 1706 → 1693MB | — | 6.5% |
| faster-whisper large-v3-turbo (CPU int8) | 수정 전 | live(10초 주기) | 4.78s → 4.73s | 5.21s | +0.011 | 1667 → 1593MB | 9.6 / 18.1s | 22.0% |
| faster-whisper large-v3-turbo (CPU int8) | 수정 후 | live(10초 주기) | 4.96s → 4.92s | 5.37s | -0.222 | 1641 → 1580MB | 12.4 / 20.8s | 4.1% |
| faster-whisper small (CPU int8) | 수정 전 | backlog(30초 창) | 2.35s → 2.65s | 3.56s | +0.138 | 1013 → 1229MB | — | 24.8% |
| faster-whisper small (CPU int8) | 수정 후 | backlog(30초 창) | 2.38s → 2.53s | 2.92s | +0.354 | 1147 → 1096MB | — | 10.3% |
| faster-whisper small (CPU int8) | 수정 전 | live(10초 주기) | 1.48s → 1.47s | 2.28s | -0.054 | 903 → 839MB | 4.9 / 12.3s | 30.7% |
| faster-whisper small (CPU int8) | 수정 후 | live(10초 주기) | 1.62s → 1.62s | 2.19s | -0.044 | 899 → 829MB | 8.1 / 14.7s | 7.3% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 전 | backlog(30초 창) | 0.44s → 0.38s | 0.58s | -0.141 | 1184 → 1133MB | — | 10.7% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 후 | backlog(30초 창) | 0.38s → 0.39s | 0.54s | +0.004 | 1142 → 956MB | — | 7.0% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 전 | live(10초 주기) | 0.16s → 0.15s | 0.22s | -0.005 | 1050 → 879MB | 0.9 / 8.2s | 17.9% |
| sherpa-onnx SenseVoice-Small (CPU int8) | 수정 후 | live(10초 주기) | 0.24s → 0.21s | 0.42s | -0.086 | 1115 → 1048MB | 6.5 / 11.2s | 6.8% |
| whisper.cpp large-v3-turbo (Metal) | 수정 전 | backlog(30초 창) | 2.35s → 2.36s | 2.55s | +0.038 | 1943 → 1956MB | — | 14.5% |
| whisper.cpp large-v3-turbo (Metal) | 수정 후 | backlog(30초 창) | 2.39s → 2.42s | 2.59s | +0.056 | 1942 → 1949MB | — | 5.8% |
| whisper.cpp large-v3-turbo (Metal) | 수정 전 | live(10초 주기) | 2.12s → 2.10s | 2.26s | -0.029 | 1927 → 1929MB | 5.1 / 12.0s | 22.8% |
| whisper.cpp large-v3-turbo (Metal) | 수정 후 | live(10초 주기) | 2.15s → 2.13s | 2.34s | -0.022 | 1933 → 1942MB | 9.3 / 16.0s | 6.6% |
| whisper.cpp small (Metal) | 수정 전 | backlog(30초 창) | 0.94s → 0.86s | 1.28s | -0.119 | 862 → 866MB | — | 30.7% |
| whisper.cpp small (Metal) | 수정 후 | backlog(30초 창) | 1.13s → 1.04s | 1.55s | -0.223 | 862 → 864MB | — | 9.3% |
| whisper.cpp small (Metal) | 수정 전 | live(10초 주기) | 0.68s → 0.68s | 0.93s | -0.018 | 857 → 836MB | 4.7 / 11.3s | 32.4% |
| whisper.cpp small (Metal) | 수정 후 | live(10초 주기) | 0.76s → 0.72s | 0.95s | -0.104 | 856 → 863MB | 6.8 / 12.5s | 9.6% |
| whisper.cpp small (Metal, no VAD) | 수정 후 | live(10초 주기) | 0.89s → 0.77s | 1.19s | -0.176 | 859 → 864MB | 7.4 / 17.3s | 13.8% |

- backlog: 음성이 이미 다 있어 매 회차가 30초 창을 꽉 채운다. 창당 비용이 세션 위치와 무관한지 보는 조건.
- live: 가상 시계로 실시간 유입을 흉내 낸다. 직전 회차가 끝나고 10초 뒤에 다음 회차를 돈다(파이프라인 워커와 같음). 자막 지연 = 회차가 끝난 시각 − 세그먼트 끝 시각.
- 기울기: 창당 처리 시간을 세션 위치에 대해 1차 회귀한 값. 0에 가까우면 일정하다는 뜻이다.

## 3. 창 경계 수정 전후 (live 10초 주기)

완료 기준: live CER ≤ 문장 단위 CER(전체) + 2.0%p.

| 엔진 | 문장 단위 CER | live CER 전 → 후 | 차이(후 − 문장) | 기준 | 자막 지연 중앙/p95 전 → 후 |
|---|---:|---:|---:|---|---:|
| faster-whisper large-v3-turbo (CPU int8) | 2.7% | 22.0% → **4.1%** | +1.4%p | 통과 | 9.6 / 18.1s → 12.4 / 20.8s |
| faster-whisper small (CPU int8) | 5.7% | 30.7% → **7.3%** | +1.7%p | 통과 | 4.9 / 12.3s → 8.1 / 14.7s |
| sherpa-onnx SenseVoice-Small (CPU int8) | 6.4% | 17.9% → **6.8%** | +0.5%p | 통과 | 0.9 / 8.2s → 6.5 / 11.2s |
| whisper.cpp large-v3-turbo (Metal) | 2.8% | 22.8% → **6.6%** | +3.8%p | 미달 | 5.1 / 12.0s → 9.3 / 16.0s |
| whisper.cpp small (Metal) | 5.6% | 32.4% → **9.6%** | +4.0%p | 미달 | 4.7 / 11.3s → 6.8 / 12.5s |

- 비교 기준은 문장 단위 '전체' CER이다. 세션도 숫자가 든 문장을 포함하기 때문이다.
- 문장 단위는 문장 하나를 통째로 넣고, 세션은 문장들을 0.6초 간격으로 이어 붙여 창으로 자른다. 그래서 같은 엔진이라도 문맥과 경계 조건이 조금 다르다.
