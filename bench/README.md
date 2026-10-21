# bench — 한국어 STT 엔진 비교

MeetButler 0.2의 엔진 결정(requirements.md §8, §13)을 위한 측정 스크립트.
결과는 [`results/RESULTS.md`](results/RESULTS.md).

## 데이터와 라이선스

| 무엇 | 출처 | 라이선스 |
|---|---|---|
| 음성 | [FLEURS](https://huggingface.co/datasets/google/fleurs) `ko_kr` test (382문장, 80분, 낭독) | CC BY 4.0 |
| faster-whisper 모델 | Systran/faster-whisper-small, mobiuslabsgmbh/faster-whisper-large-v3-turbo | MIT |
| whisper.cpp 모델 | ggerganov/whisper.cpp `ggml-small.bin`, `ggml-large-v3-turbo.bin` | MIT |
| whisper.cpp VAD | ggml-org/whisper-vad `ggml-silero-v5.1.2.bin` | MIT |
| SenseVoice | k2-fsa/sherpa-onnx `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` (int8) | FunASR Model License (모델 출처 표기 조건, 앱 배포 전 확인 필요) |
| sherpa-onnx VAD | k2-fsa/sherpa-onnx `silero_vad.onnx` (Silero v4 export, [release asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx), sha256 `9e2449e1…cbb1fd6`). The Rust engine runs the same file | MIT |

실제 회의 녹음은 쓰지 않는다. 다운로드한 음성·모델은 전부 `bench/data/`(gitignore)에만 있고,
`bench/results/raw/`(gitignore)에 엔진 출력 원문이 남는다. 커밋되는 것은 숫자뿐이다.

## 준비 (macOS)

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
# 모델: README 위 표의 파일을 bench/data/models/ 로 (SenseVoice는 models/sv/ 에 풀기)
```

faster-whisper 모델은 첫 실행 때 `bench/data/models/faster-whisper/`로 받아진다.

## 실행

```bash
sh bench/run_all.sh                                        # 전체 (M4 기준 약 1.5시간)
bench/.venv/bin/python -I bench/run_utterances.py wcpp-small --limit 20   # 하나만
bench/.venv/bin/python -I bench/run_stream.py fw-small --mode live
```

엔진 이름: `fw-small`, `fw-large-v3-turbo`, `wcpp-small`, `wcpp-large-v3-turbo`,
`wcpp-small-cpu`, `sensevoice`, `sensevoice-coreml`, 그리고 앱의 Rust 엔진 `rust-<모델>`
(`run_longform.py` 전용, 먼저 `cargo build --release -p meetbutler-engine --features whisper,silero
--example final_pass`).

VAD 기준값: `silero_reference.py`가 WAV(또는 `--fleurs 분`으로 만든 FLEURS 세션)의 onnxruntime
프레임 확률과 sherpa 구간을 JSON으로 쓴다. Rust 쪽 `tests/slow_vad.rs`가 이것과 비교한다.

`bench/`는 pytest가 수집하지 않는다(`pytest.ini`의 `norecursedirs`). 기존 테스트 스위트는
계속 모델·네트워크 없이 돈다.

## 무엇을 같게 맞췄나

- 디코딩 설정은 `legacy/backend/transcriber.py`(v0.1)와 같게 했다. 한국어로 고정했고, Whisper는 beam 5다.
  Silero VAD는 threshold 0.3, 최소 침묵 500ms, 패딩 200ms다. 창 사이에 텍스트 문맥은 넘기지
  않는다. faster-whisper는 프로덕션의 `WhisperTranscriber`를 그대로 호출한다.
- 스레드는 엔진마다 4개다(M4 성능 코어 수).
- whisper.cpp는 Homebrew의 `whisper-server`를 loopback에 띄워 모델을 한 번만 올린다. 요청마다
  WAV를 HTTP로 넘기는 비용이 들어가지만, 30초 창 기준 수 ms다.
- 꼬리 창 측정은 `legacy/backend/incremental.py`의 `IncrementalTranscriber`(v0.1)를 그대로 쓴다.

## CER 계산

- NFKC로 정규화하고 소문자로 바꾼 뒤 공백과 문장부호를 지운다. 그다음 글자 단위 편집거리를
  정답 글자 수로 나눈다(코퍼스 합산).
- FLEURS 정답에는 읽지 않는 로마자 병기가 괄호로 들어 있다(`비슈케크(Bishkek)`). 그래서
  괄호 안을 지운다.
- 숫자 표기(`2명` / `두 명`)는 정규화하지 않는다. 그 대신 정답에 숫자가 없는 문장만으로 잰 CER을
  따로 낸다. 엔진 비교는 이 값을 기준으로 본다.

## 알려진 문제와 해석

- live(10초 주기) CER이 문장 단위보다 훨씬 나쁘게 나온 것은 측정 당시
  `IncrementalTranscriber`가 창 경계에서 같은 말을 두 번 확정했기 때문이다. 엔진 차이가 아니다.
  수정 전후 결과는 RESULTS.md에 나란히 둔다.
- FLEURS는 낭독체이고 깨끗한 음성이다. 회의실 원거리 마이크, 겹치는 발화, 구어체에서는 모든
  엔진의 CER이 올라간다. 순위가 유지된다고 가정하지 말고, 대면 회의 조건은 따로 재야 한다.
- 메모리는 macOS phys_footprint다(활성 상태 보기의 "메모리" 값과 같다). Metal 할당이 포함된다.
- whisper.cpp는 수정 후에도 live CER 기준(문장 단위 + 2%p)에 못 미친다. 오류의 대부분이 문장 끝
  삭제다. VAD를 끄면(`wcpp-small-novad`) 오히려 나빠져서, VAD 시각 재매핑이 원인이라는 가설은
  기각했다. 세그먼트 끝 시각이 부정확한 것이 남은 후보다(단어 시각 DTW는 `-nfa`가 필요하며 재지 않았다).
- `run_stream.py`는 음성이 다 들어온 뒤 강제 회차만 돈다(`pipeline._drain`과 같다). 예전에는 남은
  분량이 1초 이상이면 강제하지 않았다. 그래서 창 끝까지 이어진 꼬리가 보류되면 같은 창을 끝없이
  다시 읽었다(`wcpp-large-v3-turbo-novad`에서 발생, 결과 없음). 커밋된 결과는 모두 정상 종료했으므로
  이 반복을 타지 않았다.
- whisper-server에서 VAD를 켜면 세그먼트 시각만 원래 시간축으로 되돌리고, 단어 시각과 DTW 시각
  (`t_dtw`)은 무음을 잘라낸 시간축에 그대로 남는다(30초 창 시험에서 DTW 끝이 세그먼트 시작보다
  앞섰다). 그래서 DTW는 VAD를 끈 상태(`wcpp-small-novad-dtw`)에서만 쟀다. VAD를 끄면 삽입 오류
  (무음 구간 환각, 반복)가 늘어서 DTW로 끝 시각을 바로잡아도 live CER이 기준에 못 미친다.
  이 측정 막바지에 같은 기기에서 다른 빌드가 돌기 시작했으니 그 행의 창당 시간은 참고치다.
- whisper.cpp long-form(세션 36분을 한 요청으로)은 CER이 크게 나쁘다. 오류 대부분이 삭제이고
  20~35초씩 통째로 건너뛴 구간과 반복 세그먼트가 있다. 회의 후 재전사는 엔진의 long-form에
  맡기지 말고 창이나 VAD 구간 단위로 잘라서 넣어야 한다(꼬리 창 backlog 수치 참고).
- 우리 VAD 구간 + 구간별 디코딩(`*-spans`)에서는 짧은 구간이 환각을 낸다. turbo 일괄 처리의
  2초 미만 구간 38개 중 37개가 "한글자막 by …", "시청해주셔서 감사합니다." 같은 자막 문구였다.
  저장된 출력에서 1.5초(여백 포함) 미만 구간을 빼고 다시 채점하면 CER이 6.9%에서 2.4%로 떨어진다.
  이 값은 측정이 아니라 사후 추정이다. 짧은 구간을 버리면 회의의 "네", "아니요" 같은 짧은 대답도
  사라지는데, FLEURS에는 그런 발화가 없어서 그 비용은 여기서 보이지 않는다.
- CER 정답 전처리: 처음에는 괄호를 전부 지웠다. 그런데 382문장 중 22문장은 괄호 안 한국어를 실제로
  읽는다(282자). 지금은 한글이 없는 괄호만 지운다. 모든 결과는 `rescore.py`로 저장된 출력에서
  다시 채점했다(엔진은 다시 돌리지 않았고 시간 값은 그대로다). 대부분 0.2~0.8%p 내려갔고 순위와
  기준 통과 여부는 바뀌지 않았다.
- 마지막 라운드(RESULTS.md 6절). `no_speech_prob`/`avg_logprob` 필터는 한 세그먼트도 버리지 않았다.
  whisper.cpp는 자막 문구 환각에도 `no_speech_prob`를 낮게 낸다. 환각은 모두 문장 클립 끝에서
  다음 간격으로 넘어가는 약 1초짜리 VAD 구간에서 나왔다(이어 붙인 경계. 실제 회의라면 클릭음이나
  기침 소리에 해당한다). 묶기(`packed`)는 환각을 없애지만 짧은 대답 53개 중 22개를 삼킨다.
  같은 대답을 따로 넣으면 turbo가 8개 모두 받아 적으므로 원인은 묶기다. 구간별(`spans`)은 대답을
  49/53 살린다.
- 저장된 출력에 "2초 미만 구간이 알려진 환각 문구(한글자막…, 시청해주셔서…, 지금까지 뉴스
  스토리…, …계속 됩니다, 감사합니다)를 포함하면 버림" 규칙을 사후 적용한 추정치: turbo 일괄 CER
  6.9% → 2.4%, 짧은 대답 49/53 유지. turbo live(문구만 제거)는 6.5% → 2.9%로 추정된다. small live는
  환각이 1개뿐이라 8.2%로 그대로다. 측정이 아니라 추정이며, 회의의 진짜 "감사합니다"도 버린다.
- 이 라운드의 짧은 대답 세션은 원래 세션보다 환각이 훨씬 적다(2개 대 37개). 끼워 넣기로 간격과
  잡음 배열이 달라진 탓으로 보이지만 확인하지 않았다. 대답 재현율 비교는 같은 세션 안에서만 하라.
- whisper.cpp의 ggml Silero v5.1.2는 쓰지 않는다(RESULTS.md 7절). 원래 모델은 이전 프레임의 마지막
  64샘플을 문맥으로 붙이는데 ggml 이식판은 프레임마다 반사 패딩을 한다. FLEURS 36분에서 말을 23.0분만
  찾았고(같은 int16 입력에서 v4 onnx는 26.9분) 발화를 통째로 놓쳐 Rust 최종본 CER이 6.0%였다. 합성 음성
  세션에서는 v5가 오히려 더 많이 잡아서 이 차이가 보이지 않았다. VAD·엔진을 바꿀 때는 합성 음성만으로
  판단하지 않는다.
- 같은 Silero라도 float 입력과 int16 입력의 구간이 다르다(36분 세션 sherpa 구간 205개 대 174개).
  5·6절은 float, 7절은 앱이 녹음하는 int16이다. 같은 형식끼리만 비교하라.

## 요약 모델 측정 (0.2 마일스톤 3)

실제 회의 전사는 쓸 수 없어서 회의를 합성한다. `meeting_gen.py`가 회의 종류 3가지(주간 회의, 1:1,
고객 미팅) × 길이 3가지(30분, 1시간, 2시간)를 만든다. 결정 사항, 할 일(담당자·기한), 숫자, 미해결
질문, 중간에 철회되는 제안, 짧은 대답과 잡담을 심고, 심은 것을 정답지(key)로 함께 낸다. 사람·회사·
기능 이름은 모두 지어냈다. 앱처럼 화자 표시가 없어서 담당자는 말 속에서 이름으로 불린다. 대본
본문은 `bench/data/meetings/`(커밋 안 함)에만 둔다.

`summary_score.py`는 고정 형식(SUM-1, JSON) 요약을 정답지와 맞춰 채점한다. 판정 모델은 쓰지 않는다.
항목은 다음과 같다.
- 형식 통과
- 결정·할 일·미해결 질문 재현율: 한 항목의 핵심어가 출력의 한 항목 안에 모두 있으면 찾은 것으로 친다. 할 일은 담당자와 기한이 맞는지도 따로 센다.
- 철회된 제안을 결정으로 적은 수
- 회의에 없던 사람과 숫자

한국어 자연스러움은 블라인드 사람 채점으로 본다.

```bash
bench/.venv/bin/python -I bench/meeting_gen.py                                 # 대본 9개
bench/.venv/bin/python -I -m unittest discover -s bench/tests -t bench         # bench 테스트 전부(생성기·채점기·화자 분리)
# (표준 라이브러리 unittest라 pytest가 필요 없다. bench/.venv는 위 "준비"에서 만든 venv다)
```

### 요약 모델 (직접 변환한 GGUF)

각 공식 저장소의 safetensors를 커밋 해시로 고정해 받고, sidecar와 같은 llama.cpp(`d812350`)의 `convert_hf_to_gguf.py`와 `llama-quantize`로 Q4_K_M을 만들었다(`llm_models.py`, 결과는 `llm_models.json`). 받은 원본은 변환 후 지웠다. 커밋 해시로 다시 만들 수 있고, 사용자가 받을 파일은 아래 sha256으로 확인한다.

| 이름 | 원본 저장소@커밋 | 라이선스 | 기기 묶음 | 크기 | sha256 | 변환에 쓴 transformers |
|---|---|---|---|---:|---|---|
| kanana-1.5-2.1b | `kakaocorp/kanana-1.5-2.1b-instruct-2505@7df4bc35ccd6` | Apache-2.0 | 8GB | 1.42 GiB | `622293385bb6fb46…` | 4.57.6 |
| midm-2.0-mini | `K-intelligence/Midm-2.0-Mini-Instruct@383eb221c52a` | MIT | 8GB | 1.33 GiB | `f4ba7d9ac35b523f…` | 4.57.6 |
| qwen3.5-4b | `Qwen/Qwen3.5-4B@851bf6e806ef` | Apache-2.0 | 8GB | 2.59 GiB | `60cb21ee219f5c13…` | 4.57.6 |
| ax-4.0-light | `skt/A.X-4.0-Light@ba21c20ea1b3` | Apache-2.0 (ships a NOTICE: built with Qwen 2.5) | 16GB | 4.13 GiB | `2115a098ae9c1309…` | 4.57.6 |
| kanana-1.5-8b | `kakaocorp/kanana-1.5-8b-instruct-2505@c963a5f4f649` | Apache-2.0 | 16GB | 4.58 GiB | `ea411a3934a8cbfb…` | 4.57.6 |
| qwen3.5-9b | `Qwen/Qwen3.5-9B@c20223623576` | Apache-2.0 | 16GB | 5.38 GiB | `d5e23001c0128665…` | 4.57.6 |
| midm-2.0-base | `K-intelligence/Midm-2.0-Base-Instruct@35479c5fc9a1` | MIT | 16GB | 6.55 GiB | `e590278bb19ffaf6…` | 4.57.6 |
| gemma-4-e4b | `google/gemma-4-E4B-it@ee0ef6023621` | Apache-2.0 | 8GB | 4.97 GiB | `4b9bd938e2cd55f4…` | 5.20.1 |

- A.X 4.0 Light를 채택해 배포하면 Apache-2.0 §4(d)에 따라 저장소의 NOTICE("Built with Qwen 2.5 — Alibaba Cloud")를 함께 실어야 한다.
- Gemma 4 E4B는 llama.cpp가 고정한 transformers 4.57.6이 `gemma4`를 몰라서 5.20.1로 변환했다(별도 venv). 나머지는 4.57.6.
- Gemma 4 E4B는 "유효 4B"지만 레이어별 임베딩까지 합치면 약 8B라 Q4_K_M이 4.97 GiB다. 8GB 묶음 상한(최대 메모리 4GB)을 넘는다.
- 라이선스 원문 확인 결과와 제외한 모델(EXAONE 4.0, Kanana-2, HyperCLOVA X SEED, Llama 계열)은 0.2 마일스톤 3 보고에 있다.

## 화자 분리 측정 (requirements §12 위험 1)

`run_diar.py`가 sherpa-onnx의 회의 후 화자 분리(pyannote segmentation-3.0 + 화자 embedding +
군집)를 한국어 모의 회의와 AMI에서 잰다. `diar_pipe.py`는 같은 과정을 저장한 embedding 위에서
numpy로 다시 해서, 다시 뽑지 않고 군집 방법·threshold·작은 군집 바닥을 바꿔 본다. 결과는
`results/RESULTS.md` §9.

| 무엇 | 출처 | 라이선스 |
|---|---|---|
| 한국어 음성 | [Zeroth-Korean](https://www.openslr.org/40/) (OpenSLR 40) 낭독. test 10명은 test 회의에, train 중 12명(124 139 151 158 164 169 183 187 194 200 207 208)은 dev 회의에 섞는다(`diar_mix.py`) | CC BY 4.0 |
| 영어 회의 | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) test 16회의, Mix-Headset과 Array1-01 | CC BY 4.0 |
| AMI 정답 | [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) `2509d8933721023fab4def2618aabd5c28eb82e9`, `only_words/rttms/test`, `uems/test` | Apache-2.0 |
| 모델 | sherpa-onnx GitHub releases. 출처·라이선스·sha256은 `diar_models.json` | segmentation MIT, 3D-Speaker(CAM++, ERes2Net, ERes2NetV2) Apache-2.0, TitaNet·WeSpeaker CC BY 4.0(출처 표기 필요) |

받기와 확인은 `fetch_diar.sh` 하나로 한다. 받은 파일은 `bench/data/diar/`(gitignore)에만 두고,
파일 3542개의 sha256을 `diar_data.sha256`에 커밋해 두었다. 받을 때는 Content-Length와 맞는지,
.wav가 RIFF로 시작하는지 확인하고 다섯 번까지 다시 받는다. Zeroth는 10.3 GB 압축 파일을 한 번
흘려 읽으면서 위 화자만 남긴다.

```bash
bench/.venv/bin/pip install sherpa-onnx==1.13.8 onnxruntime==1.31.0 scipy==1.16.2 numpy==2.5.4 soundfile==0.14.0
bench/fetch_diar.sh                       # models ami zeroth, 끝에 sha256 확인
```

| 패키지 | 버전 | 라이선스 | 쓰는 곳 |
|---|---|---|---|
| sherpa-onnx | 1.13.8 (자체 onnxruntime 1.28.2 포함) | Apache-2.0 | 화자 분리 전체, embedding 추출 |
| onnxruntime | 1.31.0 | MIT | `diar_pipe`의 segmentation |
| scipy | 1.16.2 | BSD-3-Clause | `diar_pipe`의 계층 군집 |
| numpy | 2.5.4 | BSD-3-Clause | |
| soundfile | 0.14.0 | BSD-3-Clause | 음성 읽기 |

결과 JSON마다 `env`에 sherpa-onnx·onnxruntime·scipy·numpy 버전과 segmentation·embedding 모델의
sha256이 들어간다.

다시 돌리는 순서. 단계마다 결과가 저장되어 있으면 건너뛰고, `bench/data/diar/STOP`이 있으면 다음
단위 앞에서 멈춘다.

```bash
bench/.venv/bin/python -I bench/run_diar.py tune korean ami          # 5분 dev로 고른 threshold, test 회의(한국어 12, AMI 32)
bench/.venv/bin/python -I bench/run_diar.py online long              # 60초 조각 실시간 대 회의 후, 4시간 회의의 시간·메모리
bench/.venv/bin/python -I bench/run_diar.py absorb                   # 30분 dev로 고른 threshold×비율 흡수(diar_post), 탐색용
bench/.venv/bin/python -I bench/run_diar.py extract-dev verify       # 30분 dev embedding 저장, sherpa 재현 확인(DER 0.5%p 이내, 화자 수 동일)
bench/.venv/bin/python -I bench/run_diar.py grid                     # 군집 방법 × threshold × 바닥, 공통·room·call 따로 고름
bench/.venv/bin/python -I bench/run_diar.py preregister            # test를 보기 전에 쓸 값을 results/diar/preregistered.json에 한 번만 기록(덮어쓰지 않음)
bench/.venv/bin/python -I bench/run_diar.py extract-test pipe        # test embedding 저장, 사전 등록한 값만 그대로 적용
bench/.venv/bin/python -I -m unittest discover -s bench/tests -t bench    # 모델·데이터 없이 도는 테스트
```

call 장면의 대역 제한은 6aa83d1부터 2의 거듭제곱 길이로 0을 채워 FFT한다(4시간 call 하나가 4.9시간 걸리던 것을 48초로). 출력은 양 끝 1초 안에서 최대 0.026, 가운데에서는 16비트 한 단계(3e-5) 안팎으로 달라지므로(잡음 30분으로 잰 값), RESULTS.md §9의 한국어 결과(call 회의와 그 회의가 들어간 평균)는 지금 코드로는 바이트 단위로 재현되지 않는다. 그대로 다시 만들려면 그 결과를 커밋한 167a9b8을 체크아웃해서 `bench/data/diar/`의 한국어 회의를 지우고 위 순서를 돌린다. room 회의와 AMI는 영향이 없다.

테스트는 bench/.venv(위 pip 고정 명령으로 scipy까지 설치한 venv)에서 돌린다. 루트 `.venv`에는 scipy가 없어 diar_pipe 테스트가 실패한다.
