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
| sherpa-onnx VAD | k2-fsa/sherpa-onnx `silero_vad.onnx` | MIT |

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
`wcpp-small-cpu`, `sensevoice`, `sensevoice-coreml`.

`bench/`는 pytest가 수집하지 않는다(`pytest.ini`의 `norecursedirs`). 기존 테스트 스위트는
계속 모델·네트워크 없이 돈다.

## 무엇을 같게 맞췄나

- 디코딩 설정은 `backend/transcriber.py`와 같게 했다. 한국어로 고정했고, Whisper는 beam 5다.
  Silero VAD는 threshold 0.3, 최소 침묵 500ms, 패딩 200ms다. 창 사이에 텍스트 문맥은 넘기지
  않는다. faster-whisper는 프로덕션의 `WhisperTranscriber`를 그대로 호출한다.
- 스레드는 엔진마다 4개다(M4 성능 코어 수).
- whisper.cpp는 Homebrew의 `whisper-server`를 loopback에 띄워 모델을 한 번만 올린다. 요청마다
  WAV를 HTTP로 넘기는 비용이 들어가지만, 30초 창 기준 수 ms다.
- 꼬리 창 측정은 `backend.incremental.IncrementalTranscriber`를 그대로 쓴다.

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
