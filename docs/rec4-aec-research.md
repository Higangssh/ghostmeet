# REC-4 에코 제거 조사 (0.3 설계 메모)

문서 조사만 했다. 코드 실험은 하지 않았고, 확인되지 않은 것은 **추정**으로 적는다.

## 왜 필요한가

스피커로 통화를 들으면 상대 목소리가 마이크로 되들어온다. 앱은 통화의 마이크 채널을 통째로 "나"로 본다(SPK-1). 그래서 새어 든 상대 목소리는 "나"의 발화가 되고 전사에도 두 번 나온다.

bench의 조각 단위 test(`bench/results/RESULTS.md` §9, 합성 통화 6회의)에서 −25 dB 누출만으로 call DER이 크게 올랐다.

| 조건 | call DER |
|---|---:|
| 누출 없음 (CAM++ / ERes2NetV2) | 0.120 / 0.111 |
| −25 dB 누출 | 0.471 / 0.463 |
| 누출 + `drop_echo` | 0.158 / 0.150 |

## 앱에 이미 있는 것

앱은 mic와 system을 채널별로 디스크에 저장한다(`PcmStore`). 두 채널의 시각은 샘플 커서에서 나온다. 그러므로 **참조 신호(system)가 이미 정렬된 채로 있다.** 이것이 OS 에코 제거와 다른 선택지를 연다.

## macOS: VoiceProcessingIO (VPIO)

- VPIO는 입력과 출력을 한 유닛으로 묶은 Audio Unit이다. 자기 출력 경로로 나가는 소리를 알고, 그것을 마이크에서 지운다.
- 외부 참조 신호를 넣는 속성은 찾지 못했다. ScreenCaptureKit이 잡은 시스템 오디오를 VPIO의 참조로 줄 수는 없다고 본다.
- WWDC23 "What's new in voice processing"의 설명:
  - 앱의 voice 스트림을 뺀 모든 소리(다른 앱 포함)가 "other audio"다.
  - voice와 섞여 출력되고, 기본으로 **볼륨이 줄어든다(ducking).**
  - `kAUVoiceIOProperty_OtherAudioDuckingConfiguration`(AVAudioEngine에서는 `voiceProcessingOtherAudioDuckingConfiguration`)으로 줄이는 정도를 min~max로 고를 수 있다.
  - 이 설명으로 보면 다른 앱의 소리도 VPIO가 보는 출력에 들어간다. 그래서 회의 앱의 상대 목소리가 지워질 수 있다(**추정**, 문서에 명시 없음).
- **공존 문제:**
  - 지금 마이크는 cpal(HAL 기반 입력)로 연다. VPIO를 쓰면 같은 장치를 VPIO가 따로 잡는다.
  - Apple 포럼에 보고가 있다. VPIO를 켠 앱이 **다른 앱(Zoom 등)의 입력 스트림을 끊었다**는 것이다. 답은 확인하지 못했다.
  - 기록 앱이 사용자의 회의 앱 마이크를 끊거나 상대 목소리를 줄이면 안 된다. 이 둘이 VPIO의 가장 큰 위험이다.
- **장치 제약:** 오래된 PJSIP 티켓과 포럼 글에 "VPIO는 기본 또는 내장 입출력에서만 동작한다"는 보고가 있다. 블루투스 헤드셋에서는 게인이 크게 떨어진다는 보고도 있다.
- 결론: VPIO는 "회의 앱 옆에서 조용히 녹음한다"는 앱의 전제와 충돌할 가능성이 크다. 쓰려면 다음 셋을 실기기에서 먼저 확인해야 한다.
  1. Zoom이나 Meet가 동시에 켜져 있을 때 입력이 끊기지 않는가
  2. ducking을 min으로 두면 사용자가 듣는 소리가 그대로인가
  3. 다른 앱의 소리가 참조로 쓰이는가

## Windows: 통신 모드 AEC

두 가지 길이 있다.

1. **통신 스트림 + 장치의 AEC 효과(APO).**
   - 캡처 스트림을 통신 범주(`AudioCategory_Communications`)로 열면 장치 드라이버의 통신 처리 체인이 켜진다. 장치에 AEC APO가 있을 때만 해당한다.
   - 참조는 기본 렌더 장치다. Windows build 22621부터는 `IAcousticEchoCancellationControl::SetEchoCancellationRenderEndpoint`로 참조 렌더 장치를 고를 수 있다.
   - 끝점이 AEC를 지원해도 참조 제어를 지원하지 않을 수 있다. 이때 `GetService`가 `E_NOINTERFACE`를 낸다(Microsoft 문서).
   - 제약 1: AEC 유무가 **장치와 드라이버에 달려 있다.**
   - 제약 2: 통신 스트림이 열리면 Windows의 "통신 활동 감지 시 다른 소리 줄이기"(기본 80%)가 사용자의 다른 소리를 줄일 수 있다(**추정**, 확인 필요).
2. **Voice Capture DSP (`CLSID_CWMAudioAEC`, Vista부터).**
   - AEC, 마이크 배열, 잡음 억제, AGC, VAD를 묶은 DMO다. 기능마다 켜고 끌 수 있다.
   - source 모드는 DMO가 장치를 직접 잡는다.
   - **filter 모드는 앱이 마이크(stream 0)와 스피커(stream 1) 샘플을 넣는다.** 앱의 WASAPI loopback(system 채널)을 참조로 줄 수 있다.
   - AEC는 단일 채널만 지원한다. 출력은 16비트, 8/11.025/16/22.05 kHz다. 앱의 16 kHz mono와 맞는다.
   - 오래된 API(DMO)라서 앞으로 유지될지는 불확실하다.

## 대안: 녹음 뒤 소프트웨어 에코 제거 (권고)

두 채널이 시각을 맞춰 디스크에 있으므로, 회의 뒤 화자 분리 전에 mic 채널에서 system 채널을 지울 수 있다. OS마다 같은 코드다. 회의 중 캡처 경로(cpal, ScreenCaptureKit, WASAPI)는 건드리지 않는다. 다른 앱의 소리를 줄이지도 않고 다른 앱의 입력을 끊지도 않는다.

- **후보 crate:**
  - `webrtc-audio-processing`: WebRTC APM 바인딩. 원본 C++는 BSD-3-Clause다. crate 자체의 라이선스는 아직 확인하지 못했다.
  - `sonora`: AEC3를 순수 Rust로 옮긴 것, BSD-3-Clause, 0.1.0으로 초기 단계다.
  - 둘 다 허용 라이선스이지만 채택 전에 Cargo.toml의 라이선스와 의존성을 확인한다.
- **풀어야 할 것:**
  - 지연: 스피커 → 공기 → 마이크, 그리고 두 캡처 경로의 버퍼 차이가 있다. AEC3에는 지연 추정기가 있다.
  - 클럭 드리프트: 두 장치의 샘플 클럭이 다르다. 4시간이면 누적된다. 샘플 커서로 맞춘 시각에 GapFiller가 채운 무음이 섞인다.
  - 이중 통화: 내가 상대와 동시에 말할 때 내 목소리를 지우면 안 된다. bench의 "talking over the far side is still me" 테스트가 이것을 지킨다.
- **회의 중 자막에는 적용하지 않는다.** live 자막은 잠정값이다. 새어 든 상대 목소리가 "나" 자막에 잠깐 보이는 것은 0.3에서 받아들인다. final pass 전에 mic 채널을 정리하면 최종 전사에서도 중복이 빠진다.
- **하한선으로 `drop_echo`:**
  - bench의 거친 대용이다. 0.1초 프레임마다 mic와 system의 0-지연 상관이 0.5를 넘으면 그 프레임을 "나"에서 뺀다.
  - 합성 통화에서 DER 0.46→0.15를 냈다. 지연과 방 울림이 없는 누출만 지우므로 실제 스피커 환경에서는 이보다 못하다.
  - 소프트웨어 AEC가 실패할 때의 대체로, 지연 추정(상호상관의 최대점)을 더하면 같은 구조로 쓸 수 있다.
  - 이 대체는 오디오를 고치지 않고 "나" 구간만 덜어낸다. 전사 중복은 남는다.

## 확인할 것 (실험 순서)

1. 소프트웨어 AEC를 bench에 붙인다. 합성 누출에 지연(10~200 ms)과 방 울림을 더해 call DER, 이중 통화 보존, 4시간 드리프트를 잰다. 공개 음성만 쓴다.
2. 실제 Mac에서 스피커 통화를 녹음해 mic 누출 수준과 지연을 잰다. 오디오는 저장소에 넣지 않고, 숫자만 기록한다.
3. VPIO와 Windows 통신 모드는 1이 실패할 때만 실기기에서 공존 문제(다른 앱 입력 끊김, ducking)부터 확인한다.

## 출처

- Apple WWDC23 "What's new in voice processing": https://developer.apple.com/videos/play/wwdc2023/10235/
- Apple 포럼, VPIO로 다른 앱 입력이 끊김: https://developer.apple.com/forums/thread/751100
- Apple 포럼, macOS VPIO 장치 제약과 게인: https://developer.apple.com/forums/thread/66953
- Microsoft `IAcousticEchoCancellationControl`: https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nn-audioclient-iacousticechocancellationcontrol
- Microsoft Voice Capture DSP: https://learn.microsoft.com/en-us/windows/win32/medfound/voicecapturedmo
- `sonora` (AEC3 Rust 포팅): https://docs.rs/crate/sonora/0.1.0
