# ghostmeet — AI Meeting Delegate

## 1. 목표

브라우저에서 화상회의(Zoom, Google Meet, Teams 등)에 참석 중일 때,
Chrome Extension이 회의 오디오를 캡처하여 실시간 전사 + AI 요약 + 대리 응답을 제공하는
**셀프호스팅 오픈소스 도구**.

### 핵심 가치
- **설치 1개** — Chrome Extension
- **외부 SaaS 의존 0** — Recall.ai 등 필요 없음
- **플랫폼 무관** — Zoom, Meet, Teams, 브라우저면 다 됨
- **API 키 최소화** — Anthropic 1개 (STT는 로컬 Whisper)

### 한 줄 설명
> "Any meeting, any platform. Just open your browser."

---

## 2. 기능 모드

### Ghost Mode (MVP, v0.1)
- 유저가 직접 회의 참석 (평소처럼)
- Extension이 탭 오디오 캡처
- 실시간 전사 (로컬 Whisper)
- 회의 중 실시간 자막 표시 (Extension popup 또는 side panel)
- 회의 종료 시 요약 + 액션 아이템 생성 (Claude API)
- **유저가 "대신 말하기" 없음 — 듣기 + 요약만**

### Agent Mode (v0.2, 이후)
- Ghost Mode 기능 전부 포함
- AI가 응답 텍스트 생성 → TTS → 가상 오디오 장치로 회의에 주입
- 사전 브리핑 (주제, 자료, 내 입장) 기반 맥락 탑재
- 발언 가드레일 ("결정은 보류", "확인 후 답변" 등 정책)
- **OS 오디오 라우팅 필요 (BlackHole 등)**

---

## 3. 아키텍처

```
┌─────────────────────────────────────┐
│         Chrome Extension            │
│                                     │
│  ┌──────────┐   ┌────────────────┐  │
│  │ tabCapture│   │  Side Panel /  │  │
│  │ Audio API │   │  Popup UI     │  │
│  └─────┬─────┘   └───────▲───────┘  │
│        │ audio chunks     │ transcript│
│        ▼                  │          │
│  ┌─────────────────────────┐        │
│  │   WebSocket Client      │        │
│  └─────────┬───────────────┘        │
└────────────┼────────────────────────┘
             │ ws://localhost:8877
             ▼
┌─────────────────────────────────────┐
│         Local Backend (Python)      │
│                                     │
│  ┌──────────┐   ┌────────────────┐  │
│  │ WebSocket │   │  Web Dashboard │  │
│  │ Server    │   │  (FastAPI)     │  │
│  └─────┬─────┘   └───────────────┘  │
│        │ audio                       │
│        ▼                             │
│  ┌──────────┐                        │
│  │ Whisper   │  (로컬 STT)           │
│  │ (local)   │                       │
│  └─────┬─────┘                        │
│        │ text                         │
│        ▼                              │
│  ┌──────────┐   ┌────────────────┐   │
│  │ Transcript│──▶│  Claude API    │   │
│  │ Buffer    │   │  (요약/응답)    │   │
│  └──────────┘   └────────────────┘   │
│                                      │
│  ┌──────────────────────────────┐    │
│  │  Meeting Context Store       │    │
│  │  (사전 브리핑, 유저 프로필)     │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

### 데이터 흐름 (Ghost Mode)
```
1. 유저가 Zoom 웹에서 회의 참석
2. Extension 활성화 → chrome.tabCapture.capture()로 탭 오디오 스트림 획득
3. Web Audio API로 PCM 청크 추출 (16kHz, mono)
4. WebSocket으로 로컬 백엔드에 스트리밍 전송
5. 백엔드: faster-whisper로 실시간 STT
6. 전사 텍스트 → Extension에 실시간 푸시 (자막)
7. 회의 종료 (유저가 "요약" 클릭)
8. 전체 전사 → Claude API → 요약 + 액션 아이템
9. 결과를 Extension popup + 웹 대시보드에 표시
```

---

## 4. 기술 스택

| 레이어 | 기술 | 이유 |
|--------|------|------|
| Chrome Extension | Manifest V3, TypeScript | tabCapture API, Side Panel API |
| 오디오 캡처 | chrome.tabCapture + Web Audio API | 브라우저 네이티브, 안정적 |
| 오디오 전송 | WebSocket (binary frames) | 실시간 스트리밍에 적합 |
| 백엔드 | Python 3.12 + FastAPI | Whisper/AI 생태계 최적 |
| STT | faster-whisper (로컬) | 외부 API 불필요, CTranslate2 기반 빠름 |
| LLM | Anthropic Claude API | 요약/응답 품질 |
| 프론트 (대시보드) | FastAPI + Jinja2 + htmx | 심플, 별도 빌드 불필요 |
| 패키징 | pip install + Chrome Extension zip | 설치 간편 |

---

## 5. 디렉토리 구조

```
ghostmeet/
├── extension/                # Chrome Extension
│   ├── manifest.json         # Manifest V3
│   ├── background.js         # Service Worker (tabCapture 관리)
│   ├── content.js            # 미팅 탭 감지
│   ├── sidepanel.html        # 실시간 자막 + 요약 UI
│   ├── sidepanel.js          # WebSocket 클라이언트
│   ├── popup.html            # 간단 설정/상태
│   ├── popup.js
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
│
├── backend/                  # Python 로컬 서버
│   ├── __main__.py           # entrypoint (python -m backend)
│   ├── app.py                # FastAPI (WebSocket + REST + 대시보드)
│   ├── transcriber.py        # faster-whisper 래퍼
│   ├── agent.py              # Claude API (요약/응답 생성)
│   ├── meeting.py            # 회의 세션 관리 (전사 버퍼, 상태)
│   ├── config.py             # .env 로딩
│   ├── templates/
│   │   ├── index.html        # 대시보드 메인
│   │   └── meeting.html      # 회의별 상세
│   └── static/
│       └── style.css
│
├── requirements.txt          # Python 의존성
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.md           # 이 문서
```

---

## 6. 완료 기준 (DoD)

### MVP (Ghost Mode, v0.1)
- [ ] Chrome Extension 설치 → Zoom/Meet 탭 감지
- [ ] tabCapture로 오디오 캡처 시작/중지
- [ ] WebSocket으로 로컬 백엔드에 오디오 스트리밍
- [ ] faster-whisper로 실시간 전사
- [ ] Extension side panel에 실시간 자막 표시
- [ ] "요약" 버튼 → Claude API → 요약 + 액션 아이템
- [ ] 웹 대시보드에서 과거 회의 목록 + 요약 조회
- [ ] README + 설치 가이드
- [ ] 데모 영상 (Zoom 회의에서 실시간 자막 → 요약)

### 기술 검증 (Day 1 우선)
- [ ] chrome.tabCapture.capture()로 탭 오디오 캡처 PoC
- [ ] 캡처된 오디오 → WebSocket → faster-whisper 전사 PoC

---

## 7. 금지 조건

- 외부 SaaS 의존 금지 (Recall.ai, Deepgram 등)
- API 키 하드코딩 금지
- 회의 오디오/전사 데이터 외부 전송 금지 (로컬에만 저장)
- 유저 동의 없이 녹음 시작 금지 (명시적 "Start" 버튼)

---

## 8. 7일 로드맵

| Day | 목표 | 산출물 |
|-----|------|--------|
| 1 | 기술 검증 — tabCapture PoC + Whisper PoC | 오디오캡처→전사 파이프라인 동작 확인 |
| 2 | Extension 뼈대 — manifest, background, WebSocket | Extension 로드 + 오디오 전송 |
| 3 | 백엔드 뼈대 — FastAPI + Whisper 연동 | 실시간 전사 API 동작 |
| 4 | 실시간 자막 — Extension↔백엔드 양방향 | Side panel에 자막 표시 |
| 5 | AI 요약 — Claude API 연동 | 요약 + 액션 아이템 생성 |
| 6 | 대시보드 + 폴리시 | 웹 UI + 에러 처리 + 설정 |
| 7 | README + 데모 영상 | 배포 가능 상태 |

---

## 9. 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| tabCapture가 Zoom 웹에서 안 될 수 있음 | 치명적 | Day 1에 PoC로 검증. 안 되면 desktopCapture 대안 |
| Whisper 로컬 실행 속도 (Mac Mini M2) | 중간 | faster-whisper + tiny/base 모델로 시작 |
| Chrome Extension 스토어 심사 지연 | 낮음 | MVP는 개발자 모드 사이드로딩 |
| 오디오 포맷 호환성 | 중간 | 48kHz→16kHz 리샘플링 필요할 수 있음 |

---

## 10. 경쟁 차별화

| 기존 도구 | 방식 | ghostmeet 차이점 |
|-----------|------|--------------|
| Otter.ai / Fireflies | SaaS, 봇이 회의 입장 | 로컬, 봇 없음, 프라이버시 |
| Recall.ai | API, 비즈니스 전용 | 오픈소스, 누구나 |
| zoom-sidekick | Recall.ai 의존 | 외부 의존 0 |
| meetingbot | AWS 인프라 필요 | pip install + Extension |

**ghostmeet = "설치 2분, 외부 의존 0, 내 데이터는 내 컴퓨터에"**
