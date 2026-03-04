# attend — AI Meeting Proxy

## 목표
줌/구글밋 링크를 넣으면 AI가 대신 회의에 참석하여 듣고, 요약하고, 질문에 대답하는 오픈소스 셀프호스팅 도구.

## 배포 모델
- **오픈소스 셀프호스팅** — 유저가 자기 API 키로 직접 실행
- 데모 영상으로 바이럴 → GitHub 유입 → 셀프호스팅
- SaaS 아님 (비용 부담 없음)

## 완료 기준 (DoD)
- [ ] 로컬에서 `python -m attend` → 웹 UI 열림
- [ ] 미팅 링크 입력 → 봇이 Zoom 회의 입장
- [ ] 실시간 전사(transcript) 웹에 표시
- [ ] 회의 종료 시 요약 + 액션 아이템 생성
- [ ] AI가 질문 감지 → 음성으로 대답
- [ ] 데모 영상 완성

## MVP 범위
### 포함
- Zoom 미팅 지원 (Recall.ai Meeting Bot API)
- 실시간 전사 (Deepgram via Recall.ai)
- LLM 요약 + 응답 (Claude API)
- TTS 음성 출력 (ElevenLabs)
- 로컬 웹 UI (FastAPI + 내장 프론트엔드)
- "내 정보" 설정 (이름, 역할, 주요 컨텍스트)
- .env 기반 API 키 설정

### 제외 (v2)
- Google Meet / Teams 지원
- 목소리 클론
- 영상 아바타
- 사용자 인증 / 결제

## 금지 조건
- API 키 하드코딩 금지
- 회의 데이터 외부 전송 금지
- 모든 데이터 로컬에만 저장

## 기술 스택
- **Backend**: Python 3.12 + FastAPI (API + 정적 파일 서빙)
- **Frontend**: 내장 (Jinja2 templates + htmx + Tailwind CDN)
- **Meeting Bot**: Recall.ai API
- **STT**: Deepgram (Recall.ai 내장 transcription)
- **LLM**: Claude API (Anthropic)
- **TTS**: ElevenLabs API

## 디렉토리 구조
```
attend/
├── attend/
│   ├── __main__.py      # entrypoint
│   ├── app.py           # FastAPI app
│   ├── meeting.py       # Recall.ai bot management
│   ├── agent.py         # LLM context + response
│   ├── tts.py           # TTS output to meeting
│   ├── config.py        # .env loading
│   ├── templates/
│   │   ├── index.html   # main page
│   │   └── meeting.html # live dashboard
│   └── static/
│       └── style.css
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## 7일 로드맵
- Day 1: 프로젝트 셋업 + Recall.ai 연동 + 봇 입장 테스트
- Day 2: 실시간 전사 파이프라인 (WebSocket → 브라우저)
- Day 3: LLM 요약 + 컨텍스트 관리
- Day 4: AI 응답 + TTS → 봇이 회의에서 발언
- Day 5: 웹 UI (대시보드 + 설정)
- Day 6: E2E 테스트 + README + install
- Day 7: 데모 영상 촬영
