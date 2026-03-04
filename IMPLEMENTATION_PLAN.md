# ghostmeet Implementation Plan (10 Steps)

## Goal
Build a self-hosted AI meeting delegate with two modes:
- Ghost Mode: listen + transcribe + summarize
- Agent Mode: respond in-meeting with policy guardrails

---

## Step 1 — Scope freeze & architecture lock (Today)
**What**
- Finalize architecture, scope, and mode split
- Define out-of-scope for v0.1

**DoD**
- requirements.md updated and approved
- This 10-step plan documented

---

## Step 2 — Audio capture transport PoC (Today)
**What**
- Chrome extension captures tab audio
- Streams chunks to local backend via WebSocket
- Backend stores chunks and exposes basic stats

**DoD**
- Start capture from extension popup
- backend receives chunks (`/ws/audio`) and writes `.webm`
- `GET /api/sessions` shows received bytes/chunks

---

## Step 3 — Real-time STT pipeline (Whisper)
**What**
- Convert incoming audio chunks to transcribable frames
- Run faster-whisper incremental transcription

**DoD**
- transcript lines appear in backend logs/API in near-real-time
- configurable model (`tiny/base/small`)

---

## Step 4 — Extension live captions UI
**What**
- Side panel for live transcript view
- Speaker segments + timestamps

**DoD**
- captions update in UI while meeting audio is streaming

---

## Step 5 — Meeting summary engine (Claude)
**What**
- Post-meeting summary generation
- Decisions / Action items / Follow-ups format

**DoD**
- summary generated from transcript via API
- deterministic output schema

---

## Step 6 — Context briefing input
**What**
- Pre-meeting context form:
  - meeting topic
  - user role
  - talking points
  - constraints

**DoD**
- summary/responses incorporate briefing context

---

## Step 7 — Agent Mode policy + response engine
**What**
- Build response policy layer:
  - when to respond
  - safe fallback lines
  - no unauthorized commitments

**DoD**
- policy tests pass on sample scenarios
- generated responses include confidence/fallback when uncertain

---

## Step 8 — Agent Mode voice output path
**What**
- TTS + local virtual audio routing path
- optional (off by default)

**DoD**
- one-click “speak response” works on local setup guide

---

## Step 9 — Packaging & DX
**What**
- one-command backend run
- extension load instructions
- sample config and troubleshooting

**DoD**
- fresh machine setup in under 10 minutes

---

## Step 10 — Demo + OSS release
**What**
- produce polished demo video
- README narrative + architecture diagram + examples

**DoD**
- public repo ready
- demo covers Ghost mode end-to-end

---

## Today’s execution boundary
- ✅ Do only Step 1 and Step 2
- ❌ Do not implement STT/summary yet
