"""ghostmeet backend — audio capture + incremental STT."""
from __future__ import annotations

import datetime as dt
import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .decoder import StreamingWebmDecoder
from .incremental import IncrementalTranscriber, Segment
from .models import Session
from .pcm_store import PcmStore
from .pipeline import SessionPipeline
from .store import SessionStore
from .summarizer import Summary, generate_summary
from .transcriber import WhisperTranscriber, get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = RECORDINGS_DIR / "ghostmeet.db"

# config from env
WHISPER_MODEL = os.environ.get("GHOSTMEET_MODEL", "base")
WHISPER_DEVICE = os.environ.get("GHOSTMEET_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("GHOSTMEET_COMPUTE_TYPE", "float32")
WHISPER_LANGUAGE = os.environ.get("GHOSTMEET_LANGUAGE") or None
CHUNK_INTERVAL = float(os.environ.get("GHOSTMEET_CHUNK_INTERVAL", "10"))

# state
sessions: Dict[str, Session] = {}
pipelines: Dict[str, SessionPipeline] = {}
summaries: Dict[str, Summary] = {}
transcript_subscribers: Dict[str, List[WebSocket]] = {}
store: SessionStore | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global store
    store = SessionStore(DB_PATH)
    pipelines.clear()
    summaries.clear()
    sessions.clear()
    sessions.update(store.load_sessions())
    # nothing is capturing yet, so anything left mid-flight died with the last process
    for session in sessions.values():
        if session.status in ("streaming", "transcribing"):
            session.status = "interrupted"
    logger.info("Loaded %d session(s) from %s", len(sessions), DB_PATH)
    try:
        yield
    finally:
        store.close()
        store = None


app = FastAPI(title="ghostmeet-backend", version="0.5.0", lifespan=lifespan)

# The API has no authentication, so the browser extension is the only intended caller.
# Echoing back any Origin let any page the user visited read their meeting transcripts.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-z]+|moz-extension://[0-9a-f-]+|http://(127\.0\.0\.1|localhost)(:\d+)?)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_transcribe_fn(language: str | None) -> WhisperTranscriber:
    """Build a per-session transcriber over the process-wide shared model."""
    model = get_model(WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
    return WhisperTranscriber(model, language=language)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "ghostmeet-backend",
        "model": WHISPER_MODEL,
        "chunk_interval_sec": CHUNK_INTERVAL,
    }


@app.get("/api/sessions")
def list_sessions():
    return {
        "count": len(sessions),
        "sessions": {k: v.to_dict() for k, v in sessions.items()},
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    return sessions[session_id].to_dict()


def _segments_for(session_id: str) -> List[Segment]:
    """Live sessions answer from the pipeline producing them, finished ones from disk."""
    if session_id in pipelines:
        return pipelines[session_id].transcriber.transcript
    return store.load_segments(session_id)


@app.get("/api/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    if session_id not in pipelines and session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    segments = _segments_for(session_id)
    return {
        "session_id": session_id,
        "segments": [s.to_dict() for s in segments],
        "full_text": " ".join(s.text for s in segments),
        "segment_count": len(segments),
    }


@app.post("/api/sessions/{session_id}/summarize")
async def summarize_session(session_id: str):
    if session_id not in pipelines and session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    text = " ".join(s.text for s in _segments_for(session_id))
    if not text.strip():
        raise HTTPException(status_code=400, detail="transcript is empty")
    summary = await generate_summary(text, session_id)
    summaries[session_id] = summary
    store.save_summary(summary)
    return summary.to_dict()


@app.get("/api/sessions/{session_id}/summary")
def get_summary(session_id: str):
    summary = summaries.get(session_id) or store.load_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="summary not found — call POST /summarize first")
    return summary.to_dict()


@app.websocket("/ws/transcript/{session_id}")
async def ws_transcript(websocket: WebSocket, session_id: str):
    await websocket.accept()
    transcript_subscribers.setdefault(session_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        subscribers = transcript_subscribers.get(session_id, [])
        if websocket in subscribers:
            subscribers.remove(websocket)


async def _broadcast_segments(session_id: str, segments: List[Segment]) -> None:
    # persist first: captions are nice to have, a lost transcript is not recoverable
    store.add_segments(session_id, segments)

    session = sessions.get(session_id)
    if session is not None and session_id in pipelines:
        session.transcript_segments = len(pipelines[session_id].transcriber.transcript)
        store.save_session(session)

    subscribers = transcript_subscribers.get(session_id, [])
    payload = {"type": "transcript", "segments": [s.to_dict() for s in segments]}
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json(payload)
        except Exception:  # noqa: BLE001 - a closed side panel is routine
            dead.append(ws)
    for ws in dead:
        subscribers.remove(ws)


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    session_id = websocket.query_params.get("session") or dt.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    language = websocket.query_params.get("lang") or WHISPER_LANGUAGE

    if session_id in pipelines and sessions.get(session_id, None) is not None:
        if sessions[session_id].status == "streaming":
            logger.warning("Rejected duplicate capture for active session %s", session_id)
            await websocket.close(code=4409, reason="session already capturing")
            return

    archive_path = RECORDINGS_DIR / f"{session_id}.webm"
    pcm_path = RECORDINGS_DIR / f"{session_id}.pcm"
    # a re-used id starts fresh: appending to a previous recording corrupts both files
    pcm_path.unlink(missing_ok=True)

    session = Session(
        session_id=session_id,
        file=archive_path.name,
        language=language,
    )
    sessions[session_id] = session
    store.save_session(session)

    pcm = PcmStore(pcm_path)
    decoder = StreamingWebmDecoder(pcm)
    pipeline = SessionPipeline(
        decoder=decoder,
        transcriber=IncrementalTranscriber(pcm, _make_transcribe_fn(language)),
        on_segments=partial(_broadcast_segments, session_id),
        interval_sec=CHUNK_INTERVAL,
    )
    pipelines[session_id] = pipeline
    await pipeline.start()

    await websocket.send_json({"session_id": session_id})

    try:
        try:
            with archive_path.open("wb") as archive:
                while True:
                    message = await websocket.receive()
                    if message.get("bytes"):
                        chunk = message["bytes"]
                        archive.write(chunk)
                        session.chunks += 1
                        session.audio_bytes += len(chunk)
                        pipeline.feed(chunk)  # returns immediately
                    elif message.get("text") == "stop":
                        break
                    elif message.get("type") == "websocket.disconnect":
                        break
        except WebSocketDisconnect:
            pass

        session.status = "transcribing"
        await pipeline.stop()

        if decoder.error is not None:
            session.status = "error"
            session.error = str(decoder.error)
        else:
            session.status = "stopped"

        session.stopped_at = dt.datetime.now().isoformat(timespec="seconds")
        session.transcript_segments = len(pipeline.transcriber.transcript)
        session.duration_sec = round(pcm.duration, 2)
        store.save_session(session)
    finally:
        # Always release the handle and reclaim the working file, even if the client
        # vanished or a pass blew up mid-session. Order matters: the decoder writes into
        # the PCM store from its own thread, so it has to be stopped first.
        decoder.close()
        pcm.close()
        # the PCM working file is large (~115 MB/hour); the webm archive is the durable copy
        pcm_path.unlink(missing_ok=True)

    # tell the client the final pass is done, so it knows the transcript is complete and
    # summarising is now safe. The socket may already be gone, which is fine.
    try:
        await websocket.send_json(
            {
                "type": "complete",
                "session_id": session_id,
                "status": session.status,
                "segment_count": session.transcript_segments,
                "duration_sec": session.duration_sec,
            }
        )
    except Exception:  # noqa: BLE001 - client hung up first
        logger.debug("Could not send completion notice for %s", session_id)

    logger.info(
        "Session %s complete: %d chunks, %d bytes, %.0fs audio, %d segments",
        session_id,
        session.chunks,
        session.audio_bytes,
        session.duration_sec,
        session.transcript_segments,
    )


def default_host() -> str:
    """Loopback unless asked otherwise — the API is unauthenticated."""
    return os.environ.get("GHOSTMEET_HOST", "127.0.0.1")


def run() -> None:
    port = int(os.environ.get("GHOSTMEET_PORT", "8877"))
    uvicorn.run(app, host=default_host(), port=port)
