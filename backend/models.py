"""Session and transcript data models."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Session:
    """Represents a capture + transcription session."""
    session_id: str
    chunks: int = 0
    audio_bytes: int = 0
    started_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    stopped_at: str | None = None
    status: str = "streaming"  # streaming | transcribing | stopped
    file: str = ""
    transcript_segments: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chunks": self.chunks,
            "audio_bytes": self.audio_bytes,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "status": self.status,
            "file": self.file,
            "transcript_segments": self.transcript_segments,
        }
