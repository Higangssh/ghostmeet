"""Real-time transcription using faster-whisper."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single transcribed segment."""
    text: str
    start: float
    end: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "timestamp": self.timestamp,
        }


class Transcriber:
    """Wraps faster-whisper for incremental transcription."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
    ):
        logger.info("Loading whisper model: %s (device=%s)", model_size, device)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self.language = language
        self.transcript: List[Segment] = []
        self._offset: float = 0.0
        logger.info("Whisper model loaded successfully")

    def transcribe_chunk(self, pcm_bytes: bytes) -> List[Segment]:
        """Transcribe a chunk of raw PCM audio (16kHz, 16-bit, mono).

        Returns list of new segments found in this chunk.
        """
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < 1600:  # less than 0.1s, skip
            return []

        segments_iter, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=3,
            best_of=3,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        new_segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            segment = Segment(
                text=text,
                start=self._offset + seg.start,
                end=self._offset + seg.end,
            )
            self.transcript.append(segment)
            new_segments.append(segment)
            logger.info("[%.1f-%.1f] %s", segment.start, segment.end, text)

        # advance offset by chunk duration
        chunk_duration = len(audio) / 16000.0
        self._offset += chunk_duration

        return new_segments

    def get_full_transcript(self) -> List[dict]:
        """Return all segments as dicts."""
        return [s.to_dict() for s in self.transcript]

    def get_full_text(self) -> str:
        """Return concatenated transcript text."""
        return " ".join(s.text for s in self.transcript)

    def reset(self):
        """Clear transcript and reset offset."""
        self.transcript.clear()
        self._offset = 0.0
