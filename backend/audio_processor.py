"""Process incoming webm/opus audio for transcription."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def transcribe_webm_file(webm_path: str | Path, transcriber) -> None:
    """Transcribe a complete webm file using faster-whisper's native file reader.

    This is more reliable than streaming PCM through ffmpeg pipe,
    because whisper can read webm/opus directly via its internal ffmpeg binding.
    """
    path = str(webm_path)
    logger.info("Transcribing file: %s", path)

    segments, info = transcriber.model.transcribe(
        path,
        language=transcriber.language,
        beam_size=5,
        best_of=3,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    new_segments = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        from .transcriber import Segment
        segment = Segment(
            text=text,
            start=seg.start,
            end=seg.end,
        )
        transcriber.transcript.append(segment)
        new_segments.append(segment)
        logger.info("[%.1f-%.1f] %s", segment.start, segment.end, text)

    return new_segments
