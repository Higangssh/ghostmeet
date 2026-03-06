"""Process incoming webm/opus audio for transcription."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe_webm_file(webm_path: str | Path, transcriber) -> list:
    """Transcribe a complete webm file using faster-whisper's native file reader.

    Returns list of new Segment objects.
    """
    from .transcriber import Segment

    path = str(webm_path)
    logger.info("Transcribing file: %s", path)

    segments_gen, info = transcriber.model.transcribe(
        path,
        language=transcriber.language,
        beam_size=5,
        best_of=3,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=1000,
            speech_pad_ms=400,
            threshold=0.3,
        ),
    )

    # force consume the generator — this is where actual inference happens
    raw_segments = list(segments_gen)
    logger.info("Whisper returned %d raw segments", len(raw_segments))

    new_segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        segment = Segment(
            text=text,
            start=seg.start,
            end=seg.end,
        )
        transcriber.transcript.append(segment)
        new_segments.append(segment)
        logger.info("[%.1f-%.1f] %s", segment.start, segment.end, text)

    logger.info("Transcription complete: %d segments", len(new_segments))
    return new_segments
