"""Convert incoming webm/opus audio chunks to 16kHz mono PCM via ffmpeg."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


async def webm_chunks_to_pcm(
    chunks: asyncio.Queue[bytes | None],
) -> AsyncIterator[bytes]:
    """Stream webm bytes through ffmpeg, yielding raw PCM chunks.

    Send None into the queue to signal EOF.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "webm",
        "-i", "pipe:0",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _feed_stdin():
        try:
            while True:
                chunk = await chunks.get()
                if chunk is None:
                    break
                if proc.stdin is not None:
                    proc.stdin.write(chunk)
                    await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if proc.stdin is not None:
                proc.stdin.close()

    feeder = asyncio.create_task(_feed_stdin())

    try:
        read_size = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH  # ~1 second of audio
        while True:
            data = await proc.stdout.read(read_size)
            if not data:
                break
            yield data
    finally:
        feeder.cancel()
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
