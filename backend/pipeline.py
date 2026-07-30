"""Per-session wiring: receive → decode → transcribe → broadcast.

The WebSocket handler only calls `feed()`, which hands bytes to the decoder thread and
returns. Transcription runs on its own schedule in a worker thread, so a slow pass slows
captions down but never stops audio from being received.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, List, Protocol

from .incremental import IncrementalTranscriber, Segment

logger = logging.getLogger(__name__)

# Bound on how many catch-up passes `stop()` will run before giving up, so a stuck
# session can never spin forever at hangup.
_MAX_DRAIN_PASSES = 2000


class Decoder(Protocol):
    def feed(self, chunk: bytes) -> None: ...
    def close(self, timeout: float = ...) -> None: ...


OnSegments = Callable[[List[Segment]], Awaitable[None]]


class SessionPipeline:
    """Owns the decode + transcribe loop for one capture session."""

    def __init__(
        self,
        decoder: Decoder,
        transcriber: IncrementalTranscriber,
        on_segments: OnSegments,
        interval_sec: float = 10.0,
    ):
        self.decoder = decoder
        self.transcriber = transcriber
        self._on_segments = on_segments
        self._interval = interval_sec
        # one worker per session: passes for a session are serialised, and a long pass
        # cannot occupy the shared default executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ghostmeet-stt")
        self._worker: asyncio.Task | None = None
        self._stopped = False

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    def feed(self, chunk: bytes) -> None:
        """Hand audio to the decoder. Returns without waiting for anything."""
        if self._stopped:
            return
        self.decoder.feed(chunk)

    async def stop(self) -> None:
        """Stop accepting audio, then transcribe whatever is left."""
        if self._stopped:
            return
        self._stopped = True

        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

        # closing joins the decoder thread, so keep it off the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.decoder.close)

        await self._drain()
        self._executor.shutdown(wait=False)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self._pass()

    async def _drain(self) -> None:
        """Catch up on any backlog, then take a final forced pass for trailing audio."""
        for _ in range(_MAX_DRAIN_PASSES):
            if self.transcriber.pending_samples <= 0:
                break
            await self._pass()
        else:
            logger.warning(
                "Gave up draining after %d passes, %d samples left",
                _MAX_DRAIN_PASSES,
                self.transcriber.pending_samples,
            )
        await self._pass(force=True)

    async def _pass(self, force: bool = False) -> None:
        loop = asyncio.get_running_loop()
        try:
            segments = await loop.run_in_executor(
                self._executor, self.transcriber.step, force
            )
        except Exception as exc:  # noqa: BLE001 - a bad pass must not end the session
            logger.error("Transcription pass failed: %s", exc, exc_info=True)
            return

        if not segments:
            return
        try:
            await self._on_segments(segments)
        except Exception as exc:  # noqa: BLE001 - a dead subscriber must not end the session
            logger.warning("Broadcasting %d segments failed: %s", len(segments), exc)
