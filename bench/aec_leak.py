"""The far side of a call leaking into the microphone, for the echo bench (REC-4).

The diarization bench's bleed is the system track at -25 dB with no delay and no room
(`diar_mix._channels`). A loudspeaker leak is later (output and input buffers, the air),
smeared (a room), and on another clock (the speaker's and the microphone's crystals
differ by tens of ppm). `leak` makes each of these from the system track; nothing in it
is a real recording.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RATE = 16000
BLOCK = 1 << 20  # samples per step of the drift's resampling, so a 4-hour track fits


@dataclass(frozen=True)
class Condition:
    level_db: float = -25.0
    delay_ms: float = 0.0
    rt60: float = 0.0       # seconds; 0 is no room
    drift_ppm: float = 0.0  # the far side's clock slower by this much
    seed: int = 0


def room(rt60: float, seed: int) -> np.ndarray:
    """A made-up impulse response: the direct path, then noise decaying 60 dB in `rt60`,
    as much energy in the tail as in the direct path; unit energy in all."""
    rng = np.random.default_rng(seed)
    n = int(1.2 * rt60 * RATE)
    t = np.arange(n) / RATE
    tail = rng.standard_normal(n) * np.exp(-np.log(1000.0) * t / rt60)
    tail[: int(0.002 * RATE)] = 0.0  # first reflections after 2 ms
    tail *= 1.0 / np.sqrt((tail ** 2).sum())
    h = tail
    h[0] = 1.0
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)


def _drift(x: np.ndarray, ppm: float) -> np.ndarray:
    eps = ppm * 1e-6
    out = np.empty(len(x), np.float32)
    grid = np.arange(len(x))
    for a in range(0, len(x), BLOCK):
        pos = np.arange(a, min(a + BLOCK, len(x)), dtype=np.float64) * (1.0 - eps)
        lo = int(pos[0])
        hi = min(len(x), int(pos[-1]) + 2)
        out[a:a + len(pos)] = np.interp(pos, grid[lo:hi], x[lo:hi])
    return out


def leak(system: np.ndarray, c: Condition) -> np.ndarray:
    """What of `system` reaches the microphone, as long as `system`."""
    from scipy.signal import oaconvolve

    y = np.asarray(system, np.float32)
    if c.drift_ppm:
        y = _drift(y, c.drift_ppm)
    if c.rt60:
        y = oaconvolve(y, room(c.rt60, c.seed))[: len(system)].astype(np.float32)
    d = int(round(c.delay_ms * RATE / 1000))
    if d:
        y = np.concatenate([np.zeros(d, np.float32), y[:-d]])
    return y * np.float32(10 ** (c.level_db / 20))
