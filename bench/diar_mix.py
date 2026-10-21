"""Mock meetings for the diarization bench, from real recordings of known speakers.

    from diar_mix import mix
    m = mix(pool, speakers=4, minutes=30, seed=1, overlap=0.1, short=0.15, scene="room")
    m.audio (16 kHz mono float32), m.turns (the reference, der.Turn), m.segments

`pool` maps a speaker to their clips (Zeroth-Korean's read sentences, one speaker each).
Each clip is trimmed to its speech, so the reference starts and ends where the voice
does. Turns follow one another with short pauses; a share of the speech overlaps the
turn before it (steered to `overlap`, the fraction of speech time with two or more
voices); a share of the turns (`short`) are short replies from a listener, 0.3 to 0.9 s
cut from one of their clips, laid over the current speaker or into the pause after.

Scenes, applied after the turns are laid out (the reference never changes):
- clean: as recorded
- room: one microphone in a room: a reverberant tail per speaker (each at its own
  distance), and a noise floor
- call: the far side of a call: band-limited to 300-3400 Hz, mu-law companded, a little
  noise

Limits, to state with every result: read speech, not conversation; each speaker's clips
come from one recording setup, so the channel can give a speaker away (the room scene's
per-speaker reverb and gain blur it but do not remove it); short replies are fragments
of sentences, not real "네"s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from der import Turn

RATE = 16_000
SHORT_MIN, SHORT_MAX = 0.3, 0.9


@dataclass
class Meeting:
    audio: np.ndarray
    turns: list[Turn]
    segments: list[tuple[float, float]]
    roles: dict[str, str] = field(default_factory=dict)  # speaker -> "quiet", "late" or "early"
    me: str | None = None          # with channels: the one speaker on the microphone
    mic: np.ndarray | None = None
    system: np.ndarray | None = None


def trim(clip: np.ndarray, floor_db: float = -35.0) -> np.ndarray:
    """The clip from its first to its last 20 ms frame within `floor_db` of its loudest."""
    frame = RATE // 50
    n = len(clip) // frame
    if n == 0:
        return clip[:0]
    rms = np.sqrt((clip[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    if rms.max() < 1e-4:
        return clip[:0]
    loud = np.nonzero(20 * np.log10(rms / rms.max()) > floor_db)[0]
    return clip[loud[0] * frame:(loud[-1] + 1) * frame]


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:n].astype(np.float32)


def _room_response(rng: np.random.Generator) -> np.ndarray:
    """A direct path and an exponentially decaying diffuse tail (RT60 0.25 to 0.6 s);
    farther speakers get a weaker direct path."""
    rt60 = rng.uniform(0.25, 0.6)
    n = int(rt60 * RATE)
    t = np.arange(n) / RATE
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * rng.uniform(0.15, 0.4)
    delay = int(rng.uniform(0.002, 0.012) * RATE)
    h = np.zeros(n + delay, dtype=np.float64)
    h[delay] = rng.uniform(0.5, 1.0)
    h[delay + 1: delay + 1 + n - 1] += tail[1:]
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)


def _band_limit(x: np.ndarray, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    # zero-padded to a power of two: at a length with a large prime factor one FFT of a
    # four-hour call takes over ten minutes
    size = 1 << (len(x) - 1).bit_length()
    spec = np.fft.rfft(x, size)
    f = np.fft.rfftfreq(size, 1 / RATE)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, size)[:len(x)].astype(np.float32)


def _mu_law(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round(y * 127) / 127  # 8 bits
    return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu).astype(np.float32)


def mix(pool: dict[str, list[np.ndarray]], speakers: int, minutes: float, seed: int,
        overlap: float, short: float, scene: str = "clean", *, quiet: int = 0, quiet_share: float = 0.02,
        late: int = 0, late_from: float = 0.65, early: int = 0, early_until: float = 0.35,
        channels: bool = False, bleed_db: float | None = None) -> Meeting:
    """`quiet` speakers take about `quiet_share` of the turns, `late` ones join at `late_from`
    of the meeting and `early` ones leave at `early_until`. With `channels` the first speaker
    is "me", rendered alone on a microphone track and everyone else on a system track as a
    call, `bleed_db` of the system track leaking into the microphone. Left at their defaults,
    these draw nothing from the generator: the same seed makes the same meeting as before."""
    rng = np.random.default_rng(seed)
    names = sorted(pool)
    chosen = sorted(rng.choice(names, size=speakers, replace=False).tolist())
    me = chosen[0] if channels else None
    free = [s for s in reversed(chosen) if s != me]  # roles go from the last speaker back
    roles = {}
    for role, n in (("quiet", quiet), ("late", late), ("early", early)):
        for _ in range(n):
            roles[free.pop(0)] = role
    if len(free) < 1:
        raise ValueError("every speaker has a role: nobody is left to carry the meeting")
    clips = {s: [c for c in (trim(c) for c in pool[s]) if len(c) > RATE // 2] for s in chosen}
    gain = {s: rng.uniform(0.5, 1.0) for s in chosen}
    next_clip = {s: int(rng.integers(len(clips[s]))) for s in chosen}

    def clip_of(s: str) -> np.ndarray:
        c = clips[s][next_clip[s] % len(clips[s])]
        next_clip[s] += 1
        return c

    placed: list[tuple[str, float, np.ndarray]] = []  # speaker, start, samples
    total = minutes * 60
    t, current = 0.5, None
    speech = np.zeros(int((total + 30) * 100), dtype=np.int16)  # 10 ms grid, for steering

    def ratio() -> float:
        busy = speech > 0
        return float((speech > 1).sum() / busy.sum()) if busy.any() else 0.0

    def lay(s: str, start: float, samples: np.ndarray):
        end = start + len(samples) / RATE
        speech[int(start * 100):int(end * 100)] += 1
        placed.append((s, start, samples))
        return end

    def present(s: str, at: float) -> bool:
        r = roles.get(s)
        # margins, so a turn that starts a little before or after `at` stays inside
        return not ((r == "late" and at < late_from * total + 2) or (r == "early" and at + 3 > early_until * total))

    def pick(among: list[str]) -> str:
        if not roles:
            return rng.choice(among)
        here = [s for s in among if present(s, t)] or [s for s in among if roles.get(s) != "late"]
        rest = sum(1 for s in here if roles.get(s) != "quiet")
        w = np.array([quiet_share * max(rest, 1) / (1 - quiet_share) if roles.get(s) == "quiet" else 1.0
                      for s in here])
        return here[int(rng.choice(len(here), p=w / w.sum()))]

    while t < total:
        cur_end = t
        if current is not None and short > 0 and rng.random() < short:
            listener = pick([s for s in chosen if s != current])
            c = clip_of(listener)
            d = int(rng.uniform(SHORT_MIN, min(SHORT_MAX, len(c) / RATE)) * RATE)
            a = int(rng.integers(0, max(1, len(c) - d)))
            frag = c[a:a + d]
            last_start, last_end = placed[-1][1], placed[-1][1] + len(placed[-1][2]) / RATE
            if ratio() < overlap and last_end - last_start > 1.2:
                start = rng.uniform(last_start + 0.3, last_end - len(frag) / RATE - 0.1)
            else:
                start = t + rng.uniform(0.15, 0.5)
            end = lay(listener, start, frag * gain[listener])
            t = max(t, end)
            continue
        speaker = chosen[int(rng.integers(len(chosen)))] if not roles else pick(chosen)
        if current is not None and speaker == current and rng.random() < 0.85:
            speaker = pick([s for s in chosen if s != current])
        c = clip_of(speaker) * gain[speaker] * rng.uniform(0.85, 1.15)
        if current is not None and speaker != current and ratio() < overlap and placed:
            last_start = placed[-1][1]
            start = max(last_start + 0.5, t - rng.uniform(0.4, 1.6))
        else:
            start = t + float(np.clip(rng.lognormal(np.log(0.45), 0.5), 0.1, 2.5))
        cur_end = lay(speaker, start, c.astype(np.float32))
        current = speaker
        t = max(t, cur_end)

    length = int((max(s + len(x) / RATE for _, s, x in placed) + 0.6) * RATE)
    audio = np.zeros(length, dtype=np.float32)
    rooms = {s: _room_response(rng) for s in chosen} if scene == "room" else {}
    for s, start, x in placed:
        if scene == "room":
            x = _fft_convolve(x, rooms[s])
        a = int(start * RATE)
        n = min(len(x), length - a)
        audio[a:a + n] += x[:n]
    if scene in ("room", "call"):
        level = float(np.sqrt((audio ** 2).mean()) + 1e-9)
        snr = 25.0 if scene == "room" else 30.0
        audio += (rng.standard_normal(length) * level / 10 ** (snr / 20)).astype(np.float32)
    if scene == "call":
        audio = _mu_law(np.clip(_band_limit(audio), -1, 1))
    peak = float(np.abs(audio).max())
    if peak > 0.95:
        audio *= 0.95 / peak

    turns = sorted((Turn(s, round(st, 3), round(st + len(x) / RATE, 3)) for s, st, x in placed),
                   key=lambda t: (t.start, t.end))
    m = Meeting(audio=audio, turns=turns, segments=[(t.start, t.end) for t in turns], roles=roles)
    if channels:
        m.me, m.mic, m.system = me, *_channels(placed, me, length, rng, bleed_db)
    return m


def _channels(placed, me: str, length: int, rng: np.random.Generator, bleed_db: float | None):
    """The microphone: me alone, dry, over a laptop's noise floor (30 dB). System audio:
    everyone else as the far side of a call."""
    mic = np.zeros(length, dtype=np.float32)
    system = np.zeros(length, dtype=np.float32)
    for s, start, x in placed:
        a = int(start * RATE)
        n = min(len(x), length - a)
        (mic if s == me else system)[a:a + n] += x[:n]
    for x, snr in ((mic, 30.0), (system, 30.0)):
        level = float(np.sqrt((x ** 2).mean()) + 1e-9)
        x += (rng.standard_normal(length) * level / 10 ** (snr / 20)).astype(np.float32)
    system = _mu_law(np.clip(_band_limit(system), -1, 1))
    if bleed_db is not None:
        mic = mic + system * np.float32(10 ** (bleed_db / 20))
    for x in (mic, system):
        peak = float(np.abs(x).max())
        if peak > 0.95:
            x *= 0.95 / peak
    return mic, system
