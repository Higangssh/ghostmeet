"""The meeting mixer for the diarization bench, on made-up "voices" (tones), not speech."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diar_mix as dm  # noqa: E402
from der import Turn  # noqa: E402

SR = dm.RATE


def tone(freq: float, sec: float, pad: float = 0.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    body = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    z = np.zeros(int(pad * SR), dtype=np.float32)
    return np.concatenate([z, body, z])


def pool(n=6, each=8):
    rng = np.random.default_rng(0)
    return {f"s{i}": [tone(200 + 60 * i, float(rng.uniform(2, 6)), pad=0.4) for _ in range(each)] for i in range(n)}


def overlap_ratio(turns: list[Turn]) -> float:
    end = max(t.end for t in turns)
    n = int(end * 100) + 1
    count = np.zeros(n, dtype=np.int32)
    for t in turns:
        count[int(t.start * 100):int(t.end * 100)] += 1
    speech = (count > 0).sum()
    return float((count > 1).sum() / speech)


class TrimTests(unittest.TestCase):
    def test_silence_around_a_clip_is_cut(self):
        clip = dm.trim(tone(300, 2.0, pad=0.5))
        self.assertAlmostEqual(len(clip) / SR, 2.0, delta=0.05)

    def test_a_silent_clip_is_dropped(self):
        self.assertEqual(len(dm.trim(np.zeros(SR, dtype=np.float32))), 0)


class MixTests(unittest.TestCase):
    def setUp(self):
        self.m = dm.mix(pool(), speakers=4, minutes=3, seed=1, overlap=0.15, short=0.2, scene="clean")

    def test_the_meeting_is_as_long_as_asked_and_the_turns_lie_in_it(self):
        sec = len(self.m.audio) / SR
        self.assertAlmostEqual(sec, 180, delta=10)
        self.assertTrue(all(0 <= t.start < t.end <= sec + 1e-6 for t in self.m.turns))
        self.assertEqual(self.m.turns, sorted(self.m.turns, key=lambda t: (t.start, t.end)))

    def test_it_has_the_speakers_asked_for(self):
        self.assertEqual(len({t.speaker for t in self.m.turns}), 4)

    def test_overlap_is_near_the_target(self):
        self.assertAlmostEqual(overlap_ratio(self.m.turns), 0.15, delta=0.07)
        clean = dm.mix(pool(), speakers=4, minutes=3, seed=1, overlap=0.0, short=0.0, scene="clean")
        self.assertLess(overlap_ratio(clean.turns), 0.01)

    def test_short_turns_are_short_and_about_as_many_as_asked(self):
        shorts = [t for t in self.m.turns if t.end - t.start <= dm.SHORT_MAX + 1e-6]
        self.assertAlmostEqual(len(shorts) / len(self.m.turns), 0.2, delta=0.1)
        self.assertTrue(all(t.end - t.start >= dm.SHORT_MIN - 1e-6 for t in shorts))

    def test_turns_are_where_the_sound_is(self):
        a = self.m.audio
        for t in self.m.turns[:20]:
            seg = a[int(t.start * SR):int(t.end * SR)]
            self.assertGreater(float(np.abs(seg).mean()), 0.01, t)
        # a gap between turns (clean scene) is silent
        ends = sorted(self.m.turns, key=lambda t: t.end)
        for t1, t2 in zip(ends, ends[1:]):
            covered = any(u.start <= t1.end + 0.02 < u.end for u in self.m.turns)
            if not covered and t2.start - t1.end > 0.2:
                gap = a[int((t1.end + 0.05) * SR):int((t1.end + 0.15) * SR)]
                self.assertLess(float(np.abs(gap).max()), 1e-6)
                break

    def test_the_same_seed_makes_the_same_meeting(self):
        again = dm.mix(pool(), speakers=4, minutes=3, seed=1, overlap=0.15, short=0.2, scene="clean")
        self.assertEqual(again.turns, self.m.turns)
        self.assertTrue(np.array_equal(again.audio, self.m.audio))

    def test_room_and_call_scenes_change_the_sound_not_the_turns(self):
        for scene in ("room", "call"):
            other = dm.mix(pool(), speakers=4, minutes=3, seed=1, overlap=0.15, short=0.2, scene=scene)
            self.assertEqual(other.turns, self.m.turns, scene)
            self.assertFalse(np.array_equal(other.audio[: len(self.m.audio)], self.m.audio), scene)
            self.assertLessEqual(float(np.abs(other.audio).max()), 1.0)

    def test_segments_are_the_turns_spans(self):
        self.assertEqual(self.m.segments, [(t.start, t.end) for t in self.m.turns])


if __name__ == "__main__":
    unittest.main()
