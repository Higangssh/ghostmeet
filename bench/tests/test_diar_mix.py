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


def pool(n=6, each=8, low=200):
    rng = np.random.default_rng(0)
    return {f"s{i}": [tone(low + 60 * i, float(rng.uniform(2, 6)), pad=0.4) for _ in range(each)] for i in range(n)}


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


def share(turns: list[Turn], speaker: str) -> float:
    total = sum(t.end - t.start for t in turns)
    return sum(t.end - t.start for t in turns if t.speaker == speaker) / total


class PeopleTests(unittest.TestCase):
    """Who is in the meeting when, and how much they say."""

    def mix(self, **kw):
        return dm.mix(pool(), speakers=5, minutes=20, seed=3, overlap=0.1, short=0.15, scene="clean", **kw)

    def test_the_new_options_left_alone_make_the_same_meeting_as_before(self):
        plain = dm.mix(pool(), speakers=4, minutes=3, seed=1, overlap=0.15, short=0.2, scene="clean")
        again = dm.mix(pool(), speakers=4, minutes=3, seed=1, overlap=0.15, short=0.2, scene="clean",
                       quiet=0, late=0, early=0)
        self.assertEqual(again.turns, plain.turns)
        self.assertTrue(np.array_equal(again.audio, plain.audio))
        self.assertEqual(plain.roles, {})

    def test_a_quiet_speaker_says_about_the_share_asked(self):
        m = self.mix(quiet=1, quiet_share=0.02)
        (who,) = [s for s, r in m.roles.items() if r == "quiet"]
        self.assertGreater(share(m.turns, who), 0.005)
        self.assertLess(share(m.turns, who), 0.04)
        self.assertEqual(len({t.speaker for t in m.turns}), 5)

    def test_a_late_joiner_says_nothing_before_joining_and_an_early_leaver_nothing_after_leaving(self):
        m = self.mix(late=1, early=1, late_from=0.65, early_until=0.35)
        (late,) = [s for s, r in m.roles.items() if r == "late"]
        (early,) = [s for s, r in m.roles.items() if r == "early"]
        self.assertTrue(all(t.start >= 0.65 * 20 * 60 for t in m.turns if t.speaker == late))
        self.assertTrue(all(t.start < 0.35 * 20 * 60 for t in m.turns if t.speaker == early))
        self.assertTrue(any(t.speaker == late for t in m.turns))
        self.assertTrue(any(t.speaker == early for t in m.turns))

    def test_quiet_and_late_are_different_people(self):
        m = self.mix(quiet=1, late=1)
        self.assertEqual(sorted(m.roles.values()), ["late", "quiet"])


class BandLimitTests(unittest.TestCase):
    def test_an_awkward_length_is_filtered_at_a_power_of_two(self):
        # a length with a large prime factor makes one whole-length FFT of a four-hour call
        # take over ten minutes; the filter pads to a power of two instead
        sizes = []
        real = np.fft.rfft
        dm.np.fft.rfft = lambda x, n=None, *a, **k: (sizes.append(n or len(x)), real(x, n, *a, **k))[1]
        try:
            n = 3 * 5 * 7 * 1009
            t = np.arange(n) / dm.RATE
            out = dm._band_limit((np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 100 * t)).astype(np.float32))
        finally:
            dm.np.fft.rfft = real
        self.assertEqual(len(out), n)
        self.assertTrue(all(s >= n and s & (s - 1) == 0 for s in sizes), sizes)
        mid = slice(n // 4, 3 * n // 4)
        kept = np.sin(2 * np.pi * 1000 * t[mid])
        self.assertLess(float(np.abs(out[mid] - kept).max()), 0.05)   # 1 kHz kept, 100 Hz gone


class ChannelTests(unittest.TestCase):
    """A call as the app records it: "me" alone on the microphone, the rest on system audio.
    The tones sit inside the call's 300-3400 Hz band."""

    def setUp(self):
        self.m = dm.mix(pool(low=500), speakers=4, minutes=3, seed=2, overlap=0.1, short=0.15, scene="call",
                        channels=True)

    def test_the_microphone_holds_only_me_and_system_audio_everyone_else(self):
        m = self.m
        self.assertIn(m.me, {t.speaker for t in m.turns})
        self.assertEqual(len(m.mic), len(m.system))
        loud = lambda x, t: float(np.abs(x[int(t.start * SR):int(t.end * SR)]).mean())  # noqa: E731
        alone = [t for t in m.turns if not any(u is not t and u.start < t.end and t.start < u.end for u in m.turns)]
        for t in alone[:20]:
            on, off = (m.mic, m.system) if t.speaker == m.me else (m.system, m.mic)
            self.assertGreater(loud(on, t), 10 * loud(off, t), t)

    def test_me_is_a_full_participant_not_a_quiet_or_late_one(self):
        m = dm.mix(pool(), speakers=5, minutes=20, seed=3, overlap=0.1, short=0.15, scene="call",
                   channels=True, quiet=1, late=1)
        self.assertNotIn(m.me, m.roles)

    def test_bleed_puts_system_audio_into_the_microphone_quietly(self):
        b = dm.mix(pool(low=500), speakers=4, minutes=3, seed=2, overlap=0.1, short=0.15, scene="call",
                   channels=True, bleed_db=-25.0)
        self.assertEqual(b.turns, self.m.turns)
        them = [t for t in b.turns if t.speaker != b.me
                and not any(u.speaker == b.me and u.start < t.end and t.start < u.end for u in b.turns)]
        t = them[0]
        seg = lambda x: float(np.sqrt((x[int(t.start * SR):int(t.end * SR)] ** 2).mean()))  # noqa: E731
        self.assertAlmostEqual(20 * np.log10(seg(b.mic) / seg(b.system)), -25.0, delta=3.0)

    def test_the_one_track_mix_is_unchanged_by_asking_for_channels(self):
        one = dm.mix(pool(low=500), speakers=4, minutes=3, seed=2, overlap=0.1, short=0.15, scene="call")
        self.assertEqual(one.turns, self.m.turns)
        self.assertTrue(np.array_equal(one.audio, self.m.audio))


if __name__ == "__main__":
    unittest.main()
