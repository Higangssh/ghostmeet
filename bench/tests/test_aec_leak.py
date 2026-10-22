"""How the far side leaks into the microphone in the echo bench, on made-up signals."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aec_leak as al  # noqa: E402

SR = al.RATE


def click(n: int, at: int) -> np.ndarray:
    x = np.zeros(n, np.float32)
    x[at] = 1.0
    return x


class Leak(unittest.TestCase):
    def test_a_delay_moves_the_far_side_later_by_that_many_samples(self):
        x = click(SR, 1000)
        y = al.leak(x, al.Condition(delay_ms=80))
        self.assertEqual(int(np.argmax(np.abs(y))), 1000 + 80 * SR // 1000)

    def test_the_leak_is_at_its_level_below_the_far_side(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4 * SR).astype(np.float32)
        y = al.leak(x, al.Condition(level_db=-25))
        self.assertAlmostEqual(20 * np.log10(np.std(y) / np.std(x)), -25, delta=0.1)

    def test_a_room_adds_a_tail_that_decays_by_its_rt60(self):
        y = al.leak(click(SR, 0), al.Condition(rt60=0.3, level_db=0))
        e = y.astype(np.float64) ** 2
        early = e[int(0.05 * SR):int(0.10 * SR)].sum()
        late = e[int(0.35 * SR):int(0.40 * SR)].sum()
        # 60 dB per 0.3 s: the 50 ms window 0.3 s later is about 60 dB down
        self.assertAlmostEqual(10 * np.log10(early / late), 60, delta=6)
        self.assertGreater(len(np.flatnonzero(np.abs(y) > 1e-4)), 0.1 * SR, "a tail, not one tap")

    def test_the_room_is_the_same_for_the_same_seed(self):
        x = click(SR, 0)
        a = al.leak(x, al.Condition(rt60=0.3, seed=7))
        self.assertTrue(np.array_equal(a, al.leak(x, al.Condition(rt60=0.3, seed=7))))
        self.assertFalse(np.array_equal(a, al.leak(x, al.Condition(rt60=0.3, seed=8))))

    def test_drift_runs_the_far_side_slow_so_a_late_click_lands_later(self):
        # 100 ppm: after 1000 s the leak is 0.1 s behind; the click lands 100 ms late
        n = 1001 * SR
        x = click(n, 1000 * SR)
        y = al.leak(x, al.Condition(drift_ppm=100, level_db=0))
        self.assertAlmostEqual((int(np.argmax(np.abs(y))) - 1000 * SR) / SR, 0.1, delta=0.002)
        self.assertEqual(len(y), n, "the microphone's length")

    def test_no_condition_is_the_old_bleed(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(SR).astype(np.float32)
        y = al.leak(x, al.Condition(level_db=-25))
        np.testing.assert_allclose(y, x * np.float32(10 ** (-25 / 20)), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
