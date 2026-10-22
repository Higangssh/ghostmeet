"""The numpy Kaldi fbank (the Rust fbank's reference), on made-up signals."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kaldi_fbank as kf  # noqa: E402


class Fbank(unittest.TestCase):
    def test_frames_every_10_ms_centred_without_snipping_the_edges(self):
        for n in (400, 1000, 16000, 16081):
            self.assertEqual(kf.fbank(np.zeros(n, np.float32)).shape, ((n + 80) // 160, 80))

    def test_silence_is_the_floor_everywhere(self):
        f = kf.fbank(np.zeros(8000, np.float32))
        self.assertTrue(np.all(f == f[0, 0]))
        self.assertAlmostEqual(float(f[0, 0]), float(np.log(np.finfo(np.float32).eps)), places=4)

    def test_a_tone_is_loudest_in_the_bin_around_its_frequency(self):
        t = np.arange(16000) / 16000
        for hz in (300, 1000, 3000):
            f = kf.fbank((0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32))
            banks = kf.mel_banks(80, 512, 16000, 20.0, -400.0)
            centre = (np.arange(257) * 16000 / 512)[banks.argmax(axis=1)]
            self.assertLess(abs(centre[int(np.median(f.argmax(axis=1)))] - hz), 0.08 * hz + 40, hz)

    def test_the_top_bin_stops_400_hz_below_nyquist(self):
        banks = kf.mel_banks(80, 512, 16000, 20.0, -400.0)
        top = np.flatnonzero(banks[-1])[-1] * 16000 / 512
        self.assertLessEqual(top, 7600)
        self.assertGreater(top, 7400)


if __name__ == "__main__":
    unittest.main()
