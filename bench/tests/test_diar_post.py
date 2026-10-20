"""Small clusters absorbed into the nearest big one, on hand-made turns and centroids."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from der import Turn  # noqa: E402
from diar_post import absorb  # noqa: E402

TURNS = [Turn("a", 0, 30), Turn("b", 30, 60), Turn("c", 60, 62), Turn("d", 62, 63)]
# c sounds like b, d sounds like a
CENTROIDS = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
             "c": np.array([0.1, 0.9]), "d": np.array([0.8, 0.3])}


class Absorb(unittest.TestCase):
    def test_a_cluster_with_less_speech_than_the_floor_takes_the_nearest_big_ones_label(self):
        out = absorb(TURNS, CENTROIDS, min_sec=5)
        self.assertEqual([t.speaker for t in out], ["a", "b", "b", "a"])
        self.assertEqual([(t.start, t.end) for t in out], [(t.start, t.end) for t in TURNS])

    def test_the_floor_counts_a_clusters_speech_over_all_its_turns(self):
        turns = TURNS + [Turn("c", 70, 74)]  # c now has 6 s in all
        self.assertEqual([t.speaker for t in absorb(turns, CENTROIDS, min_sec=5)],
                         ["a", "b", "c", "a", "c"])

    def test_a_zero_floor_changes_nothing(self):
        self.assertEqual(absorb(TURNS, CENTROIDS, min_sec=0), TURNS)

    def test_when_no_cluster_is_big_enough_nothing_is_merged(self):
        self.assertEqual(absorb(TURNS, CENTROIDS, min_sec=100), TURNS)

    def test_a_cluster_without_a_centroid_is_left_alone(self):
        cents = {k: v for k, v in CENTROIDS.items() if k != "d"}
        self.assertEqual([t.speaker for t in absorb(TURNS, cents, min_sec=5)], ["a", "b", "b", "d"])


if __name__ == "__main__":
    unittest.main()
