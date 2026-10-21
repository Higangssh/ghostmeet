"""The diarization scorer, on hand-made references and hypotheses."""
from __future__ import annotations

import itertools
import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import der  # noqa: E402

A, B, C = "A", "B", "C"


def turns(*spec):
    """(speaker, start, end) triples as Turn objects."""
    return [der.Turn(s, a, b) for s, a, b in spec]


class MappingTests(unittest.TestCase):
    def test_assignment_maximises_the_total(self):
        rng = random.Random(4)
        for _ in range(50):
            n, m = rng.randint(1, 5), rng.randint(1, 5)
            w = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
            best = max(
                (sum(w[i][j] for i, j in zip(rows, cols)) for k in range(min(n, m) + 1)
                 for rows in itertools.permutations(range(n), k)
                 for cols in itertools.permutations(range(m), k)),
                default=0,
            )
            pairs = der.assign(w)
            self.assertEqual(sum(w[i][j] for i, j in pairs), best, w)
            self.assertEqual(len({i for i, _ in pairs}), len(pairs))
            self.assertEqual(len({j for _, j in pairs}), len(pairs))


    def test_a_wide_matrix_is_assigned_exactly_and_quickly(self):
        # a grid config can leave thousands of hypothesis speakers against a few reference ones
        from scipy.optimize import linear_sum_assignment

        rng = random.Random(7)
        w = [[rng.random() for _ in range(3000)] for _ in range(4)]
        t = time.perf_counter()
        pairs = der.assign(w)
        self.assertLess(time.perf_counter() - t, 5.0)
        r, c = linear_sum_assignment([[-v for v in row] for row in w])
        self.assertAlmostEqual(sum(w[i][j] for i, j in pairs), sum(w[i][j] for i, j in zip(r, c)), places=9)
        tall = [list(col) for col in zip(*w)]
        self.assertEqual(sorted((j, i) for i, j in der.assign(tall)), pairs)


class ManyHypothesisSpeakers(unittest.TestCase):
    def test_hundreds_of_hypothesis_speakers_score_exactly_and_quickly(self):
        # A talks 0-600 s and B 600-1200 s; the hypothesis cuts it into 400 speakers of 3 s each
        ref = turns((A, 0, 600), (B, 600, 1200))
        hyp = [der.Turn(f"h{k}", 3.0 * k, 3.0 * (k + 1)) for k in range(400)]
        t = time.perf_counter()
        s = der.score(ref, hyp)
        self.assertLess(time.perf_counter() - t, 5.0)
        self.assertEqual((s.ref_speakers, s.hyp_speakers, len(s.mapping)), (2, 400, 2))
        self.assertAlmostEqual(s.missed + s.false_alarm, 0.0, places=6)
        self.assertAlmostEqual(s.confusion, 1200 - 6.0, places=6)    # two 3 s pieces are right


class DerTests(unittest.TestCase):
    def test_a_perfect_hypothesis_scores_zero_whatever_its_labels(self):
        ref = turns((A, 0, 5), (B, 5, 9), (A, 9, 12))
        hyp = turns(("x", 0, 5), ("y", 5, 9), ("x", 9, 12))
        s = der.score(ref, hyp)
        self.assertAlmostEqual(s.der, 0.0)
        self.assertAlmostEqual(s.total, 12.0, places=2)

    def test_missed_speech_false_alarm_and_confusion_are_apart(self):
        ref = turns((A, 0, 10), (B, 10, 20))
        # 2 s of B missed, 3 s of speech in silence after, 4 s of A called B
        hyp = turns(("x", 0, 6), ("y", 6, 18), ("y", 20, 23))
        s = der.score(ref, hyp)
        self.assertAlmostEqual(s.missed, 2.0, places=2)
        self.assertAlmostEqual(s.false_alarm, 3.0, places=2)
        self.assertAlmostEqual(s.confusion, 4.0, places=2)
        self.assertAlmostEqual(s.der, (2 + 3 + 4) / 20, places=3)

    def test_overlap_counts_each_speaker(self):
        ref = turns((A, 0, 10), (B, 5, 10))  # 15 s of speaker time
        hyp = turns(("x", 0, 10))  # finds A, misses the overlapping B
        s = der.score(ref, hyp)
        self.assertAlmostEqual(s.total, 15.0, places=2)
        self.assertAlmostEqual(s.missed, 5.0, places=2)
        s = der.score(ref, hyp, skip_overlap=True)
        self.assertAlmostEqual(s.total, 5.0, places=2)
        self.assertAlmostEqual(s.der, 0.0)

    def test_a_collar_forgives_boundaries(self):
        ref = turns((A, 0, 10), (B, 10, 20))
        hyp = turns(("x", 0, 10.2), ("y", 10.2, 20))
        self.assertGreater(der.score(ref, hyp).der, 0)
        self.assertAlmostEqual(der.score(ref, hyp, collar=0.25).der, 0.0)

    def test_one_label_for_everyone_is_confusion(self):
        ref = turns((A, 0, 10), (B, 10, 20))
        s = der.score(ref, turns(("x", 0, 20)))
        self.assertAlmostEqual(s.confusion, 10.0, places=2)

    def test_speaker_counts(self):
        ref = turns((A, 0, 1), (B, 1, 2), (C, 2, 3))
        hyp = turns(("x", 0, 2), ("y", 2, 3))
        s = der.score(ref, hyp)
        self.assertEqual((s.ref_speakers, s.hyp_speakers), (3, 2))


class SegmentTests(unittest.TestCase):
    """The app puts one speaker on each transcript segment; what fraction is right."""

    def test_each_segment_takes_its_main_speaker(self):
        ref = turns((A, 0, 10), (B, 10, 11), (A, 11, 20))
        hyp = turns(("x", 0, 20))  # the short "네" of B is lost in A
        segs = [(0, 10), (10, 11), (11, 20)]
        r = der.segment_accuracy(ref, hyp, segs, short=1.5)
        self.assertEqual((r.right, r.total), (2, 3))
        self.assertEqual((r.short_right, r.short_total), (0, 1))

    def test_labels_are_matched_before_counting(self):
        ref = turns((A, 0, 5), (B, 5, 10))
        hyp = turns(("q", 0, 5), ("p", 5, 10))
        r = der.segment_accuracy(ref, hyp, [(0, 5), (5, 10)])
        self.assertEqual(r.right, 2)


class RttmTests(unittest.TestCase):
    def test_round_trip(self):
        t = turns((A, 0.5, 2.25), (B, 3, 4))
        text = der.to_rttm(t, "m1")
        self.assertIn("SPEAKER m1 1 0.500 1.750 <NA> <NA> A <NA> <NA>", text)
        self.assertEqual(der.from_rttm(text), t)


if __name__ == "__main__":
    unittest.main()
