"""sherpa-onnx's offline diarization, re-done in numpy around stored embeddings, on small
made-up labels and embeddings (no model)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diar_pipe as dp  # noqa: E402

# a toy geometry: 10-frame chunks of 100 samples, shifted by 50 samples, 10 samples a frame
G = dp.Geometry(window=100, shift=50, frame_shift=10, frame_size=10, rate=100, speakers=3)


def chunk(rows: list[str]) -> np.ndarray:
    """'ab.' -> speakers a and b active, c not; one string a frame."""
    return np.array([[ch != "." for ch in r] for r in rows], dtype=np.int8)


class Powerset(unittest.TestCase):
    def test_the_seven_classes_are_silence_three_singles_and_three_pairs(self):
        m = dp.powerset_mapping(3)
        self.assertEqual(m.tolist(), [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                      [1, 1, 0], [1, 0, 1], [0, 1, 1]])

    def test_each_frame_takes_its_most_likely_class(self):
        logits = np.full((3, 7), -5.0)
        logits[0, 0] = 0   # silence
        logits[1, 2] = 0   # speaker 1
        logits[2, 5] = 0   # speakers 0 and 2
        self.assertEqual(dp.to_multilabel(logits).tolist(), [[0, 0, 0], [0, 1, 0], [1, 0, 1]])


class PerChunk(unittest.TestCase):
    def test_a_speaker_needs_ten_frames_outside_overlap_to_get_an_embedding(self):
        c = chunk(["a.."] * 10 + ["ab."] * 6 + [".b."] * 4)  # b: 4 alone, 6 overlapped
        pairs, spans = dp.chunk_speaker_samples([c], dp.Geometry(200, 100, 10, 10, 100, 3))
        self.assertEqual(pairs, [(0, 0)])
        # a alone in frames 0-9 of 20: samples 0..100 of the 200-sample window
        self.assertEqual(spans, [[(0, 100)]])

    def test_spans_are_offset_by_the_chunk_and_a_run_to_the_end_stops_one_frame_short(self):
        c = chunk(["..."] * 2 + ["a.."] * 18)
        g = dp.Geometry(200, 100, 10, 10, 100, 3)
        _, spans = dp.chunk_speaker_samples([c, c], g)
        self.assertEqual(spans, [[(20, 190)], [(120, 290)]])

    def test_speakers_a_frame_average_over_the_chunks_that_cover_it(self):
        a = chunk(["a.."] * 10)
        b = chunk(["ab."] * 10)
        # chunks start at frames 0 and 5; frames 5-9 are covered by both: (1 + 2) / 2 -> 2
        spf = dp.speakers_per_frame([a, b], G)
        self.assertEqual(spf.tolist()[:5], [1] * 5)
        self.assertEqual(spf.tolist()[5:10], [2] * 5)
        self.assertEqual(spf.tolist()[10:15], [2] * 5)


class Reconstruct(unittest.TestCase):
    def test_local_speakers_of_the_same_cluster_become_one_global_speaker(self):
        a = chunk(["a.."] * 10)
        b = chunk([".a."] * 10)   # the same voice, as the chunk's speaker 1
        turns = dp.reconstruct([a, b], [(0, 0), (1, 1)], np.array([0, 0]), 150, G,
                               min_on=0.0, min_off=0.0)
        self.assertEqual({t.speaker for t in turns}, {"0"})

    def test_each_frame_keeps_as_many_speakers_as_the_chunks_counted_there(self):
        a = chunk(["ab."] * 10)
        b = chunk(["ab."] * 5 + ["a.."] * 5)
        # frames 0-9 hold two speakers, frames 10-14 (chunk 1's last five) one: cluster 0
        turns = dp.reconstruct([a, b], [(0, 0), (0, 1), (1, 0), (1, 1)], np.array([0, 1, 0, 1]), 150, G,
                               min_on=0.0, min_off=0.0)
        spans = {t.speaker: (t.start, t.end) for t in turns}
        self.assertEqual(sorted(spans), ["0", "1"])
        self.assertAlmostEqual(spans["1"][1], 1.05, places=4)  # frame 10 * 0.1 s + half a frame
        self.assertAlmostEqual(spans["0"][1], 1.55, places=4)

    def test_tied_votes_go_to_the_lower_cluster(self):
        a = chunk(["a.."] * 10)
        b = chunk(["a.."] * 10)
        # frames 5-9: one speaker, one vote for cluster 0 and one for cluster 1
        turns = dp.reconstruct([a, b], [(0, 0), (1, 0)], np.array([0, 1]), 150, G, min_on=0.0, min_off=0.0)
        spans = {t.speaker: (t.start, t.end) for t in turns}
        self.assertAlmostEqual(spans["0"][1], 1.05, places=4)
        self.assertAlmostEqual(spans["1"][0], 1.05, places=4)

    def test_a_chunk_votes_once_for_a_cluster_two_of_its_speakers_fell_into(self):
        g = dp.Geometry(window=100, shift=30, frame_shift=10, frame_size=10, rate=100, speakers=3)
        # chunks start at frames 0, 3 and 6; frames 6-9 lie in all three and hold one
        # speaker: chunk 0 put both its speakers in cluster 1, chunk 1 says 0, chunk 2 says 2.
        # One vote each, so the tie goes to cluster 0; counted twice, cluster 1 would win.
        both, one = chunk(["ab."] * 10), chunk(["a.."] * 10)
        turns = dp.reconstruct([both, one, one], [(0, 0), (0, 1), (1, 0), (2, 0)], np.array([1, 1, 0, 2]),
                               160, g, min_on=0.0, min_off=0.0)
        at = {t.speaker for t in turns if t.start <= 0.8 <= t.end}
        self.assertIn("0", at)
        self.assertNotIn("1", at)

    def test_gaps_up_to_min_off_are_closed_and_turns_up_to_min_on_dropped(self):
        segs = [(0.0, 1.0), (1.3, 2.0), (3.0, 3.2)]
        self.assertEqual(dp.merge_and_filter(segs, min_on=0.3, min_off=0.5), [(0.0, 2.0)])


class Clustering(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        x = np.zeros((20, 8), np.float32)
        x[:10, 0] = 1
        x[10:, 1] = 1
        self.emb = x + 0.05 * rng.standard_normal(x.shape).astype(np.float32)

    def test_complete_linkage_at_a_threshold_separates_two_voices(self):
        labels, _ = dp.cluster(self.emb, "complete", 0.5)
        self.assertEqual(len(set(labels[:10])), 1)
        self.assertEqual(len(set(labels[10:])), 1)
        self.assertNotEqual(labels[0], labels[10])

    def test_a_merge_exactly_at_the_threshold_is_not_made(self):
        x = np.array([[1, 0], [0, 1]], np.float32)  # cosine distance exactly 1
        self.assertEqual(len(set(dp.cluster(x, "complete", 1.0)[0])), 2)
        self.assertEqual(len(set(dp.cluster(x, "complete", 1.0001)[0])), 1)

    def test_centroid_linkage_reads_the_threshold_on_the_cosine_scale(self):
        x = np.array([[1, 0], [0, 1]], np.float32)  # cosine 1, euclidean sqrt 2
        self.assertEqual(len(set(dp.cluster(x, "centroid", 0.99)[0])), 2)
        self.assertEqual(len(set(dp.cluster(x, "centroid", 1.01)[0])), 1)

    def test_inversions_are_counted(self):
        # an equilateral triangle: the centroid of two corners is nearer the third
        x = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], np.float64)
        self.assertEqual(dp.inversions(dp.linkage_matrix(x, "centroid", unit=False)), 1)
        self.assertEqual(dp.inversions(dp.linkage_matrix(self.emb, "complete")), 0)

    def test_clusters_are_numbered_in_order_of_first_appearance_as_sherpa_does(self):
        # tied votes go to the lower number, so the numbering changes the result
        labels, _ = dp.cluster(self.emb, "complete", 0.5)   # scipy alone numbers voice 1 first
        self.assertEqual(labels[[0, 10]].tolist(), [0, 1])
        self.assertEqual(dp.first_seen(np.array([7, 3, 7, 5])).tolist(), [0, 1, 0, 2])

    def test_folding_keeps_the_first_appearance_order(self):
        labels = np.array([2, 0, 0, 0, 1, 1, 1])   # cluster 2 (first) folds into 1
        emb = np.vstack([self.emb[12:13], self.emb[:3], self.emb[10:13]])
        self.assertEqual(dp.reassign_small(emb, labels, min_count=2).tolist(), [0, 1, 1, 1, 0, 0, 0])

    def test_small_clusters_go_to_the_nearest_big_one(self):
        labels = np.array([0] * 10 + [1] * 10 + [2])
        emb = np.vstack([self.emb, self.emb[12:13] * 0.9])  # the stray sounds like voice 1
        out = dp.reassign_small(emb, labels, min_count=2)
        self.assertEqual(out[-1], out[10])
        self.assertEqual(len(set(out)), 2)

    def test_min_share_counts_against_all_embeddings(self):
        labels = np.array([0] * 10 + [1] * 10 + [2])
        emb = np.vstack([self.emb, self.emb[3:4]])
        self.assertEqual(dp.small_floor(labels, share=0.05), 2)  # 5% of 21 -> 1.05 -> 2
        out = dp.reassign_small(emb, labels, min_count=dp.small_floor(labels, share=0.05))
        self.assertEqual(out[-1], out[0])


class ErasedPeople(unittest.TestCase):
    def test_each_embedding_is_matched_to_the_reference_speech_under_its_spans(self):
        from der import Turn
        g = dp.Geometry(200, 100, 10, 10, 100, 3)
        c = chunk(["..."] * 2 + ["a.."] * 18)          # speaker a over samples 20..190
        st = dp.Store(np.stack([c]), np.array([[0, 0]]), np.zeros((1, 2), np.float32), 200)
        ref = [Turn("x", 0.0, 1.0), Turn("y", 1.0, 3.0)]
        names, m = dp.ref_overlap(st, ref, g)
        self.assertEqual(names, ["x", "y"])
        np.testing.assert_allclose(m, [[0.8, 0.9]], atol=1e-6)

    def test_a_folded_cluster_one_person_fills_counts_as_a_person_erased(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2])
        lone = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0.1, 0.9]])
        mixed = lone.copy()
        mixed[-1] = [0.5, 0.5]
        self.assertEqual(dp.erased(lone, labels, min_count=2), 1)
        self.assertEqual(dp.erased(mixed, labels, min_count=2), 0)   # a tail, not a person
        self.assertEqual(dp.erased(lone, labels, min_count=1), 0)    # nothing folded

    def test_nothing_is_erased_when_no_cluster_is_big_enough_to_fold_into(self):
        self.assertEqual(dp.erased(np.eye(2), np.array([0, 1]), min_count=5), 0)

    def test_a_speaker_who_leads_no_final_cluster_has_vanished(self):
        overlap = np.array([[5, 1, 0], [4, 0, 0], [0, 3, 2], [0, 3, 2]], float)
        self.assertEqual(dp.vanished(overlap, np.array([0, 0, 1, 1])), 1)    # speaker 2 leads nowhere
        self.assertEqual(dp.vanished(overlap, np.array([0, 0, 1, 2])), 1)    # both led by speaker 1
        self.assertEqual(dp.vanished(overlap[:, :2], np.array([0, 0, 1, 1])), 0)


if __name__ == "__main__":
    unittest.main()
