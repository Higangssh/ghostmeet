"""Diarization in pieces with a global link, on made-up stores (no model): one embedding a
segmentation chunk, a voice a direction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diar_chunk as dc  # noqa: E402
import diar_pipe as dp  # noqa: E402

# chunks of 1 s every 0.5 s, 10 frames each: chunk c covers c*0.5 .. c*0.5 + 1 s
G = dp.Geometry(window=100, shift=50, frame_shift=10, frame_size=10, rate=100, speakers=3)


def store(voices: list[int], dim: int = 8, noise: float = 0.05, seed: int = 0, drift=None) -> dp.Store:
    """One chunk per entry, its local speaker 0 talking throughout with voice `voices[c]`.
    `drift(c)` adds to the embedding (a voice that changes over the meeting)."""
    rng = np.random.default_rng(seed)
    n = len(voices)
    labels = np.zeros((n, 10, 3), np.int8)
    labels[:, :, 0] = 1
    emb = np.zeros((n, dim), np.float32)
    for c, v in enumerate(voices):
        emb[c, v] = 1
        if drift is not None:
            emb[c] += drift(c)
    emb += noise * rng.standard_normal(emb.shape).astype(np.float32)
    pairs = np.array([[c, 0] for c in range(n)], np.int32)
    return dp.Store(labels, pairs, emb, n_samples=(n - 1) * G.shift + G.window)


def cfg(**kw) -> dc.Config:
    base = dict(piece_sec=10.0, overlap_sec=0.0, method="complete", threshold=0.5, piece_share=0.0,
                link_threshold=0.5, cannot_link=True, floor_kind="share", floor=0.0)
    base.update(kw)
    return dc.Config(**base)


class Pieces(unittest.TestCase):
    def test_each_embedding_has_one_owner_piece_by_the_middle_of_its_chunk(self):
        st = store([0] * 40)  # 40 chunks: middles at 0.5, 1.0, ... 20.0 s
        own = dc.owners(st, G, piece_sec=10.0)
        mid = np.arange(40) * 0.5 + 0.5
        self.assertEqual(own.tolist(), (mid // 10).astype(int).tolist())

    def test_an_overlap_puts_the_start_of_the_next_piece_in_this_one_too(self):
        st = store([0] * 40)
        members = dc.pieces(st, G, piece_sec=10.0, overlap_sec=2.0)
        mid = np.arange(40) * 0.5 + 0.5
        self.assertEqual(members[0].tolist(), np.flatnonzero(mid < 12.0).tolist())
        self.assertEqual(members[1].tolist(), np.flatnonzero(mid >= 10.0).tolist())


class Floors(unittest.TestCase):
    def test_small_clusters_join_the_nearest_big_one(self):
        x = np.eye(3, dtype=np.float32)[[0, 0, 0, 1, 1, 1, 2]]
        x[6] = [0.1, 0.9, 0.2]
        out = dc.keep_big(x, np.array([0, 0, 0, 1, 1, 1, 2]), np.array([1, 1, 1, 1, 1, 1, 1.0]), floor=2)
        self.assertEqual(out.tolist(), [0, 0, 0, 1, 1, 1, 1])

    def test_when_nothing_is_big_enough_the_largest_stays_and_takes_the_rest(self):
        x = np.eye(3, dtype=np.float32)[[0, 0, 1, 2]]
        out = dc.keep_big(x, np.array([0, 0, 1, 2]), np.ones(4), floor=10)
        self.assertEqual(out.tolist(), [0, 0, 0, 0])

    def test_size_is_counted_in_the_weights_given(self):
        x = np.eye(3, dtype=np.float32)[[2, 2, 1, 2]]
        labels = np.array([0, 0, 1, 2])
        # clusters 1 and 2 are one embedding each but carry 3 units of speech: cluster 0
        # (two embeddings, two units) is the small one and joins cluster 2, its own voice
        out = dc.keep_big(x, labels, np.array([1, 1, 3.0, 3.0]), floor=2.5)
        self.assertEqual(out.tolist(), [0, 0, 1, 0])
        # counted in embeddings, nothing reaches 2.5 and the largest, cluster 0, takes all
        self.assertEqual(dc.keep_big(x, labels, np.ones(4), floor=2.5).tolist(), [0, 0, 0, 0])


class Chunked(unittest.TestCase):
    def test_a_late_joiner_a_whole_meeting_floor_erases_survives_a_floor_per_piece(self):
        # voice 1 joins near the end and talks 15% of two pieces, 3% of the meeting
        voices = [0] * 170 + [0, 0, 1] * 6 + [0] * 12
        st = store(voices)
        whole = dp.reassign_small(st.embeddings, dp.cluster(st.embeddings, "complete", 0.5)[0],
                                  dp.small_floor(np.zeros(len(voices)), 0.08))
        self.assertEqual(len(set(whole.tolist())), 1)
        labels = dc.diarize_labels(st, G, cfg(piece_share=0.08))
        self.assertEqual(len(set(labels.tolist())), 2)
        self.assertEqual(len(set(labels[np.array(voices) == 1].tolist())), 1)

    def test_a_quiet_speaker_in_every_piece_survives_on_their_total_not_their_share_of_a_piece(self):
        voices = ([1] + [0] * 19) * 10  # voice 1 is one embedding, about 5%, of each piece
        voices[55] = 2                   # and voice 2 one stray embedding, 0.5 s in all
        st = store(voices)
        lost = dc.diarize_labels(st, G, cfg(piece_share=0.08))
        self.assertEqual(len(set(lost.tolist())), 1)
        kept = dc.diarize_labels(st, G, cfg(piece_share=0.0, floor_kind="sec", floor=2.0))
        self.assertEqual(len(set(kept.tolist())), 2)   # the stray is under the 2 s floor
        self.assertEqual(len(set(kept[np.array(voices) == 1].tolist())), 1)
        unfloored = dc.diarize_labels(st, G, cfg(piece_share=0.0, floor=0.0))
        self.assertEqual(len(set(unfloored.tolist())), 3)

    def test_the_same_voice_in_every_piece_is_one_speaker(self):
        voices = [0, 1] * 100
        labels = dc.diarize_labels(store(voices), G, cfg())
        self.assertEqual(labels[np.array(voices) == 0].tolist(), [0] * 100)
        self.assertEqual(labels[np.array(voices) == 1].tolist(), [1] * 100)

    def test_two_speakers_of_one_piece_are_never_linked(self):
        # voices 0 and 1 almost opposite in direction 0..1, but a high link threshold would
        # still join them; they share every piece, so the link must keep them apart
        st = store([0, 1] * 40)
        self.assertEqual(len(set(dc.diarize_labels(st, G, cfg(link_threshold=1.9)).tolist())), 2)
        self.assertEqual(len(set(dc.diarize_labels(st, G, cfg(link_threshold=1.9, cannot_link=False)).tolist())), 1)

    def test_an_overlap_links_a_voice_that_drifts_across_a_piece_boundary(self):
        # voice 0 turns steadily towards voice 2 over the meeting (1.5 rad): a piece holds
        # together, neighbouring pieces' centroids are 0.07 apart, past the link threshold,
        # but each piece overlaps the next and the shared embeddings say it is the same person
        n = 80

        def drift(c):
            a = 1.5 * c / n
            return np.array([np.cos(a) - 1, 0, np.sin(a), 0, 0, 0, 0, 0], np.float32)

        st = store([0] * n, drift=drift)
        apart = dc.diarize_labels(st, G, cfg(threshold=0.3, link_threshold=0.05))
        self.assertGreater(len(set(apart.tolist())), 1)
        linked = dc.diarize_labels(st, G, cfg(threshold=0.3, link_threshold=0.05, overlap_sec=4.0))
        self.assertEqual(len(set(linked.tolist())), 1)

    def test_labels_are_numbered_by_first_appearance_and_the_same_input_gives_the_same_output(self):
        voices = [2, 2, 0, 0, 1, 1] * 30
        st = store(voices)
        a = dc.diarize_labels(st, G, cfg())
        b = dc.diarize_labels(store(voices), G, cfg())
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(a[:6].tolist(), [0, 0, 1, 1, 2, 2])

    def test_a_cache_of_the_pieces_gives_the_same_labels(self):
        st = store([2, 2, 0, 0, 1, 1] * 30)
        cache = {}
        first = dc.diarize_labels(st, G, cfg(), cache)
        self.assertEqual(len(cache), 1)
        again = dc.diarize_labels(st, G, cfg(link_threshold=0.4), cache)
        self.assertEqual(len(cache), 1)
        self.assertEqual(again.tolist(), dc.diarize_labels(st, G, cfg(link_threshold=0.4)).tolist())
        self.assertEqual(first.tolist(), dc.diarize_labels(st, G, cfg()).tolist())

    def test_turns_come_out_through_the_same_reconstruction(self):
        st = store([0] * 20 + [1] * 20)
        turns = dc.diarize(st, G, cfg(piece_sec=5.0), min_on=0.0, min_off=0.0)
        self.assertEqual(sorted({t.speaker for t in turns}), ["0", "1"])


class OneVoice(unittest.TestCase):
    def test_wherever_any_local_speaker_talks_on_the_microphone_it_is_me(self):
        # four chunks (0 .. 2.5 s); chunk 1 has speakers 0 and 1, chunk 3 nobody
        labels = np.zeros((4, 10, 3), np.int8)
        labels[0, :, 0] = 1
        labels[1, :, 0] = labels[1, :, 1] = 1
        labels[2, :5, 2] = 1
        turns = dc.one_voice(labels, n_samples=250, g=G, name="me", min_on=0.0, min_off=0.0)
        self.assertEqual({t.speaker for t in turns}, {"me"})
        self.assertEqual(len(turns), 1)
        self.assertAlmostEqual(turns[0].start, 0.05, places=3)
        self.assertLess(turns[0].end, 2.0)   # chunk 3 is silent: frames from 1.5 s on are mostly not speech
        self.assertGreater(turns[0].end, 1.0)

    def test_a_silent_microphone_gives_no_turns(self):
        self.assertEqual(dc.one_voice(np.zeros((3, 10, 3), np.int8), 200, G, "me"), [])


if __name__ == "__main__":
    unittest.main()
