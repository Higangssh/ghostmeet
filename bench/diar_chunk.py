"""Diarization in pieces, for meetings too long to cluster at once (requirements §12 risk 1).

The whole-meeting clustering in `diar_pipe` holds an N x N distance matrix: 35,900
embeddings and 5.16 GB for four hours. Here the embeddings are cut by time into pieces
(the middle of each embedding's segmentation chunk decides its piece), each piece is
clustered on its own, and the pieces' speakers are then linked across the meeting:

1. In each piece: the same linkage and threshold as before. Clusters under
   `piece_share` of the piece's embeddings (and under `piece_min`) join the nearest
   bigger one; if none is big enough, the largest stays. What is left are the piece's
   candidate speakers.
2. Across pieces: candidates are linked by average linkage on their centroids (cosine)
   at `link_threshold`. With an overlap, a piece also clusters the first `overlap_sec`
   of the next one; a candidate whose embeddings there mostly fall into one candidate
   of the next piece is the same person (must-link). With `cannot_link`, two candidates
   of one piece are never joined.
3. A global floor on each linked speaker's total speech (`floor_kind` "share" of the
   meeting, or "sec" seconds): smaller ones join the nearest bigger one, and again the
   largest stays if none is big enough.

A floor per piece rescues someone who joins late, whose share of the whole meeting is
small but whose share of their piece is not. A floor on the linked total rescues someone
who says a little in every piece: that is why the piece floor can be 0.

Labels are numbered in order of first appearance and ties go to the lower number, as in
`diar_pipe`; turns come out of `diar_pipe.reconstruct` unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import diar_pipe as dp

# distance between two candidates that must not be joined; average linkage averages it in,
# so it has to outweigh any number of near pairs
APART = 1e6


@dataclass(frozen=True)
class Config:
    piece_sec: float
    overlap_sec: float
    method: str              # complete or average, inside a piece
    threshold: float
    piece_share: float
    link_threshold: float
    cannot_link: bool
    floor_kind: str          # "share" of the meeting's speech, or "sec"
    floor: float
    piece_min: int = 1


def _middles(store: dp.Store, g: dp.Geometry) -> np.ndarray:
    return (store.pairs[:, 0] * g.shift + g.window / 2) / g.rate


def owners(store: dp.Store, g: dp.Geometry, piece_sec: float) -> np.ndarray:
    return (_middles(store, g) // piece_sec).astype(np.int64)


def pieces(store: dp.Store, g: dp.Geometry, piece_sec: float, overlap_sec: float) -> list[np.ndarray]:
    """Embedding indices of each piece: its own, and the next piece's first `overlap_sec`."""
    mid = _middles(store, g)
    n = int(mid.max() // piece_sec) + 1 if len(mid) else 0
    return [np.flatnonzero((mid >= k * piece_sec) & (mid < (k + 1) * piece_sec + overlap_sec)) for k in range(n)]


def speech_sec(store: dp.Store, g: dp.Geometry) -> np.ndarray:
    """Seconds of speech each embedding stands for: its speaker's frames alone in its chunk,
    divided by how many chunks cover a moment."""
    lab = store.labels[store.pairs[:, 0]]                        # (n, frames, speakers)
    alone = lab * (lab.sum(axis=2, keepdims=True) < 2)
    frames = alone[np.arange(len(lab)), :, store.pairs[:, 1]].sum(axis=1)
    return frames * (g.frame_shift / g.rate) * (g.shift / g.window)


def keep_big(x: np.ndarray, labels: np.ndarray, weight: np.ndarray, floor: float) -> np.ndarray:
    """Clusters whose summed `weight` is under `floor` join the big one with the nearest
    centroid (cosine); with no big one the largest stays and takes the rest. First-seen order."""
    ids = np.unique(labels)
    size = np.array([weight[labels == c].sum() for c in ids])
    big = ids[size >= floor]
    if len(big) == 0:
        big = ids[[int(np.argmax(size))]]  # argmax: the lowest number among equals
    small = [c for c in ids if c not in set(big.tolist())]
    out = labels.copy()
    if small:
        u = dp._unit(x)
        cent = lambda c: dp._unit(u[labels == c].mean(axis=0, keepdims=True))[0]  # noqa: E731
        bc = np.stack([cent(c) for c in big])
        for s in small:
            out[labels == s] = big[int(np.argmax(bc @ cent(s)))]
    return dp.first_seen(out)


def _local(x: np.ndarray, cfg: Config) -> np.ndarray:
    if len(x) == 1:
        return np.zeros(1, np.int64)
    labels = dp.cut(dp.linkage_matrix(x, cfg.method), cfg.method, cfg.threshold)
    floor = max(cfg.piece_min, math.ceil(cfg.piece_share * len(x)))
    return keep_big(x, labels, np.ones(len(x)), floor) if floor > 1 else labels


class _Union:
    def __init__(self, n: int):
        self.up = list(range(n))

    def find(self, a: int) -> int:
        while self.up[a] != a:
            self.up[a] = self.up[self.up[a]]
            a = self.up[a]
        return a

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.up[max(a, b)] = min(a, b)


def diarize_labels(store: dp.Store, g: dp.Geometry, cfg: Config, cache: dict | None = None) -> np.ndarray:
    """A speaker label for every embedding. `cache` keeps the pieces' own clustering between
    calls on the same store that differ only in the link and the global floor."""
    x = store.embeddings
    key = (cfg.piece_sec, cfg.overlap_sec, cfg.method, cfg.threshold, cfg.piece_share, cfg.piece_min)
    if cache is not None and key in cache:
        members, local = cache[key]
    else:
        members = pieces(store, g, cfg.piece_sec, cfg.overlap_sec)
        local = [_local(x[m], cfg) if len(m) else np.zeros(0, np.int64) for m in members]
        if cache is not None:
            cache[key] = (members, local)
    own = owners(store, g, cfg.piece_sec)

    # candidates: (piece, local label), numbered in piece order
    cand, piece_of = {}, []
    for k, lab in enumerate(local):
        for c in np.unique(lab):
            cand[(k, int(c))] = len(piece_of)
            piece_of.append(k)
    n = len(piece_of)
    groups = _Union(n)
    if cfg.overlap_sec > 0:
        for k in range(len(members) - 1):
            here = dict(zip(members[k].tolist(), local[k].tolist()))
            there = dict(zip(members[k + 1].tolist(), local[k + 1].tolist()))
            shared = [i for i in members[k + 1].tolist() if i in here]
            for c in sorted({here[i] for i in shared}):
                partners = [there[i] for i in shared if here[i] == c]
                vals, counts = np.unique(partners, return_counts=True)
                if counts.max() * 2 > len(partners):
                    groups.join(cand[(k, c)], cand[(k + 1, int(vals[np.argmax(counts)]))])

    # each embedding's candidate is the one of its own piece
    of = np.empty(len(x), np.int64)
    for k, m in enumerate(members):
        for i, c in zip(m.tolist(), local[k].tolist()):
            if own[i] == k:
                of[i] = cand[(k, c)]
    root = np.array([groups.find(c) for c in range(n)])
    nodes = sorted(set(root.tolist()))
    node_of = {r: j for j, r in enumerate(nodes)}
    node = np.array([node_of[r] for r in root])[of]          # per embedding

    u = dp._unit(x)
    cent = np.stack([dp._unit(u[node == j].mean(axis=0, keepdims=True))[0] if (node == j).any()
                     else np.zeros(x.shape[1], np.float32) for j in range(len(nodes))])
    if len(nodes) > 1:
        from scipy.cluster.hierarchy import linkage

        d = 1.0 - (cent @ cent.T).astype(np.float64)
        np.maximum(d, 0.0, out=d)
        if cfg.cannot_link:
            node_pieces = [set() for _ in nodes]
            for c in range(n):
                node_pieces[node_of[root[c]]].add(piece_of[c])
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    if node_pieces[a] & node_pieces[b]:
                        d[a, b] = d[b, a] = APART
        iu = np.triu_indices(len(nodes), 1)
        z = linkage(d[iu], method="average")
        linked = dp.cut(z, "average", cfg.link_threshold)
    else:
        linked = np.zeros(len(nodes), np.int64)
    labels = dp.first_seen(linked[node])

    sec = speech_sec(store, g)
    floor = cfg.floor * sec.sum() if cfg.floor_kind == "share" else cfg.floor
    return keep_big(x, labels, sec, floor) if floor > 0 else labels


def diarize(store: dp.Store, g: dp.Geometry, cfg: Config, min_on: float = 0.3, min_off: float = 0.5):
    labels = diarize_labels(store, g, cfg)
    pairs = [tuple(p) for p in store.pairs.tolist()]
    return dp.reconstruct(list(store.labels), pairs, labels, store.n_samples, g, min_on, min_off)


def one_voice(labels: np.ndarray, n_samples: int, g: dp.Geometry, name: str,
              min_on: float = 0.3, min_off: float = 0.5):
    """Turns of one speaker wherever the segmentation hears anyone: the microphone of a call,
    which the app takes to be "me" alone (wrong in a hybrid meeting, where others sit with me).
    The same reconstruction, every local speaker in one cluster."""
    labels = list(labels)
    pairs = [(c, s) for c in range(len(labels)) for s in range(g.speakers)]
    turns = dp.reconstruct(labels, pairs, np.zeros(len(pairs), np.int64), n_samples, g, min_on, min_off)
    return [dp.Turn(name, t.start, t.end) for t in turns]
