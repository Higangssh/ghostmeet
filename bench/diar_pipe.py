"""sherpa-onnx's offline speaker diarization (v1.13.8, offline-speaker-diarization-pyannote-impl.h)
re-done in numpy, so the embeddings can be computed once and clustered many ways.

    extract(audio, segmenter, extractor) -> Store       # the slow part, once a meeting
    diarize(store, linkage, threshold, ...) -> turns    # milliseconds to seconds

The steps between the models follow sherpa line for line, float32 where it is float32:
powerset -> multi-label per 10 s chunk (1 s shift); the speakers a frame holds, averaged
over the chunks covering it; one embedding per (chunk, local speaker) with at least 10
frames outside overlap; clustering; each frame's top-k clusters by how many chunks put
them there; turns merged across gaps up to `min_off` and kept when longer than `min_on`.

sherpa clusters with complete linkage on cosine distance and applies the merges below
the threshold. Here the linkage is a choice: complete and average on that same cosine
matrix, centroid on the unit embeddings themselves (it needs Euclidean distances), its
threshold read on the cosine scale (d_euclid = sqrt(2 d_cos)). Small clusters can be
reassigned to the nearest big one by centroid, as pyannote's AgglomerativeClustering does
with min_cluster_size.

Ties in the top-k go to the lower cluster index; sherpa's std::partial_sort leaves their
order to the library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from der import Turn


@dataclass(frozen=True)
class Geometry:
    window: int        # samples a chunk
    shift: int         # samples between chunks
    frame_shift: int   # samples a frame (receptive field shift)
    frame_size: int    # receptive field size, in samples
    rate: int
    speakers: int      # local speakers a chunk


# pyannote segmentation-3.0 as sherpa runs it (window_shift_ratio 0.1)
SEGMENTATION_3 = Geometry(window=160000, shift=16000, frame_shift=270, frame_size=991, rate=16000, speakers=3)


def powerset_mapping(speakers: int) -> np.ndarray:
    rows = [[0] * speakers]
    rows += [[int(j == i) for j in range(speakers)] for i in range(speakers)]
    rows += [[int(j in (a, b)) for j in range(speakers)] for a in range(speakers) for b in range(a + 1, speakers)]
    return np.array(rows, dtype=np.int8)


def to_multilabel(logits: np.ndarray) -> np.ndarray:
    m = powerset_mapping(3)
    return m[np.argmax(logits, axis=1)]


def _starts(n_chunks: int, g: Geometry) -> np.ndarray:
    # int32(float(i) * shift / frame_shift + 0.5), in float32 as sherpa does
    i = np.arange(n_chunks, dtype=np.float32)
    return (i * np.float32(g.shift) / np.float32(g.frame_shift) + np.float32(0.5)).astype(np.int64)


def _n_frames(n_chunks: int, g: Geometry) -> int:
    return (g.window + (n_chunks - 1) * g.shift) // g.frame_shift + 1


def speakers_per_frame(labels: list[np.ndarray], g: Geometry) -> np.ndarray:
    n = _n_frames(len(labels), g)
    count = np.zeros(n, np.float32)
    weight = np.zeros(n, np.float32)
    for s, lab in zip(_starts(len(labels), g), labels):
        count[s:s + len(lab)] += lab.sum(axis=1).astype(np.float32)
        weight[s:s + len(lab)] += 1
    return (count / (weight + np.float32(1e-12)) + np.float32(0.5)).astype(np.int32)


def _sample(frame: int, n_frames: int, offset: int, g: Geometry) -> int:
    return int(np.float32(np.float32(frame) / np.float32(n_frames) * np.float32(g.window)) + np.float32(offset))


def chunk_speaker_samples(labels: list[np.ndarray], g: Geometry):
    """(chunk, local speaker) pairs worth an embedding, and each one's sample spans."""
    pairs, spans = [], []
    for c, lab in enumerate(labels):
        alone = lab * (lab.sum(axis=1, keepdims=True) < 2)
        n_frames = len(lab)
        offset = c * g.shift
        for spk in range(g.speakers):
            d = alone[:, spk]
            if d.sum() < 10:
                continue
            runs, start = [], None
            for k in range(n_frames):
                if d[k]:
                    if start is None:
                        start = k
                elif start is not None:
                    runs.append((_sample(start, n_frames, offset, g), _sample(k, n_frames, offset, g)))
                    start = None
            if start is not None:
                runs.append((_sample(start, n_frames, offset, g), _sample(n_frames - 1, n_frames, offset, g)))
            pairs.append((c, spk))
            spans.append(runs)
    return pairs, spans


def merge_and_filter(segs: list[tuple[float, float]], min_on: float, min_off: float):
    out: list[list[float]] = []
    for a, b in segs:
        if out and out[-1][1] < a and out[-1][1] + min_off >= a:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a > min_on]


def reconstruct(labels: list[np.ndarray], pairs, clusters: np.ndarray, n_samples: int, g: Geometry,
                min_on: float = 0.3, min_off: float = 0.5, block: int = 4096) -> list[Turn]:
    """sherpa's reconstruction: each frame keeps its `speakers_per_frame` clusters with the
    most chunk votes, ties to the lower cluster. Done in blocks of frames over the clusters
    voted for there, plus the lowest-numbered ones a frame may take with no vote, so a
    clustering with thousands of clusters never builds a frames x clusters count."""
    k = int(clusters.max()) + 1
    to_cluster = {p: int(c) for p, c in zip(pairs, clusters)}
    n = _n_frames(len(labels), g)
    if (n_samples - g.window) % g.shift > 0:
        n = min(n_samples // g.frame_shift, n - 1) + 1
    starts = _starts(len(labels), g)
    ends = starts + np.array([len(lab) for lab in labels], np.int64)
    spf = speakers_per_frame(labels, g)[:n]
    local = [[to_cluster.get((c, spk)) for spk in range(lab.shape[1])] for c, lab in enumerate(labels)]
    frames: dict[int, list[np.ndarray]] = {}
    for f0 in range(0, n, block):
        f1 = min(f0 + block, n)
        chunks = range(int(np.searchsorted(ends, f0, side="right")), int(np.searchsorted(starts, f1)))
        voted = sorted({c for ch in chunks for c in local[ch] if c is not None})
        # a frame takes at most spf clusters; with v voted ones the lowest spf unvoted lie below v + spf
        cols = sorted(set(voted) | set(range(min(k, len(voted) + int(spf[f0:f1].max(initial=0))))))
        at = {c: i for i, c in enumerate(cols)}
        count = np.zeros((f1 - f0, len(cols)), np.int32)
        for ch in chunks:
            lab, s0 = labels[ch], int(starts[ch])
            # a chunk votes once for a cluster in a frame, however many of its local speakers
            # the cluster took (sherpa's ReLabel sets, it does not add)
            mine = np.zeros((len(lab), len(cols)), np.int32)
            for spk, c in enumerate(local[ch]):
                if c is not None:
                    mine[:, at[c]] |= lab[:, spk]
            a, b = max(f0, s0), min(f1, s0 + len(lab))
            count[a - f0:b - f0] += mine[a - s0:b - s0]
        order = np.argsort(-count, axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(len(cols))[None, :].repeat(len(count), 0), axis=1)
        rows, picked = np.nonzero(rank < spf[f0:f1, None])
        for i in np.unique(picked):
            frames.setdefault(cols[i], []).append(rows[picked == i] + f0)
    scale = np.float32(g.frame_shift) / np.float32(g.rate)
    off = np.float32(0.5) * np.float32(g.frame_size) / np.float32(g.rate)
    turns = []
    for spk in sorted(frames):
        f = np.concatenate(frames[spk])
        f.sort()
        cut = np.flatnonzero(np.diff(f) > 1)
        run_starts = f[np.concatenate([[0], cut + 1])]
        run_ends = np.minimum(f[np.concatenate([cut, [len(f) - 1]])] + 1, n - 1)  # a run to the last frame ends on it
        f32 = np.float32
        segs = [(float(f32(a) * scale + off), float(f32(b) * scale + off)) for a, b in zip(run_starts, run_ends)]
        for a, b in merge_and_filter(segs, min_on, min_off):
            turns.append(Turn(str(spk), round(a, 3), round(b, 3)))
    return sorted(turns, key=lambda t: (t.start, t.speaker))


# --- clustering ---------------------------------------------------------------------

def _unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine_condensed(x: np.ndarray) -> np.ndarray:
    u = _unit(x)
    sim = (u @ u.T).astype(np.float64)
    iu = np.triu_indices(len(u), 1)
    return np.maximum(1.0 - sim[iu], 0.0)


def linkage_matrix(x: np.ndarray, method: str, unit: bool = True) -> np.ndarray:
    from scipy.cluster.hierarchy import linkage

    if method in ("complete", "average"):
        return linkage(cosine_condensed(x), method=method)
    if method == "centroid":
        obs = _unit(x).astype(np.float64) if unit else np.asarray(x, np.float64)
        return linkage(obs, method="centroid", metric="euclidean")
    raise ValueError(method)


def inversions(z: np.ndarray) -> int:
    """Merges lower than the one before them (centroid linkage can do that)."""
    return int((np.diff(z[:, 2]) < 0).sum())


def cut(z: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """Merges strictly below `threshold` (hclust-cpp's cutree_cdist), on the cosine scale."""
    from scipy.cluster.hierarchy import fcluster

    t = math.sqrt(2 * threshold) if method == "centroid" else threshold
    return first_seen(fcluster(z, np.nextafter(t, 0), criterion="distance"))


def first_seen(labels: np.ndarray) -> np.ndarray:
    """Clusters renumbered 0..k-1 in order of their first embedding, as sherpa numbers them.
    Tied votes in `reconstruct` go to the lower number, so this is not cosmetic: scipy's
    own numbering moved DER by 2 points on a 30-minute dev meeting."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse]


def cluster(x: np.ndarray, method: str, threshold: float) -> tuple[np.ndarray, int]:
    if len(x) == 1:
        return np.zeros(1, np.int64), 0
    z = linkage_matrix(x, method)
    return cut(z, method, threshold), inversions(z)


def small_floor(labels: np.ndarray, share: float) -> int:
    return math.ceil(share * len(labels))


def reassign_small(x: np.ndarray, labels: np.ndarray, min_count: int) -> np.ndarray:
    """Clusters with fewer than `min_count` embeddings join the big one whose centroid is
    nearest (cosine), then labels are renumbered in order of first appearance."""
    ids, counts = np.unique(labels, return_counts=True)
    big, small = ids[counts >= min_count], ids[counts < min_count]
    out = labels.copy()
    if len(big) and len(small):
        u = _unit(x)
        cent = lambda c: _unit(u[labels == c].mean(axis=0, keepdims=True))[0]  # noqa: E731
        bc = np.stack([cent(c) for c in big])
        for s in small:
            out[labels == s] = big[int(np.argmax(bc @ cent(s)))]
    return first_seen(out)


def ref_overlap(store: "Store", ref: list[Turn], g: Geometry = SEGMENTATION_3):
    """Seconds of each reference speaker's speech under each embedding's spans: (names, (n, s))."""
    names = sorted({t.speaker for t in ref})
    turns = {n: np.array([(t.start, t.end) for t in ref if t.speaker == n]) for n in names}
    pairs, spans = chunk_speaker_samples(list(store.labels), g)
    runs = dict(zip(pairs, spans))
    m = np.zeros((len(store.pairs), len(names)))
    for i, p in enumerate(store.pairs.tolist()):
        for a, b in runs[tuple(p)]:
            a, b = a / g.rate, b / g.rate
            for j, n in enumerate(names):
                t = turns[n]
                m[i, j] += np.clip(np.minimum(t[:, 1], b) - np.maximum(t[:, 0], a), 0, None).sum()
    return names, m


def erased(overlap: np.ndarray, labels: np.ndarray, min_count: int, purity: float = 0.8) -> int:
    """Clusters `reassign_small` folds away in which one reference speaker holds more than
    `purity` of the speech: a person erased, not a tail."""
    ids, counts = np.unique(labels, return_counts=True)
    if not (counts >= min_count).any():
        return 0
    n = 0
    for c in ids[counts < min_count]:
        own = overlap[labels == c].sum(axis=0)
        if own.sum() > 0 and own.max() / own.sum() > purity:
            n += 1
    return n


def vanished(overlap: np.ndarray, labels: np.ndarray) -> int:
    """Reference speakers who hold the most speech in none of the final clusters."""
    leaders = {int(np.argmax(overlap[labels == c].sum(axis=0))) for c in np.unique(labels)}
    return overlap.shape[1] - len(leaders)


# --- the models ---------------------------------------------------------------------

@dataclass
class Store:
    labels: np.ndarray          # (chunks, frames, 3) int8
    pairs: np.ndarray           # (n, 2) chunk, local speaker
    embeddings: np.ndarray      # (n, dim) float32
    n_samples: int

    def save(self, path) -> None:
        np.savez_compressed(path, labels=self.labels, pairs=self.pairs, embeddings=self.embeddings,
                            n_samples=self.n_samples)

    @classmethod
    def load(cls, path) -> "Store":
        d = np.load(path)
        return cls(d["labels"], d["pairs"], d["embeddings"], int(d["n_samples"]))


def segment(audio: np.ndarray, session, g: Geometry = SEGMENTATION_3) -> list[np.ndarray]:
    name = session.get_inputs()[0].name
    run = lambda x: session.run(None, {name: x[None, None, :]})[0][0]  # noqa: E731
    n = len(audio)
    if n <= g.window:
        raise ValueError("a meeting shorter than one chunk")
    n_chunks = (n - g.window) // g.shift + 1
    out = [to_multilabel(run(np.ascontiguousarray(audio[i * g.shift:i * g.shift + g.window], dtype=np.float32)))
           for i in range(n_chunks)]
    if (n - g.window) % g.shift > 0:
        buf = np.zeros(g.window, np.float32)
        rest = audio[n_chunks * g.shift:n]
        buf[:len(rest)] = rest
        out.append(to_multilabel(run(buf)))
    return out


def extract(audio, session, extractor, g: Geometry = SEGMENTATION_3) -> Store:
    """`audio`: an array, or anything with len() and slicing that reads it on demand (a file
    on disk), so a long meeting is never in memory at once."""
    if isinstance(audio, np.ndarray):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
    labels = segment(audio, session, g)
    pairs, spans = chunk_speaker_samples(labels, g)
    keep, embs = [], []
    for p, runs in zip(pairs, spans):
        stream = extractor.create_stream()
        for a, b in runs:
            b = min(b, len(audio))
            if b > a:
                stream.accept_waveform(g.rate, np.ascontiguousarray(audio[a:b], dtype=np.float32))
        stream.input_finished()
        e = np.array(extractor.compute(stream), np.float32)
        if not np.isnan(e).any():
            keep.append(p)
            embs.append(e)
    return Store(np.stack(labels).astype(np.int8), np.array(keep, np.int32).reshape(-1, 2),
                 np.stack(embs), len(audio))


def diarize(store: Store, method: str = "complete", threshold: float = 0.5, min_count: int = 1,
            g: Geometry = SEGMENTATION_3, min_on: float = 0.3, min_off: float = 0.5) -> tuple[list[Turn], int]:
    clusters, inv = cluster(store.embeddings, method, threshold)
    if min_count > 1:
        clusters = reassign_small(store.embeddings, clusters, min_count)
    pairs = [tuple(p) for p in store.pairs.tolist()]
    return reconstruct(list(store.labels), pairs, clusters, store.n_samples, g, min_on, min_off), inv
