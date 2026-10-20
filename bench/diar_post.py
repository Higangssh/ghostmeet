"""After clustering: a cluster with too little speech to be a person is folded into the
nearest one that has enough (by cosine similarity of their embedding centroids).

Clustering to a threshold leaves a tail of clusters of a few seconds each — overlaps,
noise, a voice in an odd moment — that count as extra speakers; forcing the number of
clusters instead merges real speakers and keeps that tail (bench/results/RESULTS.md §9).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from der import Turn


def absorb(turns: list[Turn], centroids: dict[str, np.ndarray], min_sec: float) -> list[Turn]:
    speech = Counter()
    for t in turns:
        speech[t.speaker] += t.end - t.start
    big = [s for s in speech if speech[s] >= min_sec and s in centroids]
    if not big:
        return list(turns)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)  # noqa: E731
    into = {}
    for s in speech:
        if speech[s] >= min_sec or s not in centroids:
            continue
        into[s] = max(big, key=lambda b: float(unit(centroids[s]) @ unit(centroids[b])))
    return [Turn(into.get(t.speaker, t.speaker), t.start, t.end) for t in turns]
