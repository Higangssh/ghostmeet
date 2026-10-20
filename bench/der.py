"""Diarization error rate and the app's own view of it, without third-party scorers.

    from der import Turn, score, segment_accuracy

DER is computed the usual way (NIST md-eval): time is cut into 10 ms frames; in each,
with R reference and H hypothesis speakers and the hypothesis labels mapped one to one
onto the reference ones to agree the most over the whole recording,
    missed       = max(0, R - H)
    false alarm  = max(0, H - R)
    confusion    = min(R, H) - (mapped speakers that are right)
and DER = (missed + false alarm + confusion) / total reference speaker time. A collar
around every reference boundary is left out on request, as is overlapped speech.

`segment_accuracy` scores what the app shows: each transcript segment gets the one
speaker the hypothesis has most in it; it is right when that is (mapped) the speaker
the reference has most in it. Short segments (a "네") are counted apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME = 0.01


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float


@dataclass
class Score:
    total: float
    missed: float
    false_alarm: float
    confusion: float
    ref_speakers: int
    hyp_speakers: int
    mapping: dict

    @property
    def der(self) -> float:
        return (self.missed + self.false_alarm + self.confusion) / self.total if self.total else 0.0


@dataclass
class SegmentScore:
    right: int
    total: int
    short_right: int
    short_total: int


def assign(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Rows paired with columns, one to one, for the largest total weight (Hungarian
    method on the negated, squared-up matrix). Pairs of zero weight are left out."""
    n = max(len(weights), max((len(r) for r in weights), default=0))
    if n == 0:
        return []
    big = max((v for r in weights for v in r), default=0)
    cost = [[big - (weights[i][j] if i < len(weights) and j < len(weights[i]) else 0) for j in range(n)]
            for i in range(n)]
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, n + 1):
        i = p[j] - 1
        if i < len(weights) and j - 1 < len(weights[i]) and weights[i][j - 1] > 0:
            pairs.append((i, j - 1))
    return sorted(pairs)


def _grid(turns: list[Turn], labels: list[str], frames: int) -> np.ndarray:
    """A speakers × frames boolean grid."""
    g = np.zeros((len(labels), frames), dtype=bool)
    index = {s: i for i, s in enumerate(labels)}
    for t in turns:
        a, b = int(round(t.start / FRAME)), int(round(t.end / FRAME))
        g[index[t.speaker], max(0, a):max(0, min(b, frames))] = True
    return g


def _frames(*groups: list[Turn]) -> int:
    return int(round(max((t.end for g in groups for t in g), default=0) / FRAME)) + 1


def _mapping(ref: list[Turn], hyp: list[Turn]):
    rl, hl = sorted({t.speaker for t in ref}), sorted({t.speaker for t in hyp})
    n = _frames(ref, hyp)
    R, H = _grid(ref, rl, n), _grid(hyp, hl, n)
    overlap = (H.astype(np.int32) @ R.T.astype(np.int32)).tolist()  # hyp × ref frames together
    return rl, hl, R, H, {hl[i]: rl[j] for i, j in assign(overlap)}


def score(ref: list[Turn], hyp: list[Turn], collar: float = 0.0, skip_overlap: bool = False) -> Score:
    rl, hl, R, H, mapping = _mapping(ref, hyp)
    keep = np.ones(R.shape[1], dtype=bool)
    if collar > 0:
        c = int(round(collar / FRAME))
        for t in ref:
            for edge in (t.start, t.end):
                e = int(round(edge / FRAME))
                keep[max(0, e - c):e + c] = False
    nref, nhyp = R.sum(axis=0), H.sum(axis=0)
    if skip_overlap:
        keep &= nref <= 1
    # mapped hypothesis speakers that are right, per frame
    right = np.zeros(R.shape[1], dtype=np.int32)
    for h, r in mapping.items():
        right += H[hl.index(h)] & R[rl.index(r)]
    nref, nhyp, right = nref[keep], nhyp[keep], right[keep]
    sec = lambda x: float(x.sum()) * FRAME  # noqa: E731
    return Score(
        total=sec(nref),
        missed=sec(np.maximum(0, nref - nhyp)),
        false_alarm=sec(np.maximum(0, nhyp - nref)),
        confusion=sec(np.minimum(nref, nhyp) - right),
        ref_speakers=len(rl),
        hyp_speakers=len(hl),
        mapping=mapping,
    )


def segment_accuracy(ref: list[Turn], hyp: list[Turn], segments: list[tuple[float, float]],
                     short: float = 1.5) -> SegmentScore:
    rl, hl, R, H, mapping = _mapping(ref, hyp)
    out = SegmentScore(0, 0, 0, 0)
    for a, b in segments:
        fa, fb = int(round(a / FRAME)), max(int(round(b / FRAME)), int(round(a / FRAME)) + 1)
        rs, hs = R[:, fa:fb].sum(axis=1), H[:, fa:fb].sum(axis=1)
        if not rs.any():
            continue  # no reference speech under it: not a speaker question
        want = rl[int(rs.argmax())]
        got = mapping.get(hl[int(hs.argmax())]) if hs.any() else None
        ok = got == want
        out.total += 1
        out.right += ok
        if b - a <= short:
            out.short_total += 1
            out.short_right += ok
    return out


def to_rttm(turns: list[Turn], file_id: str) -> str:
    return "".join(
        f"SPEAKER {file_id} 1 {t.start:.3f} {t.end - t.start:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"
        for t in turns
    )


def from_rttm(text: str) -> list[Turn]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            out.append(Turn(f[7], round(start, 3), round(start + dur, 3)))
    return out
