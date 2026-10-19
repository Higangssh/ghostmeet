"""One-shot transcription of a whole session: the after-the-meeting final pass.

    python -I bench/run_longform.py <engine-spec> [--minutes 36] [--replies]

Builds the same session as run_stream.py and hands the engine all of it in one call,
so the engine's own long-form decoding (not IncrementalTranscriber) decides where
windows fall. Gives CER and RTF for a batch re-transcription after the meeting.

--replies puts a short synthetic reply (bench/make_replies.sh: 네, 아니요, ...) between
every REPLY_EVERY sentences, set off by REPLY_GAP_SEC of noise on each side, and reports
how many come back. A reply counts as recalled when most of its characters align
unchanged with the hypothesis. For a packed engine the replies lost to its filter are
counted by scoring again with the dropped segments put back.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

import engines  # noqa: E402
from common import DATA, RAW, RESULTS, SAMPLE_RATE, cer_parts, footprint_mb, load_fleurs, machine, normalize  # noqa: E402
from run_stream import GAP_SEC, build_session  # noqa: E402

REPLIES = DATA / "replies"
REPLY_EVERY = 3
REPLY_GAP_SEC = 0.8


def build_session_with_replies(minutes: float):
    """Like build_session, plus a reply after every REPLY_EVERY sentences.

    Returns (audio, pieces, n_sentences, n_replies); pieces is [(text, is_reply)] in order.
    """
    replies = [line.split("\t") for line in (REPLIES / "replies.tsv").read_text().splitlines()]
    clips = [(sf.read(REPLIES / name, dtype="float32")[0], text) for name, text in replies]
    rng = np.random.default_rng(0)

    def noise(sec):
        return (rng.standard_normal(int(sec * SAMPLE_RATE)) * 1e-3).astype(np.float32)

    parts, pieces, n, k = [], [], 0, 0
    for u in load_fleurs():
        parts += [u.audio(), noise(GAP_SEC)]
        pieces.append((u.text, False))
        n += 1
        if n % REPLY_EVERY == 0:
            clip, text = clips[k % len(clips)]
            parts += [noise(REPLY_GAP_SEC - GAP_SEC), clip, noise(REPLY_GAP_SEC)]
            pieces.append((text, True))
            k += 1
        if sum(len(p) for p in parts) >= minutes * 60 * SAMPLE_RATE:
            break
    return np.concatenate(parts), pieces, n, k


def recalled(pieces, hyp: str) -> list:
    """For each reply in `pieces`, whether it survives in `hyp` (character alignment)."""
    ref, spans = "", []
    for text, is_reply in pieces:
        t = normalize(text)
        if is_reply:
            spans.append((len(ref), len(ref) + len(t)))
        ref += t
    kept = np.zeros(len(ref), bool)
    for tag, i1, i2, _, _ in Levenshtein.opcodes(ref, normalize(hyp)):
        if tag == "equal":
            kept[i1:i2] = True
    return [kept[a:b].mean() > 0.5 for a, b in spans]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--minutes", type=float, default=36)
    ap.add_argument("--replies", action="store_true")
    args = ap.parse_args()

    if args.replies:
        audio, pieces, n_utt, n_replies = build_session_with_replies(args.minutes)
        ref = " ".join(text for text, _ in pieces)
    else:
        audio, ref, n_utt = build_session(args.minutes)
    eng = engines.make(args.engine)
    eng(audio[: 5 * SAMPLE_RATE])  # warm-up, as in run_stream.py

    t = time.perf_counter()
    segs = eng(audio)
    dt = time.perf_counter() - t
    # an engine in a child process that has already exited reports its own peak
    peak_mb = getattr(eng, "peak_mb", None) or footprint_mb(eng.pid)[1]
    eng.close()

    def text(ss):
        return " ".join(s.text.strip() for s in ss if s.text.strip())

    hyp = text(segs)
    edits, ref_len = cer_parts(ref, hyp)
    dropped = getattr(eng, "dropped", [])
    session_s = len(audio) / SAMPLE_RATE
    out = {
        "engine": eng.name,
        "spec": args.engine,
        "mode": "longform",
        "session_min": round(session_s / 60, 1),
        "utterances": n_utt,
        "segments": len(segs),
        "cer": round(edits / ref_len, 4),
        "sec": round(dt, 1),
        "rtf": round(dt / session_s, 3),
        "peak_footprint_mb": round(peak_mb),
        "machine": machine(),
    }
    if hasattr(eng, "decode_sec"):  # wall time above includes loading the model
        out["decode_sec"] = round(eng.decode_sec, 1)
        out["decode_rtf"] = round(eng.decode_sec / session_s, 3)
    if hasattr(eng, "dropped"):
        out["filter_dropped_segments"] = len(dropped)
    tag = "longform"
    if args.replies:
        tag = "replies.longform"
        got = recalled(pieces, hyp)
        with_dropped = recalled(pieces, text(sorted(segs + dropped, key=lambda s: s.start)))
        out.update({
            "replies": n_replies,
            "replies_recalled": int(sum(got)),
            "replies_lost_to_filter": int(sum(w and not g for g, w in zip(got, with_dropped))),
            "replies_missed": [t for (t, r), g in zip([p for p in pieces if p[1]], got) if not g],
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.engine}.{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (RAW / f"{args.engine}.{tag}.hyp.json").write_text(
        json.dumps([[s.start, s.end, s.text] for s in segs], ensure_ascii=False)
    )
    if dropped:
        (RAW / f"{args.engine}.{tag}.dropped.json").write_text(json.dumps(
            [[s.start, s.end, s.text, s.no_speech_prob, s.avg_logprob] for s in dropped], ensure_ascii=False
        ))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
