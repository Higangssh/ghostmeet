"""One-shot transcription of a whole session: the after-the-meeting final pass.

    python -I bench/run_longform.py <engine-spec> [--minutes 36]

Builds the same session as run_stream.py and hands the engine all of it in one call,
so the engine's own long-form decoding (not IncrementalTranscriber) decides where
windows fall. Gives CER and RTF for a batch re-transcription after the meeting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engines  # noqa: E402
from common import RAW, RESULTS, SAMPLE_RATE, cer_parts, footprint_mb, machine  # noqa: E402
from run_stream import build_session  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--minutes", type=float, default=36)
    args = ap.parse_args()

    audio, ref, n_utt = build_session(args.minutes)
    eng = engines.make(args.engine)
    eng(audio[: 5 * SAMPLE_RATE])  # warm-up, as in run_stream.py

    t = time.perf_counter()
    segs = eng(audio)
    dt = time.perf_counter() - t
    peak_mb = footprint_mb(eng.pid)[1]
    eng.close()

    hyp = " ".join(s.text.strip() for s in segs if s.text.strip())
    edits, ref_len = cer_parts(ref, hyp)
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
    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.engine}.longform.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (RAW / f"{args.engine}.longform.hyp.json").write_text(
        json.dumps([[s.start, s.end, s.text] for s in segs], ensure_ascii=False)
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
