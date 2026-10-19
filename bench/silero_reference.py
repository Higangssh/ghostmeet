"""Reference Silero spans for a WAV, from the VAD the benchmark measured with.

Writes JSON to stdout or --out: the per-frame speech
probability of the benchmark's `silero_vad.onnx`, computed with onnxruntime the way
sherpa-onnx feeds it (512-sample frames, state carried across frames), and the spans
sherpa-onnx's VoiceActivityDetector cut from the same audio with the benchmark's VAD
values (`engines.VAD`). The Rust engine reimplements the span rule over probabilities;
these two together let it check that rule against sherpa without either model.

Run with the bench venv:  bench/.venv/bin/python -I bench/silero_reference.py WAV
With --fleurs MINUTES, the WAV is first built from FLEURS the way run_stream.py builds
its session (real read speech, not synthetic) and written to the given WAV path.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engines  # noqa: E402

FRAME = 512


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == engines.SAMPLE_RATE and w.getnchannels() == 1
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def probabilities(audio: np.ndarray) -> list[float]:
    import onnxruntime

    sess = onnxruntime.InferenceSession(str(engines.MODELS / "silero_vad.onnx"))
    names = [i.name for i in sess.get_inputs()]
    assert names == ["x", "h", "c"], f"expected the v4-style export sherpa uses, got {names}"
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    out = []
    # sherpa only runs whole frames; a trailing partial frame is never scored
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        prob, h, c = sess.run(None, {"x": audio[None, i : i + FRAME], "h": h, "c": c})
        out.append(round(float(prob[0, 0]), 6))
    return out


def unpadded_spans(audio: np.ndarray) -> list[list[int]]:
    """sherpa's own segments, before `speech_spans` pads them."""
    pad = engines.VAD["pad_sec"]
    engines.VAD["pad_sec"] = 0.0
    try:
        return [list(s) for s in engines.speech_spans(audio)]
    finally:
        engines.VAD["pad_sec"] = pad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fleurs", type=float, metavar="MINUTES")
    args = ap.parse_args()

    if args.fleurs:
        from run_stream import build_session

        audio, _, _ = build_session(args.fleurs)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        args.wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(args.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(engines.SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
    audio = read_wav(args.wav)
    ref = {
        "sample_rate": engines.SAMPLE_RATE,
        "frame": FRAME,
        "total_samples": len(audio),
        "vad": engines.VAD,
        "probs": probabilities(audio),
        "spans_unpadded": unpadded_spans(audio),
        "spans": [list(s) for s in engines.speech_spans(audio)],
    }
    text = json.dumps(ref)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    print(
        f"{len(ref['probs'])} frames, {len(ref['spans'])} spans:",
        [(round(a / 16000, 2), round(b / 16000, 2)) for a, b in ref["spans"]],
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
