"""Engine adapters with one shape: load once, then `engine(audio) -> [Seg]`.

Each adapter mirrors the production decode settings as closely as the engine allows
(backend/transcriber.py): Korean fixed, beam 5 for Whisper, Silero VAD with
threshold 0.3 / min silence 500 ms / pad 200 ms, no text context carried between calls.
"""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf

from common import DATA, MODELS, REPO, SAMPLE_RATE

sys.path.insert(0, str(REPO))  # reuse the production transcriber/incremental code

THREADS = 4  # M4 has 4 performance cores; same budget for every engine
LANG = "ko"


@dataclass
class Seg:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0  # Whisper only; 0 where the engine has no such score
    avg_logprob: float = 0.0


class Engine:
    name: str
    pid: int  # process whose memory counts (a child for whisper.cpp)

    def __call__(self, audio: np.ndarray) -> List[Seg]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FasterWhisper(Engine):
    def __init__(self, size: str):
        from faster_whisper import WhisperModel

        from backend.transcriber import WhisperTranscriber

        self.name = f"faster-whisper {size} (CPU int8)"
        self.pid = os.getpid()
        model = WhisperModel(
            size,
            device="cpu",
            compute_type="int8",
            cpu_threads=THREADS,
            download_root=str(MODELS / "faster-whisper"),
        )
        self._t = WhisperTranscriber(model, language=LANG)  # the exact production call

    def __call__(self, audio):
        return [Seg(s.start, s.end, s.text) for s in self._t(audio)]


class WhisperCpp(Engine):
    """whisper-server (Homebrew whisper.cpp) on loopback; model loaded once per run."""

    def __init__(self, size: str, gpu: bool = True, vad: bool = True, dtw: bool = False):
        import requests

        self._http = requests.Session()
        port = _free_port()
        self._url = f"http://127.0.0.1:{port}/inference"
        self.name = (
            f"whisper.cpp {size} ({'Metal' if gpu else 'CPU'}"
            f"{'' if vad else ', no VAD'}{', DTW' if dtw else ''})"
        )
        self._dtw = dtw
        cmd = [
            "whisper-server",
            "-m", str(MODELS / f"ggml-{size}.bin"),
            "-l", LANG,
            "-t", str(THREADS),
            "-bs", "5",
            "-mc", "0",  # = condition_on_previous_text=False
            "--host", "127.0.0.1", "--port", str(port),
        ]
        if vad:
            cmd += [
                "--vad", "-vm", str(MODELS / "ggml-silero-v5.1.2.bin"),
                "-vt", "0.3", "-vsd", "500", "-vp", "200",
            ]
        if not gpu:
            cmd.append("-ng")
        if dtw:  # token times from cross-attention; the library ignores DTW under flash attention
            cmd += ["-dtw", size.replace("-", "."), "-nfa"]
        self._log = open(MODELS.parent / f"whisper-server-{port}.log", "w")
        self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self.pid = self._proc.pid
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"whisper-server exited: see {self._log.name}")
            try:
                self._http.get(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except requests.ConnectionError:
                time.sleep(0.1)
        else:
            raise TimeoutError("whisper-server did not come up")

    def __call__(self, audio):
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        r = self._http.post(
            self._url,
            files={"file": ("w.wav", buf.getvalue(), "audio/wav")},
            data={"response_format": "verbose_json", "temperature": "0.0"},
            timeout=3600,  # a long-form run sends a whole session in one request
        )
        r.raise_for_status()
        return [
            Seg(s["start"], self._end(s), s["text"], s.get("no_speech_prob", 0.0), s.get("avg_logprob", 0.0))
            for s in r.json().get("segments", [])
        ]

    def _end(self, seg: dict) -> float:
        """Segment end; with DTW, where the last word was actually spoken."""
        if not self._dtw:
            return seg["end"]
        ends = [w["t_dtw"] * 0.01 for w in seg.get("words", []) if w.get("t_dtw", -1) >= 0]
        return min(seg["end"], max(ends)) if ends else seg["end"]

    def close(self):
        self._proc.terminate()
        self._proc.wait(timeout=30)
        self._log.close()


class SenseVoice(Engine):
    """sherpa-onnx SenseVoice-Small int8, with Silero VAD to cut speech into segments.

    SenseVoice has no notion of timestamps across a long input, so the window is split
    by VAD first (the same role vad_filter plays for faster-whisper) and each speech
    span is decoded as one batch.
    """

    DIR = MODELS / "sv" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

    def __init__(self, provider: str = "cpu"):
        import sherpa_onnx

        self.name = f"sherpa-onnx SenseVoice-Small ({provider.upper()} int8)"
        self.pid = os.getpid()
        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.DIR / "model.int8.onnx"),
            tokens=str(self.DIR / "tokens.txt"),
            num_threads=THREADS,
            language=LANG,
            use_itn=True,  # digits and punctuation, as Whisper produces
            provider=provider,
        )

    def __call__(self, audio):
        spans = speech_spans(audio)
        if not spans:
            return []
        streams = []
        for a, b in spans:
            s = self._rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio[a:b])
            streams.append(s)
        self._rec.decode_streams(streams)
        return [Seg(a / SAMPLE_RATE, b / SAMPLE_RATE, s.result.text) for s, (a, b) in zip(streams, spans)]


# Silero VAD (MIT, snakers4/silero-vad) through sherpa-onnx. One rule for every engine that
# is fed VAD spans, live and batch alike; RESULTS.md quotes these values.
VAD = {
    "threshold": 0.3,
    "min_silence_sec": 0.5,  # a pause this long ends a span
    "min_speech_sec": 0.25,  # shorter blips are dropped
    "max_speech_sec": 25.0,  # longer speech is split, so a span fits one Whisper window
    "pad_sec": 0.2,  # context kept on each side of a span
}
_vad_cfg = None


def speech_spans(audio: np.ndarray) -> List[tuple]:
    """[(start_sample, end_sample)] of speech in `audio`, padded and clipped to it."""
    global _vad_cfg
    import sherpa_onnx

    if _vad_cfg is None:
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
        cfg.silero_vad.threshold = VAD["threshold"]
        cfg.silero_vad.min_silence_duration = VAD["min_silence_sec"]
        cfg.silero_vad.min_speech_duration = VAD["min_speech_sec"]
        cfg.silero_vad.max_speech_duration = VAD["max_speech_sec"]
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        _vad_cfg = cfg
    vad = sherpa_onnx.VoiceActivityDetector(_vad_cfg, buffer_size_in_seconds=120)
    win, pad = _vad_cfg.silero_vad.window_size, int(VAD["pad_sec"] * SAMPLE_RATE)
    spans = []

    def drain():
        while not vad.empty():
            start, n = vad.front.start, len(vad.front.samples)
            spans.append((max(0, start - pad), min(len(audio), start + n + pad)))
            vad.pop()

    for i in range(0, len(audio), win):
        vad.accept_waveform(audio[i : i + win])
        drain()
    vad.flush()
    drain()
    return spans


class SpanDecoded(Engine):
    """Cut the input into our own VAD spans and decode each one on its own.

    Times come from the spans, so the engine's own VAD and long-form decoding are out of
    the picture; the inner engine runs with its VAD off.
    """

    def __init__(self, inner: Engine):
        self._inner = inner
        self.name = inner.name.replace(", no VAD", "").rstrip(")") + ", Silero spans)"
        self.pid = inner.pid

    def __call__(self, audio):
        out = []
        for a, b in speech_spans(audio):
            text = " ".join(s.text.strip() for s in self._inner(audio[a:b]) if s.text.strip())
            if text:
                out.append(Seg(a / SAMPLE_RATE, b / SAMPLE_RATE, text))
        return out

    def close(self):
        self._inner.close()


# Packing and the hallucination filter, for engines that score segments (Whisper). Both are
# plain numbers so the same rule ports to whisper-rs: whisper_full_get_segment_no_speech_prob
# and the mean token log-probability of the segment.
PACK = {
    "max_chunk_sec": 25.0,  # neighbouring spans are decoded together up to this length
    "no_speech_prob": 0.6,  # drop a segment when no_speech_prob is above this ...
    "avg_logprob": -1.0,  # ... and avg_logprob is below this (OpenAI/faster-whisper rule)
}


def pack(spans: List[tuple], max_samples: int) -> List[tuple]:
    """Merge consecutive spans while the merged chunk stays within `max_samples`."""
    chunks: List[list] = []
    for a, b in spans:
        if chunks and b - chunks[-1][0] <= max_samples:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])
    return [tuple(c) for c in chunks]


def hallucinated(seg: Seg) -> bool:
    return seg.no_speech_prob > PACK["no_speech_prob"] and seg.avg_logprob < PACK["avg_logprob"]


class PackedDecoded(Engine):
    """Our Silero spans, packed into chunks of up to max_chunk_sec, decoded per chunk.

    A lone short span (a "네" between sentences) is decoded with its neighbours instead of
    on its own, which is where Whisper invents subtitle credits. Segment times are the
    engine's own within the chunk, offset to the input; the engine's VAD is off. Segments
    the filter drops are kept in `dropped` so a run can count what the filter cost.
    """

    def __init__(self, inner: Engine, packed: bool = True):
        self._inner = inner
        how = "packed" if packed else "spans"
        self.name = inner.name.replace(", no VAD", "").rstrip(")") + f", Silero {how} + filter)"
        self.pid = inner.pid
        self._limit = int(PACK["max_chunk_sec"] * SAMPLE_RATE) if packed else 0  # 0: one span each
        self.dropped: List[Seg] = []

    def __call__(self, audio):
        out = []
        for a, b in pack(speech_spans(audio), self._limit):
            off, end = a / SAMPLE_RATE, b / SAMPLE_RATE
            for s in self._inner(audio[a:b]):
                if not s.text.strip():
                    continue
                seg = Seg(off + s.start, min(end, off + s.end), s.text, s.no_speech_prob, s.avg_logprob)
                (self.dropped if hallucinated(seg) else out).append(seg)
        return out

    def close(self):
        self._inner.close()


class RustFinal(Engine):
    """The app's own after-meeting pass: the Rust engine's `final_pass` example.

    Same rule as `PackedDecoded(packed=False)` (our spans one by one, credits on short
    spans dropped) but end to end in Rust: the same silero_vad.onnx on tract, the Rust
    span rule and blocklist, whisper-rs for decoding. Build it first:
    cargo build --release -p meetbutler-engine --features whisper,silero --example final_pass
    """

    BINARY = REPO / "target/release/examples/final_pass"

    def __init__(self, size: str):
        if not self.BINARY.exists():
            raise SystemExit(f"{self.BINARY} missing; see RustFinal")
        self.name = f"Rust engine {size} (Metal, Silero spans + blocklist)"
        self.pid = os.getpid()
        self._model = MODELS / f"ggml-{size}.bin"
        self.peak_mb = 0.0
        self.decode_sec = 0.0

    def __call__(self, audio):
        tmp = DATA / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        pcm = tmp / f"rust-final-{os.getpid()}.pcm"
        pcm.write_bytes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        try:
            run = subprocess.run(
                ["/usr/bin/time", "-l", str(self.BINARY), str(self._model),
                 str(MODELS / "silero_vad.onnx"), LANG, str(pcm)],
                capture_output=True, text=True, check=True,
            )
        finally:
            pcm.unlink(missing_ok=True)
        out = json.loads(run.stdout)
        self.decode_sec = out["decode_sec"]
        for line in run.stderr.splitlines():
            if "peak memory footprint" in line:
                self.peak_mb = max(self.peak_mb, int(line.split()[0]) / 2**20)
        return [Seg(a, b, t) for a, b, t in out["segments"]]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make(spec: str) -> Engine:
    """fw-small | fw-large-v3-turbo | wcpp-small | wcpp-small-cpu | wcpp-small-novad | wcpp-small-dtw | wcpp-small-spans | wcpp-small-packed | wcpp-small-filtered | sensevoice | ..."""
    kind, _, rest = spec.partition("-")
    if kind == "fw":
        return FasterWhisper(rest)
    if kind == "wcpp":
        parts = rest.split("-")
        flags = {f: f in parts for f in ("cpu", "novad", "dtw", "spans", "packed", "filtered")}
        size = "-".join(p for p in parts if p not in flags)
        vad = not (flags["novad"] or flags["spans"] or flags["packed"] or flags["filtered"])
        engine = WhisperCpp(size, gpu=not flags["cpu"], vad=vad, dtw=flags["dtw"])
        if flags["packed"] or flags["filtered"]:  # filtered: our spans one by one, plus the filter
            return PackedDecoded(engine, packed=flags["packed"])
        return SpanDecoded(engine) if flags["spans"] else engine
    if kind == "rust":
        return RustFinal(rest)
    if spec == "sensevoice":
        return SenseVoice("cpu")
    if spec == "sensevoice-coreml":
        return SenseVoice("coreml")
    raise ValueError(spec)
