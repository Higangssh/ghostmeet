"""Engine adapters with one shape: load once, then `engine(audio) -> [Seg]`.

Each adapter mirrors the production decode settings as closely as the engine allows
(backend/transcriber.py): Korean fixed, beam 5 for Whisper, Silero VAD with
threshold 0.3 / min silence 500 ms / pad 200 ms, no text context carried between calls.
"""
from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf

from common import MODELS, REPO, SAMPLE_RATE

sys.path.insert(0, str(REPO))  # reuse the production transcriber/incremental code

THREADS = 4  # M4 has 4 performance cores; same budget for every engine
LANG = "ko"


@dataclass
class Seg:
    start: float
    end: float
    text: str


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

    def __init__(self, size: str, gpu: bool = True, vad: bool = True):
        import requests

        self._http = requests.Session()
        port = _free_port()
        self._url = f"http://127.0.0.1:{port}/inference"
        self.name = f"whisper.cpp {size} ({'Metal' if gpu else 'CPU'}{'' if vad else ', no VAD'})"
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
            timeout=600,
        )
        r.raise_for_status()
        return [Seg(s["start"], s["end"], s["text"]) for s in r.json().get("segments", [])]

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

        self._so = sherpa_onnx
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
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
        cfg.silero_vad.threshold = 0.3
        cfg.silero_vad.min_silence_duration = 0.5
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = 25
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        self._vad_cfg = cfg
        self._pad = int(0.2 * SAMPLE_RATE)

    def _speech_spans(self, audio):
        vad = self._so.VoiceActivityDetector(self._vad_cfg, buffer_size_in_seconds=120)
        win = self._vad_cfg.silero_vad.window_size
        spans = []
        for i in range(0, len(audio), win):
            vad.accept_waveform(audio[i : i + win])
            while not vad.empty():
                spans.append((vad.front.start, len(vad.front.samples)))
                vad.pop()
        vad.flush()
        while not vad.empty():
            spans.append((vad.front.start, len(vad.front.samples)))
            vad.pop()
        return spans

    def __call__(self, audio):
        spans = self._speech_spans(audio)
        if not spans:
            return []
        streams, bounds = [], []
        for start, n in spans:
            a, b = max(0, start - self._pad), min(len(audio), start + n + self._pad)
            s = self._rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio[a:b])
            streams.append(s)
            bounds.append((a / SAMPLE_RATE, b / SAMPLE_RATE))
        self._rec.decode_streams(streams)
        return [Seg(a, b, s.result.text) for s, (a, b) in zip(streams, bounds)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make(spec: str) -> Engine:
    """fw-small | fw-large-v3-turbo | wcpp-small | wcpp-small-cpu | wcpp-small-novad | sensevoice | ..."""
    kind, _, rest = spec.partition("-")
    if kind == "fw":
        return FasterWhisper(rest)
    if kind == "wcpp":
        cpu, novad = rest.endswith("-cpu"), rest.endswith("-novad")
        return WhisperCpp(rest.removesuffix("-cpu").removesuffix("-novad"), gpu=not cpu, vad=not novad)
    if spec == "sensevoice":
        return SenseVoice("cpu")
    if spec == "sensevoice-coreml":
        return SenseVoice("coreml")
    raise ValueError(spec)
