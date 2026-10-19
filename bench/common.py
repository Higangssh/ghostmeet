"""Shared pieces for the STT benchmark: dataset loading, CER, memory sampling.

Everything here reads only from bench/data/ (gitignored). Nothing writes audio back
into the repository.
"""
from __future__ import annotations

import csv
import os
import platform
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
DATA = BENCH / "data"
MODELS = DATA / "models"
FLEURS_TSV = DATA / "downloads" / "fleurs" / "test.tsv"
FLEURS_WAV = DATA / "fleurs_ko" / "test"
RESULTS = BENCH / "results"
RAW = RESULTS / "raw"

SAMPLE_RATE = 16000


@dataclass
class Utterance:
    id: str
    path: Path
    text: str  # FLEURS raw transcription (keeps digits, punctuation stripped by CER)
    samples: int

    @property
    def duration(self) -> float:
        return self.samples / SAMPLE_RATE

    def audio(self) -> np.ndarray:
        data, sr = sf.read(self.path, dtype="float32")
        assert sr == SAMPLE_RATE, sr
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data


def load_fleurs(limit: Optional[int] = None) -> List[Utterance]:
    """FLEURS ko_kr test split, in TSV order (deterministic)."""
    out: List[Utterance] = []
    with open(FLEURS_TSV, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            out.append(Utterance(id=row[1], path=FLEURS_WAV / row[1], text=row[2], samples=int(row[5])))
    return out[:limit] if limit else out


_GLOSS = re.compile(r"\([^)]*\)")  # FLEURS keeps unspoken glosses: 비슈케크(Bishkek)
_DROP = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """CER normalisation: NFKC, lowercase, drop whitespace and punctuation.

    Korean spacing is inconsistent between engines and references, so CER is computed on
    characters with all spaces removed. Digits are kept as-is (see README on numerals).
    """
    text = _GLOSS.sub("", unicodedata.normalize("NFKC", text)).lower()
    return _DROP.sub("", text)


def edit_distance(ref: str, hyp: str) -> int:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(ref, hyp)


def cer_parts(ref: str, hyp: str) -> tuple[int, int]:
    """(edits, reference length) on normalised text, for corpus-level aggregation."""
    r, h = normalize(ref), normalize(hyp)
    return edit_distance(r, h), len(r)


def _rusage(pid: int) -> list[int]:
    import ctypes

    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 80)()
    if lib.proc_pid_rusage(pid, 4, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V4
        raise OSError(f"proc_pid_rusage failed for {pid}")
    return list(buf)[2:]  # skip the 16-byte uuid


def footprint_mb(pid: int) -> tuple[float, float]:
    """macOS (current, lifetime peak) phys_footprint in MB.

    Footprint is what Activity Monitor calls Memory; unlike RSS it also counts Metal/IOKit
    allocations, which matters for the GPU engines.
    """
    v = _rusage(pid)
    return round(v[7] / 2**20, 1), round(v[28] / 2**20, 1)


def machine() -> dict:
    def sysctl(key: str) -> str:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "cores": int(sysctl("hw.ncpu") or 0),
        "perf_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0),
        "eff_cores": int(sysctl("hw.perflevel1.physicalcpu") or 0),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
        "os": f"macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
    }


def mb(n: int) -> float:
    return round(n / 2**20, 1)


os.makedirs(RAW, exist_ok=True)
