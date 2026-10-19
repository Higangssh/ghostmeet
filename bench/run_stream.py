"""Long-session behaviour through the production tail-window code path.

    python -I bench/run_stream.py <engine-spec> [--minutes 36] [--mode backlog|live]

FLEURS ko test utterances are concatenated (TSV order, 0.6 s low-level noise gaps) into
one session of at least `--minutes`, then fed through backend.incremental's
IncrementalTranscriber (window 30 s, overlap 2 s), the same class the app uses.

  backlog  all audio is already there; every pass is a full 30 s window. Isolates the
           per-window cost and shows whether it stays flat across the session.
  live     audio "arrives" in real time on a virtual clock and a pass runs every
           `--interval` seconds after the previous one finished, like the pipeline
           worker. Gives caption latency (pass end - segment end) without waiting
           36 real minutes; only compute time is spent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import engines  # noqa: E402
from common import RAW, RESULTS, SAMPLE_RATE, cer_parts, footprint_mb, load_fleurs, machine  # noqa: E402

from backend.incremental import IncrementalTranscriber  # noqa: E402  (path set by engines)

GAP_SEC = 0.6


class ArraySource:
    """AudioSource over an in-memory array whose visible length the caller controls."""

    def __init__(self, audio: np.ndarray):
        self._audio = audio
        self.available = len(audio)

    @property
    def total_samples(self) -> int:
        return self.available

    def read(self, start_sample, end_sample=None):
        end = self.available if end_sample is None else min(end_sample, self.available)
        return self._audio[start_sample:end]


def build_session(minutes: float) -> tuple[np.ndarray, str, int]:
    rng = np.random.default_rng(0)
    parts, refs, n = [], [], 0
    for u in load_fleurs():
        parts.append(u.audio())
        parts.append((rng.standard_normal(int(GAP_SEC * SAMPLE_RATE)) * 1e-3).astype(np.float32))
        refs.append(u.text)
        n += 1
        if sum(len(p) for p in parts) >= minutes * 60 * SAMPLE_RATE:
            break
    return np.concatenate(parts), " ".join(refs), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--minutes", type=float, default=36)
    ap.add_argument("--mode", choices=["backlog", "live"], default="backlog")
    ap.add_argument("--interval", type=float, default=10.0)  # GHOSTMEET_CHUNK_INTERVAL
    ap.add_argument("--label", default="after", help="which backend code: before|after the fix")
    args = ap.parse_args()

    audio, ref, n_utt = build_session(args.minutes)
    total = len(audio)
    eng = engines.make(args.engine)
    eng(audio[: 5 * SAMPLE_RATE])  # warm-up so pass 1 is not a cold start

    src = ArraySource(audio)
    inc = IncrementalTranscriber(src, eng, window_sec=30.0, overlap_sec=2.0, min_new_sec=1.0)
    passes, latencies = [], []
    clock = 0.0  # virtual seconds since session start (live mode)
    if args.mode == "live":
        src.available = 0

    while True:
        if args.mode == "live":
            clock += args.interval
            src.available = min(total, int(clock * SAMPLE_RATE))
        before = inc._cursor
        force = src.available >= total  # all audio is in: drain with forced passes, as pipeline._drain does
        t = time.perf_counter()
        new = inc.step(force=force)
        dt = time.perf_counter() - t
        if args.mode == "live":
            clock += dt
            latencies += [clock - s.end for s in new]
        if dt > 0 and (new or inc._cursor != before):
            start = max(0, before - inc._overlap_samples)
            passes.append({
                "pos_min": round(start / SAMPLE_RATE / 60, 3),
                "window_s": round((min(src.available, start + inc._window_samples) - start) / SAMPLE_RATE, 2),
                "sec": round(dt, 3),
                "footprint_mb": footprint_mb(eng.pid)[0],
                "segments": len(new),
            })
        if src.available >= total and inc.pending_samples == 0:
            break

    peak_mb = footprint_mb(eng.pid)[1]
    eng.close()
    hyp = inc.full_text()
    edits, ref_len = cer_parts(ref, hyp)

    full = [p for p in passes if args.mode == "live" or p["window_s"] >= 29.9]
    sec = np.array([p["sec"] for p in full])
    pos = np.array([p["pos_min"] for p in full])
    mem = np.array([p["footprint_mb"] for p in full])
    first, last = pos <= 10, pos >= pos.max() - 10
    slope = float(np.polyfit(pos, sec, 1)[0]) if len(full) > 2 else 0.0
    out = {
        "engine": eng.name,
        "spec": args.engine,
        "mode": args.mode,
        "label": args.label,
        "interval_s": args.interval if args.mode == "live" else None,
        "session_min": round(total / SAMPLE_RATE / 60, 1),
        "utterances": n_utt,
        "passes": len(passes),
        "cer": round(edits / ref_len, 4),
        "pass_sec_first10min": round(float(sec[first].mean()), 3),
        "pass_sec_last10min": round(float(sec[last].mean()), 3),
        "pass_sec_p95": round(float(np.percentile(sec, 95)), 3),
        "pass_sec_slope_per_hour": round(slope * 60, 3),
        "footprint_mb_first10min": round(float(mem[first].mean()), 1),
        "footprint_mb_last10min": round(float(mem[last].mean()), 1),
        "peak_footprint_mb": peak_mb,
        "machine": machine(),
    }
    if latencies:
        lat = np.array(latencies)
        out["caption_latency_median_s"] = round(float(np.median(lat)), 2)
        out["caption_latency_p95_s"] = round(float(np.percentile(lat, 95)), 2)
    tag = f"{args.engine}.{args.mode}.{args.label}"
    (RESULTS / f"{tag}.stream.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    (RAW / f"{tag}.passes.json").write_text(json.dumps({"passes": passes, "hyp": hyp}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "machine"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
