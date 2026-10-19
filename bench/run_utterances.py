"""Short-utterance accuracy and speed: every FLEURS ko test utterance, one call each.

    python -I bench/run_utterances.py <engine-spec> [--limit N]

Writes bench/results/<spec>.utt.json (numbers only) and raw hypotheses to
bench/results/raw/ (gitignored).
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
from common import RAW, RESULTS, cer_parts, footprint_mb, load_fleurs, machine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="utt", help="output name: <engine>.<tag>.json")
    args = ap.parse_args()

    utts = load_fleurs(args.limit)
    first_audio = utts[0].audio()

    t0 = time.perf_counter()
    eng = engines.make(args.engine)
    load_s = time.perf_counter() - t0
    first = eng(first_audio)  # cold call: includes any lazy init / shader compile
    first_result_s = time.perf_counter() - t0

    rows, edits, ref_len, edits_nd, ref_nd = [], 0, 0, 0, 0
    proc_s = audio_s = 0.0
    for u in utts:
        audio = u.audio()
        t = time.perf_counter()
        hyp = " ".join(s.text.strip() for s in eng(audio))
        dt = time.perf_counter() - t
        e, n = cer_parts(u.text, hyp)
        edits += e
        ref_len += n
        if not any(c.isdigit() for c in u.text):
            edits_nd += e
            ref_nd += n
        proc_s += dt
        audio_s += u.duration
        rows.append({"id": u.id, "dur": round(u.duration, 2), "sec": round(dt, 3), "edits": e, "ref": n, "hyp": hyp})

    cur_mb, peak_mb = footprint_mb(eng.pid)
    eng.close()
    lat = np.array([r["sec"] for r in rows])
    out = {
        "engine": eng.name,
        "spec": args.engine,
        "dataset": "FLEURS ko_kr test",
        "utterances": len(rows),
        "audio_min": round(audio_s / 60, 1),
        "cer": round(edits / ref_len, 4),
        "cer_no_digits": round(edits_nd / ref_nd, 4),
        "rtf": round(proc_s / audio_s, 4),
        "load_s": round(load_s, 2),
        "first_result_s": round(first_result_s, 2),
        "first_result_audio_s": round(utts[0].duration, 2),
        "utt_latency_median_s": round(float(np.median(lat)), 3),
        "utt_latency_p95_s": round(float(np.percentile(lat, 95)), 3),
        "peak_footprint_mb": peak_mb,
        "machine": machine(),
    }
    (RESULTS / f"{args.engine}.{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    (RAW / f"{args.engine}.{args.tag}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "machine"}, ensure_ascii=False))
    _ = first


if __name__ == "__main__":
    main()
