"""Score the app's own summary (the Rust merge step, SUM-3 evidence) on the meeting sets.

    bench/.venv/bin/python -I bench/app_summary.py v2 asr [--meetings weekly-30m ...]

Runs `examples/summarise` (the app's `summary::summarise`, qwen3.5-4b, prompt p2) over
each meeting of bench/data/meetings_<set>/. The model's answers are recorded the first
time (bench/results/raw/summary_app_<set>/) and replayed after that, so a change to the
merge step is scored again in seconds: decoding is greedy, the same prompts get the same
answers. Scores go to bench/results/summary_app_<set>/, beside run_summary.py's.

Build first:
cargo build --release -p meetbutler-summarizer --features llama --example summarise --bin meetbutler-summarizer
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import summary_score  # noqa: E402

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
EXAMPLE = REPO / "target/release/examples/summarise"
SIDECAR = REPO / "target/release/meetbutler-summarizer"
MODEL = "qwen3.5-4b"
MANIFEST = BENCH / "llm_models.json"
MEETINGS = ["weekly-30m", "one_on_one-30m", "customer-30m", "weekly-60m", "one_on_one-60m", "customer-60m"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--meetings", nargs="*", default=MEETINGS)
    args = ap.parse_args()
    gguf = BENCH / "data/models/llm" / json.loads(MANIFEST.read_text())[MODEL]["file"]
    for name in args.sets:
        meetings = BENCH / "data" / f"meetings_{name}"
        results = BENCH / "results" / f"summary_app_{name}"
        raw = BENCH / "results/raw" / f"summary_app_{name}"
        results.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        for m in args.meetings:
            path = meetings / f"{m}.json"
            answers = raw / f"{MODEL}.{m}.answers.json"
            mode = ["--replay", str(answers)] if answers.exists() else ["--record", str(answers)]
            t = time.perf_counter()
            run = subprocess.run([str(EXAMPLE), str(SIDECAR), str(gguf), str(path), *mode],
                                 capture_output=True, text=True)
            wall = time.perf_counter() - t
            error = None if run.returncode == 0 else run.stderr.strip().splitlines()[-1]
            text = run.stdout if error is None else ""
            (raw / f"{MODEL}.{m}.out.json").write_text(text)
            meeting = json.loads(path.read_text())
            result = {
                "model": MODEL, "tier": "8GB", "meeting": m, "mode": "chunked", "prompt": "p2",
                "merge": "app", "error": error,
                "score": summary_score.score(text, meeting) if error is None else {},
                # only a recorded run's time means anything; a replay runs no model
                **({"wall_sec": round(wall, 1)} if mode[0] == "--record" else {}),
            }
            out = results / f"{MODEL}.{m}.chunked.app.json"
            if mode[0] == "--replay" and out.exists():
                result.setdefault("wall_sec", json.loads(out.read_text()).get("wall_sec"))
            out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
            s = result["score"]
            print(f"{name}/{m}: {mode[0][2:]} dec {s.get('decisions_found')}/{s.get('decisions_total')} "
                  f"evidence {s.get('decisions_evidence_ok')} act {s.get('actions_found')} "
                  f"evidence {s.get('actions_evidence_ok')} {error or ''}", flush=True)


if __name__ == "__main__":
    main()
