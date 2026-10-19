"""Turn the synthetic meetings into speech and back into text, the way the app would.

    bench/.venv/bin/python -I bench/asr_meetings.py weekly-30m customer-60m ... [--vocabulary] [--keep-audio]

Each line of a meeting from `meeting_gen` (bench/data/meetings_v2/) is spoken with a macOS Korean voice (`say`),
the lines are joined with short pauses into one 16 kHz mono s16le recording, and the
app's after-meeting pass (`final_pass`, Silero spans + large-v3-turbo) transcribes it.
The meeting file is written again under bench/data/meetings_asr/ with the recognised
text as `transcript`; `lines` and the answer key stay the script's, so a summary is
still scored against what was said. `run_summary.py --set asr` summarises `transcript`.

--vocabulary gives the pass the user's vocabulary (STT-7) and writes meetings_asr_vocab/:
the people named in the meeting, then two features it never mentions (decoys: a term the
prompt holds and nobody said must not show up), then its features, most said first, as
far as the prompt budget goes. Both runs count how often each of those terms is heard
against how often it is said. --keep-audio keeps the recording so the next run can
reuse it: the same audio with and without the vocabulary.

Build the pass first:
cargo build --release -p meetbutler-engine --features whisper,silero --example final_pass
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cer_parts  # noqa: E402
from meeting_gen import CLIENTS, FEATURES, OUR_COMPANY, PEOPLE  # noqa: E402

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
DATA = BENCH / "data"
MEETINGS = DATA / "meetings_v2"  # meeting_gen's current version
AUDIO = DATA / "meetings_audio"
OUT = DATA / "meetings_asr"
MODELS = DATA / "models"
FINAL_PASS = REPO / "target/release/examples/final_pass"

RATE = 16000
# macOS Korean voices; a meeting has several people, so lines take turns among them.
# The Eloquence ones need their full names: "Eddy" alone is the English voice, which
# reads Korean as a short noise.
VOICES = ["Yuna"] + [f"{v} (한국어(한국))" for v in ("Eddy", "Flo", "Reed", "Sandy", "Shelley")]
PAUSE_SEC = (0.3, 1.2)
PROMPT_CHARS = 100  # crates/meetbutler-engine/src/vocabulary.rs, MAX_PROMPT_CHARS
DECOYS = 2


def count(term: str, text: str) -> int:
    return len(re.findall(re.escape(re.sub(r"\s+", "", term)), re.sub(r"\s+", "", text)))


def vocabulary(meeting: dict) -> tuple[list[str], list[str]]:
    """The terms a user would give for this meeting, and which of them are decoys."""
    said = " ".join(meeting["lines"])
    people = [p for p in PEOPLE if count(p, said)]
    features = sorted((f for f in FEATURES if count(f, said)), key=lambda f: -count(f, said))
    decoys = [f for f in FEATURES if not count(f, said)][:DECOYS]
    terms, size = [], 0
    for t in people + decoys + [c for c in [OUR_COMPANY] + CLIENTS if count(c, said)] + features:
        if size + len(t) + 2 > PROMPT_CHARS:
            break
        terms.append(t)
        size += len(t) + 2
    return terms, decoys


def term_counts(meeting: dict, heard: str) -> dict:
    """How often the meeting's own names and terms were said, heard right and heard extra,
    and how often the decoys were heard (none were said)."""
    said = " ".join(meeting["lines"])
    terms, decoys = vocabulary(meeting)
    every = [t for t in PEOPLE + FEATURES + [OUR_COMPANY] + CLIENTS if count(t, said)]
    said_n = {t: count(t, said) for t in every}
    heard_n = {t: count(t, heard) for t in every}
    return {
        "said": sum(said_n.values()),
        "heard": sum(min(said_n[t], heard_n[t]) for t in every),
        "extra": sum(max(0, heard_n[t] - said_n[t]) for t in every),
        # the terms the vocabulary held, apart: the ones it can help
        "in_vocabulary_said": sum(said_n[t] for t in every if t in terms),
        "in_vocabulary_heard": sum(min(said_n[t], heard_n[t]) for t in every if t in terms),
        "decoys_heard": sum(count(d, heard) for d in decoys),
        "vocabulary": terms,
    }


def speak(text: str, voice: str, path: Path) -> None:
    subprocess.run(
        ["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(path), text],
        check=True,
    )


def record(meeting: dict, name: str) -> tuple[Path, float]:
    """The meeting spoken into one recording; returns its path and length in seconds."""
    work = AUDIO / name
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(meeting["seed"])
    voices = [rng.choice(VOICES) for _ in meeting["lines"]]
    paths = [work / f"{i:05d}.wav" for i in range(len(meeting["lines"]))]
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(speak, meeting["lines"], voices, paths))

    pcm = AUDIO / f"{name}.pcm"
    samples = 0
    with pcm.open("wb") as out:
        for path in paths:
            gap = int(rng.uniform(*PAUSE_SEC) * RATE)
            out.write(b"\0\0" * gap)
            with wave.open(str(path)) as w:
                assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
                out.write(w.readframes(w.getnframes()))
                samples += gap + w.getnframes()
            path.unlink()
    work.rmdir()
    return pcm, samples / RATE


def transcribe(pcm: Path, words: Path | None) -> dict:
    run = subprocess.run(
        ["/usr/bin/time", "-l", str(FINAL_PASS), str(MODELS / "ggml-large-v3-turbo.bin"),
         str(MODELS / "silero_vad.onnx"), "ko", str(pcm)] + ([str(words)] if words else []),
        capture_output=True, text=True, check=True,
    )
    out = json.loads(run.stdout)
    peak = max((int(l.split()[0]) for l in run.stderr.splitlines() if "peak memory footprint" in l), default=0)
    out["peak_mb"] = round(peak / 2**20)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("meetings", nargs="*")
    ap.add_argument("--vocabulary", action="store_true")
    ap.add_argument("--keep-audio", action="store_true")
    args = ap.parse_args()
    if not FINAL_PASS.exists():
        raise SystemExit(f"{FINAL_PASS} missing; see the module docstring")
    out_dir = OUT.with_name(OUT.name + "_vocab") if args.vocabulary else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.meetings or sorted(p.stem for p in MEETINGS.glob("*.json")):
        meeting = json.loads((MEETINGS / f"{name}.json").read_text())
        t = time.perf_counter()
        pcm = AUDIO / f"{name}.pcm"
        if pcm.exists():  # left by --keep-audio: hear the very same recording again
            audio_sec = pcm.stat().st_size / 2 / RATE
        else:
            pcm, audio_sec = record(meeting, name)
        tts_sec = time.perf_counter() - t
        words = None
        if args.vocabulary:
            words = AUDIO / f"{name}.vocabulary.txt"
            words.write_text("\n".join(vocabulary(meeting)[0]) + "\n")
        asr = transcribe(pcm, words)
        transcript = [text.strip() for _, _, text in asr["segments"] if text.strip()]
        edits, length = cer_parts(" ".join(meeting["lines"]), " ".join(transcript))
        meeting["transcript"] = transcript
        meeting["asr"] = {
            "engine": "final_pass large-v3-turbo (Metal, Silero spans + blocklist)",
            "audio_sec": round(audio_sec, 1),
            "tts_sec": round(tts_sec, 1),
            "decode_sec": round(asr["decode_sec"], 1),
            "peak_mb": asr["peak_mb"],
            "segments": len(transcript),
            "cer": round(edits / length, 4),
            "terms": term_counts(meeting, " ".join(transcript)),
            "with_vocabulary": args.vocabulary,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(meeting, ensure_ascii=False, indent=1))
        if words:
            words.unlink()
        if not args.keep_audio:
            pcm.unlink()  # synthetic, but large; the transcript is what the summaries need
        terms = meeting["asr"]["terms"]
        print(f"{name}: {audio_sec / 60:.1f} min of speech, CER {edits / length:.1%}, "
              f"decode {asr['decode_sec']:.0f}s, {len(transcript)} segments, "
              f"terms {terms['heard']}/{terms['said']} extra {terms['extra']} decoys {terms['decoys_heard']}",
              flush=True)


if __name__ == "__main__":
    main()
