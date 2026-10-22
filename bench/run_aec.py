"""The echo bench (REC-4): call DER when the far side leaks into the microphone, with no
cancellation, with the crude `drop_echo`, and with software echo cancellation (sonora's
AEC3, `bench/aec/`) run on the two recorded tracks after the meeting, with the reference
as recorded and moved 20 ms earlier.

The meetings are the chunked diarization test calls (`run_chunk.py build-test`): the
microphone is me alone, the system track the far side. The leak is made from the system
track (`aec_leak`), so the microphone holds me + leak. The system side is diarized as
preregistered (ERes2NetV2, chunked); only "me" changes with the leak and its cleaning.

    bench/.venv/bin/python -I bench/run_aec.py build    # cargo build of bench/aec, offline
    bench/.venv/bin/python -I bench/run_aec.py run

Stop between units with bench/data/diar/STOP, as the other diarization runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aec_leak as al  # noqa: E402
import der  # noqa: E402
import diar_chunk as dc  # noqa: E402
import diar_pipe as dp  # noqa: E402
import run_chunk as rc  # noqa: E402
import run_diar as rd  # noqa: E402

BENCH = Path(__file__).resolve().parent
CRATE = BENCH / "aec"
WORK = BENCH / "data" / "aec"
TARGET = WORK / "target"
BINARY = TARGET / "release" / "meetbutler-aec-bench"
RESULTS = BENCH / "results" / "aec"
EMB = "eres2netv2-zh"
RATE = al.RATE

CONDITIONS = {
    "none": None,                                   # no leak: the floor
    "flat": al.Condition(),                         # the diarization bench's bleed
    "d20": al.Condition(delay_ms=20),
    "d80": al.Condition(delay_ms=80),
    "d200": al.Condition(delay_ms=200),
    "room": al.Condition(rt60=0.3),
    "drift": al.Condition(drift_ppm=50),
    "real": al.Condition(delay_ms=80, rt60=0.3, drift_ppm=50),
}
METHODS = ["raw", "drop_echo", "sonora", "sonora_lead20"]
# the reference moved this much earlier before cancelling: AEC3 cannot cancel an echo that
# arrives with or before its reference, and the tracks on disk can always be shifted
LEAD_MS = 20


def meetings() -> list[str]:
    return [s.name for s in rc.specs("test") if s.scene == "call" and not rc.is_long(s.name)]


def build() -> None:
    subprocess.run(["cargo", "build", "--release", "--offline", "--locked"], cwd=CRATE, check=True,
                   env={**__import__("os").environ, "CARGO_TARGET_DIR": str(TARGET)})


def sonora(mic: np.ndarray, system: np.ndarray, tag: str) -> tuple[np.ndarray, float]:
    WORK.mkdir(parents=True, exist_ok=True)
    a, b, out = (WORK / f"{tag}.{k}.f32" for k in ("mic", "system", "out"))
    mic.astype("<f4").tofile(a)
    system.astype("<f4").tofile(b)
    t = time.perf_counter()
    subprocess.run([str(BINARY), str(a), str(b), str(out)], check=True, capture_output=True)
    sec = time.perf_counter() - t
    y = np.fromfile(out, "<f4")
    for p in (a, b, out):
        p.unlink()
    return np.concatenate([y, np.zeros(len(mic) - len(y), np.float32)]), sec


def me_turns(mic: np.ndarray, seg) -> list[dp.Turn]:
    labels = np.stack(dp.segment(mic, seg)).astype(np.int8)
    return dc.one_voice(labels, len(mic), rc.G, "me")


def _mask(turns, n: int, step: float = 0.01) -> np.ndarray:
    m = np.zeros(int(n / RATE / step) + 1, bool)
    for t in turns:
        m[int(t.start / step):int(t.end / step)] = True
    return m


def me_stats(ref, me_hyp, n: int) -> dict:
    """Of my speech, how much is still "me" (double talk kept); of "me", how much is not mine
    (leak taken for me)."""
    mine = _mask([t for t in ref if t.speaker == "me"], n)
    said = _mask(me_hyp, n)
    return {"me_kept": round(float((mine & said).sum() / max(1, mine.sum())), 4),
            "me_not_mine": round(float((said & ~mine).sum() / max(1, said.sum())), 4)}


def run() -> None:
    pre = json.loads(rc.PREREGISTERED.read_text())["models"][EMB]
    seg = rd.segmenter()
    for name in meetings():
        d = rc.meeting_dir("test", name)
        mic, _ = sf.read(d / "mic.wav", dtype="float32")
        system, _ = sf.read(d / "system.wav", dtype="float32")
        m = rc.Meeting("test", name, EMB, track="system")
        them = m.hyp(dc.diarize_labels(m.store, rc.G, rc.config_of(pre["chunked"])))
        them = [t for t in them if t.speaker != "me"]
        meta = m.meta
        ref = [der.Turn("me" if t.speaker == meta["me"] else t.speaker, t.start, t.end) for t in m.ref]
        for cond, c in CONDITIONS.items():
            leaked = mic if c is None else mic + al.leak(system, c)
            for method in METHODS:
                out = RESULTS / f"{name}.{cond}.{method}.json"
                if out.exists() or (c is None and method != "raw"):
                    continue
                rd.unit_boundary()
                aec_sec = None
                if method.startswith("sonora"):
                    k = LEAD_MS * RATE // 1000 if method.endswith("_lead20") else 0
                    ref_sig = np.concatenate([system[k:], np.zeros(k, np.float32)]) if k else system
                    cleaned, aec_sec = sonora(leaked, ref_sig, f"{name}.{cond}")
                    me = me_turns(cleaned, seg)
                else:
                    me = me_turns(leaked, seg)
                    if method == "drop_echo":
                        me = dc.drop_echo(me, leaked, system, RATE)
                hyp = sorted(me + them, key=lambda t: (t.start, t.speaker))
                result = {"meeting": name, "condition": cond, "leak": None if c is None else vars(c),
                          "method": method, "embedding": EMB, "system_side": pre["chunked"],
                          "aec_sec": None if aec_sec is None else round(aec_sec, 1),
                          "audio_sec": round(len(mic) / RATE, 1),
                          **me_stats(ref, me, len(mic)),
                          **rd.scored(ref, hyp, [(t.start, t.end) for t in ref]), "env": rd.env(EMB)}
                RESULTS.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=1))
                print(f"aec {name} {cond} {method}: DER {result['collar25']['der']:.3f} "
                      f"me kept {result['me_kept']:.3f} not mine {result['me_not_mine']:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["build", "run"])
    for step in ap.parse_args().steps:
        try:
            build() if step == "build" else run()
        except rd.Stopped:
            print(f"stopped: {rd.STOP} is there; remove it and run again to carry on")
            return


if __name__ == "__main__":
    main()
