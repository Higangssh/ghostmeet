"""Speaker diarization candidates on Korean mock meetings and on AMI (requirements §12, 1).

    bench/.venv/bin/python -I bench/run_diar.py tune            # thresholds, on dev meetings
    bench/.venv/bin/python -I bench/run_diar.py korean ami      # the test sets
    bench/.venv/bin/python -I bench/run_diar.py online          # live (chunked) against after-meeting
    bench/.venv/bin/python -I bench/run_diar.py long            # a 4-hour meeting: time and memory
    bench/.venv/bin/python -I bench/run_diar.py absorb          # small clusters folded away (diar_post)
    bench/.venv/bin/python -I bench/run_diar.py tune30 korean30 ami30 [--joint]   # B: 30-minute dev
    bench/.venv/bin/python -I bench/run_diar.py extract-dev verify grid preregister extract-test pipe   # diar_pipe

Pipeline: sherpa-onnx's offline diarization (pyannote segmentation-3.0 + a speaker
embedding + clustering), the one the app would embed; the embedding is what varies.

Data (bench/data/diar/, never committed; bench/fetch_diar.sh fetches and checks it):
- zeroth/: Zeroth-Korean (OpenSLR 40, CC BY 4.0), read speech laid out as meetings by
  `diar_mix` (room and call scenes, overlap, short replies): the test set's 10 speakers
  for the test meetings (raw/), 12 other speakers from the training set for tuning
  (train/; streamed out of the archive, the rest never kept)
- ami/: the AMI Meeting Corpus test set (CC BY 4.0), 16 real meetings of 4, headset mix
  and the first far-field array microphone, with BUT's reference RTTMs and UEMs
- models/: the ONNX files from sherpa-onnx's releases (bench/diar_models.json)

Writes bench/results/diar/<set>/<meeting>.<system>.json (scores only; hypotheses go to
bench/results/raw/diar/).

Stopping and resuming: a unit is one system on one meeting (one threshold of one
embedding when tuning). Before each, the run looks for bench/data/diar/STOP and, if it is
there, stops cleanly; a unit whose result is already saved is skipped, so running the
same step again carries on where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import der  # noqa: E402
import diar_mix  # noqa: E402
import diar_post  # noqa: E402

BENCH = Path(__file__).resolve().parent
DATA = BENCH / "data/diar"
MODELS = DATA / "models"
RESULTS = BENCH / "results/diar"
RAW = BENCH / "results/raw/diar"
RATE = diar_mix.RATE

SEGMENTATION = MODELS / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDINGS = {
    "campplus-zh-en": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "eres2netv2-zh": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
    "eres2net-en": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    "titanet-small": "nemo_en_titanet_small.onnx",
    "wespeaker-r34": "wespeaker_en_voxceleb_resnet34_LM.onnx",
}
# sherpa's clustering threshold: larger means fewer speakers; a 2-minute trial put the
# useful range for these embeddings between 0.7 and 1.1
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
# clusters with less than this share of the meeting's speech are folded into the nearest
# big one (diar_post); tuned with the threshold, on dev, for the first choice only. A
# share, not seconds: dev meetings are 5 minutes and test ones 30 or more, and a floor in
# seconds tuned on the one kept most of the other's tail (20 s left 11 clusters for 2)
SHARES = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
# the floor is tuned on dev meetings as long as the test ones (same scenes and sizes); the
# first absorb, a floor in seconds tuned on the 5-minute dev, failed on test, and the share
# was chosen after seeing that — so these results are exploratory, not a confirmation
ABSORB = "campplus-zh-en"
# the oracle speaker count separates counting from clustering; for the first choice only
ORACLE = ["campplus-zh-en"]
THREADS = 4  # the engine's budget

# the Korean grid: scene x speakers; dev seeds tune the threshold, test seeds report
SCENES = ["room", "call"]
SIZES = [2, 4, 6]
DEV_SEEDS, TEST_SEEDS = [100], [1, 2]
DEV_MINUTES, TEST_MINUTES = 5, 30
ABSORB_DEV_MINUTES = TEST_MINUTES  # see SHARES
OVERLAP, SHORT = 0.10, 0.15


STOP = DATA / "STOP"


class Stopped(Exception):
    pass


def unit_boundary() -> None:
    if STOP.exists():
        raise Stopped


def done(set_name: str, meeting: str, system: str) -> bool:
    return (RESULTS / set_name / f"{meeting}.{system}.json").exists()


# --- data -----------------------------------------------------------------------------

def zeroth_pool(split: str = "test") -> dict[str, list[np.ndarray]]:
    """Speaker id -> clips. Zeroth files are <speaker>_<session>_<utterance>.flac.
    `test`: the test set's 10 speakers, for the test meetings. `dev`: 12 speakers of the
    training set (none of them in the test set), for tuning, so a threshold is never
    fitted to the speakers it is scored on."""
    pool: dict[str, list[np.ndarray]] = {}
    root = DATA / ("zeroth/raw" if split == "test" else "zeroth/train")
    for path in sorted(root.rglob("*.flac")):
        speaker = path.stem.split("_")[0]
        audio, sr = sf.read(path, dtype="float32")
        assert sr == RATE, path
        pool.setdefault(speaker, []).append(audio)
    return pool


def korean_meetings(seeds: list[int], minutes: float, split: str = "test"):
    pool = zeroth_pool(split)
    if split == "dev":
        assert not set(pool) & set(zeroth_pool("test")), "dev and test speakers overlap"
    for scene in SCENES:
        for n in SIZES:
            for seed in seeds:
                m = diar_mix.mix(pool, speakers=n, minutes=minutes, seed=seed, overlap=OVERLAP,
                                 short=SHORT, scene=scene)
                yield f"ko-{scene}-{n}p-{minutes:g}m-s{seed}", m.audio, m.turns, m.segments


def ami_meetings():
    ami = DATA / "ami"
    for name in (ami / "test.meetings.txt").read_text().split():
        ref = der.from_rttm((ami / "rttm" / f"{name}.rttm").read_text())
        _, _, a, b = (ami / "uem" / f"{name}.uem").read_text().split()[:4]
        lo, hi = float(a), float(b)
        ref = [der.Turn(t.speaker, max(t.start, lo), min(t.end, hi)) for t in ref if t.end > lo and t.start < hi]
        for mic in ("Mix-Headset", "Array1-01"):
            audio, sr = sf.read(ami / "audio" / f"{name}.{mic}.wav", dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == RATE, (name, sr)
            # score inside the UEM only: silence the outside so nothing there is a false alarm
            audio = audio.copy()
            audio[: int(lo * RATE)] = 0
            audio[int(hi * RATE):] = 0
            yield f"ami-{name}-{mic}", audio, ref, [(t.start, t.end) for t in ref]


# --- the pipeline ---------------------------------------------------------------------

def diarizer(embedding: str, threshold: float = 0.5, num_clusters: int = -1):
    import sherpa_onnx as so

    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION)),
            num_threads=THREADS,
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(MODELS / EMBEDDINGS[embedding]),
                                                     num_threads=THREADS),
        clustering=so.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    assert config.validate(), embedding
    return so.OfflineSpeakerDiarization(config)


def run(sd, audio: np.ndarray, offset: float = 0.0, label: str = "") -> list[der.Turn]:
    out = sd.process(audio).sort_by_start_time()
    return [der.Turn(f"{label}{s.speaker}", round(offset + s.start, 3), round(offset + s.end, 3)) for s in out]


def scored(ref, hyp, segments) -> dict:
    s0, s25 = der.score(ref, hyp), der.score(ref, hyp, collar=0.25)
    s25o = der.score(ref, hyp, collar=0.25, skip_overlap=True)
    seg = der.segment_accuracy(ref, hyp, segments)
    pct = lambda s: {"der": round(s.der, 4), "missed": round(s.missed / s.total, 4),  # noqa: E731
                     "false_alarm": round(s.false_alarm / s.total, 4),
                     "confusion": round(s.confusion / s.total, 4)}
    return {
        "collar0": pct(s0), "collar25": pct(s25), "collar25_no_overlap": pct(s25o),
        "ref_speakers": s0.ref_speakers, "hyp_speakers": s0.hyp_speakers,
        "segments_right": seg.right, "segments": seg.total,
        "short_right": seg.short_right, "short": seg.short_total,
    }


_SHA: dict[Path, str] = {}


def sha256(path: Path) -> str:
    if path not in _SHA:
        import hashlib

        _SHA[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _SHA[path]


def env(emb: str | None) -> dict:
    """What produced a result: library versions and the model files' checksums."""
    import onnxruntime
    import scipy
    import sherpa_onnx

    out = {"sherpa_onnx": sherpa_onnx.__version__, "onnxruntime": onnxruntime.__version__,
           "scipy": scipy.__version__, "numpy": np.__version__,
           "segmentation_sha256": sha256(SEGMENTATION)}
    if emb in EMBEDDINGS:
        out["embedding_sha256"] = sha256(MODELS / EMBEDDINGS[emb])
    return out


def save(set_name: str, meeting: str, system: str, result: dict, hyp: list[der.Turn]) -> None:
    result = {**result, "env": env(system.split(".")[0])}
    (RESULTS / set_name).mkdir(parents=True, exist_ok=True)
    (RAW / set_name).mkdir(parents=True, exist_ok=True)
    (RESULTS / set_name / f"{meeting}.{system}.json").write_text(json.dumps(result, indent=1))
    (RAW / set_name / f"{meeting}.{system}.rttm").write_text(der.to_rttm(hyp, meeting))


# --- the steps ------------------------------------------------------------------------

def tune() -> dict[str, float]:
    """Each embedding's threshold, by DER (collar 0.25) over the dev meetings."""
    dev = list(korean_meetings(DEV_SEEDS, DEV_MINUTES, split="dev"))
    best = {}
    for emb in EMBEDDINGS:
        saved = RESULTS / "tune" / f"{emb}.json"
        if saved.exists():
            best[emb] = json.loads(saved.read_text())["best"]
            continue
        table = {}
        for th in THRESHOLDS:
            unit_boundary()
            sd = diarizer(emb, threshold=th)
            tot = err = 0.0
            for _, audio, ref, _ in dev:
                s = der.score(ref, run(sd, audio), collar=0.25)
                tot += s.total
                err += s.missed + s.false_alarm + s.confusion
            table[th] = round(err / tot, 4)
            print(f"tune {emb} {th}: DER {table[th]:.3f}", flush=True)
        best[emb] = min(table, key=table.get)
        (RESULTS / "tune").mkdir(parents=True, exist_ok=True)
        (RESULTS / "tune" / f"{emb}.json").write_text(json.dumps({"der_by_threshold": table, "best": best[emb]}, indent=1))
    return best


def thresholds() -> dict[str, float]:
    return {e: json.loads((RESULTS / "tune" / f"{e}.json").read_text())["best"] for e in EMBEDDINGS}


def evaluate(set_name: str, meetings) -> None:
    best = thresholds()
    for meeting, audio, ref, segments in meetings:
        speakers = len({t.speaker for t in ref})
        for emb in EMBEDDINGS:
            systems = [(f"{emb}.th", lambda: diarizer(emb, threshold=best[emb]))]
            if emb in ORACLE:
                systems.append((f"{emb}.oracle", lambda: diarizer(emb, num_clusters=speakers)))
            for system, make in systems:
                if done(set_name, meeting, system):
                    continue
                unit_boundary()
                sd = make()
                t = time.perf_counter()
                hyp = run(sd, audio)
                sec = time.perf_counter() - t
                result = {"meeting": meeting, "system": system, "embedding": emb,
                          "threshold": best[emb] if system.endswith(".th") else None,
                          "audio_sec": round(len(audio) / RATE, 1), "process_sec": round(sec, 1),
                          **scored(ref, hyp, segments)}
                save(set_name, meeting, system, result, hyp)
                print(f"{set_name} {meeting} {system}: DER {result['collar25']['der']:.3f} "
                      f"spk {result['hyp_speakers']}/{speakers} {sec:.0f}s", flush=True)


def online(audio: np.ndarray, emb: str, threshold: float, link: float, chunk_sec: float = 60.0):
    """Live-style: each chunk diarized on its own as it closes, its speakers linked to the
    meeting's so far by embedding similarity (a new speaker when none is near enough).
    The labels given are never revised."""
    import sherpa_onnx as so

    sd = diarizer(emb, threshold=threshold)
    ex = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(
        model=str(MODELS / EMBEDDINGS[emb]), num_threads=THREADS))
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    out: list[der.Turn] = []
    step = int(chunk_sec * RATE)
    for a in range(0, len(audio), step):
        piece = audio[a:a + step]
        local = run(sd, piece)
        for spk in sorted({t.speaker for t in local}):
            own = [t for t in local if t.speaker == spk]
            voice = np.concatenate([piece[int(t.start * RATE):int(t.end * RATE)] for t in own])
            if len(voice) < RATE // 2:
                gid = None
            else:
                stream = ex.create_stream()
                stream.accept_waveform(RATE, voice)
                stream.input_finished()
                e = np.array(ex.compute(stream))
                e /= np.linalg.norm(e) + 1e-9
                sims = [float(e @ (c / (np.linalg.norm(c) + 1e-9))) for c in centroids]
                gid = int(np.argmax(sims)) if sims and max(sims) >= link else None
                if gid is None:
                    centroids.append(e)
                    counts.append(1)
                    gid = len(centroids) - 1
                else:
                    centroids[gid] = centroids[gid] + e
                    counts[gid] += 1
            for t in own:
                out.append(der.Turn(f"g{gid if gid is not None else 'x'}", round(t.start + a / RATE, 3),
                                    round(t.end + a / RATE, 3)))
    return out


def evaluate_online(emb: str = "campplus-zh-en", links=(0.4, 0.5, 0.6)) -> None:
    best = thresholds()
    ami_headset = (m for m in ami_meetings() if m[0].endswith("Mix-Headset"))
    for set_name, meetings in (("korean", korean_meetings(TEST_SEEDS, TEST_MINUTES)), ("ami", ami_headset)):
        for meeting, audio, ref, segments in meetings:
            todo = [link for link in links if not done(set_name, meeting, f"{emb}.online{link}")]
            if not todo:
                continue
            unit_boundary()
            after = run(diarizer(emb, threshold=best[emb]), audio)
            for link in todo:
                unit_boundary()
                live = online(audio, emb, best[emb], link)
                result = {"meeting": meeting, "system": f"{emb}.online{link}", **scored(ref, live, segments)}
                # how many transcript segments would change speaker when the after-meeting
                # result replaces the live one
                agree = der.segment_accuracy(after, live, segments)
                result["relabelled"] = round(1 - agree.right / agree.total, 4) if agree.total else 0.0
                save(set_name, meeting, f"{emb}.online{link}", result, live)
                print(f"online {meeting} link {link}: DER {result['collar25']['der']:.3f} "
                      f"relabelled {result['relabelled']:.1%}", flush=True)


def centroids(audio: np.ndarray, hyp: list[der.Turn], ex, most_sec: float = 60.0) -> dict:
    """Each cluster's embedding, from up to `most_sec` of its longest turns."""
    out = {}
    for spk in {t.speaker for t in hyp}:
        own = sorted((t for t in hyp if t.speaker == spk), key=lambda t: t.start - t.end)
        voice, have = [], 0.0
        for t in own:
            if have >= most_sec:
                break
            voice.append(audio[int(t.start * RATE):int(t.end * RATE)])
            have += t.end - t.start
        voice = np.concatenate(voice)
        if len(voice) < RATE // 2:
            continue
        stream = ex.create_stream()
        stream.accept_waveform(RATE, voice)
        stream.input_finished()
        out[spk] = np.array(ex.compute(stream))
    return out


def extractor(emb: str):
    import sherpa_onnx as so

    return so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(
        model=str(MODELS / EMBEDDINGS[emb]), num_threads=THREADS))


def floor_sec(hyp: list[der.Turn], share: float) -> float:
    return share * sum(t.end - t.start for t in hyp)


def tune_absorb(emb: str = ABSORB) -> tuple[float, float]:
    """The threshold and floor together, by DER (collar 0.25) over the dev meetings: with
    small clusters folded away, a lower threshold can split more without counting extras."""
    saved = RESULTS / "tune" / f"{emb}.absorb.json"
    if saved.exists():
        d = json.loads(saved.read_text())
        return d["threshold"], d["share"]
    dev = list(korean_meetings(DEV_SEEDS, ABSORB_DEV_MINUTES, split="dev"))
    ex = extractor(emb)
    table = {}
    for th in THRESHOLDS:
        unit_boundary()
        sd = diarizer(emb, threshold=th)
        tot, err = 0.0, dict.fromkeys(SHARES, 0.0)
        for _, audio, ref, _ in dev:
            hyp = run(sd, audio)
            cents = centroids(audio, hyp, ex)
            for n in SHARES:
                s = der.score(ref, diar_post.absorb(hyp, cents, floor_sec(hyp, n)), collar=0.25)
                err[n] += s.missed + s.false_alarm + s.confusion
            tot += s.total
        for n in SHARES:
            table[f"{th}/{n}"] = round(err[n] / tot, 4)
        print(f"tune {emb}.absorb {th}: " + " ".join(f"{n:.1%} {err[n] / tot:.3f}" for n in SHARES), flush=True)
    th, n = min(table, key=table.get).split("/")
    saved.write_text(json.dumps({"exploratory": True, "dev_minutes": ABSORB_DEV_MINUTES,
                                 "der_by_threshold_share": table, "threshold": float(th), "share": float(n)},
                                indent=1))
    return float(th), float(n)


def evaluate_absorb(set_name: str, meetings, emb: str = ABSORB) -> None:
    th, share = tune_absorb(emb)
    system = f"{emb}.absorb"
    ex = None
    for meeting, audio, ref, segments in meetings:
        if done(set_name, meeting, system):
            continue
        unit_boundary()
        ex = ex or extractor(emb)
        t = time.perf_counter()
        hyp = run(diarizer(emb, threshold=th), audio)
        hyp = diar_post.absorb(hyp, centroids(audio, hyp, ex), floor_sec(hyp, share))
        sec = time.perf_counter() - t
        result = {"meeting": meeting, "system": system, "exploratory": True, "embedding": emb,
                  "threshold": th, "share": share,
                  "audio_sec": round(len(audio) / RATE, 1), "process_sec": round(sec, 1),
                  **scored(ref, hyp, segments)}
        save(set_name, meeting, system, result, hyp)
        print(f"{set_name} {meeting} {system}: DER {result['collar25']['der']:.3f} "
              f"spk {result['hyp_speakers']}/{len({t.speaker for t in ref})} {sec:.0f}s", flush=True)


# --- B: thresholds retuned on dev meetings as long as the test ones -------------------
# The first tune used 5-minute dev meetings; on 30-50 minute meetings its thresholds left
# dozens of extra clusters, so .th ranks threshold transfer, not embeddings. Here each
# embedding's threshold (and the absorb share, in the same table) is chosen on 30-minute
# dev meetings; a best value on the edge of the grid widens it for that embedding.
RETUNE = {
    "campplus-zh-en": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "eres2netv2-zh": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "titanet-small": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "wespeaker-r34": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
}
GRID_STEP, GRID_LOW, GRID_HIGH = 0.1, 0.1, 1.5


def tune30_point(emb: str, th: float, dev, ex) -> dict[str, float]:
    """DER (collar 0.25) over the dev meetings at one threshold, for every absorb share."""
    saved = RESULTS / "tune30" / emb / f"{th:.1f}.json"
    if saved.exists():
        return json.loads(saved.read_text())
    unit_boundary()
    sd = diarizer(emb, threshold=th)
    tot, err = 0.0, dict.fromkeys(SHARES, 0.0)
    for _, audio, ref, _ in dev:
        hyp = run(sd, audio)
        cents = centroids(audio, hyp, ex)
        for n in SHARES:
            sc = der.score(ref, diar_post.absorb(hyp, cents, floor_sec(hyp, n)), collar=0.25)
            err[n] += sc.missed + sc.false_alarm + sc.confusion
        tot += sc.total
    out = {str(n): round(err[n] / tot, 4) for n in SHARES}
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(out, indent=1))
    print(f"tune30 {emb} {th:.1f}: " + " ".join(f"{n:.1%} {out[str(n)]:.3f}" for n in SHARES), flush=True)
    return out


def tune30(emb: str, joint: bool) -> tuple[float, float]:
    """The best threshold (with share 0, or the best share too when `joint`), widening the
    grid while the best sits on its edge."""
    dev = list(korean_meetings(DEV_SEEDS, ABSORB_DEV_MINUTES, split="dev"))
    ex = extractor(emb)
    grid = list(RETUNE[emb])
    while True:
        table = {th: tune30_point(emb, th, dev, ex) for th in grid}
        cells = {(th, float(n)): d for th, row in table.items() for n, d in row.items()
                 if joint or float(n) == 0}
        th, share = min(cells, key=cells.get)
        if th == grid[0] and th - GRID_STEP >= GRID_LOW - 1e-9:
            grid.insert(0, round(th - GRID_STEP, 1))
        elif th == grid[-1] and th + GRID_STEP <= GRID_HIGH + 1e-9:
            grid.append(round(th + GRID_STEP, 1))
        else:
            return th, share


def evaluate30(set_name: str, meetings, joint: bool) -> None:
    tag = "absorb30" if joint else "th30"
    best = {emb: tune30(emb, joint) for emb in RETUNE}
    (RESULTS / "tune30").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tune30" / f"best.{tag}.json").write_text(json.dumps(
        {e: {"threshold": th, "share": sh} for e, (th, sh) in best.items()}, indent=1))
    extractors = {}
    for meeting, audio, ref, segments in meetings:
        for emb, (th, share) in best.items():
            system = f"{emb}.{tag}"
            if done(set_name, meeting, system):
                continue
            unit_boundary()
            ex = extractors.get(emb) or extractors.setdefault(emb, extractor(emb))
            t = time.perf_counter()
            hyp = run(diarizer(emb, threshold=th), audio)
            if share:
                hyp = diar_post.absorb(hyp, centroids(audio, hyp, ex), floor_sec(hyp, share))
            sec = time.perf_counter() - t
            _, overlap = diar_pipe.ref_overlap(st, ref)
            result = {"meeting": meeting, "system": system, "embedding": emb, "threshold": th,
                      "share": share, "audio_sec": round(len(audio) / RATE, 1), "process_sec": round(sec, 1),
                      **scored(ref, hyp, segments)}
            save(set_name, meeting, system, result, hyp)
            print(f"{set_name} {meeting} {system}: DER {result['collar25']['der']:.3f} "
                  f"spk {result['hyp_speakers']}/{len({t.speaker for t in ref})} {sec:.0f}s", flush=True)


# --- embeddings once, clustering many ways (diar_pipe) --------------------------------
# sherpa's pipeline re-done around stored (chunk, speaker) embeddings: one extraction a
# meeting and embedding, then linkage x threshold x small-cluster floor in minutes. The
# re-done path must reproduce sherpa's own (verify) before the grid is trusted.
STORE = DATA / "emb"  # never committed
PIPE_EMBEDDINGS = list(RETUNE)
LINKAGES = ["complete", "average", "centroid"]
PIPE_THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(29)]  # 0.10 .. 1.50
# pyannote-style floor: a cluster with fewer embeddings than this joins the nearest big one
# a best on the threshold grid's lower edge (0.10) widens it for that model only
WIDEN = {"campplus-zh-en": [0.02, 0.04, 0.06, 0.08], "wespeaker-r34": [0.02, 0.04, 0.06, 0.08]}
MIN_COUNTS = [1, 2, 3, 5, 8, 12, 20]
MIN_SHARES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16]  # the absorb run picked 8%, its grid edge


def segmenter():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(SEGMENTATION), sess_options=opts, providers=["CPUExecutionProvider"])


def store_path(set_name: str, meeting: str, emb: str) -> Path:
    return STORE / set_name / f"{meeting}.{emb}.npz"


def extract_set(set_name: str, meetings) -> None:
    import diar_pipe

    seg = None
    for meeting, audio, _, _ in meetings:
        for emb in PIPE_EMBEDDINGS:
            path = store_path(set_name, meeting, emb)
            if path.exists():
                continue
            unit_boundary()
            seg = seg or segmenter()
            t = time.perf_counter()
            store = diar_pipe.extract(audio, seg, extractor(emb))
            sec = time.perf_counter() - t
            path.parent.mkdir(parents=True, exist_ok=True)
            store.save(path)
            path.with_suffix(".json").write_text(json.dumps(
                {"audio_sec": round(len(audio) / RATE, 1), "extract_sec": round(sec, 1),
                 "embeddings": int(len(store.embeddings)), "env": env(emb)}))
            print(f"extract {set_name} {meeting} {emb}: {len(store.embeddings)} embeddings {sec:.0f}s", flush=True)


def dev30():
    return korean_meetings(DEV_SEEDS, ABSORB_DEV_MINUTES, split="dev")


def verify(threshold: float = 0.7, n_meetings: int = 2) -> None:
    """The re-done path against sherpa's own, complete linkage, the same threshold."""
    import diar_pipe

    out = []
    for meeting, audio, ref, _ in list(dev30())[:n_meetings]:
        for emb in PIPE_EMBEDDINGS:
            unit_boundary()
            theirs = run(diarizer(emb, threshold=threshold), audio)
            ours, _ = diar_pipe.diarize(diar_pipe.Store.load(store_path("dev30", meeting, emb)), "complete", threshold)
            a, b = der.score(ref, theirs, collar=0.25), der.score(ref, ours, collar=0.25)
            same = der.score(theirs, ours)  # one against the other, no collar
            row = {"meeting": meeting, "embedding": emb, "threshold": threshold,
                   "der_sherpa": round(a.der, 4), "der_ours": round(b.der, 4),
                   "speakers_sherpa": a.hyp_speakers, "speakers_ours": b.hyp_speakers,
                   "disagreement": round(same.der, 4)}
            out.append(row)
            print(f"verify {meeting} {emb}: DER {row['der_sherpa']:.4f} vs {row['der_ours']:.4f}, "
                  f"speakers {row['speakers_sherpa']} vs {row['speakers_ours']}, "
                  f"disagree {row['disagreement']:.2%}", flush=True)
    (RESULTS / "verify.json").write_text(json.dumps({"rows": out, "env": {e: env(e) for e in PIPE_EMBEDDINGS}}, indent=1))


def floors(n_emb: int):
    yield "count", 1, 1
    for c in MIN_COUNTS[1:]:
        yield "count", c, c
    for sh in MIN_SHARES:
        yield "share", sh, max(1, math.ceil(sh * n_emb))


def grid(emb: str) -> dict:
    """DER (collar 0.25) over the dev meetings for every linkage, threshold and floor, with
    the best key over all dev meetings and over each scene's own dev meetings. Thresholds
    in WIDEN[emb] are added to a saved grid (a best on the grid's edge)."""
    import diar_pipe

    saved = RESULTS / "grid" / f"{emb}.json"
    old = json.loads(saved.read_text()) if saved.exists() else None
    thresholds = PIPE_THRESHOLDS
    if old is not None:
        thresholds = [t for t in WIDEN.get(emb, []) if t not in old.get("thresholds", PIPE_THRESHOLDS)]
        if not thresholds:
            return old
    dev = [(m, ref) for m, _, ref, _ in dev30()]
    per, inv = {}, {}  # per[key][meeting] = (error, total, hyp speakers, people erased, speakers vanished)
    for meeting, ref in dev:
        unit_boundary()
        st = diar_pipe.Store.load(store_path("dev30", meeting, emb))
        pairs = [tuple(p) for p in st.pairs.tolist()]
        _, overlap = diar_pipe.ref_overlap(st, ref)
        for method in LINKAGES:
            z = diar_pipe.linkage_matrix(st.embeddings, method)
            if old is None:
                inv.setdefault(method, []).append(diar_pipe.inversions(z))
            for th in thresholds:
                base = diar_pipe.cut(z, method, th)
                for kind, value, n in floors(len(base)):
                    labels = diar_pipe.reassign_small(st.embeddings, base, n) if n > 1 else base
                    hyp = diar_pipe.reconstruct(list(st.labels), pairs, labels, st.n_samples, diar_pipe.SEGMENTATION_3)
                    sc = der.score(ref, hyp, collar=0.25)
                    per.setdefault(f"{method}/{th}/{kind}/{value}", {})[meeting] = (
                        sc.missed + sc.false_alarm + sc.confusion, sc.total, len({t.speaker for t in hyp}),
                        diar_pipe.erased(overlap, base, n) if n > 1 else 0, diar_pipe.vanished(overlap, labels))
        print(f"grid {emb} {meeting} done", flush=True)
    true = {m: len({t.speaker for t in ref}) for m, ref in dev}
    scenes = {"all": [m for m, _ in dev]}
    for m, _ in dev:
        scenes.setdefault(m.split("-")[1], []).append(m)

    def summary(ms):
        out = {}
        for k, v in per.items():
            e, t = sum(v[m][0] for m in ms), sum(v[m][1] for m in ms)
            out[k] = {"der": round(e / t, 4), "speakers_over": round(sum(v[m][2] - true[m] for m in ms) / len(ms), 2),
                      "erased": sum(v[m][3] for m in ms), "vanished": sum(v[m][4] for m in ms)}
        return out

    out = {"dev_minutes": ABSORB_DEV_MINUTES, "exploratory": True,
           "inversions": old["inversions"] if old else inv, "scenes": scenes, "true_speakers": true,
           "thresholds": sorted(set(thresholds) | set(old.get("thresholds", PIPE_THRESHOLDS) if old else [])),
           "env": env(emb)}
    for scene, ms in scenes.items():
        table = {**(old[scene]["table"] if old else {}), **summary(ms)}
        der_of = lambda k: table[k]["der"]  # noqa: E731
        best = {m: min((k for k in table if k.startswith(m + "/")), key=der_of) for m in LINKAGES}
        out[scene] = {"table": table, "best": best,
                      "best_no_floor": {m: min((k for k in table if k.startswith(m + "/") and k.endswith("/count/1")),
                                               key=der_of) for m in LINKAGES},
                      "best_all": min(best.values(), key=der_of)}
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(out, indent=1))
    return out


PREREGISTERED = RESULTS / "preregistered.json"
# fixed from the dev grid before any test or AMI result was seen:
# centroid linkage has height inversions, so a centroid value stays in the table but is
# not an app candidate
APP_CANDIDATES = {"campplus-zh-en": "average/0.25/share/0.03"}


def preregister() -> dict:
    """Write the settings test and AMI will use, once: an existing file is never rewritten,
    so nothing can be added or changed after a test result is seen."""
    if PREREGISTERED.exists():
        return json.loads(PREREGISTERED.read_text())
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "rule": "headline = each model's common value; ranking by its AMI result. "
                   "scene values are secondary (AMI takes the room value). baseline = sherpa-like "
                   "(complete, no floor). Nothing is added or changed after test is seen.",
           "korean_test_limit": "clips are reused across meetings: Korean test may be optimistic",
           "models": {}}
    for emb in PIPE_EMBEDDINGS:
        g = grid(emb)
        settings = {"best_all": g["all"]["best_all"], "scene_room": g["room"]["best_all"],
                    "scene_call": g["call"]["best_all"], "sherpa": g["all"]["best_no_floor"]["complete"]}
        out["models"][emb] = {**settings, "grid_sha256": sha256(RESULTS / "grid" / f"{emb}.json"),
                              "app_candidates": [k for k in settings.values()
                                                 if k == APP_CANDIDATES.get(emb)],
                              "not_app_candidates": sorted({k for k in settings.values()
                                                            if k.startswith("centroid/")})}
    PREREGISTERED.write_text(json.dumps(out, indent=1))
    return out


def parse_key(key: str):
    method, th, kind, value = key.split("/")
    return method, float(th), kind, float(value)


def evaluate_pipe(set_name: str, meetings, which: str = "best_all") -> None:
    """Each embedding's dev choice applied, unchanged, to a test set's stored embeddings.
    `which`: "best_all" (one value for every meeting), "scene" (the value chosen on the
    meeting's own scene; AMI meetings are in a room) or "sherpa" (complete linkage, no floor)."""
    import diar_pipe

    if not PREREGISTERED.exists():
        raise SystemExit(f"run the preregister step first: {PREREGISTERED} is missing")
    pre = json.loads(PREREGISTERED.read_text())
    for meeting, _, ref, segments in meetings:
        scene = meeting.split("-")[1] if set_name == "korean" else "room"
        for emb in PIPE_EMBEDDINGS:
            p = pre["models"][emb]
            key = {"best_all": p["best_all"], "scene": p[f"scene_{scene}"], "sherpa": p["sherpa"]}[which]
            method, th, kind, value = parse_key(key)
            system = {"best_all": f"{emb}.pipe", "scene": f"{emb}.pipe-scene", "sherpa": f"{emb}.pipe-sherpa"}[which]
            if done(set_name, meeting, system):
                continue
            unit_boundary()
            path = store_path(set_name, meeting, emb)
            st = diar_pipe.Store.load(path)
            ext = json.loads(path.with_suffix(".json").read_text())
            t = time.perf_counter()
            base, inv = diar_pipe.cluster(st.embeddings, method, th)
            n = max(1, math.ceil(value * len(base))) if kind == "share" else int(value)
            labels = diar_pipe.reassign_small(st.embeddings, base, n) if n > 1 else base
            hyp = diar_pipe.reconstruct(list(st.labels), [tuple(p) for p in st.pairs.tolist()], labels,
                                        st.n_samples, diar_pipe.SEGMENTATION_3)
            sec = time.perf_counter() - t
            _, overlap = diar_pipe.ref_overlap(st, ref)
            result = {"meeting": meeting, "system": system, "embedding": emb, "preregistered": key,
                      "preregistered_at": pre["written"], "linkage": method,
                      "threshold": th, "floor": [kind, value], "inversions": inv,
                      "erased": diar_pipe.erased(overlap, base, n) if n > 1 else 0,
                      "vanished": diar_pipe.vanished(overlap, labels),
                      "embeddings": int(len(st.embeddings)), "audio_sec": ext["audio_sec"],
                      "extract_sec": ext["extract_sec"], "cluster_sec": round(sec, 1),
                      "process_sec": round(ext["extract_sec"] + sec, 1), **scored(ref, hyp, segments)}
            save(set_name, meeting, system, result, hyp)
            print(f"{set_name} {meeting} {system}: DER {result['collar25']['der']:.3f} "
                  f"spk {result['hyp_speakers']}/{len({t.speaker for t in ref})}", flush=True)


def long_meeting() -> None:
    """A 4-hour meeting, each embedding in its own process: wall time and peak memory."""
    path = DATA / "long-4h.wav"
    if not path.exists():
        m = diar_mix.mix(zeroth_pool(), speakers=4, minutes=240, seed=7, overlap=OVERLAP, short=SHORT,
                         scene="room")
        sf.write(path, m.audio, RATE)
        (DATA / "long-4h.rttm").write_text(der.to_rttm(m.turns, "long-4h"))
    best = thresholds()
    for emb in EMBEDDINGS:
        if done("long", "long-4h", f"{emb}.th"):
            continue
        unit_boundary()
        run_one = subprocess.run(
            [sys.executable, "-I", __file__, "_long_child", emb, str(best[emb])],
            capture_output=True, text=True,
        )
        print(run_one.stdout.strip() or run_one.stderr.strip()[-400:], flush=True)


def _long_child(emb: str, threshold: float) -> None:
    audio, _ = sf.read(DATA / "long-4h.wav", dtype="float32")
    ref = der.from_rttm((DATA / "long-4h.rttm").read_text())
    t = time.perf_counter()
    hyp = run(diarizer(emb, threshold=threshold), audio)
    sec = time.perf_counter() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
    result = {"meeting": "long-4h", "system": f"{emb}.th", "audio_sec": round(len(audio) / RATE, 1),
              "process_sec": round(sec, 1), "rtf": round(sec / (len(audio) / RATE), 4),
              "peak_mb": round(peak), **scored(ref, hyp, [(x.start, x.end) for x in ref])}
    save("long", "long-4h", f"{emb}.th", result, hyp)
    print(f"long {emb}: {sec:.0f}s RTF {result['rtf']:.3f} peak {result['peak_mb']} MB "
          f"DER {result['collar25']['der']:.3f}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_long_child":
        _long_child(sys.argv[2], float(sys.argv[3]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["tune", "korean", "ami", "online", "long", "absorb",
                                                 "tune30", "korean30", "ami30",
                                                 "extract-dev", "verify", "grid", "preregister", "extract-test", "pipe"])
    ap.add_argument("--joint", action="store_true", help="B: choose the absorb share with the threshold")
    args = ap.parse_args()
    try:
        steps(args.steps, args.joint)
    except Stopped:
        print(f"stopped: {STOP} is there; remove it and run the same step again to carry on")


def steps(names: list[str], joint: bool = False) -> None:
    for step in names:
        if step == "tune":
            print(tune())
        elif step == "korean":
            evaluate("korean", korean_meetings(TEST_SEEDS, TEST_MINUTES))
        elif step == "ami":
            evaluate("ami", ami_meetings())
        elif step == "online":
            evaluate_online()
        elif step == "long":
            long_meeting()
        elif step == "absorb":
            print(tune_absorb())
            evaluate_absorb("korean", korean_meetings(TEST_SEEDS, TEST_MINUTES))
            evaluate_absorb("ami", ami_meetings())
        elif step == "tune30":
            print({emb: tune30(emb, joint) for emb in RETUNE})
        elif step == "korean30":
            evaluate30("korean", korean_meetings(TEST_SEEDS, TEST_MINUTES), joint)
        elif step == "ami30":
            evaluate30("ami", ami_meetings(), joint)
        elif step == "extract-dev":
            extract_set("dev30", dev30())
        elif step == "verify":
            verify()
        elif step == "grid":
            for emb in PIPE_EMBEDDINGS:
                g = grid(emb)
                for scene in g["scenes"]:
                    b, t = g[scene]["best_all"], g[scene]["table"]
                    print(emb, scene, b, t[b], "| sherpa-like", t[g[scene]["best_no_floor"]["complete"]], flush=True)
        elif step == "preregister":
            print(json.dumps(preregister(), indent=1))
        elif step == "extract-test":
            extract_set("korean", korean_meetings(TEST_SEEDS, TEST_MINUTES))
            extract_set("ami", ami_meetings())
        elif step == "pipe":
            for which in ("best_all", "scene", "sherpa"):
                evaluate_pipe("korean", korean_meetings(TEST_SEEDS, TEST_MINUTES), which)
                evaluate_pipe("ami", ami_meetings(), which)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    main()
