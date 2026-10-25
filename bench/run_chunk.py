"""Diarization in pieces (diar_chunk) against whole-meeting clustering, on meetings with a
late joiner, a quiet speaker and four-hour sessions, and on calls as the app records them.

    bench/.venv/bin/python -I bench/run_chunk.py build-dev extract-dev grid preregister
    bench/.venv/bin/python -I bench/run_chunk.py build-test extract-test pipe ami long

Meetings (diar_mix, Zeroth-Korean): room and call, 2, 4 and 6 people, 30 minutes, with a
late joiner and/or a quiet speaker; and a 6-person room and call of four hours with a
late joiner, a quiet speaker and one who leaves early. dev uses 12 training speakers,
test the test set's 10, never the same people. A call is two tracks: "me" alone on the
microphone and everyone else on system audio; the microphone is taken whole as "me"
(diar_chunk.one_voice) and only system audio is diarized, as the app would. The one-track
mix of the same call is kept for comparison, and on test a microphone with system audio
leaking into it at -25 dB (a laptop on speakers), never used for choosing.

Audio is written as 16-bit WAV (the app's format) under bench/data/diar/chunk/ and read
from disk window by window. Nothing here is committed but the results.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import der  # noqa: E402
import diar_chunk as dc  # noqa: E402
import diar_mix  # noqa: E402
import diar_pipe as dp  # noqa: E402
import run_diar as rd  # noqa: E402

DATA = rd.DATA / "chunk"
RESULTS = rd.RESULTS / "chunk"
RAW = rd.RAW / "chunk"
RATE = rd.RATE
G = dp.SEGMENTATION_3
MODELS = ["campplus-zh-en", "eres2netv2-zh"]  # speed, and far-field accuracy (RESULTS §9)
OVERLAP, SHORT = rd.OVERLAP, rd.SHORT
BLEED_DB = -25.0
WORKERS = 6
WIDEN_WORKERS = 4

# the grid, chosen on dev only
PIECE_SEC = [300, 600, 900]
OVERLAP_SEC = [0, 60]
LOCAL = {"complete": [0.6, 0.7, 0.8, 0.9], "average": [0.15, 0.2, 0.25, 0.3]}
PIECE_SHARE = [0.0, 0.03, 0.08]
LINK = [0.3, 0.45, 0.6, 0.75]
CANNOT_LINK = [True, False]
FLOORS = [("share", 0.0), ("share", 0.01), ("share", 0.02), ("sec", 15.0), ("sec", 45.0)]
# one widening past the first grid's edge (its best had the lowest link, local threshold and
# piece share): the axes that were flat or unanimous in its top 1% are fixed or kept
WIDEN = {"piece_sec": [600, 900, 1200, 1800], "overlap_sec": OVERLAP_SEC, "method": "average",
         "threshold": [0.1, 0.15, 0.2], "piece_share": 0.0, "link_threshold": [0.15, 0.2, 0.25, 0.3],
         "cannot_link": False, "floors": FLOORS}

PREREGISTERED = RESULTS / "preregistered.json"


@dataclass(frozen=True)
class Spec:
    name: str
    scene: str
    speakers: int
    minutes: float
    seed: int
    roles: dict = field(default_factory=dict, hash=False)


def specs(split: str) -> list[Spec]:
    a, b = (300, 301) if split == "dev" else (400, 401)
    # quiet speakers at about 2% and 1% of the speech: the mixer lands near 0.75 of the
    # share asked (their short replies and the no-repeat rule), so ask for 3% and 1.5%;
    # build() records the share each one got
    out = []
    for scene in ("room", "call"):
        for n, seed, roles in ((2, a, {}), (4, a, {"late": 1}), (4, b, {"quiet": 1, "quiet_share": 0.03}),
                               (6, a, {"late": 1, "quiet": 1, "quiet_share": 0.03}),
                               (6, b, {"late": 1, "quiet": 1, "quiet_share": 0.015})):
            tag = "".join(k[0] for k in ("late", "quiet") if k in roles) or "plain"
            out.append(Spec(f"ck-{scene}-{n}p-30m-s{seed}-{tag}", scene, n, 30, seed, roles))
        out.append(Spec(f"ck-{scene}-6p-240m-s{a}-lqe", scene, 6, 240, a,
                        {"late": 1, "quiet": 1, "quiet_share": 0.015, "early": 1}))
    return out


def meeting_dir(split: str, name: str) -> Path:
    return DATA / split / name


def is_long(name: str) -> bool:
    return "-240m-" in name


# --- building the meetings -------------------------------------------------------------

def build(split: str) -> None:
    pool = rd.zeroth_pool("test" if split == "test" else "dev")
    if split == "dev":
        assert not set(pool) & set(rd.zeroth_pool("test")), "dev and test speakers overlap"
    for s in specs(split):
        d = meeting_dir(split, s.name)
        if (d / "meta.json").exists():
            continue
        rd.unit_boundary()
        t = time.perf_counter()
        call = s.scene == "call"
        roles, tries = dict(s.roles), []
        for _ in range(5):
            m = diar_mix.mix(pool, speakers=s.speakers, minutes=s.minutes, seed=s.seed, overlap=OVERLAP,
                             short=SHORT, scene=s.scene, channels=call, **roles)
            got = _quiet_share(m)
            tries.append({"asked": roles.get("quiet_share"), "got": got})
            if got is None or QUIET[0] <= got <= QUIET[1]:
                break
            # a few turns in 30 minutes: the share moves in steps, so aim for the middle again
            roles["quiet_share"] = roles["quiet_share"] * (sum(QUIET) / 2) / max(got, 1e-3)
        else:
            raise RuntimeError(f"{s.name}: quiet share {tries}")
        d.mkdir(parents=True, exist_ok=True)
        write = lambda name, x: sf.write(d / name, np.clip(x, -1, 1), RATE, subtype="PCM_16")  # noqa: E731
        if call:
            write("mic.wav", m.mic)
            write("system.wav", m.system)
            write("mix.wav", m.audio)
            if split == "test":
                b = diar_mix.mix(pool, speakers=s.speakers, minutes=s.minutes, seed=s.seed, overlap=OVERLAP,
                                 short=SHORT, scene=s.scene, channels=True, bleed_db=BLEED_DB, **roles)
                assert b.turns == m.turns
                write("mic-bleed.wav", b.mic)
        else:
            write("audio.wav", m.audio)
        (d / "ref.rttm").write_text(der.to_rttm(m.turns, s.name))
        speech = {}
        for x in m.turns:
            speech[x.speaker] = speech.get(x.speaker, 0.0) + x.end - x.start
        total = sum(speech.values())
        meta = {"spec": asdict(s), "roles": m.roles, "me": m.me, "quiet_tries": tries,
                "share": {k: round(v / total, 4) for k, v in sorted(speech.items())},
                "first": {k: min(x.start for x in m.turns if x.speaker == k) for k in sorted(speech)},
                "last": {k: max(x.end for x in m.turns if x.speaker == k) for k in sorted(speech)},
                "audio_sec": round(len(m.audio) / RATE, 1)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"build {split} {s.name}: {meta['audio_sec'] / 60:.0f} min, roles {m.roles}, "
              f"shares {[meta['share'].get(k, 0.0) for k in m.roles]} {time.perf_counter() - t:.0f}s", flush=True)


QUIET = (0.01, 0.03)  # a quiet speaker's share of the speech, as the plan asked


def _quiet_share(m) -> float | None:
    who = [k for k, r in m.roles.items() if r == "quiet"]
    if not who:
        return None
    total = sum(t.end - t.start for t in m.turns)
    return round(sum(t.end - t.start for t in m.turns if t.speaker == who[0]) / total, 4)


class OnDisk:
    """A WAV read slice by slice (diar_pipe.extract takes it in place of an array)."""

    def __init__(self, path: Path):
        self.path = path
        self.f = sf.SoundFile(path)
        assert self.f.samplerate == RATE, path

    def __len__(self) -> int:
        return self.f.frames

    def __getitem__(self, s: slice) -> np.ndarray:
        a, b, _ = s.indices(len(self))
        self.f.seek(a)
        return self.f.read(max(0, b - a), dtype="float32")


def tracks(split: str, name: str) -> dict[str, str]:
    """track -> what it is diarized for: room -> audio; call -> system and the microphone's
    voice; on test also the one-track mix and the leaking microphone, for comparison only."""
    if name.startswith("ck-room"):
        return {"audio": "embed"}
    out = {"system": "embed", "mic": "voice"}
    if split == "test":
        out.update({"mix": "embed", "mic-bleed": "voice"})
    return out


def store_path(split: str, name: str, track: str, emb: str | None) -> Path:
    return DATA / split / name / (f"{track}.{emb}.npz" if emb else f"{track}.labels.npz")


def extract(split: str) -> None:
    seg = None
    for s in specs(split):
        for track, kind in tracks(split, s.name).items():
            wav = meeting_dir(split, s.name) / f"{track}.wav"
            for emb in (MODELS if kind == "embed" else [None]):
                path = store_path(split, s.name, track, emb)
                if path.exists():
                    continue
                rd.unit_boundary()
                seg = seg or rd.segmenter()
                t = time.perf_counter()
                audio = OnDisk(wav)
                if emb:
                    st = dp.extract(audio, seg, rd.extractor(emb))
                    st.save(path)
                    n = len(st.embeddings)
                else:
                    labels = np.stack(dp.segment(audio, seg)).astype(np.int8)
                    np.savez_compressed(path, labels=labels, n_samples=len(audio))
                    n = 0
                sec = time.perf_counter() - t
                path.with_suffix(".json").write_text(json.dumps(
                    {"audio_sec": round(len(audio) / RATE, 1), "extract_sec": round(sec, 1), "embeddings": n,
                     "env": rd.env(emb)}))
                print(f"extract {split} {s.name} {track} {emb or 'segmentation'}: {n} embeddings {sec:.0f}s",
                      flush=True)


# --- scoring -------------------------------------------------------------------------

def load_meeting(split: str, name: str):
    d = meeting_dir(split, name)
    return der.from_rttm((d / "ref.rttm").read_text()), json.loads((d / "meta.json").read_text())


def vanished_who(names: list[str], overlap: np.ndarray, labels: np.ndarray) -> set[str]:
    leaders = {int(np.argmax(overlap[labels == c].sum(axis=0))) for c in np.unique(labels)}
    return {n for i, n in enumerate(names) if i not in leaders}


class Meeting:
    """What a config is scored on: the store it clusters, and what joins its result."""

    def __init__(self, split: str, name: str, emb: str, track: str | None = None, mic: str = "mic",
                 echo: bool = False):
        self.split, self.name, self.emb = split, name, emb
        self.ref, self.meta = load_meeting(split, name)
        self.call = name.startswith("ck-call")
        self.track = track or ("system" if self.call else "audio")
        self.store = dp.Store.load(store_path(split, name, self.track, emb))
        me = self.meta["me"]
        # on the system track, "me" is not there to be found
        self.target = [t for t in self.ref if not (self.call and self.track == "system" and t.speaker == me)]
        self.names, self.overlap = dp.ref_overlap(self.store, self.target)
        self.voice = []
        if self.call and self.track == "system":
            v = np.load(store_path(split, name, mic, None))
            self.voice = dc.one_voice(v["labels"], int(v["n_samples"]), G, "me")
            if echo:
                d = meeting_dir(split, name)
                self.voice = dc.drop_echo(self.voice, OnDisk(d / f"{mic}.wav"), OnDisk(d / "system.wav"), RATE)
        self.special = set(self.meta["roles"])
        self.cache: dict = {}

    def hyp(self, labels: np.ndarray) -> list[der.Turn]:
        pairs = [tuple(p) for p in self.store.pairs.tolist()]
        turns = dp.reconstruct(list(self.store.labels), pairs, labels, self.store.n_samples, G)
        return sorted(self.voice + [der.Turn(f"s{t.speaker}", t.start, t.end) for t in turns],
                      key=lambda t: (t.start, t.speaker))

    def score(self, labels: np.ndarray) -> dict:
        hyp = self.hyp(labels)
        s = der.score(self.ref, hyp, collar=0.25)
        gone = vanished_who(self.names, self.overlap, labels)
        return {"error": s.missed + s.false_alarm + s.confusion, "total": s.total, "der": s.der,
                "speakers": len({t.speaker for t in hyp}), "true": len({t.speaker for t in self.ref}),
                "vanished": len(gone), "special_vanished": len(gone & self.special)}


def widened():
    w = WIDEN
    for p, o, th, link, (fk, fv) in itertools.product(w["piece_sec"], w["overlap_sec"], w["threshold"],
                                                      w["link_threshold"], w["floors"]):
        yield dc.Config(piece_sec=p, overlap_sec=o, method=w["method"], threshold=th,
                        piece_share=w["piece_share"], link_threshold=link, cannot_link=w["cannot_link"],
                        floor_kind=fk, floor=fv)


def configs():
    for p, o, (m, ths), share, link, cl, (fk, fv) in itertools.product(
            PIECE_SEC, OVERLAP_SEC, LOCAL.items(), PIECE_SHARE, LINK, CANNOT_LINK, FLOORS):
        for th in ths:
            yield dc.Config(piece_sec=p, overlap_sec=o, method=m, threshold=th, piece_share=share,
                            link_threshold=link, cannot_link=cl, floor_kind=fk, floor=fv)


def key(c: dc.Config) -> str:
    return (f"{c.piece_sec:g}s/{c.overlap_sec:g}/{c.method}/{c.threshold:g}/{c.piece_share:g}/"
            f"{c.link_threshold:g}/{'cl' if c.cannot_link else 'free'}/{c.floor_kind}/{c.floor:g}")


def config_of(k: str) -> dc.Config:
    p, o, m, th, share, link, cl, fk, fv = k.split("/")
    return dc.Config(piece_sec=float(p[:-1]), overlap_sec=float(o), method=m, threshold=float(th),
                     piece_share=float(share), link_threshold=float(link), cannot_link=cl == "cl",
                     floor_kind=fk, floor=float(fv))


def _grid_one(args) -> tuple[str, dict]:
    name, emb, which = args
    m = Meeting("dev", name, emb)
    # configs that share a piece clustering run one after another, so the cache stays small
    out = {}
    for c in sorted(widened() if which == "widen" else configs(), key=lambda c: (c.piece_sec, c.overlap_sec, c.method, c.threshold, c.piece_share)):
        if len(m.cache) > 1:
            m.cache.pop(next(iter(m.cache)))
        out[key(c)] = m.score(dc.diarize_labels(m.store, G, c, m.cache))
    return name, out


def summary(per: dict[str, dict], names: list[str]) -> dict:
    """Mean DER over the meetings (each meeting counts once), vanished summed."""
    v = [per[n] for n in names]
    return {"der": round(float(np.mean([x["der"] for x in v])), 4),
            "speakers_over": round(float(np.mean([x["speakers"] - x["true"] for x in v])), 2),
            "vanished": sum(x["vanished"] for x in v), "special_vanished": sum(x["special_vanished"] for x in v)}


AXES = ["piece_sec", "overlap_sec", "method", "threshold", "piece_share", "link_threshold", "cannot_link",
        "floor"]


def top_share(table: dict, share: float = 0.01) -> dict:
    """The best `share` of the grid by score: its score range and how each axis spreads in
    it. An axis spread wide across the top matters little."""
    ranked = sorted(table, key=lambda k: table[k]["score"])
    top = ranked[:max(1, math.ceil(share * len(ranked)))]
    spread = {}
    for ax in AXES:
        vals = []
        for k in top:
            c = config_of(k)
            vals.append(f"{c.floor_kind}/{c.floor:g}" if ax == "floor" else str(getattr(c, ax)))
        spread[ax] = {v: vals.count(v) for v in sorted(set(vals))}
    return {"n": len(top), "score": [table[top[0]]["score"], table[top[-1]]["score"]], "axes": spread}


def cross_seed(per_meeting: dict, short: list[str]) -> dict:
    """Choose on one seed's 30-minute dev meetings and score on the other's, both ways: how
    optimistic a dev best is, without touching test. The 4-hour dev has one seed only."""
    seeds = sorted({n.split("-s")[-1].split("-")[0] for n in short})
    out = {}
    for pick in seeds:
        a = [n for n in short if f"-s{pick}-" in n]
        b = [n for n in short if n not in a]
        keys = per_meeting[a[0]].keys()
        mean = lambda ms, k: float(np.mean([per_meeting[n][k]["der"] for n in ms]))  # noqa: E731
        k = min(keys, key=lambda k: mean(a, k))
        kb = min(keys, key=lambda k: mean(b, k))
        out[f"choose_s{pick}"] = {"key": k, "der_on_own": round(mean(a, k), 4), "der_on_other": round(mean(b, k), 4),
                                  "other_best": round(mean(b, kb), 4)}
    return out


def grid() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    names = [s.name for s in specs("dev")]
    short = [n for n in names if not is_long(n)]
    long_ = [n for n in names if is_long(n)]
    for emb in MODELS:
        path = RESULTS / "grid" / f"{emb}.json"
        if path.exists():
            continue
        rd.unit_boundary()
        t = time.perf_counter()
        with ProcessPoolExecutor(WORKERS) as pool:
            per_meeting = dict(pool.map(_grid_one, [(n, emb, "first") for n in names]))
        table = _table(per_meeting, names, short, long_)
        best = min(table, key=lambda k: table[k]["score"])
        best30 = min(table, key=lambda k: table[k]["30m"]["der"])
        best240 = min(table, key=lambda k: table[k]["240m"]["der"])
        out = {"dev": names, "rule": "score = (mean DER of the 30-minute dev + mean DER of the 4-hour dev) / 2",
               "best": best, "best_30m": best30, "best_240m": best240,
               "top1pct": top_share(table), "cross_seed": cross_seed(per_meeting, short),
               "per_meeting_best": {n: per_meeting[n][best] for n in names},
               "table": table, "grid_sec": round(time.perf_counter() - t), "env": rd.env(emb)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        print(f"grid {emb}: top 1% {out['top1pct']['score']}, cross-seed {out['cross_seed']}", flush=True)
        print(f"grid {emb}: best {best} {table[best]}\n  30m-best {best30} {table[best30]['30m']} "
              f"(on 4h {table[best30]['240m']['der']})\n  4h-best {best240} {table[best240]['240m']} "
              f"(on 30m {table[best240]['30m']['der']})", flush=True)


def _table(per_meeting: dict, names: list[str], short: list[str], long_: list[str]) -> dict:
    table = {}
    for k in per_meeting[names[0]]:
        per = {n: per_meeting[n][k] for n in names}
        s30, s240 = summary(per, short), summary(per, long_)
        table[k] = {"score": round((s30["der"] + s240["der"]) / 2, 4), "30m": s30, "240m": s240,
                    "all": summary(per, names)}
    return table


def widen_path(emb: str) -> Path:
    return RESULTS / "grid" / f"{emb}.widen.json"


def widen() -> None:
    """The one widening (WIDEN), written beside the first grid, which stays as it was. The
    configs both grids hold must score the same in both; best and top 1% are over the union,
    cross-seed over the widened configs (the first grid keeps no per-meeting scores)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    names = [s.name for s in specs("dev")]
    short = [n for n in names if not is_long(n)]
    long_ = [n for n in names if is_long(n)]
    for emb in MODELS:
        path = widen_path(emb)
        if path.exists():
            continue
        rd.unit_boundary()
        first = json.loads((RESULTS / "grid" / f"{emb}.json").read_text())
        t = time.perf_counter()
        # the widened configs peak at 4.5 GB on a 4-hour meeting, and both may run at once
        with ProcessPoolExecutor(WIDEN_WORKERS) as pool:
            per_meeting = dict(pool.map(_grid_one, [(n, emb, "widen") for n in names]))
        table = _table(per_meeting, names, short, long_)
        both = sorted(set(table) & set(first["table"]))
        differ = [k for k in both if table[k] != first["table"][k]]
        if differ:
            raise RuntimeError(f"{emb}: {len(differ)} configs score differently from the first grid: {differ[:3]}")
        union = {**first["table"], **table}
        best = min(union, key=lambda k: union[k]["score"])
        edge = {"link_threshold": min(WIDEN["link_threshold"]), "threshold": min(WIDEN["threshold"]),
                "piece_sec": max(WIDEN["piece_sec"])}
        c = config_of(best)
        out = {"widen": {k: v for k, v in WIDEN.items() if k != "floors"}, "first_grid_sha256": rd.sha256(
                   RESULTS / "grid" / f"{emb}.json"),
               "checked_same": len(both), "new": len(table) - len(both), "union": len(union),
               "best": best, "best_first": first["best"],
               "best_on_new_edge": [a for a, v in edge.items() if getattr(c, a) == v],
               "best_30m": min(union, key=lambda k: union[k]["30m"]["der"]),
               "best_240m": min(union, key=lambda k: union[k]["240m"]["der"]),
               "top1pct": top_share(union), "cross_seed": cross_seed(per_meeting, short),
               "per_meeting_best": {n: per_meeting[n][best] for n in names} if best in table else None,
               "table": table, "grid_sec": round(time.perf_counter() - t), "env": rd.env(emb)}
        path.write_text(json.dumps(out, indent=1))
        print(f"widen {emb}: {len(both)} configs as before, {out['new']} new; best {best} {union[best]} "
              f"(first {first['best']} {first['table'][first['best']]['score']}); on the new edge "
              f"{out['best_on_new_edge']}; top 1% {out['top1pct']['score']}; cross-seed {out['cross_seed']}",
              flush=True)


# --- preregistration and test ----------------------------------------------------------

def preregister() -> dict:
    if PREREGISTERED.exists():
        return json.loads(PREREGISTERED.read_text())
    old = json.loads(rd.PREREGISTERED.read_text())["models"]
    models = {}
    for emb in MODELS:
        first, w = RESULTS / "grid" / f"{emb}.json", widen_path(emb)
        ext = json.loads(w.read_text())
        union = {**json.loads(first.read_text())["table"], **ext["table"]}
        models[emb] = {"chunked": ext["best"], "dev_score": union[ext["best"]],
                       "dev_top1pct": ext["top1pct"], "best_on_grid_edge": ext["best_on_new_edge"],
                       "dev_cross_seed": ext["cross_seed"], "baseline_whole_meeting": old[emb]["best_all"],
                       "grid_sha256": rd.sha256(first), "widen_sha256": rd.sha256(w)}
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": models,
           "choice": "the lowest dev score (mean of the 30-minute and the 4-hour dev DER) over the first grid "
                     "and its one widening; a best on the widened grid's edge is kept, not chased",
           "ranking": "primary: mean of AMI Array1-01 DER and the new Korean test room DER (collar 0.25, "
                      "mean over meetings, the chunked setting); secondary: the new Korean test call DER "
                      "(two channels, no bleed). If the primary differs by less than 1 point, the model "
                      "with the shorter 4-hour diarization time is chosen.",
           "gate": "on the same Korean test, the chunked setting must leave fewer special speakers "
                   "(late joiner, quiet speaker, early leaver) vanished than the whole-meeting baseline, or "
                   "as few when the baseline leaves none, and no more speakers vanished in all. Only meetings "
                   "the baseline ran on count: if it cannot run on a 4-hour test meeting (memory), that "
                   "meeting's special speakers vanished are reported apart, as numbers. Failing it is a "
                   "failure of the design.",
           "caps": "4-hour meeting, one process from audio on disk to turns (the long step, the preregistered "
                   "setting only): peak RSS <= 1.0 GB with the models; CAM++ diarization <= 15 min (ERes2NetV2 "
                   "recorded only). Not applied to grid workers, which hold many settings and meetings. "
                   "Diarization is assumed to run after the final pass, so its memory does not add to whisper's.",
           "bleed": "test only (-25 dB system audio in the microphone), recorded, never used for choosing",
           "limits": "Korean meetings reuse clips (12 dev, 10 test speakers); the 4-hour dev reuses the most. "
                     "The microphone of a call is taken whole as 'me': wrong in a hybrid meeting. "
                     "The room and call meetings of one spec are the same mix (same seed, turns and quiet "
                     "share; the dev 4p-q is at 1.12% in both): not independent samples of a quiet speaker."}
    PREREGISTERED.write_text(json.dumps(out, indent=1))
    return out


def _baseline_labels(store: dp.Store, k: str) -> np.ndarray:
    method, th, kind, value = rd.parse_key(k)
    base, _ = dp.cluster(store.embeddings, method, th)
    n = max(1, math.ceil(value * len(base))) if kind == "share" else int(value)
    return dp.reassign_small(store.embeddings, base, n) if n > 1 else base


def _save(set_name: str, meeting: str, system: str, result: dict, hyp) -> None:
    (RESULTS / set_name).mkdir(parents=True, exist_ok=True)
    (RAW / set_name).mkdir(parents=True, exist_ok=True)
    (RESULTS / set_name / f"{meeting}.{system}.json").write_text(json.dumps(result, indent=1))
    (RAW / set_name / f"{meeting}.{system}.rttm").write_text(der.to_rttm(hyp, meeting))


def _result(m: Meeting, labels, system: str, k: str, pre: dict, sec: float):
    hyp = m.hyp(labels)
    gone = vanished_who(m.names, m.overlap, labels)
    return {"meeting": m.name, "system": system, "embedding": m.emb, "track": m.track, "preregistered": k,
            "preregistered_at": pre["written"], "roles": m.meta["roles"], "vanished": len(gone),
            "vanished_who": sorted(gone), "special_vanished": sorted(gone & m.special),
            "cluster_sec": round(sec, 1), **rd.scored(m.ref, hyp, [(t.start, t.end) for t in m.ref]),
            "env": rd.env(m.emb)}, hyp


def pipe() -> None:
    if not PREREGISTERED.exists():
        raise SystemExit("run the preregister step first")
    pre = json.loads(PREREGISTERED.read_text())
    for s in specs("test"):
        for emb in MODELS:
            p = pre["models"][emb]
            runs = [("chunk", None, "mic"), ("whole", None, "mic")]
            if s.scene == "call":
                runs += [("chunk-bleed", None, "mic-bleed"), ("chunk-bleed-echo", None, "mic-bleed"),
                         ("chunk-onetrack", "mix", "mic")]
            for system, track, mic in runs:
                name = f"{emb}.{system}"
                if (RESULTS / "test" / f"{s.name}.{name}.json").exists():
                    continue
                rd.unit_boundary()
                m = Meeting("test", s.name, emb, track=track, mic=mic, echo=system.endswith("-echo"))
                k = p["baseline_whole_meeting"] if system == "whole" else p["chunked"]
                t = time.perf_counter()
                try:
                    labels = _baseline_labels(m.store, k) if system == "whole" else \
                        dc.diarize_labels(m.store, G, config_of(k))
                except MemoryError:
                    print(f"test {s.name} {name}: out of memory, left out", flush=True)
                    continue
                result, hyp = _result(m, labels, name, k, pre, time.perf_counter() - t)
                _save("test", s.name, name, result, hyp)
                print(f"test {s.name} {name}: DER {result['collar25']['der']:.3f} spk {result['hyp_speakers']}/"
                      f"{result['ref_speakers']} vanished {result['vanished_who']}", flush=True)


def ami() -> None:
    pre = json.loads(PREREGISTERED.read_text())
    for meeting, _, ref, segments in rd.ami_meetings():
        for emb in MODELS:
            name = f"{emb}.chunk"
            if (RESULTS / "ami" / f"{meeting}.{name}.json").exists():
                continue
            rd.unit_boundary()
            st = dp.Store.load(rd.store_path("ami", meeting, emb))
            k = pre["models"][emb]["chunked"]
            t = time.perf_counter()
            labels = dc.diarize_labels(st, G, config_of(k))
            sec = time.perf_counter() - t
            names, overlap = dp.ref_overlap(st, ref)
            hyp = dp.reconstruct(list(st.labels), [tuple(x) for x in st.pairs.tolist()], labels, st.n_samples, G)
            gone = vanished_who(names, overlap, labels)
            result = {"meeting": meeting, "system": name, "embedding": emb, "preregistered": k,
                      "preregistered_at": pre["written"], "vanished": len(gone), "cluster_sec": round(sec, 1),
                      **rd.scored(ref, hyp, segments), "env": rd.env(emb)}
            _save("ami", meeting, name, result, hyp)
            print(f"ami {meeting} {name}: DER {result['collar25']['der']:.3f} vanished {len(gone)}", flush=True)


# --- four hours in one process ------------------------------------------------------------

def _long_child(name: str, emb: str, set_name: str = "long") -> None:
    """From the WAVs on disk to turns, as the app would run it after the final pass."""
    pre = json.loads(PREREGISTERED.read_text())
    k = pre["models"][emb]["chunked"]
    d = meeting_dir("test", name)
    call = name.startswith("ck-call")
    t = time.perf_counter()
    seg = rd.segmenter()
    st = dp.extract(OnDisk(d / ("system.wav" if call else "audio.wav")), seg, rd.extractor(emb))
    t_extract = time.perf_counter() - t
    # the process's peak after each stage (it only grows), and what each stage itself holds
    # at most (tracemalloc sees numpy's buffers)
    stages = {"extract": {"sec": round(t_extract, 1), "peak_mb_after": _peak_mb()}}
    labels = _stage(stages, "cluster", lambda: dc.diarize_labels(st, G, config_of(k)))
    turns = _stage(stages, "reconstruct", lambda: dp.reconstruct(
        list(st.labels), [tuple(p) for p in st.pairs.tolist()], labels, st.n_samples, G))
    hyp = [der.Turn(f"s{x.speaker}", x.start, x.end) for x in turns]
    if call:
        mic = OnDisk(d / "mic.wav")
        hyp += _stage(stages, "mic", lambda: dc.one_voice(np.stack(dp.segment(mic, seg)).astype(np.int8),
                                                          len(mic), G, "me"))
    sec = time.perf_counter() - t  # with tracemalloc's overhead in it: against the cap, on the safe side
    peak = _peak_mb()
    ref, _ = load_meeting("test", name)
    audio_sec = len(OnDisk(d / ("system.wav" if call else "audio.wav"))) / RATE
    result = {"meeting": name, "system": f"{emb}.chunk-process", "embedding": emb, "preregistered": k,
              "audio_sec": round(audio_sec, 1), "extract_sec": round(t_extract, 1), "process_sec": round(sec, 1),
              "rtf": round(sec / audio_sec, 4), "peak_mb": round(peak), "embeddings": int(len(st.embeddings)),
              "clusters": int(len(np.unique(labels))), "stages": stages,
              **rd.scored(ref, hyp, [(x.start, x.end) for x in ref]), "env": rd.env(emb)}
    if set_name in POSTHOC:
        result["posthoc"] = POSTHOC[set_name]
    _save(set_name, name, f"{emb}.chunk-process", result, hyp)
    print(f"long {name} {emb}: {sec / 60:.1f} min, peak {result['peak_mb']} MB, DER {result['collar25']['der']:.3f}",
          flush=True)


def _peak_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)  # bytes on macOS


def _stage(stages: dict, name: str, run):
    """Run one stage with tracemalloc on: its own peak, apart from what came before it."""
    import tracemalloc

    t = time.perf_counter()
    tracemalloc.start()
    out = run()
    _, top = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    sec = time.perf_counter() - t
    stages[name] = {"sec": round(sec, 1), "traced_peak_mb": round(top / 2**20), "peak_mb_after": _peak_mb()}
    return out


# re-runs of the long step after a fix that leaves every label as it was; never preregistered
POSTHOC = {"long-fixmem": "post hoc, not preregistered: the global link's condensed distances (86b3c1d), labels "
                          "bit-identical on all 88 test stores; the preregistered measurement is in long/"}


def long_runs(set_name: str = "long") -> None:
    for s in specs("test"):
        if not is_long(s.name):
            continue
        for emb in MODELS:
            if (RESULTS / set_name / f"{s.name}.{emb}.chunk-process.json").exists():
                continue
            rd.unit_boundary()
            r = subprocess.run([sys.executable, "-I", __file__, "_long_child", s.name, emb, set_name],
                               capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr[-2000:], flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_long_child":
        os.environ.setdefault("OMP_NUM_THREADS", str(rd.THREADS))
        _long_child(*sys.argv[2:5])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["build-dev", "extract-dev", "grid", "widen", "preregister",
                                                 "build-test", "extract-test", "pipe", "ami", "long", "long-fixmem"])
    args = ap.parse_args()
    try:
        for step in args.steps:
            if step == "build-dev":
                build("dev")
            elif step == "extract-dev":
                extract("dev")
            elif step == "grid":
                grid()
            elif step == "widen":
                widen()
            elif step == "preregister":
                print(json.dumps(preregister(), indent=1))
            elif step == "build-test":
                build("test")
            elif step == "extract-test":
                extract("test")
            elif step == "pipe":
                pipe()
            elif step == "ami":
                ami()
            elif step == "long":
                long_runs()
            elif step == "long-fixmem":
                long_runs("long-fixmem")
    except rd.Stopped:
        print(f"stopped: {rd.STOP} is there; remove it and run the same step again to carry on")


if __name__ == "__main__":
    main()
