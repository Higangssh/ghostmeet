"""Collect bench/results/*.json into bench/results/RESULTS.md (numbers only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS  # noqa: E402


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def lat(r: dict) -> str:
    if "caption_latency_median_s" not in r:
        return "—"
    return f"{r['caption_latency_median_s']:.1f} / {r['caption_latency_p95_s']:.1f}s"


def main() -> None:
    utt = sorted(load("*.utt.json"), key=lambda r: r["cer_no_digits"])
    cold = {r["spec"]: r for r in load("*.coldstart.json")}
    stream = load("*.stream.json")
    if not utt:
        sys.exit("no results yet")
    by_spec = {r["spec"]: r for r in utt}
    m = utt[0]["machine"]
    lines = [
        "# 한국어 STT 엔진 비교 — 측정 결과",
        "",
        f"기기: {m['cpu']} ({m['perf_cores']}P+{m['eff_cores']}E), RAM {m['ram_gb']}GB, {m['os']}, "
        f"Python {m['python']}. 모든 엔진 스레드 4개. 측정 중 다른 무거운 작업 없음.",
        "",
        f"데이터: FLEURS ko_kr test, {utt[0]['utterances']}문장 / {utt[0]['audio_min']}분 "
        "(google/fleurs, CC BY 4.0). 실제 회의 녹음은 쓰지 않았다.",
        "",
        "## 1. 문장 단위 (엔진 정확도·속도)",
        "",
        "| 엔진 | CER(숫자 없는 문장) | CER(전체) | RTF | 로드 | 첫 결과 | 문장당 지연 중앙/p95 | 최대 메모리 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in utt:
        c = cold.get(r["spec"], r)  # cached-model cold start, when measured separately
        mark = "¹" if r["spec"] in cold else ""
        lines.append(
            f"| {r['engine']} | **{pct(r['cer_no_digits'])}** | {pct(r['cer'])} | {r['rtf']:.3f} | "
            f"{c['load_s']:.1f}s{mark} | {c['first_result_s']:.1f}s{mark} | "
            f"{r['utt_latency_median_s']:.2f} / {r['utt_latency_p95_s']:.2f}s | {r['peak_footprint_mb']:.0f}MB |"
        )
    lines += [
        "",
        "- CER: 공백·문장부호를 지운 뒤 글자 편집거리 / 정답 글자 수(코퍼스 합산). 정답의 괄호 안 "
        "로마자 병기(`비슈케크(Bishkek)`)는 읽히지 않으므로 지웠다. 한글이 든 괄호는 읽으므로 남겼다.",
        "- 숫자 없는 문장 CER: 정답에 숫자가 있는 문장을 뺀 값. `2명`/`두 명` 같은 표기 차이가 엔진 "
        "비교를 흐리지 않게 한다. **엔진 비교는 이 열로 본다.**",
        "- RTF = 처리 시간 / 음성 길이. 첫 결과 = 엔진을 만들기 시작해서 첫 문장"
        f"({utt[0]['first_result_audio_s']:.0f}초) 결과가 나올 때까지(모델은 이미 디스크에 있음).",
        "- 최대 메모리: macOS phys_footprint 최대치(활성 상태 보기의 '메모리'). Metal 할당 포함. "
        "whisper.cpp는 whisper-server 프로세스 기준.",
    ]
    if cold:
        lines.append(
            "- ¹ 전체 측정 때는 첫 실행이라 모델 다운로드가 로드에 섞였다. 그래서 모델을 캐시한 상태에서 "
            "5문장으로 따로 잰 값을 넣었다."
        )

    if stream:
        lines += [
            "",
            "## 2. 긴 세션, 꼬리 창 (backend.incremental.IncrementalTranscriber 그대로)",
            "",
            f"FLEURS 문장을 이어 붙인 {stream[0]['session_min']}분 세션, 창 30초. "
            "`수정 전` = 측정 당시 main(창 경계 중복 버그 포함), `수정 후` = 창 경계 보류 수정.",
            "",
            "| 엔진 | 코드 | 조건 | 창당 처리 처음10분 → 마지막10분 | p95 | 기울기(초/시간) | 메모리 처음 → 마지막 | 자막 지연 중앙/p95 | CER |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        order = {"before": 0, "after": 1}
        for r in sorted(stream, key=lambda r: (r["spec"], r["mode"], order.get(r.get("label"), 2))):
            cond = "backlog(30초 창)" if r["mode"] == "backlog" else f"live({r['interval_s']:.0f}초 주기)"
            code = {"before": "수정 전", "after": "수정 후"}.get(r.get("label"), r.get("label", "?"))
            lines.append(
                f"| {r['engine']} | {code} | {cond} | {r['pass_sec_first10min']:.2f}s → {r['pass_sec_last10min']:.2f}s | "
                f"{r['pass_sec_p95']:.2f}s | {r['pass_sec_slope_per_hour']:+.3f} | "
                f"{r['footprint_mb_first10min']:.0f} → {r['footprint_mb_last10min']:.0f}MB | {lat(r)} | {pct(r['cer'])} |"
            )
        lines += [
            "",
            "- backlog: 음성이 이미 다 있어 매 회차가 30초 창을 꽉 채운다. 창당 비용이 세션 위치와 "
            "무관한지 보는 조건.",
            "- live: 가상 시계로 실시간 유입을 흉내 낸다. 직전 회차가 끝나고 10초 뒤에 다음 회차를 "
            "돈다(파이프라인 워커와 같음). 자막 지연 = 회차가 끝난 시각 − 세그먼트 끝 시각.",
            "- 기울기: 창당 처리 시간을 세션 위치에 대해 1차 회귀한 값. 0에 가까우면 일정하다는 뜻이다.",
        ]

        live = {(r["spec"], r.get("label")): r for r in stream if r["mode"] == "live"}
        specs = sorted({s for s, _ in live})
        if any((s, "after") in live for s in specs):
            lines += [
                "",
                "## 3. 창 경계 수정 전후 (live 10초 주기)",
                "",
                "완료 기준: live CER ≤ 문장 단위 CER(전체) + 2.0%p.",
                "",
                "| 엔진 | 문장 단위 CER | live CER 전 → 후 | 차이(후 − 문장) | 기준 | 자막 지연 중앙/p95 전 → 후 |",
                "|---|---:|---:|---:|---|---:|",
            ]
            for s in specs:
                b, a, u = live.get((s, "before")), live.get((s, "after")), by_spec.get(s)
                if not (b and a and u):
                    continue
                gap = a["cer"] - u["cer"]
                lines.append(
                    f"| {a['engine']} | {pct(u['cer'])} | {pct(b['cer'])} → **{pct(a['cer'])}** | "
                    f"{gap * 100:+.1f}%p | {'통과' if gap <= 0.02 else '미달'} | {lat(b)} → {lat(a)} |"
                )
            lines += [
                "",
                "- 비교 기준은 문장 단위 '전체' CER이다. 세션도 숫자가 든 문장을 포함하기 때문이다.",
                "- 문장 단위는 문장 하나를 통째로 넣고, 세션은 문장들을 0.6초 간격으로 이어 붙여 창으로 "
                "자른다. 그래서 같은 엔진이라도 문맥과 경계 조건이 조금 다르다.",
            ]

    longform = load("*.longform.json")
    if longform:
        backlog = {r["spec"]: r for r in stream if r["mode"] == "backlog" and r.get("label") == "after"}
        lines += [
            "",
            "## 4. 회의 후 일괄 재전사 (세션 전체를 한 번에)",
            "",
            f"같은 {longform[0]['session_min']}분 세션을 엔진에 통째로 넣었다. 창을 나누는 일은 엔진의 "
            "long-form 디코딩이 한다. 비교용으로 같은 엔진의 꼬리 창 backlog(수정 후) CER을 붙였다.",
            "",
            "| 엔진 | long-form CER | 처리 시간 | RTF | 최대 메모리 | 꼬리 창 backlog CER |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in (r for r in longform  # sections 5, 6, 7
                  if not r["spec"].endswith(("-spans", "-filtered", "-packed")) and not r["spec"].startswith("rust-")):
            b = backlog.get(r["spec"])
            lines.append(
                f"| {r['engine']} | **{pct(r['cer'])}** | {r['sec']:.0f}s | {r['rtf']:.3f} | "
                f"{r['peak_footprint_mb']:.0f}MB | {pct(b['cer']) if b else '—'} |"
            )

    spans = [r for r in stream + longform if r["spec"].endswith("-spans")]
    if spans:
        from engines import VAD

        sv = next((r for r in stream if r["spec"] == "sensevoice" and r["mode"] == "live"
                   and r.get("label") == "after"), None)
        lines += [
            "",
            "## 5. 우리 VAD 구간 + 구간별 디코딩",
            "",
            "엔진의 VAD와 long-form을 쓰지 않는다. Silero VAD(MIT)로 음성 구간을 자르고 구간마다 따로 "
            "디코딩한다. 시각은 구간에서 온다. live(꼬리 창)와 일괄(세션 전체)이 같은 함수"
            "(`engines.speech_spans`)와 같은 값을 쓴다. SenseVoice 행도 같은 규칙이다.",
            "",
            "구간 규칙: " + ", ".join(f"`{k}={v}`" for k, v in VAD.items()) + ".",
            "",
            "| 엔진 | 조건 | CER | 문장 단위 CER | 창당 처리 / 처리 시간 | 자막 지연 중앙/p95 | 측정 중 부하(1분 평균) |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for r in sorted(spans, key=lambda r: (r["spec"], r["mode"])):
            u = by_spec.get(r["spec"].removesuffix("-spans"))
            cost = (f"{r['sec']:.0f}s (RTF {r['rtf']:.3f})" if r["mode"] == "longform"
                    else f"{r['pass_sec_first10min']:.2f}s → {r['pass_sec_last10min']:.2f}s")
            lines.append(
                f"| {r['engine']} | {'일괄' if r['mode'] == 'longform' else 'live(10초 주기)'} | "
                f"**{pct(r['cer'])}** | {pct(u['cer']) if u else '—'} | {cost} | {lat(r)} | {r.get('load', '—')} |"
            )
        if sv:
            lines.append(f"| {sv['engine']} (기준) | live(10초 주기) | {pct(sv['cer'])} | "
                         f"{pct(by_spec['sensevoice']['cer'])} | {sv['pass_sec_first10min']:.2f}s → "
                         f"{sv['pass_sec_last10min']:.2f}s | {lat(sv)} | — |")

    final = [r for r in stream + longform  # *.longform.json also matches *.replies.longform.json
             if r["spec"].endswith(("-filtered", "-packed"))]
    if final:
        from engines import PACK

        def row(r, kind):
            u = by_spec.get(r["spec"].rsplit("-", 1)[0]) or by_spec.get(r["spec"])
            gap = r["cer"] - u["cer"] if u else None
            verdict = "—" if gap is None else ("통과" if gap <= 0.02 else "미달")
            reply = (f"{r['replies_recalled']}/{r['replies']} (필터로 잃음 {r['replies_lost_to_filter']})"
                     if "replies" in r else "—")
            cost = (f"{r['sec']:.0f}s (RTF {r['rtf']:.3f})" if r["mode"] == "longform"
                    else f"{r['pass_sec_first10min']:.2f}s → {r['pass_sec_last10min']:.2f}s")
            return (f"| {r['engine']} | {kind} | **{pct(r['cer'])}** | {pct(u['cer']) if u else '—'} | "
                    f"{verdict} | {reply} | {r.get('filter_dropped_segments', '—')} | {cost} | {lat(r)} | "
                    f"{r.get('load', '—')} |")

        lines += [
            "",
            "## 6. 마지막 라운드: 필터, 묶기, 짧은 대답",
            "",
            "5절의 구간 규칙에 Whisper 세그먼트 필터를 더했다. 필터는 `no_speech_prob > "
            f"{PACK['no_speech_prob']}` 이면서 `avg_logprob < {PACK['avg_logprob']}`인 세그먼트를 버린다"
            "(OpenAI/faster-whisper와 같은 규칙, whisper-rs로 그대로 옮길 수 있다). "
            f"`packed`는 이웃 구간을 {PACK['max_chunk_sec']:.0f}초까지 묶어 한 번에 디코딩하고, "
            "`spans`는 구간마다 따로 디코딩한다. 기준은 문장 단위 CER + 2.0%p다.",
            "",
            "짧은 대답: macOS `say -v Yuna`로 합성한 8개(네, 아니요, 맞아요, 좋습니다, 그렇죠, 알겠습니다, "
            "잠깐만요, 글쎄요)를 3문장마다 하나씩 앞뒤 0.8초 간격으로 끼웠다(`make_replies.sh`, "
            "합성 음성은 bench/data에만 있다). 글자 정렬에서 절반 넘게 그대로 맞으면 나온 것으로 센다.",
            "",
            "| 엔진 | 조건 | CER | 문장 단위 CER | 기준 | 짧은 대답 | 필터가 버린 세그먼트 | 창당 처리 / 처리 시간 | 자막 지연 중앙/p95 | 부하(1분 평균) |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for r in sorted(final, key=lambda r: (r["spec"], r["mode"], "replies" in r)):
            kind = ("일괄 + 짧은 대답" if "replies" in r else "일괄") if r["mode"] == "longform" else "live(10초 주기)"
            lines.append(row(r, kind))
        for spec in ("fw-small", "fw-large-v3-turbo", "sensevoice"):
            r = next((r for r in stream if r["spec"] == spec and r["mode"] == "live" and r.get("label") == "after"), None)
            if r:
                lines.append(row(r, "live(10초 주기), 비교"))

    rust = [r for r in longform if r["spec"].startswith("rust-")]
    if rust:
        lines += [
            "",
            "## 7. Rust 엔진의 회의 후 최종본 (앱이 쓸 경로 그대로)",
            "",
            "`crates/meetbutler-engine`의 `final_transcript`를 그대로 돌렸다(`examples/final_pass`). 6절 "
            "`*-filtered`와 같은 규칙(우리 구간을 하나씩 디코딩, 2초 미만 구간의 자막 문구 차단)이지만 "
            "VAD 확률, 구간 규칙, 차단 목록, whisper.cpp 호출이 모두 Rust다. 오디오는 앱이 녹음하는 "
            "형식(16 kHz s16le)으로 넘긴다. 합격선은 최종본 CER 3% 이하다.",
            "",
            "| 엔진 | 조건 | CER | 기준 | 짧은 대답 | 처리 시간 | RTF | 최대 메모리 | 부하(1분 평균) |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
        for r in sorted(rust, key=lambda r: (r["spec"], "replies" in r)):
            reply = f"{r['replies_recalled']}/{r['replies']}" if "replies" in r else "—"
            verdict = "통과" if r["cer"] <= 0.03 else "미달"
            lines.append(
                f"| {r['engine']} | {'일괄 + 짧은 대답' if 'replies' in r else '일괄'} | **{pct(r['cer'])}** | "
                f"{verdict} | {reply} | {r['sec']:.0f}s | {r['rtf']:.3f} | {r['peak_footprint_mb']:.0f}MB | "
                f"{r.get('load', '—')} |"
            )
        lines += [
            "",
            "- whisper.cpp의 ggml Silero v5.1.2(`-ggmlvad` 행)는 쓰지 않는다. 원래 Silero는 프레임마다 "
            "이전 프레임의 마지막 64샘플을 문맥으로 붙이는데, ggml 이식판은 프레임마다 반사 패딩"
            "(`ggml_pad_reflect_1d(…, 64, 64)`)을 한다. 이 세션에서 말을 23.0분만 찾았고(v4는 int16 입력에서 "
            "26.9분), 발화를 통째로 놓쳐 삭제가 314자였다. 합성 음성(TTS)에서는 오히려 더 많이 잡아서 "
            "합성 음성 대조로는 드러나지 않았다. VAD나 엔진을 바꿀 때는 FLEURS로도 확인한다"
            "(`tests/slow_vad.rs`).",
            "- 채택한 VAD는 bench와 같은 `silero_vad.onnx`(k2-fsa가 내보낸 Silero v4, MIT)를 tract(순수 Rust "
            "ONNX 런타임)로 돌린다. onnxruntime 대비 프레임 확률 차이는 합성 세션 6.2e-6, FLEURS 10분 1.9e-5이고, "
            "같은 입력에서 구간은 sherpa-onnx와 하나도 다르지 않다.",
            "- 같은 VAD라도 입력이 float인지 int16인지에 따라 구간이 달라진다. 이 36분 세션에서 sherpa 구간은 "
            "float 입력 205개, int16 입력 174개였다(겹치지 않는 구간이 많다). 5·6절은 float으로 쟀고, 앱은 "
            "int16으로 녹음하므로 이 절은 int16으로 쟀다.",
        ]

    out = RESULTS / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
