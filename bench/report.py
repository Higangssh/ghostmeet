"""Collect bench/results/*.json into bench/results/RESULTS.md (numbers only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS  # noqa: E402

BENCH = Path(__file__).resolve().parent


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def lat(r: dict) -> str:
    if "caption_latency_median_s" not in r:
        return "—"
    return f"{r['caption_latency_median_s']:.1f} / {r['caption_latency_p95_s']:.1f}s"


# peak memory a summary may use, by device tier (models are loaded after transcription
# has let go of its own)
TIER_LIMIT_MB = {"8GB": 4096, "16GB": 9216}

# f16 K+V bytes per token of context, from each GGUF's metadata: 2 × layers that keep a
# KV cache × KV heads × head size × 2 bytes. Qwen3.5 keeps one only in its 8 full-attention
# layers of 33 (the others carry a fixed recurrent state); Gemma 4 E4B in its 4 global
# layers that are not shared (sliding-window layers hold a fixed 512 tokens).
KV_BYTES_PER_TOKEN = {
    "kanana-1.5-2.1b": 2 * 32 * 8 * 128 * 2,
    "kanana-1.5-8b": 2 * 32 * 8 * 128 * 2,
    "midm-2.0-mini": 2 * 48 * 8 * 128 * 2,
    "midm-2.0-base": 2 * 48 * 8 * 128 * 2,
    "ax-4.0-light": 2 * 28 * 4 * 128 * 2,
    "qwen3.5-4b": 2 * 8 * 4 * 256 * 2,
    "qwen3.5-9b": 2 * 8 * 4 * 256 * 2,
    "gemma-4-e4b": 2 * 4 * 2 * 512 * 2,
}


def summary_table(rows: list[dict]) -> list[str]:
    from collections import defaultdict

    def ratio(a: int, b: int) -> str:
        return f"{a}/{b} ({a / b:.0%})" if b else "—"

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    by = defaultdict(list)
    for r in rows:
        prompt = r.get("prompt", "p1")
        by[(r["model"], r["mode"] + ("" if prompt == "p1" else f" · {prompt}"))].append(r)
    out = [
        "| 모델 | 묶음 | 방식 | 실행 실패 | 형식 깨짐 | 결정 | 한 번 나온 결정 | 값 틀림 | 할 일 | 담당자 맞음 | 기한 맞음 "
        "| 미해결 | 철회를 결정으로 | 지어낸 담당자 | 지어낸 숫자 | 회의당 시간 | 최대 메모리 | 생성 tok/s | 판정 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = {"8GB": 0, "16GB": 1}
    for (model, mode), rs in sorted(by.items(), key=lambda kv: (order.get(kv[1][0]["tier"], 9), kv[0])):
        tier = rs[0]["tier"]
        ran = [r for r in rs if not r.get("error")]
        failed = len(rs) - len(ran)
        scored = [r["score"] for r in ran if r["score"].get("format_ok")]
        tot = lambda k: sum(s.get(k, 0) for s in scored)  # noqa: E731
        # recall is out of every meeting that ran, so a broken format counts as missed
        key = lambda k: sum(r["score"][k] for r in ran)  # noqa: E731
        found_a = tot("actions_found")
        wall = mean([r["wall_sec"] for r in ran])
        peak = max((r.get("peak_mb") or 0 for r in rs), default=0)
        gen = mean([r["gen_tok_s"] for r in ran if r.get("gen_tok_s")])
        verdict = "탈락(메모리)" if peak > TIER_LIMIT_MB.get(tier, 1 << 30) else "—"
        if failed:
            verdict = "실행 실패" if not ran else verdict
        out.append(
            f"| {model} | {tier} | {mode} | {failed}/{len(rs)} | {len(ran) - len(scored)}/{len(ran)} | "
            f"{ratio(tot('decisions_found'), key('decisions_total'))} | "
            f"{ratio(tot('decisions_once_found'), key('decisions_once_total'))} | {tot('decisions_wrong_value')} | "
            f"{ratio(found_a, key('actions_total'))} | "
            f"{ratio(tot('actions_owner_ok'), found_a)} | {ratio(tot('actions_due_ok'), found_a)} | "
            f"{ratio(tot('open_questions_found'), key('open_questions_total'))} | "
            f"{tot('withdrawn_listed')} | "
            f"{sum(len(s.get('unknown_owners', [])) + len(s.get('invented_people', [])) for s in scored)} | "
            f"{sum(len(s.get('invented_numbers', [])) for s in scored)} | "
            f"{f'{wall:.0f}s' if wall else '—'} | {peak / 1024:.1f}GB | {f'{gen:.1f}' if gen else '—'} | {verdict} |"
        )
    return out


def single_memory_estimate(rows: list[dict]) -> list[str]:
    """What single would have peaked at with n_ctx sized to what it used, not transcript + 8192."""
    from collections import defaultdict

    # the largest answer the model gave in a well-formed run; one that ran into max_tokens
    # was looping, and a real n_ctx would not be sized for that
    longest = defaultdict(int)
    for r in rows:
        if not r.get("error") and r["score"].get("format_ok"):
            longest[r["model"]] = max(longest[r["model"]], r["gen_tokens"])
    out = [
        "| 모델 | 묶음 | 대본 | KV/토큰 | 쓴 n_ctx | 필요한 n_ctx | 잰 최대 메모리 | 줄어드는 KV | 추정 최대 메모리 | 상한 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = {"8GB": 0, "16GB": 1}
    for r in sorted(rows, key=lambda r: (order.get(r["tier"], 9), r["model"], int(r["meeting"].split("-")[-1][:-1]), r["meeting"])):
        if r.get("error") or r["model"] not in KV_BYTES_PER_TOKEN:
            continue
        per = KV_BYTES_PER_TOKEN[r["model"]]
        need = -(-(r["prompt_tokens"] + longest[r["model"]] + 64) // 256) * 256
        saved = (r["n_ctx"] - need) * per / 2**20
        est = r["peak_mb"] - saved
        limit = TIER_LIMIT_MB[r["tier"]]
        out.append(
            f"| {r['model']} | {r['tier']} | {r['meeting']} | {per // 1024}KiB | {r['n_ctx']} | {need} | "
            f"{r['peak_mb'] / 1024:.1f}GB | {saved / 1024:.1f}GB | {est / 1024:.1f}GB | "
            f"{'넘음' if est > limit else '안 넘음'} |"
        )
    return out


def v2_section(v2: list[dict], asr: list[dict]) -> list[str]:
    """The clean baseline on meeting_gen v2, the second prompt, and the speech route."""
    out = [
        "",
        "### 새 기준선(v2 대본), 프롬프트 개선, 음성 경로",
        "",
        "v2 대본은 같은 항목 틀을 한 회의에 두 번 심지 않는다. qwen3.5-4b chunked만 쟀다. p2는 첫 프롬프트에 "
        "규칙 셋을 더했다: 정하는 말(확정, 승인, 유지, 진행, 채택)이 나오면 결정이다, 할 일을 정했어도 결정은 따로 "
        "적는다, 숫자와 날짜는 전사 그대로 옮기고 바뀐 값은 마지막 값을 적는다. 정하는 말은 대본 생성기의 문장이 "
        "아닌 일반 낱말로 적었다.",
        "",
        "음성 경로(asr)는 v2 대본을 macOS 한국어 음성 여섯 개로 번갈아 읽어(`asr_meetings.py`) 앱의 회의 후 "
        "최종본 경로(`final_pass`, Silero 구간 + large-v3-turbo)로 받아 적고, 그 전사를 요약했다. 채점은 대본의 "
        "정답지 그대로다.",
        "",
    ]
    out += summary_table(v2 + [{**r, "mode": r["mode"] + " · asr"} for r in asr])
    flagged = [r["score"] for r in v2 + asr if not r.get("error") and r["score"].get("format_ok")]
    if flagged and any("decisions_flagged" in s for s in flagged):
        from collections import defaultdict

        agg = defaultdict(lambda: [0, 0, 0])
        for r in v2 + asr:
            if r.get("error") or not r["score"].get("format_ok"):
                continue
            s = r["score"]
            k = ("asr · " if r in asr else "") + r.get("prompt", "p1")
            agg[k][0] += s.get("decisions_flagged", 0)
            agg[k][1] += s.get("flagged_wrong_value", 0)
            agg[k][2] += s.get("decisions_wrong_value", 0)
        out += [
            "",
            "\"확인 필요\" 표시 후보: chunked가 구간마다 결정의 숫자(단위 한 글자까지, \"11월\", \"250만\")가 그 "
            "구간 전사에 그대로 있는지 보고, 없으면 표시한다(`summary_score.unsupported_numbers`).",
            "",
            "| 조건 | 표시한 결정 | 그중 값 틀림 | 값 틀림 전체 | 값 틀림을 잡은 비율 |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, (n, hit, wrong) in sorted(agg.items()):
            out.append(f"| {k} | {n} | {hit} | {wrong} | {f'{hit / wrong:.0%}' if wrong else '—'} |")
        out += [
            "",
            "표시는 한 번도 켜지지 않았다. 값 틀림의 숫자는 모두 그 구간 전사에 있었다: 정답 대본에서는 같은 안건의 "
            "다른 숫자(남은 버그 수, 철회된 첫 제안 금액)를 결정 값으로 옮겼고, 음성 경로에서는 받아 적기가 숫자를 "
            "바꿨다(\"28일\"→\"18일\", \"20일\"→\"21일\")는 것을 요약이 그대로 옮겼다. 전사와 대조하는 검사로는 "
            "어느 쪽도 잡을 수 없다.",
        ]
    if asr:
        a = [json.loads((BENCH / "data" / "meetings_asr" / f"{m}.json").read_text())["asr"]
             for m in sorted({r["meeting"] for r in asr}) if (BENCH / "data" / "meetings_asr" / f"{m}.json").exists()]
        if a:
            out += [
                "",
                f"음성 경로의 전사: {len(a)}개 회의, 음성 {sum(x['audio_sec'] for x in a) / 60:.0f}분, "
                f"CER {sum(x['cer'] for x in a) / len(a):.1%}(회의 평균), 처리 RTF "
                f"{sum(x['decode_sec'] for x in a) / sum(x['audio_sec'] for x in a):.3f}. "
                "CER은 낮아도 틀린 글자가 기능 이름과 사람 이름에 몰린다(\"주소록 동기화\"→\"수소로 전기화\", "
                "지어낸 담당자 20). p1이 음성 경로에서 놓친 결정 52개 중 18개는 열쇠말이 전사 어디에도 없었고, "
                "나머지는 전사에 맞게 적힌 곳이 있어도 요약이 다른 곳의 잘못 적힌 이름을 쓰거나 빠뜨렸다. "
                "이름이 한 글자 틀린 것까지 맞다고 쳐도 48→51개다.",
            ]
    return out


def vocabulary_section(plain: list[dict], vocab: list[dict]) -> list[str]:
    """STT-7: the same recordings heard with and without the user's vocabulary."""
    data = BENCH / "data"
    meetings = sorted({r["meeting"] for r in vocab})
    out = [
        "",
        "### 단어장(STT-7): 같은 녹음을 단어장 없이, 있게",
        "",
        "음성 경로의 녹음을 그대로 두고(`asr_meetings.py --keep-audio`) 최종본 경로에 단어장을 whisper "
        "initial prompt로 줬다(`--vocabulary`, `WhisperDecoder::with_vocabulary`). 단어장은 사용자가 적을 법한 "
        "것: 회의에 나온 사람 이름, 회의에서 말하지 않은 기능 이름 1~2개(미끼), 회사 이름, 많이 나온 기능 이름 "
        "순으로 100자까지. 이름·용어는 대본에서 말한 횟수 중 전사에서 들린 횟수를 셌고(\"들림\"), 말한 것보다 "
        "많이 들린 횟수(\"더 들림\")와 미끼가 들린 횟수를 따로 셌다. 요약은 4B chunked p2.",
        "",
        "| 조건 | 이름·용어 들림 | 그중 단어장 안 | 더 들림 | 미끼 들림 | CER | 디코딩 | 결정 | 할 일 | 담당자 맞음 | 지어낸 담당자 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, folder, rows in (("단어장 없음", "meetings_asr", plain), ("단어장", "meetings_asr_vocab", vocab)):
        files = [data / folder / f"{m}.json" for m in meetings]
        if not all(f.exists() for f in files):
            continue
        a = [json.loads(f.read_text())["asr"] for f in files]
        t = {k: sum(x["terms"][k] for x in a) for k in a[0]["terms"] if k != "vocabulary"}
        sc = [r["score"] for r in rows if r["meeting"] in meetings and r["score"].get("format_ok")]
        tot = lambda k: sum(x[k] for x in sc)  # noqa: E731
        out.append(
            f"| {label} | {t['heard']}/{t['said']} ({t['heard'] / t['said']:.0%}) "
            f"| {t['in_vocabulary_heard']}/{t['in_vocabulary_said']} ({t['in_vocabulary_heard'] / t['in_vocabulary_said']:.0%}) "
            f"| {t['extra']} | {t['decoys_heard']} | {sum(x['cer'] for x in a) / len(a):.1%} "
            f"| {sum(x['decode_sec'] for x in a):.0f}s "
            f"| {tot('decisions_found')}/{tot('decisions_total')} | {tot('actions_found')}/{tot('actions_total')} "
            f"| {tot('actions_owner_ok')}/{tot('actions_found')} | {sum(len(x['unknown_owners']) for x in sc)} |"
        )
    out += [
        "",
        "단어장이 없으면 전사가 이전 음성 경로 실행과 글자 하나까지 같았다(회귀 없음). 단어장은 이름·용어를 "
        "65%→74%, 결정을 48→57개로 올렸고 디코딩은 1.8% 늘었다. 미끼는 한 번도 들리지 않았다. 더 들린 14번은 "
        "두 종류다: 짧은 구간 하나가 \"지호, 지호, 지호, 지호.\"로 단어장 이름을 되풀이했고(4번), 나머지 10번은 "
        "다른 이름을 말한 자리에서 단어장의 다른 이름이 들렸다(\"시온 님이\"가 단어장 없이는 \"시원님이\", "
        "단어장으로는 \"지호님이\"). 없는 이름보다 실제 참석자 이름으로 틀리는 쪽이 더 그럴듯해 해롭지만, "
        "담당자 맞음은 전체로 올랐다.",
        "",
        "되풀이 거르기(`blocklist::is_vocabulary_echo`: 단어장 용어만 셋 이상 이어진 구간은 버림)는 이 단어장 "
        "전사에서 그 한 구간만 버렸고, 단어장 없는 전사와 대본 문장에서는 하나도 버리지 않았다.",
        "",
        "단어장 실행에서 사라진 할 일 10개(새로 찾은 것을 빼면 5개 순감): 3개는 전사가 바뀌었다(\"다음 주 "
        "화요일\"이 \"다음 주와 요일\"로, 날짜를 못 들음). 7개는 전사에 열쇠말이 그대로 있는데 요약이 같은 틀의 "
        "다른 할 일(\"배포 전 점검표 정리\", \"견적 다시 요청\")과 날짜를 뒤섞었다. 채점 매칭 탓은 없었다. "
        "greedy 디코딩이라 같은 입력에는 같은 출력이 나오니, 이 차이는 전사의 작은 변화가 요약을 흔든 결과다.",
    ]
    return out


def app_section(bench: dict[str, list[dict]], app: dict[str, list[dict]]) -> list[str]:
    """The app's own summary (Rust, `summary::summarise` with the merge step) against the
    bench's join, on the same model answers, and how often its evidence is right."""
    out = [
        "",
        "### 앱의 요약: 합치기 단계와 근거 구간(SUM-3)",
        "",
        "앱이 쓰는 Rust 경로(`examples/summarise`, `app_summary.py`)로 같은 4B chunked p2를 돌렸다. 프롬프트가 "
        "같고 greedy라 구간마다 모델의 답은 bench와 같고, 다른 것은 합치기뿐이다. 앱의 합치기는 나중 값이 같은 "
        "항목의 앞 값을 대신하고, 미룬 결정은 미해결로 옮기고, 의장 선포(\"계속 심사\", \"가결되었음을 "
        "선포\")가 있는 회의에서는 계속 심사된 안건 구간의 결정을 미해결로 내린다. 근거가 맞음은 찾은 항목의 "
        "근거 줄(앞뒤 한 줄 포함)이 그 항목의 값(값이 없으면 열쇠말)을 말하는 비율이다.",
        "",
        "| 대본 | 합치기 | 결정 | 값 틀림 | 철회를 결정으로 | 미해결 | 할 일 | 결정 근거 맞음 | 할 일 근거 맞음 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("v2", "asr"):
        for label, rows in (("bench", [r for r in bench[name] if r.get("prompt") == "p2"]), ("앱", app[name])):
            sc = [r["score"] for r in rows if not r.get("error") and r["score"].get("format_ok")]
            if not sc:
                continue
            t = lambda k: sum(x.get(k, 0) for x in sc)  # noqa: E731
            ev = lambda k, n: f"{t(k)}/{t(n)} ({t(k) / t(n):.0%})" if label == "앱" and t(n) else "—"  # noqa: E731
            out.append(
                f"| {name} | {label} | {t('decisions_found')}/{t('decisions_total')} | {t('decisions_wrong_value')} "
                f"| {t('withdrawn_listed')} | {t('open_questions_found')}/{t('open_questions_total')} "
                f"| {t('actions_found')}/{t('actions_total')} | {ev('decisions_evidence_ok', 'decisions_found')} "
                f"| {ev('actions_evidence_ok', 'actions_found')} |"
            )
    out += [
        "",
        "합성 대본에서 앱의 합치기는 점수를 바꾸지 않았다. 바뀐 것은 한 건뿐으로, 결정에 적힌 \"다음 회의 때 "
        "다시 보며 오늘 정하지 않습니다\"를 미해결로 옮겼다. 철회된 값이 결정으로 남은 경우는 한 구간 안에서 "
        "모델이 적은 것이라 합치기가 볼 수 없다(음성 경로의 하나는 받아 적기가 최종 금액을 놓쳤다). 채점기는 "
        "\"500만 원에서 400만 원으로 확정\"을 철회 값으로 세고 있었다. 이제 \"에서 + 숫자\"를 바뀐 값으로 "
        "보고, 모든 표를 다시 채점했다(결과 9개에서 \"철회를 결정으로\" 12건이 빠짐).",
        "",
        "의장 선포가 있는 회의(국회 상임위 소위 회의록 한 건, 로컬에서만 돌리고 원문과 요약은 커밋하지 "
        "않음)에서 같은 모델 답에 앱의 합치기를 쓰면: 계속 심사된 두 안건 구간의 결정 18개가 미해결로 "
        "내려갔고(계속 심사된 법안을 \"통과\"로 적은 오류가 사라짐), 가결된 안건 안에서 미룬 조항 2개만 "
        "미해결로 갔고, 의장의 선포 문장이 근거 줄과 함께 목록에 들어갔다. 결정 75→62, 미해결 16→38.",
    ]
    return out


def summary_section(rows: list[dict]) -> list[str]:
    """Section 8: local summary models on the synthetic meetings (run_summary.py)."""
    short = [r for r in rows if not r["meeting"].endswith("-120m")]
    long = [r for r in rows if r["meeting"].endswith("-120m")]
    out = [
        "",
        "## 8. 로컬 요약 모델 (합성 회의 대본, 정답 대본 입력)",
        "",
        "`run_summary.py`가 앱의 요약 sidecar를 앱과 같은 방식으로 돌렸다. 요약마다 프로세스를 새로 띄우고, "
        "GGUF 안의 chat template, 고정 형식(SUM-1)을 강제하는 grammar, repeat penalty 1.1, greedy를 쓴다. "
        "대본은 `meeting_gen.py`의 주간·1:1·고객 × 30·60·120분이고 정답지와 맞춰 채점했다"
        "(`summary_score.py`). 모든 모델에 같은 프롬프트를 썼다. single은 전사를 통째로 넣고, chunked는 "
        "약 10분씩 요약한 뒤 목록을 잇는다.",
        "",
        "재현율은 대본 합계다(형식이 깨진 요약은 하나도 못 찾은 것으로 센다). 값 틀림은 안건은 맞게 적었는데 "
        "숫자나 날짜가 정답과 다른 결정의 수다(읽는 사람에게 가장 해로운 오류라 재현율과 같은 무게로 본다). "
        "한 번 나온 결정은 같은 틀이 숫자만 바뀌어 다시 나오지 않은 결정만의 재현율이다. 지어낸 담당자는 회의에 없는 "
        "이름이나 사람이 아닌 담당자의 수이고, 지어낸 숫자는 회의에서 말하지 않은 숫자의 수다. "
        "최대 메모리는 peak footprint와 max RSS 중 큰 값이다. "
        f"기기 상한은 8GB 묶음 {TIER_LIMIT_MB['8GB'] / 1024:.0f}GB, 16GB 묶음 {TIER_LIMIT_MB['16GB'] / 1024:.0f}GB이고, "
        "넘으면 탈락이다. single의 n_ctx는 전사 + 8192(출력 상한)로 잡았다.",
        "",
        "아래 두 표는 이전 생성기(v1, 1ac7bf2)의 대본으로 쟀다. v1은 같은 항목 틀을 한 회의에 숫자만 바꿔 여러 번 "
        "심었다(\"개발자 N명 채용\"이 한 회의에 네 번). 채점기를 고친 뒤 다시 채점해 b76f90b의 표와 수치가 다르다: "
        "철회를 결정으로는 철회를 밝힌 문장(\"600만 원 취소 후 800만 원으로 확정\")을 더는 세지 않아 거의 0이 됐고, "
        "같은 정답 항목은 한 번만 세어 결정 분모가 줄었다.",
        "",
        "### 30·60분 대본 6개, 모든 모델 (이전 생성기 v1)",
        "",
    ]
    out += summary_table(short)
    if long:
        out += [
            "",
            "### 120분 대본 3개, 30·60분 결과로 고른 모델 (이전 생성기 v1)",
            "",
            "묶음마다 상한 안에 든 방식의 결정+할 일 재현율이 높은 두 모델이다. llama 구조 모델(Kanana, Mi:dm)의 "
            "single은 시작하자마자 디코딩이 실패했다(`Decode Error -3`, 최대 메모리 11GB대). 실패한 실행은 n_ctx를 "
            "남기지 못했지만 같은 식(전사 + 8192)이면 57344로, 학습 문맥 32768을 넘고 KV 캐시만 Mi:dm Mini 10.5GB, "
            "Kanana 8B 7GB다. 30·60분 표의 Mi:dm Base single 60분 3개도 같은 실패다(n_ctx 약 36864, KV 6.75GB + "
            "모델 6.5GB). 측정 설정 탓이지만, 프롬프트를 60분의 두 배로 잡아(Mi:dm 약 2.3만, Kanana 약 3.1만 토큰) "
            "n_ctx를 맞춰도 KV만 Mi:dm Mini 약 4.2GB, Kanana 8B 약 3.9GB다. Mi:dm Mini는 8GB 묶음 상한을 넘고 "
            "Kanana 8B는 9GB 상한 근처이며 학습 문맥 끝에 걸린다.",
            "",
        ]
        out += summary_table(long)
    out += [
        "",
        "### single의 메모리를 n_ctx 때문에 얼마나 크게 쟀나 (추정, 다시 돌리지 않음)",
        "",
        "llama.cpp는 n_ctx만큼의 KV 캐시를 처음에 다 잡는다. 필요한 n_ctx = 프롬프트 + 그 모델이 형식을 "
        "지킨 요약 중 가장 긴 출력 + 64를 256 단위로 올린 값이다. 추정 최대 메모리 = 잰 값 − (쓴 n_ctx − "
        "필요한 n_ctx) × KV/토큰. 계산 버퍼는 n_ctx에 따라 조금 더 줄어들 수 있으니 상한 쪽 추정이다.",
        "",
    ]
    out += single_memory_estimate([r for r in rows if r["mode"] == "single"])
    return out


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
            "## 2. 긴 세션, 꼬리 창 (legacy/backend/incremental.py의 IncrementalTranscriber 그대로)",
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
            "형식(16 kHz s16le)으로 넘긴다. 합격선은 최종본 CER 3% 이하다. 짧은 대답을 끼운 세션은 "
            "재현율로 판정한다: 대답 51/53 이상, 차단 규칙으로 잃은 대답 0, CER은 같은 경로의 본 세션 "
            "대비 +1.0%p 이내.",
            "",
            "| 엔진 | 조건 | CER | 기준 | 짧은 대답 | 처리 시간 | RTF | 최대 메모리 | 부하(1분 평균) |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
        plain = {r["spec"]: r for r in rust if "replies" not in r}

        def verdict(r):
            """The plain session against the 3% bar; the session with replies by recall
            (>= 51/53, none lost to the blocklist) and CER within +1.0%p of the plain one."""
            if "replies" not in r:
                return "통과" if r["cer"] <= 0.03 else "미달"
            base = plain.get(r["spec"])
            if base is not None and base["cer"] > 0.03:
                return "— (본 세션 미달)"
            ok = (r["replies_recalled"] >= r["replies"] - 2 and r["replies_lost_to_filter"] == 0
                  and base is not None and r["cer"] - base["cer"] <= 0.01)
            return "통과" if ok else "미달"

        for r in sorted(rust, key=lambda r: (r["spec"], "replies" in r)):
            reply = f"{r['replies_recalled']}/{r['replies']}" if "replies" in r else "—"
            verdict_text = verdict(r)
            lines.append(
                f"| {r['engine']} | {'일괄 + 짧은 대답' if 'replies' in r else '일괄'} | **{pct(r['cer'])}** | "
                f"{verdict_text} | {reply} | {r['sec']:.0f}s | {r['rtf']:.3f} | {r['peak_footprint_mb']:.0f}MB | "
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

    summaries = load("summary/*.json")
    if summaries:
        lines += summary_section(summaries)
        v2, asr = load("summary_v2/*.json"), load("summary_asr/*.json")
        if v2 or asr:
            lines += v2_section(v2, asr)
        vocab = load("summary_asr_vocab/*.json")
        if vocab:
            lines += vocabulary_section([r for r in asr if r.get("prompt") == "p2"], vocab)
        app = {k: load(f"summary_app_{k}/*.json") for k in ("v2", "asr")}
        if any(app.values()):
            lines += app_section({"v2": v2, "asr": asr}, app)

    out = RESULTS / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
