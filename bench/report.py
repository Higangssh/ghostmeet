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
        "| Model | Tier | Mode | Run failed | Broken format | Decisions | One-off decisions | Wrong value | Action items "
        "| Owner right | Due date right | Open questions | Retracted as decision | Invented owner | Invented number "
        "| Time per meeting | Peak memory | Generation tok/s | Verdict |",
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
        verdict = "out (memory)" if peak > TIER_LIMIT_MB.get(tier, 1 << 30) else "—"
        if failed:
            verdict = "run failed" if not ran else verdict
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
        "| Model | Tier | Script | KV/token | n_ctx used | n_ctx needed | Measured peak memory | KV saved "
        "| Estimated peak memory | Cap |",
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
            f"{'over' if est > limit else 'under'} |"
        )
    return out


def v2_section(v2: list[dict], asr: list[dict]) -> list[str]:
    """The clean baseline on meeting_gen v2, the second prompt, and the speech route."""
    out = [
        "",
        "### New baseline (v2 scripts), a better prompt, the speech route",
        "",
        "The v2 scripts never plant the same item template twice in one meeting. Only qwen3.5-4b chunked was "
        "measured. p2 adds three rules to the first prompt: a deciding word (확정, 승인, 유지, 진행, 채택: "
        "confirmed, approved, kept, go ahead, adopted) makes a decision; a decision is written down even when "
        "an action item was also set; numbers and dates are copied as transcribed, and a changed value is "
        "written as its last value. The deciding words are common words, not the script generator's sentences.",
        "",
        "The speech route (asr) reads the v2 scripts aloud with six macOS Korean voices in turn "
        "(`asr_meetings.py`), transcribes them with the app's after-meeting final pass (`final_pass`, Silero "
        "spans + large-v3-turbo), and summarises that transcript. Scoring uses the scripts' answer key as is.",
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
            "A candidate \"확인 필요\" (needs checking) flag: for each span, chunked checks that a decision's "
            "numbers (down to a one-letter unit, \"11월\", \"250만\") appear as is in that span's transcript, "
            "and flags the decision if not (`summary_score.unsupported_numbers`).",
            "",
            "| Condition | Decisions flagged | Of which wrong value | Wrong value in all | Share of wrong values caught |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, (n, hit, wrong) in sorted(agg.items()):
            out.append(f"| {k} | {n} | {hit} | {wrong} | {f'{hit / wrong:.0%}' if wrong else '—'} |")
        out += [
            "",
            "The flag never fired. Every number in a wrong value was in that span's transcript: on the "
            "answer scripts the summary moved another number from the same agenda item (bugs left, the "
            "first offer that was withdrawn) into the decision; on the speech route the transcription "
            "changed a number (\"28일\"→\"18일\", \"20일\"→\"21일\") and the summary copied it. A check "
            "against the transcript cannot catch either.",
        ]
    if asr:
        a = [json.loads((BENCH / "data" / "meetings_asr" / f"{m}.json").read_text())["asr"]
             for m in sorted({r["meeting"] for r in asr}) if (BENCH / "data" / "meetings_asr" / f"{m}.json").exists()]
        if a:
            out += [
                "",
                f"Speech route transcripts: {len(a)} meetings, {sum(x['audio_sec'] for x in a) / 60:.0f} min of "
                f"speech, CER {sum(x['cer'] for x in a) / len(a):.1%} (meeting mean), processing RTF "
                f"{sum(x['decode_sec'] for x in a) / sum(x['audio_sec'] for x in a):.3f}. "
                "CER is low, but the wrong letters gather in feature names and people's names "
                "(\"주소록 동기화\"→\"수소로 전기화\", contact sync heard as nonsense; 20 invented owners). Of "
                "the 52 decisions p1 missed on the speech route, 18 had their keywords nowhere in the "
                "transcript; for the rest the transcript had them right somewhere, but the summary used a "
                "misheard name from elsewhere or left them out. Counting names one letter off as right "
                "still gives only 48→51.",
            ]
    return out


def vocabulary_section(plain: list[dict], vocab: list[dict]) -> list[str]:
    """STT-7: the same recordings heard with and without the user's vocabulary."""
    data = BENCH / "data"
    meetings = sorted({r["meeting"] for r in vocab})
    out = [
        "",
        "### Vocabulary (STT-7): the same recordings without and with a vocabulary",
        "",
        "The speech route's recordings were kept (`asr_meetings.py --keep-audio`) and the final pass was "
        "given a vocabulary as whisper's initial prompt (`--vocabulary`, `WhisperDecoder::with_vocabulary`). "
        "The vocabulary is what a user would plausibly type: the people's names in the meeting, 1–2 feature "
        "names never said in the meeting (decoys), the company name, then the most frequent feature names, "
        "up to 100 characters. For names and terms we counted how many of the times said in the script were "
        "heard in the transcript (\"heard\"), and separately how many times one was heard more often than "
        "said (\"extra\") and how many times a decoy was heard. Summaries are 4B chunked p2.",
        "",
        "| Condition | Names and terms heard | Of which in vocabulary | Extra | Decoys heard | CER | Decoding "
        "| Decisions | Action items | Owner right | Invented owner |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, folder, rows in (("no vocabulary", "meetings_asr", plain), ("vocabulary", "meetings_asr_vocab", vocab)):
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
        "Without a vocabulary the transcript matched the earlier speech route run to the letter (no "
        "regression). The vocabulary raised names and terms from 65%→74% and decisions from 48→57, and "
        "decoding took 1.8% longer. No decoy was ever heard. The 14 extras are of two kinds: one short span "
        "repeated a vocabulary name as \"지호, 지호, 지호, 지호.\" (4 times), and the other 10 heard another "
        "vocabulary name where a different name was said (\"시온 님이\" was \"시원님이\" without the "
        "vocabulary and \"지호님이\" with it). Getting it wrong with a real attendee's name is more "
        "believable than a name that does not exist, so it does more harm, but owner right went up overall.",
        "",
        "The echo filter (`blocklist::is_vocabulary_echo`: drop a span made only of three or more "
        "vocabulary terms in a row) dropped only that one span in this vocabulary transcript, and nothing "
        "in the transcript without a vocabulary or in the script sentences.",
        "",
        "The 10 action items lost in the vocabulary run (5 net, after the newly found ones): in 3 the "
        "transcript changed (\"다음 주 화요일\", next Tuesday, became \"다음 주와 요일\", so the date was "
        "not heard). In 7 the keywords were in the transcript as is, but the summary mixed up the date with "
        "another action item from the same template (\"배포 전 점검표 정리\", \"견적 다시 요청\"). None came "
        "from score matching. Greedy decoding gives the same output for the same input, so the difference is "
        "a small change in the transcript shaking the summary.",
    ]
    return out


def app_section(bench: dict[str, list[dict]], app: dict[str, list[dict]]) -> list[str]:
    """The app's own summary (Rust, `summary::summarise` with the merge step) against the
    bench's join, on the same model answers, and how often its evidence is right."""
    out = [
        "",
        "### The app's summary: the merge step and evidence spans (SUM-3)",
        "",
        "The same 4B chunked p2 was run through the Rust path the app uses (`examples/summarise`, "
        "`app_summary.py`). The prompt is the same and decoding is greedy, so the model's answer for each "
        "span is the same as in bench; only the merge differs. The app's merge lets a later value replace "
        "an earlier one for the same item, moves deferred decisions to open questions, and in a meeting "
        "with chair rulings (\"계속 심사\", held over for further review; \"가결되었음을 선포\", declared "
        "passed) moves the decisions of a held-over item's spans to open questions. Evidence right is the "
        "share of found items whose evidence line (with one line either side) states the item's value (or "
        "its keyword when it has no value).",
        "",
        "| Script | Merge | Decisions | Wrong value | Retracted as decision | Open questions | Action items "
        "| Decision evidence right | Action item evidence right |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("v2", "asr"):
        for label, rows in (("bench", [r for r in bench[name] if r.get("prompt") == "p2"]), ("app", app[name])):
            sc = [r["score"] for r in rows if not r.get("error") and r["score"].get("format_ok")]
            if not sc:
                continue
            t = lambda k: sum(x.get(k, 0) for x in sc)  # noqa: E731
            ev = lambda k, n: f"{t(k)}/{t(n)} ({t(k) / t(n):.0%})" if label == "app" and t(n) else "—"  # noqa: E731
            out.append(
                f"| {name} | {label} | {t('decisions_found')}/{t('decisions_total')} | {t('decisions_wrong_value')} "
                f"| {t('withdrawn_listed')} | {t('open_questions_found')}/{t('open_questions_total')} "
                f"| {t('actions_found')}/{t('actions_total')} | {ev('decisions_evidence_ok', 'decisions_found')} "
                f"| {ev('actions_evidence_ok', 'actions_found')} |"
            )
    out += [
        "",
        "On the synthetic scripts the app's merge did not change the scores. It changed one thing only: "
        "it moved a decision that read \"다음 회의 때 다시 보며 오늘 정하지 않습니다\" (we will look again "
        "at the next meeting and not decide today) to open questions. A retracted value left as a decision "
        "was written by the model inside one span, so the merge cannot see it (one on the speech route was "
        "the transcription missing the final amount). The scorer was counting \"500만 원에서 400만 원으로 "
        "확정\" (confirmed at 4 million won, down from 5 million) as a retracted value. It now reads \"에서 + "
        "number\" (from + number) as a changed value, and every table was scored again (12 \"retracted as "
        "decision\" fewer across 9 results).",
        "",
        "In a meeting with chair rulings (one set of National Assembly subcommittee minutes, run locally "
        "only; neither the text nor the summary is committed), the app's merge on the same model answers: "
        "moved the 18 decisions in the two held-over items' spans to open questions (the error of writing "
        "a held-over bill as \"통과\", passed, went away), moved only 2 deferred clauses inside passed items "
        "to open questions, and put the chair's ruling sentences in the list with their evidence lines. "
        "Decisions 75→62, open questions 16→38.",
    ]
    return out


def summary_section(rows: list[dict]) -> list[str]:
    """Section 8: local summary models on the synthetic meetings (run_summary.py)."""
    short = [r for r in rows if not r["meeting"].endswith("-120m")]
    long = [r for r in rows if r["meeting"].endswith("-120m")]
    out = [
        "",
        "## 8. Local summary models (synthetic meeting scripts, answer script as input)",
        "",
        "`run_summary.py` ran the app's summary sidecar the way the app does: a new process per summary, "
        "the chat template inside the GGUF, a grammar that forces the fixed format (SUM-1), repeat penalty "
        "1.1, greedy. The scripts are `meeting_gen.py`'s weekly, 1:1 and customer meetings × 30, 60 and "
        "120 min, scored against the answer key (`summary_score.py`). Every model got the same prompt. "
        "single takes the whole transcript; chunked summarises about 10 min at a time and joins the lists.",
        "",
        "Recall is summed over the scripts (a summary with a broken format counts as finding nothing). "
        "Wrong value is the number of decisions with the right agenda item but a number or date that "
        "differs from the answer (the error that hurts a reader most, so it weighs as much as recall). "
        "One-off decisions is the recall over only the decisions whose template does not come back with "
        "different numbers. Invented owner counts owners who are not in the meeting or are not people; "
        "invented number counts numbers never said in the meeting. "
        "Peak memory is the larger of peak footprint and max RSS. "
        f"The device caps are {TIER_LIMIT_MB['8GB'] / 1024:.0f}GB for the 8GB tier and "
        f"{TIER_LIMIT_MB['16GB'] / 1024:.0f}GB for the 16GB tier; "
        "going over is out. single's n_ctx was set to transcript + 8192 (the output cap).",
        "",
        "The two tables below were measured on the earlier generator's scripts (v1, 1ac7bf2). v1 planted "
        "the same item template several times in one meeting with only the numbers changed (\"개발자 N명 "
        "채용\", hire N developers, four times in one meeting). After the scorer was fixed everything was "
        "scored again, so the numbers differ from b76f90b's tables: retracted as decision no longer counts "
        "a sentence that states the retraction (\"600만 원 취소 후 800만 원으로 확정\", 6 million won "
        "cancelled, then confirmed at 8 million) and fell to almost 0, and the same answer item is counted "
        "once, so the decisions denominator shrank.",
        "",
        "### Six 30 and 60 min scripts, every model (earlier generator v1)",
        "",
    ]
    out += summary_table(short)
    if long:
        out += [
            "",
            "### Three 120 min scripts, models chosen by the 30 and 60 min results (earlier generator v1)",
            "",
            "For each tier, the two models with the highest decision + action item recall in a mode that "
            "fit the cap. single on the llama-architecture models (Kanana, Mi:dm) failed to decode right "
            "at the start (`Decode Error -3`, peak memory around 11GB). A failed run leaves no n_ctx, but "
            "by the same formula (transcript + 8192) it would be 57344, past the training context of "
            "32768, with the KV cache alone at 10.5GB for Mi:dm Mini and 7GB for Kanana 8B. The three "
            "Mi:dm Base single 60 min runs in the 30 and 60 min table are the same failure (n_ctx about "
            "36864, KV 6.75GB + model 6.5GB). The measurement setup is to blame, but even with n_ctx sized "
            "to a prompt twice the 60 min one (about 23 thousand tokens for Mi:dm, about 31 thousand for "
            "Kanana), the KV alone is about 4.2GB for Mi:dm Mini and about 3.9GB for Kanana 8B. Mi:dm "
            "Mini goes over the 8GB tier's cap, and Kanana 8B sits near the 9GB cap and at the end of its "
            "training context.",
            "",
        ]
        out += summary_table(long)
    out += [
        "",
        "### How much n_ctx inflated single's measured memory (estimated, not run again)",
        "",
        "llama.cpp reserves the KV cache for all of n_ctx up front. n_ctx needed = prompt + the longest "
        "output among that model's well-formed summaries + 64, rounded up to a multiple of 256. Estimated "
        "peak memory = measured − (n_ctx used − n_ctx needed) × KV/token. Compute buffers may shrink a "
        "little more with n_ctx, so this is an upper-side estimate.",
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
        "# Korean STT engine comparison — results",
        "",
        f"Machine: {m['cpu']} ({m['perf_cores']}P+{m['eff_cores']}E), RAM {m['ram_gb']}GB, {m['os']}, "
        f"Python {m['python']}. 4 threads for every engine. No other heavy work while measuring.",
        "",
        f"Data: FLEURS ko_kr test, {utt[0]['utterances']} sentences / {utt[0]['audio_min']} min "
        "(google/fleurs, CC BY 4.0). No real meeting recordings were used.",
        "",
        "## 1. Sentence by sentence (engine accuracy and speed)",
        "",
        "| Engine | CER (sentences without digits) | CER (all) | RTF | Load | First result "
        "| Latency per sentence median/p95 | Peak memory |",
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
        "- CER: letter edit distance after removing spaces and punctuation / letters in the reference "
        "(summed over the corpus). Romanised glosses in brackets in the reference (`비슈케크(Bishkek)`) "
        "are not read aloud, so they were removed. Brackets that hold Hangul are read, so they were kept.",
        "- CER without digits: leaving out sentences whose reference has a digit, so spelling differences "
        "like `2명`/`두 명` (2 people / two people) do not blur the engine comparison. **Compare engines by "
        "this column.**",
        "- RTF = processing time / speech length. First result = from starting to build the engine until the "
        f"first sentence ({utt[0]['first_result_audio_s']:.0f} s) has a result (the model already on disk).",
        "- Peak memory: the macOS phys_footprint peak (\"Memory\" in Activity Monitor), Metal allocations "
        "included. For whisper.cpp it is the whisper-server process.",
    ]
    if cold:
        lines.append(
            "- ¹ In the full run this was the first run, so a model download got mixed into load. The values "
            "are from a separate 5-sentence run with the model cached."
        )

    if stream:
        lines += [
            "",
            "## 2. Long session, tail window (legacy/backend/incremental.py's IncrementalTranscriber as is)",
            "",
            f"A {stream[0]['session_min']} min session of FLEURS sentences joined together, 30 s window. "
            "`before fix` = main at the time of measuring (with the window boundary duplicate bug), "
            "`after fix` = the window boundary hold-back fix.",
            "",
            "| Engine | Code | Condition | Per window first 10 min → last 10 min | p95 | Slope (s/hour) "
            "| Memory first → last | Caption delay median/p95 | CER |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        order = {"before": 0, "after": 1}
        for r in sorted(stream, key=lambda r: (r["spec"], r["mode"], order.get(r.get("label"), 2))):
            cond = "backlog (30 s window)" if r["mode"] == "backlog" else f"live ({r['interval_s']:.0f} s period)"
            code = {"before": "before fix", "after": "after fix"}.get(r.get("label"), r.get("label", "?"))
            lines.append(
                f"| {r['engine']} | {code} | {cond} | {r['pass_sec_first10min']:.2f}s → {r['pass_sec_last10min']:.2f}s | "
                f"{r['pass_sec_p95']:.2f}s | {r['pass_sec_slope_per_hour']:+.3f} | "
                f"{r['footprint_mb_first10min']:.0f} → {r['footprint_mb_last10min']:.0f}MB | {lat(r)} | {pct(r['cer'])} |"
            )
        lines += [
            "",
            "- backlog: the speech is all there already, so every pass fills the 30 s window. This condition "
            "shows whether the cost per window is independent of the position in the session.",
            "- live: a virtual clock imitates audio arriving in real time. The next pass runs 10 s after the "
            "last one ends (as the pipeline worker does). Caption delay = when the pass ended − when the "
            "segment ended.",
            "- Slope: a linear fit of the time per window against the position in the session. Near 0 means "
            "flat.",
        ]

        live = {(r["spec"], r.get("label")): r for r in stream if r["mode"] == "live"}
        specs = sorted({s for s, _ in live})
        if any((s, "after") in live for s in specs):
            lines += [
                "",
                "## 3. Before and after the window boundary fix (live, 10 s period)",
                "",
                "Done when: live CER ≤ sentence-by-sentence CER (all) + 2.0%p.",
                "",
                "| Engine | Sentence CER | live CER before → after | Gap (after − sentence) | Bar "
                "| Caption delay median/p95 before → after |",
                "|---|---:|---:|---:|---|---:|",
            ]
            for s in specs:
                b, a, u = live.get((s, "before")), live.get((s, "after")), by_spec.get(s)
                if not (b and a and u):
                    continue
                gap = a["cer"] - u["cer"]
                lines.append(
                    f"| {a['engine']} | {pct(u['cer'])} | {pct(b['cer'])} → **{pct(a['cer'])}** | "
                    f"{gap * 100:+.1f}%p | {'pass' if gap <= 0.02 else 'fail'} | {lat(b)} → {lat(a)} |"
                )
            lines += [
                "",
                "- The comparison is against the sentence-by-sentence CER on 'all', because the session also "
                "holds sentences with digits.",
                "- Sentence by sentence feeds one whole sentence; the session joins sentences 0.6 s apart and "
                "cuts them into windows. So even the same engine sees slightly different context and boundaries.",
            ]

    longform = load("*.longform.json")
    if longform:
        backlog = {r["spec"]: r for r in stream if r["mode"] == "backlog" and r.get("label") == "after"}
        lines += [
            "",
            "## 4. Batch re-transcription after the meeting (the whole session at once)",
            "",
            f"The same {longform[0]['session_min']} min session fed to the engine whole. The engine's "
            "long-form decoding does the windowing. For comparison, the same engine's tail-window backlog "
            "(after fix) CER is beside it.",
            "",
            "| Engine | long-form CER | Processing time | RTF | Peak memory | Tail-window backlog CER |",
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
            "## 5. Our VAD spans + decoding span by span",
            "",
            "Neither the engine's VAD nor long-form is used. Silero VAD (MIT) cuts the speech spans and each "
            "span is decoded on its own. Times come from the spans. live (tail window) and batch (whole "
            "session) use the same function (`engines.speech_spans`) with the same values. The SenseVoice "
            "row follows the same rules.",
            "",
            "Span rules: " + ", ".join(f"`{k}={v}`" for k, v in VAD.items()) + ".",
            "",
            "| Engine | Condition | CER | Sentence CER | Per window / processing time | Caption delay median/p95 "
            "| Load while measuring (1 min average) |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for r in sorted(spans, key=lambda r: (r["spec"], r["mode"])):
            u = by_spec.get(r["spec"].removesuffix("-spans"))
            cost = (f"{r['sec']:.0f}s (RTF {r['rtf']:.3f})" if r["mode"] == "longform"
                    else f"{r['pass_sec_first10min']:.2f}s → {r['pass_sec_last10min']:.2f}s")
            lines.append(
                f"| {r['engine']} | {'batch' if r['mode'] == 'longform' else 'live (10 s period)'} | "
                f"**{pct(r['cer'])}** | {pct(u['cer']) if u else '—'} | {cost} | {lat(r)} | {r.get('load', '—')} |"
            )
        if sv:
            lines.append(f"| {sv['engine']} (reference) | live (10 s period) | {pct(sv['cer'])} | "
                         f"{pct(by_spec['sensevoice']['cer'])} | {sv['pass_sec_first10min']:.2f}s → "
                         f"{sv['pass_sec_last10min']:.2f}s | {lat(sv)} | — |")

    final = [r for r in stream + longform  # *.longform.json also matches *.replies.longform.json
             if r["spec"].endswith(("-filtered", "-packed"))]
    if final:
        from engines import PACK

        def row(r, kind):
            u = by_spec.get(r["spec"].rsplit("-", 1)[0]) or by_spec.get(r["spec"])
            gap = r["cer"] - u["cer"] if u else None
            verdict = "—" if gap is None else ("pass" if gap <= 0.02 else "fail")
            reply = (f"{r['replies_recalled']}/{r['replies']} (lost to the filter {r['replies_lost_to_filter']})"
                     if "replies" in r else "—")
            cost = (f"{r['sec']:.0f}s (RTF {r['rtf']:.3f})" if r["mode"] == "longform"
                    else f"{r['pass_sec_first10min']:.2f}s → {r['pass_sec_last10min']:.2f}s")
            return (f"| {r['engine']} | {kind} | **{pct(r['cer'])}** | {pct(u['cer']) if u else '—'} | "
                    f"{verdict} | {reply} | {r.get('filter_dropped_segments', '—')} | {cost} | {lat(r)} | "
                    f"{r.get('load', '—')} |")

        lines += [
            "",
            "## 6. Last round: filter, packing, short replies",
            "",
            "Section 5's span rules plus a Whisper segment filter. The filter drops a segment with "
            f"`no_speech_prob > {PACK['no_speech_prob']}` and `avg_logprob < {PACK['avg_logprob']}` "
            "(the same rule as OpenAI/faster-whisper, and it carries over to whisper-rs as is). "
            f"`packed` joins neighbouring spans up to {PACK['max_chunk_sec']:.0f} s and decodes them at once; "
            "`spans` decodes each span on its own. The bar is sentence CER + 2.0%p.",
            "",
            "Short replies: 8 replies made with macOS `say -v Yuna` (네, 아니요, 맞아요, 좋습니다, 그렇죠, "
            "알겠습니다, 잠깐만요, 글쎄요: yes, no, right, good, sure, got it, one moment, well) were put in "
            "after every third sentence with 0.8 s on either side (`make_replies.sh`; the synthetic audio lives "
            "only in bench/data). A reply counts as found when more than half of it matches as is in the "
            "letter alignment.",
            "",
            "| Engine | Condition | CER | Sentence CER | Bar | Short replies | Segments dropped by the filter "
            "| Per window / processing time | Caption delay median/p95 | Load (1 min average) |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for r in sorted(final, key=lambda r: (r["spec"], r["mode"], "replies" in r)):
            kind = ("batch + short replies" if "replies" in r else "batch") if r["mode"] == "longform" else "live (10 s period)"
            lines.append(row(r, kind))
        for spec in ("fw-small", "fw-large-v3-turbo", "sensevoice"):
            r = next((r for r in stream if r["spec"] == spec and r["mode"] == "live" and r.get("label") == "after"), None)
            if r:
                lines.append(row(r, "live (10 s period), comparison"))

    rust = [r for r in longform if r["spec"].startswith("rust-")]
    if rust:
        lines += [
            "",
            "## 7. The Rust engine's final pass after the meeting (the app's own path)",
            "",
            "`crates/meetbutler-engine`'s `final_transcript` run as is (`examples/final_pass`). The same rules "
            "as section 6's `*-filtered` (decode our spans one by one, block subtitle credits in spans under "
            "2 s), but the VAD probabilities, span rules, blocklist and whisper.cpp calls are all Rust. The "
            "audio is passed in the format the app records (16 kHz s16le). The bar is a final pass CER of 3% "
            "or less. The session with short replies is judged by recall: at least 51/53 replies, 0 replies "
            "lost to the blocklist, and CER within +1.0%p of the plain session on the same path.",
            "",
            "| Engine | Condition | CER | Bar | Short replies | Processing time | RTF | Peak memory "
            "| Load (1 min average) |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
        plain = {r["spec"]: r for r in rust if "replies" not in r}

        def verdict(r):
            """The plain session against the 3% bar; the session with replies by recall
            (>= 51/53, none lost to the blocklist) and CER within +1.0%p of the plain one."""
            if "replies" not in r:
                return "pass" if r["cer"] <= 0.03 else "fail"
            base = plain.get(r["spec"])
            if base is not None and base["cer"] > 0.03:
                return "— (plain session failed)"
            ok = (r["replies_recalled"] >= r["replies"] - 2 and r["replies_lost_to_filter"] == 0
                  and base is not None and r["cer"] - base["cer"] <= 0.01)
            return "pass" if ok else "fail"

        for r in sorted(rust, key=lambda r: (r["spec"], "replies" in r)):
            reply = f"{r['replies_recalled']}/{r['replies']}" if "replies" in r else "—"
            verdict_text = verdict(r)
            lines.append(
                f"| {r['engine']} | {'batch + short replies' if 'replies' in r else 'batch'} | **{pct(r['cer'])}** | "
                f"{verdict_text} | {reply} | {r['sec']:.0f}s | {r['rtf']:.3f} | {r['peak_footprint_mb']:.0f}MB | "
                f"{r.get('load', '—')} |"
            )
        lines += [
            "",
            "- whisper.cpp's ggml Silero v5.1.2 (the `-ggmlvad` rows) is not used. The original Silero "
            "prepends the last 64 samples of the previous frame to each frame as context; the ggml port "
            "reflect-pads each frame instead (`ggml_pad_reflect_1d(…, 64, 64)`). On this session it found "
            "only 23.0 min of speech (v4 found 26.9 min on int16 input) and missed whole utterances, for 314 "
            "deleted letters. On synthetic speech (TTS) it caught more, so a synthetic comparison did not "
            "show it. When changing the VAD or the engine, check against FLEURS too (`tests/slow_vad.rs`).",
            "- The VAD we adopted is the same `silero_vad.onnx` as bench (Silero v4 as exported by k2-fsa, "
            "MIT), run on tract (a pure Rust ONNX runtime). Against onnxruntime the frame probabilities differ "
            "by 6.2e-6 on the synthetic session and 1.9e-5 on 10 min of FLEURS, and on the same input not a "
            "single span differs from sherpa-onnx.",
            "- The same VAD gives different spans for float and int16 input. On this 36 min session sherpa "
            "gave 205 spans on float input and 174 on int16 (many do not overlap). Sections 5 and 6 were "
            "measured on float; the app records int16, so this section was measured on int16.",
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
