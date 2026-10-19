"""Summarise the synthetic meetings with each local model through the app's sidecar.

    bench/data/tools/venv-convert/bin/python -I bench/run_summary.py MODEL [MODEL ...]
        [--meetings weekly-30m ...] [--mode single|chunked|both]

Runs `target/release/meetbutler-summarizer` (build it with `--features llama`) the way
the app will: one process per summary, so process start and model load are part of
the cost, JSON lines on stdin/stdout, output held to the fixed summary format (SUM-1)
by a grammar. The prompt is rendered with the model's own chat template from its GGUF,
thinking turned off where the template has the switch.

Two ways to fit a meeting:
- single: the whole transcript in one prompt (context sized to it)
- chunked: about CHUNK_SYLLABLES of transcript per prompt, each summarised in the same
  format; the lists are joined and one short last prompt writes the overall summary

Writes bench/results/summary/<model>.<meeting>.<mode>.json (scores, timings, peak
memory) and the raw output to bench/results/raw/summary/ (ignored).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jinja2  # noqa: E402  (the conversion venv has it, with gguf)
from gguf import GGUFReader  # noqa: E402

import summary_score  # noqa: E402
from meeting_gen import syllables  # noqa: E402

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
SIDECAR = REPO / "target/release/meetbutler-summarizer"
MODELS = BENCH / "data/models/llm"
MEETINGS = BENCH / "data/meetings"
RESULTS = BENCH / "results/summary"
RAW = BENCH / "results/raw/summary"
MANIFEST = BENCH / "llm_models.json"

CHUNK_SYLLABLES = 3000  # about ten minutes of speech
# room for the lists of a two-hour meeting (some 80 items) in one answer; a chunk of
# about ten minutes needs far less
MAX_TOKENS = 8192
MAX_TOKENS_CHUNK = 2048
# llama.cpp's usual repeat penalty: without it small models loop on one list item
REPEAT_PENALTY = 1.1
# rough Korean tokens per Hangul syllable, to size the context; the sidecar refuses a
# prompt that does not fit, and the run retries once with the size it reports
TOKENS_PER_SYLLABLE = 1.3

# the fixed summary format (requirements SUM-1); whitespace bounded so greedy decoding
# cannot wander off into blank space
GRAMMAR = r'''
root ::= "{" ws "\"summary\"" ws ":" ws string "," ws "\"decisions\"" ws ":" ws strlist "," ws "\"action_items\"" ws ":" ws actlist "," ws "\"open_questions\"" ws ":" ws strlist ws "}"
strlist ::= "[" ws ( string ( ws "," ws string )* )? ws "]"
actlist ::= "[" ws ( action ( ws "," ws action )* )? ws "]"
action ::= "{" ws "\"owner\"" ws ":" ws string "," ws "\"task\"" ws ":" ws string "," ws "\"due\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]{0,8}
'''.strip()

INSTRUCTIONS = """다음은 회의를 받아 적은 전사입니다. 누가 말했는지는 표시되어 있지 않습니다.
이 회의를 정리해서 JSON 하나로만 답하세요. 필드는 다음과 같습니다.
- summary: 회의 전체 요약, 3~5문장
- decisions: 회의에서 최종 확정된 결정, 하나에 한 항목
- action_items: 할 일마다 {"owner": 담당자 이름, "task": 할 일, "due": 기한}
- open_questions: 결론을 내지 못하고 다음으로 미룬 문제, 하나에 한 항목

규칙:
- 결정은 최종 결정만 적습니다. 회의 중에 취소하거나 철회한 제안은 적지 않습니다.
- 결정마다 무엇에 대한 결정인지(기능이나 안건 이름)와 정한 내용(날짜, 금액, 인원 등)을 함께 적습니다.
- owner에는 할 일을 맡은 사람의 이름만 적습니다("님"은 빼도 됩니다). 팀 이름이나 문장은 적지 않습니다. 맡은 사람이 말해지지 않았으면 그 할 일은 적지 않습니다.
- due에는 말한 기한을 그대로 적습니다("12월 3일", "다음 주 수요일"). 기한을 task 안에 쓰지 않습니다.
- 같은 내용을 두 번 적지 않습니다. 잡담은 적지 않습니다.
- 전사에 없는 사람, 숫자, 날짜를 지어내지 않습니다.
- 한국어로 씁니다."""

# the second prompt, aimed at what the first one missed on the synthetic meetings: decisions
# written down only as the action that follows them, and numbers changed on the way out
INSTRUCTIONS_P2 = INSTRUCTIONS.replace(
    "- 결정마다 무엇에 대한 결정인지(기능이나 안건 이름)와 정한 내용(날짜, 금액, 인원 등)을 함께 적습니다.\n",
    # generic words for settling something, not the generator's own sentences
    "- 확정, 승인, 유지, 진행, 채택처럼 무언가를 정하는 말이 나오면 그것은 결정입니다. 안건마다 빠짐없이 적습니다.\n"
    "- 결정에 이어 할 일과 담당자를 정했어도 결정은 decisions에 따로 적습니다. 할 일만 적고 결정을 빼지 않습니다.\n"
    "- 결정마다 무엇에 대한 결정인지(기능이나 안건 이름)와 정한 내용(날짜, 금액, 인원 등)을 함께 적습니다.\n"
    "- 숫자와 날짜는 전사에 나온 글자 그대로 옮겨 적습니다. 처음 제안한 값이 바뀌었으면 마지막에 확정한 값을 적습니다.\n",
)
assert INSTRUCTIONS_P2 != INSTRUCTIONS
PROMPTS = {"p1": INSTRUCTIONS, "p2": INSTRUCTIONS_P2}

MERGE = """다음은 한 회의를 부분별로 요약한 글입니다. 이 내용을 바탕으로 회의 전체를 3~5문장으로
요약하세요. 요약문만 쓰고 다른 말은 덧붙이지 마세요."""


def chat_template(model: Path) -> str:
    reader = GGUFReader(str(model))
    field = reader.fields["tokenizer.chat_template"]
    return bytes(field.parts[field.data[0]]).decode()


def render(template: str, user: str) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
    return env.from_string(template).render(
        messages=[{"role": "user", "content": user}],
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3.5 and others with the switch; ignored elsewhere
        bos_token="",  # the sidecar's tokenizer adds BOS itself
        eos_token="",
    )


def n_ctx_for(prompt: str, max_tokens: int) -> int:
    need = int(syllables(prompt) * TOKENS_PER_SYLLABLE + len(prompt) * 0.1) + max_tokens + 512
    return min(131072, -(-need // 2048) * 2048)


class Sidecar:
    """One sidecar process, measured from spawn to exit."""

    def __init__(self, err_path: Path):
        self.err_path = err_path
        self.started = time.perf_counter()
        self.proc = subprocess.Popen(
            ["/usr/bin/time", "-l", str(SIDECAR)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err_path.open("w"),
            text=True,
        )
        self.n = 0

    def generate(self, model: Path, prompt: str, grammar: str | None, max_tokens: int) -> dict:
        n_ctx = n_ctx_for(prompt, max_tokens)
        for _attempt in range(2):
            self.n += 1
            req = {"type": "generate", "id": f"r{self.n}", "model": str(model), "prompt": prompt,
                   "max_tokens": max_tokens, "n_ctx": n_ctx, "repeat_penalty": REPEAT_PENALTY}
            if grammar:
                req["grammar"] = grammar
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            loaded = None
            for line in self.proc.stdout:
                event = json.loads(line)
                if event.get("type") == "loaded":
                    loaded = event["sec"]
                if event.get("type") in ("done", "error") and event.get("id") == req["id"]:
                    break
            else:
                raise RuntimeError("the sidecar exited")
            if event["type"] == "done":
                return {**event, "loaded_sec": loaded, "n_ctx": n_ctx}
            fit = re.search(r"prompt of (\d+) tokens", event["message"])
            if not fit:
                raise RuntimeError(f"{event['kind']}: {event['message']}")
            n_ctx = -(-(int(fit.group(1)) + max_tokens + 256) // 1024) * 1024
        raise RuntimeError("context still too small")

    def close(self) -> dict:
        self.proc.stdin.close()
        self.proc.wait(timeout=60)
        err = self.err_path.read_text()
        footprint = re.search(r"(\d+)\s+peak memory footprint", err)
        rss = re.search(r"(\d+)\s+maximum resident set size", err)
        mb = lambda m: round(int(m.group(1)) / 2**20) if m else None  # noqa: E731
        return {
            "process_sec": round(time.perf_counter() - self.started, 2),
            # footprint leaves out the model weights, which are mapped from the file and
            # handed to Metal as they are; the resident set counts them, so the device
            # limits are checked against the larger of the two
            "peak_footprint_mb": mb(footprint),
            "max_rss_mb": mb(rss),
            "peak_mb": max(filter(None, [mb(footprint), mb(rss)]), default=None),
            "exit": self.proc.returncode,
        }


def unverified(decisions: list, lines: list[str]) -> list[str]:
    """Decisions with a number their own stretch of transcript never says: what the app
    would mark "확인 필요". Kept beside the summary, not in it, so scoring ignores it."""
    source = "\n".join(lines)
    return [str(d) for d in decisions if summary_score.unsupported_numbers(str(d), source)]


def summarise(model: Path, template: str, lines: list[str], mode: str, sidecar: Sidecar,
              instructions: str = INSTRUCTIONS) -> tuple[str, list[dict]]:
    """The summary JSON text and the stats of every generation it took."""
    if mode == "single":
        prompt = render(template, instructions + "\n\n전사:\n" + "\n".join(lines))
        done = sidecar.generate(model, prompt, GRAMMAR, MAX_TOKENS)
        whole = summary_score.parse(done["text"])
        if whole is None:
            return done["text"], [done]
        whole["unverified"] = unverified(whole["decisions"], lines)
        return json.dumps(whole, ensure_ascii=False), [done]

    chunks, cur = [], []
    for line in lines:
        cur.append(line)
        if syllables("\n".join(cur)) >= CHUNK_SYLLABLES:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    merged = {"summary": "", "decisions": [], "action_items": [], "open_questions": [], "unverified": []}
    parts, runs = [], []
    for chunk in chunks:
        done = sidecar.generate(model, render(template, instructions + "\n\n전사:\n" + "\n".join(chunk)),
                                GRAMMAR, MAX_TOKENS_CHUNK)
        runs.append(done)
        part = summary_score.parse(done["text"])
        if part is None:
            continue  # counted below: a chunk that broke the format loses its items
        parts.append(str(part["summary"]))
        merged["unverified"] += [d for d in unverified(part["decisions"], chunk) if d not in merged["unverified"]]
        for field in ("decisions", "action_items", "open_questions"):
            for item in part[field]:
                if item not in merged[field]:
                    merged[field].append(item)
    done = sidecar.generate(model, render(template, MERGE + "\n\n" + "\n".join(parts)), None, 400)
    runs.append(done)
    merged["summary"] = done["text"].strip()
    merged["chunks"] = len(chunks)
    merged["chunks_broken"] = len(chunks) - len(parts)
    return json.dumps(merged, ensure_ascii=False), runs


def rescore() -> None:
    for path in sorted(RESULTS.glob("*.json")):
        result = json.loads(path.read_text())
        if result["error"]:
            continue
        tag = path.name.removesuffix(".json")
        meeting = json.loads((MEETINGS / f"{result['meeting']}.json").read_text())
        result["score"] = summary_score.score((RAW / f"{tag}.out.json").read_text(), meeting)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"rescored {RESULTS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--meetings", nargs="*")
    ap.add_argument("--mode", choices=["single", "chunked", "both"], default="both")
    ap.add_argument("--set", default="",
                    help="another meeting set: bench/data/meetings_SET, results in summary_SET "
                         "(asr: what the engine heard, from asr_meetings.py; v2: meeting_gen.py version 2)")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="p1")
    ap.add_argument("--rescore", action="store_true",
                    help="score the saved outputs again with the current scorer; runs nothing")
    args = ap.parse_args()
    global MEETINGS, RESULTS, RAW
    if args.set:
        MEETINGS, RESULTS, RAW = (d.with_name(f"{d.name}_{args.set}") for d in (MEETINGS, RESULTS, RAW))
    if args.rescore:
        rescore()
        return

    manifest = json.loads(MANIFEST.read_text())
    meetings = args.meetings or sorted(p.stem for p in MEETINGS.glob("*.json"))
    modes = ["single", "chunked"] if args.mode == "both" else [args.mode]
    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        model = MODELS / manifest[name]["file"]
        template = chat_template(model)
        for meeting_name in meetings:
            meeting = json.loads((MEETINGS / f"{meeting_name}.json").read_text())
            for mode in modes:
                tag = f"{name}.{meeting_name}.{mode}" + (f".{args.prompt}" if args.prompt != "p1" else "")
                sidecar = Sidecar(RAW / f"{tag}.stderr")
                t = time.perf_counter()
                try:
                    text, runs = summarise(model, template, meeting.get("transcript", meeting["lines"]), mode, sidecar,
                                           PROMPTS[args.prompt])
                    error = None
                except RuntimeError as exc:
                    text, runs, error = "", [], str(exc)
                wall = time.perf_counter() - t
                proc = sidecar.close()
                result = {
                    "model": name,
                    "tier": manifest[name]["tier"],
                    "meeting": meeting_name,
                    "mode": mode,
                    "prompt": args.prompt,
                    "error": error,
                    "score": summary_score.score(text, meeting) if not error else None,
                    "wall_sec": round(wall, 1),
                    "load_sec": round(runs[0]["loaded_sec"] or 0, 2) if runs else None,
                    "prompt_tokens": sum(r["stats"]["prompt_tokens"] for r in runs),
                    "gen_tokens": sum(r["stats"]["gen_tokens"] for r in runs),
                    "prompt_tok_s": round(sum(r["stats"]["prompt_tokens"] for r in runs)
                                          / max(1e-9, sum(r["stats"]["prompt_sec"] for r in runs)), 1) if runs else None,
                    "gen_tok_s": round(sum(r["stats"]["gen_tokens"] for r in runs)
                                       / max(1e-9, sum(r["stats"]["gen_sec"] for r in runs)), 1) if runs else None,
                    "n_ctx": max((r["n_ctx"] for r in runs), default=None),
                    "generations": len(runs),
                    **proc,
                }
                (RESULTS / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
                (RAW / f"{tag}.out.json").write_text(text)
                s = result["score"] or {}
                print(f"{tag}: {'ERROR ' + error if error else ''}"
                      f"dec {s.get('decisions_found')}/{s.get('decisions_total')} "
                      f"act {s.get('actions_found')}/{s.get('actions_total')} "
                      f"wall {result['wall_sec']}s peak {result['peak_mb']}MB", flush=True)


if __name__ == "__main__":
    main()
