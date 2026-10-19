"""Put summaries side by side for people to score without knowing which model wrote them.

    python3 -I bench/blind_pack.py

For each meeting, the chunked summaries of the chosen models go into
`bench/data/blind/<meeting>.md` under the letters A, B, C in a shuffled order, with the
transcript after them; which letter is which model goes into `bench/data/blind/key.json`.
Keep the key away from whoever scores. Everything stays in the ignored bench/data/.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RAW = BENCH / "results" / "raw" / "summary"
MEETINGS = BENCH / "data" / "meetings"
OUT = BENCH / "data" / "blind"

PACK = ["weekly-30m", "customer-60m"]
MODELS = ["qwen3.5-4b", "qwen3.5-9b", "gemma-4-e4b"]
MODE = "chunked"


def render(summary: dict) -> list[str]:
    lines = ["**요약**", "", str(summary["summary"]), "", "**결정**", ""]
    lines += [f"- {d}" for d in summary["decisions"]] or ["- (없음)"]
    lines += ["", "**할 일**", ""]
    lines += [
        f"- {a.get('owner', '')}: {a.get('task', '')} ({a.get('due', '')})" for a in summary["action_items"]
    ] or ["- (없음)"]
    lines += ["", "**미해결 질문**", ""]
    lines += [f"- {q}" for q in summary["open_questions"]] or ["- (없음)"]
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.SystemRandom()
    key = {}
    for meeting in PACK:
        order = MODELS[:]
        rng.shuffle(order)
        letters = dict(zip("ABC", order))
        key[meeting] = letters
        doc = [f"# {meeting}", "", "각 요약을 읽고 정확성·빠진 것·자연스러움을 채점해 주세요. 모델 이름은 가렸습니다.", ""]
        for letter, model in letters.items():
            summary = json.loads((RAW / f"{model}.{meeting}.{MODE}.out.json").read_text())
            doc += [f"## 요약 {letter}", ""] + render(summary) + [""]
        transcript = json.loads((MEETINGS / f"{meeting}.json").read_text())["lines"]
        doc += ["## 대본", ""] + transcript
        (OUT / f"{meeting}.md").write_text("\n".join(doc) + "\n")
    (OUT / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"wrote {', '.join(PACK)} and key.json to {OUT}")


if __name__ == "__main__":
    main()
