"""Score a model's meeting summary against the meeting's answer key (`meeting_gen`).

The summary is the app's fixed format (requirements SUM-1), as JSON:

    {"summary": "...", "decisions": ["..."],
     "action_items": [{"owner": "...", "task": "...", "due": "..."}],
     "open_questions": ["..."]}

Scored without a judge model:
- format: the output parses and has every field
- recall of decisions, action items (and for each found one, whether its owner and
  deadline are right) and open questions: an item counts as found when one output entry
  contains all of its keywords (spacing ignored)
- withdrawn: proposals the meeting took back that the summary lists as decisions
- invented: people named in the summary who were not in the meeting, and numbers in it
  the meeting never said
Korean naturalness is left to people, blind (bench/README.md).
"""
from __future__ import annotations

import json
import re

from meeting_gen import PEOPLE

FIELDS = ("summary", "decisions", "action_items", "open_questions")


def squash(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def parse(output: str) -> dict | None:
    """The summary object, or None when the output is not the fixed format. A model
    may wrap it in a code fence or add a sentence around it; the first object counts."""
    start = output.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(output[start:])
        except json.JSONDecodeError:
            start = output.find("{", start + 1)
            continue
        if isinstance(obj, dict) and all(f in obj for f in FIELDS):
            if all(isinstance(obj[f], list) for f in FIELDS[1:]) and all(
                isinstance(a, dict) for a in obj["action_items"]
            ):
                return obj
        return None
    return None


def has_all(text: str, keywords: list[str]) -> bool:
    t = squash(text)
    return all(squash(k) in t for k in keywords)


def match(items: list[dict], entries: list[str]) -> list[int | None]:
    """For each key item, the index of the first unused entry containing all its keywords."""
    used: set[int] = set()
    out: list[int | None] = []
    for item in items:
        hit = next((i for i, e in enumerate(entries) if i not in used and has_all(e, item["keywords"])), None)
        if hit is not None:
            used.add(hit)
        out.append(hit)
    return out


def score(output: str, meeting: dict) -> dict:
    key = meeting["key"]
    summary = parse(output)
    totals = {
        "decisions": len(key["decisions"]),
        "actions": len(key["actions"]),
        "open_questions": len(key["open_questions"]),
        "withdrawn": len(key["withdrawn"]),
    }
    if summary is None:
        return {"format_ok": False, **{f"{k}_total": v for k, v in totals.items()}}

    decisions = [str(d) for d in summary["decisions"]]
    actions = summary["action_items"]
    action_text = [" ".join(str(a.get(f, "")) for f in ("owner", "task", "due")) for a in actions]
    questions = [str(q) for q in summary["open_questions"]]

    found_d = match(key["decisions"], decisions)
    found_a = match(key["actions"], action_text)
    found_q = match(key["open_questions"], questions)
    owner_ok = due_ok = 0
    for want, hit in zip(key["actions"], found_a):
        if hit is None:
            continue
        got = actions[hit]
        owner_ok += want["owner"] in squash(got.get("owner", ""))
        due_ok += squash(want["due"]) in squash(got.get("due", ""))
    # a withdrawn proposal counts against the summary only when it is listed as decided
    withdrawn = sum(any(has_all(d, w["keywords"]) for d in decisions) for w in key["withdrawn"])

    everything = json.dumps(summary, ensure_ascii=False)
    present = set(key["participants"])
    invented_people = sorted({p for p in PEOPLE if p in everything and p not in present})
    owners = {squash(a.get("owner", "")) for a in actions}
    unknown_owners = sorted(o for o in owners if o and not any(p in o for p in present))
    said = set(re.findall(r"\d+", "\n".join(meeting["lines"])))
    invented_numbers = sorted(set(re.findall(r"\d+", everything)) - said)

    return {
        "format_ok": True,
        "decisions_found": sum(h is not None for h in found_d),
        "decisions_total": totals["decisions"],
        "actions_found": sum(h is not None for h in found_a),
        "actions_owner_ok": owner_ok,
        "actions_due_ok": due_ok,
        "actions_total": totals["actions"],
        "open_questions_found": sum(h is not None for h in found_q),
        "open_questions_total": totals["open_questions"],
        "withdrawn_listed": withdrawn,
        "withdrawn_total": totals["withdrawn"],
        "invented_people": invented_people,
        "unknown_owners": unknown_owners,
        "invented_numbers": invented_numbers,
        "summary_chars": len(str(summary["summary"])),
    }
