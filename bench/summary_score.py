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
- withdrawn: proposals the meeting took back that the summary lists as decisions, unless
  the same entry says it was taken back ("600만 원 취소 후 800만 원으로 확정")
- wrong value: a decision missed whose topic is in an entry with another number or date,
  the worst error for a reader, since it looks right
- invented: people named in the summary who were not in the meeting, and numbers in it
  the meeting never said
- evidence (SUM-3), when the summary carries it: of the decisions and action items
  found, how many point at a transcript line that, with the lines either side of it,
  says the item's value (its keywords with digits: the settling line, "그럼 250만 원으로
  확정하죠", often leaves the topic unsaid), or all its keywords when it has no value
Korean naturalness is left to people, blind (bench/README.md).
"""
from __future__ import annotations

import json
import re

from meeting_gen import PEOPLE

FIELDS = ("summary", "decisions", "action_items", "open_questions")

# an entry that names a withdrawn proposal with one of these says it was taken back
# ("500만 원에서 400만 원으로 확정" names the first value on the way to the final one)
TAKEN_BACK = re.compile(r"취소|철회|대신|거절|무산|없던|않|아닌|변경|바꾸|바꿔|바뀌|→|에서\s*\d")


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


def unsupported_numbers(entry: str, source: str) -> list[str]:
    """Numbers in a summary entry, each with the unit after it ("11월", "250만", "2명"),
    that the source transcript never says. A candidate for the app's "확인 필요" mark:
    the model may not copy a number faithfully, and grammar cannot make it."""
    said = squash(source)
    out = []
    for num, unit in re.findall(r"(\d+)\s*([^\s\d.,()]?)", entry):
        if not re.search(rf"(?<!\d){num}{re.escape(unit)}", said):
            out.append(num + unit)
    return out


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


def unique(items: list[dict]) -> list[dict]:
    """A meeting that plants the same item twice asks for it once."""
    seen: set[str] = set()
    return [i for i in items if not (i["text"] in seen or seen.add(i["text"]))]


def wrong_values(items: list[dict], hits: list[int | None], entries: list[str]) -> set[int]:
    """The entries left over that carry a missed item's topic with other numbers: the
    decision is there and says the wrong thing."""
    used = {h for h in hits if h is not None}
    wrong = set()
    for item, hit in zip(items, hits):
        values = [k for k in item["keywords"] if re.search(r"\d", k)]
        topic = [k for k in item["keywords"] if k not in values]
        if hit is not None or not values or not topic:
            continue
        at = next((i for i, e in enumerate(entries)
                   if i not in used and has_all(e, topic) and re.search(r"\d", e)), None)
        if at is not None:
            used.add(at)
            wrong.add(at)
    return wrong


def evidence_ok(items: list[dict], hits: list[int | None], at: list, lines: list[str]) -> int:
    """Found items whose evidence line, with its neighbours, says all their keywords."""
    ok = 0
    for item, hit in zip(items, hits):
        if hit is None or hit >= len(at) or at[hit] is None:
            continue
        near = lines[max(0, at[hit] - 1): at[hit] + 2]
        values = [k for k in item["keywords"] if re.search(r"\d", k)]
        ok += has_all(" ".join(near), values or item["keywords"])
    return ok


def score(output: str, meeting: dict) -> dict:
    key = {**meeting["key"], **{f: unique(meeting["key"][f]) for f in ("decisions", "open_questions", "withdrawn")}}
    summary = parse(output)
    totals = {
        "decisions": len(key["decisions"]),
        "actions": len(key["actions"]),
        "open_questions": len(key["open_questions"]),
        "withdrawn": len(key["withdrawn"]),
    }
    shapes = [re.sub(r"\d+", "#", d["text"]) for d in key["decisions"]]
    once = {s: shapes.count(s) == 1 for s in shapes}
    totals["decisions_once"] = sum(once[s] for s in shapes)
    if summary is None:
        return {"format_ok": False, **{f"{k}_total": v for k, v in totals.items()}}

    decisions = [str(d) for d in summary["decisions"]]
    actions = summary["action_items"]
    action_text = [" ".join(str(a.get(f, "")) for f in ("owner", "task", "due")) for a in actions]
    questions = [str(q) for q in summary["open_questions"]]

    found_d = match(key["decisions"], decisions)
    wrong = wrong_values(key["decisions"], found_d, decisions)
    marks = summary.get("unverified", [])
    flagged = [i for i, d in enumerate(decisions) if d in marks]
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
    withdrawn = sum(
        any(has_all(d, w["keywords"]) and not TAKEN_BACK.search(d) for d in decisions) for w in key["withdrawn"]
    )

    # only the format's own fields; anything else a caller attached is not the summary
    everything = json.dumps({f: summary[f] for f in FIELDS}, ensure_ascii=False)
    present = set(key["participants"])
    invented_people = sorted({p for p in PEOPLE if p in everything and p not in present})
    owners = {squash(a.get("owner", "")) for a in actions}
    unknown_owners = sorted(o for o in owners if o and not any(p in o for p in present))
    said = set(re.findall(r"\d+", "\n".join(meeting["lines"])))
    invented_numbers = sorted(set(re.findall(r"\d+", everything)) - said)

    evidence = {}
    if isinstance(summary.get("evidence"), dict):
        at = summary["evidence"]
        lines = meeting.get("transcript", meeting["lines"])
        evidence = {
            "decisions_evidence_ok": evidence_ok(key["decisions"], found_d, at.get("decisions", []), lines),
            "actions_evidence_ok": evidence_ok(key["actions"], found_a, at.get("action_items", []), lines),
        }

    return {
        **evidence,
        "format_ok": True,
        "decisions_found": sum(h is not None for h in found_d),
        "decisions_total": totals["decisions"],
        "decisions_wrong_value": len(wrong),
        # meeting_gen v1 planted some items again with other numbers ("개발자 N명 채용"
        # twice), which a summary may fairly read as one item revised
        "decisions_once_found": sum(h is not None for h, s in zip(found_d, shapes) if once[s]),
        "decisions_once_total": totals["decisions_once"],
        # entries the run marked "확인 필요" (a number its transcript never says)
        "decisions_flagged": len(flagged),
        "flagged_wrong_value": sum(i in wrong for i in flagged),
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
