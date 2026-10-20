"""The summary scorer, against summaries built from a meeting's own key."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import meeting_gen as mg  # noqa: E402
import summary_score as ss  # noqa: E402


def meeting(kind="weekly", minutes=30) -> dict:
    return mg.to_json(mg.generate(kind, minutes))


def perfect(m: dict) -> dict:
    k = m["key"]
    return {
        "summary": "회의 요약",
        "decisions": [d["text"] for d in k["decisions"]],
        "action_items": [{"owner": a["owner"], "task": a["task"], "due": a["due"]} for a in k["actions"]],
        "open_questions": [q["text"] for q in k["open_questions"]],
    }


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.m = meeting()

    def test_a_summary_built_from_the_key_finds_everything(self):
        s = ss.score(json.dumps(perfect(self.m), ensure_ascii=False), self.m)
        self.assertTrue(s["format_ok"])
        self.assertEqual(s["decisions_found"], s["decisions_total"])
        self.assertEqual(s["actions_found"], s["actions_total"])
        self.assertEqual(s["actions_owner_ok"], s["actions_total"])
        self.assertEqual(s["actions_due_ok"], s["actions_total"])
        self.assertEqual(s["open_questions_found"], s["open_questions_total"])
        self.assertEqual((s["withdrawn_listed"], s["invented_people"], s["invented_numbers"]), (0, [], []))

    def test_an_empty_summary_finds_nothing(self):
        empty = {"summary": "", "decisions": [], "action_items": [], "open_questions": []}
        s = ss.score(json.dumps(empty), self.m)
        self.assertTrue(s["format_ok"])
        self.assertEqual((s["decisions_found"], s["actions_found"], s["open_questions_found"]), (0, 0, 0))

    def test_output_that_is_not_the_format_is_a_format_failure(self):
        for bad in ["회의 잘 했습니다", '{"summary": "x"}', '{"summary": "", "decisions": "x", "action_items": [], "open_questions": []}', ""]:
            self.assertFalse(ss.score(bad, self.m)["format_ok"], bad)

    def test_a_code_fence_or_a_sentence_around_the_json_is_fine(self):
        text = "다음은 요약입니다.\n```json\n" + json.dumps(perfect(self.m), ensure_ascii=False) + "\n```"
        self.assertTrue(ss.score(text, self.m)["format_ok"])

    def test_spacing_does_not_matter(self):
        p = perfect(self.m)
        p["decisions"] = [d.replace(" ", "") for d in p["decisions"]]
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["decisions_found"], s["decisions_total"])

    def test_one_entry_is_not_counted_for_two_items(self):
        p = perfect(self.m)
        p["decisions"] = [" ".join(p["decisions"])]  # everything in one line
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["decisions_found"], 1)

    def test_a_withdrawn_proposal_listed_as_decided_is_counted(self):
        p = perfect(self.m)
        w = self.m["key"]["withdrawn"][0]
        p["decisions"].append(w["text"])
        self.assertEqual(ss.score(json.dumps(p, ensure_ascii=False), self.m)["withdrawn_listed"], 1)

    def test_a_withdrawn_proposal_written_down_as_taken_back_is_not_counted(self):
        # "600만 원 취소 후 800만 원으로 확정" names the withdrawn amount and gets it right
        p = perfect(self.m)
        w = self.m["key"]["withdrawn"][0]
        for marked in [f"{w['text']} 취소 후 다시 확정", f"{w['text']} 제안이 거절됨", f"{w['text']} 대신 다른 안",
                       f"{w['text']}에서 900만 원으로 확정"]:
            q = {**p, "decisions": p["decisions"] + [marked]}
            self.assertEqual(ss.score(json.dumps(q, ensure_ascii=False), self.m)["withdrawn_listed"], 0, marked)

    def test_identical_key_items_count_once(self):
        # a meeting that plants the same item twice asks for it once
        m = meeting()
        totals = ss.score(json.dumps(perfect(m), ensure_ascii=False), m)
        m["key"]["decisions"].append(dict(m["key"]["decisions"][0]))
        m["key"]["withdrawn"].append(dict(m["key"]["withdrawn"][0]))
        p = perfect(m)
        p["decisions"].append(m["key"]["withdrawn"][0]["text"])
        s = ss.score(json.dumps(p, ensure_ascii=False), m)
        self.assertEqual(s["decisions_total"], totals["decisions_total"])
        self.assertEqual(s["decisions_found"], s["decisions_total"])
        self.assertEqual((s["withdrawn_total"], s["withdrawn_listed"]), (totals["withdrawn_total"], 1))

    def test_a_decision_with_the_right_topic_and_another_number_has_a_wrong_value(self):
        p = perfect(self.m)
        i, d = next((i, d) for i, d in enumerate(self.m["key"]["decisions"])
                    if any(re.search(r"\d", k) for k in d["keywords"]))
        p["decisions"][i] = re.sub(r"\d+", "4321", d["text"])
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["decisions_found"], s["decisions_total"] - 1)
        self.assertEqual(s["decisions_wrong_value"], 1)

    def test_a_decision_missing_its_number_is_missed_not_wrong(self):
        p = perfect(self.m)
        i, d = next((i, d) for i, d in enumerate(self.m["key"]["decisions"])
                    if any(re.search(r"\d", k) for k in d["keywords"]))
        p["decisions"][i] = re.sub(r"\d+\S*", "", d["text"])
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual((s["decisions_found"], s["decisions_wrong_value"]), (s["decisions_total"] - 1, 0))

    def test_a_right_summary_has_no_wrong_values(self):
        self.assertEqual(ss.score(json.dumps(perfect(self.m), ensure_ascii=False), self.m)["decisions_wrong_value"], 0)

    def test_a_wrong_owner_is_caught_but_the_action_still_found(self):
        p = perfect(self.m)
        k = self.m["key"]
        others = [x for x in k["participants"] if x != k["actions"][0]["owner"]]
        p["action_items"][0]["owner"] = others[0]
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["actions_found"], s["actions_total"])
        self.assertEqual(s["actions_owner_ok"], s["actions_total"] - 1)

    def test_an_action_with_the_wrong_deadline_is_not_that_action(self):
        # the deadline is what picks an action out; without it "점검표" could be any
        p = perfect(self.m)
        p["action_items"][1]["due"] = "언젠가"
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["actions_found"], s["actions_total"] - 1)

    def test_an_owner_written_with_honorifics_is_still_right(self):
        p = perfect(self.m)
        p["action_items"][0]["owner"] = p["action_items"][0]["owner"] + " 님"
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["actions_owner_ok"], s["actions_total"])

    def test_people_and_numbers_the_meeting_never_had_are_invented(self):
        p = perfect(self.m)
        stranger = next(x for x in mg.PEOPLE if x not in self.m["key"]["participants"])
        p["action_items"].append({"owner": stranger, "task": "보고서", "due": "다음 달"})
        p["summary"] = "예산은 98765만 원으로 정했다"
        s = ss.score(json.dumps(p, ensure_ascii=False), self.m)
        self.assertEqual(s["invented_people"], [stranger])
        self.assertEqual(s["unknown_owners"], [stranger])
        self.assertIn("98765", s["invented_numbers"])

    def test_an_owner_who_is_not_a_person_is_unknown(self):
        p = perfect(self.m)
        p["action_items"][0]["owner"] = "담당자 미정"
        self.assertEqual(ss.score(json.dumps(p, ensure_ascii=False), self.m)["unknown_owners"], ["담당자미정"])


if __name__ == "__main__":
    unittest.main()


class ActionMatchTests(unittest.TestCase):
    def test_an_action_is_found_by_its_deadline_and_task_word_however_it_is_worded(self):
        # "배포 전 점검표 작성" without the feature name is still that action: deadlines do
        # not repeat within a meeting
        m = meeting()
        a = m["key"]["actions"][0]
        word = a["keywords"][1]
        out = {"summary": "", "decisions": [], "open_questions": [],
               "action_items": [{"owner": a["owner"], "task": f"{word} 마무리", "due": a["due"]}]}
        s = ss.score(json.dumps(out, ensure_ascii=False), m)
        self.assertEqual(s["actions_found"], 1)
        self.assertEqual((s["actions_owner_ok"], s["actions_due_ok"]), (1, 1))

    def test_a_deadline_written_into_the_task_finds_the_action_but_not_the_due_field(self):
        m = meeting()
        a = m["key"]["actions"][0]
        out = {"summary": "", "decisions": [], "open_questions": [],
               "action_items": [{"owner": a["owner"], "task": f"{a['keywords'][1]} ({a['due']}까지)", "due": ""}]}
        s = ss.score(json.dumps(out, ensure_ascii=False), m)
        self.assertEqual((s["actions_found"], s["actions_due_ok"]), (1, 0))


class FieldTests(unittest.TestCase):
    def test_fields_a_caller_attached_are_not_scored(self):
        # chunked runs attach their own counters; a 0 there is not an invented number
        m = meeting()
        p = perfect(m)
        p["chunks_broken"] = 0
        p["chunks"] = 987654
        self.assertEqual(ss.score(json.dumps(p, ensure_ascii=False), m)["invented_numbers"], [])


class UnsupportedNumberTests(unittest.TestCase):
    SOURCE = "관리자 페이지 배포는 11월 11일로 확정하죠. 예산은 750만 원이고 개발자 2명을 뽑아요."

    def test_numbers_said_in_the_source_are_supported(self):
        for entry in ["관리자 페이지 배포를 11월 11일로 확정", "예산 750만 원", "개발자 2 명 채용", "배포 일정 확정"]:
            self.assertEqual(ss.unsupported_numbers(entry, self.SOURCE), [], entry)

    def test_a_number_the_source_never_said_is_unsupported(self):
        self.assertEqual(ss.unsupported_numbers("예산 250만 원", self.SOURCE), ["250만"])

    def test_a_number_inside_a_longer_one_is_not_support(self):
        # "1일" is not said because "11일" was
        self.assertEqual(ss.unsupported_numbers("배포를 11월 1일로 확정", self.SOURCE), ["1일"])
        self.assertEqual(ss.unsupported_numbers("50만 원", self.SOURCE), ["50만"])


class SplitAndFlagTests(unittest.TestCase):
    def test_decisions_planted_once_are_counted_apart_from_repeated_ones(self):
        m = meeting()
        d = m["key"]["decisions"]
        twin = next(x for x in d if re.search(r"\d", x["text"]))
        other = re.sub(r"\d+", "4321", twin["text"])
        d.append({"text": other, "keywords": [re.sub(r"\d+", "4321", k) for k in twin["keywords"]]})
        p = perfect(m)
        p["decisions"].remove(other)
        s = ss.score(json.dumps(p, ensure_ascii=False), m)
        self.assertEqual((s["decisions_once_total"], s["decisions_once_found"]), (len(d) - 2, len(d) - 2))
        self.assertEqual(s["decisions_found"], len(d) - 1)

    def test_flagged_decisions_are_checked_against_the_wrong_values(self):
        m = meeting()
        p = perfect(m)
        i, d = next((i, d) for i, d in enumerate(m["key"]["decisions"]) if any(re.search(r"\d", k) for k in d["keywords"]))
        p["decisions"][i] = re.sub(r"\d+", "4321", d["text"])
        p["unverified"] = [p["decisions"][i], p["decisions"][i - 1]]  # one wrong, one right
        s = ss.score(json.dumps(p, ensure_ascii=False), m)
        self.assertEqual((s["decisions_flagged"], s["flagged_wrong_value"]), (2, 1))


class BrokenFormatTotalsTests(unittest.TestCase):
    def test_a_broken_summary_still_reports_what_it_was_asked_for(self):
        # a run that broke the format counts as having found nothing, out of the same totals
        m = meeting()
        bad, good = ss.score("회의 잘 했습니다", m), ss.score(json.dumps(perfect(m), ensure_ascii=False), m)
        for k in good:
            if k.endswith("_total"):
                self.assertEqual(bad.get(k), good[k], k)


class EvidenceTests(unittest.TestCase):
    """SUM-3: each found item's evidence line, as the app's merge step points at it."""

    def setUp(self):
        self.m = meeting()
        self.p = perfect(self.m)
        lines = self.m["lines"]
        # the last line that says each item's value (or keywords), as a right merge would
        def at(kws):
            want = [k for k in kws if re.search(r"\d", k)] or kws
            return max(i for i, line in enumerate(lines) if ss.has_all(line, want))
        self.right = {
            "decisions": [at(d["keywords"]) for d in self.m["key"]["decisions"]],
            "action_items": [at(a["keywords"]) for a in self.m["key"]["actions"]],
            "open_questions": [None for _ in self.m["key"]["open_questions"]],
        }

    def test_evidence_at_the_line_that_says_the_item_is_right(self):
        s = ss.score(json.dumps({**self.p, "evidence": self.right}, ensure_ascii=False), self.m)
        self.assertEqual(s["decisions_evidence_ok"], s["decisions_found"])
        self.assertEqual(s["actions_evidence_ok"], s["actions_found"])

    def test_evidence_elsewhere_is_wrong(self):
        far = {f: [0 for _ in v] for f, v in self.right.items()}
        s = ss.score(json.dumps({**self.p, "evidence": far}, ensure_ascii=False), self.m)
        self.assertLess(s["decisions_evidence_ok"], s["decisions_found"])

    def test_without_evidence_there_is_no_count(self):
        s = ss.score(json.dumps(self.p, ensure_ascii=False), self.m)
        self.assertNotIn("decisions_evidence_ok", s)
