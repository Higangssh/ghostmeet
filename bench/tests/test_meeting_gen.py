"""The synthetic meetings: every planted fact is in the transcript, nothing else is.

    bench/.venv/bin/python -I -m unittest discover -s bench/tests -t bench
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import meeting_gen as mg  # noqa: E402


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


class MeetingTests(unittest.TestCase):
    def setUp(self):
        self.meetings = [mg.generate(kind, minutes) for kind in mg.KINDS for minutes in (30, 60)]

    def test_the_same_seed_gives_the_same_meeting(self):
        a, b = mg.generate("weekly", 30, seed=7), mg.generate("weekly", 30, seed=7)
        self.assertEqual(a.lines, b.lines)
        self.assertNotEqual(a.lines, mg.generate("weekly", 30, seed=8).lines)

    def test_a_meeting_is_about_as_long_as_asked(self):
        for m in self.meetings:
            self.assertAlmostEqual(m.estimated_minutes(), m.minutes, delta=m.minutes * 0.1, msg=m.kind)

    def test_every_planted_fact_is_said_in_the_meeting(self):
        for m in self.meetings:
            text = squash(m.transcript())
            k = m.key
            for item in k.decisions + k.open_questions + k.withdrawn:
                for kw in item.keywords:
                    self.assertIn(squash(kw), text, f"{m.kind}: {item.text}")
            for a in k.actions:
                self.assertIn(a.owner, k.participants)
                for kw in a.keywords + [a.due, f"{a.owner} 님"]:
                    self.assertIn(squash(kw), text, f"{m.kind}: {a.task}")

    def test_an_owner_is_named_out_loud_with_the_task_and_deadline(self):
        # no speaker labels: the name, the task and the date come in one line, as said
        for m in self.meetings:
            for a in m.key.actions:
                self.assertTrue(
                    any(f"{a.owner} 님" in line and a.due in line for line in m.lines), f"{m.kind}: {a.task}"
                )

    def test_a_withdrawn_proposal_is_taken_back_out_loud(self):
        for m in self.meetings:
            text = m.transcript()
            for w in m.key.withdrawn:
                at = text.find(w.keywords[-1])
                self.assertNotEqual(at, -1)
                later = text[at:]
                self.assertTrue(any(word in later for word in ("취소", "철회", "없던 걸로")), w.text)

    def test_deadlines_never_repeat_within_a_meeting(self):
        for m in self.meetings:
            dues = [a.due for a in m.key.actions]
            self.assertEqual(len(dues), len(set(dues)), m.kind)

    def test_only_participants_are_named(self):
        for m in self.meetings:
            text = m.transcript()
            for p in mg.PEOPLE:
                if p not in m.key.participants:
                    self.assertNotIn(p, text, f"{m.kind} names {p}, who is not there")

    def test_every_number_said_is_in_the_key_or_a_date(self):
        # so a number in a summary that is not in the transcript was invented
        for m in self.meetings:
            planted = set(m.key.numbers)
            dates = {n for a in m.key.actions for n in re.findall(r"\d+", a.due)}
            dates |= {n for d in m.key.decisions for k in d.keywords for n in re.findall(r"\d+", k) if "월" in k}
            for n in re.findall(r"\d+", m.transcript()):
                self.assertTrue(n in planted or n in dates, f"{m.kind}: {n} is not planted")

    def test_there_is_no_speaker_label(self):
        for m in self.meetings:
            for line in m.lines:
                self.assertIsNone(re.match(r"^\S+\s*:", line), line)

    def test_every_kind_plants_each_kind_of_fact(self):
        for m in self.meetings:
            k = m.key
            self.assertTrue(k.decisions and k.actions and k.open_questions and k.withdrawn, m.kind)


if __name__ == "__main__":
    unittest.main()
