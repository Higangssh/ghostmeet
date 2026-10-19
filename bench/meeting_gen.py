"""Synthetic Korean meetings with an answer key, for scoring local summary models.

    bench/.venv/bin/python -I bench/meeting_gen.py            # writes bench/data/meetings/

Real meeting transcripts cannot be used, so meetings are assembled from made-up parts:
agenda items with a decision, owners and deadlines for follow-ups, numbers, a question
left open, a proposal that is withdrawn later in the meeting, short replies and small
talk. Every fact the summary should contain is planted, so the key knows it; nothing
the key does not plant (a person, a date, an amount) exists in the meeting. People,
companies and products are invented.

The app does not know who is speaking (0.2 has only the me/them channel), so the
transcript has no speaker labels and owners are named out loud, as in a real meeting.
Generation is deterministic per (kind, minutes, seed). Script text stays in the ignored
bench/data/; only this generator, the key format and scores are committed.
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "meetings"

# Speech rate used to size a meeting: Hangul syllables per minute of read-aloud speech.
# Calibrated against the macOS Yuna voice when the audio is made; see SYLLABLES_PER_MIN use.
SYLLABLES_PER_MIN = 300

KINDS = ("weekly", "one_on_one", "customer")
LENGTHS = (30, 60, 120)

# --- invented parts ------------------------------------------------------------------

# Given names only, spoken with 님 as colleagues do. Surnames are used once, in introductions.
PEOPLE = ["하늘", "도윤", "서준", "유나", "지호", "민서", "태오", "수아", "건우", "다인", "시온", "예린"]
SURNAMES = ["윤", "장", "임", "한", "오", "신", "권", "황", "안", "송", "류", "홍"]
# Invented company names for the customer meeting (ours, theirs).
OUR_COMPANY = "누리온"
CLIENTS = ["바람결상사", "푸른솔물류", "새벽빛제약", "다솜유통"]
FEATURES = [
    "결제 화면", "알림 설정", "검색 필터", "회원 가입 절차", "관리자 페이지", "영수증 내보내기",
    "다크 모드", "오프라인 저장", "공유 링크", "사진 업로드", "예약 기능", "리포트 대시보드",
    "쿠폰 적용", "배송 조회", "주소록 동기화", "음성 메모", "일정 공유", "파일 미리보기",
    "팀 초대", "사용량 통계", "비밀번호 재설정", "언어 설정", "접근성 옵션", "데이터 백업",
]
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일"]
MONTHS = [10, 11, 12]

SMALL_TALK = [
    "오늘 아침에 비가 많이 와서 길이 엄청 막히더라고요.",
    "점심에 새로 생긴 국숫집 가 보셨어요? 생각보다 괜찮았어요.",
    "주말에 다들 잘 쉬셨어요? 저는 집에서 계속 잤어요.",
    "회의실 에어컨이 또 너무 세네요. 혹시 추우신 분 계세요?",
    "커피 드실 분 계시면 말씀해 주세요. 제가 내려갈 때 사 올게요.",
    "요즘 날씨가 갑자기 쌀쌀해져서 감기 걸리기 딱 좋은 것 같아요.",
]
SHORT_REPLIES = ["네.", "맞아요.", "그렇죠.", "좋습니다.", "아, 네.", "음, 그렇네요.", "알겠습니다.", "글쎄요."]
FILLERS = [
    "그 부분은 지난번에도 얘기가 나왔던 것 같은데요.",
    "제가 이해한 게 맞는지 한번 확인해 보고 싶은데요.",
    "현장에서 들은 얘기로는 사용자들이 꽤 헷갈려 한다고 하더라고요.",
    "일정이 빠듯하긴 한데 아예 못 할 정도는 아닌 것 같아요.",
    "그건 우선순위를 다시 봐야 할 것 같습니다.",
    "자료는 회의 끝나고 공유 폴더에 올려 두겠습니다.",
    "이 부분은 숫자를 조금 더 봐야 판단할 수 있을 것 같아요.",
    "다른 팀 의견도 한번 들어 보면 좋겠네요.",
]

# Talk around an item that plants nothing: no names, numbers, dates or products, so the
# key stays complete however much of it a meeting has.
DISCUSSION = [
    "이건 사용자 입장에서 한번 생각해 봐야 할 것 같아요.",
    "지난번 설문에서도 비슷한 얘기가 나왔었죠.",
    "기술적으로는 어렵지 않은데 테스트가 좀 오래 걸릴 거예요.",
    "솔직히 저는 이게 그렇게 급한지 잘 모르겠어요.",
    "다른 회사들은 이걸 어떻게 하는지 찾아본 사람 있어요?",
    "제가 찾아봤는데 다들 방식이 조금씩 다르더라고요.",
    "그럼 우리한테 맞는 방식이 뭔지가 중요하겠네요.",
    "운영팀에서는 문의가 줄어들 거라고 기대하고 있어요.",
    "문서도 같이 바꿔야 하니까 그 시간도 생각해 주세요.",
    "디자인 쪽 의견은 어떠세요?",
    "디자인은 큰 문제 없을 것 같고, 문구만 좀 다듬으면 될 것 같아요.",
    "혹시 예전 버전을 쓰는 사람들은 어떻게 되나요?",
    "그분들은 당분간 그대로 쓰시게 하고 천천히 옮기면 될 것 같아요.",
    "이거 하다가 다른 기능이 깨지는 일은 없겠죠?",
    "그래서 이번에는 자동 테스트를 먼저 붙이자는 얘기가 나왔어요.",
    "좋은 생각이에요. 지난번처럼 급하게 고치는 일은 없어야죠.",
    "현실적으로 이번 달 안에 다 하기는 어려울 수도 있어요.",
    "그래도 핵심만 먼저 하고 나머지는 나눠서 하면 되지 않을까요?",
    "고객 입장에서는 그게 제일 눈에 띄는 변화일 거예요.",
    "제가 보기엔 방향은 맞는데 속도를 조절할 필요는 있어 보여요.",
    "다들 비슷하게 생각하시는 것 같네요.",
    "한 가지만 더 확인하고 싶은데, 이건 누가 최종 결정하나요?",
    "그건 이 자리에서 정하면 될 것 같아요.",
    "아, 그리고 예전에 정리해 둔 자료가 있는데 찾아서 보내 드릴게요.",
    "그 자료 보면 이해가 훨씬 빠를 거예요.",
    "말씀하신 대로 하면 사용자 불편은 확실히 줄어들 것 같네요.",
    "다만 비용이 얼마나 들지는 아직 감이 잘 안 와요.",
    "그건 대략적으로라도 따로 계산해 보는 게 좋겠어요.",
    "회의 시간이 좀 남았으니까 이 얘기는 조금 더 해 보죠.",
    "저는 처음에 반대했는데, 얘기 듣다 보니 괜찮은 것 같아요.",
    "근데 이걸 지금 바꾸면 기존에 하던 일정이 다 밀리지 않을까요?",
    "밀리는 건 맞는데, 나중에 고치는 것보다는 지금이 싸게 먹혀요.",
    "그 말도 일리가 있네요. 저도 비슷한 경험이 있어요.",
    "예전에 한 번 미뤘다가 결국 두 배로 고생했던 기억이 나네요.",
    "그때는 사람도 지금보다 적었으니까 상황이 좀 다르긴 하죠.",
    "요즘 들어오는 문의를 보면 이 문제가 제일 많긴 해요.",
    "문의 내용이 대부분 비슷해서 안내만 잘해도 많이 줄 것 같아요.",
    "안내 문구는 제가 초안을 한번 잡아 볼 수 있어요.",
    "좋아요, 그건 이따가 따로 얘기해요.",
    "화면을 보면서 얘기하면 더 빠를 것 같은데 공유해 주실 수 있어요?",
    "네, 지금 띄울게요. 잘 보이세요?",
    "네, 잘 보여요. 여기 이 부분 말씀하시는 거죠?",
    "맞아요, 여기서 사람들이 제일 많이 멈춰요.",
    "그럼 여기 단계를 하나 줄이는 게 핵심이겠네요.",
    "단계를 줄이면 실수로 누르는 경우가 생길 수도 있어요.",
    "그건 확인 창을 하나 두면 해결될 것 같아요.",
    "확인 창이 너무 많으면 또 귀찮아하실 텐데요.",
    "어느 정도 선이 맞는지는 몇 명한테 직접 써 보게 하면 알 수 있겠죠.",
    "그 테스트는 오래 안 걸리니까 해 볼 만하네요.",
    "다들 다른 의견 없으시면 이쯤에서 정리할게요.",
]


@dataclass
class Decision:
    text: str
    keywords: list[str]


@dataclass
class Action:
    owner: str
    task: str
    due: str
    # the deadline and the word that names the task: deadlines never repeat within a
    # meeting, so these two pick out one action however the summary words it
    keywords: list[str]


@dataclass
class Key:
    participants: list[str]
    decisions: list[Decision] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    open_questions: list[Decision] = field(default_factory=list)
    # proposals made and then withdrawn in the meeting: a summary that lists one as
    # decided got the meeting wrong
    withdrawn: list[Decision] = field(default_factory=list)
    # every number the meeting says out loud; a number in a summary that is not here
    # (or a date) was invented
    numbers: list[str] = field(default_factory=list)


@dataclass
class Meeting:
    kind: str
    minutes: int
    seed: int
    lines: list[str]
    key: Key

    def transcript(self) -> str:
        return "\n".join(self.lines)

    def syllables(self) -> int:
        return syllables(self.transcript())

    def estimated_minutes(self) -> float:
        return self.syllables() / SYLLABLES_PER_MIN


def syllables(text: str) -> int:
    return sum(1 for c in text if "가" <= c <= "힣")


# --- agenda items --------------------------------------------------------------------


class Builder:
    """Draws invented values without repeating one, so every planted fact is unique."""

    def __init__(self, rng: random.Random, people: list[str]):
        self.rng = rng
        self.people = people
        self.features = rng.sample(FEATURES, len(FEATURES))
        self.used_dates: set[str] = set()
        self.used_numbers: set[int] = set()

    def feature(self) -> str:
        if not self.features:
            self.features = self.rng.sample(FEATURES, len(FEATURES))
        return self.features.pop()

    def person(self, exclude: tuple[str, ...] = ()) -> str:
        return self.rng.choice([p for p in self.people if p not in exclude])

    def date(self) -> str:
        """A deadline not used yet. Dates are what tie an action to its owner in the
        key, so they never repeat; there are over eighty to draw from."""
        while True:
            if self.rng.random() < 0.5:
                d = f"{self.rng.choice(MONTHS)}월 {self.rng.randint(1, 28)}일"
            else:
                d = f"다음 주 {self.rng.choice(WEEKDAYS)}"
            if d not in self.used_dates:
                self.used_dates.add(d)
                return d

    def number(self, lo: int, hi: int, step: int = 1) -> int:
        """A number not used yet when the range still has one, else any in range: a small
        range (head counts, days off) runs out in a long meeting, and a repeated number
        is still a number the meeting said."""
        free = [n for n in range(lo, hi, step) if n not in self.used_numbers]
        n = self.rng.choice(free) if free else self.rng.randrange(lo, hi, step)
        self.used_numbers.add(n)
        return n


def item_release(b: Builder, key: Key, lead: str) -> list[str]:
    f, date, owner = b.feature(), b.date(), b.person((lead,))
    bugs = b.number(3, 30)
    key.decisions.append(Decision(f"{f} 배포를 {date}로 확정", [f, date]))
    key.actions.append(Action(owner, f"{f} 배포 전 점검표 정리", date, [date, "점검표"]))
    key.numbers.append(str(bugs))
    return [
        f"다음은 {f} 배포 일정입니다. 지금 남은 버그가 {bugs}개 정도예요.",
        f"{bugs}개 중에 급한 건 몇 개 안 되고, 나머지는 배포 뒤에 고쳐도 될 것 같아요.",
        "그럼 일정은 그대로 가도 괜찮을까요?",
        f"네, 그러면 {f} 배포는 {date}로 확정하죠.",
        f"{owner} 님이 배포 전 점검표를 {date}까지 정리해 주실 수 있을까요?",
        "네, 제가 맡겠습니다.",
    ]


def item_budget(b: Builder, key: Key, lead: str) -> list[str]:
    f = b.feature()
    first, final = b.number(300, 900, 50), b.number(200, 900, 50)
    owner, date = b.person((lead,)), b.date()
    key.withdrawn.append(Decision(f"{f} 외주 예산 {first}만 원", [f, f"{first}만 원"]))
    key.decisions.append(Decision(f"{f} 외주 예산을 {final}만 원으로 확정", [f, f"{final}만 원"]))
    key.actions.append(Action(owner, f"{f} 외주 업체에 견적 다시 요청", date, [date, "견적"]))
    key.numbers += [str(first), str(final)]
    return [
        f"{f} 외주 건인데요, 처음 제안은 {first}만 원이었어요.",
        f"일단 {first}만 원으로 진행하는 걸로 하죠.",
        "잠깐만요, 그 금액이면 이번 분기 예산을 넘지 않나요?",
        "아, 맞네요. 방금 말한 건 취소할게요.",
        f"{first}만 원은 없던 걸로 하고, {final}만 원으로 확정하겠습니다.",
        f"{owner} 님이 업체에 견적을 {date}까지 다시 받아 주세요.",
        "알겠습니다.",
    ]


def item_metrics(b: Builder, key: Key, lead: str) -> list[str]:
    f = b.feature()
    before, after = b.number(5, 40), b.number(41, 80)
    key.numbers += [str(before), str(after)]
    key.decisions.append(Decision(f"{f} 개편안을 유지", [f, "유지"]))
    return [
        f"{f} 개편 뒤에 사용률이 {before}퍼센트에서 {after}퍼센트로 올랐습니다.",
        "생각보다 많이 올랐네요. 표본이 충분한 건가요?",
        "이 주 정도 데이터라 아직 이르긴 한데 방향은 맞는 것 같아요.",
        f"그럼 {f} 개편안은 그대로 유지하죠.",
    ]


def item_question(b: Builder, key: Key, lead: str) -> list[str]:
    f = b.feature()
    q = f"{f}의 개인정보 보관 기간을 몇 년으로 할지"
    key.open_questions.append(Decision(q, [f, "보관"]))
    return [
        f"하나 더 있는데요, {f}에서 개인정보를 얼마나 보관할지 아직 안 정했어요.",
        "법무팀 의견을 먼저 들어야 할 것 같은데요.",
        "맞아요. 이건 오늘 정하기는 어렵겠네요.",
        "그럼 보관 기간은 다음 회의 때 다시 보죠.",
    ]


def item_hiring(b: Builder, key: Key, lead: str) -> list[str]:
    n, owner, date = b.number(1, 4), b.person((lead,)), b.date()
    key.numbers.append(str(n))
    key.decisions.append(Decision(f"개발자 {n}명 추가 채용 진행", ["채용", f"{n}명"]))
    key.actions.append(Action(owner, "채용 공고 초안 작성", date, [date, "채용 공고"]))
    return [
        "인력 얘기도 해야 할 것 같아요. 지금 인원으로는 다음 분기가 힘들어요.",
        f"몇 명 정도 필요하다고 보세요?",
        f"최소 {n}명은 있어야 할 것 같습니다.",
        f"좋아요, 개발자 {n}명 추가 채용을 진행하는 걸로 하죠.",
        f"{owner} 님이 채용 공고 초안을 {date}까지 써 주세요.",
        "네, 알겠습니다.",
    ]


def item_incident(b: Builder, key: Key, lead: str) -> list[str]:
    f, minutes, owner, date = b.feature(), b.number(10, 90), b.person((lead,)), b.date()
    key.numbers.append(str(minutes))
    key.decisions.append(Decision(f"{f} 장애 재발 방지를 위해 모니터링 경보 추가", [f, "경보"]))
    key.actions.append(Action(owner, f"{f} 장애 보고서 작성", date, [date, "보고서"]))
    return [
        f"지난주에 {f}가 {minutes}분 동안 멈췄던 건 다들 아시죠.",
        "원인은 설정 파일이 잘못 올라간 거였어요.",
        "그럼 다음에 같은 일이 생기면 바로 알 수 있게 해야겠네요.",
        f"네, {f}에 모니터링 경보를 추가하기로 하죠.",
        f"{owner} 님이 장애 보고서를 {date}까지 정리해 주세요.",
        "네.",
    ]


# 1:1
def item_goal(b: Builder, key: Key, lead: str) -> list[str]:
    f, owner, date = b.feature(), b.person((lead,)), b.date()
    key.decisions.append(Decision(f"다음 분기 목표를 {f} 개선으로 설정", [f, "목표"]))
    key.actions.append(Action(owner, f"{f} 개선 계획서 작성", date, [date, "계획서"]))
    return [
        "다음 분기 목표를 같이 정해 보죠. 어떤 걸 해 보고 싶어요?",
        f"저는 {f}를 제대로 고쳐 보고 싶어요. 불만이 제일 많이 들어오는 부분이라서요.",
        f"좋아요. 그럼 다음 분기 목표는 {f} 개선으로 하죠.",
        f"{owner} 님, 개선 계획서를 {date}까지 한 장으로 정리해 줄래요?",
        "네, 그렇게 할게요.",
    ]


def item_vacation(b: Builder, key: Key, lead: str) -> list[str]:
    date, days = b.date(), b.number(2, 6)
    person = b.person((lead,))
    key.numbers.append(str(days))
    key.decisions.append(Decision(f"{person} 님 휴가 {date}부터 {days}일 승인", ["휴가", date, f"{days}일"]))
    return [
        f"아, 그리고 휴가 신청한 거 봤어요. {date}부터 {days}일 맞죠?",
        "네, 맞아요. 그때 일이 좀 몰려 있어서 걱정이긴 한데요.",
        "괜찮아요. 미리 인수인계만 해 주면 돼요. 휴가는 승인할게요.",
        "감사합니다.",
    ]


def item_training(b: Builder, key: Key, lead: str) -> list[str]:
    amount, owner, date = b.number(20, 150, 10), b.person((lead,)), b.date()
    key.numbers.append(str(amount))
    key.withdrawn.append(Decision("외부 컨퍼런스 참가", ["컨퍼런스"]))
    key.decisions.append(Decision(f"온라인 강의 수강비 {amount}만 원 지원", ["강의", f"{amount}만 원"]))
    key.actions.append(Action(owner, "수강할 강의 목록 공유", date, [date, "강의 목록"]))
    return [
        "교육 지원 얘기도 해 볼까요? 지난번에 컨퍼런스 가고 싶다고 했잖아요.",
        "네, 외부 컨퍼런스에 가 보고 싶었어요.",
        "그럼 일단 컨퍼런스 참가로 진행해 보죠.",
        "아, 그런데 그 날짜가 출시랑 겹치네요. 컨퍼런스는 취소하는 게 좋겠어요.",
        f"대신 온라인 강의 수강비를 {amount}만 원까지 지원할게요.",
        f"{owner} 님이 들을 강의 목록을 {date}까지 공유해 주세요.",
        "좋습니다.",
    ]


def item_feedback(b: Builder, key: Key, lead: str) -> list[str]:
    q = "코드 리뷰 담당을 누구에게 넘길지"
    key.open_questions.append(Decision(q, ["코드 리뷰"]))
    return [
        "요즘 코드 리뷰가 너무 몰려서 다른 일을 못 하고 있어요.",
        "그건 저도 느꼈어요. 누구한테 나눠 줄 수 있을지 생각해 봐야겠네요.",
        "당장 맡길 사람이 떠오르진 않네요.",
        "이건 제가 조금 더 고민해 보고 다음에 다시 얘기하죠.",
    ]


# customer
def item_price(b: Builder, key: Key, lead: str) -> list[str]:
    asked, agreed = b.number(5000, 9000, 500), b.number(3000, 8000, 500)
    key.numbers += [str(asked), str(agreed)]
    key.withdrawn.append(Decision(f"건당 단가 {asked}원", [f"{asked}원"]))
    key.decisions.append(Decision(f"건당 단가 {agreed}원으로 합의", ["단가", f"{agreed}원"]))
    return [
        f"단가 말씀드리면, 저희 쪽 제안은 건당 {asked}원입니다.",
        "음, 그 가격이면 내부 승인이 어려울 것 같아요.",
        f"그러면 {asked}원은 철회하고, 물량을 고려해서 건당 {agreed}원으로 하겠습니다.",
        f"네, {agreed}원이면 진행할 수 있겠네요. 그걸로 합의하죠.",
    ]


def item_timeline(b: Builder, key: Key, lead: str) -> list[str]:
    f, date, owner = b.feature(), b.date(), b.person((lead,))
    key.decisions.append(Decision(f"{f} 시범 운영을 {date}에 시작", [f, "시범", date]))
    key.actions.append(Action(owner, f"{f} 시범 운영 계정 발급", date, [date, "계정"]))
    return [
        f"{f}는 언제부터 써 볼 수 있을까요?",
        f"{date}부터 시범 운영이 가능합니다.",
        f"좋습니다. 그럼 {f} 시범 운영은 {date}에 시작하는 걸로 하죠.",
        f"{owner} 님이 {date} 전까지 시범 운영 계정을 발급해 드릴 겁니다.",
        "감사합니다.",
    ]


def item_sla(b: Builder, key: Key, lead: str) -> list[str]:
    hours = b.number(2, 24)
    key.numbers.append(str(hours))
    key.decisions.append(Decision(f"장애 대응 시간 {hours}시간 이내로 계약서에 명시", ["장애", f"{hours}시간"]))
    q = "주말 장애 대응 비용을 누가 부담할지"
    key.open_questions.append(Decision(q, ["주말", "비용"]))
    return [
        "장애가 났을 때 얼마나 빨리 대응해 주실 수 있나요?",
        f"평일 기준으로 {hours}시간 이내에 대응합니다.",
        f"그럼 {hours}시간 이내 대응을 계약서에 넣어 주세요.",
        "네, 반영하겠습니다. 다만 주말 대응 비용은 따로 논의가 필요할 것 같아요.",
        "그 부분은 내부에서 확인하고 다시 연락드릴게요.",
    ]


ITEMS = {
    "weekly": [item_release, item_budget, item_metrics, item_question, item_hiring, item_incident],
    "one_on_one": [item_goal, item_vacation, item_training, item_feedback, item_release, item_metrics],
    "customer": [item_price, item_timeline, item_sla, item_release, item_question, item_metrics],
}


# --- assembling a meeting ------------------------------------------------------------


def participants(kind: str, rng: random.Random) -> list[str]:
    n = {"weekly": 5, "one_on_one": 2, "customer": 4}[kind]
    return rng.sample(PEOPLE, n)


def generate(kind: str, minutes: int, seed: int = 0) -> Meeting:
    rng = random.Random(f"{kind}-{minutes}-{seed}")
    people = participants(kind, rng)
    lead = people[0]
    key = Key(participants=people)
    b = Builder(rng, people)
    lines: list[str] = []

    surnames = rng.sample(SURNAMES, len(people))
    if kind == "customer":
        client = rng.choice(CLIENTS)
        lines += [
            f"안녕하세요, {OUR_COMPANY}의 {surnames[0]}{lead}입니다. 시간 내 주셔서 감사합니다.",
            f"네, 안녕하세요. {client}의 {surnames[2]}{people[2]}이고 이쪽은 {surnames[3]}{people[3]}입니다.",
        ]
    else:
        lines.append(f"자, 다들 오셨으면 시작할게요. 오늘 진행은 {lead}가 맡겠습니다.")
    lines.append(rng.choice(SMALL_TALK))
    lines.append(rng.choice(SHORT_REPLIES))

    target = minutes * SYLLABLES_PER_MIN
    items = ITEMS[kind]
    order: list = []
    while syllables("\n".join(lines)) < target * 0.92:
        if not order:
            order = rng.sample(items, len(items))
        block = order.pop()(b, key, lead)
        # talk the item over before it is settled: after its opening line, a stretch
        # of discussion that plants nothing
        # (kept in bank order, so a question still comes before its answer)
        picked = sorted(rng.sample(range(len(DISCUSSION)), rng.randint(14, 26)))
        talk = [DISCUSSION[i] for i in picked]
        block = block[:1] + talk + block[1:]
        for line in block:
            lines.append(line)
            r = rng.random()
            if r < 0.25:
                lines.append(rng.choice(SHORT_REPLIES))
            elif r < 0.40:
                lines.append(rng.choice(FILLERS))
        if rng.random() < 0.3:
            lines.append(rng.choice(SMALL_TALK))
    lines.append("오늘은 여기까지 하죠. 다들 수고하셨습니다.")
    lines.append(rng.choice(["수고하셨습니다.", "네, 감사합니다."]))
    return Meeting(kind, minutes, seed, lines, key)


def to_json(m: Meeting) -> dict:
    return {"kind": m.kind, "minutes": m.minutes, "seed": m.seed, "lines": m.lines, "key": asdict(m.key)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        for minutes in LENGTHS:
            m = generate(kind, minutes)
            path = OUT / f"{kind}-{minutes}m.json"
            path.write_text(json.dumps(to_json(m), ensure_ascii=False, indent=1))
            k = m.key
            print(f"{path.name}: {len(m.lines)} lines, ~{m.estimated_minutes():.0f} min, "
                  f"{len(k.decisions)} decisions, {len(k.actions)} actions, "
                  f"{len(k.open_questions)} open, {len(k.withdrawn)} withdrawn", file=sys.stderr)


if __name__ == "__main__":
    main()
