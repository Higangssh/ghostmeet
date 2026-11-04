// A made-up meeting for the README's screenshots and demo. Every name, line and number
// here is invented; nothing comes from a recording. A small team (지호, 도윤 and "나")
// sets the release date of an address-book sync feature in a fictional app.
//
// Dates are relative to the capture clock in capture.mjs (Wednesday 2026-11-04 14:00 KST).

window.SCENARIO = {
  vocabulary: ["지호", "도윤", "주소록 동기화"],

  // meetings already in the library, oldest first
  past: [
    { id: "m1", title: "주간 제품 회의", at: "2026-10-26T10:00:00+09:00", min: 42 },
    { id: "m2", title: "디자인 리뷰 · 온보딩 화면", at: "2026-10-28T15:30:00+09:00", min: 35 },
    { id: "m3", title: "1:1 지호", at: "2026-10-30T11:00:00+09:00", min: 24 },
    { id: "m4", title: "주간 제품 회의", at: "2026-11-02T10:00:00+09:00", min: 51 },
  ],
  pastTranscript: [
    { start: 3, channel: "me", text: "지난주 지표부터 볼게요." },
    { start: 9, channel: "them", text: "가입 전환율이 조금 올랐어요." },
    { start: 16, channel: "them", text: "온보딩 두 번째 화면을 줄인 효과 같아요." },
    { start: 24, channel: "me", text: "그 실험은 한 주 더 보죠." },
  ],
  pastSummary: {
    summary: "가입 전환율이 올랐고, 온보딩을 줄인 실험을 한 주 더 보기로 했다.",
    decisions: ["온보딩 실험을 한 주 연장한다"],
    action_items: [],
    open_questions: [],
    evidence: { decisions: [3], action_items: [], open_questions: [] },
  },

  // the meeting the demo records: [seconds, channel, live caption, final transcript]
  // the live line is the small model's provisional caption; the final pass corrects it
  lines: [
    [4, "me", "오늘은 주소록 동기와 출시 일정만 정하고 끝낼게요.", "오늘은 주소록 동기화 출시 일정만 정하고 끝낼게요."],
    [11, "them", "네, 지호예요. 베타에서 충돌 보고가 두건 남아 있어요.", "네, 지호예요. 베타에서 충돌 보고가 두 건 남아 있어요."],
    [19, "them", "둘 다 오래된 안드로이드 버전에서만 나요.", "둘 다 오래된 안드로이드 버전에서만 나요."],
    [27, "me", "그럼 그 두 건 고치고 출시하는 걸로 하죠.", "그럼 그 두 건 고치고 출시하는 걸로 하죠."],
    [35, "them", "도윤입니다. 수정은 목요일까지 가능해요.", "도윤입니다. 수정은 목요일까지 가능해요."],
    [44, "me", "출시는 다음 주 화요일, 11월 10일로 잡겠습니다.", "출시는 다음 주 화요일, 11월 10일로 잡겠습니다."],
    [52, "them", "공지 문구는 누가 쓰나요?", "공지 문구는 누가 쓰나요?"],
    [60, "me", "제가 금요일까지 초안 올릴게요.", "제가 금요일까지 초안 올릴게요."],
    [68, "them", "오래된 버전 사용자한테 따로 안내할지는 정해야 해요.", "오래된 버전 사용자한테 따로 안내할지는 정해야 해요."],
    [75, "me", "그건 다음 회의에서 숫자 보고 정하죠.", "그건 다음 회의에서 숫자 보고 정하죠."],
  ],
  duration: 79,
  title: "주소록 동기화 출시 일정",

  // the fixed summary format (SUM-1), with each item's evidence line (SUM-3)
  summary: {
    summary: "베타에서 남은 충돌 두 건을 고친 뒤 주소록 동기화를 11월 10일에 출시하기로 했다.",
    decisions: ["오래된 안드로이드 버전의 충돌 두 건을 고친 뒤 출시한다", "출시일은 11월 10일(화)"],
    action_items: [
      { owner: "도윤", task: "충돌 두 건 수정", due: "11월 5일(목)" },
      { owner: "나", task: "출시 공지 초안", due: "11월 6일(금)" },
    ],
    open_questions: ["오래된 버전 사용자에게 따로 안내할지"],
    evidence: { decisions: [3, 5], action_items: [4, 7], open_questions: [8] },
  },
};
