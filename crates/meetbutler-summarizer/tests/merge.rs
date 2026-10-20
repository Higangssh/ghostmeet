//! Joining the chunk summaries into one: the last word on an item wins, a deferral takes
//! its item's earlier decisions back to open questions, a chair's ruling outranks what
//! a chunk made of the discussion, and every item points at the transcript line it
//! came from (SUM-3). The scripts are invented; none is a real meeting.

use meetbutler_summarizer::merge::{merge, Part};
use meetbutler_summarizer::summary::{ActionItem, Summary};

fn lines(text: &[&str]) -> Vec<String> {
    text.iter().map(|s| s.to_string()).collect()
}

fn part(range: std::ops::Range<usize>, decisions: &[&str], open: &[&str]) -> Part {
    Part {
        lines: range,
        summary: Summary {
            decisions: decisions.iter().map(|s| s.to_string()).collect(),
            open_questions: open.iter().map(|s| s.to_string()).collect(),
            ..Summary::default()
        },
    }
}

#[test]
fn a_value_changed_in_a_later_chunk_replaces_the_earlier_one() {
    // the withdrawn first proposal (bench v2): 500 proposed in one chunk, 300 settled in the next
    let t = lines(&[
        "결제 화면 외주 건인데요. 처음 제안은 500만 원이었어요.",
        "그 금액이면 이번 분기 안에 맞출 수 있어요.",
        "검토해 보니 500만 원은 어렵겠습니다.",
        "그럼 결제 화면 외주 예산은 300만 원으로 확정하죠.",
    ]);
    let m = merge(
        &t,
        vec![
            part(0..2, &["결제 화면 외주 예산 500만 원 확정"], &[]),
            part(2..4, &["결제 화면 외주 예산을 300만 원으로 확정"], &[]),
        ],
    );
    assert_eq!(m.decisions, ["결제 화면 외주 예산을 300만 원으로 확정"]);
    assert_eq!(m.evidence.decisions, [Some(3)]);
}

#[test]
fn look_alike_items_about_different_things_are_all_kept() {
    let t = lines(&[
        "결제 화면 배포는 11월 3일로 확정하죠.",
        "검색 필터 배포는 12월 1일로 확정하죠.",
        "결제 화면 외주 예산은 300만 원으로 확정하죠.",
    ]);
    let m = merge(
        &t,
        vec![
            part(
                0..2,
                &[
                    "결제 화면 배포를 11월 3일로 확정",
                    "검색 필터 배포를 12월 1일로 확정",
                ],
                &[],
            ),
            part(2..3, &["결제 화면 외주 예산 300만 원 확정"], &[]),
        ],
    );
    assert_eq!(m.decisions.len(), 3, "{:?}", m.decisions);
    assert_eq!(m.evidence.decisions, [Some(0), Some(1), Some(2)]);
}

#[test]
fn a_deferral_takes_the_items_earlier_decision_back_to_open_questions() {
    let t = lines(&[
        "알림 설정 개편은 이번에 바로 가는 걸로 하죠.",
        "그런데 법무 검토가 아직 안 끝났네요.",
        "알림 설정 개편은 결론을 내지 못했으니 다음 회의에서 다시 보죠.",
    ]);
    let m = merge(
        &t,
        vec![
            part(0..1, &["알림 설정 개편안 진행 확정"], &[]),
            part(1..3, &["알림 설정 개편은 다음 회의에서 다시 논의"], &[]),
        ],
    );
    assert!(m.decisions.is_empty(), "{:?}", m.decisions);
    assert_eq!(
        m.open_questions,
        [
            "알림 설정 개편안 진행 확정",
            "알림 설정 개편은 다음 회의에서 다시 논의"
        ]
    );
}

#[test]
fn a_ruling_to_keep_reviewing_outranks_what_a_chunk_made_of_the_discussion() {
    // the committee case: a bill discussed across chunks, then held over by the chair
    let t = lines(&[
        "위원장 다음은 의사일정 제3항 가나다 특별법안입니다.",
        "위원 가나다 특별법은 제정이 필요하다고 봅니다.",
        "위원 위원회 구성은 더 따져 봐야 합니다.",
        "정부 부처 의견을 다음 소위에 제출하겠습니다.",
        "위원장 의사일정 제3항은 보다 심도 있는 논의를 위하여 소위에서 계속 심사하도록 하겠습니다.",
        "위원장 다음은 의사일정 제4항 라마바 법안입니다.",
        "위원 이견 없습니다.",
        "위원장 의사일정 제4항은 원안대로 의결하고자 하는데 이의 없으십니까? 가결되었음을 선포합니다.",
    ]);
    let m = merge(
        &t,
        vec![
            part(0..3, &["가나다 특별법 제정안을 통과시켰다"], &[]),
            part(
                3..8,
                &[
                    "부처가 다음 소위에 의견을 제출하기로 결정",
                    "라마바 법안 원안 가결",
                ],
                &[],
            ),
        ],
    );
    assert!(
        !m.decisions.iter().any(|d| d.contains("통과")),
        "a held-over bill must not read as passed: {:?}",
        m.decisions
    );
    assert!(
        m.open_questions.iter().any(|q| q.contains("가나다 특별법")),
        "{:?}",
        m.open_questions
    );
    assert!(
        m.open_questions.iter().any(|q| q.contains("계속 심사")),
        "{:?}",
        m.open_questions
    );
    assert!(
        m.decisions.iter().any(|d| d.contains("라마바")),
        "{:?}",
        m.decisions
    );
    // the ruling itself is there, pointing at its line
    let ruled = m
        .decisions
        .iter()
        .position(|d| d.contains("가결되었음을 선포"))
        .unwrap();
    assert_eq!(m.evidence.decisions[ruled], Some(7));
}

#[test]
fn without_rulings_or_deferrals_the_lists_are_joined_in_order_without_repeats() {
    let t = lines(&["가", "나", "다"]);
    let mut first = part(0..2, &["하나"], &["질문"]);
    first.summary.action_items = vec![ActionItem {
        owner: "지호".into(),
        task: "점검표 정리".into(),
        due: "11월 3일".into(),
    }];
    let m = merge(&t, vec![first, part(2..3, &["하나", "둘"], &["질문"])]);
    assert_eq!(m.decisions, ["하나", "둘"]);
    assert_eq!(m.open_questions, ["질문"]);
    assert_eq!(m.action_items.len(), 1);
    assert_eq!(m.evidence.decisions.len(), 2);
    assert_eq!(m.evidence.action_items.len(), 1);
    assert_eq!(m.evidence.open_questions.len(), 1);
}

#[test]
fn evidence_is_the_line_in_the_items_chunk_that_says_it() {
    let t = lines(&[
        "지난주 장애 얘기부터 하죠.",
        "지호 님이 배포 전 점검표를 11월 3일까지 정리해 주세요.",
        "도윤 님은 견적을 다시 받아 주세요.",
        "지호 님이 배포 전 점검표를 12월 1일까지 정리해 주세요.",
    ]);
    let mut first = part(0..3, &[], &[]);
    first.summary.action_items = vec![ActionItem {
        owner: "지호".into(),
        task: "배포 전 점검표 정리".into(),
        due: "11월 3일".into(),
    }];
    let mut second = part(3..4, &[], &[]);
    second.summary.action_items = vec![ActionItem {
        owner: "지호".into(),
        task: "배포 전 점검표 정리".into(),
        due: "12월 1일".into(),
    }];
    let m = merge(&t, vec![first, second]);
    // two tasks alike but for their dates, both kept, each at its own line
    assert_eq!(m.action_items.len(), 2);
    assert_eq!(m.evidence.action_items, [Some(1), Some(3)]);
}

#[test]
fn one_point_put_off_in_an_item_that_passed_takes_only_itself_to_open_questions() {
    // a bill passed with its clauses settled one by one, one clause left for later; the
    // clauses all carry the bill's long name, which must not make them one subject
    let t = lines(&[
        "위원장 다음은 의사일정 제1항 사바아 지원법 일부개정법률안입니다.",
        "위원 목적 조문에 지역 균형을 넣는 데 동의합니다.",
        "위원 시행일은 공포한 날로 하죠.",
        "위원 할인율 상향은 다음에 다시 보죠.",
        "위원장 의사일정 제1항은 수정한 대로 의결하고자 합니다. 가결되었음을 선포합니다.",
    ]);
    let m = merge(
        &t,
        vec![part(
            0..5,
            &[
                "사바아 지원법 일부개정법률안의 목적 조문에 지역 균형 추가",
                "사바아 지원법 일부개정법률안의 시행일을 공포한 날로 결정",
                "사바아 지원법 일부개정법률안의 할인율 상향은 최종 결정을 미룸",
            ],
            &[],
        )],
    );
    assert_eq!(
        m.open_questions,
        ["사바아 지원법 일부개정법률안의 할인율 상향은 최종 결정을 미룸"]
    );
    assert!(
        m.decisions.iter().any(|d| d.contains("목적 조문")),
        "{:?}",
        m.decisions
    );
    assert!(
        m.decisions.iter().any(|d| d.contains("시행일")),
        "{:?}",
        m.decisions
    );
    assert!(
        m.decisions.iter().any(|d| d.contains("가결되었음을 선포")),
        "{:?}",
        m.decisions
    );
}
