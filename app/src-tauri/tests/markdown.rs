//! OUT-1: a meeting as Markdown, for the clipboard or a file. Synthetic meeting text only.

use meetbutler_app::markdown::markdown;
use meetbutler_engine::incremental::Segment;
use meetbutler_engine::store::Session;
use meetbutler_summarizer::merge::Evidence;
use meetbutler_summarizer::summary::{ActionItem, Summary};

fn seg(text: &str, start: f64, channel: &str) -> Segment {
    let mut s = Segment::new(text, start, start + 2.0);
    s.channel = Some(channel.into());
    s
}

fn session() -> Session {
    Session {
        title: "주간 회의".into(),
        duration_sec: 2520.0,
        sources: vec!["mic".into(), "system".into()],
        ..Session::new("m1", 1.0)
    }
}

fn transcript() -> Vec<Segment> {
    vec![
        seg("결제 화면 배포 얘기부터 하죠.", 3.0, "me"),
        seg("그럼 11월 3일로 확정하죠.", 754.5, "them"),
        seg("지호 님이 점검표를 정리해 주세요.", 3725.0, "me"),
    ]
}

fn summary() -> Summary {
    Summary {
        summary: "배포 일정을 정했다.".into(),
        decisions: vec!["결제 화면 배포를 11월 3일로 확정".into()],
        action_items: vec![ActionItem {
            owner: "지호".into(),
            task: "점검표 정리".into(),
            due: "".into(),
        }],
        open_questions: vec!["보관 기간\n몇 년으로 할지".into()],
        evidence: Evidence {
            decisions: vec![Some(1)],
            action_items: vec![Some(2)],
            open_questions: vec![None],
        },
        ..Summary::default()
    }
}

#[test]
fn a_meeting_reads_title_facts_summary_lists_and_transcript_in_that_order() {
    let md = markdown(
        &session(),
        "2026년 10월 20일 오후 2:03",
        &transcript(),
        Some(&summary()),
    );
    let at = |s: &str| md.find(s).unwrap_or_else(|| panic!("no {s:?} in\n{md}"));
    assert!(md.starts_with("# 주간 회의\n"));
    assert!(at("2026년 10월 20일 오후 2:03") < at("## 요약"));
    assert!(md.contains("42분"));
    assert!(md.contains("마이크(나), 시스템 오디오(상대)"));
    assert!(at("## 요약") < at("## 결정") && at("## 결정") < at("## 할 일"));
    assert!(at("## 할 일") < at("## 미해결 질문") && at("## 미해결 질문") < at("## 전사"));
    assert!(md.contains("배포 일정을 정했다."));
}

#[test]
fn each_item_carries_the_time_of_its_evidence() {
    let md = markdown(&session(), "d", &transcript(), Some(&summary()));
    assert!(
        md.contains("- 결제 화면 배포를 11월 3일로 확정 (12:34)"),
        "{md}"
    );
    assert!(md.contains("- [ ] 지호: 점검표 정리 (1:02:05)"), "{md}");
    // no evidence, no time; an item stays on one line
    assert!(md.contains("- 보관 기간 몇 년으로 할지\n"), "{md}");
}

#[test]
fn the_transcript_lists_time_and_side_for_every_segment() {
    let md = markdown(&session(), "d", &transcript(), None);
    assert!(
        md.contains("**0:03 나** 결제 화면 배포 얘기부터 하죠."),
        "{md}"
    );
    assert!(
        md.contains("**12:34 상대** 그럼 11월 3일로 확정하죠."),
        "{md}"
    );
    assert!(!md.contains("## 요약"), "no summary, no summary sections");
}

#[test]
fn an_untitled_meeting_is_named_by_its_date() {
    let s = Session {
        title: "  ".into(),
        ..session()
    };
    assert!(markdown(&s, "2026년 10월 20일", &[], None).starts_with("# 2026년 10월 20일 회의\n"));
}

#[test]
fn model_text_cannot_start_a_heading_or_a_list_of_its_own() {
    let mut s = summary();
    s.decisions = vec!["# 큰 제목\n- 가짜 항목".into()];
    s.evidence.decisions = vec![None];
    let md = markdown(&session(), "d", &[], Some(&s));
    assert!(md.contains("- \\# 큰 제목 - 가짜 항목\n"), "{md}");
}
