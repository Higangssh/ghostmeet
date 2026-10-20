//! A meeting as Markdown (OUT-1): title, when and how long, the summary with each item's
//! time, then the transcript. Summary items are model output: each is kept to one line
//! and cannot open a heading or a list of its own.

use meetbutler_engine::incremental::Segment;
use meetbutler_engine::store::Session;
use meetbutler_summarizer::summary::Summary;

/// "0:03", "12:34", "1:02:05".
fn clock(sec: f64) -> String {
    let s = sec.max(0.0) as u64;
    let (h, m, r) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{r:02}")
    } else {
        format!("{m}:{r:02}")
    }
}

/// One line, with nothing at its start that Markdown would read as structure.
fn line(text: &str) -> String {
    let one = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match one.chars().next() {
        Some('#' | '>' | '-' | '*' | '+' | '|' | '`' | '=') => format!("\\{one}"),
        _ => one,
    }
}

fn side(channel: Option<&str>) -> &'static str {
    match channel {
        Some("them") => "상대",
        _ => "나",
    }
}

fn source(name: &str) -> &str {
    match name {
        "mic" => "마이크(나)",
        "system" => "시스템 오디오(상대)",
        other => other,
    }
}

/// `date` is the meeting's start as the reader's clock shows it; the window formats it.
pub fn markdown(
    session: &Session,
    date: &str,
    transcript: &[Segment],
    summary: Option<&Summary>,
) -> String {
    let title = match session.title.trim() {
        "" => format!("{date} 회의"),
        t => t.to_string(),
    };
    let mut out = format!("# {}\n\n", line(&title));
    out += &format!("- 날짜: {date}\n");
    out += &format!(
        "- 길이: {}분\n",
        (session.duration_sec / 60.0).round() as u64
    );
    let sources: Vec<&str> = session.sources.iter().map(|s| source(s)).collect();
    if !sources.is_empty() {
        out += &format!("- 소스: {}\n", sources.join(", "));
    }

    let at = |i: &Option<usize>| -> String {
        i.and_then(|i| transcript.get(i))
            .map_or(String::new(), |s| format!(" ({})", clock(s.start)))
    };
    if let Some(s) = summary {
        out += &format!("\n## 요약\n\n{}\n", line(&s.summary));
        let section = |out: &mut String, name: &str, items: Vec<String>, ev: &[Option<usize>]| {
            if items.is_empty() {
                return;
            }
            *out += &format!("\n## {name}\n\n");
            for (i, item) in items.iter().enumerate() {
                *out += &format!("{item}{}\n", at(ev.get(i).unwrap_or(&None)));
            }
        };
        let decisions = s
            .decisions
            .iter()
            .map(|d| format!("- {}", line(d)))
            .collect();
        section(&mut out, "결정", decisions, &s.evidence.decisions);
        let actions = s
            .action_items
            .iter()
            .map(|a| {
                let due = if a.due.trim().is_empty() {
                    String::new()
                } else {
                    format!(" — {}", line(&a.due))
                };
                format!("- [ ] {}: {}{due}", line(&a.owner), line(&a.task))
            })
            .collect();
        section(&mut out, "할 일", actions, &s.evidence.action_items);
        let open = s
            .open_questions
            .iter()
            .map(|q| format!("- {}", line(q)))
            .collect();
        section(&mut out, "미해결 질문", open, &s.evidence.open_questions);
    }

    if !transcript.is_empty() {
        out += "\n## 전사\n\n";
        for seg in transcript {
            out += &format!(
                "**{} {}** {}\n\n",
                clock(seg.start),
                side(seg.channel.as_deref()),
                line(&seg.text)
            );
        }
    }
    out
}
