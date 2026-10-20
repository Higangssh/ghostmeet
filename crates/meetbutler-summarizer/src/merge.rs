//! Joining the chunk summaries into the meeting's (the merge step).
//!
//! Each chunk is summarised on its own, so a chunk lists what it heard as settled even
//! when a later chunk changes it, and an item discussed across chunks is never seen
//! whole. The committee minutes in the benchmark showed both: a bill held over read as
//! passed, and discussion points listed as decisions (`bench/results/RESULTS.md` §8).
//! Joined here, in this order:
//!
//! 1. Every item gets its evidence: the line of its own chunk that says it (SUM-3).
//! 2. The last word wins: a later decision on the same thing with another value of the
//!    same kind ("500만 원" then "300만 원") replaces the earlier one. Look-alike items
//!    about different things ("결제 화면 배포" and "검색 필터 배포") are kept apart.
//! 3. A deferral ("다음 회의에서 다시", "보류") is an open question, and takes the
//!    earlier decisions on the same subject with it; inside an item the chair passed it
//!    is one point put off and goes alone.
//! 4. A chair's ruling outranks the chunks: in a meeting that rules item by item
//!    ("…가결되었음을 선포합니다", "…계속 심사하도록 하겠습니다"), the decisions whose
//!    evidence lies in an item held over become open questions, and each ruling is
//!    listed itself.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::summary::{ActionItem, Summary};

/// A chunk's summary and the transcript lines it was made from.
#[derive(Debug, Clone)]
pub struct Part {
    pub lines: Range<usize>,
    pub summary: Summary,
}

/// For each listed item, the transcript line (an index into the lines summarised) it
/// came from, if one says it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub decisions: Vec<Option<usize>>,
    pub action_items: Vec<Option<usize>>,
    pub open_questions: Vec<Option<usize>>,
}

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("a valid pattern"))
}

/// A number with the unit after it: "300만", "11월", "3일", "2명".
fn values_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(
        &R,
        r"(\d+)\s*(요일|시간|월|일|원|만|억|명|개|건|%|분|년|주|차)?",
    )
}

fn defer_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(
        &R,
        r"계속\s*심사|다음\s*회의|다음에\s*다시|다시\s*논의|다시\s*보|보류|미루|미룸|미뤄|결론을?\s*내지\s*못|추후|재논의|확정되지\s*않|결정되지\s*않",
    )
}

fn ruled_passed_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(가결|의결|채택)되었음을\s*선포")
}

fn ruled_held_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"계속\s*심사하(도록\s*하겠습니다|겠습니다|기로)")
}

/// Subjects this alike (letter pairs shared over all) are one item.
const SAME_SUBJECT: f64 = 0.6;

/// Particles a word ends with, longest first; "배포를" and "배포는" are one word here.
const PARTICLES: &[&str] = &[
    "으로", "에서", "까지", "부터", "에게", "을", "를", "은", "는", "이", "가", "로", "에", "의",
    "와", "과", "도",
];

/// Words that say something was settled or put off, not what about.
const NOT_SUBJECT: &[&str] = &[
    "확정",
    "결정",
    "진행",
    "채택",
    "승인",
    "유지",
    "합의",
    "동의",
    "다음",
    "회의",
    "다시",
    "논의",
    "검토",
    "하기로",
    "하기",
    "함",
    "계속",
    "심사",
    "보류",
    "했습니다",
    "합니다",
    "하기로함",
];

fn hangul_or_alnum(c: char) -> bool {
    c.is_alphanumeric()
}

fn squash(text: &str) -> String {
    text.chars().filter(|&c| hangul_or_alnum(c)).collect()
}

/// What an item is about: its words without numbers, units, particles or the words of
/// settling, run together.
fn subject(text: &str) -> String {
    let mut out = String::new();
    for word in text.split(|c: char| !hangul_or_alnum(c)) {
        if word.is_empty() || word.chars().any(|c| c.is_ascii_digit()) {
            continue;
        }
        let mut w = word;
        if let Some(p) = PARTICLES
            .iter()
            .find(|p| w.len() > p.len() && w.ends_with(*p))
        {
            w = &w[..w.len() - p.len()];
        }
        if !NOT_SUBJECT.contains(&w) {
            out.push_str(w);
        }
    }
    out
}

fn bigrams(text: &str) -> HashSet<(char, char)> {
    let c: Vec<char> = text.chars().collect();
    c.windows(2).map(|w| (w[0], w[1])).collect()
}

fn jaccard(a: &HashSet<(char, char)>, b: &HashSet<(char, char)>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Date,
    Money,
    Count,
    Other,
}

fn values(text: &str) -> Vec<(String, Kind)> {
    values_re()
        .captures_iter(text)
        .map(|c| {
            let unit = c.get(2).map_or("", |m| m.as_str());
            let kind = match unit {
                "월" | "일" | "요일" => Kind::Date,
                "원" | "만" | "억" => Kind::Money,
                "명" | "개" | "건" => Kind::Count,
                _ => Kind::Other,
            };
            (format!("{}{unit}", &c[1]), kind)
        })
        .collect()
}

/// Two decisions about one thing that differ in a value of the same kind.
fn same_item_new_value(earlier: &str, later: &str) -> bool {
    let (a, b) = (values(earlier), values(later));
    if a.is_empty() || b.is_empty() || a == b {
        return false;
    }
    let kinds = |v: &[(String, Kind)]| v.iter().map(|x| x.1).collect::<HashSet<_>>();
    kinds(&a) == kinds(&b)
        && jaccard(&bigrams(&subject(earlier)), &bigrams(&subject(later))) >= SAME_SUBJECT
}

/// The line in `range` that says `text` best: shared letter pairs, and three more for
/// each of its numbers the line has. A tie goes to the later line, where a value is
/// usually settled.
fn evidence(lines: &[String], range: &Range<usize>, text: &str) -> Option<usize> {
    let want = bigrams(&squash(text));
    let nums: Vec<String> = values(text).into_iter().map(|v| v.0).collect();
    let mut best: Option<(usize, usize)> = None;
    for i in range.clone().filter(|&i| i < lines.len()) {
        let line = squash(&lines[i]);
        let mut score = bigrams(&line).intersection(&want).count();
        score += 3 * nums.iter().filter(|n| line.contains(n.as_str())).count();
        if score > 0 && best.is_none_or(|(_, s)| score >= s) {
            best = Some((i, score));
        }
    }
    best.map(|b| b.0)
}

fn action_text(a: &ActionItem) -> String {
    format!("{} {} {}", a.owner, a.task, a.due)
}

/// The ruling's own words: from "의사일정" (if the line says it) to the end of the
/// sentence that rules.
fn ruling_text(line: &str, at: usize) -> String {
    let end = line[at..]
        .find(['.', '?', '!'])
        .map_or(line.len(), |i| at + i + 1);
    let start = line[..at]
        .rfind("의사일정")
        .unwrap_or_else(|| line[..at].rfind(['.', '?', '!']).map_or(0, |i| i + 1));
    line[start..end].trim().to_string()
}

/// Join the chunk summaries (in transcript order) into one. The overall `summary` text
/// is left empty for the caller.
pub fn merge(lines: &[String], parts: Vec<Part>) -> Summary {
    let mut decisions: Vec<(String, Option<usize>)> = Vec::new();
    let mut actions: Vec<(ActionItem, Option<usize>)> = Vec::new();
    let mut open: Vec<(String, Option<usize>)> = Vec::new();
    let mut out = Summary::default();
    for part in parts {
        let s = part.summary;
        for d in s.decisions {
            if !decisions.iter().any(|x| x.0 == d) {
                let at = evidence(lines, &part.lines, &d);
                decisions.push((d, at));
            }
        }
        for a in s.action_items {
            if !actions.iter().any(|x| x.0 == a) {
                let at = evidence(lines, &part.lines, &action_text(&a));
                actions.push((a, at));
            }
        }
        for q in s.open_questions {
            if !open.iter().any(|x| x.0 == q) {
                let at = evidence(lines, &part.lines, &q);
                open.push((q, at));
            }
        }
    }

    // 2. the last word wins
    let mut i = 0;
    while i < decisions.len() {
        let replaced =
            (i + 1..decisions.len()).any(|j| same_item_new_value(&decisions[i].0, &decisions[j].0));
        if replaced {
            decisions.remove(i);
        } else {
            i += 1;
        }
    }

    // the chair's rulings, each closing the item that runs from the line after the last
    let mut rulings: Vec<(String, usize, bool)> = Vec::new();
    let mut items: Vec<(Range<usize>, bool)> = Vec::new();
    let mut start = 0;
    for (n, line) in lines.iter().enumerate() {
        let held = ruled_held_re().find(line);
        let Some(m) = held.or_else(|| ruled_passed_re().find(line)) else {
            continue;
        };
        rulings.push((ruling_text(line, m.start()), n, held.is_some()));
        items.push((start..n + 1, held.is_some()));
        start = n + 1;
    }
    let in_passed = |at: Option<usize>| {
        at.is_some_and(|at| items.iter().any(|(r, held)| !held && r.contains(&at)))
    };

    // 3. a deferral is an open question, and takes its subject's earlier decisions along,
    // except inside an item the chair passed: there it is one point put off
    let mut moved: Vec<usize> = Vec::new();
    for j in 0..decisions.len() {
        if !defer_re().is_match(&decisions[j].0) {
            continue;
        }
        if !in_passed(decisions[j].1) {
            let about = bigrams(&subject(&decisions[j].0));
            for (i, d) in decisions[..j].iter().enumerate() {
                if !moved.contains(&i) && jaccard(&bigrams(&subject(&d.0)), &about) >= SAME_SUBJECT
                {
                    moved.push(i);
                }
            }
        }
        moved.push(j);
    }

    // 4. an item held over: what the chunks decided in it is open
    for (range, held) in &items {
        if !held {
            continue;
        }
        for (i, d) in decisions.iter().enumerate() {
            if d.1.is_some_and(|at| range.contains(&at)) && !moved.contains(&i) {
                moved.push(i);
            }
        }
    }

    moved.sort_unstable();
    for &i in &moved {
        let d = decisions[i].clone();
        if !open.iter().any(|x| x.0 == d.0) {
            open.push(d);
        }
    }
    let mut kept: Vec<(String, Option<usize>)> = decisions
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !moved.contains(i))
        .map(|(_, d)| d)
        .collect();
    for (text, at, held) in rulings {
        let list = if held { &mut open } else { &mut kept };
        if !list.iter().any(|x| x.0 == text) {
            list.push((text, Some(at)));
        }
    }

    (out.decisions, out.evidence.decisions) = kept.into_iter().unzip();
    (out.action_items, out.evidence.action_items) = actions.into_iter().unzip();
    (out.open_questions, out.evidence.open_questions) = open.into_iter().unzip();
    out
}
