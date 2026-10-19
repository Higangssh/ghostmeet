//! SessionStore — meetings that outlive the process.
//!
//! Segments land as they are recognised, so a crash costs only the newest window. When
//! the meeting ends, the after-meeting pass replaces the live transcript with the final
//! one in a single transaction: a reader sees one or the other, never a mix.

use std::sync::Arc;

use meetbutler_engine::incremental::Segment;
use meetbutler_engine::store::{Session, SessionStatus, SessionStore, Summary, TranscriptKind};

fn open(dir: &tempfile::TempDir) -> SessionStore {
    SessionStore::open(dir.path().join("meetbutler.db")).unwrap()
}

fn make_session(id: &str) -> Session {
    Session {
        status: SessionStatus::Stopped,
        title: "Weekly sync".into(),
        stopped_at: Some(1_785_480_000.0),
        language: Some("ko".into()),
        duration_sec: 128.5,
        transcript_segments: 3,
        ..Session::new(id, 1_785_470_000.0)
    }
}

fn seg(text: &str, start: f64, end: f64) -> Segment {
    Segment::new(text, start, end)
}

fn texts(segs: &[Segment]) -> Vec<String> {
    segs.iter().map(|s| s.text.clone()).collect()
}

#[test]
fn a_fresh_store_has_no_sessions() {
    let dir = tempfile::tempdir().unwrap();
    assert!(open(&dir).load_sessions().unwrap().is_empty());
}

#[test]
fn a_new_session_is_recording_with_a_live_transcript() {
    let s = Session::new("s1", 1.0);
    assert_eq!(s.status, SessionStatus::Recording);
    assert_eq!(s.transcript, TranscriptKind::Live);
    assert_eq!(s.stopped_at, None);
}

#[test]
fn session_round_trips_every_field() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    let original = Session {
        error: Some("microphone unplugged".into()),
        sources: vec!["mic".into(), "system".into()],
        transcript: TranscriptKind::Final,
        ..make_session("20260731-090000")
    };

    store.save_session(&original).unwrap();

    assert_eq!(
        store.load_session("20260731-090000").unwrap(),
        Some(original)
    );
}

#[test]
fn saving_the_same_session_updates_it() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    let mut session = make_session("s1");
    store.save_session(&session).unwrap();

    session.transcript_segments = 99;
    store.save_session(&session).unwrap();

    let sessions = store.load_sessions().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].transcript_segments, 99);
}

#[test]
fn sessions_are_listed_oldest_first() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&Session::new("later", 200.0)).unwrap();
    store.save_session(&Session::new("earlier", 100.0)).unwrap();

    let ids: Vec<String> = store
        .load_sessions()
        .unwrap()
        .into_iter()
        .map(|s| s.session_id)
        .collect();
    assert_eq!(ids, ["earlier", "later"]);
}

#[test]
fn an_unknown_session_reads_as_none() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(open(&dir).load_session("nope").unwrap(), None);
}

#[test]
fn segments_come_back_in_the_order_they_were_added() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    store.add_segments("s1", &[seg("first", 0.0, 1.0)]).unwrap();
    store
        .add_segments("s1", &[seg("second", 1.0, 2.0), seg("third", 2.0, 3.0)])
        .unwrap();

    assert_eq!(
        texts(&store.load_segments("s1").unwrap()),
        ["first", "second", "third"]
    );
}

#[test]
fn segment_fields_survive_including_speaker() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    let original = Segment {
        speaker: Some("me".into()),
        timestamp: 1234.5,
        ..seg("hello", 1.5, 2.25)
    };
    store
        .add_segments("s1", std::slice::from_ref(&original))
        .unwrap();

    assert_eq!(store.load_segments("s1").unwrap(), [original]);
}

#[test]
fn segments_do_not_leak_between_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    store.save_session(&make_session("s2")).unwrap();
    store.add_segments("s1", &[seg("mine", 0.0, 1.0)]).unwrap();
    store.add_segments("s2", &[seg("yours", 0.0, 1.0)]).unwrap();

    assert_eq!(texts(&store.load_segments("s1").unwrap()), ["mine"]);
    assert_eq!(texts(&store.load_segments("s2").unwrap()), ["yours"]);
}

#[test]
fn unknown_session_has_no_segments() {
    let dir = tempfile::tempdir().unwrap();
    assert!(open(&dir)
        .load_segments("never-happened")
        .unwrap()
        .is_empty());
}

#[test]
fn adding_no_segments_is_harmless() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();

    store.add_segments("s1", &[]).unwrap();

    assert!(store.load_segments("s1").unwrap().is_empty());
}

// --- the final transcript ------------------------------------------------------------

#[test]
fn the_final_transcript_replaces_the_live_one() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    store
        .add_segments(
            "s1",
            &[seg("live one", 0.0, 1.0), seg("live two", 1.0, 2.0)],
        )
        .unwrap();

    store
        .replace_transcript("s1", &[seg("final", 0.0, 2.0)])
        .unwrap();

    assert_eq!(texts(&store.load_segments("s1").unwrap()), ["final"]);
    let session = store.load_session("s1").unwrap().unwrap();
    assert_eq!(session.transcript, TranscriptKind::Final);
    assert_eq!(session.transcript_segments, 1);
}

#[test]
fn a_failed_replacement_leaves_the_live_transcript_untouched() {
    // all or nothing: a half-written final transcript would be worse than either
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    store.add_segments("s1", &[seg("live", 0.0, 1.0)]).unwrap();

    let broken = [seg("good", 0.0, 1.0), seg("bad", f64::NAN, 2.0)]; // NaN binds as NULL
    assert!(store.replace_transcript("s1", &broken).is_err());

    assert_eq!(texts(&store.load_segments("s1").unwrap()), ["live"]);
    assert_eq!(
        store.load_session("s1").unwrap().unwrap().transcript,
        TranscriptKind::Live
    );
}

#[test]
fn replacing_the_transcript_of_an_unknown_session_fails() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    assert!(store
        .replace_transcript("nope", &[seg("x", 0.0, 1.0)])
        .is_err());
    assert!(store.load_segments("nope").unwrap().is_empty());
}

#[test]
fn readers_never_see_a_mix_of_live_and_final() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(open(&dir));
    store.save_session(&make_session("s1")).unwrap();
    let live: Vec<Segment> = (0..200)
        .map(|i| seg("live", i as f64, i as f64 + 1.0))
        .collect();
    store.add_segments("s1", &live).unwrap();

    let reader = {
        let store = store.clone();
        std::thread::spawn(move || {
            for _ in 0..200 {
                let segs = store.load_segments("s1").unwrap();
                let kinds: std::collections::HashSet<&str> =
                    segs.iter().map(|s| s.text.as_str()).collect();
                assert_eq!(kinds.len(), 1, "saw {kinds:?}");
            }
        })
    };
    let finals: Vec<Segment> = (0..150)
        .map(|i| seg("final", i as f64, i as f64 + 1.0))
        .collect();
    store.replace_transcript("s1", &finals).unwrap();
    reader.join().unwrap();
}

// --- summaries -----------------------------------------------------------------------

#[test]
fn summary_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    let summary = Summary {
        content: "## Decisions\n- ship it".into(),
        model: "local".into(),
        input_tokens: 120,
        output_tokens: 45,
        status: "done".into(),
        error: None,
        ..Summary::new("s1")
    };

    store.save_summary(&summary).unwrap();

    assert_eq!(store.load_summary("s1").unwrap(), Some(summary));
}

#[test]
fn regenerating_a_summary_replaces_the_old_one() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    store
        .save_summary(&Summary {
            content: "first pass".into(),
            ..Summary::new("s1")
        })
        .unwrap();
    store
        .save_summary(&Summary {
            content: "second pass".into(),
            ..Summary::new("s1")
        })
        .unwrap();

    assert_eq!(
        store.load_summary("s1").unwrap().unwrap().content,
        "second pass"
    );
}

#[test]
fn missing_summary_reads_as_none() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(open(&dir).load_summary("s1").unwrap(), None);
}

// --- library -------------------------------------------------------------------------

#[test]
fn a_session_can_be_renamed() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();

    store.rename_session("s1", "Launch review").unwrap();

    assert_eq!(
        store.load_session("s1").unwrap().unwrap().title,
        "Launch review"
    );
}

#[test]
fn deleting_a_session_removes_its_transcript_and_summary() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(&dir);
    store.save_session(&make_session("s1")).unwrap();
    store.save_session(&make_session("s2")).unwrap();
    store.add_segments("s1", &[seg("gone", 0.0, 1.0)]).unwrap();
    store.add_segments("s2", &[seg("kept", 0.0, 1.0)]).unwrap();
    store.save_summary(&Summary::new("s1")).unwrap();

    store.delete_session("s1").unwrap();

    assert_eq!(store.load_session("s1").unwrap(), None);
    assert!(store.load_segments("s1").unwrap().is_empty());
    assert_eq!(store.load_summary("s1").unwrap(), None);
    assert_eq!(texts(&store.load_segments("s2").unwrap()), ["kept"]);
}

// --- durability ----------------------------------------------------------------------

#[test]
fn everything_survives_reopening_the_database() {
    // the actual point: restarting the app must not lose a meeting
    let dir = tempfile::tempdir().unwrap();
    {
        let first = open(&dir);
        first.save_session(&make_session("s1")).unwrap();
        first
            .add_segments("s1", &[seg("we decided to ship", 0.0, 2.0)])
            .unwrap();
        first
            .save_summary(&Summary {
                content: "shipped".into(),
                ..Summary::new("s1")
            })
            .unwrap();
    }

    let reopened = open(&dir);
    assert!(reopened.load_session("s1").unwrap().is_some());
    assert_eq!(
        texts(&reopened.load_segments("s1").unwrap()),
        ["we decided to ship"]
    );
    assert_eq!(
        reopened.load_summary("s1").unwrap().unwrap().content,
        "shipped"
    );
}

#[test]
fn writes_from_several_threads_all_land() {
    // segments are written from the transcription workers, one per channel
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(open(&dir));
    store.save_session(&make_session("s1")).unwrap();

    let threads: Vec<_> = (0..20)
        .map(|i| {
            let store = store.clone();
            std::thread::spawn(move || {
                store
                    .add_segments("s1", &[seg(&format!("line {i}"), i as f64, i as f64 + 1.0)])
                    .unwrap();
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(store.load_segments("s1").unwrap().len(), 20);
}

#[test]
fn reopening_an_existing_database_does_not_wipe_it() {
    let dir = tempfile::tempdir().unwrap();
    open(&dir).save_session(&make_session("s1")).unwrap();

    let second = open(&dir);
    second.save_session(&make_session("s2")).unwrap();

    let ids: Vec<String> = second
        .load_sessions()
        .unwrap()
        .into_iter()
        .map(|s| s.session_id)
        .collect();
    assert_eq!(ids.len(), 2);
}

#[test]
fn sessions_cut_off_by_a_crash_are_marked_interrupted_on_reopen() {
    // nothing is capturing yet at startup, so anything left mid-flight died with the
    // last process; its transcript so far is kept
    let dir = tempfile::tempdir().unwrap();
    {
        let first = open(&dir);
        first.save_session(&Session::new("cut", 1.0)).unwrap();
        first
            .save_session(&Session {
                status: SessionStatus::Finalizing,
                ..Session::new("final", 2.0)
            })
            .unwrap();
        first.save_session(&make_session("done")).unwrap();
        first
            .add_segments("cut", &[seg("said before the crash", 0.0, 1.0)])
            .unwrap();
    }

    let reopened = open(&dir);
    assert_eq!(reopened.recover_interrupted().unwrap(), 2);

    let status = |id| reopened.load_session(id).unwrap().unwrap().status;
    assert_eq!(status("cut"), SessionStatus::Interrupted);
    assert_eq!(status("final"), SessionStatus::Interrupted);
    assert_eq!(status("done"), SessionStatus::Stopped);
    assert_eq!(
        texts(&reopened.load_segments("cut").unwrap()),
        ["said before the crash"]
    );
}
