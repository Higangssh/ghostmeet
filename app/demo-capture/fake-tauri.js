// A stand-in for window.__TAURI__, so the app's real pages (app/ui) run in a browser on
// the made-up meeting in scenario.js. Loaded before any page script by capture.mjs. The
// stage page owns the state and the event bus; the three windows are its iframes and
// reach it through window.top. Only the commands the pages call are here.
(() => {
  const top = window.top;
  if (window === top) {
    const S = window.SCENARIO;
    const at = (iso) => Date.parse(iso) / 1000;
    const session = (id, title, startedAt, min, extra) => ({
      session_id: id, title, started_at: startedAt, stopped_at: startedAt + min * 60,
      status: "stopped", sources: ["mic", "system"], language: "ko", duration_sec: min * 60,
      transcript: "final", transcript_segments: 2, error: null, vocabulary: S.vocabulary, ...extra,
    });
    const seg = (start, channel, text) => ({ text, start, end: start + 5, channel, speaker: null, timestamp: 0 });
    const demo = {
      sessions: S.past.map((p) => session(p.id, p.title, at(p.at), p.min)),
      transcripts: {},
      summaries: {},
      recording: null,
      listeners: {},
    };
    for (const p of S.past) {
      demo.transcripts[p.id] = S.pastTranscript.map((l) => seg(l.start, l.channel, l.text));
      demo.summaries[p.id] = { session_id: p.id, status: "done", model: "qwen3.5-4b", error: null,
        content: JSON.stringify(S.pastSummary) };
    }
    const find = (id) => demo.sessions.find((s) => s.session_id === id);
    const copy = (v) => (v === undefined ? null : JSON.parse(JSON.stringify(v)));

    demo.emit = (event, payload) => {
      for (const cb of demo.listeners[event] || []) cb({ event, payload: copy(payload) });
    };

    const commands = {
      list_sessions: () => demo.sessions,
      get_transcript: ({ sessionId }) => demo.transcripts[sessionId] || null,
      get_summary: ({ sessionId }) => demo.summaries[sessionId] || null,
      recording: () => demo.recording,
      last_vocabulary: () => S.vocabulary,
      show_window: () => null,
      cancel_summary: () => null,
      summarize: () => null,
      get_markdown: () => "",
      rename_session: ({ sessionId, title }) => {
        find(sessionId).title = title;
        demo.emit("sessions", null);
      },
      start_recording: ({ sources }) => {
        const id = `m${Date.now()}`;
        demo.sessions.push(session(id, "", Date.now() / 1000, 0, {
          status: "recording", stopped_at: null, transcript: "live", sources,
        }));
        demo.transcripts[id] = [];
        demo.recording = id;
        demo.emit("recording", id);
        return { session_id: id, sources, warnings: [] };
      },
      stop_recording: () => {
        const s = find(demo.recording);
        demo.recording = null;
        Object.assign(s, { status: "stopped", stopped_at: Date.now() / 1000, duration_sec: S.duration });
        demo.emit("recording", null);
        return s;
      },
    };

    // what Rust does on its own: captions as they are recognised, the final pass, the summary
    demo.caption = (i) => {
      const [start, channel, live] = S.lines[i];
      const s = seg(start, channel, live);
      demo.transcripts[demo.recording].push(s);
      demo.emit("segments", { session_id: demo.recording, segments: [s] });
    };
    demo.latest = () => demo.sessions[demo.sessions.length - 1].session_id;
    demo.finalProgress = (progress) => {
      const id = demo.latest();
      const done = progress >= 1;
      if (done) {
        demo.transcripts[id] = S.lines.map(([start, channel, , text]) => seg(start, channel, text));
        Object.assign(find(id), { transcript: "final", transcript_segments: S.lines.length });
      }
      demo.emit("final", { session_id: id, progress, done, error: null });
    };
    demo.summaryProgress = (progress) => {
      const id = demo.latest();
      const done = progress >= 1;
      if (done) {
        demo.summaries[id] = { session_id: id, status: "done", model: "qwen3.5-4b", error: null,
          content: JSON.stringify(S.summary) };
      }
      demo.emit("summary", { session_id: id, progress: done ? 1 : progress, done, cancelled: false, error: null });
    };
    demo.invoke = async (cmd, args) => {
      if (!commands[cmd]) throw new Error(`demo: no command ${cmd}`);
      return copy(commands[cmd](args || {}));
    };
    window.__demo = demo;
  }

  const demo = () => window.top.__demo;
  window.__TAURI__ = {
    core: { invoke: (cmd, args) => demo().invoke(cmd, args) },
    event: {
      listen: async (event, cb) => {
        (demo().listeners[event] ||= []).push(cb);
        return () => {};
      },
    },
  };
})();
