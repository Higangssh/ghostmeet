// The panel (about 360×560): captions / summary / record tabs for the meeting on screen —
// the one recording, else the last one — and, below, its state and Start/Stop.
let recording = null;
let shown = null;      // session id on screen
let shownSession = null;
let startedAt = 0;
let timer = null;

function status(text) {
  $("status").textContent = text || "";
}

// --- tabs ---

function selectTab(name) {
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== `tab-${name}`));
}

document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));

// --- the next meeting's setup ---

async function loadSetup() {
  const s = await nextSetup();
  document.querySelectorAll("input[name=source]").forEach((el) => (el.checked = s.sources.includes(el.value)));
  $("language").value = s.language;
  $("vocabulary").value = s.vocabulary.join(", ");
}

function saveSetup() {
  writeSetup({
    sources: [...document.querySelectorAll("input[name=source]:checked")].map((el) => el.value),
    language: $("language").value,
    vocabulary: parseVocabulary($("vocabulary").value),
  });
}

document.querySelectorAll("#setup input, #setup select, #setup textarea").forEach((el) => el.addEventListener("change", saveSetup));

// --- the meeting on screen ---

function tick() {
  $("elapsed").textContent = clock((Date.now() - startedAt) / 1000);
}

function setRecording(id, since) {
  recording = id;
  $("indicator").hidden = !id;
  $("heading").hidden = !!id;
  $("toggle").textContent = id ? "녹음 정지" : "녹음 시작";
  $("toggle").classList.toggle("recording", !!id);
  $("setup").hidden = !!id;
  clearInterval(timer);
  if (id) {
    startedAt = since || Date.now();
    tick();
    timer = setInterval(tick, 1000);
  }
}

async function showSummary() {
  const s = await loadSummary(shown);
  $("summary-run").hidden = true;
  $("summary-state").textContent = s.state || "아직 요약이 없습니다";
  $("summary-again").hidden = !s.status || s.status === "done";
  // nothing kept and nothing running: the mascot says where the summary will be
  $("summary-empty").hidden = !!s.status || !$("summary-run").hidden;
  if (s.content) {
    renderSummary($("summary-body"), s.content, (line) => {
      selectTab("captions");
      jumpTo($("captions"), line);
    });
  } else {
    $("summary-body").replaceChildren();
  }
}

async function show(session) {
  shown = session ? session.session_id : null;
  shownSession = session;
  $("meeting").hidden = !session;
  if (!session) {
    renderTranscript($("captions"), [], false);
    $("kind").textContent = "";
    $("summary-state").textContent = "아직 요약이 없습니다";
    $("summary-body").replaceChildren();
    $("summary-empty").hidden = false;
    return;
  }
  const final = session.transcript === "final";
  renderTranscript($("captions"), (await invoke("get_transcript", { sessionId: shown })) || [], !final);
  $("captions").scrollTop = $("captions").scrollHeight;
  $("kind").textContent = final ? "최종본" : "임시 자막";
  $("title").value = session.title;
  $("facts").textContent = `${meetingDate(session)} · ${Math.round(session.duration_sec / 60)}분`;
  await showSummary();
}

async function sessionById(id) {
  return (await invoke("list_sessions")).find((s) => s.session_id === id) || null;
}

async function showLatest() {
  const sessions = await invoke("list_sessions");
  await show(sessions.length ? sessions[sessions.length - 1] : null);
}

$("title").addEventListener("change", async () => {
  if (!shown) return;
  try {
    await invoke("rename_session", { sessionId: shown, title: $("title").value });
  } catch (err) {
    status(String(err));
  }
});

$("copy").addEventListener("click", async () => {
  try {
    const md = await invoke("get_markdown", { sessionId: shown, date: meetingDate(shownSession) });
    await navigator.clipboard.writeText(md);
    status("Markdown을 복사했습니다.");
  } catch (err) {
    status(`복사하지 못했습니다: ${err}`);
  }
});

$("export").addEventListener("click", async () => {
  try {
    const path = await invoke("export_markdown", { sessionId: shown, date: meetingDate(shownSession) });
    status(`저장했습니다: ${path}`);
  } catch (err) {
    status(`저장하지 못했습니다: ${err}`);
  }
});

// --- summary controls ---

$("summary-cancel").addEventListener("click", async () => {
  $("summary-cancel").disabled = true;
  await invoke("cancel_summary");
});

$("summary-again").addEventListener("click", async () => {
  if (!shown) return;
  $("summary-state").textContent = "요약하는 중…";
  $("summary-empty").hidden = true;
  $("summary-again").hidden = true;
  $("summary-body").replaceChildren();
  await invoke("summarize", { sessionId: shown });
});

// --- start / stop ---

$("toggle").addEventListener("click", async () => {
  $("toggle").disabled = true;
  try {
    if (!recording) {
      saveSetup();
      const started = await startMeeting();
      status(started.warnings.length ? `일부 소스 없이 녹음합니다: ${started.warnings.join(" / ")}` : "");
    } else {
      await stopMeeting();
      status("회의가 끝났습니다. 최종본을 만드는 중…");
    }
  } catch (err) {
    status(String(err));
  } finally {
    $("toggle").disabled = false;
  }
});

// --- events from Rust, sent to every window ---

listen("recording", async ({ payload }) => {
  if (payload) {
    setRecording(payload);
    selectTab("captions");
    await show(await sessionById(payload));
  } else {
    const id = recording;
    setRecording(null);
    await show(await sessionById(id));
  }
});

listen("segments", ({ payload }) => {
  if (payload.session_id !== shown) return;
  const list = $("captions");
  payload.segments.forEach((s) => list.append(captionItem(s, true)));
  list.scrollTop = list.scrollHeight;
});

listen("final", async ({ payload }) => {
  if (payload.session_id !== shown) return;
  if (!payload.done) {
    status(`최종본을 만드는 중… ${Math.round(payload.progress * 100)}%`);
    return;
  }
  status(payload.error ? `최종본을 만들지 못했습니다. 임시 자막은 그대로 있습니다. (${payload.error})` : "최종본으로 바꿨습니다.");
  if (!payload.error) await show(await sessionById(shown));
});

listen("summary", async ({ payload }) => {
  if (payload.session_id !== shown) return;
  if (!payload.done) {
    $("summary-empty").hidden = true;
    $("summary-run").hidden = false;
    $("summary-cancel").disabled = false;
    $("summary-progress").value = payload.progress;
    $("summary-state").textContent = `요약하는 중… ${Math.round(payload.progress * 100)}%`;
    status($("summary-state").textContent);
    return;
  }
  status(payload.cancelled ? "요약을 취소했습니다." : payload.error ? "요약하지 못했습니다." : "요약이 끝났습니다.");
  await showSummary();
});

listen("sessions", async () => {
  if (!shown) return;
  const s = await sessionById(shown);
  if (s) {
    shownSession = s;
    if (document.activeElement !== $("title")) $("title").value = s.title;
  } else if (!recording) {
    await showLatest(); // the meeting on screen was deleted in the library
  }
});

(async () => {
  await loadSetup();
  const id = await invoke("recording");
  if (id) {
    const s = await sessionById(id);
    setRecording(id, s ? s.started_at * 1000 : undefined);
    await show(s);
  } else {
    setRecording(null);
    await showLatest();
  }
})();
