// The library window (MEM-1, OUT-1): past meetings, a title filter, and for the one
// picked its summary and transcript, rename, Markdown copy/save, and delete.
let sessions = [];
let picked = null;
let armed = null; // delete asks twice; the second click must come within a few seconds

function status(text) {
  $("status").textContent = text || "";
}

function label(s) {
  if (s.status === "recording") return "녹음 중";
  if (s.status === "finalizing") return "정리 중";
  if (s.status === "interrupted") return "중단됨";
  return s.transcript === "final" ? "최종본" : "임시";
}

function renderList() {
  const q = $("filter").value.trim().toLowerCase();
  const rows = [...sessions].reverse().filter((s) => !q || meetingTitle(s).toLowerCase().includes(q));
  $("sessions").replaceChildren(...rows.map((s) => {
    const li = el("li", s.session_id === (picked && picked.session_id) ? "picked" : "");
    li.tabIndex = 0;
    li.append(el("span", "name", meetingTitle(s)), el("span", "meta", `${Math.round(s.duration_sec / 60)}분 · ${label(s)}`));
    li.addEventListener("click", () => pick(s.session_id));
    li.addEventListener("keydown", (e) => {
      if (e.key === "Enter") pick(s.session_id);
    });
    return li;
  }));
}

async function refresh() {
  sessions = await invoke("list_sessions");
  if (picked) picked = sessions.find((s) => s.session_id === picked.session_id) || null;
  renderList();
  if (!picked) {
    $("meeting").hidden = true;
    $("empty").hidden = false;
  }
}

async function showSummary() {
  const s = await loadSummary(picked.session_id);
  $("summary-state").textContent = s.state;
  $("summary-again").hidden = !s.status || s.status === "done";
  if (s.content) renderSummary($("summary-body"), s.content, (line) => jumpTo($("transcript"), line));
  else $("summary-body").replaceChildren();
}

async function pick(id) {
  picked = sessions.find((s) => s.session_id === id) || null;
  disarm();
  status("");
  renderList();
  $("empty").hidden = !!picked;
  $("meeting").hidden = !picked;
  if (!picked) return;
  $("title").value = picked.title;
  $("title").placeholder = meetingDate(picked);
  const sources = picked.sources.map((n) => (n === "system" ? "상대" : "나")).join("·");
  $("facts").textContent = `${meetingDate(picked)} · ${Math.round(picked.duration_sec / 60)}분 · ${sources}`;
  const final = picked.transcript === "final";
  renderTranscript($("transcript"), (await invoke("get_transcript", { sessionId: id })) || [], !final);
  $("kind").textContent = final ? "최종본" : "임시 자막";
  await showSummary();
}

$("filter").addEventListener("input", renderList);

$("title").addEventListener("change", async () => {
  if (!picked) return;
  try {
    await invoke("rename_session", { sessionId: picked.session_id, title: $("title").value });
  } catch (err) {
    status(String(err));
  }
});

$("copy").addEventListener("click", async () => {
  try {
    const md = await invoke("get_markdown", { sessionId: picked.session_id, date: meetingDate(picked) });
    await navigator.clipboard.writeText(md);
    status("Markdown을 복사했습니다.");
  } catch (err) {
    status(`복사하지 못했습니다: ${err}`);
  }
});

$("export").addEventListener("click", async () => {
  try {
    status(`저장했습니다: ${await invoke("export_markdown", { sessionId: picked.session_id, date: meetingDate(picked) })}`);
  } catch (err) {
    status(`저장하지 못했습니다: ${err}`);
  }
});

function disarm() {
  clearTimeout(armed);
  armed = null;
  $("delete").textContent = "삭제";
  $("delete").classList.remove("armed");
}

$("delete").addEventListener("click", async () => {
  if (!picked) return;
  if (!armed) {
    $("delete").textContent = "녹음까지 지웁니다. 한 번 더 누르면 삭제";
    $("delete").classList.add("armed");
    armed = setTimeout(disarm, 4000);
    return;
  }
  disarm();
  try {
    await invoke("delete_session", { sessionId: picked.session_id });
    picked = null;
    await refresh();
  } catch (err) {
    status(`삭제하지 못했습니다: ${err}`);
  }
});

$("summary-again").addEventListener("click", async () => {
  if (!picked) return;
  $("summary-state").textContent = "요약하는 중…";
  $("summary-again").hidden = true;
  await invoke("summarize", { sessionId: picked.session_id });
});

// --- events from Rust ---

listen("sessions", refresh);
listen("recording", refresh);

listen("final", async ({ payload }) => {
  if (!payload.done) return;
  await refresh();
  if (picked && picked.session_id === payload.session_id) await pick(picked.session_id);
});

listen("summary", async ({ payload }) => {
  if (!picked || picked.session_id !== payload.session_id) return;
  if (!payload.done) {
    $("summary-state").textContent = `요약하는 중… ${Math.round(payload.progress * 100)}%`;
    return;
  }
  await showSummary();
});

refresh();
