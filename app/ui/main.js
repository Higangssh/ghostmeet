// MeetButler window. Talks to Rust over Tauri IPC only; transcript and summary text are
// model output derived from meeting audio, so they only ever go into the page through
// textContent.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
let recording = null; // session id while recording
let shown = null;     // session whose transcript is on screen
let startedAt = 0;
let timer = null;

function notice(text) {
  $("notice").hidden = !text;
  $("notice").textContent = text || "";
}

function caption(seg, provisional) {
  const li = document.createElement("li");
  if (provisional) li.className = "provisional";
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = seg.channel === "them" ? "상대" : "나";
  li.append(who, document.createTextNode(seg.text));
  return li;
}

function showTranscript(segments, kind) {
  const list = $("captions");
  list.replaceChildren(...segments.map((s) => caption(s, kind !== "final")));
  list.scrollTop = list.scrollHeight;
  $("kind").textContent = kind === "final" ? "최종본" : kind === "live" ? "임시 자막" : "";
}

function clock() {
  const s = Math.floor((Date.now() - startedAt) / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = String(s % 60).padStart(2, "0");
  $("elapsed").textContent = h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
}

function setRecording(id) {
  recording = id;
  $("indicator").hidden = !id;
  $("idle-title").hidden = !!id;
  $("toggle").textContent = id ? "녹음 정지" : "녹음 시작";
  $("toggle").classList.toggle("recording", !!id);
  document.querySelectorAll("#controls input, #controls select").forEach((el) => (el.disabled = !!id));
  clearInterval(timer);
  if (id) {
    startedAt = Date.now();
    clock();
    timer = setInterval(clock, 1000);
  }
}

async function refreshSessions() {
  const sessions = await invoke("list_sessions");
  const list = $("sessions");
  list.replaceChildren(...sessions.reverse().map((s) => {
    const li = document.createElement("li");
    const title = document.createElement("span");
    title.textContent = s.title || new Date(s.started_at * 1000).toLocaleString();
    const meta = document.createElement("span");
    meta.className = "meta";
    const min = Math.round(s.duration_sec / 60);
    meta.textContent = `${min}분 · ${s.transcript === "final" ? "최종본" : s.status === "finalizing" ? "정리 중" : "임시"}`;
    li.append(title, meta);
    li.addEventListener("click", () => open(s.session_id, s.transcript));
    return li;
  }));
}

async function open(id, kind) {
  shown = id;
  showTranscript((await invoke("get_transcript", { sessionId: id })) || [], kind);
  await showSummary(id);
}

// --- summary: model output, so text nodes only ---

// SUM-3: an item's evidence is the index of the transcript line it came from; the
// summary was made from the transcript in the order it is listed on screen
function jumpTo(line) {
  const li = $("captions").children[line];
  if (!li) return;
  li.scrollIntoView({ block: "center", behavior: "smooth" });
  li.classList.add("evidence");
  setTimeout(() => li.classList.remove("evidence"), 2000);
}

function list(title, items, evidence = []) {
  if (!items.length) return [];
  const h = document.createElement("h3");
  h.textContent = title;
  const ul = document.createElement("ul");
  ul.replaceChildren(...items.map((text, i) => {
    const li = document.createElement("li");
    li.textContent = text;
    const at = evidence[i];
    if (Number.isInteger(at)) {
      const go = document.createElement("button");
      go.type = "button";
      go.className = "go";
      go.title = "이 내용이 나온 곳";
      go.textContent = "↗";
      go.addEventListener("click", () => jumpTo(at));
      li.append(" ", go);
    }
    return li;
  }));
  return [h, ul];
}

function renderSummary(s) {
  const p = document.createElement("p");
  p.textContent = s.summary;
  const actions = s.action_items.map((a) => [a.owner, a.task, a.due && `(${a.due})`].filter(Boolean).join(" · "));
  const ev = s.evidence || {};
  $("summary-body").replaceChildren(
    p,
    ...list("결정", s.decisions, ev.decisions),
    ...list("할 일", actions, ev.action_items),
    ...list("미해결 질문", s.open_questions, ev.open_questions),
  );
}

function summaryRunning(on, progress = 0) {
  $("summary-run").hidden = !on;
  $("summary-progress").value = progress;
  $("summary-cancel").disabled = false;
  if (on) $("summary-again").hidden = true;
}

async function showSummary(id) {
  const kept = await invoke("get_summary", { sessionId: id });
  $("summary").hidden = !kept;
  summaryRunning(false);
  $("summary-body").replaceChildren();
  if (!kept) return;
  $("summary-again").hidden = kept.status === "done";
  if (kept.status === "done") {
    $("summary-state").textContent = "로컬 모델";
    try {
      renderSummary(JSON.parse(kept.content));
    } catch {
      $("summary-state").textContent = "요약을 읽지 못했습니다";
    }
  } else {
    $("summary-state").textContent = kept.status === "cancelled" ? "취소됨" : `실패: ${kept.error || ""}`;
  }
}

$("summary-cancel").addEventListener("click", async () => {
  $("summary-cancel").disabled = true;
  await invoke("cancel_summary");
});

$("summary-again").addEventListener("click", async () => {
  if (!shown) return;
  $("summary").hidden = false;
  $("summary-state").textContent = "요약하는 중…";
  $("summary-body").replaceChildren();
  summaryRunning(true);
  await invoke("summarize", { sessionId: shown });
});

$("toggle").addEventListener("click", async () => {
  $("toggle").disabled = true;
  try {
    if (!recording) {
      const sources = [...document.querySelectorAll("input[name=source]:checked")].map((el) => el.value);
      const started = await invoke("start_recording", { sources, language: $("language").value });
      shown = started.session_id;
      showTranscript([], "live");
      notice(started.warnings.length ? `일부 소스 없이 녹음합니다: ${started.warnings.join(" / ")}` : "");
      setRecording(started.session_id);
    } else {
      const id = recording;
      await invoke("stop_recording");
      setRecording(null);
      notice("회의가 끝났습니다. 최종본을 만드는 중…");
      await open(id, "live");
      await refreshSessions();
    }
  } catch (err) {
    notice(String(err));
  } finally {
    $("toggle").disabled = false;
  }
});

listen("segments", ({ payload }) => {
  if (payload.session_id !== shown) return;
  const list = $("captions");
  payload.segments.forEach((s) => list.append(caption(s, true)));
  list.scrollTop = list.scrollHeight;
});

listen("final", async ({ payload }) => {
  if (!payload.done) {
    notice(`최종본을 만드는 중… ${Math.round(payload.progress * 100)}%`);
    return;
  }
  notice(payload.error ? `최종본을 만들지 못했습니다. 임시 자막은 그대로 있습니다. (${payload.error})` : "최종본으로 바꿨습니다.");
  if (!payload.error && payload.session_id === shown) await open(shown, "final");
  await refreshSessions();
});

listen("summary", async ({ payload }) => {
  if (payload.session_id !== shown) return;
  if (!payload.done) {
    $("summary").hidden = false;
    $("summary-state").textContent = `요약하는 중… ${Math.round(payload.progress * 100)}%`;
    summaryRunning(true, payload.progress);
    return;
  }
  await showSummary(shown);
});

(async () => {
  const id = await invoke("recording");
  if (id) {
    shown = id;
    setRecording(id);
    await open(id, "live");
  }
  await refreshSessions();
})();
