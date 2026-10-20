// Shared by the mini bar, the panel and the library. Talks to Rust over Tauri IPC only.
// Transcript and summary text are model output derived from meeting audio: they only
// ever go into a page as text nodes (textContent), never as HTML.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// "0:03", "12:34", "1:02:05"
function clock(sec) {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
}

function meetingDate(session) {
  return new Date(session.started_at * 1000).toLocaleString();
}

function meetingTitle(session) {
  return session.title.trim() || meetingDate(session);
}

function side(seg) {
  return seg.channel === "them" ? "상대" : "나";
}

// one transcript line: time, side, text
function captionItem(seg, provisional) {
  const li = el("li", provisional ? "provisional" : "");
  li.append(el("span", "time", clock(seg.start)), el("span", "who", side(seg)), document.createTextNode(seg.text));
  return li;
}

function renderTranscript(list, segments, provisional) {
  list.replaceChildren(...segments.map((s) => captionItem(s, provisional)));
}

// SUM-3: an item's evidence is the index of the transcript line it came from, in the
// order the transcript is listed
function jumpTo(list, line) {
  const li = list.children[line];
  if (!li) return;
  li.scrollIntoView({ block: "center", behavior: "smooth" });
  li.classList.add("evidence");
  setTimeout(() => li.classList.remove("evidence"), 2000);
}

function summaryList(title, items, evidence, onJump) {
  if (!items.length) return [];
  const ul = el("ul");
  ul.replaceChildren(...items.map((text, i) => {
    const li = el("li", "", text);
    const at = (evidence || [])[i];
    if (Number.isInteger(at)) {
      const go = el("button", "go", "↗");
      go.type = "button";
      go.title = "이 내용이 나온 곳";
      go.addEventListener("click", () => onJump(at));
      li.append(" ", go);
    }
    return li;
  }));
  return [el("h3", "", title), ul];
}

function renderSummary(body, s, onJump) {
  const ev = s.evidence || {};
  const actions = s.action_items.map((a) => [a.owner, a.task, a.due && `(${a.due})`].filter(Boolean).join(" · "));
  body.replaceChildren(
    el("p", "", s.summary),
    ...summaryList("결정", s.decisions, ev.decisions, onJump),
    ...summaryList("할 일", actions, ev.action_items, onJump),
    ...summaryList("미해결 질문", s.open_questions, ev.open_questions, onJump),
  );
}

// the kept summary as a state line and, when done, the content
async function loadSummary(sessionId) {
  const kept = await invoke("get_summary", { sessionId });
  if (!kept) return { state: "", content: null, status: null };
  if (kept.status === "done") {
    try {
      return { state: "로컬 모델", content: JSON.parse(kept.content), status: "done" };
    } catch {
      return { state: "요약을 읽지 못했습니다", content: null, status: "error" };
    }
  }
  const state = kept.status === "cancelled" ? "취소됨" : `실패: ${kept.error || ""}`;
  return { state, content: null, status: kept.status };
}

// --- the next meeting's setup, shared by every window (same origin, same storage) ---

function readSetup() {
  try {
    return JSON.parse(localStorage.getItem("setup")) || {};
  } catch {
    return {};
  }
}

function writeSetup(setup) {
  try {
    localStorage.setItem("setup", JSON.stringify(setup));
  } catch {
    // storage may be unavailable; the defaults still record
  }
}

function parseVocabulary(text) {
  return text.split(/[,\n]/).map((t) => t.trim()).filter(Boolean);
}

async function nextSetup() {
  const s = readSetup();
  return {
    sources: s.sources && s.sources.length ? s.sources : ["mic", "system"],
    language: s.language || "ko",
    vocabulary: s.vocabulary !== undefined ? s.vocabulary : await invoke("last_vocabulary"),
  };
}

// Start with the setup the panel holds; any window may start or stop
async function startMeeting() {
  const setup = await nextSetup();
  return invoke("start_recording", setup);
}

async function stopMeeting() {
  return invoke("stop_recording");
}
