// The mini bar (the main window, always on top): recording state, elapsed time, sources,
// the newest caption line, Start/Stop, and the way to the panel and the library.
let recording = null;
let startedAt = 0;
let timer = null;

function tick() {
  $("elapsed").textContent = clock((Date.now() - startedAt) / 1000);
}

function showSources(names) {
  $("sources").replaceChildren(...names.map((n) => el("span", "chip", n === "system" ? "상대" : "나")));
}

function setRecording(id, sources, since) {
  recording = id;
  $("indicator").hidden = !id;
  $("idle-title").hidden = !!id;
  $("toggle").textContent = id ? "정지" : "녹음 시작";
  $("toggle").classList.toggle("recording", !!id);
  clearInterval(timer);
  if (id) {
    startedAt = since || Date.now();
    tick();
    timer = setInterval(tick, 1000);
    if (sources) showSources(sources);
  } else {
    $("sources").replaceChildren();
  }
}

async function toggle() {
  $("toggle").disabled = true;
  try {
    if (!recording) {
      const started = await startMeeting();
      setRecording(started.session_id, started.sources);
      $("last").textContent = started.warnings.length ? `일부 소스 없이 녹음합니다: ${started.warnings.join(" / ")}` : "";
    } else {
      await stopMeeting();
      setRecording(null);
      $("last").textContent = "최종본을 만드는 중…";
    }
  } catch (err) {
    $("last").textContent = String(err);
  } finally {
    $("toggle").disabled = false;
  }
}

$("toggle").addEventListener("click", toggle);
// the menu bar's 녹음 시작/정지 lands here, so it starts with the same setup
listen("tray-toggle", () => {
  if (!$("toggle").disabled) toggle();
});

$("expand").addEventListener("click", () => invoke("show_window", { label: "panel" }));
$("library").addEventListener("click", () => invoke("show_window", { label: "library" }));

listen("recording", ({ payload }) => {
  if (payload && payload !== recording) setRecording(payload);
  if (!payload && recording) setRecording(null);
});

listen("segments", ({ payload }) => {
  if (payload.session_id !== recording || !payload.segments.length) return;
  const seg = payload.segments[payload.segments.length - 1];
  $("last").textContent = `${side(seg)} · ${seg.text}`;
});

listen("final", ({ payload }) => {
  if (recording) return;
  $("last").textContent = !payload.done
    ? `최종본을 만드는 중… ${Math.round(payload.progress * 100)}%`
    : payload.error ? "최종본을 만들지 못했습니다. 임시 자막은 그대로 있습니다." : "최종본으로 바꿨습니다.";
});

listen("summary", ({ payload }) => {
  if (recording) return;
  $("last").textContent = !payload.done
    ? `요약하는 중… ${Math.round(payload.progress * 100)}%`
    : payload.cancelled ? "요약을 취소했습니다." : payload.error ? "요약하지 못했습니다." : "요약이 끝났습니다. 패널에서 보세요.";
});

(async () => {
  const id = await invoke("recording");
  if (id) {
    const s = await invoke("list_sessions");
    const cur = s.find((x) => x.session_id === id);
    setRecording(id, cur ? cur.sources : [], cur ? cur.started_at * 1000 : undefined);
  }
})();
