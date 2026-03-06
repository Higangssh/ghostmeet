const BACKEND_URL = '127.0.0.1:8877';

let ws = null;
let segmentCount = 0;
let startTime = null;
let durationTimer = null;

const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');
const emptyStateEl = document.getElementById('empty-state');
const segmentCountEl = document.getElementById('segment-count');
const durationEl = document.getElementById('duration');
const btnClear = document.getElementById('btn-clear');

// --- helpers ---

function formatTime(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function formatTimestamp(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function setStatus(state, label) {
  statusEl.textContent = label || state;
  statusEl.className = `status ${state}`;
}

function addSegment(seg) {
  emptyStateEl.classList.add('hidden');

  const div = document.createElement('div');
  div.className = 'segment new';

  const timeSpan = document.createElement('div');
  timeSpan.className = 'time';
  timeSpan.textContent = `${formatTimestamp(seg.start)} → ${formatTimestamp(seg.end)}`;

  const textSpan = document.createElement('div');
  textSpan.className = 'text';
  textSpan.textContent = seg.text;

  div.appendChild(timeSpan);
  div.appendChild(textSpan);
  transcriptEl.appendChild(div);

  // remove "new" highlight after a moment
  setTimeout(() => div.classList.remove('new'), 2000);

  // auto-scroll
  const container = document.getElementById('transcript-container');
  container.scrollTop = container.scrollHeight;

  segmentCount++;
  segmentCountEl.textContent = `${segmentCount} segment${segmentCount !== 1 ? 's' : ''}`;
}

function clearTranscript() {
  transcriptEl.innerHTML = '';
  segmentCount = 0;
  segmentCountEl.textContent = '0 segments';
  emptyStateEl.classList.remove('hidden');
}

// --- duration timer ---

function startDurationTimer() {
  startTime = Date.now();
  durationTimer = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    durationEl.textContent = formatTime(elapsed);
  }, 1000);
}

function stopDurationTimer() {
  if (durationTimer) {
    clearInterval(durationTimer);
    durationTimer = null;
  }
}

// --- WebSocket connection ---

function connectTranscript(sessionId) {
  if (ws) {
    ws.close();
  }

  setStatus('connecting', 'connecting...');

  ws = new WebSocket(`ws://${BACKEND_URL}/ws/transcript/${sessionId}`);

  ws.onopen = () => {
    setStatus('connected', `live — ${sessionId}`);
    startDurationTimer();
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'transcript' && data.segments) {
        for (const seg of data.segments) {
          addSegment(seg);
        }
      }
    } catch (e) {
      console.error('Failed to parse transcript message:', e);
    }
  };

  ws.onclose = () => {
    setStatus('disconnected', 'disconnected');
    stopDurationTimer();
  };

  ws.onerror = () => {
    setStatus('disconnected', 'connection error');
    stopDurationTimer();
  };
}

function disconnect() {
  if (ws) {
    ws.close();
    ws = null;
  }
  stopDurationTimer();
}

// --- listen for messages from background/popup ---

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === 'transcript_start' && message.sessionId) {
    connectTranscript(message.sessionId);
    sendResponse({ ok: true });
  } else if (message.action === 'transcript_stop') {
    disconnect();
    sendResponse({ ok: true });
  }
  return true;
});

// --- on load: check if there's an active session ---

chrome.storage.local.get(['activeSessionId'], (result) => {
  if (result.activeSessionId) {
    connectTranscript(result.activeSessionId);
  }
});

// --- summarize button ---

const btnSummarize = document.getElementById('btn-summarize');

btnSummarize.addEventListener('click', async () => {
  const sessionId = await getActiveSessionId();
  if (!sessionId) {
    return;
  }

  btnSummarize.disabled = true;
  btnSummarize.textContent = '⏳ Generating...';

  // show loading in transcript area
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'summary-loading';
  loadingDiv.textContent = '🤖 Generating summary with Claude...';
  transcriptEl.appendChild(loadingDiv);
  const container = document.getElementById('transcript-container');
  container.scrollTop = container.scrollHeight;

  try {
    const resp = await fetch(`http://${BACKEND_URL}/api/sessions/${sessionId}/summarize`, {
      method: 'POST',
    });
    const data = await resp.json();

    loadingDiv.remove();

    if (data.status === 'done' && data.content) {
      const summaryDiv = document.createElement('div');
      summaryDiv.className = 'summary-block';
      summaryDiv.innerHTML = markdownToHtml(data.content);
      transcriptEl.appendChild(summaryDiv);
      container.scrollTop = container.scrollHeight;
    } else {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'summary-loading';
      errorDiv.textContent = `❌ ${data.error || 'Summary generation failed'}`;
      transcriptEl.appendChild(errorDiv);
    }
  } catch (e) {
    loadingDiv.remove();
    const errorDiv = document.createElement('div');
    errorDiv.className = 'summary-loading';
    errorDiv.textContent = `❌ Failed to connect: ${e.message}`;
    transcriptEl.appendChild(errorDiv);
  } finally {
    btnSummarize.disabled = false;
    btnSummarize.textContent = '📋 Summarize';
  }
});

function getActiveSessionId() {
  return new Promise((resolve) => {
    chrome.storage.local.get(['activeSessionId'], (result) => {
      resolve(result.activeSessionId || null);
    });
  });
}

function markdownToHtml(md) {
  // minimal markdown → html for summary display
  return md
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^- (.+)$/gm, '• $1<br>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

// --- clear button ---

btnClear.addEventListener('click', clearTranscript);
