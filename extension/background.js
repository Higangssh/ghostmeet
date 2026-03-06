let mediaRecorder = null;
let mediaStream = null;
let socket = null;
let sessionId = null;

function newSessionId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

async function startCapture() {
  if (mediaRecorder) {
    return { ok: false, error: 'already capturing', sessionId };
  }

  sessionId = newSessionId();
  // persist for side panel
  chrome.storage.local.set({ activeSessionId: sessionId });

  return new Promise((resolve) => {
    chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
      if (chrome.runtime.lastError || !stream) {
        resolve({ ok: false, error: chrome.runtime.lastError?.message || 'failed to capture tab audio' });
        return;
      }

      mediaStream = stream;
      socket = new WebSocket(`ws://127.0.0.1:8877/ws/audio?session=${sessionId}`);
      socket.binaryType = 'arraybuffer';

      socket.onopen = () => {
        try {
          mediaRecorder = new MediaRecorder(mediaStream, {
            mimeType: 'audio/webm;codecs=opus',
            audioBitsPerSecond: 128000,
          });
        } catch (e) {
          resolve({ ok: false, error: `MediaRecorder init failed: ${e.message}` });
          return;
        }

        mediaRecorder.ondataavailable = async (event) => {
          if (!event.data || event.data.size === 0) return;
          if (socket && socket.readyState === WebSocket.OPEN) {
            const buf = await event.data.arrayBuffer();
            socket.send(buf);
          }
        };

        mediaRecorder.start(1000);
        resolve({ ok: true, sessionId, message: 'capture started' });
      };

      socket.onerror = () => {
        resolve({ ok: false, error: 'websocket connection failed (is backend running on 127.0.0.1:8877?)' });
      };
    });
  });
}

async function stopCapture() {
  if (!mediaRecorder) {
    return { ok: false, error: 'not capturing' };
  }

  mediaRecorder.stop();
  mediaRecorder = null;

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send('stop');
    socket.close();
  }

  const stoppedSession = sessionId;
  socket = null;
  sessionId = null;

  // clear persisted session
  chrome.storage.local.remove('activeSessionId');

  return { ok: true, sessionId: stoppedSession, message: 'capture stopped' };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message.action === 'start_capture') {
      sendResponse(await startCapture());
      return;
    }

    if (message.action === 'stop_capture') {
      sendResponse(await stopCapture());
      return;
    }

    sendResponse({ ok: false, error: 'unknown action' });
  })();

  return true;
});
