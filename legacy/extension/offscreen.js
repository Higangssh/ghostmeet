// Does the actual recording. Lives in an offscreen document because a Manifest V3
// service worker has no MediaRecorder and cannot hold a MediaStream.
import { audioSocketUrl } from './shared.js';

let recorder = null;
let socket = null;
let stream = null;
let audioContext = null;
// serialises sends so the final chunk always reaches the backend before "stop"
let sendChain = Promise.resolve();

async function start({ streamId, sessionId, language, backend }) {
  const tabStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId },
    },
  });

  // Capturing a tab takes its audio away from the speakers; play it back so the user
  // can still hear the meeting they are in.
  audioContext = new AudioContext();
  audioContext.createMediaStreamSource(tabStream).connect(audioContext.destination);

  await startRecording(tabStream, { sessionId, language, backend });
}

/**
 * Record a stream and ship it to the backend.
 *
 * Split out from acquiring the tab stream: getting the stream needs a real toolbar
 * click (activeTab), everything from here on does not, so this half can be exercised
 * on its own.
 */
export async function startRecording(mediaStream, { sessionId, language, backend }) {
  stream = mediaStream;
  socket = new WebSocket(audioSocketUrl(backend, sessionId, language));
  socket.binaryType = 'arraybuffer';
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener(
      'error',
      () => reject(new Error(`cannot reach the ghostmeet backend at ${backend}`)),
      { once: true },
    );
  });
  socket.addEventListener('message', onBackendMessage);

  recorder = new MediaRecorder(stream, {
    mimeType: 'audio/webm;codecs=opus',
    audioBitsPerSecond: 128000,
  });
  recorder.ondataavailable = (event) => {
    if (!event.data || event.data.size === 0) return;
    sendChain = sendChain.then(async () => {
      const buffer = await event.data.arrayBuffer();
      if (socket && socket.readyState === WebSocket.OPEN) socket.send(buffer);
    });
  };
  recorder.onstop = () => {
    sendChain = sendChain.then(() => {
      if (socket && socket.readyState === WebSocket.OPEN) socket.send('stop');
    });
  };
  recorder.start(1000);
}

function onBackendMessage(event) {
  let data;
  try {
    data = JSON.parse(event.data);
  } catch {
    return;
  }
  chrome.runtime.sendMessage({ target: 'panel', action: 'backend_message', data }).catch(() => {});

  // the backend sends this once the final transcription pass is done
  if (data.type === 'complete') {
    releaseCapture();
    chrome.runtime.sendMessage({ target: 'background', action: 'capture_finished' }).catch(() => {});
  }
}

export function stopRecording() {
  // Only stops the recorder. The socket stays open until the backend reports the final
  // pass is complete, otherwise the tail of the meeting is never transcribed.
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  recorder = null;
}

function releaseCapture() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  if (socket) {
    socket.removeEventListener('message', onBackendMessage);
    socket.close();
    socket = null;
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== 'offscreen') return;

  (async () => {
    try {
      if (message.action === 'start') {
        await start(message);
        sendResponse({ ok: true });
      } else if (message.action === 'stop') {
        stopRecording();
        sendResponse({ ok: true });
      } else {
        sendResponse({ ok: false, error: `unknown action ${message.action}` });
      }
    } catch (error) {
      releaseCapture();
      sendResponse({ ok: false, error: error.message });
    }
  })();

  return true;
});
