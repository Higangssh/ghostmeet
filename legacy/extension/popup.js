const statusEl = document.getElementById('status');

function show(text) {
  statusEl.textContent = text;
}

async function send(action) {
  const response = await chrome.runtime.sendMessage({ action });
  if (!response) {
    show('no response from the extension background');
    return;
  }

  if (!response.ok) {
    show(`⚠ ${response.error}`);
    return;
  }

  if (action === 'start_capture') {
    show(`● recording — ${response.sessionId}`);
    chrome.runtime.sendMessage({
      target: 'panel',
      action: 'transcript_start',
      sessionId: response.sessionId,
    }).catch(() => {});
    if (chrome.sidePanel) {
      const window = await chrome.windows.getCurrent();
      chrome.sidePanel.open({ windowId: window.id }).catch(() => {});
    }
  } else if (action === 'stop_capture') {
    show('■ stopped — finishing transcription...');
  }
}

document.getElementById('startBtn').addEventListener('click', () => send('start_capture'));
document.getElementById('stopBtn').addEventListener('click', () => send('stop_capture'));
