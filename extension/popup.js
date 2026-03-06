const statusEl = document.getElementById('status');

async function send(action) {
  const response = await chrome.runtime.sendMessage({ action });
  if (!response) {
    statusEl.textContent = 'no response from background';
    return;
  }
  statusEl.textContent = JSON.stringify(response, null, 2);

  // on successful start, store session ID and notify side panel
  if (response.ok && action === 'start_capture' && response.sessionId) {
    chrome.storage.local.set({ activeSessionId: response.sessionId });
    chrome.runtime.sendMessage({
      action: 'transcript_start',
      sessionId: response.sessionId,
    }).catch(() => {});
    // open side panel
    if (chrome.sidePanel) {
      chrome.sidePanel.open({ windowId: (await chrome.windows.getCurrent()).id }).catch(() => {});
    }
  }

  // on stop, clear session and notify side panel
  if (response.ok && action === 'stop_capture') {
    chrome.storage.local.remove('activeSessionId');
    chrome.runtime.sendMessage({ action: 'transcript_stop' }).catch(() => {});
  }
}

document.getElementById('startBtn').addEventListener('click', () => send('start_capture'));
document.getElementById('stopBtn').addEventListener('click', () => send('stop_capture'));
