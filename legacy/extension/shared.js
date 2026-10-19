// Pure helpers shared by the popup, side panel and offscreen document.
// No chrome.* access in here, so it can be unit tested outside the browser.

export function audioSocketUrl(backend, sessionId, language) {
  const session = `session=${encodeURIComponent(sessionId)}`;
  const lang = language ? `&lang=${encodeURIComponent(language)}` : '';
  return `ws://${backend}/ws/audio?${session}${lang}`;
}

export function newSessionId(now = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  );
}

export function formatClock(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Render a model-generated summary as HTML.
 *
 * The summary is derived from meeting audio and produced by an LLM, so it is treated as
 * untrusted: everything is escaped first and only then are a few markdown constructs
 * turned back into markup.
 */
export function renderSummary(markdown) {
  return escapeHtml(markdown)
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^- (.+)$/gm, '• $1<br>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}
