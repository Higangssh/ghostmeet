/**
 * Verifies the extension's recording path in a real browser against a real backend.
 *
 *   1. python -m backend                  (in another shell)
 *   2. npm install playwright && npx playwright install chromium
 *   3. node tests/extension/verify-capture.mjs
 *
 * What it covers: that a Manifest V3 service worker genuinely cannot record (no
 * MediaRecorder, no tabCapture.capture), that the offscreen document loads, and that
 * recording through it reaches the backend as a decodable stream with the right
 * per-session language.
 *
 * What it does NOT cover: acquiring the tab stream. chrome.tabCapture.getMediaStreamId
 * requires the activeTab grant that only a real toolbar click produces, and there is no
 * way to synthesise that click. Verify that by hand: load the extension unpacked, click
 * the icon on a tab with audio, press Start, and confirm audio_bytes climbs in
 * /api/sessions — and that you can still hear the tab.
 *
 * Runs muted, so the test tone never reaches the speakers.
 */
import { chromium } from 'playwright';
import WebSocket from 'ws';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXTENSION = path.resolve(HERE, '..', '..', 'extension');
const BACKEND = process.env.GHOSTMEET_URL || 'http://127.0.0.1:8877';
const CDP_PORT = 9411;
const SESSION = `verify-${Date.now()}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const api = (p) => fetch(BACKEND + p).then((r) => r.json());

function cdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m);
      pending.delete(m.id);
    }
  });
  return {
    ready: new Promise((r) => ws.on('open', r)),
    send: (method, params = {}) =>
      new Promise((res) => {
        const i = ++id;
        pending.set(i, res);
        ws.send(JSON.stringify({ id: i, method, params }));
      }),
    close: () => ws.close(),
  };
}

const failures = [];
function check(ok, label) {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label}`);
  if (!ok) failures.push(label);
}

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'ghostmeet-verify-'));
let context = null;
let offscreen = null;

try {
  await api('/api/health');
} catch {
  console.error(`No backend at ${BACKEND}. Start it with: python -m backend`);
  process.exit(2);
}

try {
  context = await chromium.launchPersistentContext(profile, {
    // Bundled Chromium, not installed Chrome: stable Chrome ignores --load-extension.
    channel: 'chromium',
    headless: false,
    args: [
      `--disable-extensions-except=${EXTENSION}`,
      `--load-extension=${EXTENSION}`,
      '--mute-audio',
      '--autoplay-policy=no-user-gesture-required',
      `--remote-debugging-port=${CDP_PORT}`,
      '--window-position=-2400,-2400',
    ],
  });

  let [worker] = context.serviceWorkers();
  if (!worker) worker = await context.waitForEvent('serviceworker', { timeout: 20000 });
  console.log(`extension ${worker.url().split('/')[2]}\n`);

  console.log('service worker capabilities');
  check(
    (await worker.evaluate('typeof MediaRecorder')) === 'undefined',
    'MediaRecorder is absent, so recording cannot happen in the worker',
  );
  check(
    (await worker.evaluate('typeof (chrome.tabCapture && chrome.tabCapture.capture)')) === 'undefined',
    'tabCapture.capture is absent — the pre-offscreen code could never have worked',
  );
  check(
    (await worker.evaluate('typeof (chrome.tabCapture && chrome.tabCapture.getMediaStreamId)')) === 'function',
    'tabCapture.getMediaStreamId is available',
  );
  check((await worker.evaluate('typeof chrome.offscreen')) === 'object', 'chrome.offscreen is available');

  console.log('\noffscreen document');
  await worker.evaluate(`chrome.offscreen.createDocument({
    url: 'offscreen.html', reasons: ['USER_MEDIA'], justification: 'verification run' })`);
  await sleep(1500);
  check(
    (await worker.evaluate(
      `chrome.runtime.getContexts({contextTypes:['OFFSCREEN_DOCUMENT']}).then(c => c.length)`,
    )) === 1,
    'exactly one offscreen document exists',
  );

  let target = null;
  for (let i = 0; i < 30 && !target; i++) {
    await sleep(300);
    const targets = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)
      .then((r) => r.json())
      .catch(() => []);
    target = targets.find((t) => t.url.includes('offscreen.html'));
  }
  check(Boolean(target), 'offscreen document is reachable (Playwright does not list it as a page)');
  if (!target) throw new Error('offscreen document not reachable');

  offscreen = cdp(target.webSocketDebuggerUrl);
  await offscreen.ready;
  await offscreen.send('Runtime.enable');

  console.log('\nrecording');
  const run = await offscreen.send('Runtime.evaluate', {
    expression: `(async () => {
      const mod = await import('./offscreen.js');
      const ac = new AudioContext();
      const osc = ac.createOscillator(); osc.type = 'sine'; osc.frequency.value = 440;
      const gain = ac.createGain(); gain.gain.value = 0.5;
      const dest = ac.createMediaStreamDestination();
      osc.connect(gain); gain.connect(dest); osc.start();
      await mod.startRecording(dest.stream, {
        sessionId: ${JSON.stringify(SESSION)}, language: 'en', backend: ${JSON.stringify(BACKEND.replace(/^https?:\/\//, ''))},
      });
      return 'recording';
    })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  check(run.result?.result?.value === 'recording', 'startRecording resolved, so the socket opened');

  await sleep(11000);
  const live = (await api('/api/sessions')).sessions[SESSION];
  check(Boolean(live), 'the backend registered the session');
  check(live?.audio_bytes > 0, `the backend received audio (${live?.audio_bytes} bytes)`);
  check(live?.chunks > 5, `audio arrived as a stream (${live?.chunks} chunks)`);
  check(live?.language === 'en', 'the per-session language reached the backend');

  console.log('\nstop');
  await offscreen.send('Runtime.evaluate', {
    expression: `import('./offscreen.js').then(m => m.stopRecording())`,
    awaitPromise: true,
    returnByValue: true,
  });

  let final = null;
  for (let i = 0; i < 90; i++) {
    await sleep(1000);
    final = (await api('/api/sessions')).sessions[SESSION];
    if (final && !['streaming', 'transcribing'].includes(final.status)) break;
  }
  check(final?.status === 'stopped', 'the session finished cleanly');
  check(
    final?.duration_sec > 8,
    `the backend decoded the recording (${final?.duration_sec}s of audio)`,
  );

  const transcript = await api(`/api/sessions/${SESSION}/transcript`);
  check(typeof transcript.segment_count === 'number', 'the transcript endpoint serves the session');
  console.log(`  (a pure tone yields ${transcript.segment_count} segments — voice detection rejects it, as it should)`);
} catch (error) {
  console.error('ERROR:', error.message);
  failures.push(error.message);
} finally {
  if (offscreen) offscreen.close();
  if (context) await context.close();
  fs.rmSync(profile, { recursive: true, force: true });
  console.log(`\n${failures.length ? `FAIL (${failures.length})` : 'PASS'}`);
  process.exit(failures.length ? 1 : 0);
}
