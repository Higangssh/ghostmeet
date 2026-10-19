/**
 * Pure helpers shared by the extension's contexts.
 *
 * Run with: node --test tests/extension/
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  audioSocketUrl,
  escapeHtml,
  formatClock,
  newSessionId,
  renderSummary,
} from '../../extension/shared.js';

test('audio socket url carries the session id', () => {
  assert.equal(
    audioSocketUrl('127.0.0.1:8877', '20260802-101500', null),
    'ws://127.0.0.1:8877/ws/audio?session=20260802-101500',
  );
});

test('audio socket url passes the chosen language through', () => {
  const url = audioSocketUrl('127.0.0.1:8877', 's1', 'ko');
  assert.equal(url, 'ws://127.0.0.1:8877/ws/audio?session=s1&lang=ko');
});

test('auto-detect leaves the language off entirely', () => {
  for (const value of ['', null, undefined]) {
    assert.ok(!audioSocketUrl('h:1', 's1', value).includes('lang='));
  }
});

test('session ids and languages are url encoded', () => {
  const url = audioSocketUrl('127.0.0.1:8877', 'team sync&x=1', 'zh-CN');
  assert.ok(url.includes('session=team%20sync%26x%3D1'));
  assert.ok(url.endsWith('&lang=zh-CN'));
});

test('session id is derived from the given time', () => {
  assert.equal(newSessionId(new Date(2026, 7, 2, 9, 5, 3)), '20260802-090503');
});

test('clock formatting pads to mm:ss', () => {
  assert.equal(formatClock(0), '00:00');
  assert.equal(formatClock(9.7), '00:09');
  assert.equal(formatClock(75), '01:15');
  assert.equal(formatClock(3600), '60:00');
});

test('escapeHtml neutralises markup', () => {
  assert.equal(
    escapeHtml('<img src=x onerror="alert(1)">'),
    '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;',
  );
});

test('summary rendering escapes html before applying markdown', () => {
  // The summary is model output derived from meeting audio, so it is not trusted input.
  const html = renderSummary('## <script>alert(1)</script>');

  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('&lt;script&gt;'));
});

test('summary rendering turns headings and bullets into markup', () => {
  const html = renderSummary('## Decisions\n### Actions\n- ship it\n**bold**');

  assert.ok(html.includes('<h2>Decisions</h2>'));
  assert.ok(html.includes('<h3>Actions</h3>'));
  assert.ok(html.includes('ship it'));
  assert.ok(html.includes('<strong>bold</strong>'));
});

test('summary rendering keeps a plain paragraph intact', () => {
  assert.equal(renderSummary('just a line'), 'just a line');
});
